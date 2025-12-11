"""
LangGraph Node implementations.

Each node represents a step in the EDA workflow.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.agents import (
    AnalystAgent,
    AnalystInput,
    CodeAgent,
    CodeAgentInput,
    CriticAgent,
    CriticInput,
    PlannerAgent,
    PlannerInput,
)
from src.agents.planner import AnalysisPlan, AnalysisStep, Hypothesis

from .state import EDAState, WorkflowPhase

logger = logging.getLogger(__name__)


# === Node Implementations ===

async def context_fusion_node(state: EDAState) -> dict[str, Any]:
    """
    Context Fusion Layer node.
    
    Retrieves relevant schema and episodic context for the question.
    Uses hybrid search: Vector (semantic) + Keyword + Entity matching.
    """
    from src.context_fusion import ContextBuilder
    
    logger.info(f"[Context Fusion] Processing: {state['user_question'][:50]}...")
    
    try:
        # Build enriched context using hybrid search
        builder = ContextBuilder(use_llm=True, domain="vnfilm_ticketing")
        context = await builder.build(state["user_question"], top_k=10)
        builder.close()
        
        logger.info(
            f"[Context Fusion] Found: {len(context.sub_graph.tables)} tables, "
            f"{len(context.sub_graph.columns)} columns, "
            f"{len(context.sub_graph.joins)} joins"
        )
        
        return {
            "enriched_context": context.to_dict(),
            "prompt_context": context.prompt_context,
            "sub_graph": context.sub_graph.to_dict(),
            "analyzed_query": {
                "intent": context.analyzed_query.intent.value,
                "keywords": context.analyzed_query.keywords,
                "entities": [
                    {"text": e.text, "type": e.entity_type, "normalized": e.normalized_name}
                    for e in context.analyzed_query.entities
                ],
                "rewritten_query": context.analyzed_query.rewritten_query,
            },
            "schema_context": context.sub_graph.to_dict(),  # For backward compatibility
            "current_phase": WorkflowPhase.PLANNING.value,
            "messages": [AIMessage(content=f"[ContextFusion] Found {len(context.sub_graph.tables)} relevant tables")],
        }
    except Exception as e:
        logger.error(f"[Context Fusion] Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty context on error
        return {
            "enriched_context": {},
            "prompt_context": "",
            "sub_graph": {},
            "analyzed_query": {},
            "schema_context": {},
            "error_message": f"Context fusion failed: {str(e)}",
            "current_phase": WorkflowPhase.PLANNING.value,
        }


async def planner_node(state: EDAState) -> dict[str, Any]:
    """
    Planner Agent node.
    
    Generates or refines analysis plan based on enriched context from Context Fusion.
    """
    logger.info(f"[Planner] Iteration {state.get('debate_iteration', 0) + 1}")
    
    # Initialize planner
    planner = PlannerAgent()
    
    # Build input
    previous_plan = None
    if state.get("current_plan"):
        # Reconstruct AnalysisPlan from state dict
        plan_dict = state["current_plan"]
        previous_plan = AnalysisPlan(
            question=plan_dict.get("question", state["user_question"]),
            hypotheses=[
                Hypothesis.from_dict(h) for h in plan_dict.get("hypotheses", [])
            ],
            steps=[
                AnalysisStep.from_dict(s) for s in plan_dict.get("steps", [])
            ],
            version=plan_dict.get("version", 1),
        )
    
    # Use prompt_context from Context Fusion for richer schema info
    enriched_context = state.get("enriched_context", {})
    prompt_context = state.get("prompt_context", "")
    
    # Enhance enriched_context with prompt_context
    if prompt_context:
        enriched_context["schema_description"] = prompt_context
    
    planner_input = PlannerInput(
        question=state["user_question"],
        enriched_context=enriched_context,
        previous_plan=previous_plan,
        critic_feedback=state.get("critic_feedback"),
    )
    
    # Log what's being passed
    logger.info(f"[Planner] Intent: {state.get('analyzed_query', {}).get('intent', 'unknown')}")
    logger.info(f"[Planner] Tables available: {len(state.get('sub_graph', {}).get('tables', []))}")
    
    # Generate plan
    output = await planner.process(planner_input)
    
    # Update state with new plan
    plan_history = list(state.get("plan_history", []))
    if state.get("current_plan"):
        plan_history.append(state["current_plan"])
    
    return {
        "current_plan": output.plan.to_dict(),
        "plan_history": plan_history,
        "hypotheses": [h.to_dict() for h in output.plan.hypotheses],
        "current_phase": WorkflowPhase.VALIDATION.value,
        "messages": [AIMessage(content=f"[Planner] Generated plan with {len(output.plan.steps)} steps")],
    }


async def critic_node(state: EDAState) -> dict[str, Any]:
    """
    Critic Agent node with 3-layer verification.
    
    Layer 1: Data Availability (Neo4j query)
    Layer 2: Logical Consistency (Rules-based)
    Layer 3: Business Logic (Optional LLM)
    """
    from src.validation import PlanVerifier
    
    iteration = state.get("debate_iteration", 0) + 1
    max_iterations = state.get("max_debate_iterations", 3)
    
    logger.info(f"[Critic] ═══════════════════════════════════════════════")
    logger.info(f"[Critic] PLAN VERIFICATION (Iteration {iteration}/{max_iterations})")
    logger.info(f"[Critic] ═══════════════════════════════════════════════")
    
    if not state.get("current_plan"):
        return {
            "error_message": "No plan to validate",
            "current_phase": WorkflowPhase.ERROR.value,
        }
    
    plan_dict = state["current_plan"]
    
    # Log plan summary
    hypotheses = plan_dict.get("hypotheses", [])
    steps = plan_dict.get("steps", [])
    logger.info(f"[Critic] Plan has {len(hypotheses)} hypotheses, {len(steps)} steps")
    
    # =========================================================================
    # Run 3-Layer Verification
    # =========================================================================
    
    verifier = PlanVerifier(domain="vnfilm_ticketing")
    
    # Only run Layer 3 (LLM) if we've been iterating for a while
    run_layer3 = iteration >= 2
    
    result = await verifier.verify(plan_dict, run_layer3=run_layer3)
    
    # =========================================================================
    # Make Decision
    # =========================================================================
    
    plan_approved = result.passed
    
    # Force approve if max iterations reached and no critical errors
    has_critical_errors = any(
        i.severity == "error" for i in result.issues
    )
    
    if iteration >= max_iterations and not has_critical_errors:
        logger.warning(f"[Critic] Max iterations ({max_iterations}) reached, forcing approval")
        plan_approved = True
    
    # Format feedback for Planner
    feedback = ""
    if not plan_approved:
        feedback = verifier.format_issues(result.issues)
    
    logger.info(f"[Critic] Final Decision: {'✅ APPROVED' if plan_approved else '❌ REJECTED'}")
    
    # Calculate approval score based on issues
    total_issues = len(result.issues)
    errors = sum(1 for i in result.issues if i.severity == "error")
    warnings = sum(1 for i in result.issues if i.severity == "warning")
    
    # Score: 1.0 if no issues, decreases with issues
    approval_score = max(0.0, 1.0 - (errors * 0.3) - (warnings * 0.1))
    
    return {
        "validation_result": {
            "status": "approved" if plan_approved else "rejected",
            "approval_score": approval_score,
            "layer1_passed": result.layer1_passed,
            "layer2_passed": result.layer2_passed,
            "layer3_passed": result.layer3_passed,
            "issues": [i.to_dict() for i in result.issues],
            "total_errors": errors,
            "total_warnings": warnings,
        },
        "critic_feedback": feedback if not plan_approved else None,
        "debate_iteration": iteration,
        "plan_approved": plan_approved,
        "current_phase": WorkflowPhase.EXECUTION.value if plan_approved else WorkflowPhase.PLANNING.value,
        "messages": [AIMessage(content=f"[Critic] {'Approved' if plan_approved else 'Rejected'} (L1:{'✅' if result.layer1_passed else '❌'} L2:{'✅' if result.layer2_passed else '❌'} L3:{'✅' if result.layer3_passed else '⏭️'})")],
    }



async def code_agent_node(state: EDAState) -> dict[str, Any]:
    """
    Code Agent node.
    
    Generates and executes code for the analysis plan.
    """
    logger.info("Generating and executing code...")
    
    if not state.get("current_plan"):
        return {
            "error_message": "No plan to execute",
            "current_phase": WorkflowPhase.ERROR.value,
        }
    
    # Initialize code agent
    code_agent = CodeAgent()
    
    # Reconstruct plan
    plan_dict = state["current_plan"]
    plan = AnalysisPlan(
        question=plan_dict.get("question", ""),
        hypotheses=[Hypothesis.from_dict(h) for h in plan_dict.get("hypotheses", [])],
        steps=[AnalysisStep.from_dict(s) for s in plan_dict.get("steps", [])],
        version=plan_dict.get("version", 1),
    )
    
    # Build input
    error_to_fix = None
    if state.get("execution_errors"):
        error_to_fix = state["execution_errors"][-1]
    
    code_input = CodeAgentInput(
        plan=plan,
        previous_results={
            int(k): v for k, v in state.get("execution_results", {}).items()
        },
        error_to_fix=error_to_fix,
        retry_count=state.get("code_retry_count", 0),
    )
    
    # Execute
    output = await code_agent.process(code_input)
    
    # Handle errors
    execution_errors = list(state.get("execution_errors", []))
    for step_num, result in output.execution_results.items():
        if result.status.value == "error":
            execution_errors.append(result.error_message or "Unknown error")
    
    # Update retry count if there were errors
    code_retry_count = state.get("code_retry_count", 0)
    if not output.all_success:
        code_retry_count += 1
    
    return {
        "generated_code": [c.to_dict() for c in output.generated_code],
        "execution_results": {
            str(k): v.to_dict() for k, v in output.execution_results.items()
        },
        "execution_errors": execution_errors,
        "code_retry_count": code_retry_count,
        "current_phase": WorkflowPhase.ANALYSIS.value if output.all_success else WorkflowPhase.EXECUTION.value,
        "messages": [AIMessage(content=f"[CodeAgent] Executed {len(output.generated_code)} steps")],
    }


async def analyst_node(state: EDAState) -> dict[str, Any]:
    """
    Analyst Agent node.
    
    Evaluates execution results and generates insights.
    """
    logger.info("Analyzing results...")
    
    if not state.get("execution_results"):
        return {
            "error_message": "No execution results to analyze",
            "current_phase": WorkflowPhase.ERROR.value,
        }
    
    # Initialize analyst
    analyst = AnalystAgent()
    
    # Reconstruct plan and results
    plan_dict = state["current_plan"]
    plan = AnalysisPlan(
        question=plan_dict.get("question", ""),
        hypotheses=[Hypothesis.from_dict(h) for h in plan_dict.get("hypotheses", [])],
        steps=[AnalysisStep.from_dict(s) for s in plan_dict.get("steps", [])],
        version=plan_dict.get("version", 1),
    )
    
    # Import here to avoid circular import
    from src.agents.code_agent import ExecutionResult, ExecutionStatus, OutputType
    
    execution_results = {}
    for step_num_str, result_dict in state.get("execution_results", {}).items():
        execution_results[int(step_num_str)] = ExecutionResult(
            status=ExecutionStatus(result_dict.get("status", "success")),
            output_type=OutputType(result_dict.get("outputType", "text")),
            output=result_dict.get("output"),
            execution_time_ms=result_dict.get("executionTimeMs", 0),
            error_message=result_dict.get("errorMessage"),
        )
    
    analyst_input = AnalystInput(
        plan=plan,
        execution_results=execution_results,
        original_question=state["user_question"],
    )
    
    # Analyze
    output = await analyst.process(analyst_input)
    
    return {
        "hypothesis_evaluations": [e.to_dict() for e in output.hypothesis_evaluations],
        "insights": [i.to_dict() for i in output.insights],
        "analysis_summary": output.summary,
        "is_insight_sufficient": output.answers_question and output.confidence >= 0.7,
        "current_phase": WorkflowPhase.APPROVAL.value,
        "messages": [AIMessage(content=f"[Analyst] Confidence: {output.confidence:.2f}")],
    }


async def approval_node(state: EDAState) -> dict[str, Any]:
    """
    Insight Approval node.
    
    Checks if insights are sufficient and generates final report.
    """
    logger.info("Checking insight sufficiency...")
    
    is_sufficient = state.get("is_insight_sufficient", False)
    
    if is_sufficient:
        # Generate final report
        final_report = {
            "question": state["user_question"],
            "summary": state.get("analysis_summary", ""),
            "hypotheses": state.get("hypotheses", []),
            "evaluations": state.get("hypothesis_evaluations", []),
            "insights": state.get("insights", []),
            "generated_at": state["created_at"],
        }
        
        return {
            "final_report": final_report,
            "current_phase": WorkflowPhase.COMPLETE.value,
            "messages": [AIMessage(content="[System] Analysis complete!")],
        }
    else:
        # Need more analysis - go back to planning
        return {
            "critic_feedback": "Insights were not sufficient. Please refine the analysis.",
            "current_phase": WorkflowPhase.PLANNING.value,
            "messages": [AIMessage(content="[System] Needs more analysis, returning to planning...")],
        }


async def error_node(state: EDAState) -> dict[str, Any]:
    """
    Error handling node.
    """
    logger.error(f"Workflow error: {state.get('error_message', 'Unknown')}")
    return {
        "final_report": {
            "error": True,
            "message": state.get("error_message", "Unknown error"),
            "question": state["user_question"],
        },
        "current_phase": WorkflowPhase.COMPLETE.value,
    }
