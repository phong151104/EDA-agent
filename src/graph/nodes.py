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
    analysis_phase = state.get("analysis_phase", "exploration")
    
    # When transitioning to deep_dive, DON'T pass previous exploration plan
    # This forces Planner to generate fresh hypotheses based on findings
    previous_plan = None
    if state.get("current_plan"):
        plan_dict = state["current_plan"]
        plan_phase = plan_dict.get("phase", "exploration")
        
        # Only carry over previous_plan if same phase (for refinement within phase)
        if plan_phase == analysis_phase:
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
        else:
            logger.info(f"[Planner] Phase transition {plan_phase} → {analysis_phase}, clearing previous_plan")
    
    # Use prompt_context from Context Fusion for richer schema info
    enriched_context = state.get("enriched_context", {})
    prompt_context = state.get("prompt_context", "")
    sub_graph = state.get("sub_graph", {})
    
    # Enhance enriched_context with prompt_context
    if prompt_context:
        enriched_context["schema_description"] = prompt_context
    
    # CRITICAL: Merge SubGraph into enriched_context so Planner sees actual tables
    if sub_graph:
        enriched_context["tables"] = sub_graph.get("tables", [])
        enriched_context["columns"] = sub_graph.get("columns", [])
        enriched_context["joins"] = sub_graph.get("joins", [])
        logger.info(f"[Planner] Merged SubGraph: {len(enriched_context.get('tables', []))} tables")
    
    planner_input = PlannerInput(
        question=state["user_question"],
        enriched_context=enriched_context,
        previous_plan=previous_plan,
        critic_feedback=state.get("critic_feedback"),
        # Two-phase analysis fields
        analysis_phase=analysis_phase,
        exploration_summary=state.get("exploration_summary"),
    )
    
    # Log what's being passed
    logger.info(f"[Planner] Phase: {analysis_phase}")
    logger.info(f"[Planner] Intent: {state.get('analyzed_query', {}).get('intent', 'unknown')}")
    logger.info(f"[Planner] Tables available: {len(state.get('sub_graph', {}).get('tables', []))}")
    
    # Log exploration summary if in deep_dive phase
    exploration_summary = state.get("exploration_summary")
    if analysis_phase == "deep_dive":
        if exploration_summary:
            keys = list(exploration_summary.keys()) if isinstance(exploration_summary, dict) else []
            logger.info(f"[Planner] Received exploration_summary with keys: {keys}")
        else:
            logger.warning("[Planner] Deep dive phase but NO exploration_summary!")
    
    # Generate plan
    output = await planner.process(planner_input)
    
    # Update state with new plan
    plan_history = list(state.get("plan_history", []))
    if state.get("current_plan"):
        plan_history.append(state["current_plan"])
    
    # Save plan with current phase info
    plan_dict = output.plan.to_dict()
    plan_dict["phase"] = analysis_phase  # Tag plan with its phase
    
    return {
        "current_plan": plan_dict,
        "plan_history": plan_history,
        "hypotheses": [h.to_dict() for h in output.plan.hypotheses],
        "current_phase": WorkflowPhase.VALIDATION.value,
        "messages": [AIMessage(content=f"[Planner] Generated plan with {len(output.plan.steps)} steps (phase: {analysis_phase})")],
    }


async def critic_node(state: EDAState) -> dict[str, Any]:
    """
    Critic Agent node with 3-layer verification.
    
    Layer 1: Data Availability (Neo4j query)
    Layer 2: Logical Consistency (Rules-based)
    Layer 3: Business Logic (Optional LLM)
    
    On max iterations: Filters out invalid hypotheses and approves valid ones.
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
    filtered_plan = None
    
    # On max iterations: Filter invalid hypotheses and approve valid ones
    if iteration >= max_iterations and not plan_approved:
        logger.info(f"[Critic] ───────────────────────────────────────────────")
        logger.info(f"[Critic] MAX ITERATIONS REACHED - FILTERING PLAN")
        logger.info(f"[Critic] ───────────────────────────────────────────────")
        
        # Collect step IDs with table errors
        invalid_step_ids = set()
        for issue in result.issues:
            if issue.severity == "error" and issue.issue_type == "table_not_found":
                step_id = issue.details.get("step_id") if issue.details else None
                if step_id:
                    invalid_step_ids.add(step_id)
        
        # Find hypothesis IDs associated with invalid steps
        invalid_hypothesis_ids = set()
        for step in steps:
            step_id = step.get("id", "")
            if step_id in invalid_step_ids:
                hypo_id = step.get("hypothesis_id", "")
                if hypo_id:
                    invalid_hypothesis_ids.add(hypo_id)
        
        # Filter hypotheses and steps
        valid_hypotheses = [
            h for h in hypotheses 
            if h.get("id", "") not in invalid_hypothesis_ids
        ]
        valid_steps = [
            s for s in steps 
            if s.get("hypothesis_id", "") not in invalid_hypothesis_ids
        ]
        
        logger.info(f"[Critic] Removed hypotheses: {invalid_hypothesis_ids}")
        logger.info(f"[Critic] Remaining: {len(valid_hypotheses)} hypotheses, {len(valid_steps)} steps")
        
        if valid_hypotheses and valid_steps:
            # Create filtered plan
            filtered_plan = {
                **plan_dict,
                "hypotheses": valid_hypotheses,
                "steps": valid_steps,
                "filtered": True,
                "removed_hypotheses": list(invalid_hypothesis_ids),
            }
            plan_approved = True
            logger.info(f"[Critic] ✅ SOFT APPROVAL - Proceeding with valid hypotheses only")
        else:
            logger.error(f"[Critic] ❌ No valid hypotheses remaining!")
    
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
    
    # Use filtered plan if available
    final_plan = filtered_plan if filtered_plan else plan_dict
    
    # Build return dict - only include current_plan if we filtered it
    result_dict = {
        "validation_result": {
            "status": "approved" if plan_approved else "rejected",
            "approval_score": approval_score,
            "layer1_passed": result.layer1_passed,
            "layer2_passed": result.layer2_passed,
            "layer3_passed": result.layer3_passed,
            "issues": [i.to_dict() for i in result.issues],
            "total_errors": errors,
            "total_warnings": warnings,
            "filtered": filtered_plan is not None,
        },
        "critic_feedback": feedback if not plan_approved else None,
        "debate_iteration": iteration,
        "plan_approved": plan_approved,
        "current_phase": WorkflowPhase.EXECUTION.value if plan_approved else WorkflowPhase.PLANNING.value,
        "messages": [AIMessage(content=f"[Critic] {'Approved' if plan_approved else 'Rejected'} (L1:{'✅' if result.layer1_passed else '❌'} L2:{'✅' if result.layer2_passed else '❌'} L3:{'✅' if result.layer3_passed else '⏭️'}){' [FILTERED]' if filtered_plan else ''}")],
    }
    
    # Only update current_plan if we filtered it
    if filtered_plan:
        result_dict["current_plan"] = filtered_plan
    
    return result_dict
async def code_agent_node(state: EDAState) -> dict[str, Any]:
    """
    Code Agent node.
    
    Receives approved plan from Critic and generates/executes code.
    """
    from src.agents.code_agent import CodeAgentInput, ExecutionResult, ExecutionStatus, OutputType
    
    logger.info("[CodeAgent] ═══════════════════════════════════════════════")
    logger.info("[CodeAgent] GENERATING CODE FROM APPROVED PLAN")
    logger.info("[CodeAgent] ═══════════════════════════════════════════════")
    
    # Check if plan is approved
    if not state.get("plan_approved"):
        logger.warning("[CodeAgent] Plan not approved yet, cannot execute")
        return {
            "error_message": "Plan not approved by Critic",
            "current_phase": WorkflowPhase.PLANNING.value,
        }
    
    if not state.get("current_plan"):
        return {
            "error_message": "No plan to execute",
            "current_phase": WorkflowPhase.ERROR.value,
        }
    
    plan_dict = state["current_plan"]
    
    logger.info(f"[CodeAgent] Plan version: {plan_dict.get('version', 1)}")
    logger.info(f"[CodeAgent] Hypotheses: {len(plan_dict.get('hypotheses', []))}")
    logger.info(f"[CodeAgent] Steps: {len(plan_dict.get('steps', []))}")
    
    # Initialize code agent
    code_agent = CodeAgent()
    
    # Reconstruct plan with new format
    plan = AnalysisPlan(
        question=plan_dict.get("question", state.get("question", "")),
        hypotheses=[Hypothesis.from_dict(h) for h in plan_dict.get("hypotheses", [])],
        steps=[AnalysisStep.from_dict(s) for s in plan_dict.get("steps", [])],
        version=plan_dict.get("version", 1),
    )
    
    # Get schema context from sub_graph (passed from Context Fusion)
    sub_graph = state.get("sub_graph", {})
    schema_context = {
        "tables": sub_graph.get("tables", []),
        "columns": sub_graph.get("columns", []),
        "joins": sub_graph.get("joins", []),
    }
    
    logger.info(f"[CodeAgent] Schema context: {len(schema_context['tables'])} tables, "
                f"{len(schema_context['columns'])} columns, {len(schema_context['joins'])} joins")
    
    # Build input with schema context
    error_to_fix = None
    if state.get("execution_errors"):
        error_to_fix = state["execution_errors"][-1]
    
    code_input = CodeAgentInput(
        plan=plan,
        schema_context=schema_context,
        previous_results={
            str(k): v for k, v in state.get("execution_results", {}).items()
            if isinstance(v, ExecutionResult)
        },
        error_to_fix=error_to_fix,
        retry_count=state.get("code_retry_count", 0),
    )
    
    # Execute
    output = await code_agent.process(code_input)
    
    # Log results
    logger.info("[CodeAgent] ───────────────────────────────────────────────")
    logger.info("[CodeAgent] EXECUTION RESULTS:")
    for step_id, result in output.execution_results.items():
        status_icon = "✅" if result.status == ExecutionStatus.SUCCESS else "❌"
        logger.info(f"[CodeAgent]   {status_icon} {step_id}: {result.status.value}")
    
    # Handle errors
    execution_errors = list(state.get("execution_errors", []))
    for step_id, result in output.execution_results.items():
        if result.status == ExecutionStatus.ERROR:
            execution_errors.append(result.error_message or f"Error in {step_id}")
    
    # Update retry count if there were errors
    code_retry_count = state.get("code_retry_count", 0)
    if not output.all_success:
        code_retry_count += 1
        logger.warning(f"[CodeAgent] Errors occurred, retry count: {code_retry_count}")
    
    # Determine next phase
    # ALWAYS proceed to ANALYSIS even with partial errors
    # Analyst can work with whatever results we have
    if output.all_success:
        next_phase = WorkflowPhase.ANALYSIS.value
        logger.info("[CodeAgent] ✅ All steps executed successfully!")
    elif code_retry_count >= code_agent.max_retries:
        # Max retries reached - proceed with partial results instead of ERROR
        next_phase = WorkflowPhase.ANALYSIS.value
        success_count = sum(1 for r in output.execution_results.values() if r.status == ExecutionStatus.SUCCESS)
        total_count = len(output.execution_results)
        logger.warning(f"[CodeAgent] ⚠️ Max retries ({code_agent.max_retries}) reached. "
                      f"Proceeding to analysis with {success_count}/{total_count} successful steps")
    else:
        next_phase = WorkflowPhase.EXECUTION.value
        logger.info(f"[CodeAgent] Will retry (attempt {code_retry_count + 1})")
    
    logger.info("[CodeAgent] ═══════════════════════════════════════════════")
    
    return {
        "generated_code": [c.to_dict() for c in output.generated_code],
        "execution_results": {
            str(k): v.to_dict() for k, v in output.execution_results.items()
        },
        "execution_errors": execution_errors,
        "code_retry_count": code_retry_count,
        "all_code_success": output.all_success,
        "current_phase": next_phase,
        "messages": [AIMessage(content=f"[CodeAgent] Executed {len(output.generated_code)} steps ({'✅ All success' if output.all_success else '❌ Has errors'})")],
    }



async def analyst_node(state: EDAState) -> dict[str, Any]:
    """
    Analyst Agent node.
    
    Evaluates execution results and generates insights.
    For 2-phase flow: saves exploration_summary and switches phase after Phase 1.
    """
    analysis_phase = state.get("analysis_phase", "exploration")
    logger.info(f"Analyzing results... [Phase: {analysis_phase}]")
    
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
    for step_id, result_dict in state.get("execution_results", {}).items():
        # Step IDs are strings like 's1', 's2' - keep as string
        execution_results[step_id] = ExecutionResult(
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
        analysis_phase=analysis_phase,  # Pass current phase
    )
    
    # Analyze
    output = await analyst.process(analyst_input)
    
    # Build result based on phase
    result = {
        "hypothesis_evaluations": [e.to_dict() for e in output.hypothesis_evaluations],
        "insights": [i.to_dict() for i in output.insights],
        "analysis_summary": output.summary,
        "messages": [AIMessage(content=f"[Analyst] Phase: {analysis_phase}, Confidence: {output.confidence:.2f}")],
    }
    
    if analysis_phase == "exploration":
        # Phase 1 complete → save exploration_summary and switch to deep_dive
        result["exploration_summary"] = output.exploration_summary
        result["analysis_phase"] = "deep_dive"  # Switch to Phase 2
        result["current_phase"] = WorkflowPhase.PLANNING.value  # Go back to Planner
        result["is_insight_sufficient"] = False  # Not done yet
        
        # Log exploration_summary for debugging
        if output.exploration_summary:
            keys = list(output.exploration_summary.keys()) if isinstance(output.exploration_summary, dict) else []
            logger.info(f"[Analyst] Exploration complete, summary keys: {keys}")
        else:
            logger.warning("[Analyst] Exploration complete but NO exploration_summary generated!")
        logger.info(f"[Analyst] Switching to deep_dive phase")
    else:
        # Phase 2 (deep_dive)
        result["current_phase"] = WorkflowPhase.APPROVAL.value
        result["is_insight_sufficient"] = output.answers_question and output.confidence >= 0.7
        
        # Increment deep_dive iteration counter
        deep_dive_iteration = state.get("deep_dive_iteration", 0) + 1
        result["deep_dive_iteration"] = deep_dive_iteration
        logger.info(f"[Analyst] Deep dive iteration {deep_dive_iteration}")
    
    return result


async def approval_node(state: EDAState) -> dict[str, Any]:
    """
    Insight Approval node.
    
    Checks if insights are sufficient and generates final report.
    For 2-phase flow: checks deep_dive_iteration max (3 loops).
    When not sufficient, provides detailed feedback to Planner.
    """
    logger.info("Checking insight sufficiency...")
    
    is_sufficient = state.get("is_insight_sufficient", False)
    deep_dive_iteration = state.get("deep_dive_iteration", 0)
    max_deep_dive_loops = 3
    
    # Force complete if max iterations reached
    if deep_dive_iteration >= max_deep_dive_loops:
        logger.warning(f"Max deep dive iterations ({max_deep_dive_loops}) reached, forcing completion")
        is_sufficient = True
    
    if is_sufficient:
        # Generate final report
        final_report = {
            "question": state["user_question"],
            "summary": state.get("analysis_summary", ""),
            "hypotheses": state.get("hypotheses", []),
            "evaluations": state.get("hypothesis_evaluations", []),
            "insights": state.get("insights", []),
            "exploration_summary": state.get("exploration_summary"),
            "deep_dive_iterations": deep_dive_iteration,
            "generated_at": state["created_at"],
        }
        
        return {
            "final_report": final_report,
            "is_insight_sufficient": True,  # MUST be set for routing!
            "current_phase": WorkflowPhase.COMPLETE.value,
            "messages": [AIMessage(content=f"[System] Analysis complete! (Deep dive iterations: {deep_dive_iteration})")],
        }
    else:
        # Build detailed feedback for Planner
        # Include analysis summary and previous hypotheses to avoid repetition
        analysis_summary = state.get("analysis_summary", "")
        current_plan = state.get("current_plan", {})
        previous_hypotheses = current_plan.get("hypotheses", [])
        
        # Extract hypothesis statements that have already been tested
        tested_hypotheses = [
            h.get("statement", h.get("title", ""))
            for h in previous_hypotheses
        ]
        
        # Build detailed critic feedback
        detailed_feedback = f"""## Phân tích chưa đủ sâu - Cần giả thuyết MỚI

### Tóm tắt từ Analyst:
{analysis_summary[:1500] if analysis_summary else 'Không có tóm tắt'}

### ⚠️ CÁC GIẢ THUYẾT ĐÃ KIỂM TRA (KHÔNG LẶP LẠI):
{chr(10).join(f'- {h}' for h in tested_hypotheses[:6])}

### Yêu cầu cho kế hoạch tiếp theo:
1. KHÔNG lặp lại các giả thuyết đã kiểm tra ở trên
2. Đào SÂU hơn vào các phát hiện từ dữ liệu thực tế
3. Tìm NGUYÊN NHÂN GỐC RỄ dựa trên pattern đã thấy
4. Đề xuất các dimension/breakdown KHÁC để phân tích

### Gợi ý hướng đi mới:
- Phân tích theo segment khách hàng khác
- So sánh hiệu suất theo nguồn/kênh khác
- Tìm correlation với các yếu tố chưa xét
- Drill down vào điểm dị thường đã phát hiện
"""
        
        return {
            "critic_feedback": detailed_feedback,
            "deep_dive_iteration": deep_dive_iteration,  # For display
            "current_phase": WorkflowPhase.PLANNING.value,
            "messages": [AIMessage(content=f"[System] Needs more analysis (iteration {deep_dive_iteration}/{max_deep_dive_loops}), returning to planning with detailed feedback...")],
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
