"""
LangGraph Conditional Edge functions.

Defines routing logic between nodes based on state.
"""

from __future__ import annotations

from typing import Literal

from config import config

from .state import EDAState, WorkflowPhase


def route_after_critic(
    state: EDAState,
) -> Literal["planner", "code_agent"]:
    """
    Route after Critic validation.
    
    Decision:
    - If plan approved → proceed to Code Agent
    - If not approved and under max iterations → return to Planner
    - If max iterations reached → proceed anyway (force)
    """
    if state.get("plan_approved", False):
        return "code_agent"
    
    debate_iteration = state.get("debate_iteration", 0)
    max_iterations = config.max_debate_iterations
    
    if debate_iteration >= max_iterations:
        # Force proceed after max iterations
        return "code_agent"
    
    return "planner"


def route_after_code_agent(
    state: EDAState,
) -> Literal["analyst", "code_agent", "error"]:
    """
    Route after Code Agent execution.
    
    Decision:
    - If all success → proceed to Analyst
    - If errors and under max retries → retry Code Agent
    - If max retries reached → proceed to Analyst with partial results (NOT error)
    """
    # Check if there are any execution results
    execution_results = state.get("execution_results", {})
    if not execution_results:
        return "error"
    
    # Check for errors
    has_errors = any(
        r.get("status") == "error"
        for r in execution_results.values()
    )
    
    if not has_errors:
        return "analyst"
    
    # Check retry count
    retry_count = state.get("code_retry_count", 0)
    max_retries = config.max_code_retries
    
    if retry_count >= max_retries:
        # Proceed to analyst with partial results - don't go to error!
        # Analyst can work with whatever successful steps we have
        return "analyst"
    
    return "code_agent"


def route_after_analyst(
    state: EDAState,
) -> Literal["approval", "code_agent", "planner"]:
    """
    Route after Analyst evaluation.
    
    Decision for TWO-PHASE FLOW:
    - Phase 1 (exploration) complete → switch to deep_dive, go to planner
    - Phase 2 (deep_dive) → go to approval (or planner if needs more loops)
    """
    analysis_phase = state.get("analysis_phase", "exploration")
    
    # Phase 1 complete → move to Phase 2
    if analysis_phase == "exploration":
        # Exploration done, need to go to Planner for deep dive
        return "planner"
    
    # Phase 2 - check if we need more analysis
    evaluations = state.get("hypothesis_evaluations", [])
    
    # Check if any evaluation indicates need for re-execution
    needs_rerun = any(
        e.get("newStatus") == "error" or e.get("confidence", 0) < 0.3
        for e in evaluations
    )
    
    if needs_rerun:
        return "code_agent"
    
    return "approval"


def route_after_approval(
    state: EDAState,
) -> Literal["end", "planner"]:
    """
    Route after Insight Approval.
    
    Decision:
    - If insights sufficient → end
    - If not sufficient → return to planning for deeper analysis
    """
    if state.get("is_insight_sufficient", False):
        return "end"
    
    # Check if we've already iterated too many times
    plan_versions = len(state.get("plan_history", []))
    if plan_versions >= 5:  # Max 5 full iterations
        return "end"
    
    return "planner"


def route_by_phase(
    state: EDAState,
) -> Literal[
    "context_fusion",
    "planner",
    "critic",
    "code_agent",
    "analyst",
    "approval",
    "error",
    "end",
]:
    """
    General routing based on current phase.
    
    Used for entry point or recovery routing.
    """
    phase = state.get("current_phase", WorkflowPhase.CONTEXT_FUSION.value)
    
    if phase == WorkflowPhase.CONTEXT_FUSION.value:
        return "context_fusion"
    elif phase == WorkflowPhase.PLANNING.value:
        return "planner"
    elif phase == WorkflowPhase.VALIDATION.value:
        return "critic"
    elif phase == WorkflowPhase.EXECUTION.value:
        return "code_agent"
    elif phase == WorkflowPhase.ANALYSIS.value:
        return "analyst"
    elif phase == WorkflowPhase.APPROVAL.value:
        return "approval"
    elif phase == WorkflowPhase.ERROR.value:
        return "error"
    elif phase == WorkflowPhase.COMPLETE.value:
        return "end"
    else:
        return "error"
