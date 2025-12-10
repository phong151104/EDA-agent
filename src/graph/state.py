"""
LangGraph Shared State definitions.

Defines the state schema for the EDA workflow graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class WorkflowPhase(str, Enum):
    """Current phase of the workflow."""
    
    CONTEXT_FUSION = "context_fusion"
    PLANNING = "planning"
    VALIDATION = "validation"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    APPROVAL = "approval"
    COMPLETE = "complete"
    ERROR = "error"


class EDAState(TypedDict, total=False):
    """
    Shared state for the EDA workflow.
    
    This state is passed between all nodes in the LangGraph.
    Uses TypedDict for LangGraph compatibility.
    """
    
    # === Session Info ===
    session_id: str
    created_at: str
    
    # === User Input ===
    user_question: str
    
    # === Context Fusion Layer ===
    enriched_context: dict[str, Any]
    schema_context: dict[str, Any]
    episodic_context: list[dict[str, Any]]
    
    # === Planning Layer ===
    current_plan: dict[str, Any] | None
    plan_history: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    
    # === Validation Layer ===
    validation_result: dict[str, Any] | None
    critic_feedback: str | None
    debate_iteration: int
    plan_approved: bool
    
    # === Execution Layer ===
    generated_code: list[dict[str, Any]]
    execution_results: dict[int, dict[str, Any]]
    execution_errors: list[str]
    code_retry_count: int
    
    # === Analysis Layer ===
    hypothesis_evaluations: list[dict[str, Any]]
    insights: list[dict[str, Any]]
    analysis_summary: str | None
    
    # === Approval Layer ===
    is_insight_sufficient: bool
    final_report: dict[str, Any] | None
    
    # === Workflow Control ===
    current_phase: str
    error_message: str | None
    messages: Annotated[list[BaseMessage], add_messages]


def create_initial_state(question: str) -> EDAState:
    """
    Create initial state for a new EDA session.
    
    Args:
        question: User's question to analyze
        
    Returns:
        Initial EDAState with defaults
    """
    return EDAState(
        # Session
        session_id=str(uuid4()),
        created_at=datetime.utcnow().isoformat(),
        
        # Input
        user_question=question,
        
        # Context
        enriched_context={},
        schema_context={},
        episodic_context=[],
        
        # Planning
        current_plan=None,
        plan_history=[],
        hypotheses=[],
        
        # Validation
        validation_result=None,
        critic_feedback=None,
        debate_iteration=0,
        plan_approved=False,
        
        # Execution
        generated_code=[],
        execution_results={},
        execution_errors=[],
        code_retry_count=0,
        
        # Analysis
        hypothesis_evaluations=[],
        insights=[],
        analysis_summary=None,
        
        # Approval
        is_insight_sufficient=False,
        final_report=None,
        
        # Control
        current_phase=WorkflowPhase.CONTEXT_FUSION.value,
        error_message=None,
        messages=[],
    )


# === State Update Helpers ===

def update_phase(state: EDAState, phase: WorkflowPhase) -> dict[str, str]:
    """Create state update for phase change."""
    return {"current_phase": phase.value}


def add_to_plan_history(state: EDAState) -> dict[str, list]:
    """Add current plan to history before updating."""
    if state.get("current_plan"):
        history = list(state.get("plan_history", []))
        history.append(state["current_plan"])
        return {"plan_history": history}
    return {"plan_history": state.get("plan_history", [])}


def increment_debate_iteration(state: EDAState) -> dict[str, int]:
    """Increment the debate iteration counter."""
    return {"debate_iteration": state.get("debate_iteration", 0) + 1}


def increment_code_retry(state: EDAState) -> dict[str, int]:
    """Increment the code retry counter."""
    return {"code_retry_count": state.get("code_retry_count", 0) + 1}
