"""
Main EDA Workflow Graph.

Assembles all nodes and edges into the complete LangGraph workflow.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .state import EDAState, create_initial_state
from .nodes import (
    analyst_node,
    approval_node,
    code_agent_node,
    context_fusion_node,
    critic_node,
    error_node,
    planner_node,
)
from .edges import (
    route_after_analyst,
    route_after_approval,
    route_after_code_agent,
    route_after_critic,
)

logger = logging.getLogger(__name__)


def create_eda_workflow() -> StateGraph:
    """
    Create the complete EDA workflow graph.
    
    Graph structure:
    ```
    START
      │
      ▼
    context_fusion
      │
      ▼
    planner ◄─────────────────┐
      │                       │
      ▼                       │
    critic ───(rejected)──────┘
      │
      │ (approved)
      ▼
    code_agent ◄──────────────┐
      │                       │
      │ (success)    (error)  │
      ▼                       │
    analyst ──────────────────┘
      │
      ▼
    approval
      │
      ├──(sufficient)──► END
      │
      └──(not sufficient)──► planner (loop back)
    ```
    
    Returns:
        Compiled StateGraph
    """
    # Create graph with state schema
    workflow = StateGraph(EDAState)
    
    # === Add Nodes ===
    workflow.add_node("context_fusion", context_fusion_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("code_agent", code_agent_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("error", error_node)
    
    # === Add Edges ===
    
    # Entry point
    workflow.set_entry_point("context_fusion")
    
    # Context fusion → Planner
    workflow.add_edge("context_fusion", "planner")
    
    # Planner → Critic
    workflow.add_edge("planner", "critic")
    
    # Critic → Planner (rejected) or Code Agent (approved)
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "planner": "planner",
            "code_agent": "code_agent",
        },
    )
    
    # Code Agent → Analyst (success) or retry (error) or Error (max retries)
    workflow.add_conditional_edges(
        "code_agent",
        route_after_code_agent,
        {
            "analyst": "analyst",
            "code_agent": "code_agent",
            "error": "error",
        },
    )
    
    # Analyst → Approval or Code Agent (needs re-run)
    workflow.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {
            "approval": "approval",
            "code_agent": "code_agent",
        },
    )
    
    # Approval → END or Planner (not sufficient)
    workflow.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "end": END,
            "planner": "planner",
        },
    )
    
    # Error → END
    workflow.add_edge("error", END)
    
    return workflow


def compile_workflow(
    checkpointer: MemorySaver | None = None,
) -> Any:
    """
    Compile the workflow with optional checkpointing.
    
    Args:
        checkpointer: Optional checkpointer for state persistence
        
    Returns:
        Compiled graph ready for execution
    """
    workflow = create_eda_workflow()
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)


class EDAWorkflowRunner:
    """
    High-level runner for the EDA workflow.
    
    Provides a simple interface for executing the workflow
    and handling results.
    """
    
    def __init__(self, checkpointer: MemorySaver | None = None):
        """
        Initialize the workflow runner.
        
        Args:
            checkpointer: Optional checkpointer for persistence
        """
        self.graph = compile_workflow(checkpointer)
    
    async def run(
        self,
        question: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run the EDA workflow for a question.
        
        Args:
            question: User's question to analyze
            config: Optional LangGraph config (thread_id, etc.)
            
        Returns:
            Final state with report
        """
        initial_state = create_initial_state(question)
        
        if config is None:
            config = {"configurable": {"thread_id": initial_state["session_id"]}}
        
        logger.info(f"Starting EDA workflow for: {question[:50]}...")
        
        final_state = await self.graph.ainvoke(initial_state, config)
        
        logger.info(f"Workflow complete. Phase: {final_state.get('current_phase')}")
        
        return final_state
    
    async def stream(
        self,
        question: str,
        config: dict[str, Any] | None = None,
    ):
        """
        Stream the EDA workflow execution.
        
        Yields state updates as they occur.
        
        Args:
            question: User's question
            config: Optional config
            
        Yields:
            State updates from each node
        """
        initial_state = create_initial_state(question)
        
        if config is None:
            config = {"configurable": {"thread_id": initial_state["session_id"]}}
        
        async for event in self.graph.astream(initial_state, config):
            yield event
    
    def get_graph_image(self) -> bytes | None:
        """
        Get a visualization of the workflow graph.
        
        Returns:
            PNG image bytes or None if not available
        """
        try:
            return self.graph.get_graph().draw_mermaid_png()
        except Exception as e:
            logger.warning(f"Could not generate graph image: {e}")
            return None
