#!/usr/bin/env python3
"""
Test script for full Planner → Critic → Code Agent flow.

Usage:
    python scripts/test/test_full_flow.py "Tại sao doanh thu giảm?"
"""

import asyncio
import sys
import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("neo4j").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(override=True)


async def test_full_flow(question: str, max_iterations: int = 3):
    """Test full Planner → Critic → Code Agent flow."""
    
    from src.graph.state import create_initial_state
    from src.graph.nodes import context_fusion_node, planner_node, critic_node, code_agent_node
    
    print("=" * 70)
    print("TESTING: Full EDA Agent Flow")
    print("=" * 70)
    print(f"\nQuestion: {question}")
    print(f"Max iterations: {max_iterations}\n")
    
    # Step 1: Create initial state
    state = create_initial_state(question)
    state["max_debate_iterations"] = max_iterations
    
    # =========================================================================
    # Phase 1: Context Fusion
    # =========================================================================
    print("-" * 70)
    print("PHASE 1: CONTEXT FUSION")
    print("-" * 70)
    
    context_result = await context_fusion_node(state)
    state.update(context_result)
    
    tables_count = len(state.get("sub_graph", {}).get("tables", []))
    columns_count = len(state.get("sub_graph", {}).get("columns", []))
    print(f"  ✅ Found {tables_count} tables, {columns_count} columns")
    
    # =========================================================================
    # Phase 2: Planner → Critic Loop
    # =========================================================================
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        print(f"\n{'='*70}")
        print(f"PHASE 2: PLANNING - Iteration {iteration}")
        print("=" * 70)
        
        # Planner generates/refines plan
        print(f"\n--- Planner ---")
        
        if state.get("critic_feedback"):
            print(f"  📝 Received feedback from Critic")
        
        planner_result = await planner_node(state)
        state.update(planner_result)
        
        plan = state.get("current_plan", {})
        print(f"  ✅ Generated plan v{plan.get('version', 1)}")
        print(f"     Hypotheses: {len(plan.get('hypotheses', []))}")
        print(f"     Steps: {len(plan.get('steps', []))}")
        
        # Critic validates
        print(f"\n--- Critic ---")
        
        critic_result = await critic_node(state)
        state.update(critic_result)
        
        validation = state.get("validation_result", {})
        print(f"  Layer 1 (Data): {'✅' if validation.get('layer1_passed') else '❌'}")
        print(f"  Layer 2 (Logic): {'✅' if validation.get('layer2_passed') else '❌'}")
        print(f"  Errors: {validation.get('total_errors', 0)}, Warnings: {validation.get('total_warnings', 0)}")
        
        # Check if approved
        if state.get("plan_approved"):
            print(f"\n  🎉 PLAN APPROVED after {iteration} iteration(s)!")
            break
        else:
            print(f"  ❌ Plan rejected, refining...")
    
    if not state.get("plan_approved"):
        print("\n❌ Plan not approved after max iterations")
        return state
    
    # =========================================================================
    # Phase 3: Code Agent
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: CODE GENERATION")
    print("=" * 70)
    
    code_result = await code_agent_node(state)
    state.update(code_result)
    
    generated_code = state.get("generated_code", [])
    print(f"\n  📝 Generated {len(generated_code)} code blocks:\n")
    
    for code in generated_code:
        step_id = code.get("step_id", "?")
        language = code.get("language", "?")
        desc = code.get("description", "N/A")[:60]
        hypo_id = code.get("hypothesis_id", "?")
        
        print(f"  ┌─ {step_id} ({hypo_id}) [{language.upper()}] {desc}...")
        print(f"  │")
        
        # Show FULL code
        code_content = code.get("code", "")
        if code_content:
            for line in code_content.split("\n"):
                print(f"  │  {line}")
        
        print(f"  └{'─' * 50}\n")
    
    # Check execution results
    execution_results = state.get("execution_results", {})
    all_success = state.get("all_code_success", False)
    
    print(f"\n  📊 Execution Results:")
    for step_id, result in execution_results.items():
        status = result.get("status", "unknown")
        icon = "✅" if status == "success" else "❌"
        print(f"     {icon} {step_id}: {status}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FLOW COMPLETE!")
    print("=" * 70)
    
    final_plan = state.get("current_plan") or {}
    
    print(f"\n📋 Final Summary:")
    print(f"   Plan Version: {final_plan.get('version', 1)}")
    print(f"   Iterations: {state.get('debate_iteration', 0)}")
    print(f"   Hypotheses: {len(final_plan.get('hypotheses', []))}")
    print(f"   Steps: {len(final_plan.get('steps', []))}")
    print(f"   Code Blocks: {len(generated_code)}")
    print(f"   All Success: {'✅' if all_success else '❌'}")
    
    print("\n" + "=" * 70)
    
    return state


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Tại sao doanh thu giảm?"
    asyncio.run(test_full_flow(question))
