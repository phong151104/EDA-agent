#!/usr/bin/env python3
"""
Test script for Planner-Critic Communication Loop.

Usage:
    python scripts/test/test_planner_critic_loop.py "Tại sao doanh thu giảm?"
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


async def test_planner_critic_loop(question: str, max_iterations: int = 3):
    """Test Planner → Critic debate loop."""
    
    from src.graph.state import create_initial_state
    from src.graph.nodes import context_fusion_node, planner_node, critic_node
    
    print("=" * 70)
    print("TESTING: Planner-Critic Debate Loop")
    print("=" * 70)
    print(f"\nQuestion: {question}")
    print(f"Max iterations: {max_iterations}\n")
    
    # Step 1: Create initial state
    state = create_initial_state(question)
    state["max_debate_iterations"] = max_iterations
    
    print("-" * 70)
    print("STEP 1: Context Fusion")
    print("-" * 70)
    
    context_result = await context_fusion_node(state)
    state.update(context_result)
    
    print(f"  ✅ Found {len(state.get('sub_graph', {}).get('tables', []))} tables")
    
    # Step 2: Planner → Critic Loop
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print("=" * 70)
        
        # Planner generates/refines plan
        print(f"\n--- Planner (iteration {iteration}) ---")
        
        if state.get("critic_feedback"):
            print(f"  📝 Received feedback from Critic:")
            print(f"     {state['critic_feedback'][:200]}...")
        
        planner_result = await planner_node(state)
        state.update(planner_result)
        
        plan = state.get("current_plan", {})
        print(f"\n  ✅ Generated plan v{plan.get('version', 1)}")
        
        # Print hypotheses
        print(f"\n  💡 Hypotheses ({len(plan.get('hypotheses', []))}):")
        for h in plan.get("hypotheses", []):
            statement = h.get("statement", h.get("hypothesis", "N/A"))
            print(f"     • [{h.get('id', '?')}] {statement[:70]}...")
            if h.get("rationale"):
                print(f"       Rationale: {h.get('rationale', '')[:50]}...")
        
        # Print steps
        print(f"\n  📝 Analysis Steps ({len(plan.get('steps', []))}):")
        for step in plan.get("steps", []):
            sn = step.get("step_number", step.get("stepNumber", "?"))
            desc = step.get("description", "N/A")
            action = step.get("action_type", step.get("actionType", "sql"))
            print(f"     {sn}. [{action}] {desc[:60]}...")
            details = step.get("details", {})
            if details.get("tables"):
                print(f"        Tables: {details['tables']}")
            if details.get("sql"):
                print(f"        SQL: {details['sql'][:50]}...")
        
        # Critic validates
        print(f"\n--- Critic (iteration {iteration}) ---")
        
        critic_result = await critic_node(state)
        state.update(critic_result)
        
        validation = state.get("validation_result", {})
        print(f"  Status: {validation.get('status', 'N/A')}")
        print(f"  Score: {validation.get('approval_score', 0):.2f}")
        print(f"  Layer 1 (Data): {'✅' if validation.get('layer1_passed') else '❌'}")
        print(f"  Layer 2 (Logic): {'✅' if validation.get('layer2_passed') else '❌'}")
        print(f"  Layer 3 (Biz): {'✅' if validation.get('layer3_passed') else '⏭️'}")
        print(f"  Errors: {validation.get('total_errors', 0)}, Warnings: {validation.get('total_warnings', 0)}")
        
        # Check if approved
        if state.get("plan_approved"):
            print(f"\n  🎉 PLAN APPROVED after {iteration} iteration(s)!")
            break
        else:
            print(f"  ❌ Plan rejected, will refine...")
            if iteration < max_iterations:
                feedback = state.get("critic_feedback", "")
                if feedback:
                    print(f"\n  Feedback for Planner:")
                    # Print formatted feedback
                    for line in feedback.split("\n")[:10]:
                        print(f"    {line}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    
    final_plan = state.get("current_plan", {})
    print(f"\n📋 Final Plan:")
    print(f"   Version: {final_plan.get('version', 1)}")
    print(f"   Approved: {state.get('plan_approved', False)}")
    print(f"   Iterations: {state.get('debate_iteration', 0)}")
    
    print(f"\n💡 Hypotheses ({len(final_plan.get('hypotheses', []))}):")
    for i, h in enumerate(final_plan.get("hypotheses", []), 1):
        stmt = h.get("statement", h.get("hypothesis", "N/A"))
        h_id = h.get("id", f"h{i}")
        print(f"   [{h_id}] {stmt[:70]}...")
    
    print(f"\n📝 Analysis Steps ({len(final_plan.get('steps', []))}):")
    
    # Group steps by hypothesis
    steps_by_hypo = {}
    for s in final_plan.get("steps", []):
        h_id = s.get("hypothesis_id", "unknown")
        steps_by_hypo.setdefault(h_id, []).append(s)
    
    for h_id, steps in steps_by_hypo.items():
        print(f"\n   [{h_id}]")
        for s in steps:
            s_id = s.get("id", "?")
            desc = s.get("description", "N/A")[:55]
            action = s.get("action_type", "?")
            deps = s.get("depends_on", [])
            deps_str = f" (→{deps[0]})" if deps else ""
            print(f"      {s_id}. [{action}] {desc}...{deps_str}")
            
            # Show requirements
            reqs = s.get("requirements", {})
            if reqs.get("tables_hint"):
                print(f"          Tables: {reqs['tables_hint']}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)
    
    return state


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Tại sao doanh thu giảm?"
    asyncio.run(test_planner_critic_loop(question))
