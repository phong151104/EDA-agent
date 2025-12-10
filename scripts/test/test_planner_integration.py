#!/usr/bin/env python3
"""
Test script for Context Fusion → Planner integration.

Usage:
    python scripts/test/test_planner_integration.py "Tại sao doanh thu giảm?"
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

# Add project root to path (go up 2 levels: test -> scripts -> project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()


async def test_integration(question: str):
    """Test Context Fusion → Planner integration."""
    
    from src.graph.state import create_initial_state
    from src.graph.nodes import context_fusion_node, planner_node
    
    print("=" * 70)
    print("TESTING: Context Fusion → Planner Integration")
    print("=" * 70)
    print(f"\nQuestion: {question}\n")
    
    # Step 1: Create initial state
    print("-" * 70)
    print("STEP 1: Create Initial State")
    print("-" * 70)
    
    state = create_initial_state(question)
    print(f"  Session ID: {state['session_id']}")
    print(f"  Initial Phase: {state['current_phase']}")
    
    # Step 2: Run Context Fusion
    print("\n" + "-" * 70)
    print("STEP 2: Context Fusion Node")
    print("-" * 70)
    
    context_result = await context_fusion_node(state)
    
    # Merge result into state
    state.update(context_result)
    
    print(f"\n  ✅ Context Fusion Complete!")
    print(f"  Phase: {state['current_phase']}")
    
    # Show analyzed query
    analyzed = state.get("analyzed_query", {})
    print(f"\n  📊 Analyzed Query:")
    print(f"     Intent: {analyzed.get('intent', 'N/A')}")
    print(f"     Keywords: {analyzed.get('keywords', [])[:5]}...")
    print(f"     Entities: {len(analyzed.get('entities', []))} found")
    
    # Show sub-graph summary
    sub_graph = state.get("sub_graph", {})
    print(f"\n  📋 Sub-graph:")
    print(f"     Tables: {len(sub_graph.get('tables', []))}")
    print(f"     Columns: {len(sub_graph.get('columns', []))}")
    print(f"     Joins: {len(sub_graph.get('joins', []))}")
    
    # Show tables
    tables = sub_graph.get("tables", [])
    if tables:
        print(f"\n  📑 Tables found:")
        for t in tables[:5]:
            print(f"     - {t.get('table_name')}: {t.get('business_name', '')[:40]}...")
    
    # Show prompt context (first 500 chars)
    prompt_context = state.get("prompt_context", "")
    if prompt_context:
        print(f"\n  📝 Prompt Context (preview):")
        print("-" * 50)
        print(prompt_context[:800])
        if len(prompt_context) > 800:
            print(f"\n... ({len(prompt_context)} chars total)")
        print("-" * 50)
    
    # Step 3: Run Planner (if you want to test full flow)
    print("\n" + "-" * 70)
    print("STEP 3: Planner Node")
    print("-" * 70)
    
    try:
        planner_result = await planner_node(state)
        state.update(planner_result)
        
        print(f"\n  ✅ Planner Complete!")
        print(f"  Phase: {state['current_phase']}")
        
        plan = state.get("current_plan", {})
        print(f"\n  📋 Generated Plan:")
        print(f"     Question: {plan.get('question', 'N/A')[:50]}...")
        print(f"     Hypotheses: {len(plan.get('hypotheses', []))}")
        print(f"     Steps: {len(plan.get('steps', []))}")
        
        # Show hypotheses
        hypotheses = plan.get("hypotheses", [])
        if hypotheses:
            print(f"\n  💡 Hypotheses:")
            for i, h in enumerate(hypotheses[:3], 1):
                title = h.get("title", h.get("hypothesis", "N/A"))
                print(f"     {i}. {title[:60]}...")
        
        # Show steps
        steps = plan.get("steps", [])
        if steps:
            print(f"\n  📝 Analysis Steps:")
            for i, s in enumerate(steps[:5], 1):
                desc = s.get("description", s.get("objective", "N/A"))
                print(f"     {i}. {desc[:60]}...")
        
    except Exception as e:
        print(f"\n  ❌ Planner Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)
    
    return state


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Tại sao doanh thu giảm?"
    asyncio.run(test_integration(question))
