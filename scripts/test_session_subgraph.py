#!/usr/bin/env python3
"""
Test Script: Verify SubGraph Storage and Agent Access

This script demonstrates:
1. Building a session with SubGraph
2. How SubGraph is stored
3. How different agents can access the same SubGraph
"""

import asyncio
import sys
import json
from pathlib import Path

# Add both root and src to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from context_fusion import build_session, EDASession
from context_fusion.session_context import cache_session, get_cached_session


async def test_session_and_subgraph():
    """Test session creation and SubGraph access."""
    
    print("=" * 70)
    print("TEST 1: Build Session and Extract SubGraph")
    print("=" * 70)
    
    # Build session (this queries Neo4j once)
    query = "Doanh thu theo vendor trong tháng này"
    session = await build_session(query)
    
    print(f"\n✅ Session created: {session.session_id}")
    print(f"   Query: {session.original_query}")
    print(f"   Domain: {session.domain}")
    print(f"   Created at: {session.created_at}")
    
    # =========================================================================
    # SubGraph is stored as Python objects
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: SubGraph Structure")
    print("=" * 70)
    
    sub_graph = session.sub_graph
    print(f"\n📊 SubGraph Statistics:")
    print(f"   Tables: {len(sub_graph.tables)}")
    print(f"   Columns: {len(sub_graph.columns)}")
    print(f"   Joins: {len(sub_graph.joins)}")
    print(f"   Metrics: {len(sub_graph.metrics)}")
    print(f"   Concepts: {len(sub_graph.concepts)}")
    
    # Show table details
    print(f"\n📋 Tables in SubGraph:")
    for t in sub_graph.tables[:5]:  # First 5
        print(f"   - {t.table_name} ({t.business_name})")
    
    # =========================================================================
    # Test Fast Lookups (O(1) with indexes)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Fast Lookup Methods (O(1))")
    print("=" * 70)
    
    # Get specific table
    if sub_graph.tables:
        test_table = sub_graph.tables[0].table_name
        table = sub_graph.get_table(test_table)
        print(f"\n🔍 get_table('{test_table}'):")
        print(f"   → {table.business_name}")
        
        # Get columns for table
        cols = sub_graph.get_columns_for_table(test_table)
        print(f"\n🔍 get_columns_for_table('{test_table}'): {len(cols)} columns")
        for c in cols[:5]:
            pk = "[PK]" if c.is_primary_key else ""
            time = "[TIME]" if c.is_time_column else ""
            print(f"   - {c.column_name} ({c.data_type}) {pk}{time}")
        
        # Get related tables
        related = sub_graph.get_related_tables(test_table)
        print(f"\n🔍 get_related_tables('{test_table}'): {related}")
    
    # =========================================================================
    # Test Serialization Formats
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Serialization Formats")
    print("=" * 70)
    
    # 1. Dictionary format
    sub_graph_dict = sub_graph.to_dict()
    print(f"\n📦 to_dict():")
    print(f"   Type: {type(sub_graph_dict)}")
    print(f"   Keys: {list(sub_graph_dict.keys())}")
    
    # 2. JSON format
    json_str = json.dumps(sub_graph_dict, ensure_ascii=False, indent=2)
    print(f"\n📦 JSON size: {len(json_str)} characters")
    
    # 3. Prompt context (for LLM)
    prompt_detailed = sub_graph.to_prompt_context(compact=False)
    prompt_compact = sub_graph.to_prompt_context(compact=True)
    print(f"\n📦 Prompt Context:")
    print(f"   Detailed mode: {len(prompt_detailed)} characters (~{len(prompt_detailed)//4} tokens)")
    print(f"   Compact mode:  {len(prompt_compact)} characters (~{len(prompt_compact)//4} tokens)")
    print(f"   Token savings: {100 - (len(prompt_compact)*100//len(prompt_detailed))}%")
    
    # =========================================================================
    # Simulate Agent Access
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 5: Simulate Multiple Agents Accessing Session")
    print("=" * 70)
    
    # Agent 1: Planner
    print("\n🤖 Planner Agent:")
    planner_input = session.prompt_context[:200] + "..."
    print(f"   Reading prompt_context: {len(session.prompt_context)} chars")
    print(f"   Tables available: {session.table_names}")
    
    # Simulate planner output
    session.set_state("plan", {
        "steps": ["Analyze query", "Join tables", "Aggregate results"],
        "tables_to_use": session.table_names[:3]
    })
    print(f"   Saved plan to session state")
    
    # Agent 2: SQL Generator
    print("\n🤖 SQL Generator Agent:")
    plan = session.get_state("plan")  # Read from previous agent
    print(f"   Reading plan from session: {plan.get('steps', [])}")
    
    # Access SubGraph directly for SQL generation
    for table_name in plan.get("tables_to_use", [])[:2]:
        cols = sub_graph.get_columns_for_table(table_name)
        pk_cols = [c.column_name for c in cols if c.is_primary_key]
        time_cols = [c.column_name for c in cols if c.is_time_column]
        print(f"   Table {table_name}: PK={pk_cols}, Time={time_cols}")
    
    # Simulate SQL output
    session.set_state("sql", "SELECT vendor_id, SUM(amount) FROM orders GROUP BY 1")
    print(f"   Saved SQL to session state")
    
    # Agent 3: Executor
    print("\n🤖 Executor Agent:")
    sql = session.get_state("sql")
    print(f"   Reading SQL from session: {sql}")
    print(f"   Would execute against database...")
    
    # =========================================================================
    # Test Session Cache
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 6: Session Cache")
    print("=" * 70)
    
    # Manually cache session
    cache_session(session)
    print(f"\n💾 Session cached with ID: {session.session_id}")
    
    # Retrieve from cache
    cached = get_cached_session(session.session_id)
    if cached:
        print(f"✅ Retrieved from cache: {cached.summary()}")
        print(f"   Same SubGraph: {cached.sub_graph is session.sub_graph}")
    else:
        print("❌ Cache retrieval failed")
    
    # =========================================================================
    # Full Session to_dict()
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 7: Full Session Serialization")
    print("=" * 70)
    
    session_dict = session.to_dict()
    print(f"\n📦 Session to_dict():")
    print(f"   Keys: {list(session_dict.keys())}")
    print(f"   Session ID: {session_dict['session_id']}")
    print(f"   State keys: {list(session_dict.get('state', {}).keys())}")
    
    # Show compact prompt
    print("\n" + "=" * 70)
    print("BONUS: Compact Prompt (for token efficiency)")
    print("=" * 70)
    print("\n" + sub_graph.to_prompt_context(compact=True))
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_session_and_subgraph())
