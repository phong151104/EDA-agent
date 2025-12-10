#!/usr/bin/env python3
"""
Test script for Context Fusion Layer.

Usage:
    python scripts/test/test_context_fusion.py "Tại sao doanh thu giảm?"
"""

import asyncio
import sys
import os
import warnings
import logging

# Suppress Neo4j verbose warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*neo4j.*")

# Set log level for neo4j to ERROR only
logging.getLogger("neo4j").setLevel(logging.ERROR)

# Add project root to path (go up 2 levels: test -> scripts -> project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(override=True)  # Force reload

from src.context_fusion import (
    QueryRewriter,
    SchemaRetriever,
    ContextBuilder,
    quick_analyze,
)


async def test_query_rewriter(query: str):
    """Test Query Rewriter alone."""
    print("\n" + "="*60)
    print("QUERY REWRITER TEST")
    print("="*60)
    
    # Quick analysis (no LLM)
    print("\n1. Quick Analysis (no LLM):")
    result = quick_analyze(query)
    print(f"   Intent: {result.intent.value}")
    print(f"   Keywords: {result.keywords}")
    print(f"   Entities: {[(e.text, e.entity_type) for e in result.entities]}")
    print(f"   Search Terms: {result.get_search_terms()}")
    
    # Full analysis with LLM
    print("\n2. Full Analysis (with LLM):")
    rewriter = QueryRewriter(use_llm=True)
    try:
        result = await rewriter.analyze(query)
        print(f"   Intent: {result.intent.value}")
        print(f"   Keywords: {result.keywords}")
        print(f"   Entities: {[(e.text, e.entity_type, e.normalized_name) for e in result.entities]}")
        print(f"   Rewritten: {result.rewritten_query}")
    except Exception as e:
        print(f"   LLM Error: {e}")


async def test_schema_retriever(query: str, domain: str = "vnfilm_ticketing"):
    """Test Schema Retriever with Neo4j."""
    print("\n" + "="*60)
    print("SCHEMA RETRIEVER TEST")
    print("="*60)
    
    # First analyze query
    analyzed = quick_analyze(query)
    print(f"\nSearch terms: {analyzed.get_search_terms()}")
    
    # Retrieve sub-graph
    retriever = SchemaRetriever()
    try:
        sub_graph = await retriever.retrieve(
            analyzed,
            domain=domain,
            top_k=5,
        )
        
        print(f"\nSub-graph retrieved:")
        print(f"   Tables: {[t.table_name for t in sub_graph.tables]}")
        print(f"   Columns: {len(sub_graph.columns)} columns")
        print(f"   Joins: {[(j.from_table, j.to_table) for j in sub_graph.joins]}")
        print(f"   Metrics: {[m.name for m in sub_graph.metrics]}")
        print(f"   Concepts: {[c.name for c in sub_graph.concepts]}")
        
        print("\n--- Prompt Context ---")
        print(sub_graph.to_prompt_context()[:2000])
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        retriever.close()


async def test_full_context(query: str, domain: str = "vnfilm_ticketing"):
    """Test full Context Builder pipeline."""
    print("\n" + "="*60)
    print("FULL CONTEXT BUILDER TEST")
    print("="*60)
    
    print(f"\nQuery: {query}")
    
    builder = ContextBuilder(use_llm=False, domain=domain)
    try:
        context = await builder.build(query)
        
        print(f"\n--- Enriched Context ---")
        print(f"Intent: {context.analyzed_query.intent.value}")
        print(f"Tables: {context.sub_graph.get_table_names()}")
        print(f"Columns: {len(context.sub_graph.columns)}")
        print(f"Joins: {len(context.sub_graph.joins)}")
        print(f"Metrics: {len(context.sub_graph.metrics)}")
        
        print("\n--- Full Prompt Context ---")
        print(context.prompt_context[:3000])
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        builder.close()


async def main():
    """Main entry point."""
    # Default test query
    query = "Tại sao doanh thu giảm?"
    
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    print(f"\n{'#'*60}")
    print(f"# Testing Context Fusion Layer")
    print(f"# Query: {query}")
    print(f"{'#'*60}")
    
    # Test each component
    await test_query_rewriter(query)
    await test_schema_retriever(query)
    await test_full_context(query)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
