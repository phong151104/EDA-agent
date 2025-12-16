"""
Text-to-SQL + Execute SQL Test Script.

Shows the full pipeline:
1. Query Analysis (intent, keywords, entities)
2. Hybrid Search (vector + fulltext + keyword)
3. Smart Column Selection
4. SQL Generation
5. SQL Execution on PostgreSQL

Run: python scripts/test/test_text_to_sql_execute.py
"""

import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

# Suppress verbose logs
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("asyncpg").setLevel(logging.WARNING)


@dataclass
class StepTiming:
    """Track timing for each step."""
    name: str
    duration_ms: float
    details: str = ""


def format_sql(sql: str) -> str:
    """Format SQL for readability."""
    keywords = ['SELECT', 'FROM', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN',
                'WHERE', 'AND', 'OR', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'ON']
    
    formatted = sql.strip()
    for kw in keywords:
        formatted = re.sub(rf'(?<!^)\b({kw})\b', rf'\n  \1', formatted, flags=re.IGNORECASE)
    
    formatted = re.sub(r'\n+', '\n', formatted)
    return formatted.strip()


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_step(step_num: int, title: str, icon: str = "🔷"):
    """Print a step header."""
    print(f"\n{icon} STEP {step_num}: {title}")
    print("-" * 50)


def print_table(rows: List[Dict], max_rows: int = 10):
    """Print query results as a table."""
    if not rows:
        print("  (No rows returned)")
        return
    
    columns = list(rows[0].keys())
    
    # Calculate column widths
    widths = {}
    for col in columns:
        max_val_len = max(len(str(r.get(col, ""))) for r in rows[:max_rows])
        widths[col] = max(len(col), min(max_val_len, 30))
    
    # Print header
    header = " | ".join(f"{col[:widths[col]]:^{widths[col]}}" for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    print(f"  {header}")
    print(f"  {separator}")
    
    # Print rows
    for i, row in enumerate(rows[:max_rows]):
        values = []
        for col in columns:
            val = str(row.get(col, ""))[:widths[col]]
            values.append(f"{val:<{widths[col]}}")
        print(f"  {' | '.join(values)}")
    
    if len(rows) > max_rows:
        print(f"  ... and {len(rows) - max_rows} more rows")


async def run_full_test(prompt: str):
    """Run text-to-sql with SQL execution."""
    from src.context_fusion import ContextBuilder, QueryRewriter, SchemaRetriever
    from src.mcp.tools import TextToSQL
    from src.mcp.db import Database
    
    timings: List[StepTiming] = []
    
    print_header(f"📝 PROMPT: {prompt}")
    
    # =========================================================================
    # STEP 1: Query Analysis
    # =========================================================================
    print_step(1, "Query Analysis", "🔍")
    
    start = time.perf_counter()
    rewriter = QueryRewriter(use_llm=True)
    analyzed = await rewriter.analyze(prompt)
    duration = (time.perf_counter() - start) * 1000
    timings.append(StepTiming("Query Analysis", duration))
    
    print(f"  Intent:     {analyzed.intent.value}")
    print(f"  Keywords:   {analyzed.keywords[:8]}{'...' if len(analyzed.keywords) > 8 else ''}")
    print(f"  Entities:   {[(e.text, e.entity_type) for e in analyzed.entities[:5]]}")
    print(f"  ⏱️  {duration:.0f}ms")
    
    # =========================================================================
    # STEP 2: Hybrid Search
    # =========================================================================
    print_step(2, "Hybrid Search (Schema Retrieval)", "🔎")
    
    start = time.perf_counter()
    retriever = SchemaRetriever()
    sub_graph = await retriever.retrieve(
        analyzed_query=analyzed,
        domain="vnfilm_ticketing",
        top_k=10,
    )
    duration = (time.perf_counter() - start) * 1000
    timings.append(StepTiming("Hybrid Search", duration))
    
    print(f"  Tables Found: {len(sub_graph.tables)}")
    for t in sub_graph.tables[:5]:
        print(f"    • {t.table_name}")
    if len(sub_graph.tables) > 5:
        print(f"    ... and {len(sub_graph.tables) - 5} more")
    print(f"  ⏱️  {duration:.0f}ms")
    
    retriever.close()
    
    # =========================================================================
    # STEP 3: SQL Generation
    # =========================================================================
    print_step(3, "SQL Generation (LLM)", "🤖")
    
    start = time.perf_counter()
    tool = TextToSQL()
    result = await tool.generate(prompt, sub_graph=sub_graph)
    duration = (time.perf_counter() - start) * 1000
    timings.append(StepTiming("SQL Generation", duration))
    
    if not result.success:
        print(f"  Status: ❌ FAILED")
        print(f"  Error: {result.error}")
        return None
    
    print(f"  Status: ✅ SUCCESS")
    print(f"  Tables Used: {', '.join(result.tables_used or [])}")
    print(f"  ⏱️  {duration:.0f}ms")
    
    print(f"\n  📄 Generated SQL:")
    print("  " + "-" * 46)
    formatted_sql = format_sql(result.sql)
    for line in formatted_sql.split('\n'):
        print(f"  {line}")
    print("  " + "-" * 46)
    
    # =========================================================================
    # STEP 4: SQL Execution
    # =========================================================================
    print_step(4, "SQL Execution (PostgreSQL)", "🗄️")
    
    start = time.perf_counter()
    try:
        rows = await Database.execute(result.sql)
        duration = (time.perf_counter() - start) * 1000
        timings.append(StepTiming("SQL Execution", duration))
        
        print(f"  Status: ✅ SUCCESS")
        print(f"  Rows Returned: {len(rows)}")
        print(f"  ⏱️  {duration:.0f}ms")
        
        # Print results
        print(f"\n  📊 Query Results:")
        print("  " + "-" * 46)
        print_table(rows)
        print("  " + "-" * 46)
        
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        timings.append(StepTiming("SQL Execution", duration))
        
        print(f"  Status: ❌ FAILED")
        print(f"  Error: {e}")
        rows = []
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("⏱️  TIMING SUMMARY", "─")
    
    total_time = sum(t.duration_ms for t in timings)
    for t in timings:
        pct = (t.duration_ms / total_time) * 100 if total_time > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {t.name:20} {t.duration_ms:6.0f}ms  {bar} {pct:4.0f}%")
    
    print(f"  {'─' * 50}")
    print(f"  {'TOTAL':20} {total_time:6.0f}ms")
    
    # Cleanup
    await Database.disconnect()
    
    print()
    return {"sql": result.sql, "rows": rows}


async def interactive_mode():
    """Run in interactive mode."""
    print_header("TEXT-TO-SQL + EXECUTE TEST")
    print("  Commands:")
    print("    'quit' or 'q' - Exit")
    print("    Enter any prompt to test")
    print()
    
    while True:
        try:
            prompt = input("📝 Enter prompt: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['quit', 'q', 'exit']:
                print("👋 Goodbye!")
                break
            
            await run_full_test(prompt)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


async def demo_mode():
    """Run with demo prompts."""
    demo_prompts = [
        "Tổng doanh thu theo rạp chiếu",
        # "Số lượng đơn hàng tháng 1/2025",
        # "Top 5 phim có doanh thu cao nhất",
    ]
    
    print_header("TEXT-TO-SQL + EXECUTE DEMO")
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n{'=' * 70}")
        print(f"  DEMO {i}/{len(demo_prompts)}")
        print(f"{'=' * 70}")
        
        await run_full_test(prompt)
        
        if i < len(demo_prompts):
            print("\n" + "─" * 70)
            input("  Press Enter for next demo...")


async def test_db_connection():
    """Test PostgreSQL connection only."""
    from src.mcp.db import Database
    
    print_header("DATABASE CONNECTION TEST")
    
    try:
        print("  Connecting to PostgreSQL...")
        await Database.connect()
        print("  ✅ Connection successful!")
        
        # Test query
        rows = await Database.execute("SELECT COUNT(*) as count FROM information_schema.tables")
        print(f"  Total tables in DB: {rows[0]['count']}")
        
        # Show schemas
        rows = await Database.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
        """)
        print(f"  Schemas: {[r['schema_name'] for r in rows]}")
        
        await Database.disconnect()
        print("  ✅ Test passed!")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Text-to-SQL + Execute Test")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample prompts")
    parser.add_argument("--prompt", type=str, help="Run with specific prompt")
    parser.add_argument("--test-db", action="store_true", help="Test database connection only")
    args = parser.parse_args()
    
    if args.test_db:
        asyncio.run(test_db_connection())
    elif args.demo:
        asyncio.run(demo_mode())
    elif args.prompt:
        asyncio.run(run_full_test(args.prompt))
    else:
        asyncio.run(interactive_mode())
