"""
Text-to-SQL Verbose Test Script.

Shows detailed step-by-step visualization of the entire pipeline:
1. Query Analysis (intent, keywords, entities)
2. Hybrid Search (vector + fulltext + keyword)
3. Smart Column Selection (with savings metrics)
4. SQL Generation

Run: python scripts/test/test_text_to_sql_verbose.py
"""

import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging to see internal steps
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

# Suppress verbose logs from other modules during test
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


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


async def run_verbose_test(prompt: str):
    """Run text-to-sql with detailed step-by-step output."""
    from src.context_fusion import ContextBuilder, QueryRewriter
    from src.mcp.tools import TextToSQL
    
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
    if analyzed.time_range:
        print(f"  Time Range: {analyzed.time_range}")
    print(f"  ⏱️  {duration:.0f}ms")
    
    # =========================================================================
    # STEP 2: Hybrid Search (Vector + Fulltext + Keyword)
    # =========================================================================
    print_step(2, "Hybrid Search (Vector + Fulltext + Keyword)", "🔎")
    
    start = time.perf_counter()
    from src.context_fusion import SchemaRetriever
    retriever = SchemaRetriever()
    
    # Perform the retrieval (this does all 3 searches)
    sub_graph = await retriever.retrieve(
        analyzed_query=analyzed,
        domain="vnfilm_ticketing",
        top_k=10,
    )
    duration = (time.perf_counter() - start) * 1000
    timings.append(StepTiming("Hybrid Search", duration))
    
    print(f"  Tables Found: {len(sub_graph.tables)}")
    for t in sub_graph.tables:
        print(f"    • {t.table_name} ({t.business_name})")
    
    print(f"\n  Joins Found: {len(sub_graph.joins)}")
    for j in sub_graph.joins:
        on_clause = j.on_clause[0] if j.on_clause else "N/A"
        print(f"    • {j.from_table} → {j.to_table}")
        print(f"      ON {on_clause}")
    
    print(f"  ⏱️  {duration:.0f}ms")
    
    retriever.close()
    
    # =========================================================================
    # STEP 3: Smart Column Selection
    # =========================================================================
    print_step(3, "Smart Column Selection", "📊")
    
    total_cols = sum(1 for c in sub_graph.columns)
    
    # Group columns by table for display
    cols_by_table: Dict[str, List[str]] = {}
    for c in sub_graph.columns:
        if c.table_name not in cols_by_table:
            cols_by_table[c.table_name] = []
        
        markers = []
        if c.is_primary_key:
            markers.append("PK")
        if c.is_time_column:
            markers.append("TIME")
        
        marker_str = f" [{','.join(markers)}]" if markers else ""
        cols_by_table[c.table_name].append(f"{c.column_name}{marker_str}")
    
    print(f"  Columns Selected: {total_cols}")
    for table_name, cols in cols_by_table.items():
        print(f"    {table_name}:")
        for col in cols[:6]:
            print(f"      • {col}")
        if len(cols) > 6:
            print(f"      ... and {len(cols) - 6} more")
    
    # Show column statistics
    print(f"\n  📉 Token Optimization:")
    print(f"     Selected columns shown above")
    print(f"     (Only PK/FK/Time + semantically relevant columns)")
    
    # =========================================================================
    # STEP 4: SQL Generation
    # =========================================================================
    print_step(4, "SQL Generation (LLM)", "🤖")
    
    start = time.perf_counter()
    tool = TextToSQL()
    result = await tool.generate(prompt, sub_graph=sub_graph)
    duration = (time.perf_counter() - start) * 1000
    timings.append(StepTiming("SQL Generation", duration))
    
    if result.success:
        print(f"  Status: ✅ SUCCESS")
        print(f"  Tables Used: {', '.join(result.tables_used or [])}")
        print(f"  ⏱️  {duration:.0f}ms")
        
        print(f"\n  📄 Generated SQL:")
        print("  " + "-" * 46)
        formatted_sql = format_sql(result.sql)
        for line in formatted_sql.split('\n'):
            print(f"  {line}")
        print("  " + "-" * 46)
    else:
        print(f"  Status: ❌ FAILED")
        print(f"  Error: {result.error}")
    
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
    
    print()
    return result


async def interactive_mode():
    """Run in interactive mode."""
    print_header("TEXT-TO-SQL VERBOSE TEST")
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
            
            await run_verbose_test(prompt)
            
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
        "Doanh thu theo vendor Q1 2025",
        # "Số lượng đơn hàng thành công theo rạp chiếu tháng này",
        # "Top 10 khách hàng có doanh thu cao nhất",
    ]
    
    print_header("TEXT-TO-SQL DEMO MODE")
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n{'=' * 70}")
        print(f"  DEMO {i}/{len(demo_prompts)}")
        print(f"{'=' * 70}")
        
        await run_verbose_test(prompt)
        
        if i < len(demo_prompts):
            print("\n" + "─" * 70)
            input("  Press Enter for next demo...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Text-to-SQL Verbose Test")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample prompts")
    parser.add_argument("--prompt", type=str, help="Run with specific prompt")
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(demo_mode())
    elif args.prompt:
        asyncio.run(run_verbose_test(args.prompt))
    else:
        asyncio.run(interactive_mode())
