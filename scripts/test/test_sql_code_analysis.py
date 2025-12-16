#!/usr/bin/env python3
"""
Test: SQL → Cache → Enriched Context → Code Interpreter Flow

This test demonstrates the complete dynamic flow:
1. Execute SQL query (any query)
2. Auto-enrich with column metadata from YAML
3. Store enriched data in Redis
4. Code Agent generates Python code with full context
5. Code Interpreter executes with injected DataFrames

Usage:
    python scripts/test/test_sql_code_analysis.py
"""

import asyncio
import sys
import json
import uuid
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(override=True)


def print_header(title: str, char: str = "═"):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def print_section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


async def test_full_flow(question: str):
    """Run the complete flow for any question."""
    
    from src.mcp.server import MCPServer
    from src.mcp.tools.code_interpreter import CodeInterpreter
    from src.metadata_enricher import MetadataEnricher
    from src.cache import SessionCache
    from openai import OpenAI
    from config import config
    
    # Create session
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    cache = SessionCache(session_id)
    mcp = MCPServer()
    enricher = MetadataEnricher()
    interpreter = CodeInterpreter()
    llm = OpenAI(api_key=config.openai.api_key)
    
    print_header(f"SESSION: {session_id}")
    print(f"\n📝 Question: {question}")
    
    # =========================================================================
    # STEP 1: Generate SQL from natural language
    # =========================================================================
    print_section("STEP 1: Text-to-SQL")
    
    sql_result = await mcp.call_tool("text_to_sql", {"prompt": question})
    
    if not sql_result.success:
        print(f"❌ SQL generation failed: {sql_result.error}")
        return
    
    sql = sql_result.output.get("sql", "")
    tables_used = sql_result.output.get("tables_used", [])
    
    print(f"✅ SQL Generated:")
    print(f"   {sql}")
    print(f"   Tables: {tables_used}")
    
    # =========================================================================
    # STEP 2: Execute SQL
    # =========================================================================
    print_section("STEP 2: Execute SQL")
    
    exec_result = await mcp.call_tool("execute_sql", {"query": sql})
    
    if not exec_result.success:
        print(f"❌ SQL execution failed: {exec_result.error}")
        return
    
    rows = exec_result.output.get("rows", [])
    print(f"✅ Executed: {len(rows)} rows returned")
    
    if rows:
        print(f"   Columns: {list(rows[0].keys())}")
        print(f"   Sample: {json.dumps(rows[:2], ensure_ascii=False, default=str)}")
    
    # =========================================================================
    # STEP 3: Enrich with Metadata
    # =========================================================================
    print_section("STEP 3: Enrich with Metadata")
    
    enriched = enricher.enrich(
        sql=sql,
        data=rows,
        tables_hint=tables_used
    )
    
    print(f"✅ Enriched data:")
    print(f"   Tables: {enriched['tables_used']}")
    print(f"   Context:")
    print(f"   {enriched['context_text']}")
    
    # =========================================================================
    # STEP 4: Save to Redis Cache
    # =========================================================================
    print_section("STEP 4: Save to Redis Cache")
    
    step_id = "s1"  # First step
    cache.save_step_result(
        step_id=step_id,
        sql=sql,
        data=enriched["data"],
        column_metadata=enriched["columns"],
        tables_used=enriched["tables_used"],
        context_text=enriched["context_text"],
    )
    
    print(f"✅ Saved step '{step_id}' to cache")
    print(f"   Session keys: {cache.get_all_keys()}")
    
    # =========================================================================
    # STEP 5: Code Agent generates Python code with context
    # =========================================================================
    print_section("STEP 5: Code Agent - Generate Python Code")
    
    # Get enriched data from cache
    enriched_data = cache.get_enriched_step_data([step_id])
    
    code_gen_prompt = f"""Bạn là Code Agent, nhiệm vụ viết Python code để phân tích dữ liệu.

## Câu hỏi: {question}

## Dữ liệu có sẵn:
- `df` (hoặc `df_{step_id}`): DataFrame với {len(rows)} rows

{enriched_data['context']}

## Sample data:
{json.dumps(rows[:2] if rows else [], ensure_ascii=False, default=str)}

## Yêu cầu:
1. Phân tích dữ liệu để trả lời câu hỏi
2. Tính toán các chỉ số quan trọng
3. In kết quả rõ ràng

## Lưu ý:
- pandas, numpy, matplotlib đã được import sẵn
- df đã được load sẵn
- CHỈ viết code, không giải thích

Python code:"""

    print("🤖 Code Agent generating code with context...")
    
    response = llm.chat.completions.create(
        model=config.openai.model,
        messages=[
            {"role": "system", "content": "Bạn là Senior Python Developer. Viết code clean, efficient. CHỈ trả về code."},
            {"role": "user", "content": code_gen_prompt}
        ],
        temperature=0,
        max_tokens=1500
    )
    
    generated_code = response.choices[0].message.content.strip()
    
    # Clean up code
    if generated_code.startswith("```python"):
        generated_code = generated_code[9:]
    if generated_code.startswith("```"):
        generated_code = generated_code[3:]
    if generated_code.endswith("```"):
        generated_code = generated_code[:-3]
    generated_code = generated_code.strip()
    
    print(f"\n📝 Generated Code:")
    print("─" * 40)
    print(generated_code)
    print("─" * 40)
    
    # =========================================================================
    # STEP 6: Execute code with Code Interpreter
    # =========================================================================
    print_section("STEP 6: Code Interpreter - Execute")
    
    exec_output = await interpreter.execute_with_multiple_dataframes(
        code=generated_code,
        dataframes={step_id: rows}
    )
    
    if exec_output.success:
        print(f"✅ Execution Output:")
        print(exec_output.output)
        
        # Save output to cache
        cache.set("analysis_output", {
            "code": generated_code,
            "output": exec_output.output,
            "images": exec_output.images,
        })
    else:
        print(f"❌ Execution failed: {exec_output.error}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("SESSION SUMMARY")
    summary = cache.get_session_summary()
    print(f"""
   Session ID: {summary['session_id']}
   Total keys: {summary['key_count']}
   Keys: {summary['keys']}
   
   Steps completed:
   ├── text_to_sql: ✅
   ├── execute_sql: ✅ ({len(rows)} rows)
   ├── enrich: ✅ (tables: {enriched['tables_used']})
   ├── cache: ✅ 
   ├── code_gen: ✅
   └── execute: {'✅' if exec_output.success else '❌'}
""")
    
    return session_id


async def interactive_mode():
    """Interactive mode - ask any question."""
    print("\n" + "=" * 70)
    print("  EDA Agent - Interactive Test Mode")
    print("  Type your question or 'exit' to quit")
    print("=" * 70)
    
    while True:
        try:
            question = input("\n📝 Your question: ").strip()
            
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            await test_full_flow(question)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single question mode
        question = " ".join(sys.argv[1:])
        asyncio.run(test_full_flow(question))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())
