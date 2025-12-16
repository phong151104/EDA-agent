#!/usr/bin/env python3
"""
Test script for full Planner → Critic → Code Agent flow.

Shows clean input/output at each phase instead of detailed processing logs.

Usage:
    python scripts/test/test_full_flow.py "Tại sao doanh thu giảm?"
"""

import asyncio
import sys
import os
import warnings
import logging

# Configure logging - enable for debugging
warnings.filterwarnings("ignore")
# logging.disable(logging.CRITICAL)  # Disabled to see errors
logging.basicConfig(
    level=logging.WARNING,  # Show WARNING and above (less noise than DEBUG/INFO)
    format='%(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(override=True)


def print_header(title: str, char: str = "═"):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def print_section(title: str):
    """Print a section title."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def print_table(headers: list, rows: list):
    """Print a simple ASCII table."""
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    # Header
    header_line = "│ " + " │ ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " │"
    border = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    separator = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    bottom = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
    
    print(border)
    print(header_line)
    print(separator)
    for row in rows:
        print("│ " + " │ ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))) + " │")
    print(bottom)


async def test_full_flow(question: str, max_iterations: int = 3):
    """Test full Planner → Critic → Code Agent flow with clean output."""
    
    from src.graph.state import create_initial_state
    from src.graph.nodes import context_fusion_node, planner_node, critic_node, code_agent_node
    
    print_header("EDA AGENT - FULL FLOW TEST")
    print(f"\n📝 Question: {question}")
    print(f"⚙️  Max iterations: {max_iterations}")
    
    # Create initial state
    state = create_initial_state(question)
    state["max_debate_iterations"] = max_iterations
    
    # =========================================================================
    # PHASE 1: CONTEXT FUSION
    # =========================================================================
    print_header("PHASE 1: CONTEXT FUSION", "─")
    
    print("\n📥 INPUT:")
    print(f"   Question: \"{question}\"")
    
    context_result = await context_fusion_node(state)
    state.update(context_result)
    
    sub_graph = state.get("sub_graph", {})
    tables = sub_graph.get("tables", [])
    columns = sub_graph.get("columns", [])
    joins = sub_graph.get("joins", [])
    analyzed_query = state.get("analyzed_query", {})
    
    print("\n📤 OUTPUT:")
    print(f"   Intent: {analyzed_query.get('intent', 'N/A')}")
    print(f"   Keywords: {analyzed_query.get('keywords', [])[:5]}")
    # Format entities safely
    entities = analyzed_query.get('entities', [])[:3]
    entities_str = []
    for e in entities:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            entities_str.append(f"({e[0]}, {e[1]})")
        elif isinstance(e, dict):
            entities_str.append(f"({e.get('text', '?')}, {e.get('type', '?')})")
        else:
            entities_str.append(str(e))
    print(f"   Entities: {entities_str}")
    print(f"\n   📊 SubGraph:")
    print(f"      Tables: {len(tables)}")
    print(f"      Columns: {len(columns)}")
    print(f"      Joins: {len(joins)}")
    
    if tables:
        print(f"\n   📋 Top Tables:")
        for t in tables[:5]:
            name = t.get("table_name", t) if isinstance(t, dict) else t
            print(f"      • {name}")
    
    # =========================================================================
    # PHASE 2: PLANNER → CRITIC LOOP
    # =========================================================================
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        print_header(f"PHASE 2: PLANNING (Iteration {iteration}/{max_iterations})", "─")
        
        # --- Planner ---
        print("\n🎯 PLANNER")
        print(f"   📥 Input: SubGraph with {len(tables)} tables")
        if state.get("critic_feedback"):
            print(f"   📝 Feedback: {len(state.get('validation_issues', []))} issues to fix")
        
        planner_result = await planner_node(state)
        state.update(planner_result)
        
        plan = state.get("current_plan", {})
        hypotheses = plan.get("hypotheses", [])
        steps = plan.get("steps", [])
        
        print(f"\n   📤 Output: Plan v{plan.get('version', 1)}")
        
        # Show hypotheses
        print(f"\n   💡 Hypotheses ({len(hypotheses)}):")
        for h in hypotheses[:4]:
            h_id = h.get("id", "?")
            statement = h.get("statement", h.get("title", "N/A"))[:60]
            print(f"      [{h_id}] {statement}")
        
        # Show steps grouped by type
        step_types = {}
        for s in steps:
            action = s.get("action_type", "unknown")
            step_types[action] = step_types.get(action, 0) + 1
        
        print(f"\n   📋 Steps ({len(steps)}): {step_types}")
        
        # --- Critic ---
        print(f"\n   🔍 CRITIC")
        print(f"   📥 Input: Plan with {len(hypotheses)} hypotheses, {len(steps)} steps")
        
        critic_result = await critic_node(state)
        state.update(critic_result)
        
        validation = state.get("validation_result", {})
        issues = validation.get("issues", [])  # Issues is inside validation_result
        
        layer1 = "✅" if validation.get("layer1_passed") else "❌"
        layer2 = "✅" if validation.get("layer2_passed") else "❌"
        layer3 = "✅" if validation.get("layer3_passed", True) else "❌"
        
        print(f"\n   📤 Output:")
        print(f"      Layer 1 (Data):   {layer1}")
        print(f"      Layer 2 (Logic):  {layer2}")
        print(f"      Layer 3 (Biz):    {layer3}")
        print(f"      Errors: {validation.get('total_errors', 0)}, Warnings: {validation.get('total_warnings', 0)}")
        
        # Show issues if any
        errors = [i for i in issues if i.get("severity") == "error"]
        if errors:
            print(f"\n   ❌ Errors ({len(errors)}):")
            for err in errors[:6]:  # Show up to 6 errors
                layer = err.get('layer', '?')
                msg = err.get('message', 'Unknown')[:80]
                item_id = err.get('hypothesis_id') or err.get('step_id') or ''
                print(f"      [{layer}] ({item_id}) {msg}")
        
        # Check approval
        if state.get("plan_approved"):
            print(f"\n   ✅ PLAN APPROVED!")
            break
        else:
            print(f"\n   ⏳ Plan rejected, refining...")
    
    if not state.get("plan_approved"):
        print("\n❌ Plan not approved after max iterations")
        return state
    
    # =========================================================================
    # PHASE 3: CODE AGENT
    # =========================================================================
    print_header("PHASE 3: CODE GENERATION", "─")
    
    final_plan = state.get("current_plan", {})
    print(f"\n📥 INPUT:")
    print(f"   Plan: v{final_plan.get('version', 1)} with {len(final_plan.get('steps', []))} steps")
    print(f"   Schema: {len(tables)} tables, {len(columns)} columns")
    
    code_result = await code_agent_node(state)
    state.update(code_result)
    
    generated_code = state.get("generated_code", [])
    execution_results = state.get("execution_results", {})
    
    print(f"\n📤 OUTPUT: {len(generated_code)} code blocks\n")
    
    for code in generated_code:
        step_id = code.get("step_id", "?")
        hypo_id = code.get("hypothesis_id", "?")
        language = code.get("language", "?").upper()
        desc = code.get("description", "N/A")
        code_content = code.get("code", "")
        
        exec_result = execution_results.get(step_id, {})
        status = exec_result.get("status", "unknown")
        icon = "✅" if status == "success" else "❌"
        exec_time = exec_result.get("execution_time_ms", 0)
        
        print(f"\n┌{'─' * 70}┐")
        print(f"│ {icon} [{step_id}] ({hypo_id}) {language:6} │ {desc[:55]}")
        print(f"├{'─' * 70}┤")
        
        # Show code (truncated to 10 lines for readability)
        lines = code_content.split("\n")
        for i, line in enumerate(lines[:10]):
            print(f"│ {line[:68]}")
        if len(lines) > 10:
            print(f"│ ... ({len(lines) - 10} more lines)")
        
        # Show execution output if available
        output = exec_result.get("output", {})
        if isinstance(output, dict):
            stdout = output.get("stdout", "")
            images = output.get("images", [])
            
            if stdout or images:
                print(f"├{'─' * 70}┤")
                print(f"│ 🔧 EXECUTION OUTPUT ({exec_time}ms):")
                
                if stdout:
                    for line in str(stdout)[:200].split("\n")[:5]:
                        print(f"│   {line[:65]}")
                    if len(stdout) > 200:
                        print(f"│   ... (truncated)")
                
                if images:
                    print(f"│   📊 Generated {len(images)} image(s)")
                    # Save images to output folder
                    import base64
                    import os
                    output_dir = "scripts/test/outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    for idx, img_b64 in enumerate(images):
                        img_path = f"{output_dir}/{step_id}_chart_{idx+1}.png"
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(img_b64))
                        print(f"│   💾 Saved: {img_path}")
        
        print(f"└{'─' * 70}┘")
    
    # =========================================================================
    # PHASE 4: ANALYST AGENT - Hypothesis Verification
    # =========================================================================
    print_header("PHASE 4: ANALYST AGENT - HYPOTHESIS VERIFICATION", "─")
    
    from openai import OpenAI
    from config import config
    
    llm = OpenAI(api_key=config.openai.api_key)
    
    # Collect all outputs from Code Agent
    all_outputs = []
    for code in generated_code:
        step_id = code.get("step_id", "?")
        hypo_id = code.get("hypothesis_id", "?")
        exec_result = execution_results.get(step_id, {})
        output = exec_result.get("output", {})
        
        if isinstance(output, dict):
            stdout = output.get("stdout", "")
            sql_data = output.get("sql_data", [])
        else:
            stdout = str(output)
            sql_data = []
        
        all_outputs.append({
            "step_id": step_id,
            "hypothesis_id": hypo_id,
            "code": code.get("code", "")[:500],
            "output": str(stdout)[:800] if stdout else str(sql_data)[:800],
        })
    
    # Get hypotheses
    hypotheses = final_plan.get("hypotheses", [])
    
    print(f"\n📥 INPUT:")
    print(f"   • {len(hypotheses)} hypotheses to verify")
    print(f"   • {len(all_outputs)} step outputs from Code Agent")
    
    # Build Analyst prompt
    hypothesis_text = "\n".join([
        f"- [{h.get('id')}] {h.get('statement', h.get('title', 'N/A'))}"
        for h in hypotheses
    ])
    
    outputs_text = "\n".join([
        f"### Step {o['step_id']} (Hypothesis: {o['hypothesis_id']}):\n```\n{o['output'][:400]}\n```"
        for o in all_outputs
    ])
    
    analyst_prompt = f"""Bạn là Senior Data Analyst chuyên phân tích doanh thu và đưa ra chiến lược kinh doanh.

## Câu hỏi gốc: {question}

## Các giả thuyết đã kiểm tra:
{hypothesis_text}

## Kết quả phân tích từ Code Agent:
{outputs_text}

## YÊU CẦU PHÂN TÍCH:

### 1. Xác thực từng giả thuyết
Với MỖI giả thuyết, phân loại:
- ✅ **VALID** - Dữ liệu chứng minh đúng
- ❌ **INVALID** - Dữ liệu bác bỏ
- ⚠️ **INCONCLUSIVE** - Không đủ dữ liệu

### 2. Phân tích xu hướng
- So sánh doanh thu giữa các tháng (tăng/giảm bao nhiêu %)
- Xác định tháng cao nhất/thấp nhất
- Nhận diện pattern (cuối tuần, đầu tháng, v.v.)

### 3. Tìm nguyên nhân gốc rễ
- Liệt kê 2-3 nguyên nhân chính dựa trên dữ liệu
- Đưa ra bằng chứng cụ thể từ số liệu

### 4. Đề xuất hành động cho THÁNG TỚI (QUAN TRỌNG)
- Min 3 recommendations cụ thể, có thể thực hiện ngay
- Mỗi recommendation phải có: Hành động + Mục tiêu + KPI đo lường
- Ví dụ: "Tăng số suất chiếu phim hot 20% → mục tiêu tăng 15% doanh thu vé"

## FORMAT JSON:
```json
{{
  "verified_hypotheses": [
    {{"id": "H1", "status": "VALID|INVALID|INCONCLUSIVE", "evidence": "Căn cứ cụ thể...", "confidence": 0.0-1.0}},
  ],
  "trend_analysis": {{
    "month_comparison": "Mô tả so sánh giữa các tháng với % cụ thể",
    "peak_month": "Tháng cao nhất",
    "lowest_month": "Tháng thấp nhất", 
    "trend_direction": "up|down|stable",
    "percentage_change": "X%"
  }},
  "root_causes": [
    {{"cause": "Nguyên nhân 1", "evidence": "Bằng chứng từ dữ liệu", "impact": "high|medium|low"}},
  ],
  "key_insights": ["Insight quan trọng 1", "Insight 2"],
  "summary": "Tóm tắt tổng quan kết quả phân tích",
  "recommendations": [
    {{"action": "Hành động cụ thể", "target": "Mục tiêu đạt được", "kpi": "Cách đo lường", "priority": "high|medium|low"}},
  ]
}}
```

Trả lời bằng JSON:"""

    print("\n🤖 Analyst Agent đang xác thực...")
    
    response = llm.chat.completions.create(
        model=config.openai.model,
        messages=[
            {"role": "system", "content": "Bạn là Senior Data Analyst. Phân tích kết quả và xác thực giả thuyết. Trả lời bằng JSON."},
            {"role": "user", "content": analyst_prompt}
        ],
        temperature=0.2,
        max_tokens=2000
    )
    
    analyst_output = response.choices[0].message.content.strip()
    
    # Try parse JSON
    import json
    try:
        # Extract JSON from response
        if "```json" in analyst_output:
            json_str = analyst_output.split("```json")[1].split("```")[0]
        elif "```" in analyst_output:
            json_str = analyst_output.split("```")[1].split("```")[0]
        else:
            json_str = analyst_output
        
        analyst_result = json.loads(json_str)
    except:
        analyst_result = {"raw_output": analyst_output, "verified_hypotheses": []}
    
    # Save to state
    state["analyst_result"] = analyst_result
    
    print(f"\n📤 OUTPUT:")
    
    # Show hypothesis verification results
    verified = analyst_result.get("verified_hypotheses", [])
    
    valid_count = sum(1 for h in verified if h.get("status") == "VALID")
    invalid_count = sum(1 for h in verified if h.get("status") == "INVALID")
    inconclusive_count = sum(1 for h in verified if h.get("status") == "INCONCLUSIVE")
    
    print(f"\n   📊 HYPOTHESIS VERIFICATION:")
    print(f"      ✅ Valid:       {valid_count}")
    print(f"      ❌ Invalid:     {invalid_count}")
    print(f"      ⚠️  Inconclusive: {inconclusive_count}")
    
    if verified:
        print(f"\n   📋 DETAILS:")
        for h in verified:
            h_id = h.get("id", "?")
            status = h.get("status", "?")
            evidence = h.get("evidence", "N/A")[:80]
            confidence = h.get("confidence", 0)
            
            if status == "VALID":
                icon = "✅"
            elif status == "INVALID":
                icon = "❌"
            else:
                icon = "⚠️"
            
            print(f"\n      {icon} [{h_id}] {status} (confidence: {confidence:.0%})")
            print(f"         └─ {evidence}")
    
    # Show key insights
    insights = analyst_result.get("key_insights", [])
    if insights:
        print(f"\n   💡 KEY INSIGHTS:")
        for i, insight in enumerate(insights[:3], 1):
            print(f"      {i}. {insight}")
    
    # Show summary
    summary = analyst_result.get("summary", "")
    if summary:
        print(f"\n   📝 SUMMARY:")
        print(f"      {summary}")
    # Show trend analysis if available
    trend = analyst_result.get("trend_analysis", {})
    if trend:
        print(f"\n   📈 TREND ANALYSIS:")
        print(f"      • {trend.get('month_comparison', '')}")
        print(f"      • Peak: {trend.get('peak_month', '?')} | Lowest: {trend.get('lowest_month', '?')}")
        print(f"      • Trend: {trend.get('trend_direction', '?')} ({trend.get('percentage_change', '?')})")
    
    # Show root causes if available
    root_causes = analyst_result.get("root_causes", [])
    if root_causes:
        print(f"\n   🔍 ROOT CAUSES:")
        for rc in root_causes[:3]:
            if isinstance(rc, dict):
                print(f"      • [{rc.get('impact', '?').upper()}] {rc.get('cause', '')}")
                print(f"        Evidence: {rc.get('evidence', '')[:60]}")
            else:
                print(f"      • {rc}")
    
    # Show recommendations (enhanced format)
    recommendations = analyst_result.get("recommendations", [])
    if recommendations:
        print(f"\n   🎯 RECOMMENDATIONS FOR NEXT MONTH:")
        for i, r in enumerate(recommendations[:5], 1):
            if isinstance(r, dict):
                priority = r.get('priority', 'medium')
                priority_icon = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
                print(f"\n      {priority_icon} [{i}] {r.get('action', '')}")
                print(f"         → Target: {r.get('target', '')}")
                print(f"         → KPI: {r.get('kpi', '')}")
            else:
                print(f"      • {r}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_header("FINAL SUMMARY", "═")
    
    all_success = state.get("all_code_success", False)
    
    summary_rows = [
        ["Plan Version", str(final_plan.get("version", 1))],
        ["Iterations", str(state.get("debate_iteration", iteration))],
        ["Hypotheses", str(len(final_plan.get("hypotheses", [])))],
        ["Steps Executed", str(len(generated_code))],
        ["Code Success", "✅ Yes" if all_success else "❌ No"],
        ["─" * 15, "─" * 20],
        ["Valid Hypotheses", f"✅ {valid_count}"],
        ["Invalid Hypotheses", f"❌ {invalid_count}"],
        ["Inconclusive", f"⚠️  {inconclusive_count}"],
    ]
    
    print()
    for row in summary_rows:
        print(f"   {row[0]:20} : {row[1]}")
    
    print("\n" + "═" * 70)
    
    return state


if __name__ == "__main__":
    # Check for command line argument
    if len(sys.argv) > 1:
        # Single run mode with CLI argument
        question = sys.argv[1]
        asyncio.run(test_full_flow(question))
    else:
        # Interactive mode
        print_header("EDA AGENT - INTERACTIVE TEST")
        print("  Enter a question to test the full flow.")
        print("  Type 'quit' or 'q' to exit.\n")
        
        while True:
            try:
                question = input("📝 Enter prompt: ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ['quit', 'q', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                asyncio.run(test_full_flow(question))
                print("\n" + "=" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

