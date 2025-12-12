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

# Suppress ALL library logs - only show our custom output
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

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
        issues = state.get("validation_issues", [])
        
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
            print(f"\n   ❌ Errors:")
            for err in errors[:3]:
                print(f"      • {err.get('message', 'Unknown')[:50]}")
        
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
        
        print(f"└{'─' * 70}┘")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("SUMMARY", "═")
    
    all_success = state.get("all_code_success", False)
    
    summary_rows = [
        ["Plan Version", str(final_plan.get("version", 1))],
        ["Iterations", str(state.get("debate_iteration", iteration))],
        ["Hypotheses", str(len(final_plan.get("hypotheses", [])))],
        ["Steps", str(len(final_plan.get("steps", [])))],
        ["Code Blocks", str(len(generated_code))],
        ["All Success", "✅ Yes" if all_success else "❌ No"],
    ]
    
    print()
    for row in summary_rows:
        print(f"   {row[0]:15} : {row[1]}")
    
    print("\n" + "═" * 70)
    
    return state


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Tại sao doanh thu giảm?"
    asyncio.run(test_full_flow(question))
