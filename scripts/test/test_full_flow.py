"""
Test Full EDA Workflow using LangGraph.

Now uses EDAWorkflowRunner for proper 2-phase flow:
  Phase 1 (Exploration): 2-3 overview hypotheses → exploration_summary
  Phase 2 (Deep Dive): 5-6 detailed hypotheses based on Phase 1 findings
"""

import asyncio
import sys
import os
import base64
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup config
os.environ.setdefault("DOMAIN", "vnfilm_ticketing")

from src.graph.workflow import EDAWorkflowRunner


def print_header(title: str, char: str = "═"):
    """Print a formatted header."""
    width = 70
    border = char * width
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}")


def print_subheader(title: str):
    """Print a subheader."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def save_images(output: dict, step_id: str, output_dir: Path, iteration_prefix: str = ""):
    """Save base64 images to output folder with iteration prefix."""
    if not output or not isinstance(output, dict):
        return
    
    images = output.get("images", [])
    for i, img_data in enumerate(images, 1):
        if img_data:
            try:
                img_bytes = base64.b64decode(img_data)
                # Add iteration prefix to avoid overwriting
                prefix = f"{iteration_prefix}_" if iteration_prefix else ""
                filename = f"{prefix}{step_id}_chart_{i}.png"
                filepath = output_dir / filename
                filepath.write_bytes(img_bytes)
                print(f"   💾 Saved: {filepath}")
            except Exception as e:
                print(f"   ⚠️ Failed to save image: {e}")


async def test_langgraph_workflow(question: str):
    """Test using the actual LangGraph workflow with 2-phase flow."""
    
    print_header("EDA AGENT - LANGGRAPH WORKFLOW TEST")
    print(f"\n📝 Question: {question}")
    print(f"🔄 Two-Phase Analysis: Exploration → Deep Dive")
    
    # Create output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Create workflow runner
    runner = EDAWorkflowRunner()
    
    # Track events
    events_by_node = {}
    current_phase = "exploration"
    phase_iteration = {"exploration": 0, "deep_dive": 0}
    
    try:
        async for event in runner.stream(question):
            # event is a dict with node name as key
            for node_name, node_output in event.items():
                if node_name == "__start__":
                    continue
                
                # Track phase changes
                analysis_phase = node_output.get("analysis_phase", current_phase)
                if analysis_phase != current_phase:
                    print_subheader(f"🔄 PHASE TRANSITION: {current_phase.upper()} → {analysis_phase.upper()}")
                    current_phase = analysis_phase
                
                # Display node output based on type
                if node_name == "context_fusion":
                    print_subheader("📥 CONTEXT FUSION")
                    sub_graph = node_output.get("sub_graph", {})
                    analyzed = node_output.get("analyzed_query", {})
                    print(f"   Intent: {analyzed.get('intent', 'N/A')}")
                    print(f"   Tables: {len(sub_graph.get('tables', []))}")
                    print(f"   Columns: {len(sub_graph.get('columns', []))}")
                    
                elif node_name == "planner":
                    phase_iteration[current_phase] = phase_iteration.get(current_phase, 0) + 1
                    phase_name = "EXPLORATION" if current_phase == "exploration" else "DEEP DIVE"
                    print_subheader(f"🎯 PLANNER [{phase_name}] (Iteration {phase_iteration[current_phase]})")
                    
                    plan = node_output.get("current_plan", {})
                    hypotheses = plan.get("hypotheses", [])
                    steps = plan.get("steps", [])
                    
                    print(f"   Plan Version: {plan.get('version', 1)}")
                    print(f"   Hypotheses: {len(hypotheses)}")
                    for h in hypotheses[:6]:
                        h_id = h.get("id", "?")
                        statement = h.get("statement", h.get("title", ""))[:60]
                        print(f"      [{h_id}] {statement}")
                    print(f"   Steps: {len(steps)}")
                    
                elif node_name == "critic":
                    print_subheader("🔍 CRITIC")
                    validation = node_output.get("validation_result", {})
                    approved = node_output.get("plan_approved", False)
                    print(f"   Layer 1 (Data): {'✅' if validation.get('layer1_passed') else '❌'}")
                    print(f"   Layer 2 (Logic): {'✅' if validation.get('layer2_passed') else '❌'}")
                    print(f"   Approved: {'✅ YES' if approved else '❌ NO'}")
                    
                elif node_name == "code_agent":
                    print_subheader("💻 CODE AGENT")
                    generated = node_output.get("generated_code", [])
                    results = node_output.get("execution_results", {})
                    
                    print(f"   Generated: {len(generated)} code blocks")
                    print(f"   Executed: {len(results)} steps")
                    
                    # Show each step briefly
                    for code in generated[:8]:
                        step_id = code.get("step_id", "?")
                        hypo_id = code.get("hypothesis_id", "?")
                        lang = code.get("language", "?")
                        desc = code.get("description", "")[:50]
                        
                        step_result = results.get(step_id, {})
                        status = step_result.get("status", "unknown")
                        # Show warning for failed steps instead of error
                        if status == "success":
                            status_icon = "✅"
                        else:
                            status_icon = "⚠️"  # Show as "chưa xác minh" not error
                        
                        print(f"      {status_icon} [{step_id}] ({hypo_id}) {lang.upper()}: {desc}")
                        
                        # Save images with iteration prefix
                        output = step_result.get("output", {})
                        if isinstance(output, dict):
                            # Build iteration prefix: e.g., "exp1" or "dd2"
                            phase_prefix = "exp" if current_phase == "exploration" else "dd"
                            iter_num = phase_iteration.get(current_phase, 1)
                            iteration_prefix = f"{phase_prefix}{iter_num}"
                            save_images(output, step_id, output_dir, iteration_prefix)
                    
                elif node_name == "analyst":
                    phase_name = "EXPLORATION" if current_phase == "exploration" else "DEEP DIVE"
                    print_subheader(f"📊 ANALYST [{phase_name}]")
                    
                    summary = node_output.get("analysis_summary", "")
                    exploration_summary = node_output.get("exploration_summary")
                    
                    # Show exploration summary if Phase 1
                    if exploration_summary:
                        print("   📋 Exploration Summary (for Phase 2):")
                        if isinstance(exploration_summary, dict):
                            findings = exploration_summary.get("key_findings", [])
                            for f in findings[:5]:
                                print(f"      • {str(f)[:80]}")
                            trends = exploration_summary.get("trends", [])
                            for t in trends[:3]:
                                print(f"      📈 {str(t)[:60]}")
                        else:
                            print(f"      {str(exploration_summary)[:200]}")
                    
                    # Show summary (truncated)
                    if summary:
                        summary_lines = summary.split('\n')[:10]
                        print("   📝 Summary:")
                        for line in summary_lines:
                            if line.strip():
                                print(f"      {line.strip()[:80]}")
                    
                elif node_name == "approval":
                    print_subheader("✅ APPROVAL")
                    is_sufficient = node_output.get("is_insight_sufficient", False)
                    final_report = node_output.get("final_report")
                    deep_dive_iter = node_output.get("deep_dive_iteration", 0)
                    
                    print(f"   Insight Sufficient: {'✅ YES' if is_sufficient else '❌ NO'}")
                    print(f"   Deep Dive Iterations: {deep_dive_iter}")
                    
                    if final_report:
                        print("   🎯 Final Report Generated!")
                        print(f"      Exploration Summary: {'Yes' if final_report.get('exploration_summary') else 'No'}")
                        print(f"      Insights: {len(final_report.get('insights', []))}")
                    
                elif node_name == "error":
                    print_subheader("❌ ERROR")
                    print(f"   Error: {node_output.get('error_message', 'Unknown')}")
                
                # Store event
                events_by_node[node_name] = node_output
        
        # Final summary
        print_header("FINAL SUMMARY")
        
        final_state = events_by_node.get("approval", events_by_node.get("analyst", {}))
        final_report = final_state.get("final_report", {})
        
        print(f"""
   Analysis Phase     : {current_phase}
   Exploration Iter   : {phase_iteration.get('exploration', 0)}
   Deep Dive Iter     : {phase_iteration.get('deep_dive', 0)}
   ──────────────────   ────────────────────
   Final Report       : {'✅ Generated' if final_report else '❌ Not Generated'}
   Insight Sufficient : {'✅ Yes' if final_state.get('is_insight_sufficient') else '❌ No'}
""")
        
        if final_report:
            print("   📊 Report Contents:")
            print(f"      • Exploration Summary: {'✅' if final_report.get('exploration_summary') else '❌'}")
            print(f"      • Hypotheses: {len(final_report.get('hypotheses', []))}")
            print(f"      • Evaluations: {len(final_report.get('evaluations', []))}")
            print(f"      • Insights: {len(final_report.get('insights', []))}")
            
            # Save detailed report to file
            report_path = save_final_report(
                final_report, 
                events_by_node, 
                question, 
                output_dir,
                phase_iteration
            )
            print(f"\n   💾 Full Report Saved: {report_path}")
        
        print("\n" + "═" * 70)
        
        return events_by_node
        
    except Exception as e:
        print(f"\n❌ Workflow Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_final_report(final_report: dict, events: dict, question: str, output_dir: Path, phase_iteration: dict) -> Path:
    """Save detailed final report as markdown file with embedded charts and Vietnamese content."""
    from datetime import datetime
    import re
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"report_{timestamp}.md"
    
    # Collect all generated chart images
    chart_files = list(output_dir.glob("*.png"))
    exp_charts = sorted([cf for cf in chart_files if "exp" in cf.name.lower()])
    dd_charts = sorted([cf for cf in chart_files if "dd" in cf.name.lower()])
    
    # Get hypotheses and evaluations
    hypotheses = final_report.get("hypotheses", [])
    evaluations = final_report.get("evaluations", [])
    exploration_summary = final_report.get("exploration_summary", {})
    
    # Get analysis summary from analyst
    analyst_event = events.get("analyst", {})
    analysis_summary = analyst_event.get("analysis_summary", "")
    
    # Parse validated/invalidated from analysis_summary text
    validated_findings = []
    invalidated_findings = []
    uncertain_findings = []
    
    # Extract findings from analysis summary
    if analysis_summary:
        # Look for VALIDATED patterns
        validated_pattern = r"(?:VALIDATED|Xác thực|xác thực)[\s\S]*?(?=###|$)"
        matches = re.findall(r"###\s*Hypothesis\s*\d+[:\s]*([^\n]+)\n[^#]*?(?:VALIDATED|xác thực)[^#]*?Evidence[^:]*:\s*([^\n]+)", analysis_summary, re.IGNORECASE)
        for m in matches:
            validated_findings.append({"statement": m[0].strip(), "evidence": m[1].strip()})
        
        # Look for INVALIDATED patterns
        matches = re.findall(r"###\s*Hypothesis\s*\d+[:\s]*([^\n]+)\n[^#]*?(?:INVALIDATED|bác bỏ)[^#]*?Evidence[^:]*:\s*([^\n]+)", analysis_summary, re.IGNORECASE)
        for m in matches:
            invalidated_findings.append({"statement": m[0].strip(), "evidence": m[1].strip()})
    
    # Build lines
    lines = []
    
    # ========== EXECUTIVE SUMMARY ==========
    lines.extend([
        "# 📊 Báo Cáo Phân Tích Dữ Liệu",
        "",
        f"**Ngày tạo:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "",
        "---",
        "",
        "## 📋 Tóm Tắt Điều Hành",
        "",
        f"> **Câu hỏi:** {question}",
        "",
    ])
    
    # Key numbers
    total_hypotheses = len(hypotheses) or (len(validated_findings) + len(invalidated_findings) + len(uncertain_findings))
    lines.extend([
        "### Kết Quả Tổng Quan",
        "",
        f"| Chỉ số | Giá trị |",
        f"|--------|---------|",
        f"| Số vòng phân tích | {phase_iteration.get('deep_dive', 0) + 1} |",
        f"| Tổng giả thuyết | {total_hypotheses} |",
        f"| Đã xác thực | {len(validated_findings)} |",
        f"| Bị bác bỏ | {len(invalidated_findings)} |",
        f"| Biểu đồ tạo ra | {len(exp_charts) + len(dd_charts)} |",
        "",
    ])
    
    # Key findings from exploration
    if isinstance(exploration_summary, dict) and "key_findings" in exploration_summary:
        lines.extend([
            "### Phát Hiện Quan Trọng Nhất",
            "",
        ])
        for finding in exploration_summary["key_findings"][:3]:
            lines.append(f"- ✅ {finding}")
        lines.append("")
    
    # Validated hypotheses summary
    if validated_findings:
        lines.extend([
            "### Các Nguyên Nhân Đã Xác Thực",
            "",
        ])
        for vf in validated_findings[:3]:
            lines.append(f"- 🎯 **{vf['statement'][:80]}**")
        lines.append("")
    
    lines.extend([
        "---",
        "",
    ])
    
    # ========== PHASE 1: EXPLORATION ==========
    lines.extend([
        "## 🔍 Giai Đoạn 1: Khám Phá Dữ Liệu",
        "",
        "*Giai đoạn này xây dựng bức tranh tổng quan về tình hình.*",
        "",
    ])
    
    if isinstance(exploration_summary, dict):
        if "key_findings" in exploration_summary:
            lines.append("### 📌 Phát Hiện Chính")
            lines.append("")
            for finding in exploration_summary["key_findings"]:
                lines.append(f"- {finding}")
            lines.append("")
        
        if "data_overview" in exploration_summary:
            lines.append("### 📊 Số Liệu Tổng Quan")
            lines.append("")
            lines.append("| Chỉ số | Giá trị |")
            lines.append("|--------|---------|")
            for key, value in exploration_summary["data_overview"].items():
                lines.append(f"| {key} | {value} |")
            lines.append("")
        
        if "trends" in exploration_summary:
            lines.append("### 📈 Xu Hướng")
            lines.append("")
            for trend in exploration_summary["trends"]:
                lines.append(f"- 📈 {trend}")
            lines.append("")
    
    # Exploration charts with context
    if exp_charts:
        lines.append("### 📊 Biểu Đồ Khám Phá")
        lines.append("")
        for i, chart in enumerate(exp_charts, 1):
            lines.append(f"#### Biểu đồ {i}")
            lines.append("")
            lines.append(f"![{chart.stem}]({chart.name})")
            lines.append("")
    
    lines.extend(["---", ""])
    
    # ========== PHASE 2: DEEP DIVE ==========
    lines.extend([
        "## 🔬 Giai Đoạn 2: Phân Tích Chuyên Sâu",
        "",
        f"*Đào sâu tìm nguyên nhân gốc rễ qua {phase_iteration.get('deep_dive', 0)} vòng lặp.*",
        "",
    ])
    
    # Validated findings with deep dive charts
    if validated_findings:
        lines.append("### ✅ Các Giả Thuyết Đã Xác Thực")
        lines.append("")
        
        dd_chart_index = 0
        for i, vf in enumerate(validated_findings, 1):
            lines.append(f"#### {i}. {vf['statement']}")
            lines.append("")
            lines.append(f"**Bằng chứng:** {vf['evidence']}")
            lines.append("")
            
            # Add corresponding chart if available
            if dd_chart_index < len(dd_charts):
                lines.append(f"![{dd_charts[dd_chart_index].stem}]({dd_charts[dd_chart_index].name})")
                lines.append("")
                dd_chart_index += 1
            
            lines.append("---")
            lines.append("")
    
    # Invalidated findings
    if invalidated_findings:
        lines.append("### ❌ Các Giả Thuyết Bị Bác Bỏ")
        lines.append("")
        for i, ivf in enumerate(invalidated_findings, 1):
            lines.append(f"- **{ivf['statement']}**")
            lines.append(f"  - *Lý do:* {ivf['evidence'][:150]}")
        lines.append("")
    
    # Remaining deep dive charts
    remaining_dd_charts = dd_charts[len(validated_findings):] if len(validated_findings) < len(dd_charts) else []
    if remaining_dd_charts:
        lines.append("### 📊 Biểu Đồ Phân Tích Bổ Sung")
        lines.append("")
        for chart in remaining_dd_charts:
            lines.append(f"![{chart.stem}]({chart.name})")
            lines.append("")
    
    lines.extend(["---", ""])
    
    # ========== FULL ANALYST OUTPUT (Vietnamese translation note) ==========
    if analysis_summary:
        # Translate key terms to Vietnamese
        vn_summary = analysis_summary
        vn_summary = vn_summary.replace("### Hypothesis", "### Giả thuyết")
        vn_summary = vn_summary.replace("Evaluation", "Đánh giá")
        vn_summary = vn_summary.replace("VALIDATED", "XÁC THỰC")
        vn_summary = vn_summary.replace("INVALIDATED", "BÁC BỎ")
        vn_summary = vn_summary.replace("NEEDS MORE DATA", "CẦN THÊM DỮ LIỆU")
        vn_summary = vn_summary.replace("Evidence Summary", "Tóm tắt bằng chứng")
        vn_summary = vn_summary.replace("Confidence Level", "Độ tin cậy")
        vn_summary = vn_summary.replace("Root Cause Analysis", "Phân tích nguyên nhân gốc rễ")
        vn_summary = vn_summary.replace("Insights and Recommendations", "Insight và Khuyến nghị")
        vn_summary = vn_summary.replace("Action", "Hành động")
        vn_summary = vn_summary.replace("Additional Analysis Needed", "Phân tích bổ sung cần thiết")
        
        lines.extend([
            "## 📝 Chi Tiết Phân Tích",
            "",
            vn_summary,
            "",
            "---",
            "",
        ])
    
    # ========== RECOMMENDATIONS ==========
    lines.extend([
        "## 💡 Kết Luận Và Khuyến Nghị",
        "",
    ])
    
    # Conclusions from validated findings
    if validated_findings:
        lines.append("### 🎯 Kết Luận Chính")
        lines.append("")
        for i, vf in enumerate(validated_findings, 1):
            first_sentence = vf['evidence'].split('.')[0] + '.' if vf['evidence'] else ''
            lines.append(f"{i}. **{vf['statement']}**")
            if first_sentence:
                lines.append(f"   - {first_sentence}")
        lines.append("")
    
    # Action recommendations
    lines.append("### 🚀 Khuyến Nghị Hành Động")
    lines.append("")
    
    if validated_findings:
        for i, vf in enumerate(validated_findings[:3], 1):
            lines.append(f"#### {i}. Xử lý: {vf['statement'][:50]}...")
            lines.append("- Cần đánh giá và đưa ra giải pháp cụ thể")
            lines.append("- Theo dõi các chỉ số liên quan")
            lines.append("")
    else:
        lines.append("- Thu thập thêm dữ liệu để có kết luận chính xác hơn")
        lines.append("- Xem xét mở rộng phạm vi phân tích")
        lines.append("- Kiểm tra chất lượng dữ liệu nguồn")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        f"*Báo cáo được tạo tự động bởi EDA Agent - {len(exp_charts) + len(dd_charts)} biểu đồ đã tạo*",
    ])
    
    # Write to file
    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding="utf-8")
    
    return report_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
        asyncio.run(test_langgraph_workflow(question))
    else:
        print_header("EDA AGENT - INTERACTIVE TEST")
        print("  Enter a question to test the full LangGraph workflow.")
        print("  Type 'quit' or 'q' to exit.\n")
        
        while True:
            try:
                question = input("📝 Enter prompt: ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ['quit', 'q', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                asyncio.run(test_langgraph_workflow(question))
                print("\n" + "=" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
