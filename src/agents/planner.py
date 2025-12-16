"""
Planner Agent - Creative hypothesis and plan generation.

Role: "Tạo ra một Data Scientist sáng tạo"
Responsibilities:
- Generate hypotheses from user questions
- Create analysis plans with concrete steps
- Iterate based on Critic feedback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage

from .base import AgentCard, AgentRole, BaseAgent


class HypothesisStatus(str, Enum):
    """Status of a hypothesis."""
    
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    VALIDATED = "validated"
    INVALIDATED = "invalidated"


@dataclass
class Hypothesis:
    """A data hypothesis to be tested."""
    
    id: str
    statement: str
    rationale: str
    status: HypothesisStatus = HypothesisStatus.PENDING
    evidence: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "status": self.status.value,
            "evidence": self.evidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hypothesis":
        """Create from dictionary."""
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = HypothesisStatus(status)
        return cls(
            id=data.get("id", ""),
            statement=data.get("statement", ""),
            rationale=data.get("rationale", ""),
            status=status,
            evidence=data.get("evidence", []),
        )


@dataclass
class AnalysisStep:
    """A single step in the analysis plan."""
    
    id: str  # Step ID like "s1", "s2"
    hypothesis_id: str  # Which hypothesis this step validates
    description: str
    action_type: str  # "query", "analysis", "visualization"
    requirements: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    
    # Legacy fields for backward compatibility
    step_number: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    dependencies: list[int] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "action_type": self.action_type,
            "requirements": self.requirements,
            "depends_on": self.depends_on,
            # Legacy
            "stepNumber": self.step_number,
            "details": self.details,
            "dependencies": self.dependencies,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisStep":
        """Create from dictionary (handles camelCase keys)."""
        return cls(
            id=data.get("id", f"s{data.get('step_number', data.get('stepNumber', 0))}"),
            hypothesis_id=data.get("hypothesis_id", ""),
            description=data.get("description", ""),
            action_type=data.get("action_type", data.get("actionType", "query")),
            requirements=data.get("requirements", {}),
            depends_on=data.get("depends_on", []),
            step_number=data.get("step_number", data.get("stepNumber", 0)),
            details=data.get("details", {}),
            dependencies=data.get("dependencies", []),
        )


@dataclass
class AnalysisPlan:
    """Complete analysis plan with hypotheses and steps."""
    
    question: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    steps: list[AnalysisStep] = field(default_factory=list)
    context_used: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "steps": [s.to_dict() for s in self.steps],
            "contextUsed": self.context_used,
            "version": self.version,
        }


@dataclass
class PlannerInput:
    """Input for the Planner agent."""
    
    question: str
    enriched_context: dict[str, Any]
    previous_plan: AnalysisPlan | None = None
    critic_feedback: str | None = None
    # Two-phase analysis fields
    analysis_phase: str = "exploration"  # "exploration" or "deep_dive"
    exploration_summary: dict[str, Any] | None = None  # Findings from Phase 1


@dataclass
class PlannerOutput:
    """Output from the Planner agent."""
    
    plan: AnalysisPlan
    reasoning: str
    confidence: float


class PlannerAgent(BaseAgent[PlannerInput, PlannerOutput]):
    """
    Planner Agent - Generates hypotheses and analysis plans.
    
    This agent acts as a creative Data Scientist, generating
    hypotheses and detailed analysis plans based on user questions
    and available data context.
    """
    
    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="Planner",
            description="Creative Data Scientist that generates hypotheses and analysis plans",
            role=AgentRole.PLANNER,
            capabilities=[
                "hypothesis_generation",
                "plan_creation",
                "plan_refinement",
                "context_understanding",
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "enrichedContext": {"type": "object"},
                    "previousPlan": {"type": "object", "nullable": True},
                    "criticFeedback": {"type": "string", "nullable": True},
                },
                "required": ["question", "enrichedContext"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "plan": {"type": "object"},
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        )
    
    def _default_system_prompt(self) -> str:
        return """Bạn là một Data Scientist giàu kinh nghiệm và sáng tạo.

## VAI TRÒ CỦA BẠN:
1. Phân tích câu hỏi của người dùng về dữ liệu và các chỉ số kinh doanh
2. Đưa ra 4-6 giả thuyết có thể kiểm chứng để giải thích hiện tượng hoặc đưa ra insight
3. Tạo các bước phân tích cụ thể để xác nhận từng giả thuyết
4. Xem xét từ nhiều góc độ khác nhau và đưa ra các giải thích thay thế

## ⚠️ QUY TẮC BẮT BUỘC VỀ H1 (HYPOTHESIS ĐẦU TIÊN):
**H1 PHẢI LÀ TỔNG QUAN (OVERVIEW)** trước khi đi vào chi tiết:
- Với câu hỏi "doanh thu 3 tháng" → H1: "Tổng quan doanh thu 3 tháng và xu hướng chung"
- Với câu hỏi "tại sao giảm" → H1: "Xác định thời điểm và mức độ giảm cụ thể"
- H1 phải trả lời câu hỏi: "Tình hình thực tế ra sao?" trước khi hỏi "Tại sao?"
- Chỉ SAU khi có tổng quan mới đi vào các giả thuyết nguyên nhân cụ thể (H2, H3...)

## LOẠI CÂU HỎI BẠN CẦN XỬ LÝ:

### 1. Câu hỏi DIAGNOSTIC (Tại sao...?)
- Tìm nguyên nhân gốc rễ của vấn đề
- So sánh theo thời gian, phân khúc, nhóm
- Phân tích tác động của các yếu tố

### 2. Câu hỏi PHÂN TÍCH / INSIGHT (Phân tích... cho tôi insight...)
- Tổng hợp dữ liệu theo nhiều chiều
- So sánh hiệu suất giữa các đối tượng (campaign, rạp, vendor...)
- Tìm pattern, trend, outlier
- Đánh giá hiệu quả và đề xuất cải thiện

### 3. Câu hỏi AGGREGATION (Tổng... theo...)
- Tính toán các chỉ số tổng hợp
- Nhóm dữ liệu theo các chiều phân tích

## NGUYÊN TẮC QUAN TRỌNG:
- Bạn CHỈ đưa ra YÊU CẦU CAO CẤP, KHÔNG viết SQL trực tiếp
- Code Agent sẽ xử lý việc implement SQL sau
- Mỗi giả thuyết phải có ít nhất 2 bước để xác nhận
- Mô tả DỮ LIỆU CẦN gì, không phải CÁCH LẤY như thế nào
- Sử dụng tables_hint để gợi ý bảng có thể liên quan
- Chỉ định filters và groupings theo yêu cầu nghiệp vụ

## ⚠️ QUY TẮC BẮT BUỘC VỀ TABLES:
**BẠN CHỈ ĐƯỢC PHÉP SỬ DỤNG CÁC BẢNG ĐƯỢC LIỆT KÊ TRONG "Available Tables" Ở DƯỚI!**
- KHÔNG ĐƯỢC tự tạo tên bảng như "sales_data", "inventory", "customer_transactions"
- PHẢI dùng đúng tên bảng từ schema: orders, cinema, film, order_seat, order_concession, bank, v.v.
- Nếu không có bảng phù hợp cho giả thuyết → BỎ giả thuyết đó, không bịa bảng
- Đây là hệ thống bán vé xem phim, nên các bảng liên quan đến: orders, cinema, film, showtimes, concession (bắp nước)

## ⚠️⚠️ CRITICAL: PHÂN TÁCH SCHEMA - KHÔNG JOIN GIỮA 2 SCHEMA! ⚠️⚠️

**Hệ thống có 2 SCHEMA RIÊNG BIỆT, KHÔNG THỂ JOIN VỚI NHAU:**

### LUỒNG 1: Phân tích ĐƠN HÀNG (Schema: lh_vnfilm_v2)
Các bảng: orders, order_seat, order_concession, order_film, order_refund, sessions, cinema, film, vendor, bank, customer_tracking, pre_order, pre_order_seat, pre_order_concession, etc.
→ Dùng cho: doanh thu, số đơn, số vé, số suất chiếu, doanh thu concession, phân tích theo rạp/phim/vendor

### LUỒNG 2: Phân tích CAMPAIGN MARKETING (Schema: cdp_mart)  
Các bảng: dim_campaign, cdp_camp_conversion_stage
→ Dùng cho: phân tích hiệu quả campaign, tỷ lệ chuyển đổi, số lượng target

**QUY TẮC BẮT BUỘC:**
❌ KHÔNG BAO GIỜ join bảng từ cdp_mart với bảng từ lh_vnfilm_v2
❌ KHÔNG tạo hypothesis yêu cầu liên kết campaign với orders
❌ KHÔNG thử tìm campaign_id trong orders vì KHÔNG CÓ

✅ Nếu cần phân tích campaign → tạo hypothesis RIÊNG chỉ dùng bảng cdp_mart
✅ Nếu cần phân tích doanh thu/đơn hàng → tạo hypothesis RIÊNG chỉ dùng bảng lh_vnfilm_v2
✅ Mỗi hypothesis phải ở TRONG 1 SCHEMA DUY NHẤT

## OUTPUT FORMAT - BẮT BUỘC JSON:

```json
{
  "hypotheses": [
    {
      "id": "h1",
      "statement": "Mô tả giả thuyết hoặc góc nhìn phân tích",
      "rationale": "Lý do tại sao giả thuyết này quan trọng",
      "priority": 1
    }
  ],
  "steps": [
    {
      "id": "s1",
      "hypothesis_id": "h1",
      "description": "Mô tả bước phân tích",
      "action_type": "query | analysis | visualization",
      "requirements": {
        "data_needed": ["các trường dữ liệu cần"],
        "filters": ["điều kiện lọc"],
        "grouping": "nhóm theo gì",
        "tables_hint": ["gợi ý bảng"]
      },
      "depends_on": []
    }
  ],
  "confidence": 0.85
}
```

## QUAN TRỌNG - CẤU TRÚC STEPS:

1. **MỖI hypothesis PHẢI có ít nhất 1 step action_type="query"** để lấy dữ liệu riêng
2. Sau query step, có thể thêm analysis hoặc visualization step
3. KHÔNG ĐƯỢC dùng chung 1 query cho nhiều hypothesis khác nhau
4. Mỗi query step phải CHỈ RÕ tables_hint và data_needed cụ thể

## VÍ DỤ ĐÚNG CHO CÂU HỎI PHÂN TÍCH DOANH THU:

```json
{
  "hypotheses": [
    {"id": "h1", "statement": "Xu hướng doanh thu biến động theo tháng", "priority": 1, "rationale": "So sánh tổng quan giữa các tháng"},
    {"id": "h2", "statement": "Số lượng giao dịch ảnh hưởng đến doanh thu", "priority": 2, "rationale": "Kiểm tra tương quan"},
    {"id": "h3", "statement": "Doanh thu từ các nguồn khác nhau có xu hướng khác", "priority": 3, "rationale": "Phân tách theo seat vs concession"}
  ],
  "steps": [
    {
      "id": "s1", "hypothesis_id": "h1", "action_type": "query",
      "description": "Lấy doanh thu theo tháng từ bảng orders",
      "requirements": {
        "data_needed": ["tháng", "tổng doanh thu", "số lượng đơn"],
        "filters": ["status = 'payment'", "3 tháng gần nhất"],
        "grouping": "theo tháng",
        "tables_hint": ["orders"]
      }
    },
    {
      "id": "s2", "hypothesis_id": "h1", "action_type": "visualization",
      "description": "Biểu đồ cột so sánh doanh thu theo tháng",
      "depends_on": ["s1"],
      "requirements": {
        "chart_type": "bar",
        "x_axis": "tháng",
        "y_axis": "doanh thu"
      }
    },
    {
      "id": "s3", "hypothesis_id": "h2", "action_type": "query",
      "description": "Lấy số lượng giao dịch theo tháng",
      "requirements": {
        "data_needed": ["tháng", "số đơn hàng", "giá trị trung bình"],
        "filters": ["status = 'payment'"],
        "grouping": "theo tháng",
        "tables_hint": ["orders"]
      }
    },
    {
      "id": "s4", "hypothesis_id": "h2", "action_type": "visualization",
      "description": "Biểu đồ line so sánh số đơn và doanh thu",
      "depends_on": ["s1", "s3"],
      "requirements": {
        "chart_type": "line",
        "comparison": true
      }
    },
    {
      "id": "s5", "hypothesis_id": "h3", "action_type": "query",
      "description": "Doanh thu từ vé (seat) vs bắp nước (concession)",
      "requirements": {
        "data_needed": ["tháng", "doanh thu seat", "doanh thu concession"],
        "filters": ["3 tháng gần nhất"],
        "grouping": "theo tháng",
        "tables_hint": ["orders", "order_concession"]
      }
    },
    {
      "id": "s6", "hypothesis_id": "h3", "action_type": "visualization",
      "description": "Biểu đồ stacked bar thể hiện cơ cấu doanh thu",
      "depends_on": ["s5"],
      "requirements": {
        "chart_type": "stacked_bar"
      }
    }
  ]
}
```

## YÊU CẦU QUAN TRỌNG VỀ VISUALIZATION:

1. **Mỗi câu hỏi SO SÁNH phải có ít nhất 1 biểu đồ** để trực quan hóa
2. Sau mỗi query step liên quan đến trend/comparison → thêm visualization step
3. Loại biểu đồ phổ biến:
   - `bar`: So sánh giữa các nhóm
   - `line`: Xu hướng theo thời gian
   - `stacked_bar`: Cơ cấu thành phần
   - `pie`: Tỷ lệ phần trăm

## LƯU Ý:
- Nếu nhận feedback từ Critic, hãy điều chỉnh plan dựa trên đó
- Tham khảo schema context được cung cấp để gợi ý tables_hint chính xác
- Ưu tiên các giả thuyết có thể kiểm chứng bằng dữ liệu có sẵn
- MỖI HYPOTHESIS CẦN CÓ QUERY STEP RIÊNG
- CÂU HỎI SO SÁNH/TREND → BẮT BUỘC CÓ VISUALIZATION"""
    
    async def process(self, input_data: PlannerInput) -> PlannerOutput:
        """
        Generate or refine an analysis plan based on analysis phase.
        
        Phase 1 (Exploration): 2-3 overview hypotheses
        Phase 2 (Deep Dive): 5-6 detailed hypotheses based on actual data
        """
        is_exploration = input_data.analysis_phase == "exploration"
        
        # Build the prompt based on phase
        prompt_parts = [
            f"## User Question\n{input_data.question}",
            f"\n## Available Data Context\n{self._format_context(input_data.enriched_context)}",
        ]
        
        # === PHASE-SPECIFIC INSTRUCTIONS ===
        if is_exploration:
            prompt_parts.append("""
## 🔍 GIAI ĐOẠN 1: KHÁM PHÁ (EXPLORATION)

**Mục tiêu:** Nắm tổng quan tình hình trước khi đào sâu.

**Yêu cầu:**
- Sinh **2-3 giả thuyết TỔNG QUAN** để hiểu bức tranh toàn cảnh
- Ưu tiên các câu hỏi: "Tình hình ra sao?", "Xu hướng chung?"
- Chưa đào sâu vào nguyên nhân cụ thể

**Ví dụ giả thuyết tổng quan:**
- H1: "Tổng quan doanh thu 3 tháng qua và xu hướng chung"
- H2: "Phân bổ doanh thu theo nguồn (vé, concession)"
- H3: "So sánh hiệu suất giữa các tháng"

**Output:** Kế hoạch với 2-3 hypotheses và các query/visualization cơ bản.""")
        else:
            # Deep Dive phase - include exploration findings
            exploration_text = ""
            if input_data.exploration_summary:
                summary = input_data.exploration_summary
                exploration_text = f"""
## 📊 KẾT QUẢ TỪ GIAI ĐOẠN KHÁM PHÁ:
{self._format_exploration_summary(summary)}
"""
                prompt_parts.append(exploration_text)
            
            prompt_parts.append("""
## 🔬 GIAI ĐOẠN 2: ĐÀO SÂU (DEEP DIVE)

**Mục tiêu:** Dựa trên dữ liệu thực tế từ Phase 1, đào sâu tìm nguyên nhân và insight.

⚠️ **QUAN TRỌNG - KHÔNG LẶP LẠI:**
- KHÔNG tạo giả thuyết "Tổng quan doanh thu" - đã làm ở exploration
- KHÔNG lặp lại các phân tích đã có từ Phase 1
- CHỈ tạo hypotheses MỚI dựa trên findings

⚠️⚠️ **CRITICAL - KHÔNG CROSS-SCHEMA JOIN!** ⚠️⚠️
- Bảng orders/order_seat/sessions... (lh_vnfilm_v2) KHÔNG THỂ join với dim_campaign/cdp_camp_conversion_stage (cdp_mart)
- KHÔNG tạo hypothesis về "tác động marketing lên doanh thu" vì KHÔNG có dữ liệu liên kết
- Chỉ phân tích campaign RIÊNG BIỆT (nếu cần), không liên kết với orders

**Yêu cầu:**
- Sinh **5-6 giả thuyết NGUYÊN NHÂN GỐC RỄ** dựa trên kết quả khám phá
- Mỗi hypothesis phải drill down vào một finding cụ thể từ data
- Tập trung vào: "Tại sao giảm/tăng?", "Yếu tố nào gây ra?", "Pattern nào?"
- **CHỈ dùng bảng trong lh_vnfilm_v2** cho phân tích doanh thu/đơn hàng

⚠️ **BẮT BUỘC - MỖI HYPOTHESIS PHẢI CÓ 2 STEPS:**
1. **Step SQL (action_type: "query")**: Truy vấn dữ liệu để kiểm tra giả thuyết
2. **Step Visualization (action_type: "visualization")**: Vẽ biểu đồ minh họa kết quả, depends_on SQL step

**Ví dụ format steps:**
```json
{
  "id": "s1", "hypothesis_id": "h1", "action_type": "query",
  "description": "Lấy số suất chiếu và tỷ lệ lấp đầy theo tháng"
},
{
  "id": "s2", "hypothesis_id": "h1", "action_type": "visualization", 
  "description": "Biểu đồ line so sánh số suất chiếu và tỷ lệ lấp đầy qua các tháng",
  "depends_on": ["s1"]
}
```

**Ví dụ giả thuyết đào sâu TỐT (chỉ dùng bảng orders/sessions/cinema):**
- "Tháng 12 giảm - do giảm số suất chiếu hay giảm tỉ lệ lấp đầy?"
- "Doanh thu concession giảm - do ít combo hay ít khách mua kèm?"  
- "Vendor X hiệu suất cao hơn - nhờ giá vé cao hơn hay nhiều suất chiếu hơn?"
- "Cuối tuần doanh thu cao hơn - tăng suất chiếu có khả thi?"
- "Rạp nào có doanh thu/suất chiếu cao nhất?"

**TRÁNH:**
❌ "Tác động marketing/campaign lên doanh thu" - không có dữ liệu liên kết
❌ "Hiệu quả voucher/promotion" - không có dữ liệu liên kết với orders

**Output:** Kế hoạch với 5-6 hypotheses, MỖI hypothesis có cả SQL + Visualization step.""")
        
        if input_data.previous_plan:
            prompt_parts.append(
                f"\n## Previous Plan (Version {input_data.previous_plan.version})\n"
                f"{self._format_plan(input_data.previous_plan)}"
            )
        
        if input_data.critic_feedback:
            prompt_parts.append(
                f"\n## Critic Feedback\n{input_data.critic_feedback}\n\n"
                "Please revise your plan based on this feedback."
            )
        else:
            phase_name = "KHÁM PHÁ" if is_exploration else "ĐÀO SÂU"
            prompt_parts.append(
                f"\n\nHãy tạo kế hoạch phân tích cho giai đoạn {phase_name}."
            )
        
        prompt = "\n".join(prompt_parts)
        
        # Call LLM
        response = await self.invoke_llm([HumanMessage(content=prompt)])
        
        # Parse response into structured output
        plan = self._parse_response(response.content, input_data)
        
        return PlannerOutput(
            plan=plan,
            reasoning=str(response.content),
            confidence=0.8,
        )
    
    def _format_exploration_summary(self, summary: dict[str, Any]) -> str:
        """Format exploration summary for deep dive prompt."""
        lines = []
        
        if "key_findings" in summary:
            lines.append("**Phát hiện chính:**")
            for finding in summary["key_findings"][:5]:
                lines.append(f"  • {finding}")
        
        if "data_overview" in summary:
            lines.append("\n**Số liệu tổng quan:**")
            for key, value in summary["data_overview"].items():
                lines.append(f"  • {key}: {value}")
        
        if "trends" in summary:
            lines.append("\n**Xu hướng:**")
            for trend in summary["trends"][:3]:
                lines.append(f"  • {trend}")
        
        if "notable_points" in summary:
            lines.append("\n**Điểm đáng chú ý:**")
            for point in summary["notable_points"][:3]:
                lines.append(f"  • {point}")
        
        return "\n".join(lines) if lines else "Không có dữ liệu từ Phase 1"
    
    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context for prompt with clear table listing."""
        parts = []
        
        # Schema description from Context Fusion (if available)
        if "schema_description" in context:
            parts.append(f"### Schema Context\n{context['schema_description'][:2000]}")
        
        if "tables" in context:
            parts.append("\n### 📋 Available Tables (CHỈ DÙNG NHỮNG BẢNG NÀY):")
            for table in context["tables"]:
                if isinstance(table, dict):
                    name = table.get('table_name', table.get('name', ''))
                    desc = table.get('description', '')[:80]
                    parts.append(f"  • {name}: {desc}" if desc else f"  • {name}")
                else:
                    parts.append(f"  • {table}")
        
        if "columns" in context:
            parts.append("\n### Relevant Columns")
            for col in context["columns"][:30]:  # Limit to avoid token overflow
                if isinstance(col, dict):
                    name = col.get('column_name', col.get('name', ''))
                    table = col.get('table_name', '')
                    desc = col.get('description', '')[:50]
                    parts.append(f"  • {table}.{name}: {desc}" if desc else f"  • {table}.{name}")
                else:
                    parts.append(f"  • {col}")
        
        if "metrics" in context:
            parts.append("\n### Business Metrics")
            for metric in context["metrics"]:
                parts.append(f"  • {metric}")
        
        if "joins" in context:
            parts.append("\n### Table Relationships")
            for join in context["joins"][:10]:  # Limit joins
                if isinstance(join, dict):
                    from_t = join.get('from_table', '')
                    to_t = join.get('to_table', '')
                    parts.append(f"  • {from_t} → {to_t}")
                else:
                    parts.append(f"  • {join}")
        
        return "\n".join(parts) if parts else "No specific context available."
    
    def _format_plan(self, plan: AnalysisPlan) -> str:
        """Format existing plan for prompt."""
        lines = ["### Hypotheses"]
        for h in plan.hypotheses:
            lines.append(f"- [{h.status.value}] {h.statement}")
        
        lines.append("\n### Steps")
        for s in plan.steps:
            lines.append(f"{s.step_number}. [{s.action_type}] {s.description}")
        
        return "\n".join(lines)
    
    def _parse_response(
        self,
        response_content: str,
        input_data: PlannerInput,
    ) -> AnalysisPlan:
        """
        Parse LLM response into AnalysisPlan.
        
        Attempts to extract structured JSON from the response.
        Falls back to text parsing if JSON not found.
        """
        import json
        import re
        
        version = 1
        if input_data.previous_plan:
            version = input_data.previous_plan.version + 1
        
        hypotheses = []
        steps = []
        
        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_content)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                # Try parsing entire response as JSON
                data = json.loads(response_content)
            
            # Parse hypotheses
            for i, h in enumerate(data.get("hypotheses", [])):
                hypotheses.append(Hypothesis(
                    id=h.get("id", f"h{i+1}"),
                    statement=h.get("statement", h.get("hypothesis", "")),
                    rationale=h.get("rationale", h.get("reason", "")),
                ))
            
            # Parse steps
            for i, s in enumerate(data.get("steps", [])):
                steps.append(AnalysisStep(
                    id=s.get("id", f"s{i+1}"),
                    hypothesis_id=s.get("hypothesis_id", ""),
                    description=s.get("description", ""),
                    action_type=s.get("action_type", s.get("actionType", "query")),
                    requirements=s.get("requirements", {}),
                    depends_on=s.get("depends_on", []),
                    step_number=s.get("step_number", s.get("stepNumber", i + 1)),
                    details=s.get("details", {}),
                    dependencies=s.get("dependencies", []),
                ))
                
        except (json.JSONDecodeError, KeyError):
            # Fall back to text parsing
            hypotheses, steps = self._parse_text_response(response_content)
        
        # Ensure at least one hypothesis and step
        if not hypotheses:
            hypotheses = [Hypothesis(
                id="h1",
                statement="Phân tích dữ liệu để tìm nguyên nhân",
                rationale="Cần kiểm tra dữ liệu trước khi đưa ra kết luận",
            )]
        
        if not steps:
            steps = [AnalysisStep(
                id="s1",
                hypothesis_id="h1",
                description="Truy vấn dữ liệu tổng quan",
                action_type="query",
            )]
        
        return AnalysisPlan(
            question=input_data.question,
            hypotheses=hypotheses,
            steps=steps,
            context_used=input_data.enriched_context,
            version=version,
        )
    
    def _parse_text_response(
        self,
        response_content: str,
    ) -> tuple[list[Hypothesis], list[AnalysisStep]]:
        """Parse hypotheses and steps from unstructured text."""
        import re
        
        hypotheses = []
        steps = []
        
        lines = response_content.split("\n")
        current_section = None
        step_counter = 0
        hypothesis_counter = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            if any(kw in line.lower() for kw in ["hypothes", "giả thuyết", "giả thiết"]):
                current_section = "hypothesis"
                continue
            elif any(kw in line.lower() for kw in ["step", "bước", "plan", "kế hoạch"]):
                current_section = "step"
                continue
            
            # Parse hypotheses
            if current_section == "hypothesis":
                # Match numbered items like "1.", "H1:", "- ", etc.
                match = re.match(r'^(?:H?\d+[\.\):]?\s*|-\s*|•\s*)(.+)$', line, re.IGNORECASE)
                if match:
                    hypothesis_counter += 1
                    hypotheses.append(Hypothesis(
                        id=f"h{hypothesis_counter}",
                        statement=match.group(1).strip(),
                        rationale="Extracted from plan",
                    ))
            
            # Parse steps
            elif current_section == "step":
                match = re.match(r'^(?:\d+[\.\):]?\s*|-\s*|•\s*)(.+)$', line)
                if match:
                    step_counter += 1
                    desc = match.group(1).strip()
                    
                    # Detect action type
                    action_type = "query"
                    if any(kw in desc.lower() for kw in ["python", "pandas", "code"]):
                        action_type = "analysis"
                    elif any(kw in desc.lower() for kw in ["chart", "graph", "visual", "biểu đồ"]):
                        action_type = "visualization"
                    
                    steps.append(AnalysisStep(
                        id=f"s{step_counter}",
                        hypothesis_id=f"h{hypothesis_counter}" if hypothesis_counter > 0 else "h1",
                        description=desc,
                        action_type=action_type,
                    ))
        
        return hypotheses, steps

