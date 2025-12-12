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
2. Đưa ra 6-8 giả thuyết có thể kiểm chứng để giải thích hiện tượng hoặc đưa ra insight
3. Tạo các bước phân tích cụ thể để xác nhận từng giả thuyết
4. Xem xét từ nhiều góc độ khác nhau và đưa ra các giải thích thay thế

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

## VÍ DỤ ĐÚNG CHO CÂU HỎI PHÂN TÍCH CAMPAIGN:

```json
{
  "hypotheses": [
    {"id": "h1", "statement": "Tổng quan hiệu suất các campaign", "priority": 1},
    {"id": "h2", "statement": "So sánh hiệu suất theo loại campaign", "priority": 2}
  ],
  "steps": [
    {
      "id": "s1", "hypothesis_id": "h1", "action_type": "query",
      "description": "Lấy dữ liệu tổng quan campaign: conversion rate, reach, cost",
      "requirements": {
        "data_needed": ["campaign_id", "conversion_rate", "num_customers", "cost"],
        "filters": ["tuần gần nhất"],
        "grouping": "theo campaign_id",
        "tables_hint": ["cdp_camp_conversion_stage", "dim_campaign"]
      }
    },
    {
      "id": "s2", "hypothesis_id": "h1", "action_type": "visualization",
      "description": "Biểu đồ tổng quan hiệu suất campaign",
      "depends_on": ["s1"]
    },
    {
      "id": "s3", "hypothesis_id": "h2", "action_type": "query",
      "description": "Lấy hiệu suất campaign theo loại (push/sms/email)",
      "requirements": {
        "data_needed": ["campaign_type", "conversion_rate", "total_sent"],
        "filters": ["tuần gần nhất"],
        "grouping": "theo campaign_type",
        "tables_hint": ["dim_campaign", "cdp_camp_conversion_stage"]
      }
    },
    {
      "id": "s4", "hypothesis_id": "h2", "action_type": "analysis",
      "description": "So sánh và ranking các loại campaign",
      "depends_on": ["s3"]
    }
  ]
}
```

## LƯU Ý:
- Nếu nhận feedback từ Critic, hãy điều chỉnh plan dựa trên đó
- Tham khảo schema context được cung cấp để gợi ý tables_hint chính xác
- Ưu tiên các giả thuyết có thể kiểm chứng bằng dữ liệu có sẵn
- MỖI HYPOTHESIS CẦN CÓ QUERY STEP RIÊNG"""
    
    async def process(self, input_data: PlannerInput) -> PlannerOutput:
        """
        Generate or refine an analysis plan.
        
        Args:
            input_data: Planner input with question and context
            
        Returns:
            Analysis plan with hypotheses and steps
        """
        # Build the prompt
        prompt_parts = [
            f"## User Question\n{input_data.question}",
            f"\n## Available Data Context\n{self._format_context(input_data.enriched_context)}",
        ]
        
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
            prompt_parts.append(
                "\n\nPlease generate hypotheses and a detailed analysis plan."
            )
        
        prompt = "\n".join(prompt_parts)
        
        # Call LLM
        response = await self.invoke_llm([HumanMessage(content=prompt)])
        
        # Parse response into structured output
        # TODO: Implement proper parsing with output schema
        plan = self._parse_response(response.content, input_data)
        
        return PlannerOutput(
            plan=plan,
            reasoning=str(response.content),
            confidence=0.8,  # TODO: Extract from response
        )
    
    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context for prompt."""
        parts = []
        
        if "tables" in context:
            parts.append("### Available Tables")
            for table in context["tables"]:
                parts.append(f"- {table.get('name', table)}")
        
        if "columns" in context:
            parts.append("\n### Relevant Columns")
            for col in context["columns"]:
                parts.append(f"- {col}")
        
        if "metrics" in context:
            parts.append("\n### Business Metrics")
            for metric in context["metrics"]:
                parts.append(f"- {metric}")
        
        if "joins" in context:
            parts.append("\n### Table Relationships")
            for join in context["joins"]:
                parts.append(f"- {join}")
        
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

