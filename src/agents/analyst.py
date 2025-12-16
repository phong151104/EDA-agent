"""
Analyst Agent - Evidence evaluation and hypothesis validation.

Role: "Tạo ra một Data Analyst chuyên nghiệp"
Responsibilities:
- Evaluate execution results against hypotheses
- Determine if hypotheses are validated or invalidated
- Generate insights and conclusions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage

from .base import AgentCard, AgentRole, BaseAgent
from .code_agent import ExecutionResult
from .planner import AnalysisPlan, Hypothesis, HypothesisStatus


class InsightType(str, Enum):
    """Type of insight generated."""
    
    FINDING = "finding"
    ANOMALY = "anomaly"
    TREND = "trend"
    CORRELATION = "correlation"
    RECOMMENDATION = "recommendation"


@dataclass
class Insight:
    """A data insight derived from analysis."""
    
    id: str
    type: InsightType
    title: str
    description: str
    supporting_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.8
    related_hypothesis: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "supportingEvidence": self.supporting_evidence,
            "confidence": self.confidence,
            "relatedHypothesis": self.related_hypothesis,
        }


@dataclass
class HypothesisEvaluation:
    """Evaluation of a single hypothesis."""
    
    hypothesis_id: str
    new_status: HypothesisStatus
    evidence_summary: str
    confidence: float
    supporting_data: Any = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesisId": self.hypothesis_id,
            "newStatus": self.new_status.value,
            "evidenceSummary": self.evidence_summary,
            "confidence": self.confidence,
        }


@dataclass
class AnalystInput:
    """Input for the Analyst agent."""
    
    plan: AnalysisPlan
    execution_results: dict[str, ExecutionResult]  # Step IDs are strings like 's1'
    original_question: str
    # Two-phase analysis
    analysis_phase: str = "exploration"  # "exploration" or "deep_dive"


@dataclass
class AnalystOutput:
    """Output from the Analyst agent."""
    
    hypothesis_evaluations: list[HypothesisEvaluation]
    insights: list[Insight]
    summary: str
    answers_question: bool
    confidence: float
    needs_more_analysis: bool = False
    suggested_follow_up: str | None = None
    # Two-phase analysis: summary for Phase 1 to pass to Phase 2
    exploration_summary: dict[str, Any] | None = None


class AnalystAgent(BaseAgent[AnalystInput, AnalystOutput]):
    """
    Analyst Agent - Evaluates evidence and validates hypotheses.
    
    This agent acts as a professional Data Analyst, examining
    execution results to validate or invalidate hypotheses and
    generate actionable insights.
    """
    
    def __init__(
        self,
        insight_threshold: float = 0.7,
        **kwargs: Any,
    ):
        """
        Initialize Analyst agent.
        
        Args:
            insight_threshold: Minimum confidence for insights
            **kwargs: Base agent arguments
        """
        super().__init__(**kwargs)
        self.insight_threshold = insight_threshold
    
    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="Analyst",
            description="Professional Data Analyst that evaluates evidence and generates insights",
            role=AgentRole.ANALYST,
            capabilities=[
                "hypothesis_evaluation",
                "insight_generation",
                "evidence_synthesis",
                "question_answering",
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "plan": {"type": "object"},
                    "executionResults": {"type": "object"},
                    "originalQuestion": {"type": "string"},
                },
                "required": ["plan", "executionResults", "originalQuestion"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "hypothesisEvaluations": {"type": "array"},
                    "insights": {"type": "array"},
                    "summary": {"type": "string"},
                    "answersQuestion": {"type": "boolean"},
                    "confidence": {"type": "number"},
                },
            },
        )
    
    def _default_system_prompt(self) -> str:
        return """You are a professional Data Analyst with expertise in evidence evaluation.

Your role is to:
1. Examine execution results from data analysis
2. Evaluate whether hypotheses are supported or refuted by the data
3. Generate clear insights from the findings
4. Synthesize results to answer the original question

When evaluating evidence:
- Look for statistical significance
- Consider data quality and sample sizes
- Identify patterns, trends, and anomalies
- Be objective and evidence-based

For each hypothesis:
- Determine if it's VALIDATED, INVALIDATED, or needs more data
- Provide evidence summary
- Assign confidence level (0.0-1.0)

Generate insights that are:
- Actionable and business-relevant
- Clearly supported by data
- Ranked by importance/confidence

If the analysis doesn't fully answer the question:
- Suggest what additional analysis is needed
- Identify data gaps"""
    
    async def process(self, input_data: AnalystInput) -> AnalystOutput:
        """
        Evaluate execution results and generate insights.
        
        Phase 1 (Exploration): Generate exploration_summary for Phase 2
        Phase 2 (Deep Dive): Generate detailed insights and recommendations
        """
        is_exploration = input_data.analysis_phase == "exploration"
        
        # Build analysis prompt
        prompt_parts = [
            f"## Original Question\n{input_data.original_question}",
            "\n## Hypotheses to Evaluate",
        ]
        
        for h in input_data.plan.hypotheses:
            prompt_parts.append(f"- **{h.id}**: {h.statement}")
            prompt_parts.append(f"  Rationale: {h.rationale}")
        
        prompt_parts.append("\n## Execution Results")
        
        for step_num, result in input_data.execution_results.items():
            step = next(
                (s for s in input_data.plan.steps if s.step_number == step_num),
                None
            )
            step_desc = step.description if step else "Unknown step"
            
            prompt_parts.append(
                f"\n### Step {step_num}: {step_desc}\n"
                f"Status: {result.status.value}\n"
                f"Output Type: {result.output_type.value}"
            )
            
            if result.output:
                output_str = self._format_output(result)
                prompt_parts.append(f"\n```\n{output_str}\n```")
            
            if result.error_message:
                prompt_parts.append(f"Error: {result.error_message}")
        
        # Phase-specific instructions
        if is_exploration:
            prompt_parts.append("""
## 🔍 GIAI ĐOẠN EXPLORATION - YÊU CẦU OUTPUT:

Trả về JSON với format sau:
```json
{
  "phase": "exploration",
  "key_findings": ["Phát hiện chính 1", "Phát hiện chính 2", ...],
  "data_overview": {
    "tong_doanh_thu": "X triệu VND",
    "thang_cao_nhat": "Tháng Y",
    "thang_thap_nhat": "Tháng Z",
    ...các số liệu quan trọng...
  },
  "trends": ["Xu hướng 1: tăng/giảm X%", "Xu hướng 2", ...],
  "notable_points": ["Điểm đáng chú ý 1", "Điểm đáng chú ý 2", ...],
  "summary": "Tóm tắt ngắn gọn tình hình"
}
```

QUAN TRỌNG: Output này sẽ được dùng để tạo giả thuyết đào sâu ở Phase 2.""")
        else:
            prompt_parts.append("""
## 🔬 GIAI ĐOẠN DEEP DIVE - YÊU CẦU OUTPUT:

Hãy:
1. Đánh giá từng hypothesis (validated/invalidated/needs more data)
2. Tìm NGUYÊN NHÂN GỐC RỄ cho các hiện tượng
3. Đưa ra INSIGHTS chi tiết và KHUYẾN NGHỊ hành động
4. Đánh giá mức độ confident của kết luận

Trả về phân tích chi tiết, không cần JSON format.""")
        
        prompt = "\n".join(prompt_parts)
        
        # Call LLM
        response = await self.invoke_llm([HumanMessage(content=prompt)])
        
        # Parse response based on phase
        return self._parse_response(response.content, input_data, is_exploration)
    
    def _format_output(self, result: ExecutionResult) -> str:
        """Format execution output for prompt."""
        output = result.output
        
        if isinstance(output, dict):
            # JSON-like output
            import json
            return json.dumps(output, indent=2, default=str)[:2000]
        elif isinstance(output, list):
            # Table/list output - show first few rows
            if len(output) > 10:
                return str(output[:10]) + f"\n... and {len(output) - 10} more rows"
            return str(output)
        else:
            return str(output)[:2000]
    
    def _parse_response(
        self,
        response_content: str,
        input_data: AnalystInput,
        is_exploration: bool = False,
    ) -> AnalystOutput:
        """
        Parse LLM response into AnalystOutput.
        
        For exploration phase, tries to extract JSON with exploration_summary.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Placeholder evaluations
        evaluations = []
        for h in input_data.plan.hypotheses:
            evaluations.append(
                HypothesisEvaluation(
                    hypothesis_id=h.id,
                    new_status=HypothesisStatus.PENDING,
                    evidence_summary="To be parsed from LLM response",
                    confidence=0.5,
                )
            )
        
        exploration_summary = None
        if is_exploration:
            # Try to parse JSON from response
            import json
            import re
            
            # Try multiple patterns to find JSON
            patterns = [
                r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
                r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
                r'\{[\s\S]*?"phase"[\s\S]*?\}',  # Inline JSON with "phase"
            ]
            
            for pattern in patterns:
                json_match = re.search(pattern, response_content)
                if json_match:
                    try:
                        json_str = json_match.group(1) if '```' in pattern else json_match.group(0)
                        exploration_summary = json.loads(json_str)
                        logger.info(f"[Analyst] Parsed exploration_summary with keys: {list(exploration_summary.keys())}")
                        break
                    except json.JSONDecodeError:
                        continue
            
            # Fallback: extract key info from text
            if not exploration_summary:
                logger.warning("[Analyst] Could not parse JSON, creating fallback exploration_summary")
                exploration_summary = {
                    "key_findings": self._extract_findings_from_text(response_content),
                    "summary": response_content[:1500],
                    "raw_response": True,
                }
        
        return AnalystOutput(
            hypothesis_evaluations=evaluations,
            insights=[],
            summary=str(response_content),
            answers_question=not is_exploration,  # Phase 1 doesn't answer yet
            confidence=0.5,
            needs_more_analysis=is_exploration,  # Phase 1 always needs Phase 2
            exploration_summary=exploration_summary,
        )
    
    def _extract_findings_from_text(self, text: str) -> list[str]:
        """Extract bullet points or numbered items from text as findings."""
        import re
        findings = []
        
        # Look for bullet points or numbered items
        patterns = [
            r'[-•]\s*(.{20,100})',   # Bullet points
            r'\d+\.\s*(.{20,100})',   # Numbered items
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            findings.extend(matches[:3])
        
        if not findings:
            # Take first 3 sentences
            sentences = text.split('.')[:3]
            findings = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
        
        return findings[:5]
