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
    execution_results: dict[int, ExecutionResult]
    original_question: str


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
        
        Args:
            input_data: Analyst input with results and plan
            
        Returns:
            Evaluations, insights, and summary
        """
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
                # Format output based on type
                output_str = self._format_output(result)
                prompt_parts.append(f"\n```\n{output_str}\n```")
            
            if result.error_message:
                prompt_parts.append(f"Error: {result.error_message}")
        
        prompt_parts.append(
            "\n\nPlease:\n"
            "1. Evaluate each hypothesis (validated/invalidated/needs more data)\n"
            "2. Generate key insights from the data\n"
            "3. Provide a summary answering the original question\n"
            "4. Rate your confidence in the overall answer"
        )
        
        prompt = "\n".join(prompt_parts)
        
        # Call LLM
        response = await self.invoke_llm([HumanMessage(content=prompt)])
        
        # Parse response
        return self._parse_response(response.content, input_data)
    
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
    ) -> AnalystOutput:
        """
        Parse LLM response into AnalystOutput.
        
        TODO: Implement proper structured output parsing.
        """
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
        
        return AnalystOutput(
            hypothesis_evaluations=evaluations,
            insights=[],
            summary=str(response_content),
            answers_question=False,  # TODO: Parse from response
            confidence=0.5,
            needs_more_analysis=True,
        )
