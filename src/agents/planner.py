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


@dataclass
class AnalysisStep:
    """A single step in the analysis plan."""
    
    step_number: int
    description: str
    action_type: str  # "sql", "python", "visualization"
    details: dict[str, Any] = field(default_factory=dict)
    dependencies: list[int] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stepNumber": self.step_number,
            "description": self.description,
            "actionType": self.action_type,
            "details": self.details,
            "dependencies": self.dependencies,
        }


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
        return """You are a creative and experienced Data Scientist.

Your role is to:
1. Analyze user questions about data and business metrics
2. Generate testable hypotheses that could explain the phenomena
3. Create detailed analysis plans with concrete SQL/Python steps
4. Consider multiple angles and alternative explanations

When creating plans:
- Break down complex questions into smaller, testable hypotheses
- Specify exact SQL queries or Python operations needed
- Consider data availability and quality
- Think about business context and domain knowledge
- Be creative but grounded in data

Output your response in a structured format with:
- List of hypotheses with rationale
- Step-by-step analysis plan
- Confidence level in your approach

If you receive feedback from the Critic, incorporate it to improve your plan."""
    
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
        
        TODO: Implement proper structured output parsing.
        For now, returns a placeholder plan.
        """
        version = 1
        if input_data.previous_plan:
            version = input_data.previous_plan.version + 1
        
        # Placeholder - actual implementation should parse LLM output
        return AnalysisPlan(
            question=input_data.question,
            hypotheses=[
                Hypothesis(
                    id="h1",
                    statement="Placeholder hypothesis",
                    rationale="To be parsed from LLM response",
                )
            ],
            steps=[
                AnalysisStep(
                    step_number=1,
                    description="Placeholder step",
                    action_type="sql",
                )
            ],
            context_used=input_data.enriched_context,
            version=version,
        )
