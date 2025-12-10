"""
Critic Agent - Plan validation and feedback.

Role: "Tạo ra một Business Expert kỹ tính"
Responsibilities:
- Validate plans against metadata and business rules
- Check execution feasibility with Code Agent
- Provide constructive feedback for plan refinement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage

from .base import AgentCard, AgentRole, BaseAgent
from .planner import AnalysisPlan


class ValidationStatus(str, Enum):
    """Plan validation status."""
    
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class ValidationCategory(str, Enum):
    """Category of validation issue."""
    
    SCHEMA = "schema"  # Schema/column doesn't exist
    BUSINESS_LOGIC = "business_logic"  # Violates business rules
    FEASIBILITY = "feasibility"  # Can't be executed
    DATA_QUALITY = "data_quality"  # Data issues
    SEMANTIC = "semantic"  # Logical/semantic error
    HALLUCINATION = "hallucination"  # Made up information


@dataclass
class ValidationIssue:
    """A single validation issue found in the plan."""
    
    category: ValidationCategory
    severity: str  # "error", "warning", "info"
    message: str
    location: str  # Where in the plan (e.g., "hypothesis_1", "step_3")
    suggestion: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
        }


@dataclass
class CriticInput:
    """Input for the Critic agent."""
    
    plan: AnalysisPlan
    metadata_context: dict[str, Any]
    feasibility_check: dict[str, Any] | None = None  # From Code Agent dry-run


@dataclass
class CriticOutput:
    """Output from the Critic agent."""
    
    status: ValidationStatus
    approval_score: float  # 0.0 - 1.0
    issues: list[ValidationIssue] = field(default_factory=list)
    feedback: str = ""
    approved_plan: AnalysisPlan | None = None


class CriticAgent(BaseAgent[CriticInput, CriticOutput]):
    """
    Critic Agent - Validates and critiques analysis plans.
    
    This agent acts as a meticulous Business Expert, ensuring
    plans are valid, feasible, and aligned with business rules.
    """
    
    def __init__(
        self,
        approval_threshold: float = 0.8,
        **kwargs: Any,
    ):
        """
        Initialize Critic agent.
        
        Args:
            approval_threshold: Minimum score to approve a plan
            **kwargs: Base agent arguments
        """
        super().__init__(**kwargs)
        self.approval_threshold = approval_threshold
    
    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="Critic",
            description="Meticulous Business Expert that validates analysis plans",
            role=AgentRole.CRITIC,
            capabilities=[
                "schema_validation",
                "business_rule_check",
                "feasibility_assessment",
                "feedback_generation",
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "plan": {"type": "object"},
                    "metadataContext": {"type": "object"},
                    "feasibilityCheck": {"type": "object", "nullable": True},
                },
                "required": ["plan", "metadataContext"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "approvalScore": {"type": "number"},
                    "issues": {"type": "array"},
                    "feedback": {"type": "string"},
                },
            },
        )
    
    def _default_system_prompt(self) -> str:
        return """You are a meticulous Business Expert and Data Quality Analyst.

Your role is to:
1. Validate analysis plans against available schema and metadata
2. Check for business logic errors and inconsistencies
3. Identify potential hallucinations or made-up information
4. Assess execution feasibility
5. Provide constructive, actionable feedback

When reviewing plans:
- Verify all referenced tables and columns exist in the schema
- Check that SQL queries are syntactically valid
- Ensure business metrics are calculated correctly
- Look for logical errors in hypotheses
- Consider edge cases and data quality issues

Be critical but constructive. Your feedback should help improve the plan.

For each issue found, specify:
- Category (schema, business_logic, feasibility, semantic, hallucination)
- Severity (error, warning, info)
- Clear description of the problem
- Suggested fix if possible

Provide an overall approval score from 0.0 to 1.0."""
    
    async def process(self, input_data: CriticInput) -> CriticOutput:
        """
        Validate an analysis plan.
        
        Args:
            input_data: Critic input with plan and metadata
            
        Returns:
            Validation result with issues and feedback
        """
        # Build validation prompt
        prompt_parts = [
            "## Analysis Plan to Review",
            f"Question: {input_data.plan.question}",
            f"\n### Hypotheses",
        ]
        
        for h in input_data.plan.hypotheses:
            prompt_parts.append(f"- {h.id}: {h.statement}")
            prompt_parts.append(f"  Rationale: {h.rationale}")
        
        prompt_parts.append("\n### Analysis Steps")
        for s in input_data.plan.steps:
            prompt_parts.append(
                f"{s.step_number}. [{s.action_type}] {s.description}"
            )
            if s.details:
                prompt_parts.append(f"   Details: {s.details}")
        
        prompt_parts.append("\n## Available Schema Context")
        prompt_parts.append(self._format_metadata(input_data.metadata_context))
        
        if input_data.feasibility_check:
            prompt_parts.append("\n## Feasibility Check Results")
            prompt_parts.append(self._format_feasibility(input_data.feasibility_check))
        
        prompt_parts.append(
            "\n\nPlease review this plan and provide:\n"
            "1. List of issues found (with category, severity, and suggestions)\n"
            "2. Overall approval score (0.0-1.0)\n"
            "3. Summary feedback for the Planner"
        )
        
        prompt = "\n".join(prompt_parts)
        
        # Call LLM
        response = await self.invoke_llm([HumanMessage(content=prompt)])
        
        # Parse response
        result = self._parse_response(response.content, input_data.plan)
        
        # Determine final status
        if result.approval_score >= self.approval_threshold:
            if not any(i.severity == "error" for i in result.issues):
                result.status = ValidationStatus.APPROVED
                result.approved_plan = input_data.plan
            else:
                result.status = ValidationStatus.NEEDS_REVISION
        else:
            result.status = ValidationStatus.NEEDS_REVISION
        
        return result
    
    def _format_metadata(self, metadata: dict[str, Any]) -> str:
        """Format metadata context for prompt."""
        parts = []
        
        if "tables" in metadata:
            parts.append("### Tables")
            for table in metadata["tables"]:
                if isinstance(table, dict):
                    parts.append(f"- {table.get('name', table)}: {table.get('description', '')}")
                else:
                    parts.append(f"- {table}")
        
        if "columns" in metadata:
            parts.append("\n### Columns")
            for col in metadata["columns"]:
                if isinstance(col, dict):
                    parts.append(
                        f"- {col.get('table', '')}.{col.get('name', col)}: "
                        f"{col.get('type', '')} - {col.get('description', '')}"
                    )
                else:
                    parts.append(f"- {col}")
        
        if "business_rules" in metadata:
            parts.append("\n### Business Rules")
            for rule in metadata["business_rules"]:
                parts.append(f"- {rule}")
        
        return "\n".join(parts) if parts else "No metadata available."
    
    def _format_feasibility(self, feasibility: dict[str, Any]) -> str:
        """Format feasibility check results."""
        parts = []
        
        if "sql_valid" in feasibility:
            status = "✓" if feasibility["sql_valid"] else "✗"
            parts.append(f"SQL Syntax: {status}")
        
        if "tables_exist" in feasibility:
            status = "✓" if feasibility["tables_exist"] else "✗"
            parts.append(f"Tables Exist: {status}")
        
        if "estimated_rows" in feasibility:
            parts.append(f"Estimated Rows: {feasibility['estimated_rows']}")
        
        if "errors" in feasibility:
            parts.append("Errors:")
            for error in feasibility["errors"]:
                parts.append(f"  - {error}")
        
        return "\n".join(parts) if parts else "No feasibility check performed."
    
    def _parse_response(
        self,
        response_content: str,
        plan: AnalysisPlan,
    ) -> CriticOutput:
        """
        Parse LLM response into CriticOutput.
        
        TODO: Implement proper structured output parsing.
        """
        # Placeholder - actual implementation should parse LLM output
        return CriticOutput(
            status=ValidationStatus.NEEDS_REVISION,
            approval_score=0.7,
            issues=[],
            feedback=str(response_content),
        )
