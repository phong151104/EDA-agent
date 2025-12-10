"""
Code Agent - SQL/Python code generation and execution.

Role: "Tạo ra một Senior Python/SQL Developer"
Responsibilities:
- Generate SQL queries and Python code from plans
- Execute code in sandbox environment
- Handle errors and retry with fixes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage

from .base import AgentCard, AgentRole, BaseAgent
from .planner import AnalysisPlan, AnalysisStep


class ExecutionStatus(str, Enum):
    """Code execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class OutputType(str, Enum):
    """Type of execution output."""
    
    DATAFRAME = "dataframe"
    JSON = "json"
    IMAGE = "image"
    TEXT = "text"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result of code execution."""
    
    status: ExecutionStatus
    output_type: OutputType
    output: Any
    execution_time_ms: int = 0
    error_message: str | None = None
    traceback: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "outputType": self.output_type.value,
            "output": self.output,
            "executionTimeMs": self.execution_time_ms,
            "errorMessage": self.error_message,
            "traceback": self.traceback,
        }


@dataclass
class GeneratedCode:
    """Generated code from a plan step."""
    
    step_number: int
    language: str  # "sql" or "python"
    code: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stepNumber": self.step_number,
            "language": self.language,
            "code": self.code,
            "description": self.description,
            "dependencies": self.dependencies,
        }


@dataclass
class CodeAgentInput:
    """Input for the Code Agent."""
    
    plan: AnalysisPlan
    step_to_execute: AnalysisStep | None = None  # If None, generate all code
    previous_results: dict[int, ExecutionResult] = field(default_factory=dict)
    error_to_fix: str | None = None
    retry_count: int = 0


@dataclass
class CodeAgentOutput:
    """Output from the Code Agent."""
    
    generated_code: list[GeneratedCode]
    execution_results: dict[int, ExecutionResult] = field(default_factory=dict)
    all_success: bool = False


class CodeAgent(BaseAgent[CodeAgentInput, CodeAgentOutput]):
    """
    Code Agent - Generates and executes SQL/Python code.
    
    This agent acts as a Senior Developer, translating analysis
    plans into executable code and handling execution errors.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """
        Initialize Code Agent.
        
        Args:
            max_retries: Maximum code execution retries
            **kwargs: Base agent arguments
        """
        super().__init__(**kwargs)
        self.max_retries = max_retries
    
    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name="CodeAgent",
            description="Senior Python/SQL Developer that generates and executes code",
            role=AgentRole.CODE_AGENT,
            capabilities=[
                "sql_generation",
                "python_generation",
                "code_execution",
                "error_handling",
                "visualization",
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "plan": {"type": "object"},
                    "stepToExecute": {"type": "object", "nullable": True},
                    "previousResults": {"type": "object"},
                    "errorToFix": {"type": "string", "nullable": True},
                },
                "required": ["plan"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "generatedCode": {"type": "array"},
                    "executionResults": {"type": "object"},
                    "allSuccess": {"type": "boolean"},
                },
            },
        )
    
    def _default_system_prompt(self) -> str:
        return """You are a Senior Python/SQL Developer with expertise in data analysis.

Your role is to:
1. Translate analysis plans into executable SQL and Python code
2. Generate clean, efficient, and correct code
3. Handle execution errors and generate fixes
4. Create visualizations when needed

When generating SQL:
- Use proper PostgreSQL syntax
- Include appropriate JOINs based on schema
- Add comments explaining complex logic
- Handle NULL values appropriately
- Use parameterized queries when needed

When generating Python:
- Use pandas for data manipulation
- Use matplotlib/seaborn for visualization
- Handle edge cases gracefully
- Add error handling

If you receive an error:
- Analyze the error message and traceback
- Identify the root cause
- Generate a corrected version of the code

Output code in markdown code blocks with language specification."""
    
    async def process(self, input_data: CodeAgentInput) -> CodeAgentOutput:
        """
        Generate and execute code for the analysis plan.
        
        Args:
            input_data: Code agent input with plan and context
            
        Returns:
            Generated code and execution results
        """
        # Determine which steps to process
        if input_data.step_to_execute:
            steps = [input_data.step_to_execute]
        else:
            steps = input_data.plan.steps
        
        # Build prompt
        prompt_parts = [
            f"## Analysis Plan\nQuestion: {input_data.plan.question}",
            "\n### Steps to Implement",
        ]
        
        for step in steps:
            prompt_parts.append(
                f"\n**Step {step.step_number}** [{step.action_type}]\n"
                f"{step.description}"
            )
            if step.details:
                prompt_parts.append(f"Details: {step.details}")
        
        if input_data.previous_results:
            prompt_parts.append("\n### Previous Step Results")
            for step_num, result in input_data.previous_results.items():
                prompt_parts.append(
                    f"Step {step_num}: {result.status.value} - "
                    f"{result.output_type.value}"
                )
        
        if input_data.error_to_fix:
            prompt_parts.append(
                f"\n### Error to Fix (Retry {input_data.retry_count}/{self.max_retries})\n"
                f"```\n{input_data.error_to_fix}\n```\n"
                "Please analyze the error and generate corrected code."
            )
        else:
            prompt_parts.append(
                "\n\nPlease generate the code for each step. "
                "Use markdown code blocks with language specification."
            )
        
        prompt = "\n".join(prompt_parts)
        
        # Call LLM
        response = await self.invoke_llm([HumanMessage(content=prompt)])
        
        # Parse generated code
        generated_code = self._parse_code_response(response.content, steps)
        
        # Execute code (placeholder - actual execution via MCP)
        execution_results = await self._execute_code(generated_code)
        
        all_success = all(
            r.status == ExecutionStatus.SUCCESS
            for r in execution_results.values()
        )
        
        return CodeAgentOutput(
            generated_code=generated_code,
            execution_results=execution_results,
            all_success=all_success,
        )
    
    def _parse_code_response(
        self,
        response_content: str,
        steps: list[AnalysisStep],
    ) -> list[GeneratedCode]:
        """
        Parse LLM response to extract code blocks.
        
        TODO: Implement proper code block extraction.
        """
        # Placeholder - actual implementation should parse markdown code blocks
        generated = []
        for step in steps:
            language = "sql" if step.action_type == "sql" else "python"
            generated.append(
                GeneratedCode(
                    step_number=step.step_number,
                    language=language,
                    code=f"-- Placeholder code for step {step.step_number}",
                    description=step.description,
                )
            )
        return generated
    
    async def _execute_code(
        self,
        code_list: list[GeneratedCode],
    ) -> dict[int, ExecutionResult]:
        """
        Execute generated code.
        
        TODO: Implement actual execution via MCP/sandbox.
        """
        results = {}
        for code in code_list:
            # Placeholder - actual execution via MCP server
            results[code.step_number] = ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output_type=OutputType.TEXT,
                output="Execution placeholder - implement via MCP",
                execution_time_ms=100,
            )
        return results
    
    async def validate_sql_syntax(self, sql: str) -> dict[str, Any]:
        """
        Validate SQL syntax without executing.
        
        Used by Critic for feasibility checking.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Validation result with errors if any
        """
        # TODO: Implement via MCP with EXPLAIN
        return {
            "valid": True,
            "errors": [],
        }
