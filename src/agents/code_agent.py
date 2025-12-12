"""
Code Agent - SQL/Python code generation and execution.

Role: "Tạo ra một Senior Python/SQL Developer"
Responsibilities:
- Generate SQL queries and Python code from approved plans
- Execute code in sandbox environment
- Handle errors and retry with fixes
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from .base import AgentCard, AgentRole, BaseAgent
from .planner import AnalysisPlan, AnalysisStep

logger = logging.getLogger(__name__)


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
            "output_type": self.output_type.value,
            "output": self.output,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "traceback": self.traceback,
        }


@dataclass
class GeneratedCode:
    """Generated code from a plan step."""
    
    step_id: str  # e.g., "s1", "s2"
    hypothesis_id: str  # e.g., "h1", "h2"
    language: str  # "sql" or "python"
    code: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "hypothesis_id": self.hypothesis_id,
            "language": self.language,
            "code": self.code,
            "description": self.description,
            "dependencies": self.dependencies,
        }


@dataclass
class CodeAgentInput:
    """Input for the Code Agent."""
    
    plan: AnalysisPlan
    schema_context: Dict[str, Any] = field(default_factory=dict)  # Tables, columns, joins
    step_to_execute: AnalysisStep | None = None  # If None, generate all code
    previous_results: Dict[str, ExecutionResult] = field(default_factory=dict)
    error_to_fix: str | None = None
    retry_count: int = 0


@dataclass
class CodeAgentOutput:
    """Output from the Code Agent."""
    
    generated_code: List[GeneratedCode]
    execution_results: Dict[str, ExecutionResult] = field(default_factory=dict)
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
                    "schema_context": {"type": "object"},
                    "step_to_execute": {"type": "object", "nullable": True},
                    "previous_results": {"type": "object"},
                    "error_to_fix": {"type": "string", "nullable": True},
                },
                "required": ["plan"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "generated_code": {"type": "array"},
                    "execution_results": {"type": "object"},
                    "all_success": {"type": "boolean"},
                },
            },
        )
    
    def _default_system_prompt(self) -> str:
        return """You are a Senior Python/SQL Developer with expertise in data analysis.

Your role is to:
1. Translate analysis plan requirements into executable SQL and Python code
2. Generate clean, efficient, and correct code
3. Handle execution errors and generate fixes
4. Create visualizations when needed

IMPORTANT: You receive HIGH-LEVEL requirements, NOT specific SQL. You must:
- Understand the data requirements (data_needed, filters, grouping)
- Use the provided schema context (tables, columns, joins) to write correct SQL
- Generate proper queries that match the schema

When generating SQL:
- Use proper PostgreSQL syntax
- Use exact column and table names from the schema
- Include appropriate JOINs based on schema relationships
- Add comments explaining the query logic
- Handle NULL values appropriately

When generating Python for visualization:
- Use pandas for data manipulation
- Use matplotlib/plotly for visualization
- Generate clean, readable charts
- Add proper labels and titles

If you receive an error:
- Analyze the error message and traceback
- Identify the root cause
- Generate a corrected version of the code

OUTPUT FORMAT:
For each step, output code in markdown code blocks with the step ID:

### Step s1: [description]
```sql
-- Your SQL query here
SELECT ...
```

### Step s2: [description]
```python
# Your Python code here
import pandas as pd
...
```
"""
    
    async def process(self, input_data: CodeAgentInput) -> CodeAgentOutput:
        """
        Generate and execute code for the analysis plan.
        
        Args:
            input_data: Code agent input with plan and context
            
        Returns:
            Generated code and execution results
        """
        logger.info(f"[CodeAgent] Processing plan with {len(input_data.plan.steps)} steps")
        
        # Determine which steps to process
        if input_data.step_to_execute:
            steps = [input_data.step_to_execute]
        else:
            steps = input_data.plan.steps
        
        # Build prompt with schema context
        prompt = self._build_prompt(input_data, steps)
        
        # Call LLM
        logger.info("[CodeAgent] Generating code...")
        response = await self.invoke_llm([HumanMessage(content=prompt)])
        
        # Parse generated code
        generated_code = self._parse_code_response(response.content, steps)
        
        logger.info(f"[CodeAgent] Generated {len(generated_code)} code blocks")
        for code in generated_code:
            logger.info(f"  - {code.step_id}: [{code.language}] {code.description[:50]}...")
        
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
    
    def _build_prompt(
        self,
        input_data: CodeAgentInput,
        steps: List[AnalysisStep],
    ) -> str:
        """Build the prompt for LLM with schema context."""
        prompt_parts = [
            f"## Analysis Task\n**Question:** {input_data.plan.question}",
        ]
        
        # Add schema context
        schema = input_data.schema_context
        if schema:
            prompt_parts.append("\n## Available Schema\n")
            
            # Tables
            tables = schema.get("tables", [])
            if tables:
                prompt_parts.append("### Tables")
                for t in tables:
                    if isinstance(t, dict):
                        prompt_parts.append(f"- **{t.get('table_name', t)}**: {t.get('description', '')[:100]}")
                    else:
                        prompt_parts.append(f"- {t}")
            
            # Columns
            columns = schema.get("columns", [])
            if columns:
                prompt_parts.append("\n### Key Columns")
                # Group by table
                by_table = {}
                for c in columns:
                    if isinstance(c, dict):
                        tbl = c.get("table_name", "unknown")
                        by_table.setdefault(tbl, []).append(c)
                
                for tbl, cols in by_table.items():
                    prompt_parts.append(f"\n**{tbl}**:")
                    for c in cols[:10]:  # Limit columns shown
                        col_name = c.get("column_name", "")
                        col_type = c.get("data_type", "")
                        col_desc = c.get("description", "")[:50]
                        prompt_parts.append(f"  - `{col_name}` ({col_type}): {col_desc}")
            
            # Joins
            joins = schema.get("joins", [])
            if joins:
                prompt_parts.append("\n### Available Joins")
                for j in joins[:5]:  # Limit joins shown
                    if isinstance(j, dict):
                        prompt_parts.append(
                            f"- {j.get('from_table')} → {j.get('to_table')}: {j.get('on_clause', '')}"
                        )
        
        # Add steps to implement
        prompt_parts.append("\n## Steps to Implement\n")
        
        for step in steps:
            step_id = step.id if hasattr(step, 'id') else f"s{step.step_number}"
            hypo_id = step.hypothesis_id if hasattr(step, 'hypothesis_id') else ""
            action = step.action_type
            desc = step.description
            
            prompt_parts.append(f"### {step_id} ({hypo_id}): {desc}")
            prompt_parts.append(f"**Action Type:** {action}")
            
            # Add requirements
            reqs = step.requirements if hasattr(step, 'requirements') else step.details
            if reqs:
                prompt_parts.append("**Requirements:**")
                if isinstance(reqs, dict):
                    for k, v in reqs.items():
                        prompt_parts.append(f"  - {k}: {v}")
            
            # Add dependencies
            deps = step.depends_on if hasattr(step, 'depends_on') else step.dependencies
            if deps:
                prompt_parts.append(f"**Depends on:** {deps}")
            
            prompt_parts.append("")
        
        # Add previous results if any
        if input_data.previous_results:
            prompt_parts.append("\n## Previous Step Results")
            for step_id, result in input_data.previous_results.items():
                prompt_parts.append(
                    f"- {step_id}: {result.status.value} - {result.output_type.value}"
                )
        
        # Add error to fix if any
        if input_data.error_to_fix:
            prompt_parts.append(
                f"\n## ⚠️ Error to Fix (Retry {input_data.retry_count}/{self.max_retries})\n"
                f"```\n{input_data.error_to_fix}\n```\n"
                "Please analyze the error and generate corrected code."
            )
        else:
            prompt_parts.append(
                "\n\nPlease generate the code for each step. "
                "Use markdown code blocks with language specification (sql or python)."
            )
        
        return "\n".join(prompt_parts)
    
    def _parse_code_response(
        self,
        response_content: str,
        steps: List[AnalysisStep],
    ) -> List[GeneratedCode]:
        """Parse LLM response to extract code blocks."""
        generated = []
        
        # Create step map for reference
        step_map = {}
        for step in steps:
            step_id = step.id if hasattr(step, 'id') else f"s{step.step_number}"
            step_map[step_id] = step
        
        # Pattern to match step headers and code blocks
        # Match: ### Step s1: description
        step_pattern = r'###\s*(?:Step\s*)?(\w+):?\s*(.+?)(?=\n)'
        code_pattern = r'```(\w+)\n(.*?)```'
        
        # Find all step sections
        sections = re.split(r'(###\s*(?:Step\s*)?\w+)', response_content)
        
        current_step_id = None
        current_description = ""
        
        for section in sections:
            # Check if this is a step header
            step_match = re.match(r'###\s*(?:Step\s*)?(\w+)', section)
            if step_match:
                current_step_id = step_match.group(1).lower()
                continue
            
            # Find code blocks in this section
            code_blocks = re.findall(code_pattern, section, re.DOTALL)
            
            for language, code in code_blocks:
                # Determine step_id
                if current_step_id and current_step_id in step_map:
                    step = step_map[current_step_id]
                    step_id = current_step_id
                    hypo_id = step.hypothesis_id if hasattr(step, 'hypothesis_id') else ""
                    description = step.description
                elif steps:
                    # Use first unprocessed step
                    for s in steps:
                        s_id = s.id if hasattr(s, 'id') else f"s{s.step_number}"
                        if s_id not in [g.step_id for g in generated]:
                            step_id = s_id
                            hypo_id = s.hypothesis_id if hasattr(s, 'hypothesis_id') else ""
                            description = s.description
                            break
                    else:
                        step_id = f"s{len(generated) + 1}"
                        hypo_id = ""
                        description = "Generated code"
                else:
                    step_id = f"s{len(generated) + 1}"
                    hypo_id = ""
                    description = "Generated code"
                
                generated.append(GeneratedCode(
                    step_id=step_id,
                    hypothesis_id=hypo_id,
                    language=language.lower(),
                    code=code.strip(),
                    description=description,
                ))
        
        # If no code was parsed, create placeholders
        if not generated:
            logger.warning("[CodeAgent] No code blocks found in response, creating placeholders")
            for step in steps:
                step_id = step.id if hasattr(step, 'id') else f"s{step.step_number}"
                hypo_id = step.hypothesis_id if hasattr(step, 'hypothesis_id') else ""
                language = "python" if step.action_type in ["visualization", "analysis"] else "sql"
                
                generated.append(GeneratedCode(
                    step_id=step_id,
                    hypothesis_id=hypo_id,
                    language=language,
                    code=f"-- Placeholder for {step_id}\n-- {step.description}",
                    description=step.description,
                ))
        
        return generated
    
    async def _execute_code(
        self,
        code_list: List[GeneratedCode],
    ) -> Dict[str, ExecutionResult]:
        """
        Execute generated code.
        
        TODO: Implement actual execution via MCP/sandbox.
        """
        results = {}
        for code in code_list:
            # Placeholder - actual execution via MCP server
            logger.info(f"[CodeAgent] Executing {code.step_id} ({code.language})...")
            
            results[code.step_id] = ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output_type=OutputType.TEXT if code.language == "sql" else OutputType.DATAFRAME,
                output=f"[Placeholder] Code executed: {code.step_id}",
                execution_time_ms=100,
            )
        
        return results
    
    async def fix_error(
        self,
        error_message: str,
        failed_code: GeneratedCode,
        traceback: str | None = None,
    ) -> GeneratedCode:
        """
        Fix code that produced an error.
        
        Args:
            error_message: The error that occurred
            failed_code: The code that failed
            traceback: Optional traceback
            
        Returns:
            Fixed code
        """
        traceback_section = ""
        if traceback:
            traceback_section = f"## Traceback\n```\n{traceback}\n```\n"
        
        prompt = f"""The following code produced an error. Please fix it.

## Original Code ({failed_code.language})
```{failed_code.language}
{failed_code.code}
```

## Error Message
```
{error_message}
```

{traceback_section}
Please provide the corrected code in a markdown code block.
"""
        response = await self.invoke_llm([HumanMessage(content=prompt)])
        
        # Extract fixed code
        code_match = re.search(
            rf'```{failed_code.language}\n(.*?)```',
            response.content,
            re.DOTALL
        )
        
        if code_match:
            fixed_code = code_match.group(1).strip()
        else:
            # Try without language specifier
            code_match = re.search(r'```\n(.*?)```', response.content, re.DOTALL)
            fixed_code = code_match.group(1).strip() if code_match else failed_code.code
        
        return GeneratedCode(
            step_id=failed_code.step_id,
            hypothesis_id=failed_code.hypothesis_id,
            language=failed_code.language,
            code=fixed_code,
            description=f"[Fixed] {failed_code.description}",
            dependencies=failed_code.dependencies,
        )
    
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
