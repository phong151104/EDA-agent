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
    SKIPPED = "skipped"  # Visualization skipped due to failed SQL dependency


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
        Generate code for the analysis plan using MCP tools.
        
        For SQL steps: Uses MCP text_to_sql tool
        For Python steps: Generates visualization code based on SQL results
        
        Args:
            input_data: Code agent input with plan and context
            
        Returns:
            Generated code and execution results
        """
        from src.mcp.server import MCPServer
        
        logger.info(f"[CodeAgent] Processing plan with {len(input_data.plan.steps)} steps")
        
        # Initialize MCP Server
        mcp = MCPServer()
        
        # Determine which steps to process
        if input_data.step_to_execute:
            steps = [input_data.step_to_execute]
        else:
            steps = input_data.plan.steps
        
        generated_code: List[GeneratedCode] = []
        execution_results: Dict[str, ExecutionResult] = {}
        
        for step in steps:
            step_id = step.id if hasattr(step, 'id') else f"s{step.step_number}"
            hypo_id = step.hypothesis_id if hasattr(step, 'hypothesis_id') else ""
            action_type = step.action_type.lower()
            
            logger.info(f"[CodeAgent] Processing step {step_id} ({action_type})")
            
            if action_type in ["query", "sql", "aggregation"]:
                # Use MCP text_to_sql tool
                code, result = await self._generate_sql_via_mcp(
                    mcp=mcp,
                    step=step,
                    step_id=step_id,
                    hypo_id=hypo_id,
                    question=input_data.plan.question,
                    schema_context=input_data.schema_context,
                )
                generated_code.append(code)
                execution_results[step_id] = result
                
            elif action_type in ["visualization", "chart", "plot"]:
                # Generate and execute Python visualization code
                code, result = await self._generate_visualization_code(
                    mcp=mcp,
                    step=step,
                    step_id=step_id,
                    hypo_id=hypo_id,
                    previous_results=execution_results,
                )
                generated_code.append(code)
                execution_results[step_id] = result
                
            elif action_type in ["analysis", "calculation", "comparison"]:
                # Generate and execute Python analysis code
                code, result = await self._generate_analysis_code(
                    mcp=mcp,
                    step=step,
                    step_id=step_id,
                    hypo_id=hypo_id,
                    previous_results=execution_results,
                )
                generated_code.append(code)
                execution_results[step_id] = result
                
            else:
                # Fallback: use LLM to generate code
                code, result = await self._generate_code_with_llm(
                    step=step,
                    step_id=step_id,
                    hypo_id=hypo_id,
                    input_data=input_data,
                )
                generated_code.append(code)
                execution_results[step_id] = result
        
        logger.info(f"[CodeAgent] Generated {len(generated_code)} code blocks")
        for code in generated_code:
            logger.info(f"  - {code.step_id}: [{code.language}] {code.description[:50]}...")
        
        all_success = all(
            r.status == ExecutionStatus.SUCCESS
            for r in execution_results.values()
        )
        
        return CodeAgentOutput(
            generated_code=generated_code,
            execution_results=execution_results,
            all_success=all_success,
        )
    
    async def _generate_sql_via_mcp(
        self,
        mcp,
        step: AnalysisStep,
        step_id: str,
        hypo_id: str,
        question: str,
        schema_context=None,  # NEW: schema context for error hints
        max_retries: int = 3,  # max retry attempts
    ) -> tuple[GeneratedCode, ExecutionResult]:
        """Generate SQL using MCP text_to_sql tool and execute it with retry on error."""
        
        # Build natural language prompt from step requirements
        reqs = step.requirements if hasattr(step, 'requirements') else step.details or {}
        
        data_needed = reqs.get("data_needed", [])
        filters = reqs.get("filters", [])
        grouping = reqs.get("grouping", [])
        tables_hint = reqs.get("tables_hint", [])
        
        # Build natural language prompt for text_to_sql
        nl_prompt = f"""Yêu cầu: {step.description}

Dữ liệu cần lấy: {', '.join(data_needed) if data_needed else 'theo mô tả'}
Điều kiện lọc: {', '.join(filters) if filters else 'không có'}
Nhóm theo: {grouping if grouping else 'không nhóm'}
Bảng gợi ý: {', '.join(tables_hint) if tables_hint else 'tự chọn'}

Câu hỏi gốc: {question}"""
        
        last_error = None
        last_sql = None
        
        # === RETRY LOOP with error feedback ===
        for attempt in range(1, max_retries + 1):
            logger.info(f"[CodeAgent] SQL attempt {attempt}/{max_retries}: {nl_prompt[:80]}...")
            
            # Add error feedback to prompt if retrying
            prompt_with_feedback = nl_prompt
            if last_error and last_sql:
                # Build detailed error context
                error_hints = self._build_sql_error_hints(last_error, schema_context)
                
                prompt_with_feedback = f"""{nl_prompt}

⚠️ **LẦN TRƯỚC THẤT BẠI - CẦN SỬA LỖI:**

**SQL đã sinh:**
```sql
{last_sql}
```

**Lỗi gặp phải:** {last_error}

**Hướng dẫn sửa:**
{error_hints}

**YÊU CẦU:** Sinh SQL MỚI sửa lỗi trên. PHẢI kiểm tra:
1. Tên cột/bảng đúng chính xác (case-sensitive)
2. Data types phù hợp khi so sánh/aggregate
3. PostgreSQL syntax (EXTRACT, DATE_TRUNC, COALESCE, etc.)"""
            
            # Call MCP text_to_sql tool
            result = await mcp.call_tool("text_to_sql", {"prompt": prompt_with_feedback})
            
            if not result.success:
                last_error = result.error
                logger.warning(f"[CodeAgent] ⚠️ text_to_sql failed (attempt {attempt}): {result.error}")
                continue  # Retry
            
            sql = result.output.get("sql", "")
            tables_used = result.output.get("tables_used", [])
            last_sql = sql
            
            logger.info(f"[CodeAgent] ✅ SQL generated ({len(sql)} chars)")
            
            # Execute the SQL
            exec_result = await mcp.call_tool("execute_sql", {"query": sql})
            
            if exec_result.success:
                sql_data = exec_result.output.get("rows", [])
                columns = exec_result.output.get("columns", [])
                row_count = exec_result.output.get("row_count", 0)
                logger.info(f"[CodeAgent] ✅ SQL executed: {row_count} rows")
                
                return (
                    GeneratedCode(
                        step_id=step_id,
                        hypothesis_id=hypo_id,
                        language="sql",
                        code=sql,
                        description=step.description,
                    ),
                    ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        output_type=OutputType.DATAFRAME if sql_data else OutputType.TEXT,
                        output={
                            "sql": sql,
                            "tables_used": tables_used,
                            "sql_data": sql_data,
                            "columns": columns if sql_data else [],
                            "row_count": row_count if sql_data else 0,
                        },
                        execution_time_ms=result.execution_time_ms + exec_result.execution_time_ms,
                    ),
                )
            else:
                # SQL execution failed - save error for retry
                last_error = exec_result.error
                logger.warning(f"[CodeAgent] ⚠️ SQL execution failed (attempt {attempt}): {exec_result.error}")
                # Continue to next retry
        
        # All retries exhausted
        logger.error(f"[CodeAgent] ❌ SQL failed after {max_retries} attempts: {last_error}")
        
        return (
            GeneratedCode(
                step_id=step_id,
                hypothesis_id=hypo_id,
                language="sql",
                code=last_sql or f"-- Error: {last_error}",
                description=step.description,
            ),
            ExecutionResult(
                status=ExecutionStatus.ERROR,
                output_type=OutputType.ERROR,
                output=None,
                error_message=f"Failed after {max_retries} attempts: {last_error}",
            ),
        )
    
    def _build_sql_error_hints(self, error: str, schema_context) -> str:
        """Build detailed error hints based on error message and schema."""
        hints = []
        error_lower = error.lower()
        
        # Column does not exist
        if "column" in error_lower and "does not exist" in error_lower:
            hints.append("""**Lỗi: Column không tồn tại**
- Kiểm tra lại tên cột chính xác từ schema context
- Alias bảng phải đúng (vd: o.vnpay_final_amount thay vì orders.vnpay_final_amount)
- Không đoán tên cột, chỉ dùng cột có trong schema""")
        
        # Function does not exist
        if "function" in error_lower and "does not exist" in error_lower:
            hints.append("""**Lỗi: Function không tồn tại (PostgreSQL syntax)**

POSTGRESQL DATE FUNCTIONS:
- EXTRACT(DOW FROM timestamp) -- 0=Sunday, 6=Saturday
- EXTRACT(MONTH FROM timestamp), EXTRACT(YEAR FROM timestamp)
- DATE_TRUNC('month', timestamp)
- TO_CHAR(timestamp, 'YYYY-MM-DD')
- CURRENT_DATE, CURRENT_TIMESTAMP, NOW()

KHÔNG DÙNG:
- day_of_week() → Dùng: EXTRACT(DOW FROM date)
- WEEK() → Dùng: EXTRACT(WEEK FROM date)
- DAYNAME() → Dùng: TO_CHAR(date, 'Day')""")
        
        # Type mismatch / invalid input syntax
        if "invalid input syntax" in error_lower or "type" in error_lower:
            hints.append("""**Lỗi: Type mismatch**
- Không so sánh text với integer
- Cast nếu cần: CAST(column AS INTEGER) hoặc CAST(column AS VARCHAR)
- SUM/AVG chỉ dùng với số, không dùng với text
- Kiểm tra data type thực sự trong schema trước khi aggregate""")
        
        # Sum/aggregate on wrong type
        if "sum(text)" in error_lower or "aggregate" in error_lower:
            hints.append("""**Lỗi: Aggregate trên sai type**
- SUM() chỉ dùng với cột số (INTEGER, BIGINT, DOUBLE)
- Kiểm tra data type thực sự của cột
- Có thể cần: SUM(CAST(column AS DOUBLE))
- Hoặc dùng cột số khác có nghĩa tương đương""")
        
        # Extract relevant schema info
        if schema_context:
            tables_info = self._extract_tables_info(schema_context)
            if tables_info:
                hints.append(f"""**Schema context (columns có sẵn):**
{tables_info}""")
        
        if not hints:
            hints.append(f"""**Lỗi không xác định:** {error}
- Kiểm tra syntax Trino SQL
- Verify tên bảng/cột từ schema
- Đảm bảo data types phù hợp""")
        
        return "\n\n".join(hints)
    
    def _extract_tables_info(self, schema_context) -> str:
        """Extract relevant table/column info from schema context."""
        if not schema_context:
            return ""
        
        info_lines = []
        tables = schema_context.tables if hasattr(schema_context, 'tables') else []
        
        for table in tables[:5]:  # Limit to 5 tables
            table_name = table.name if hasattr(table, 'name') else str(table)
            cols = []
            
            if hasattr(table, 'columns'):
                for col in table.columns[:8]:  # Limit columns shown
                    col_name = col.name if hasattr(col, 'name') else str(col)
                    col_type = col.data_type if hasattr(col, 'data_type') else "unknown"
                    cols.append(f"    - {col_name} ({col_type})")
            
            if cols:
                info_lines.append(f"• {table_name}:")
                info_lines.extend(cols)
        
        return "\n".join(info_lines) if info_lines else ""
    
    async def _generate_visualization_code(
        self,
        mcp,  # MCPServer instance
        step: AnalysisStep,
        step_id: str,
        hypo_id: str,
        previous_results: Dict[str, ExecutionResult],
    ) -> tuple[GeneratedCode, ExecutionResult]:
        """Generate and execute Python visualization code with SQL data from ALL dependencies."""
        
        # Find dependent SQL steps
        deps = step.depends_on if hasattr(step, 'depends_on') else step.dependencies or []
        
        # Check if all dependencies succeeded - SKIP visualization if SQL failed
        for dep_id in deps:
            dep_result = previous_results.get(dep_id)
            if not dep_result or dep_result.status != ExecutionStatus.SUCCESS:
                logger.warning(f"[CodeAgent] ⚠️ Skipping visualization {step_id} - dependency {dep_id} failed or missing")
                return (
                    GeneratedCode(
                        step_id=step_id,
                        hypothesis_id=hypo_id,
                        language="python",
                        code="# Skipped - SQL dependency failed",
                        description=f"Biểu đồ bị bỏ qua do SQL step {dep_id} thất bại",
                    ),
                    ExecutionResult(
                        step_id=step_id,
                        status=ExecutionStatus.SKIPPED,
                        output={"skipped": True, "reason": f"SQL dependency {dep_id} failed"},
                        error_message=f"Skipped due to failed SQL dependency: {dep_id}",
                    ),
                )
        
        # Collect SQL data from ALL dependencies and get column info
        dataframes: Dict[str, list] = {}
        data_info = []
        for dep_id in deps:
            dep_result = previous_results.get(dep_id)
            if dep_result and dep_result.output:
                sql_data = dep_result.output.get("sql_data", [])
                columns = dep_result.output.get("columns", [])
                if sql_data:
                    dataframes[dep_id] = sql_data
                    data_info.append(f"- df_{dep_id}: columns = {columns}, rows = {len(sql_data)}")
        
        # Get visualization requirements from step
        reqs = step.requirements if hasattr(step, 'requirements') else step.details or {}
        chart_type = reqs.get("chart_type", "auto")
        x_axis = reqs.get("x_axis", "")
        y_axis = reqs.get("y_axis", "")
        
        # Chart description for report - use step description
        chart_description = step.description
        
        # Ask LLM to generate visualization code
        viz_prompt = f"""Tạo code Python để vẽ biểu đồ với matplotlib/seaborn.

## Yêu cầu:
- Mô tả: {step.description}
- Loại biểu đồ: {chart_type}
- Trục X: {x_axis if x_axis else 'tự động'}
- Trục Y: {y_axis if y_axis else 'tự động'}

## Dữ liệu có sẵn:
{chr(10).join(data_info) if data_info else "- df: DataFrame từ SQL result"}

## Lưu ý:
1. df đã có sẵn (được inject), KHÔNG cần tạo lại
2. Dùng matplotlib/seaborn
3. Thêm title, xlabel, ylabel đầy đủ (TIẾNG VIỆT)
4. Lưu chart ra file: plt.savefig("{step_id}_chart.png")
5. Print thông tin về chart đã tạo
6. Xử lý trường hợp df rỗng
7. Title phải mô tả nội dung chart rõ ràng bằng tiếng Việt

Trả về CHỈ CODE Python (không có markdown fences)."""

        from langchain_core.messages import HumanMessage
        response = await self.invoke_llm([HumanMessage(content=viz_prompt)])
        code = response.content.strip()
        
        # Remove markdown fences if present
        if code.startswith("```"):
            code = code.split("\n", 1)[-1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        code = code.strip()
        
        logger.info(f"[CodeAgent] LLM generated visualization code ({len(code)} chars)")
        
        # Execute code with injected DataFrames - with retry loop
        from src.mcp.tools.code_interpreter import CodeInterpreter
        from src.mcp.server import ToolResult
        
        max_retries = 3
        last_error = None
        current_code = code
        
        for attempt in range(1, max_retries + 1):
            logger.info(f"[CodeAgent] Visualization attempt {attempt}/{max_retries}")
            
            if dataframes:
                interpreter = CodeInterpreter()
                mcp_result = await interpreter.execute_with_multiple_dataframes(
                    code=current_code,
                    dataframes=dataframes
                )
                # Wrap in ToolResult format
                mcp_result = ToolResult(
                    success=mcp_result.success,
                    output={"stdout": mcp_result.output, "images": mcp_result.images},
                    error=mcp_result.error,
                    execution_time_ms=mcp_result.execution_time_ms,
                )
            else:
                mcp_result = await mcp.call_tool("execute_python", {"code": current_code})
            
            if mcp_result.success:
                logger.info(f"[CodeAgent] ✅ Visualization executed ({mcp_result.execution_time_ms}ms)")
                
                return (
                    GeneratedCode(
                        step_id=step_id,
                        hypothesis_id=hypo_id,
                        language="python",
                        code=current_code,
                        description=step.description,
                        dependencies=deps,
                    ),
                    ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        output_type=OutputType.IMAGE,
                        output={
                            "stdout": mcp_result.output.get("stdout", ""),
                            "images": mcp_result.output.get("images", []),
                            "chart_description": chart_description,  # Save for report
                        },
                        execution_time_ms=mcp_result.execution_time_ms,
                    ),
                )
            else:
                last_error = mcp_result.error
                logger.warning(f"[CodeAgent] ⚠️ Visualization failed (attempt {attempt}): {last_error}")
                
                # Ask LLM to fix the code
                if attempt < max_retries:
                    fix_prompt = f"""Code Python sau bị lỗi:

```python
{current_code}
```

Lỗi: {last_error}

Sửa code để tránh lỗi trên. Trả về CHỈCODE Python mới (không có markdown fences).
Lưu ý: df là DataFrame có sẵn từ SQL result."""
                    
                    from langchain_core.messages import HumanMessage
                    response = await self.invoke_llm([HumanMessage(content=fix_prompt)])
                    current_code = response.content.strip()
                    # Remove markdown fences if present
                    if current_code.startswith("```"):
                        current_code = current_code.split("\n", 1)[-1]
                    if current_code.endswith("```"):
                        current_code = current_code.rsplit("```", 1)[0]
                    current_code = current_code.strip()
                    logger.info(f"[CodeAgent] LLM generated fixed code ({len(current_code)} chars)")
        
        # All retries exhausted
        logger.error(f"[CodeAgent] ❌ Visualization failed after {max_retries} attempts: {last_error}")
        
        return (
            GeneratedCode(
                step_id=step_id,
                hypothesis_id=hypo_id,
                language="python",
                code=current_code,
                description=step.description,
                dependencies=deps,
            ),
            ExecutionResult(
                status=ExecutionStatus.ERROR,
                output_type=OutputType.ERROR,
                error_message=f"Failed after {max_retries} attempts: {last_error}",
            ),
        )
    
    async def _generate_analysis_code(
        self,
        mcp,  # MCPServer instance
        step: AnalysisStep,
        step_id: str,
        hypo_id: str,
        previous_results: Dict[str, ExecutionResult],
    ) -> tuple[GeneratedCode, ExecutionResult]:
        """Generate and execute Python analysis code with SQL data from ALL dependencies."""
        
        deps = step.depends_on if hasattr(step, 'depends_on') else step.dependencies or []
        
        # Collect SQL data from ALL dependencies
        dataframes: Dict[str, list] = {}
        for dep_id in deps:
            dep_result = previous_results.get(dep_id)
            if dep_result and dep_result.output:
                sql_data = dep_result.output.get("sql_data", [])
                if sql_data:
                    dataframes[dep_id] = sql_data
        
        # Generate clean analysis code
        # Code can reference: df (if single dependency) or df_s1, df_s2, etc.
        if len(dataframes) == 1:
            df_note = f"# Data available as 'df' from step {list(dataframes.keys())[0]}"
        elif len(dataframes) > 1:
            df_names = ", ".join(f"df_{k}" for k in dataframes.keys())
            df_note = f"# Data available as: {df_names}"
        else:
            df_note = "# No data from previous steps"
        
        code = f'''# Analysis for: {step.description}
{df_note}

# Perform analysis
print("="*50)
print("Analysis: {step.description}")
print("="*50)

# Check available DataFrames
if 'df' in dir() and not df.empty:
    print(f"\\nData shape: {{df.shape}}")
    print(f"\\nColumns: {{list(df.columns)}}")
    print(f"\\nFirst 5 rows:")
    print(df.head())
    print(f"\\nSummary statistics:")
    print(df.describe())
    
    # Basic insights for numeric columns
    for col in df.select_dtypes(include=[np.number]).columns[:5]:
        print(f"\\n{{col}}:")
        print(f"  Mean: {{df[col].mean():.2f}}")
        print(f"  Max: {{df[col].max():.2f}}")
        print(f"  Min: {{df[col].min():.2f}}")
    
    row_count = len(df)
else:
    print("No data available for analysis")
    row_count = 0

result = {{
    "description": "{step.description}",
    "status": "completed",
    "row_count": row_count
}}
print(f"\\nResult: {{result}}")
'''
        
        # Execute code with injected DataFrames - with retry loop
        from src.mcp.tools.code_interpreter import CodeInterpreter
        from src.mcp.server import ToolResult
        
        max_retries = 3
        last_error = None
        current_code = code
        
        for attempt in range(1, max_retries + 1):
            logger.info(f"[CodeAgent] Analysis attempt {attempt}/{max_retries}")
            
            if dataframes:
                interpreter = CodeInterpreter()
                mcp_result = await interpreter.execute_with_multiple_dataframes(
                    code=current_code,
                    dataframes=dataframes
                )
                # Wrap in ToolResult format
                mcp_result = ToolResult(
                    success=mcp_result.success,
                    output={"stdout": mcp_result.output, "images": mcp_result.images},
                    error=mcp_result.error,
                    execution_time_ms=mcp_result.execution_time_ms,
                )
            else:
                # No data, execute directly
                mcp_result = await mcp.call_tool("execute_python", {"code": current_code})
            
            if mcp_result.success:
                logger.info(f"[CodeAgent] ✅ Analysis executed ({mcp_result.execution_time_ms}ms)")
                
                return (
                    GeneratedCode(
                        step_id=step_id,
                        hypothesis_id=hypo_id,
                        language="python",
                        code=current_code,
                        description=step.description,
                        dependencies=deps,
                    ),
                    ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        output_type=OutputType.JSON,
                        output={
                            "stdout": mcp_result.output.get("stdout", ""),
                        },
                        execution_time_ms=mcp_result.execution_time_ms,
                    ),
                )
            else:
                last_error = mcp_result.error
                logger.warning(f"[CodeAgent] ⚠️ Analysis failed (attempt {attempt}): {last_error}")
                
                # Ask LLM to fix the code
                if attempt < max_retries:
                    fix_prompt = f"""Code Python sau bị lỗi:

```python
{current_code}
```

Lỗi: {last_error}

Sửa code để tránh lỗi trên. Trả về CHỈ CODE Python mới."""
                    
                    from langchain_core.messages import HumanMessage
                    response = await self.invoke_llm([HumanMessage(content=fix_prompt)])
                    current_code = response.content.strip()
                    # Remove markdown fences
                    if current_code.startswith("```"):
                        current_code = current_code.split("\n", 1)[-1]
                    if current_code.endswith("```"):
                        current_code = current_code.rsplit("```", 1)[0]
                    current_code = current_code.strip()
                    logger.info(f"[CodeAgent] LLM generated fixed code ({len(current_code)} chars)")
        
        # All retries exhausted
        logger.error(f"[CodeAgent] ❌ Analysis failed after {max_retries} attempts: {last_error}")
        
        return (
            GeneratedCode(
                step_id=step_id,
                hypothesis_id=hypo_id,
                language="python",
                code=current_code,
                description=step.description,
                dependencies=deps,
            ),
            ExecutionResult(
                status=ExecutionStatus.ERROR,
                output_type=OutputType.ERROR,
                error_message=f"Failed after {max_retries} attempts: {last_error}",
            ),
        )
    
    async def _generate_code_with_llm(
        self,
        step: AnalysisStep,
        step_id: str,
        hypo_id: str,
        input_data: CodeAgentInput,
    ) -> tuple[GeneratedCode, ExecutionResult]:
        """Fallback: Generate code with LLM."""
        
        prompt = f"""Generate code for this step:
        
Step: {step.description}
Action Type: {step.action_type}
Requirements: {step.requirements if hasattr(step, 'requirements') else step.details}

Output the code in a markdown code block.
"""
        
        response = await self.invoke_llm([HumanMessage(content=prompt)])
        
        # Parse code from response
        code_match = re.search(r'```(\w+)\n(.*?)```', response.content, re.DOTALL)
        if code_match:
            language = code_match.group(1).lower()
            code = code_match.group(2).strip()
        else:
            language = "python"
            code = response.content.strip()
        
        return (
            GeneratedCode(
                step_id=step_id,
                hypothesis_id=hypo_id,
                language=language,
                code=code,
                description=step.description,
            ),
            ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output_type=OutputType.TEXT,
                output={"code_generated": True},
            ),
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
