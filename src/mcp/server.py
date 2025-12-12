"""
MCP Server integration.

Provides tool execution capabilities for the Code Agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    
    success: bool
    output: Any
    error: str | None = None
    execution_time_ms: int = 0


class MCPServer:
    """
    MCP Server integration for tool execution.
    
    Exposes tools for:
    - SQL execution
    - Python code execution
    - Schema validation
    - Visualization generation
    """
    
    def __init__(self):
        """Initialize MCP server."""
        self._tools: dict[str, Any] = {}
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register available tools."""
        self._tools = {
            "execute_sql": self.execute_sql,
            "execute_python": self.execute_python,
            "validate_sql_syntax": self.validate_sql_syntax,
            "get_table_schema": self.get_table_schema,
            "create_visualization": self.create_visualization,
            "text_to_sql": self.text_to_sql,
        }
    
    async def execute_sql(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> ToolResult:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query to execute
            parameters: Query parameters
            database: Target database
            
        Returns:
            ToolResult with query results
        """
        logger.info(f"Executing SQL: {query[:100]}...")
        
        # TODO: Implement actual SQL execution
        # Should connect to PostgreSQL and execute query
        
        return ToolResult(
            success=True,
            output={"rows": [], "columns": []},
            execution_time_ms=50,
        )
    
    async def execute_python(
        self,
        code: str,
        data: dict[str, Any] | None = None,
        timeout_seconds: int = 60,
    ) -> ToolResult:
        """
        Execute Python code using OpenAI Code Interpreter.
        
        Args:
            code: Python code to execute
            data: Optional data to pass (e.g., DataFrame from SQL result)
            timeout_seconds: Execution timeout
            
        Returns:
            ToolResult with execution output, images, files
        """
        from src.mcp.tools.code_interpreter import CodeInterpreter
        
        logger.info(f"[MCP] Executing Python code ({len(code)} chars)")
        
        interpreter = CodeInterpreter()
        result = await interpreter.execute(
            code=code, 
            data=data, 
            timeout_seconds=timeout_seconds
        )
        
        return ToolResult(
            success=result.success,
            output={
                "stdout": result.output,
                "images": result.images,
                "files": result.files,
            },
            error=result.error,
            execution_time_ms=result.execution_time_ms,
        )
    
    async def validate_sql_syntax(
        self,
        query: str,
    ) -> ToolResult:
        """
        Validate SQL syntax without executing.
        
        Args:
            query: SQL query to validate
            
        Returns:
            ToolResult with validation result
        """
        logger.info("Validating SQL syntax")
        
        # TODO: Implement via EXPLAIN (without ANALYZE)
        
        return ToolResult(
            success=True,
            output={"valid": True, "errors": []},
        )
    
    async def get_table_schema(
        self,
        table_name: str,
        include_samples: bool = False,
    ) -> ToolResult:
        """
        Get schema information for a table.
        
        Args:
            table_name: Table to describe
            include_samples: Include sample data
            
        Returns:
            ToolResult with schema info
        """
        logger.info(f"Getting schema for: {table_name}")
        
        # TODO: Implement via information_schema query
        
        return ToolResult(
            success=True,
            output={"columns": [], "row_count": 0},
        )
    
    async def create_visualization(
        self,
        data: dict[str, Any],
        chart_type: str,
        options: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Create a visualization.
        
        Args:
            data: Data to visualize
            chart_type: Type of chart (bar, line, scatter, etc.)
            options: Chart options
            
        Returns:
            ToolResult with image path
        """
        logger.info(f"Creating {chart_type} visualization")
        
        # TODO: Implement via matplotlib/seaborn in sandbox
        
        return ToolResult(
            success=True,
            output={"image_path": None, "image_base64": None},
        )
    
    async def text_to_sql(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> ToolResult:
        """
        Generate Trino SQL from natural language.
        
        Args:
            prompt: Natural language query (any text)
            session_id: Optional - reuse existing session's SubGraph
            
        Returns:
            ToolResult with SQL string
        """
        from src.mcp.tools.text_to_sql import TextToSQL
        
        tool = TextToSQL()
        result = await tool.generate(prompt=prompt, session_id=session_id)
        
        return ToolResult(
            success=result.success,
            output={
                "sql": result.sql,
                "session_id": result.session_id,
                "tables_used": result.tables_used,
            },
            error=result.error,
        )
    
    def list_tools(self) -> list[dict[str, Any]]:
        """
        List available tools (MCP tools/list).
        
        Returns:
            List of tool descriptions
        """
        return [
            {
                "name": "execute_sql",
                "description": "Execute a SQL query against the database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "parameters": {"type": "object"},
                        "database": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "execute_python",
                "description": "Execute Python code in a sandbox environment",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "timeout_seconds": {"type": "integer", "default": 30},
                    },
                    "required": ["code"],
                },
            },
            {
                "name": "validate_sql_syntax",
                "description": "Validate SQL syntax without executing the query",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_table_schema",
                "description": "Get schema information for a database table",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string"},
                        "include_samples": {"type": "boolean", "default": False},
                    },
                    "required": ["table_name"],
                },
            },
            {
                "name": "create_visualization",
                "description": "Create a data visualization",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "chart_type": {"type": "string"},
                        "options": {"type": "object"},
                    },
                    "required": ["data", "chart_type"],
                },
            },
            {
                "name": "text_to_sql",
                "description": "Generate Trino SQL from natural language. Uses Graph RAG to find relevant schema.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Natural language query"},
                        "session_id": {"type": "string", "description": "Optional: reuse existing session"},
                    },
                    "required": ["prompt"],
                },
            },
        ]
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Call a tool by name (MCP tools/call).
        
        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments
            
        Returns:
            ToolResult
        """
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {tool_name}",
            )
        
        tool_func = self._tools[tool_name]
        try:
            result = await tool_func(**arguments)
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )
