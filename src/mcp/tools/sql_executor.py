"""
SQL Executor MCP Tool.

Provides SQL execution capabilities with safety checks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import asyncpg

from config import config

logger = logging.getLogger(__name__)


# Dangerous SQL patterns
DANGEROUS_PATTERNS = [
    r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b",
    r"\bTRUNCATE\b",
    r"\bDELETE\s+FROM\b(?!\s+.*\bWHERE\b)",  # DELETE without WHERE
    r"\bALTER\s+(TABLE|DATABASE)\b",
    r"\bCREATE\s+(TABLE|DATABASE|SCHEMA)\b",
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
]


@dataclass
class SQLResult:
    """Result from SQL execution."""
    
    success: bool
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    execution_time_ms: int
    error: str | None = None


class SQLExecutor:
    """
    Safe SQL execution tool.
    
    Features:
    - Connection pooling
    - Query validation
    - Timeout handling
    - Result size limits
    """
    
    def __init__(
        self,
        max_rows: int = 10000,
        timeout_seconds: int = 30,
    ):
        """
        Initialize SQL executor.
        
        Args:
            max_rows: Maximum rows to return
            timeout_seconds: Query timeout
        """
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        self._pool: asyncpg.Pool | None = None
    
    async def get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=config.postgres.host,
                port=config.postgres.port,
                user=config.postgres.user,
                password=config.postgres.password,
                database=config.postgres.db,
                min_size=1,
                max_size=10,
            )
        return self._pool
    
    def validate_query(self, query: str) -> tuple[bool, str | None]:
        """
        Validate SQL query for safety.
        
        Args:
            query: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        query_upper = query.upper()
        
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, query_upper, re.IGNORECASE):
                return False, f"Query contains dangerous operation: {pattern}"
        
        # Only allow SELECT queries
        if not query_upper.strip().startswith("SELECT"):
            return False, "Only SELECT queries are allowed"
        
        return True, None
    
    async def execute(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> SQLResult:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query
            parameters: Query parameters
            
        Returns:
            SQLResult with data or error
        """
        import time
        start_time = time.monotonic()
        
        # Validate query
        is_valid, error = self.validate_query(query)
        if not is_valid:
            return SQLResult(
                success=False,
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=0,
                error=error,
            )
        
        # Add LIMIT if not present
        query_limited = self._add_limit(query)
        
        try:
            pool = await self.get_pool()
            
            async with pool.acquire() as conn:
                # Set statement timeout
                await conn.execute(
                    f"SET statement_timeout = '{self.timeout_seconds}s'"
                )
                
                # Execute query
                rows = await conn.fetch(query_limited, *(parameters or {}).values())
                
                if rows:
                    columns = list(rows[0].keys())
                    data = [list(row.values()) for row in rows]
                else:
                    columns = []
                    data = []
                
                execution_time = int((time.monotonic() - start_time) * 1000)
                
                return SQLResult(
                    success=True,
                    columns=columns,
                    rows=data,
                    row_count=len(data),
                    execution_time_ms=execution_time,
                )
                
        except asyncpg.exceptions.QueryCanceledError:
            return SQLResult(
                success=False,
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=self.timeout_seconds * 1000,
                error="Query timeout exceeded",
            )
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return SQLResult(
                success=False,
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=int((time.monotonic() - start_time) * 1000),
                error=str(e),
            )
    
    def _add_limit(self, query: str) -> str:
        """Add LIMIT clause if not present."""
        query_upper = query.upper()
        if "LIMIT" not in query_upper:
            return f"{query.rstrip().rstrip(';')} LIMIT {self.max_rows}"
        return query
    
    async def explain(
        self,
        query: str,
    ) -> dict[str, Any]:
        """
        Get query execution plan without running.
        
        Args:
            query: Query to explain
            
        Returns:
            Execution plan details
        """
        try:
            pool = await self.get_pool()
            
            async with pool.acquire() as conn:
                explain_query = f"EXPLAIN (FORMAT JSON) {query}"
                result = await conn.fetchval(explain_query)
                
                return {
                    "success": True,
                    "plan": result,
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
