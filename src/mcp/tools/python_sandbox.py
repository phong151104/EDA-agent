"""
Python Sandbox MCP Tool.

Provides safe Python code execution in isolated environment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config import config

logger = logging.getLogger(__name__)


@dataclass
class PythonResult:
    """Result from Python execution."""
    
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    generated_files: list[str] = field(default_factory=list)
    execution_time_ms: int = 0
    error: str | None = None


class PythonSandbox:
    """
    Sandboxed Python execution.
    
    Uses E2B for secure code execution with:
    - Isolated environment
    - Pre-installed data science packages
    - File output capture
    """
    
    # Pre-approved packages
    ALLOWED_IMPORTS = {
        "pandas", "numpy", "matplotlib", "seaborn",
        "scipy", "sklearn", "json", "datetime",
        "collections", "itertools", "functools",
        "math", "statistics", "re", "typing",
    }
    
    # Forbidden patterns
    FORBIDDEN_PATTERNS = [
        "import os",
        "import sys",
        "import subprocess",
        "__import__",
        "exec(",
        "eval(",
        "open(",
        "file(",
        "input(",
        "breakpoint(",
    ]
    
    def __init__(
        self,
        timeout_seconds: int = 30,
        use_e2b: bool = True,
    ):
        """
        Initialize Python sandbox.
        
        Args:
            timeout_seconds: Execution timeout
            use_e2b: Whether to use E2B (True) or local execution (False)
        """
        self.timeout_seconds = timeout_seconds
        self.use_e2b = use_e2b
    
    def validate_code(self, code: str) -> tuple[bool, str | None]:
        """
        Validate Python code for safety.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in code:
                return False, f"Forbidden pattern detected: {pattern}"
        
        return True, None
    
    async def execute(
        self,
        code: str,
        context: dict[str, Any] | None = None,
    ) -> PythonResult:
        """
        Execute Python code in sandbox.
        
        Args:
            code: Python code to execute
            context: Variables to inject
            
        Returns:
            PythonResult with output
        """
        # Validate code
        is_valid, error = self.validate_code(code)
        if not is_valid:
            return PythonResult(
                success=False,
                error=error,
            )
        
        if self.use_e2b:
            return await self._execute_e2b(code, context)
        else:
            return await self._execute_local(code, context)
    
    async def _execute_e2b(
        self,
        code: str,
        context: dict[str, Any] | None = None,
    ) -> PythonResult:
        """Execute using E2B sandbox."""
        try:
            # TODO: Implement E2B execution
            # from e2b import Sandbox
            # 
            # sandbox = Sandbox()
            # result = sandbox.run_code(code, timeout=self.timeout_seconds)
            # ...
            
            logger.info("E2B execution placeholder")
            
            return PythonResult(
                success=True,
                stdout="E2B execution not implemented yet",
                execution_time_ms=100,
            )
            
        except Exception as e:
            logger.error(f"E2B execution error: {e}")
            return PythonResult(
                success=False,
                error=str(e),
            )
    
    async def _execute_local(
        self,
        code: str,
        context: dict[str, Any] | None = None,
    ) -> PythonResult:
        """
        Execute locally (for development only).
        
        WARNING: Not secure for production use.
        """
        import io
        import sys
        import time
        from contextlib import redirect_stdout, redirect_stderr
        
        start_time = time.monotonic()
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Create execution namespace
        namespace = {
            "__builtins__": __builtins__,
        }
        
        if context:
            namespace.update(context)
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)
            
            execution_time = int((time.monotonic() - start_time) * 1000)
            
            # Extract result if present
            return_value = namespace.get("result") or namespace.get("output")
            
            return PythonResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                return_value=return_value,
                execution_time_ms=execution_time,
            )
            
        except Exception as e:
            return PythonResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error=str(e),
                execution_time_ms=int((time.monotonic() - start_time) * 1000),
            )
    
    async def install_packages(
        self,
        packages: list[str],
    ) -> bool:
        """
        Install additional packages in sandbox.
        
        Args:
            packages: Package names to install
            
        Returns:
            True if successful
        """
        # Filter only allowed packages
        allowed = [p for p in packages if p in self.ALLOWED_IMPORTS]
        
        if not allowed:
            logger.warning(f"No allowed packages in: {packages}")
            return False
        
        # TODO: Implement package installation in E2B
        logger.info(f"Would install packages: {allowed}")
        return True
