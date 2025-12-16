"""
Code Interpreter MCP Tool.

Executes Python code using OpenAI Code Interpreter (Assistants API).

Usage:
    from src.mcp.tools import CodeInterpreter
    
    tool = CodeInterpreter()
    result = await tool.execute(code="import pandas as pd\nprint(df.head())")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)


@dataclass
class CodeExecutionResult:
    """Result from code execution."""
    
    success: bool
    output: str = ""
    error: str | None = None
    images: list[str] = field(default_factory=list)  # Base64 encoded images
    files: list[dict[str, str]] = field(default_factory=list)  # {filename, content}
    execution_time_ms: int = 0
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "images": self.images,
            "files": self.files,
            "execution_time_ms": self.execution_time_ms,
        }


class CodeInterpreter:
    """Execute Python code using OpenAI Code Interpreter."""
    
    ASSISTANT_INSTRUCTIONS = """Bạn là Code Executor.
Chạy code Python được cung cấp và trả về kết quả.
Nếu code tạo biểu đồ, hãy lưu file và trả về.
Nếu có lỗi, mô tả chi tiết lỗi gì và cách fix."""
    
    def __init__(self, model: str | None = None):
        self.client = OpenAI(api_key=config.openai.api_key)
        self.model = model or "gpt-4o"  # Code interpreter works best with gpt-4o
        self._assistant_id: str | None = None
    
    def _get_or_create_assistant(self) -> str:
        """Get or create the code interpreter assistant."""
        if self._assistant_id:
            return self._assistant_id
        
        # Check if assistant already exists
        assistants = self.client.beta.assistants.list(limit=20)
        for assistant in assistants.data:
            if assistant.name == "EDA Code Executor":
                self._assistant_id = assistant.id
                logger.info(f"[CodeInterpreter] Reusing assistant: {assistant.id}")
                return self._assistant_id
        
        # Create new assistant
        assistant = self.client.beta.assistants.create(
            name="EDA Code Executor",
            instructions=self.ASSISTANT_INSTRUCTIONS,
            tools=[{"type": "code_interpreter"}],
            model=self.model,
        )
        self._assistant_id = assistant.id
        logger.info(f"[CodeInterpreter] Created assistant: {assistant.id}")
        return self._assistant_id
    
    async def execute(
        self,
        code: str,
        data: dict[str, Any] | None = None,
        timeout_seconds: int = 60,
    ) -> CodeExecutionResult:
        """
        Execute Python code.
        
        Args:
            code: Python code to execute
            data: Optional data to pass to code (as JSON)
            timeout_seconds: Maximum execution time
            
        Returns:
            CodeExecutionResult with output, errors, and generated files
        """
        start_time = time.time()
        
        try:
            assistant_id = self._get_or_create_assistant()
            
            # Create thread
            thread = self.client.beta.threads.create()
            
            # Build message content
            content = f"Execute this Python code:\n\n```python\n{code}\n```"
            if data:
                import json
                content += f"\n\nWith this data:\n```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```"
            
            # Add message
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=content,
            )
            
            # Run and wait for completion
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant_id,
                timeout=timeout_seconds,
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            if run.status == "completed":
                # Get messages
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread.id,
                    order="desc",
                    limit=1,
                )
                
                output_text = ""
                images = []
                files = []
                
                for message in messages.data:
                    if message.role == "assistant":
                        for content_block in message.content:
                            if content_block.type == "text":
                                output_text += content_block.text.value
                            elif content_block.type == "image_file":
                                # Get file content as base64
                                file_id = content_block.image_file.file_id
                                if not file_id:
                                    logger.debug("[CodeInterpreter] Skipping empty file_id")
                                    continue
                                try:
                                    file_data = self.client.files.content(file_id)
                                    import base64
                                    images.append(base64.b64encode(file_data.content).decode())
                                except Exception as e:
                                    logger.warning(f"Failed to get image {file_id}: {e}")
                
                logger.info(f"[CodeInterpreter] ✅ Execution completed in {execution_time}ms")
                
                return CodeExecutionResult(
                    success=True,
                    output=output_text,
                    images=images,
                    files=files,
                    execution_time_ms=execution_time,
                )
            
            elif run.status == "failed":
                error_msg = run.last_error.message if run.last_error else "Unknown error"
                logger.error(f"[CodeInterpreter] ❌ Run failed: {error_msg}")
                
                return CodeExecutionResult(
                    success=False,
                    error=error_msg,
                    execution_time_ms=execution_time,
                )
            
            else:
                logger.error(f"[CodeInterpreter] ❌ Unexpected run status: {run.status}")
                
                return CodeExecutionResult(
                    success=False,
                    error=f"Unexpected status: {run.status}",
                    execution_time_ms=execution_time,
                )
                
        except Exception as e:
            logger.exception("[CodeInterpreter] Execution error")
            return CodeExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
    
    async def execute_with_dataframe(
        self,
        code: str,
        df_json: str,
        df_name: str = "df",
    ) -> CodeExecutionResult:
        """
        Execute code with a pre-loaded DataFrame.
        
        Args:
            code: Python code (can reference df_name directly)
            df_json: DataFrame as JSON string
            df_name: Variable name for the DataFrame
            
        Returns:
            CodeExecutionResult
        """
        # Prepend DataFrame loading
        full_code = f"""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from previous SQL query
{df_name} = pd.read_json('''{df_json}''')

# User code
{code}
"""
        return await self.execute(full_code)
    
    async def execute_with_multiple_dataframes(
        self,
        code: str,
        dataframes: dict[str, list],  # {step_id: sql_data}
    ) -> CodeExecutionResult:
        """
        Execute code with multiple pre-loaded DataFrames.
        
        Args:
            code: Python code (can reference df_<step_id> variables)
            dataframes: Dict mapping step_id to SQL result data
            
        Returns:
            CodeExecutionResult
            
        Example:
            dataframes = {"s1": [...], "s2": [...]}
            # Code can use: df_s1, df_s2
        """
        import json as json_module
        
        # Build data loading code
        load_lines = [
            "import pandas as pd",
            "import matplotlib.pyplot as plt", 
            "import seaborn as sns",
            "import numpy as np",
            "",
            "# Load data from previous SQL queries",
        ]
        
        for step_id, data in dataframes.items():
            df_name = f"df_{step_id}"
            json_str = json_module.dumps(data, ensure_ascii=False, default=str)
            load_lines.append(f"{df_name} = pd.read_json('''{json_str}''')")
        
        # Also create convenient 'df' alias for the first or only DataFrame
        if len(dataframes) == 1:
            first_step = list(dataframes.keys())[0]
            load_lines.append(f"df = df_{first_step}  # Alias for convenience")
        elif len(dataframes) > 1:
            load_lines.append("")
            load_lines.append("# Available DataFrames: " + ", ".join(f"df_{k}" for k in dataframes.keys()))
        
        load_lines.append("")
        load_lines.append("# User code")
        
        full_code = "\n".join(load_lines) + "\n" + code
        
        return await self.execute(full_code)


async def execute_code(code: str, data: dict[str, Any] | None = None) -> CodeExecutionResult:
    """Quick function to execute code."""
    return await CodeInterpreter().execute(code, data)

