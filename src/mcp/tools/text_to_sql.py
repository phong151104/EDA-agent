"""
Text-to-SQL MCP Tool.

Generates Trino SQL from natural language using SubGraph context.

Usage:
    from src.mcp.tools import TextToSQL
    
    tool = TextToSQL()
    result = await tool.generate("Doanh thu theo vendor tháng này")
    print(result.sql)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import OpenAI

from config import config
from src.context_fusion import SubGraph, build_session, get_cached_session

logger = logging.getLogger(__name__)

METADATA_DIR = Path(__file__).parent.parent.parent.parent / "metadata" / "domains"


@dataclass
class TextToSQLResult:
    """Result from text-to-sql generation."""
    
    success: bool
    sql: str
    session_id: str = ""
    tables_used: list[str] | None = None
    error: str | None = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "sql": self.sql,
            "session_id": self.session_id,
            "tables_used": self.tables_used,
            "error": self.error,
        }


class TextToSQL:
    """Generate Trino SQL from natural language."""
    
    SYSTEM_PROMPT = """Bạn là chuyên gia Trino SQL. Tạo SQL CHỈ dùng schema được cung cấp.

=== LUẬT BẮT BUỘC ===
1. CHỈ dùng cột có trong "## Columns" - không được bịa cột
2. CHỈ dùng JOIN từ "## Joins" - không được tự nghĩ ra cách join
3. Nếu cần thông tin nhưng không có cột phù hợp → dùng cột thay thế gần nhất

=== DOMAIN KNOWLEDGE ===
• "khách hàng" = orders.bank_identity_hash  
• "giao dịch thành công" = status='payment'
• orders.status: 'initial', 'payment', 'expired'
• Tên vendor: vendor.name_vi
• Tên rạp: cinema.name_vi hoặc order_film.cinema_name_vi (nếu có)

=== TRINO SYNTAX ===
• Table: lakehouse.lh_vnfilm_v2.<table_name>
• Q1 2025: created_date >= DATE '2025-01-01' AND created_date < DATE '2025-04-01'
• DISTINCT khi đếm unique

=== OUTPUT ===
JSON: {"sql": "SELECT ...;"}"""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0,
    ):
        self.client = OpenAI(api_key=config.openai.api_key)
        self.model = model or config.openai.model
        self.temperature = temperature
    
    async def generate(
        self,
        prompt: str,
        session_id: str | None = None,
        sub_graph: SubGraph | None = None,
    ) -> TextToSQLResult:
        """
        Generate Trino SQL from natural language.
        
        Args:
            prompt: Natural language query
            session_id: Reuse existing session's SubGraph
            sub_graph: Provide SubGraph directly
            
        Returns:
            TextToSQLResult with SQL string
        """
        try:
            # Get SubGraph
            session_id_out, sub_graph = await self._get_subgraph(prompt, session_id, sub_graph)
            
            if not sub_graph or not sub_graph.tables:
                return TextToSQLResult(
                    success=False,
                    sql="",
                    error="No relevant tables found",
                )
            
            # Generate SQL
            sql = self._generate_sql(prompt, sub_graph)
            
            if not sql:
                return TextToSQLResult(
                    success=False,
                    sql="",
                    session_id=session_id_out,
                    error="Failed to generate valid SQL",
                )
            
            return TextToSQLResult(
                success=True,
                sql=sql,
                session_id=session_id_out,
                tables_used=sub_graph.get_table_names(),
            )
            
        except Exception as e:
            logger.exception("Text-to-SQL error")
            return TextToSQLResult(success=False, sql="", error=str(e))
    
    async def _get_subgraph(
        self,
        prompt: str,
        session_id: str | None,
        sub_graph: SubGraph | None,
    ) -> tuple[str, SubGraph | None]:
        """Get SubGraph from cache or build new."""
        if sub_graph:
            return "", sub_graph
        
        if session_id:
            session = get_cached_session(session_id)
            if session and session.sub_graph:
                return session_id, session.sub_graph
        
        session = await build_session(prompt)
        return session.session_id, session.sub_graph
    
    def _generate_sql(self, prompt: str, sub_graph: SubGraph) -> str:
        """Generate SQL using LLM."""
        schema_context = sub_graph.to_prompt_context(compact=False)
        samples_context = self._load_samples(sub_graph.get_table_names())
        
        user_prompt = f"Schema:\n{schema_context}"
        if samples_context:
            user_prompt += f"\n\n{samples_context}"
        user_prompt += f"\n\nQ: {prompt}\nSQL:"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_tokens=1000,
        )
        
        return self._parse_sql(response.choices[0].message.content)
    
    def _load_samples(self, table_names: list[str], domain: str = "vnfilm_ticketing") -> str:
        """Load sample queries from YAML metadata."""
        samples = []
        
        for table_name in table_names:
            yaml_path = METADATA_DIR / domain / "tables" / f"{table_name}.yml"
            if not yaml_path.exists():
                continue
            
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    metadata = yaml.safe_load(f)
                
                for sq in metadata.get("sample_queries", []):
                    if isinstance(sq, dict) and sq.get("sql"):
                        samples.append(f"Q: {sq.get('description', '')}\nA: {sq['sql'].strip()}")
            except Exception:
                pass
        
        return "Examples:\n" + "\n".join(samples) if samples else ""
    
    def _parse_sql(self, content: str) -> str:
        """Parse and validate SQL from LLM response."""
        logger.info(f"[LLM_RESPONSE] {content[:500]}...")
        
        try:
            result = json.loads(content)
            sql = result.get("sql", "").strip()
            
            # Handle error response from LLM
            if not sql and result.get("error"):
                logger.warning(f"[LLM] Error from LLM: {result.get('error')}")
                return ""
            
            # Check if LLM indicated missing data
            if not sql and result.get("message"):
                logger.warning(f"[LLM] Message: {result.get('message')}")
                return ""
                
        except json.JSONDecodeError:
            logger.warning(f"[LLM] Failed to parse JSON, trying regex extraction")
            sql_match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL)
            sql = sql_match.group(1).strip() if sql_match else content.strip()
        
        if not sql:
            logger.warning("[LLM] Empty SQL returned from LLM")
            return ""
        
        if not self._validate_sql(sql):
            logger.warning(f"[LLM] SQL validation failed: {sql[:200]}")
            return ""
        
        return sql if sql.endswith(";") else sql + ";"
    
    def _validate_sql(self, sql: str) -> bool:
        """Validate SQL for security."""
        if not sql:
            return False
        
        sql_upper = sql.upper().strip()
        
        # Must be SELECT or WITH (CTE) only
        if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
            logger.warning(f"[SQL] SQL must start with SELECT or WITH, got: {sql[:30]}")
            return False
        
        # Forbidden operations
        forbidden = ["DELETE", "DROP", "TRUNCATE", "UPDATE", "INSERT", "ALTER", "CREATE", "GRANT", "EXEC"]
        for word in forbidden:
            if re.search(rf'\b{word}\b', sql_upper):
                logger.warning(f"[SQL] Forbidden keyword found: {word}")
                return False
        
        # Check balanced parentheses and quotes
        if sql.count('(') != sql.count(')'):
            logger.warning("[SQL] Unbalanced parentheses")
            return False
        if sql.count("'") % 2 != 0:
            logger.warning("[SQL] Unbalanced quotes")
            return False
        
        return True


async def generate_sql(prompt: str, session_id: str | None = None) -> TextToSQLResult:
    """Quick function to generate SQL."""
    return await TextToSQL().generate(prompt, session_id=session_id)
