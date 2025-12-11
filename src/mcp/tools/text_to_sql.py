"""
Text-to-SQL MCP Tool.

Generates Trino SQL from natural language using SubGraph context.
Optimized for token efficiency and reusability across agents.

Usage:
    from src.mcp.tools import TextToSQL
    
    text_to_sql = TextToSQL()
    result = await text_to_sql.generate("Doanh thu theo vendor tháng này")
    # result.sql = "SELECT ..."
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from openai import OpenAI

from config import config
from src.context_fusion import SubGraph, build_session, get_cached_session

if TYPE_CHECKING:
    from src.context_fusion import EDASession

logger = logging.getLogger(__name__)

# Metadata directory path
METADATA_DIR = Path(__file__).parent.parent.parent.parent / "metadata" / "domains"


# =============================================================================
# Result Model
# =============================================================================

@dataclass
class TextToSQLResult:
    """Result from text-to-sql generation."""
    
    success: bool
    sql: str
    session_id: str = ""
    tables_used: list[str] | None = None
    error: str | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for MCP response."""
        return {
            "success": self.success,
            "sql": self.sql,
            "session_id": self.session_id,
            "tables_used": self.tables_used,
            "error": self.error,
        }


# =============================================================================
# Main Tool Class
# =============================================================================

class TextToSQL:
    """
    Generate Trino SQL from natural language.
    
    Features:
    - Auto-fetches SubGraph via context_fusion (Graph RAG)
    - Smart few-shot: loads relevant sample queries from YAML metadata
    - Token-optimized prompts (compact schema + 1-2 best examples)
    - Supports session reuse to avoid redundant Neo4j queries
    
    Example:
        >>> tool = TextToSQL()
        >>> result = await tool.generate("Tổng doanh thu theo vendor")
        >>> print(result.sql)
        SELECT v.vendor_name, SUM(b.revenue) AS total_revenue
        FROM fact_booking b
        JOIN dim_vendor v ON b.vendor_id = v.vendor_id
        GROUP BY v.vendor_name;
    """
    
    # System prompt with strict SubGraph-only rules
    SYSTEM_PROMPT = """You are a Trino SQL expert. Generate SQL using ONLY the provided schema.

=== ⚠️ MANDATORY - DO NOT VIOLATE ===
• DO NOT invent table names - use ONLY tables from "## Available Tables"
• DO NOT invent column names - use ONLY columns from "## Columns"
• DO NOT invent JOIN conditions - use ONLY joins from "## Joins"
• If a join doesn't exist in "## Joins", those tables CANNOT be joined

=== DOMAIN KNOWLEDGE ===
• "khách hàng" identifier = orders.bank_identity_hash
• "giao dịch thành công" = orders with status='payment'
• To get vendor info: orders → order_film → vendor (via order_film.vendor_id)
• order_film has cinema info in columns (cinema_id, cinema_name_vi) - no need to join cinema table
• orders.status: 'initial', 'payment', 'expired'

=== RULES ===
1. JOINS: Use ONLY from "## Joins". Check the ON clause carefully.
2. DATA TYPES: Don't compare VARCHAR with BIGINT (e.g., vendor_cinema_id is VARCHAR)
3. TABLE NAMES: lakehouse.lh_vnfilm_v2.<table_name>
4. Use DISTINCT when counting unique customers
5. VENDOR NAME: use name_vi (NOT name)

=== TRINO DATE ===
- Q1 2025: created_date >= DATE '2025-01-01' AND created_date < DATE '2025-04-01'

OUTPUT: Return ONLY valid JSON: {"sql": "SELECT ...;"}"""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0,
        max_samples: int = 2,
    ):
        """
        Initialize TextToSQL tool.
        
        Args:
            model: OpenAI model (defaults to config.openai.model)
            temperature: LLM temperature (0 for deterministic)
            max_samples: Maximum number of sample queries to include (default: 2)
        """
        self.client = OpenAI(api_key=config.openai.api_key)
        self.model = model or config.openai.model
        self.temperature = temperature
        self.max_samples = max_samples
    
    # =========================================================================
    # Main API
    # =========================================================================
    
    async def generate(
        self,
        prompt: str,
        session_id: str | None = None,
        sub_graph: SubGraph | None = None,
    ) -> TextToSQLResult:
        """
        Generate Trino SQL from natural language prompt.
        
        Args:
            prompt: Natural language query (any text from agents)
            session_id: Reuse existing session's SubGraph (skip Neo4j query)
            sub_graph: Provide SubGraph directly (highest priority)
            
        Returns:
            TextToSQLResult with clean SQL string
        """
        try:
            # Step 1: Get SubGraph
            session_id_out, sub_graph = await self._get_subgraph(
                prompt=prompt,
                session_id=session_id,
                sub_graph=sub_graph,
            )
            
            if not sub_graph or not sub_graph.tables:
                return TextToSQLResult(
                    success=False,
                    sql="",
                    error="No relevant tables found for this query",
                )
            
            # Step 2: Generate SQL (with samples)
            sql = await self._generate_sql(prompt, sub_graph)
            
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
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    async def _get_subgraph(
        self,
        prompt: str,
        session_id: str | None,
        sub_graph: SubGraph | None,
    ) -> tuple[str, SubGraph | None]:
        """Get SubGraph from various sources."""
        
        # Priority 1: Direct SubGraph
        if sub_graph:
            return "", sub_graph
        
        # Priority 2: From cached session
        if session_id:
            session = get_cached_session(session_id)
            if session and session.sub_graph:
                logger.debug(f"Reusing session: {session_id}")
                return session_id, session.sub_graph
            logger.warning(f"Session not found: {session_id}")
        
        # Priority 3: Build new session (queries Neo4j, auto-caches)
        logger.info(f"Building new session for: {prompt[:50]}...")
        session = await build_session(prompt)
        return session.session_id, session.sub_graph
    
    def _load_table_metadata(self, table_name: str, domain: str = "vnfilm_ticketing") -> dict | None:
        """Load metadata YAML for a table."""
        yaml_path = METADATA_DIR / domain / "tables" / f"{table_name}.yml"
        
        if not yaml_path.exists():
            logger.debug(f"YAML not found: {yaml_path}")
            return None
        
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load {yaml_path}: {e}")
            return None
    
    def _score_sample(self, prompt: str, sample_question: str, sample_sql: str) -> float:
        """Score relevance of a sample to the prompt using keyword matching."""
        prompt_lower = prompt.lower()
        question_lower = sample_question.lower() if sample_question else ""
        
        # Extract keywords from prompt
        prompt_words = set(re.findall(r'\w+', prompt_lower))
        question_words = set(re.findall(r'\w+', question_lower))
        
        # Score based on word overlap
        overlap = len(prompt_words & question_words)
        return overlap
    
    def _load_relevant_samples(
        self, 
        table_names: list[str], 
        prompt: str,
        domain: str = "vnfilm_ticketing",
    ) -> str:
        """
        Load ALL sample queries from ALL relevant tables.
        
        Returns formatted few-shot examples for the prompt.
        """
        all_samples = []
        
        logger.info(f"Loading samples for tables: {table_names}")
        logger.info(f"METADATA_DIR: {METADATA_DIR}")
        
        for table_name in table_names:
            metadata = self._load_table_metadata(table_name, domain)
            if not metadata:
                logger.debug(f"No metadata for table: {table_name}")
                continue
            
            # Get sample_queries
            sample_queries = metadata.get("sample_queries", [])
            logger.info(f"Table {table_name}: found {len(sample_queries)} sample_queries")
            
            for sq in sample_queries:
                if isinstance(sq, dict):
                    desc = sq.get("description", "")
                    sql = sq.get("sql", "")
                    if sql:
                        all_samples.append({
                            "question": desc,
                            "sql": sql.strip(),
                            "table": table_name,
                        })
        
        logger.info(f"Total samples loaded: {len(all_samples)}")
        
        if not all_samples:
            return ""
        
        # Format ALL samples as few-shot examples
        lines = ["Examples:"]
        for i, sample in enumerate(all_samples, 1):
            lines.append(f"Q{i}: {sample['question']}")
            # Keep SQL readable but compact
            sql = sample["sql"].replace("\n", " ").strip()
            sql = re.sub(r'\s+', ' ', sql)
            lines.append(f"A{i}: {sql}")
        
        return "\n".join(lines)
    
    async def _generate_sql(self, prompt: str, sub_graph: SubGraph) -> str:
        """Generate SQL using LLM with detailed schema + few-shot samples."""
        
        # Build DETAILED schema context (includes all columns per table)
        # This is critical - LLM needs to see which columns belong to which table
        schema_context = sub_graph.to_prompt_context(compact=False)
        
        # Load relevant samples from YAML (token-efficient)
        table_names = sub_graph.get_table_names()
        samples_context = self._load_relevant_samples(table_names, prompt)
        
        # Build user prompt
        if samples_context:
            user_prompt = f"""Schema:
{schema_context}

{samples_context}

Q: {prompt}
SQL:"""
        else:
            user_prompt = f"""Schema:
{schema_context}

Q: {prompt}
SQL:"""

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
        
        content = response.choices[0].message.content
        return self._parse_sql(content)
    
    def _parse_sql(self, content: str) -> str:
        """Parse and validate SQL from LLM response."""
        try:
            result = json.loads(content)
            sql = result.get("sql", "").strip()
        except json.JSONDecodeError:
            # Fallback: extract SQL from text
            sql_match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL)
            if sql_match:
                sql = sql_match.group(1).strip()
            else:
                sql = content.strip()
        
        # Basic validation
        sql_upper = sql.upper()
        if not sql_upper.startswith("SELECT"):
            logger.warning(f"Invalid SQL (not SELECT): {sql[:50]}")
            return ""
        
        # Ensure semicolon
        if not sql.endswith(";"):
            sql += ";"
        
        return sql


# =============================================================================
# Convenience Functions
# =============================================================================

async def generate_sql(
    prompt: str,
    session_id: str | None = None,
) -> TextToSQLResult:
    """
    Quick function to generate SQL.
    
    Usage:
        from src.mcp.tools.text_to_sql import generate_sql
        
        result = await generate_sql("Doanh thu theo vendor")
        if result.success:
            print(result.sql)
    """
    tool = TextToSQL()
    return await tool.generate(prompt, session_id=session_id)

