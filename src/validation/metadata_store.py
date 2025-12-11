"""
MetadataStore for Critic Agent.

Provides full schema validation against Neo4j database.
Used by Critic to validate Planner's analysis plan.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from neo4j import GraphDatabase, Driver

from config import config

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A validation issue found by Critic."""
    
    issue_type: str  # table_not_found, column_not_found, invalid_join, etc.
    severity: str    # error, warning, suggestion
    message: str
    details: Dict[str, Any] = None
    suggestion: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details or {},
            "suggestion": self.suggestion,
        }


class MetadataStore:
    """
    Full schema store for Critic validation.
    
    Unlike SchemaRetriever (which returns filtered sub-graph),
    MetadataStore provides access to the ENTIRE schema
    for comprehensive validation.
    """
    
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        domain: str = "vnfilm_ticketing",
    ):
        self.uri = uri or config.neo4j.uri
        self.user = user or config.neo4j.user
        self.password = password or config.neo4j.password
        self.domain = domain
        self._driver: Driver | None = None
        
        # Cache for schema data
        self._tables_cache: Dict[str, Dict] = {}
        self._columns_cache: Dict[str, List[Dict]] = {}
        self._joins_cache: List[Dict] = []
        self._loaded = False
    
    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            logger.info(f"Connected to Neo4j at {self.uri}")
        return self._driver
    
    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # =========================================================================
    # Load Full Schema
    # =========================================================================
    
    def load_schema(self) -> None:
        """Load entire schema into memory."""
        if self._loaded:
            return
            
        logger.info(f"Loading full schema for domain: {self.domain}")
        
        with self.driver.session() as session:
            # Load all tables
            tables = session.run("""
                MATCH (t:Table {domain: $domain})
                RETURN t.table_name AS table_name,
                       t.business_name AS business_name,
                       t.table_type AS table_type,
                       t.description AS description,
                       t.grain AS grain,
                       t.tags AS tags
            """, domain=self.domain)
            
            for record in tables:
                self._tables_cache[record["table_name"]] = dict(record)
            
            # Load all columns
            columns = session.run("""
                MATCH (t:Table {domain: $domain})-[r:HAS_COLUMN]->(c:Column)
                RETURN t.table_name AS table_name,
                       c.column_name AS column_name,
                       c.data_type AS data_type,
                       c.business_name AS business_name,
                       c.description AS description,
                       c.semantics AS semantics,
                       r.primary_key AS is_primary_key,
                       r.time_column AS is_time_column
            """, domain=self.domain)
            
            for record in columns:
                table_name = record["table_name"]
                if table_name not in self._columns_cache:
                    self._columns_cache[table_name] = []
                self._columns_cache[table_name].append(dict(record))
            
            # Load all joins
            joins = session.run("""
                MATCH (t1:Table)-[j:JOIN]->(t2:Table)
                WHERE t1.domain = $domain OR t2.domain = $domain
                RETURN t1.table_name AS from_table,
                       t2.table_name AS to_table,
                       j.join_type AS join_type,
                       j.on AS on_clause
            """, domain=self.domain)
            
            for record in joins:
                self._joins_cache.append(dict(record))
            
            # Also load FK relationships
            fks = session.run("""
                MATCH (t1:Table)-[fk:FK]->(t2:Table)
                WHERE t1.domain = $domain OR t2.domain = $domain
                RETURN t1.table_name AS from_table,
                       t2.table_name AS to_table,
                       fk.column AS column,
                       fk.references_column AS references_column
            """, domain=self.domain)
            
            for record in fks:
                self._joins_cache.append({
                    "from_table": record["from_table"],
                    "to_table": record["to_table"],
                    "join_type": "fk",
                    "on_clause": [
                        f"{record['from_table']}.{record['column']} = "
                        f"{record['to_table']}.{record['references_column']}"
                    ],
                })
        
        self._loaded = True
        logger.info(
            f"Loaded: {len(self._tables_cache)} tables, "
            f"{sum(len(c) for c in self._columns_cache.values())} columns, "
            f"{len(self._joins_cache)} joins"
        )
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_all_tables(self) -> List[str]:
        """Get all table names."""
        self.load_schema()
        return list(self._tables_cache.keys())
    
    def get_table(self, table_name: str) -> Optional[Dict]:
        """Get table metadata."""
        self.load_schema()
        return self._tables_cache.get(table_name)
    
    def get_columns(self, table_name: str) -> List[Dict]:
        """Get all columns for a table."""
        self.load_schema()
        return self._columns_cache.get(table_name, [])
    
    def get_column(self, table_name: str, column_name: str) -> Optional[Dict]:
        """Get specific column metadata."""
        cols = self.get_columns(table_name)
        for col in cols:
            if col["column_name"] == column_name:
                return col
        return None
    
    def get_all_joins(self) -> List[Dict]:
        """Get all join definitions."""
        self.load_schema()
        return self._joins_cache
    
    def can_join(self, from_table: str, to_table: str) -> bool:
        """Check if two tables can be joined."""
        self.load_schema()
        for join in self._joins_cache:
            if (join["from_table"] == from_table and join["to_table"] == to_table) or \
               (join["from_table"] == to_table and join["to_table"] == from_table):
                return True
        return False
    
    def get_join_path(self, from_table: str, to_table: str) -> Optional[Dict]:
        """Get join clause between two tables."""
        self.load_schema()
        for join in self._joins_cache:
            if (join["from_table"] == from_table and join["to_table"] == to_table) or \
               (join["from_table"] == to_table and join["to_table"] == from_table):
                return join
        return None
    
    def find_similar_table(self, table_name: str, use_llm: bool = True) -> Optional[str]:
        """
        Find similar table name using LLM semantic matching.
        
        Args:
            table_name: The table name to find alternatives for
            use_llm: Whether to use LLM for semantic matching
            
        Returns:
            Similar table name or None
        """
        self.load_schema()
        
        if not self._tables_cache:
            return None
        
        # Prepare table info for matching
        all_tables = []
        for name, info in self._tables_cache.items():
            all_tables.append({
                "name": name,
                "description": info.get("description", ""),
                "business_name": info.get("business_name", ""),
            })
        
        if use_llm:
            return self._find_similar_table_llm(table_name, all_tables)
        else:
            # Fallback to simple matching
            for table in all_tables:
                name = table["name"]
                if table_name in name or name in table_name:
                    return name
            return None
    
    def _find_similar_table_llm(
        self,
        table_name: str,
        all_tables: List[Dict],
    ) -> Optional[str]:
        """Use LLM to find semantically similar table."""
        try:
            from langchain_openai import ChatOpenAI
            
            # Format table list for LLM
            table_list = "\n".join([
                f"- {t['name']}: {t.get('business_name', '')} - {t.get('description', '')[:100]}"
                for t in all_tables
            ])
            
            prompt = f"""Người dùng yêu cầu bảng "{table_name}" nhưng không tồn tại.

Danh sách các bảng có sẵn:
{table_list}

Hãy tìm bảng có ý nghĩa gần nhất với "{table_name}".
Ví dụ: "customers" có thể tương đương với "customer_tracking" hoặc "user_info".

Chỉ trả lời TÊN BẢNG duy nhất phù hợp nhất, không giải thích.
Nếu không có bảng nào phù hợp, trả lời "NONE"."""

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(prompt)
            
            result = response.content.strip()
            
            # Validate result exists
            if result != "NONE" and result in self._tables_cache:
                logger.info(f"[LLM] Found similar table: '{table_name}' → '{result}'")
                return result
            
            return None
            
        except Exception as e:
            logger.warning(f"LLM table matching failed: {e}")
            # Fallback to simple matching
            for table in all_tables:
                name = table["name"]
                if table_name in name or name in table_name:
                    return name
            return None
    
    def find_similar_column(
        self,
        table_name: str,
        column_name: str,
        use_llm: bool = True,
    ) -> Optional[str]:
        """
        Find similar column name using LLM semantic matching.
        
        Args:
            table_name: The table to search in
            column_name: The column name to find alternatives for
            use_llm: Whether to use LLM for semantic matching
        """
        cols = self.get_columns(table_name)
        if not cols:
            return None
        
        if use_llm:
            return self._find_similar_column_llm(table_name, column_name, cols)
        else:
            # Fallback to simple matching
            for col in cols:
                name = col["column_name"]
                if column_name in name or name in column_name:
                    return name
            return None
    
    def _find_similar_column_llm(
        self,
        table_name: str,
        column_name: str,
        cols: List[Dict],
    ) -> Optional[str]:
        """Use LLM to find semantically similar column."""
        try:
            from langchain_openai import ChatOpenAI
            
            # Format column list for LLM
            col_list = "\n".join([
                f"- {c['column_name']}: {c.get('business_name', '')} ({c.get('data_type', '')}) - {c.get('description', '')[:80]}"
                for c in cols
            ])
            
            prompt = f"""Người dùng tìm column "{column_name}" trong bảng "{table_name}" nhưng không tồn tại.

Danh sách các columns có sẵn trong bảng {table_name}:
{col_list}

Hãy tìm column có ý nghĩa gần nhất với "{column_name}".
Ví dụ: "revenue" có thể tương đương với "vnpay_final_amount" hoặc "total_amount".

Chỉ trả lời TÊN COLUMN duy nhất phù hợp nhất, không giải thích.
Nếu không có column nào phù hợp, trả lời "NONE"."""

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(prompt)
            
            result = response.content.strip()
            
            # Validate result exists
            col_names = {c["column_name"] for c in cols}
            if result != "NONE" and result in col_names:
                logger.info(f"[LLM] Found similar column: '{column_name}' → '{result}'")
                return result
            
            return None
            
        except Exception as e:
            logger.warning(f"LLM column matching failed: {e}")
            # Fallback
            for col in cols:
                name = col["column_name"]
                if column_name in name or name in column_name:
                    return name
            return None
    
    # =========================================================================
    # Validation Methods
    # =========================================================================
    
    def validate_table(self, table_name: str) -> Optional[ValidationIssue]:
        """Validate that a table exists."""
        if not self.get_table(table_name):
            suggestion = self.find_similar_table(table_name)
            return ValidationIssue(
                issue_type="table_not_found",
                severity="error",
                message=f"Table '{table_name}' không tồn tại trong schema",
                details={"table_name": table_name},
                suggestion=f"Có thể bạn muốn nói '{suggestion}'?" if suggestion else None,
            )
        return None
    
    def validate_column(self, table_name: str, column_name: str) -> Optional[ValidationIssue]:
        """Validate that a column exists in a table."""
        if not self.get_column(table_name, column_name):
            suggestion = self.find_similar_column(table_name, column_name)
            return ValidationIssue(
                issue_type="column_not_found",
                severity="error",
                message=f"Column '{column_name}' không tồn tại trong table '{table_name}'",
                details={"table_name": table_name, "column_name": column_name},
                suggestion=f"Có thể bạn muốn nói '{suggestion}'?" if suggestion else None,
            )
        return None
    
    def validate_join(self, from_table: str, to_table: str) -> Optional[ValidationIssue]:
        """Validate that two tables can be joined."""
        if not self.can_join(from_table, to_table):
            return ValidationIssue(
                issue_type="invalid_join",
                severity="warning",
                message=f"Không có join path giữa '{from_table}' và '{to_table}'",
                details={"from_table": from_table, "to_table": to_table},
                suggestion="Cần thêm bảng trung gian hoặc xem lại logic join",
            )
        return None
    
    def validate_plan(self, plan: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate an entire analysis plan.
        
        Args:
            plan: Analysis plan dict with 'steps', 'hypotheses', etc.
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Extract tables and columns from plan
        steps = plan.get("steps", [])
        
        for step in steps:
            # Check tables mentioned in step
            tables_used = step.get("tables", [])
            if isinstance(tables_used, str):
                tables_used = [tables_used]
            
            for table in tables_used:
                issue = self.validate_table(table)
                if issue:
                    issues.append(issue)
            
            # Check columns mentioned in step
            columns_used = step.get("columns", [])
            for col_ref in columns_used:
                if isinstance(col_ref, dict):
                    table = col_ref.get("table")
                    column = col_ref.get("column")
                    if table and column:
                        issue = self.validate_column(table, column)
                        if issue:
                            issues.append(issue)
            
            # Check SQL if present
            sql = step.get("sql", "")
            if sql:
                sql_issues = self._validate_sql_references(sql)
                issues.extend(sql_issues)
        
        return issues
    
    def _validate_sql_references(self, sql: str) -> List[ValidationIssue]:
        """Basic SQL validation (check table/column references)."""
        issues = []
        sql_lower = sql.lower()
        
        # Simple check: see if tables in SQL exist
        for table_name in self._tables_cache.keys():
            if table_name in sql_lower:
                continue  # Table found in SQL, OK
        
        # Could add more SQL parsing here
        return issues
    
    def format_feedback(self, issues: List[ValidationIssue]) -> str:
        """Format validation issues as feedback for Planner."""
        if not issues:
            return ""
        
        lines = ["## Validation Issues Found:\n"]
        
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        
        if errors:
            lines.append("### ❌ Errors (must fix):\n")
            for i, issue in enumerate(errors, 1):
                lines.append(f"{i}. **{issue.issue_type}**: {issue.message}")
                if issue.suggestion:
                    lines.append(f"   → Suggestion: {issue.suggestion}")
                lines.append("")
        
        if warnings:
            lines.append("### ⚠️ Warnings:\n")
            for i, issue in enumerate(warnings, 1):
                lines.append(f"{i}. **{issue.issue_type}**: {issue.message}")
                if issue.suggestion:
                    lines.append(f"   → Suggestion: {issue.suggestion}")
                lines.append("")
        
        return "\n".join(lines)
