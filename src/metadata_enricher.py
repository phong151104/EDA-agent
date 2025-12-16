"""
Dynamic Metadata Enricher for EDA Agent.

Automatically enriches SQL query results with column descriptions
from metadata YAML files.

Usage:
    from src.metadata_enricher import MetadataEnricher
    
    enricher = MetadataEnricher()
    enriched = enricher.enrich(
        sql="SELECT vnpay_final_amount FROM orders",
        data=[{"vnpay_final_amount": 100000}]
    )
    # Returns: {
    #   "data": [...],
    #   "columns": {"vnpay_final_amount": {"description": "...", ...}},
    #   "tables_used": ["orders"]
    # }
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

METADATA_DIR = Path(__file__).parent.parent / "metadata" / "domains" / "vnfilm_ticketing" / "tables"


class MetadataEnricher:
    """Enriches SQL results with column metadata from YAML files."""
    
    def __init__(self, domain: str = "vnfilm_ticketing"):
        self.domain = domain
        self._metadata_cache: dict[str, dict] = {}
    
    def enrich(
        self,
        sql: str,
        data: list[dict],
        tables_hint: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Enrich SQL result with column metadata.
        
        Args:
            sql: The SQL query that was executed
            data: The query result rows
            tables_hint: Optional list of table names (if already known)
            
        Returns:
            Enriched result with data, column metadata, and table info
        """
        # 1. Extract tables from SQL
        tables_used = tables_hint or self._extract_tables_from_sql(sql)
        
        # 2. Get columns from data
        result_columns = list(data[0].keys()) if data else []
        
        # 3. Load metadata for columns
        column_metadata = self._get_column_metadata(result_columns, tables_used)
        
        # 4. Build context string for Code Agent
        context_text = self._build_context_text(column_metadata)
        
        return {
            "data": data,
            "row_count": len(data),
            "columns": column_metadata,
            "tables_used": tables_used,
            "context_text": context_text,
        }
    
    def _extract_tables_from_sql(self, sql: str) -> list[str]:
        """Extract table names from SQL query."""
        tables = []
        
        # Match: FROM schema.table or FROM table
        from_pattern = r'FROM\s+(?:[\w.]+\.)?(\w+)'
        tables.extend(re.findall(from_pattern, sql, re.IGNORECASE))
        
        # Match: JOIN schema.table or JOIN table
        join_pattern = r'JOIN\s+(?:[\w.]+\.)?(\w+)'
        tables.extend(re.findall(join_pattern, sql, re.IGNORECASE))
        
        # Remove duplicates
        return list(set(tables))
    
    def _load_table_metadata(self, table_name: str) -> dict | None:
        """Load metadata YAML for a table."""
        if table_name in self._metadata_cache:
            return self._metadata_cache[table_name]
        
        yaml_path = METADATA_DIR / f"{table_name}.yml"
        if not yaml_path.exists():
            logger.debug(f"[Enricher] No metadata for table: {table_name}")
            return None
        
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
            self._metadata_cache[table_name] = metadata
            return metadata
        except Exception as e:
            logger.error(f"[Enricher] Error loading {yaml_path}: {e}")
            return None
    
    def _get_column_metadata(
        self,
        columns: list[str],
        tables: list[str]
    ) -> dict[str, dict]:
        """Get metadata for columns from relevant tables."""
        column_info = {}
        
        # Build a map of column_name -> metadata from all tables
        all_columns = {}
        for table_name in tables:
            metadata = self._load_table_metadata(table_name)
            if not metadata:
                continue
            
            # YAML structure: columns is a dict of {column_name: {properties}}
            columns_dict = metadata.get("columns", {})
            
            # Handle both dict and list formats
            if isinstance(columns_dict, dict):
                for col_name, col_props in columns_dict.items():
                    if isinstance(col_props, dict):
                        all_columns[col_name] = {
                            "table": table_name,
                            "description": col_props.get("description", ""),
                            "data_type": col_props.get("data_type", ""),
                            "business_name": col_props.get("business_name", ""),
                            "semantics": col_props.get("semantics", []),
                        }
            elif isinstance(columns_dict, list):
                for col in columns_dict:
                    if isinstance(col, dict):
                        col_name = col.get("name", "")
                        all_columns[col_name] = {
                            "table": table_name,
                            "description": col.get("description", ""),
                            "data_type": col.get("data_type", ""),
                            "business_name": col.get("business_name", ""),
                        }
        
        # Match result columns to metadata
        for col in columns:
            # Direct match
            if col in all_columns:
                column_info[col] = all_columns[col]
            else:
                # Try alias matching (e.g., total_revenue might be SUM(vnpay_final_amount))
                # For now, just mark as computed
                column_info[col] = {
                    "table": "derived",
                    "description": f"Computed column: {col}",
                    "data_type": "unknown",
                }
        
        return column_info
    
    def _build_context_text(self, column_metadata: dict[str, dict]) -> str:
        """Build human-readable context for Code Agent."""
        if not column_metadata:
            return ""
        
        lines = ["## Column Descriptions:"]
        for col_name, info in column_metadata.items():
            desc = info.get("description", "No description")
            dtype = info.get("data_type", "unknown")
            lines.append(f"- `{col_name}` ({dtype}): {desc}")
        
        return "\n".join(lines)
    
    def get_table_context(self, table_name: str) -> dict | None:
        """Get full table context for Code Agent."""
        metadata = self._load_table_metadata(table_name)
        if not metadata:
            return None
        
        return {
            "table_name": table_name,
            "description": metadata.get("description", ""),
            "columns": [
                {
                    "name": c.get("name"),
                    "description": c.get("description", ""),
                    "data_type": c.get("data_type", ""),
                }
                for c in metadata.get("columns", [])[:20]  # Limit to top 20
            ],
            "business_context": metadata.get("business_context", ""),
        }


# Convenience function
def enrich_sql_result(
    sql: str,
    data: list[dict],
    tables_hint: list[str] | None = None
) -> dict[str, Any]:
    """Quick function to enrich SQL result."""
    return MetadataEnricher().enrich(sql, data, tables_hint)
