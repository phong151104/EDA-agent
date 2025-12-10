"""
Context Fusion Layer - Data Models.

Models for query analysis and sub-graph representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntent(str, Enum):
    """Types of analytical intent."""
    
    EXPLORATORY = "exploratory"     # "Doanh thu như thế nào?"
    DIAGNOSTIC = "diagnostic"       # "Tại sao doanh thu giảm?"
    COMPARATIVE = "comparative"     # "So sánh Q1 vs Q2"
    TREND = "trend"                 # "Xu hướng doanh thu"
    AGGREGATION = "aggregation"     # "Tổng doanh thu theo tháng"
    DETAIL = "detail"               # "Chi tiết đơn hàng X"
    UNKNOWN = "unknown"


@dataclass
class ExtractedEntity:
    """An entity extracted from user query."""
    
    text: str                       # Original text in query
    entity_type: str                # table, column, metric, concept, time, value
    normalized_name: str | None = None  # Mapped to actual name in schema
    confidence: float = 1.0


@dataclass
class AnalyzedQuery:
    """Result of query analysis."""
    
    original_query: str
    intent: QueryIntent
    entities: list[ExtractedEntity] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    time_range: dict[str, str] | None = None  # {"start": "...", "end": "..."}
    clarifications_needed: list[str] = field(default_factory=list)
    rewritten_query: str | None = None
    
    def get_search_terms(self) -> list[str]:
        """Get terms for searching Neo4j."""
        terms = list(self.keywords)
        for entity in self.entities:
            terms.append(entity.text)
            if entity.normalized_name:
                terms.append(entity.normalized_name)
        return list(set(terms))


# === Sub-graph Models ===

@dataclass
class TableNode:
    """A table in the sub-graph."""
    
    table_name: str
    domain: str
    business_name: str = ""
    description: str = ""
    grain: str = ""
    table_type: str = ""
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "table_name": self.table_name,
            "domain": self.domain,
            "business_name": self.business_name,
            "description": self.description,
            "grain": self.grain,
            "table_type": self.table_type,
            "tags": self.tags,
        }


@dataclass
class ColumnNode:
    """A column in the sub-graph."""
    
    table_name: str
    column_name: str
    data_type: str = ""
    business_name: str = ""
    description: str = ""
    semantics: list[str] = field(default_factory=list)
    is_primary_key: bool = False
    is_time_column: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "table_name": self.table_name,
            "column_name": self.column_name,
            "data_type": self.data_type,
            "business_name": self.business_name,
            "description": self.description,
            "semantics": self.semantics,
            "is_primary_key": self.is_primary_key,
            "is_time_column": self.is_time_column,
        }


@dataclass
class JoinEdge:
    """A join relationship in the sub-graph."""
    
    from_table: str
    to_table: str
    join_type: str = "inner"
    on_clause: list[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "from_table": self.from_table,
            "to_table": self.to_table,
            "join_type": self.join_type,
            "on_clause": self.on_clause,
            "description": self.description,
        }


@dataclass
class MetricNode:
    """A metric in the sub-graph."""
    
    name: str
    business_name: str = ""
    expression: str = ""
    base_table: str = ""
    description: str = ""
    unit: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "business_name": self.business_name,
            "expression": self.expression,
            "base_table": self.base_table,
            "description": self.description,
            "unit": self.unit,
        }


@dataclass
class ConceptNode:
    """A concept/semantic in the sub-graph."""
    
    name: str
    synonyms: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "synonyms": self.synonyms,
        }


@dataclass
class SubGraph:
    """
    Sub-graph extracted from Neo4j.
    
    Contains all relevant schema information for the query.
    
    Features:
    - Pre-computed indexes for O(1) lookups
    - Bidirectional join traversal
    - Compact prompt generation for LLM
    """
    
    tables: list[TableNode] = field(default_factory=list)
    columns: list[ColumnNode] = field(default_factory=list)
    joins: list[JoinEdge] = field(default_factory=list)
    metrics: list[MetricNode] = field(default_factory=list)
    concepts: list[ConceptNode] = field(default_factory=list)
    
    # Pre-computed indexes (built on first access)
    _table_index: dict[str, TableNode] = field(default_factory=dict, repr=False)
    _columns_by_table: dict[str, list[ColumnNode]] = field(default_factory=dict, repr=False)
    _joins_by_table: dict[str, list[JoinEdge]] = field(default_factory=dict, repr=False)
    _indexed: bool = field(default=False, repr=False)
    
    def _build_indexes(self) -> None:
        """Build lookup indexes for fast access."""
        if self._indexed:
            return
        
        # Table index: name -> TableNode
        self._table_index = {t.table_name: t for t in self.tables}
        
        # Columns grouped by table
        self._columns_by_table = {}
        for c in self.columns:
            if c.table_name not in self._columns_by_table:
                self._columns_by_table[c.table_name] = []
            self._columns_by_table[c.table_name].append(c)
        
        # Joins by table (bidirectional)
        self._joins_by_table = {}
        for j in self.joins:
            # Forward
            if j.from_table not in self._joins_by_table:
                self._joins_by_table[j.from_table] = []
            self._joins_by_table[j.from_table].append(j)
            # Reverse
            if j.to_table not in self._joins_by_table:
                self._joins_by_table[j.to_table] = []
            if j not in self._joins_by_table[j.to_table]:
                self._joins_by_table[j.to_table].append(j)
        
        self._indexed = True
    
    # =========================================================================
    # Fast Lookups (O(1) with indexes)
    # =========================================================================
    
    def get_table(self, name: str) -> TableNode | None:
        """Get table by name (O(1))."""
        self._build_indexes()
        return self._table_index.get(name)
    
    def get_columns_for_table(self, table_name: str) -> list[ColumnNode]:
        """Get all columns for a table (O(1))."""
        self._build_indexes()
        return self._columns_by_table.get(table_name, [])
    
    def get_joins_for_table(self, table_name: str) -> list[JoinEdge]:
        """Get all joins involving a table (O(1))."""
        self._build_indexes()
        return self._joins_by_table.get(table_name, [])
    
    def get_related_tables(self, table_name: str) -> list[str]:
        """Get tables that can be joined with given table."""
        joins = self.get_joins_for_table(table_name)
        related = set()
        for j in joins:
            if j.from_table == table_name:
                related.add(j.to_table)
            else:
                related.add(j.from_table)
        return list(related)
    
    def get_primary_keys(self, table_name: str | None = None) -> list[ColumnNode]:
        """Get primary key columns."""
        self._build_indexes()
        if table_name:
            return [c for c in self._columns_by_table.get(table_name, []) if c.is_primary_key]
        return [c for c in self.columns if c.is_primary_key]
    
    def get_time_columns(self, table_name: str | None = None) -> list[ColumnNode]:
        """Get time columns."""
        self._build_indexes()
        if table_name:
            return [c for c in self._columns_by_table.get(table_name, []) if c.is_time_column]
        return [c for c in self.columns if c.is_time_column]
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "tables": [t.to_dict() for t in self.tables],
            "columns": [c.to_dict() for c in self.columns],
            "joins": [j.to_dict() for j in self.joins],
            "metrics": [m.to_dict() for m in self.metrics],
            "concepts": [c.to_dict() for c in self.concepts],
        }
    
    # =========================================================================
    # Prompt Generation (Compact format to save tokens)
    # =========================================================================
    
    def to_prompt_context(self, compact: bool = False) -> str:
        """
        Convert sub-graph to text for LLM prompt.
        
        Args:
            compact: If True, use shorter format to save tokens
        """
        if compact:
            return self._to_compact_prompt()
        return self._to_detailed_prompt()
    
    def _to_detailed_prompt(self) -> str:
        """Detailed format with full descriptions."""
        lines = []
        
        # Tables
        if self.tables:
            lines.append("## Available Tables")
            for t in self.tables:
                lines.append(f"- **{t.table_name}** ({t.business_name})")
                if t.description:
                    lines.append(f"  {t.description}")
                if t.grain:
                    lines.append(f"  Grain: {t.grain}")
        
        # Columns (grouped by table)
        if self.columns:
            lines.append("\n## Columns")
            self._build_indexes()
            for table_name, cols in self._columns_by_table.items():
                lines.append(f"\n### {table_name}")
                for c in cols:
                    pk = " [PK]" if c.is_primary_key else ""
                    time = " [TIME]" if c.is_time_column else ""
                    lines.append(f"- {c.column_name} ({c.data_type}){pk}{time}")
                    if c.business_name:
                        lines.append(f"  → {c.business_name}")
        
        # Joins
        if self.joins:
            lines.append("\n## Joins")
            for j in self.joins:
                lines.append(f"- {j.from_table} → {j.to_table} ({j.join_type})")
                for clause in j.on_clause:
                    lines.append(f"  ON {clause}")
        
        # Metrics
        if self.metrics:
            lines.append("\n## Metrics")
            for m in self.metrics:
                lines.append(f"- **{m.name}** ({m.business_name})")
                lines.append(f"  Expression: `{m.expression}`")
        
        return "\n".join(lines)
    
    def _to_compact_prompt(self) -> str:
        """Compact format to minimize tokens."""
        lines = []
        
        # Tables (one line each)
        if self.tables:
            lines.append("## Tables")
            for t in self.tables:
                lines.append(f"- {t.table_name}: {t.business_name}")
        
        # Key columns only (PK + TIME)
        key_cols = [c for c in self.columns if c.is_primary_key or c.is_time_column]
        if key_cols:
            lines.append("\n## Key Columns")
            self._build_indexes()
            for table_name in self._columns_by_table:
                table_keys = [c for c in self._columns_by_table[table_name] 
                              if c.is_primary_key or c.is_time_column]
                if table_keys:
                    cols_str = ", ".join([
                        f"{c.column_name}{'[PK]' if c.is_primary_key else '[T]'}" 
                        for c in table_keys
                    ])
                    lines.append(f"- {table_name}: {cols_str}")
        
        # Joins (compact format)
        if self.joins:
            lines.append("\n## Joins")
            for j in self.joins:
                on = j.on_clause[0] if j.on_clause else ""
                lines.append(f"- {j.from_table}→{j.to_table}: {on}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # Backward Compatible Methods
    # =========================================================================
    
    def get_table_names(self) -> list[str]:
        """Get list of table names."""
        return [t.table_name for t in self.tables]
    
    def get_column_names(self, table_name: str | None = None) -> list[str]:
        """Get column names, optionally filtered by table."""
        if table_name:
            return [c.column_name for c in self.columns if c.table_name == table_name]
        return [f"{c.table_name}.{c.column_name}" for c in self.columns]

