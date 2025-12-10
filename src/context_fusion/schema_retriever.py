"""
Schema Retriever with Hybrid Search.

Combines:
1. Vector search (semantic similarity)
2. Rule-based keyword matching
3. LLM-extracted entities

Uses score fusion to produce final ranked results.
"""

from __future__ import annotations

import logging
from typing import Any, List, Dict, Set
from dataclasses import dataclass

from neo4j import GraphDatabase, Driver

from config import config
from .models import (
    AnalyzedQuery,
    ColumnNode,
    ConceptNode,
    JoinEdge,
    MetricNode,
    SubGraph,
    TableNode,
)
from .vector_index import Neo4jVectorIndex

logger = logging.getLogger(__name__)


@dataclass
class ScoredTable:
    """A table with combined relevance score."""
    table_name: str
    vector_score: float = 0.0
    keyword_score: float = 0.0
    entity_score: float = 0.0
    final_score: float = 0.0
    match_sources: List[str] = None
    
    def __post_init__(self):
        if self.match_sources is None:
            self.match_sources = []


class SchemaRetriever:
    """
    Retrieves relevant schema sub-graph from Neo4j.
    
    Hybrid Strategy:
    1. Vector search with original query (semantic)
    2. Vector search with LLM-extracted entities
    3. Keyword boost for rule-based matches
    4. Score fusion to rank final results
    """
    
    # Score weights
    VECTOR_WEIGHT = 0.5      # α: Vector similarity weight
    KEYWORD_WEIGHT = 0.3     # β: Rule-based keyword match weight  
    ENTITY_WEIGHT = 0.2      # γ: LLM entity match weight
    
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        vector_index: Neo4jVectorIndex | None = None,
    ):
        """Initialize with Neo4j connection."""
        self.uri = uri or config.neo4j.uri
        self.user = user or config.neo4j.user
        self.password = password or config.neo4j.password
        self._driver: Driver | None = None
        self.vector_index = vector_index or Neo4jVectorIndex(
            uri=self.uri, user=self.user, password=self.password
        )
    
    @property
    def driver(self) -> Driver:
        """Get or create Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
        return self._driver
    
    def execute_query(self, query: str, params: dict = None) -> List[Dict]:
        """Execute a read query and return results."""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
        self.vector_index.close()
    
    async def retrieve(
        self,
        analyzed_query: AnalyzedQuery,
        domain: str = "vnfilm_ticketing",
        top_k: int = 10,
        expand_depth: int = 2,
    ) -> SubGraph:
        """
        Retrieve relevant sub-graph using hybrid search.
        
        Combines:
        - Vector search on original query
        - Vector search on LLM-extracted entities
        - Keyword boost from rule-based matching
        """
        question = analyzed_query.original_query
        keywords = analyzed_query.keywords
        entities = analyzed_query.entities
        
        logger.info(f"[HYBRID] 🔍 Query: {question[:50]}...")
        logger.info(f"[HYBRID] Keywords: {keywords[:5]}...")
        logger.info(f"[HYBRID] Entities: {[(e.text, e.normalized_name) for e in entities[:3]]}...")
        
        # === STEP 1: Multi-signal Search ===
        scored_tables: Dict[str, ScoredTable] = {}
        
        # 1A: Vector search with original query
        logger.info(f"[STEP 1A] Vector search with original query...")
        vector_results = self.vector_index.vector_search(question, top_k=top_k * 2)
        
        for result in vector_results:
            table_name = self._extract_table_from_result(result)
            if table_name:
                if table_name not in scored_tables:
                    scored_tables[table_name] = ScoredTable(table_name=table_name)
                scored_tables[table_name].vector_score = max(
                    scored_tables[table_name].vector_score,
                    result.get("score", 0)
                )
                scored_tables[table_name].match_sources.append("vector_query")
        
        logger.info(f"[STEP 1A] ✅ Found {len(scored_tables)} tables from vector search")
        
        # 1B: Vector search with LLM-extracted entities
        entity_terms = [e.normalized_name or e.text for e in entities if e.normalized_name]
        if entity_terms:
            logger.info(f"[STEP 1B] Vector search with entities: {entity_terms}...")
            
            for entity_term in entity_terms[:5]:  # Limit to top 5 entities
                try:
                    entity_results = self.vector_index.vector_search(entity_term, top_k=5)
                    for result in entity_results:
                        table_name = self._extract_table_from_result(result)
                        if table_name:
                            if table_name not in scored_tables:
                                scored_tables[table_name] = ScoredTable(table_name=table_name)
                            scored_tables[table_name].entity_score = max(
                                scored_tables[table_name].entity_score,
                                result.get("score", 0) * 0.8  # Slightly lower weight
                            )
                            if "entity_search" not in scored_tables[table_name].match_sources:
                                scored_tables[table_name].match_sources.append("entity_search")
                except Exception as e:
                    logger.warning(f"Entity search failed for '{entity_term}': {e}")
        
        # 1C: Keyword boost from rule-based matching
        if keywords:
            logger.info(f"[STEP 1C] Keyword matching boost...")
            keyword_matches = self._keyword_search(keywords, domain)
            
            for table_name, match_count in keyword_matches.items():
                if table_name not in scored_tables:
                    scored_tables[table_name] = ScoredTable(table_name=table_name)
                # Normalize keyword score (max 1.0)
                scored_tables[table_name].keyword_score = min(match_count / 3.0, 1.0)
                if "keyword_match" not in scored_tables[table_name].match_sources:
                    scored_tables[table_name].match_sources.append("keyword_match")
        
        # === STEP 2: Score Fusion ===
        logger.info(f"[STEP 2] 📊 Score fusion...")
        
        for table in scored_tables.values():
            table.final_score = (
                self.VECTOR_WEIGHT * table.vector_score +
                self.KEYWORD_WEIGHT * table.keyword_score +
                self.ENTITY_WEIGHT * table.entity_score
            )
            
            # Bonus for multi-source matches
            if len(table.match_sources) >= 2:
                table.final_score *= 1.2
            if len(table.match_sources) >= 3:
                table.final_score *= 1.1
        
        # Sort by final score
        ranked_tables = sorted(
            scored_tables.values(),
            key=lambda t: t.final_score,
            reverse=True
        )[:top_k]
        
        # Log top results
        logger.info(f"[STEP 2] ✅ Top tables by combined score:")
        for i, t in enumerate(ranked_tables[:5], 1):
            logger.info(
                f"  #{i} {t.table_name}: final={t.final_score:.3f} "
                f"(vec={t.vector_score:.2f}, kw={t.keyword_score:.2f}, ent={t.entity_score:.2f}) "
                f"sources={t.match_sources}"
            )
        
        # === STEP 3: Expand Context ===
        table_names = {t.table_name for t in ranked_tables}
        relevant_columns = self._extract_relevant_columns(vector_results)
        
        logger.info(f"[STEP 3] 🕸️ Expanding context for {len(table_names)} tables...")
        expanded_context = self._expand_context(table_names, expand_depth, relevant_columns, domain)
        
        # === STEP 4: Build SubGraph ===
        tables = [
            TableNode(
                table_name=t.get("table_name", ""),
                domain=domain,
                business_name=t.get("business_name", ""),
                description=t.get("description", ""),
                grain=t.get("grain", ""),
                table_type=t.get("table_type", ""),
                tags=t.get("tags", []) or [],
            )
            for t in expanded_context["tables"]
        ]
        
        columns = [
            ColumnNode(
                table_name=c.get("table_name", ""),
                column_name=c.get("column_name", ""),
                data_type=c.get("data_type", ""),
                business_name=c.get("business_name", ""),
                description=c.get("description", ""),
                semantics=c.get("semantics", []) or [],
                is_primary_key=c.get("is_primary_key", False) or False,
                is_time_column=c.get("is_time_column", False) or False,
            )
            for c in expanded_context["columns"]
        ]
        
        joins = [
            JoinEdge(
                from_table=j.get("from_table", ""),
                to_table=j.get("to_table", ""),
                join_type=j.get("join_type", "left"),
                on_clause=j.get("on_clause", []) or [],
                description=j.get("description", ""),
            )
            for j in expanded_context["joins"]
        ]
        
        metrics = [
            MetricNode(
                name=m.get("name", ""),
                business_name=m.get("business_name", ""),
                expression=m.get("expression", ""),
                base_table=m.get("base_table", ""),
                description=m.get("description", ""),
                unit=m.get("unit", ""),
            )
            for m in expanded_context["metrics"]
        ]
        
        # Get concepts from entities
        concepts = [
            ConceptNode(name=e.normalized_name or e.text, synonyms=[e.text])
            for e in entities if e.entity_type == "concept"
        ]
        
        logger.info(
            f"[DONE] ✅ Sub-graph: {len(tables)} tables, {len(columns)} columns, "
            f"{len(joins)} joins, {len(metrics)} metrics"
        )
        
        return SubGraph(
            tables=tables,
            columns=columns,
            joins=joins,
            metrics=metrics,
            concepts=concepts,
        )
    
    def _extract_table_from_result(self, result: Dict) -> str | None:
        """Extract table name from vector search result."""
        label = result.get("label")
        props = result.get("props", {})
        
        if label == "Table":
            return props.get("table_name")
        elif label == "Column":
            return props.get("table_name")
        elif label == "Metric":
            return props.get("base_table")
        return None
    
    def _extract_relevant_columns(
        self,
        vector_results: List[Dict[str, Any]],
    ) -> Set[tuple]:
        """Extract (table_name, column_name) pairs from vector search results."""
        columns = set()
        
        for result in vector_results:
            label = result.get("label")
            props = result.get("props", {})
            
            if label == "Column":
                table_name = props.get("table_name", "")
                column_name = props.get("column_name", "")
                if table_name and column_name:
                    columns.add((table_name, column_name))
        
        return columns
    
    def _keyword_search(
        self,
        keywords: List[str],
        domain: str,
    ) -> Dict[str, int]:
        """
        Search for tables matching keywords via Cypher.
        Returns table_name -> match_count mapping.
        """
        if not keywords:
            return {}
        
        query = """
        MATCH (t:Table {domain: $domain})
        OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
        OPTIONAL MATCH (t)-[:HAS_CONCEPT]->(con:Concept)
        WITH t, collect(DISTINCT c) AS cols, collect(DISTINCT con) AS cons, $keywords AS keywords
        WITH t, cols, cons, keywords,
             size([kw IN keywords WHERE 
                toLower(t.table_name) CONTAINS toLower(kw) OR
                toLower(t.business_name) CONTAINS toLower(kw) OR
                toLower(t.description) CONTAINS toLower(kw) OR
                any(col IN cols WHERE 
                    toLower(col.column_name) CONTAINS toLower(kw) OR
                    toLower(col.business_name) CONTAINS toLower(kw) OR
                    any(sem IN col.semantics WHERE toLower(sem) CONTAINS toLower(kw))
                ) OR
                any(concept IN cons WHERE 
                    toLower(concept.name) CONTAINS toLower(kw) OR
                    any(syn IN concept.synonyms WHERE toLower(syn) CONTAINS toLower(kw))
                )
             ]) AS match_count
        WHERE match_count > 0
        RETURN t.table_name AS table_name, match_count
        ORDER BY match_count DESC
        LIMIT 20
        """
        
        try:
            results = self.execute_query(query, {"keywords": keywords, "domain": domain})
            return {r["table_name"]: r.get("match_count", 1) for r in results}
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return {}
    
    def _expand_context(
        self,
        table_names: Set[str],
        depth: int,
        relevant_columns: Set[tuple],
        domain: str,
    ) -> Dict[str, Any]:
        """Expand context using graph traversal."""
        if not table_names:
            return {"tables": [], "columns": [], "joins": [], "metrics": []}
        
        table_names_list = list(table_names)
        
        # Get tables
        tables_query = """
        MATCH (t:Table)
        WHERE t.table_name IN $table_names
        RETURN t.table_name AS table_name,
               t.business_name AS business_name,
               t.table_type AS table_type,
               t.description AS description,
               t.grain AS grain,
               t.tags AS tags
        """
        tables = self.execute_query(tables_query, {"table_names": table_names_list})
        
        # Get key columns (PK, time)
        key_columns_query = """
        MATCH (t:Table)-[r:HAS_COLUMN]->(c:Column)
        WHERE t.table_name IN $table_names
          AND (r.primary_key = true OR r.time_column = true)
        RETURN t.table_name AS table_name,
               c.column_name AS column_name,
               c.data_type AS data_type,
               c.business_name AS business_name,
               c.description AS description,
               c.semantics AS semantics,
               r.primary_key AS is_primary_key,
               r.time_column AS is_time_column
        ORDER BY t.table_name, c.column_name
        """
        key_columns = self.execute_query(key_columns_query, {"table_names": table_names_list})
        
        # Get vector-matched columns
        vector_columns = []
        if relevant_columns:
            all_columns_query = """
            MATCH (t:Table)-[r:HAS_COLUMN]->(c:Column)
            WHERE t.table_name IN $table_names
            RETURN t.table_name AS table_name,
                   c.column_name AS column_name,
                   c.data_type AS data_type,
                   c.business_name AS business_name,
                   c.description AS description,
                   c.semantics AS semantics,
                   r.primary_key AS is_primary_key,
                   r.time_column AS is_time_column
            """
            all_cols = self.execute_query(all_columns_query, {"table_names": table_names_list})
            for col in all_cols:
                if (col['table_name'], col['column_name']) in relevant_columns:
                    vector_columns.append(col)
        
        # Merge columns
        columns_dict = {}
        for col in key_columns:
            key = (col['table_name'], col['column_name'])
            columns_dict[key] = col
        for col in vector_columns:
            key = (col['table_name'], col['column_name'])
            if key not in columns_dict:
                columns_dict[key] = col
        
        columns = sorted(columns_dict.values(), key=lambda x: (x['table_name'], x['column_name']))
        
        # Get joins
        joins_query = """
        MATCH (t1:Table)-[j:JOIN]->(t2:Table)
        WHERE t1.table_name IN $table_names OR t2.table_name IN $table_names
        RETURN t1.table_name AS from_table,
               t2.table_name AS to_table,
               j.join_type AS join_type,
               j.on AS on_clause,
               j.description AS description
        """
        joins = self.execute_query(joins_query, {"table_names": table_names_list})
        
        # Get FK relationships
        fk_query = """
        MATCH (t1:Table)-[fk:FK]->(t2:Table)
        WHERE t1.table_name IN $table_names OR t2.table_name IN $table_names
        RETURN t1.table_name AS from_table,
               t2.table_name AS to_table,
               fk.column AS column,
               fk.references_column AS references_column
        """
        fks = self.execute_query(fk_query, {"table_names": table_names_list})
        
        for fk in fks:
            joins.append({
                "from_table": fk["from_table"],
                "to_table": fk["to_table"],
                "join_type": "left",
                "on_clause": [f"{fk['from_table']}.{fk['column']} = {fk['to_table']}.{fk['references_column']}"],
                "description": "",
            })
        
        # Get metrics
        metrics_query = """
        MATCH (m:Metric)
        WHERE m.base_table IN $table_names
        RETURN m.name AS name,
               m.business_name AS business_name,
               m.description AS description,
               m.expression AS expression,
               m.base_table AS base_table,
               m.unit AS unit
        """
        metrics = self.execute_query(metrics_query, {"table_names": table_names_list})
        
        return {
            "tables": tables,
            "columns": columns,
            "joins": joins,
            "metrics": metrics,
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
