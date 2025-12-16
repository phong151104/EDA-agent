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
                catalog=t.get("catalog") or "lakehouse",
                schema=t.get("schema") or "lh_vnfilm_v2",
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
    
    def _find_join_paths(
        self,
        table_names: List[str],
        domain: str,
        max_depth: int = 4,
    ) -> Dict[str, Any]:
        """
        Find optimal join paths between tables using Neo4j shortestPath.
        
        This ensures we get the correct joins to connect all required tables,
        including any intermediate tables needed for multi-hop joins.
        
        Args:
            table_names: List of table names to connect
            domain: Domain to search in
            max_depth: Maximum path depth (default 4 hops)
            
        Returns:
            Dict with:
            - joins: List of join dicts on the optimal paths
            - intermediate_tables: Set of table names needed for multi-hop joins
        """
        if len(table_names) < 2:
            return {"joins": [], "intermediate_tables": set()}
        
        all_joins = []
        intermediate_tables = set()
        seen_join_keys = set()  # Avoid duplicate joins
        
        # Use first table as anchor
        anchor_table = table_names[0]
        other_tables = table_names[1:]
        
        logger.info(f"[PATH] Finding join paths from anchor '{anchor_table}' to {other_tables}")
        
        for target_table in other_tables:
            # Find shortest path between anchor and target
            path_query = """
            MATCH path = shortestPath(
                (t1:Table {table_name: $anchor})-[:JOIN|FK*1..""" + str(max_depth) + """]-(t2:Table {table_name: $target})
            )
            WHERE t1.domain = $domain AND t2.domain = $domain
            WITH path, relationships(path) AS rels, nodes(path) AS nodes
            UNWIND range(0, size(rels)-1) AS idx
            WITH nodes[idx] AS from_node, nodes[idx+1] AS to_node, rels[idx] AS rel
            RETURN 
                from_node.table_name AS from_table,
                to_node.table_name AS to_table,
                type(rel) AS rel_type,
                rel.join_type AS join_type,
                rel.on AS on_clause,
                rel.column AS fk_column,
                rel.references_column AS fk_ref_column,
                rel.description AS description
            """
            
            try:
                results = self.execute_query(path_query, {
                    "anchor": anchor_table,
                    "target": target_table,
                    "domain": domain,
                })
                
                if not results:
                    logger.warning(f"[PATH] No path found: {anchor_table} → {target_table}")
                    continue
                
                logger.info(f"[PATH] ✅ Found path {anchor_table} → {target_table} with {len(results)} hops")
                
                for r in results:
                    from_t = r["from_table"]
                    to_t = r["to_table"]
                    
                    # Create unique key for deduplication (order-independent)
                    join_key = tuple(sorted([from_t, to_t]))
                    if join_key in seen_join_keys:
                        continue
                    seen_join_keys.add(join_key)
                    
                    # Track intermediate tables
                    if from_t not in table_names:
                        intermediate_tables.add(from_t)
                    if to_t not in table_names:
                        intermediate_tables.add(to_t)
                    
                    # Build join dict
                    if r["rel_type"] == "JOIN":
                        all_joins.append({
                            "from_table": from_t,
                            "to_table": to_t,
                            "join_type": r.get("join_type") or "left",
                            "on_clause": r.get("on_clause") or [],
                            "description": r.get("description") or "",
                        })
                    else:  # FK relationship
                        fk_col = r.get("fk_column", "")
                        ref_col = r.get("fk_ref_column", "")
                        if fk_col and ref_col:
                            all_joins.append({
                                "from_table": from_t,
                                "to_table": to_t,
                                "join_type": "left",
                                "on_clause": [f"{from_t}.{fk_col} = {to_t}.{ref_col}"],
                                "description": "",
                            })
                            
            except Exception as e:
                logger.warning(f"[PATH] Path search failed {anchor_table} → {target_table}: {e}")
        
        if intermediate_tables:
            logger.info(f"[PATH] Intermediate tables discovered: {intermediate_tables}")
        
        logger.info(f"[PATH] Total joins on paths: {len(all_joins)}")
        return {"joins": all_joins, "intermediate_tables": intermediate_tables}
    
    def _keyword_search(
        self,
        keywords: List[str],
        domain: str,
    ) -> Dict[str, int]:
        """
        Search for tables matching keywords using Neo4j Fulltext Index.
        
        Uses fulltext search with fuzzy matching for better performance
        and typo tolerance compared to CONTAINS queries.
        
        Returns:
            table_name -> match_count mapping
        """
        if not keywords:
            return {}
        
        logger.info(f"[FULLTEXT] Searching with keywords: {keywords}")
        
        # Use fulltext search with fuzzy matching
        fulltext_results = self.vector_index.fulltext_search(
            search_terms=keywords,
            label=None,  # Search all labels
            top_k=30,
            fuzzy=True,
        )
        
        # Count matches per table
        table_match_counts: Dict[str, int] = {}
        
        for result in fulltext_results:
            label = result.get("label", "")
            props = result.get("props", {})
            score = result.get("score", 0)
            
            # Extract table name based on node type
            if label == "Table":
                table_name = props.get("table_name", "")
                # Filter by domain
                if props.get("domain") == domain:
                    table_match_counts[table_name] = table_match_counts.get(table_name, 0) + 1
            elif label == "Column":
                table_name = props.get("table_name", "")
                if table_name:  # Columns belong to tables
                    table_match_counts[table_name] = table_match_counts.get(table_name, 0) + 1
            elif label == "Concept":
                # Concepts may be linked to multiple tables - need to query
                # For now, skip concept matches in table counting
                pass
        
        logger.info(f"[FULLTEXT] Found {len(table_match_counts)} tables from fulltext search")
        return table_match_counts
    
    def _expand_context(
        self,
        table_names: Set[str],
        depth: int,
        relevant_columns: Set[tuple],
        domain: str,
        query_embedding: List[float] | None = None,
    ) -> Dict[str, Any]:
        """
        Expand context using graph traversal with smart column filtering.
        
        Optimized to only return relevant columns instead of all columns,
        reducing token usage significantly.
        """
        if not table_names:
            return {"tables": [], "columns": [], "joins": [], "metrics": []}
        
        table_names_list = list(table_names)
        
        # === STEP 1: Use path search to find optimal joins ===
        path_result = self._find_join_paths(table_names_list, domain, max_depth=depth + 2)
        joins = path_result["joins"]
        intermediate_tables = path_result["intermediate_tables"]
        
        # Expand table list with intermediate tables (needed for multi-hop joins)
        all_table_names = set(table_names_list) | intermediate_tables
        all_table_names_list = list(all_table_names)
        
        if intermediate_tables:
            logger.info(f"[EXPAND] Added intermediate tables: {intermediate_tables}")
        
        # === STEP 2: Get table metadata ===
        tables_query = """
        MATCH (t:Table)
        WHERE t.table_name IN $table_names
        RETURN t.table_name AS table_name,
               t.catalog AS catalog,
               t.schema AS schema,
               t.business_name AS business_name,
               t.table_type AS table_type,
               t.description AS description,
               t.grain AS grain,
               t.tags AS tags
        """
        tables = self.execute_query(tables_query, {"table_names": all_table_names_list})
        
        # === STEP 3: Smart Column Filtering ===
        columns = self._get_relevant_columns(
            table_names=all_table_names_list,
            relevant_columns=relevant_columns,
            query_embedding=query_embedding,
        )
        
        # === STEP 4: Fallback - if no paths found, use legacy join fetch ===
        if not joins and len(table_names_list) > 1:
            logger.warning("[EXPAND] No paths found, falling back to legacy join fetch")
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
            
            # Also get FK relationships
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
        
        # === STEP 5: Get metrics ===
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
        metrics = self.execute_query(metrics_query, {"table_names": all_table_names_list})
        
        return {
            "tables": tables,
            "columns": columns,
            "joins": joins,
            "metrics": metrics,
        }
    
    def _get_relevant_columns(
        self,
        table_names: List[str],
        relevant_columns: Set[tuple],
        query_embedding: List[float] | None = None,
        max_per_table: int = 10,
    ) -> List[Dict]:
        """
        Get only relevant columns instead of all columns.
        
        Strategy:
        1. Always include: PK, FK, Time columns (needed for joins/filters)
        2. Include columns matched by vector search
        3. If embedding provided, find additional relevant columns via vector search
        
        This significantly reduces tokens sent to LLM.
        
        Args:
            table_names: Tables to get columns for
            relevant_columns: Set of (table_name, column_name) from vector search
            query_embedding: Optional embedding for additional column search
            max_per_table: Maximum columns per table (excluding must-have)
            
        Returns:
            List of column dictionaries
        """
        logger.info(f"[SMART_COLUMNS] Getting relevant columns for {len(table_names)} tables")
        
        # === STEP 1: Get must-have columns (PK, FK, Time) ===
        must_have_query = """
        MATCH (t:Table)-[r:HAS_COLUMN]->(c:Column)
        WHERE t.table_name IN $table_names
          AND (r.primary_key = true OR r.foreign_key = true OR r.time_column = true)
        RETURN t.table_name AS table_name,
               c.column_name AS column_name,
               c.data_type AS data_type,
               c.business_name AS business_name,
               c.description AS description,
               c.semantics AS semantics,
               r.primary_key AS is_primary_key,
               r.foreign_key AS is_foreign_key,
               r.time_column AS is_time_column,
               'must_have' AS source
        """
        must_have_cols = self.execute_query(must_have_query, {"table_names": table_names})
        logger.info(f"[SMART_COLUMNS] Must-have columns (PK/FK/Time): {len(must_have_cols)}")
        
        # === STEP 2: Get vector-matched columns ===
        vector_matched_cols = []
        if relevant_columns:
            vector_query = """
            MATCH (t:Table)-[r:HAS_COLUMN]->(c:Column)
            WHERE t.table_name IN $table_names
            RETURN t.table_name AS table_name,
                   c.column_name AS column_name,
                   c.data_type AS data_type,
                   c.business_name AS business_name,
                   c.description AS description,
                   c.semantics AS semantics,
                   r.primary_key AS is_primary_key,
                   r.foreign_key AS is_foreign_key,
                   r.time_column AS is_time_column,
                   'vector_match' AS source
            """
            all_cols = self.execute_query(vector_query, {"table_names": table_names})
            for col in all_cols:
                if (col['table_name'], col['column_name']) in relevant_columns:
                    vector_matched_cols.append(col)
            logger.info(f"[SMART_COLUMNS] Vector-matched columns: {len(vector_matched_cols)}")
        
        # === STEP 3: Additional vector search on columns (if embedding provided) ===
        vector_search_cols = []
        if query_embedding:
            try:
                vector_results = self.vector_index._search_single_label(
                    query_embedding=query_embedding,
                    label="Column",
                    top_k=15,
                )
                for result in vector_results:
                    props = result.get("props", {})
                    table_name = props.get("table_name", "")
                    if table_name in table_names and result.get("score", 0) > 0.6:
                        vector_search_cols.append({
                            "table_name": table_name,
                            "column_name": props.get("column_name", ""),
                            "data_type": props.get("data_type", ""),
                            "business_name": props.get("business_name", ""),
                            "description": props.get("description", ""),
                            "semantics": props.get("semantics", []),
                            "is_primary_key": False,
                            "is_foreign_key": False,
                            "is_time_column": False,
                            "source": "vector_search",
                            "score": result.get("score", 0),
                        })
                logger.info(f"[SMART_COLUMNS] Vector search found: {len(vector_search_cols)} columns")
            except Exception as e:
                logger.warning(f"[SMART_COLUMNS] Vector column search failed: {e}")
        
        # === STEP 4: Include display/name columns ===
        # These are essential for GROUP BY and human-readable output
        # Include from both dimension tables AND denormalized columns in fact tables
        display_cols_query = """
        MATCH (t:Table)-[r:HAS_COLUMN]->(c:Column)
        WHERE t.table_name IN $table_names
          AND (
            // Dimension tables - get name/title/code columns
            ((t.table_type = 'dimension' OR t.table_type = 'dim') AND (
              c.column_name CONTAINS 'name' OR 
              c.column_name CONTAINS 'title' OR
              c.column_name CONTAINS 'code'
            ))
            OR
            // Any table - get denormalized display columns (cinema_name, vendor_name, etc.)
            c.column_name CONTAINS 'cinema_name' OR
            c.column_name CONTAINS 'vendor_name' OR
            c.column_name CONTAINS 'film_name' OR
            c.column_name CONTAINS 'bank_name' OR
            c.column_name CONTAINS 'status'
            OR
            // Semantic hints
            any(sem IN c.semantics WHERE sem IN ['name', 'title', 'label', 'display'])
          )
        RETURN t.table_name AS table_name,
               c.column_name AS column_name,
               c.data_type AS data_type,
               c.business_name AS business_name,
               c.description AS description,
               c.semantics AS semantics,
               false AS is_primary_key,
               false AS is_foreign_key,
               false AS is_time_column,
               'display_column' AS source
        """
        try:
            display_cols = self.execute_query(display_cols_query, {"table_names": table_names})
            logger.info(f"[SMART_COLUMNS] Display columns (dim + denormalized): {len(display_cols)}")
        except Exception as e:
            logger.warning(f"[SMART_COLUMNS] Display column query failed: {e}")
            display_cols = []
        
        # === STEP 5: Merge and deduplicate ===
        columns_dict = {}
        
        # Priority 1: Must-have columns
        for col in must_have_cols:
            key = (col['table_name'], col['column_name'])
            columns_dict[key] = col
        
        # Priority 2: Display columns (name_vi, etc.)
        for col in display_cols:
            key = (col['table_name'], col['column_name'])
            if key not in columns_dict:
                columns_dict[key] = col
        
        # Priority 3: Vector-matched columns
        for col in vector_matched_cols:
            key = (col['table_name'], col['column_name'])
            if key not in columns_dict:
                columns_dict[key] = col
        
        # Priority 4: Vector search columns (limit per table)
        table_counts = {}
        for col in sorted(vector_search_cols, key=lambda x: x.get("score", 0), reverse=True):
            key = (col['table_name'], col['column_name'])
            table_name = col['table_name']
            
            if key not in columns_dict:
                table_counts[table_name] = table_counts.get(table_name, 0) + 1
                if table_counts[table_name] <= max_per_table:
                    columns_dict[key] = col
        
        columns = sorted(columns_dict.values(), key=lambda x: (x['table_name'], x['column_name']))
        
        # Log summary
        total_possible = self._count_total_columns(table_names)
        savings = ((total_possible - len(columns)) / total_possible * 100) if total_possible > 0 else 0
        logger.info(f"[SMART_COLUMNS] ✅ Selected {len(columns)}/{total_possible} columns (~{savings:.0f}% reduction)")
        
        return columns
    
    def _count_total_columns(self, table_names: List[str]) -> int:
        """Count total columns in tables for statistics."""
        query = """
        MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
        WHERE t.table_name IN $table_names
        RETURN count(c) AS total
        """
        result = self.execute_query(query, {"table_names": table_names})
        return result[0]["total"] if result else 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
