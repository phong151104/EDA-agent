"""
Neo4j Graph RAG module.

Provides schema retrieval using hybrid vector + graph search.
Adapted from text2sql project architecture.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase, Driver

from config import config

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j database client with connection management.
    
    Provides methods for executing Cypher queries and managing connections.
    """
    
    def __init__(self):
        """Initialize Neo4j client from config."""
        self._driver: Driver | None = None
    
    @property
    def driver(self) -> Driver:
        """Get or create Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                config.neo4j.uri,
                auth=(config.neo4j.user, config.neo4j.password),
            )
            logger.info(f"Connected to Neo4j at {config.neo4j.uri}")
        return self._driver
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Closed Neo4j connection")
    
    def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        with self.driver.session(database=config.neo4j.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Execute a write query."""
        with self.driver.session(database=config.neo4j.database) as session:
            session.execute_write(
                lambda tx: tx.run(query, parameters or {})
            )
    
    def __enter__(self) -> "Neo4jClient":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class GraphRAG:
    """
    Graph-based Retrieval Augmented Generation.
    
    Uses Neo4j to retrieve relevant schema information
    based on semantic similarity and graph relationships.
    """
    
    def __init__(self, client: Neo4jClient | None = None):
        """
        Initialize GraphRAG.
        
        Args:
            client: Optional Neo4j client (creates new if not provided)
        """
        self.client = client or Neo4jClient()
    
    async def retrieve_context(
        self,
        question: str,
        domain: str | None = None,
        top_k: int = 10,
        expand_depth: int = 2,
    ) -> dict[str, Any]:
        """
        Retrieve relevant schema context for a question.
        
        Uses hybrid approach:
        1. Vector search to find semantically relevant nodes
        2. Graph traversal to expand context with related nodes
        
        Args:
            question: Natural language question
            domain: Optional domain filter
            top_k: Number of initial vector search results
            expand_depth: Depth of graph expansion
            
        Returns:
            Context with tables, columns, joins, metrics
        """
        # Step 1: Vector search for relevant tables/columns
        vector_results = await self._vector_search(question, top_k)
        
        # Step 2: Extract relevant entity names
        relevant_tables = self._extract_tables(vector_results)
        relevant_columns = self._extract_columns(vector_results)
        
        # Step 3: Expand context via graph traversal
        expanded_context = await self._expand_context(
            relevant_tables,
            relevant_columns,
            expand_depth,
        )
        
        return {
            "tables": expanded_context.get("tables", []),
            "columns": expanded_context.get("columns", []),
            "joins": expanded_context.get("joins", []),
            "metrics": expanded_context.get("metrics", []),
            "concepts": expanded_context.get("concepts", []),
        }
    
    async def _vector_search(
        self,
        query_text: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Perform vector similarity search.
        
        TODO: Implement actual vector search using Neo4j vector index.
        """
        # Placeholder - actual implementation uses vector index
        logger.debug(f"Vector search for: {query_text[:50]}...")
        return []
    
    def _extract_tables(
        self,
        vector_results: list[dict[str, Any]],
    ) -> set[str]:
        """Extract table names from vector search results."""
        tables = set()
        for result in vector_results:
            if "table_name" in result:
                tables.add(result["table_name"])
            elif "labels" in result and "Table" in result.get("labels", []):
                tables.add(result.get("name", ""))
        return tables
    
    def _extract_columns(
        self,
        vector_results: list[dict[str, Any]],
    ) -> set[tuple[str, str]]:
        """Extract (table_name, column_name) pairs."""
        columns = set()
        for result in vector_results:
            if "column_name" in result and "table_name" in result:
                columns.add((result["table_name"], result["column_name"]))
        return columns
    
    async def _expand_context(
        self,
        table_names: set[str],
        relevant_columns: set[tuple[str, str]],
        depth: int,
    ) -> dict[str, Any]:
        """
        Expand context using graph traversal.
        
        Gets table details, related joins, and metrics.
        """
        if not table_names:
            return {}
        
        # Get table details
        tables_query = """
        MATCH (t:Table)
        WHERE t.table_name IN $table_names
        OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
        RETURN t, collect(c) as columns
        """
        
        tables_result = self.client.execute_query(
            tables_query,
            {"table_names": list(table_names)}
        )
        
        tables = []
        columns = []
        
        for record in tables_result:
            table = record.get("t", {})
            tables.append({
                "name": table.get("table_name"),
                "business_name": table.get("business_name"),
                "description": table.get("description"),
                "grain": table.get("grain"),
            })
            
            for col in record.get("columns", []):
                columns.append({
                    "table": table.get("table_name"),
                    "name": col.get("column_name"),
                    "type": col.get("data_type"),
                    "description": col.get("description"),
                })
        
        # Get joins between tables
        joins_query = """
        MATCH (t1:Table)-[j:JOINS]->(t2:Table)
        WHERE t1.table_name IN $table_names OR t2.table_name IN $table_names
        RETURN t1.table_name as from_table, 
               t2.table_name as to_table,
               j.join_type as join_type,
               j.on as on_clause
        """
        
        joins_result = self.client.execute_query(
            joins_query,
            {"table_names": list(table_names)}
        )
        
        joins = [
            {
                "from": r["from_table"],
                "to": r["to_table"],
                "type": r.get("join_type", "INNER"),
                "on": r.get("on_clause", []),
            }
            for r in joins_result
        ]
        
        # Get metrics
        metrics_query = """
        MATCH (m:Metric)-[:METRIC_BASE_TABLE]->(t:Table)
        WHERE t.table_name IN $table_names
        RETURN m
        """
        
        metrics_result = self.client.execute_query(
            metrics_query,
            {"table_names": list(table_names)}
        )
        
        metrics = [
            {
                "name": r["m"].get("name"),
                "business_name": r["m"].get("business_name"),
                "expression": r["m"].get("expression"),
                "description": r["m"].get("description"),
            }
            for r in metrics_result
        ]
        
        return {
            "tables": tables,
            "columns": columns,
            "joins": joins,
            "metrics": metrics,
        }
    
    async def get_full_schema(self, domain: str | None = None) -> dict[str, Any]:
        """
        Get the complete schema for a domain.
        
        Args:
            domain: Domain name filter
            
        Returns:
            Complete schema information
        """
        query = """
        MATCH (t:Table)
        WHERE $domain IS NULL OR t.domain = $domain
        OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
        RETURN t, collect(c) as columns
        ORDER BY t.table_name
        """
        
        results = self.client.execute_query(query, {"domain": domain})
        
        tables = []
        for record in results:
            table = record.get("t", {})
            columns = [
                {
                    "name": c.get("column_name"),
                    "type": c.get("data_type"),
                    "description": c.get("description"),
                }
                for c in record.get("columns", [])
            ]
            tables.append({
                "name": table.get("table_name"),
                "columns": columns,
            })
        
        return {"tables": tables}
