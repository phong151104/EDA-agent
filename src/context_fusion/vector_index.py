"""
Neo4j Vector Index management and operations.
"""

from __future__ import annotations

import logging
from typing import Any, List, Dict

from neo4j import GraphDatabase, Driver

from config import config
from src.embeddings import OpenAIEmbedder, NodeTextBuilder

logger = logging.getLogger(__name__)


class Neo4jVectorIndex:
    """
    Manages Neo4j vector indexes for semantic search.
    
    Creates embeddings for graph nodes and stores them in Neo4j's
    native vector index for efficient similarity search.
    """
    
    # Node labels to index
    INDEXED_LABELS = ["Table", "Column", "Concept", "Metric"]
    
    # Fulltext index configuration
    FULLTEXT_INDEX_NAME = "schema_fulltext_search"
    
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        embedder: OpenAIEmbedder | None = None,
    ):
        self.uri = uri or config.neo4j.uri
        self.user = user or config.neo4j.user
        self.password = password or config.neo4j.password
        self._driver: Driver | None = None
        self.embedder = embedder or OpenAIEmbedder()
        self.index_name = config.vector_index.index_name
        self.similarity_function = config.vector_index.similarity_function
    
    @property
    def driver(self) -> Driver:
        """Get or create Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            logger.info(f"Connected to Neo4j at {self.uri}")
        return self._driver
    
    def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def execute_query(self, query: str, params: dict = None) -> List[Dict]:
        """Execute a read query and return results."""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def execute_write(self, query: str, params: dict = None) -> None:
        """Execute a write query."""
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, params or {}))
    
    def create_vector_index(self, label: str) -> None:
        """
        Create a vector index for a specific node label.
        
        Args:
            label: Node label (Table, Column, etc.)
        """
        index_name = f"{self.index_name}_{label.lower()}"
        
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{label})
        ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedder.dimensions},
                `vector.similarity_function`: '{self.similarity_function}'
            }}
        }}
        """
        
        try:
            self.execute_write(query)
            logger.info(f"Created vector index: {index_name}")
        except Exception as e:
            logger.warning(f"Vector index creation note: {e}")
    
    def create_all_indexes(self) -> None:
        """Create vector indexes for all indexed labels."""
        for label in self.INDEXED_LABELS:
            self.create_vector_index(label)
    
    def drop_vector_index(self, label: str) -> None:
        """Drop a vector index for a specific node label."""
        index_name = f"{self.index_name}_{label.lower()}"
        
        query = f"DROP INDEX {index_name} IF EXISTS"
        self.execute_write(query)
        logger.info(f"Dropped vector index: {index_name}")
    
    # =========================================================================
    # Fulltext Index (for keyword search with fuzzy matching)
    # =========================================================================
    
    def create_fulltext_index(self) -> None:
        """
        Create fulltext index for text-based search.
        
        This enables fuzzy matching and is faster than CONTAINS queries.
        """
        # Note: Neo4j fulltext index syntax requires creating one index per label combination
        # We create a single index covering the most important searchable fields
        queries = [
            # Index for Table nodes
            f"""
            CREATE FULLTEXT INDEX {self.FULLTEXT_INDEX_NAME}_table IF NOT EXISTS
            FOR (n:Table)
            ON EACH [n.table_name, n.business_name, n.description]
            """,
            # Index for Column nodes  
            f"""
            CREATE FULLTEXT INDEX {self.FULLTEXT_INDEX_NAME}_column IF NOT EXISTS
            FOR (n:Column)
            ON EACH [n.column_name, n.business_name, n.description]
            """,
            # Index for Concept nodes
            f"""
            CREATE FULLTEXT INDEX {self.FULLTEXT_INDEX_NAME}_concept IF NOT EXISTS
            FOR (n:Concept)
            ON EACH [n.name, n.synonyms]
            """,
        ]
        
        for query in queries:
            try:
                self.execute_write(query)
                logger.info(f"Created fulltext index")
            except Exception as e:
                logger.warning(f"Fulltext index creation note: {e}")
    
    def fulltext_search(
        self,
        search_terms: List[str],
        label: str | None = None,
        top_k: int = 20,
        fuzzy: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform fulltext search with optional fuzzy matching.
        
        Args:
            search_terms: List of terms to search for
            label: Restrict search to specific label (Table, Column, Concept)
            top_k: Maximum results to return
            fuzzy: Enable fuzzy matching (~)
            
        Returns:
            List of matching nodes with scores
        """
        if not search_terms:
            return []
        
        # Build search string with OR and optional fuzzy
        if fuzzy:
            # Fuzzy search: "term~" allows 1-2 character edits
            search_text = " OR ".join([f"{term}~" for term in search_terms])
        else:
            search_text = " OR ".join(search_terms)
        
        # Determine which index to use
        if label:
            index_name = f"{self.FULLTEXT_INDEX_NAME}_{label.lower()}"
        else:
            # Search all indexes and combine
            return self._fulltext_search_all(search_terms, top_k, fuzzy)
        
        query = """
        CALL db.index.fulltext.queryNodes($index_name, $search_text)
        YIELD node, score
        RETURN labels(node)[0] AS label, 
               properties(node) AS props, 
               score
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        try:
            return self.execute_query(query, {
                "index_name": index_name,
                "search_text": search_text,
                "top_k": top_k,
            })
        except Exception as e:
            logger.warning(f"Fulltext search failed for {label}: {e}")
            return []
    
    def _fulltext_search_all(
        self,
        search_terms: List[str],
        top_k: int,
        fuzzy: bool,
    ) -> List[Dict[str, Any]]:
        """Search all fulltext indexes and combine results."""
        all_results = []
        
        for label in ["table", "column", "concept"]:
            results = self.fulltext_search(search_terms, label=label, top_k=top_k, fuzzy=fuzzy)
            all_results.extend(results)
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:top_k]
    
    def generate_and_store_embeddings(
        self,
        label: str,
        batch_size: int = 50,
    ) -> int:
        """
        Generate embeddings for all nodes of a label and store in Neo4j.
        
        Args:
            label: Node label to process
            batch_size: Number of nodes to process per batch
            
        Returns:
            Number of nodes processed
        """
        logger.info(f"Generating embeddings for {label} nodes...")
        
        # Fetch all nodes of this label
        query = f"""
        MATCH (n:{label})
        RETURN elementId(n) AS node_id, properties(n) AS props
        """
        results = self.execute_query(query)
        
        if not results:
            logger.info(f"No {label} nodes found")
            return 0
        
        # Build texts for embedding
        node_ids = []
        texts = []
        
        for record in results:
            node_id = record["node_id"]
            props = record["props"]
            
            text = NodeTextBuilder.build_text(props, label)
            if text.strip():
                node_ids.append(node_id)
                texts.append(text)
        
        logger.info(f"Embedding {len(texts)} {label} nodes...")
        
        # Generate embeddings
        embeddings = self.embedder.embed_texts(texts, batch_size=100)
        
        # Store embeddings in batches
        for i in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            
            self._store_embeddings_batch(batch_ids, batch_embeddings, batch_texts)
        
        logger.info(f"Stored embeddings for {len(node_ids)} {label} nodes")
        return len(node_ids)
    
    def _store_embeddings_batch(
        self,
        node_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
    ) -> None:
        """Store a batch of embeddings in Neo4j."""
        query = """
        UNWIND $batch AS item
        MATCH (n) WHERE elementId(n) = item.node_id
        SET n.embedding = item.embedding,
            n.embedding_text = item.text
        """
        
        batch = [
            {"node_id": nid, "embedding": emb, "text": txt}
            for nid, emb, txt in zip(node_ids, embeddings, texts)
        ]
        
        self.execute_write(query, {"batch": batch})
    
    def index_all_nodes(self) -> Dict[str, int]:
        """
        Create indexes and generate embeddings for all node types.
        
        Returns:
            Dictionary of label -> count of nodes processed
        """
        # Create indexes first
        self.create_all_indexes()
        
        # Generate and store embeddings
        counts = {}
        for label in self.INDEXED_LABELS:
            count = self.generate_and_store_embeddings(label)
            counts[label] = count
        
        logger.info(f"Indexed all nodes: {counts}")
        return counts
    
    def vector_search(
        self,
        query_text: str,
        label: str | None = None,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query_text: Query text to search for
            label: Optional label to restrict search
            top_k: Number of results to return
            
        Returns:
            List of matching nodes with scores
        """
        top_k = top_k or config.vector_index.top_k
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query_text)
        
        if label:
            # Search specific label
            return self._search_single_label(query_embedding, label, top_k)
        else:
            # Search all labels and combine results
            return self._search_all_labels(query_embedding, top_k)
    
    def _search_single_label(
        self,
        query_embedding: List[float],
        label: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search a single label's vector index."""
        index_name = f"{self.index_name}_{label.lower()}"
        
        query = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        RETURN 
            elementId(node) AS node_id,
            labels(node)[0] AS label,
            properties(node) AS props,
            score
        ORDER BY score DESC
        """
        
        return self.execute_query(query, {
            "index_name": index_name,
            "top_k": top_k,
            "embedding": query_embedding,
        })
    
    def _search_all_labels(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search all label indexes and combine results."""
        all_results = []
        results_by_label = {}
        
        for label in self.INDEXED_LABELS:
            try:
                results = self._search_single_label(query_embedding, label, top_k)
                results_by_label[label] = len(results)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Vector search failed for {label}: {e}")
                results_by_label[label] = 0
        
        # Sort by score and take top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Log summary
        summary = ", ".join([f"{k}: {v}" for k, v in results_by_label.items()])
        logger.info(f"[Vector] Searched {len(self.INDEXED_LABELS)} indexes ({summary})")
        
        return all_results[:top_k]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
