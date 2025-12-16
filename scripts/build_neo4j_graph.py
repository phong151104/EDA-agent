#!/usr/bin/env python3
"""
build_neo4j_graph.py

Reads YAML metadata files and builds a Neo4j knowledge graph for Text-to-SQL.

Usage:
    python build_neo4j_graph.py --domain vnfilm_ticketing --metadata-root metadata/domains
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import yaml
from neo4j import GraphDatabase, Driver, ManagedTransaction

load_dotenv(override=True)  # Force reload

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ForeignKey:
    """Represents a foreign key relationship."""
    column: str
    references_table: str
    references_column: str
    relation: str
    description: str


@dataclass
class Concept:
    """Represents a business concept with synonyms."""
    name: str
    synonyms: list[str] = field(default_factory=list)


@dataclass
class Column:
    """Represents a table column."""
    table_name: str
    column_name: str
    data_type: str
    business_name: str | None = None
    description: str = ""
    semantics: list[str] = field(default_factory=list)
    unit: str | None = None
    pii: bool = False
    sensitive: bool = False


@dataclass
class Table:
    """Represents a database table."""
    catalog: str
    schema: str
    table_name: str
    domain: str
    table_type: str
    business_name: str
    grain: str
    description: str
    tags: list[str] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    time_columns: list[str] = field(default_factory=list)
    recommended_filters: list[str] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)
    columns: list[Column] = field(default_factory=list)
    sample_questions: list[str] = field(default_factory=list)


@dataclass
class Join:
    """Represents a join relationship between tables."""
    from_table: str
    to_table: str
    join_type: str
    on: list[str]
    description: str


@dataclass
class Metric:
    """Represents a business metric."""
    name: str
    business_name: str
    description: str
    expression: str
    base_table: str
    grain: str
    unit: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class DomainMetadata:
    """Container for all metadata in a domain."""
    domain: str
    tables: list[Table] = field(default_factory=list)
    joins: list[Join] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    concepts: dict[str, Concept] = field(default_factory=dict)


# =============================================================================
# YAML Loader
# =============================================================================

class MetadataLoader:
    """Loads metadata from YAML files."""

    def __init__(self, metadata_root: Path, domain: str):
        self.metadata_root = metadata_root
        self.domain = domain
        self.domain_path = metadata_root / domain

    def load(self) -> DomainMetadata:
        """Load all metadata for the domain."""
        logger.info(f"Loading metadata from: {self.domain_path}")

        if not self.domain_path.exists():
            raise FileNotFoundError(f"Domain path not found: {self.domain_path}")

        metadata = DomainMetadata(domain=self.domain)

        # Load tables
        tables_path = self.domain_path / "tables"
        if tables_path.exists():
            metadata.tables = self._load_tables(tables_path)
            logger.info(f"Loaded {len(metadata.tables)} tables")

        # Extract all concepts from tables
        for table in metadata.tables:
            for concept in table.concepts:
                if concept.name not in metadata.concepts:
                    metadata.concepts[concept.name] = concept

        # Also extract concepts from column semantics
        for table in metadata.tables:
            for column in table.columns:
                for semantic in column.semantics:
                    if semantic not in metadata.concepts:
                        # Create a concept for semantic tags that don't have explicit concepts
                        metadata.concepts[semantic] = Concept(name=semantic, synonyms=[])

        logger.info(f"Extracted {len(metadata.concepts)} unique concepts")

        # Load joins
        joins_file = self.domain_path / "joins.yaml"
        if joins_file.exists():
            metadata.joins = self._load_joins(joins_file)
            logger.info(f"Loaded {len(metadata.joins)} joins")

        # Load metrics
        metrics_file = self.domain_path / "metrics.yaml"
        if metrics_file.exists():
            metadata.metrics = self._load_metrics(metrics_file)
            logger.info(f"Loaded {len(metadata.metrics)} metrics")

        return metadata

    def _load_tables(self, tables_path: Path) -> list[Table]:
        """Load all table YAML files."""
        tables = []

        for yaml_file in tables_path.glob("*.yaml"):
            tables.append(self._parse_table_file(yaml_file))

        for yml_file in tables_path.glob("*.yml"):
            tables.append(self._parse_table_file(yml_file))

        return tables

    def _parse_table_file(self, file_path: Path) -> Table:
        """Parse a single table YAML file."""
        logger.debug(f"Parsing table file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Parse foreign keys
        foreign_keys = []
        for fk_data in data.get("foreign_keys", []) or []:
            foreign_keys.append(ForeignKey(
                column=fk_data.get("column", ""),
                references_table=fk_data.get("references_table", ""),
                references_column=fk_data.get("references_column", ""),
                relation=fk_data.get("relation", ""),
                description=fk_data.get("description", "")
            ))

        # Parse concepts
        concepts = []
        for concept_data in data.get("concepts", []) or []:
            concepts.append(Concept(
                name=concept_data.get("name", ""),
                synonyms=concept_data.get("synonyms", []) or []
            ))

        # Parse columns
        columns = []
        columns_data = data.get("columns", {}) or {}
        table_name = data.get("table_name", "")

        for col_name, col_data in columns_data.items():
            if col_data is None:
                col_data = {}
            columns.append(Column(
                table_name=table_name,
                column_name=col_name,
                data_type=col_data.get("data_type", "unknown"),
                business_name=col_data.get("business_name"),
                description=col_data.get("description", ""),
                semantics=col_data.get("semantics", []) or [],
                unit=col_data.get("unit"),
                pii=col_data.get("pii", False) or False,
                sensitive=col_data.get("sensitive", False) or False
            ))

        return Table(
            catalog=data.get("catalog", ""),
            schema=data.get("schema", ""),
            table_name=table_name,
            domain=data.get("domain", self.domain),
            table_type=data.get("table_type", ""),
            business_name=data.get("business_name", ""),
            grain=data.get("grain", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []) or [],
            primary_key=data.get("primary_key", []) or [],
            foreign_keys=foreign_keys,
            time_columns=data.get("time_columns", []) or [],
            recommended_filters=data.get("recommended_filters", []) or [],
            concepts=concepts,
            columns=columns,
            sample_questions=data.get("sample_questions", []) or []
        )

    def _load_joins(self, joins_file: Path) -> list[Join]:
        """Load joins from YAML file."""
        with open(joins_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        joins = []
        for join_data in data.get("joins", []) or []:
            joins.append(Join(
                from_table=join_data.get("from", ""),
                to_table=join_data.get("to", ""),
                join_type=join_data.get("type", "inner"),
                on=join_data.get("on", []) or [],
                description=join_data.get("description", "")
            ))

        return joins

    def _load_metrics(self, metrics_file: Path) -> list[Metric]:
        """Load metrics from YAML file."""
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        metrics = []
        for metric_data in data.get("metrics", []) or []:
            metrics.append(Metric(
                name=metric_data.get("name", ""),
                business_name=metric_data.get("business_name", ""),
                description=metric_data.get("description", ""),
                expression=metric_data.get("expression", ""),
                base_table=metric_data.get("base_table", ""),
                grain=metric_data.get("grain", ""),
                unit=metric_data.get("unit"),
                tags=metric_data.get("tags", []) or []
            ))

        return metrics


# =============================================================================
# Neo4j Graph Builder
# =============================================================================

class Neo4jGraphBuilder:
    """Builds the knowledge graph in Neo4j."""

    BATCH_SIZE = 100

    def __init__(self, uri: str, user: str, password: str):
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self) -> None:
        """Close the Neo4j connection."""
        self.driver.close()
        logger.info("Closed Neo4j connection")

    def build_graph(self, metadata: DomainMetadata) -> None:
        """Build the complete knowledge graph for the domain."""
        logger.info(f"Building graph for domain: {metadata.domain}")

        with self.driver.session() as session:
            # Step 1: Clear existing data for this domain
            self._clear_domain(session, metadata.domain)

            # Step 2: Create constraints
            self._create_constraints(session)

            # Step 3: Create Table nodes
            self._create_table_nodes(session, metadata)

            # Step 4: Create Column nodes and HAS_COLUMN relationships
            self._create_column_nodes(session, metadata)

            # Step 5: Create Concept nodes
            self._create_concept_nodes(session, metadata)

            # Step 6: Create Metric nodes
            self._create_metric_nodes(session, metadata)

            # Step 7: Create JOIN relationships
            self._create_join_relationships(session, metadata)

            # Step 8: Create FK relationships
            self._create_fk_relationships(session, metadata)

            # Step 9: Create HAS_CONCEPT relationships (table -> concept)
            self._create_table_concept_relationships(session, metadata)

            # Step 10: Create HAS_SEMANTIC relationships (column -> concept)
            self._create_column_semantic_relationships(session, metadata)

            # Step 11: Create METRIC_BASE_TABLE and HAS_METRIC relationships
            self._create_metric_relationships(session, metadata)

        logger.info("Graph build complete!")

    def _clear_domain(self, session: Any, domain: str) -> None:
        """Clear all nodes for the given domain."""
        logger.info(f"Clearing existing data for domain: {domain}")

        # Get all table names in this domain first
        result = session.run("""
            MATCH (t:Table {domain: $domain})
            RETURN t.table_name AS table_name
        """, domain=domain)
        table_names = [record["table_name"] for record in result]
        logger.info(f"Found {len(table_names)} tables to clear: {table_names}")

        # Delete all columns belonging to tables in this domain
        if table_names:
            session.execute_write(
                lambda tx: tx.run("""
                    MATCH (c:Column)
                    WHERE c.table_name IN $table_names
                    DETACH DELETE c
                """, table_names=table_names)
            )

        # Delete all tables in this domain
        session.execute_write(
            lambda tx: tx.run("""
                MATCH (t:Table {domain: $domain})
                DETACH DELETE t
            """, domain=domain)
        )

        # Delete all metrics with base_table in this domain's tables
        if table_names:
            session.execute_write(
                lambda tx: tx.run("""
                    MATCH (m:Metric)
                    WHERE m.base_table IN $table_names
                    DETACH DELETE m
                """, table_names=table_names)
            )

        # Clean up orphaned Concept nodes (those with no relationships)
        session.execute_write(
            lambda tx: tx.run("""
                MATCH (c:Concept)
                WHERE NOT (c)<-[:HAS_CONCEPT]-() AND NOT (c)<-[:HAS_SEMANTIC]-()
                DELETE c
            """)
        )

        logger.info("Domain data cleared")

    def _create_constraints(self, session: Any) -> None:
        """Create uniqueness constraints."""
        logger.info("Creating constraints...")

        constraints = [
            # Table constraint: unique on (domain, table_name)
            """
            CREATE CONSTRAINT table_unique IF NOT EXISTS
            FOR (t:Table) REQUIRE (t.domain, t.table_name) IS UNIQUE
            """,
            # Column constraint: unique on (table_name, column_name)
            """
            CREATE CONSTRAINT column_unique IF NOT EXISTS
            FOR (c:Column) REQUIRE (c.table_name, c.column_name) IS UNIQUE
            """,
            # Metric constraint: unique on name
            """
            CREATE CONSTRAINT metric_unique IF NOT EXISTS
            FOR (m:Metric) REQUIRE m.name IS UNIQUE
            """,
            # Concept constraint: unique on name
            """
            CREATE CONSTRAINT concept_unique IF NOT EXISTS
            FOR (k:Concept) REQUIRE k.name IS UNIQUE
            """
        ]

        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception as e:
                # Constraint may already exist, or syntax may differ by Neo4j version
                logger.warning(f"Constraint creation note: {e}")

        logger.info("Constraints created/verified")
        
        # Create fulltext indexes for hybrid search
        self._create_fulltext_indexes(session)
    
    def _create_fulltext_indexes(self, session: Any) -> None:
        """
        Create fulltext indexes for hybrid search.
        
        Fulltext indexes enable:
        - Fuzzy matching (typo tolerance)
        - Multi-field search (table_name + business_name + description)
        - Fast keyword search without regex
        
        How it works:
        1. Neo4j creates an inverted index from text fields
        2. When querying, it tokenizes the search term and finds matches
        3. Supports wildcards (*) and fuzzy search (~)
        
        Example query:
            CALL db.index.fulltext.queryNodes('schema_fulltext_search_table', 'order* OR revenue~')
            YIELD node, score
            RETURN node.table_name, score
        """
        logger.info("Creating fulltext indexes for hybrid search...")
        
        fulltext_indexes = [
            # Fulltext index for Table nodes
            # Searches: table_name, business_name, description
            """
            CREATE FULLTEXT INDEX schema_fulltext_search_table IF NOT EXISTS
            FOR (t:Table)
            ON EACH [t.table_name, t.business_name, t.description]
            """,
            
            # Fulltext index for Column nodes
            # Searches: column_name, business_name, description
            """
            CREATE FULLTEXT INDEX schema_fulltext_search_column IF NOT EXISTS
            FOR (c:Column)
            ON EACH [c.column_name, c.business_name, c.description]
            """,
            
            # Fulltext index for Concept nodes
            # Searches: name (synonyms are in a separate property)
            """
            CREATE FULLTEXT INDEX schema_fulltext_search_concept IF NOT EXISTS
            FOR (c:Concept)
            ON EACH [c.name]
            """,
        ]
        
        for index_query in fulltext_indexes:
            try:
                session.run(index_query)
                # Extract index name for logging
                index_name = index_query.split("INDEX ")[1].split(" IF")[0].strip()
                logger.info(f"  ✅ Created fulltext index: {index_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"  ⏭️ Fulltext index already exists")
                else:
                    logger.warning(f"  ⚠️ Fulltext index creation note: {e}")
        
        logger.info("Fulltext indexes created/verified")

    def _create_table_nodes(self, session: Any, metadata: DomainMetadata) -> None:
        """Create Table nodes."""
        logger.info(f"Creating {len(metadata.tables)} Table nodes...")

        def create_tables(tx: ManagedTransaction, tables: list[dict[str, Any]]) -> None:
            tx.run("""
                UNWIND $tables AS t
                MERGE (table:Table {domain: t.domain, table_name: t.table_name})
                SET table.name = t.table_name,
                    table.catalog = t.catalog,
                    table.schema = t.schema,
                    table.table_type = t.table_type,
                    table.business_name = t.business_name,
                    table.grain = t.grain,
                    table.description = t.description,
                    table.tags = t.tags
            """, tables=tables)

        table_data = [
            {
                "catalog": t.catalog,
                "schema": t.schema,
                "table_name": t.table_name,
                "domain": t.domain,
                "table_type": t.table_type,
                "business_name": t.business_name,
                "grain": t.grain,
                "description": t.description,
                "tags": t.tags
            }
            for t in metadata.tables
        ]

        # Batch create
        for i in range(0, len(table_data), self.BATCH_SIZE):
            batch = table_data[i:i + self.BATCH_SIZE]
            session.execute_write(create_tables, batch)

        logger.info("Table nodes created")

    def _create_column_nodes(self, session: Any, metadata: DomainMetadata) -> None:
        """Create Column nodes and HAS_COLUMN relationships."""
        all_columns = []

        for table in metadata.tables:
            pk_set = set(table.primary_key)
            time_set = set(table.time_columns)
            # Build FK column set from foreign_keys
            fk_set = {fk.column for fk in table.foreign_keys}

            for col in table.columns:
                all_columns.append({
                    "table_name": col.table_name,
                    "column_name": col.column_name,
                    "data_type": col.data_type,
                    "business_name": col.business_name or "",
                    "description": col.description,
                    "semantics": col.semantics,
                    "unit": col.unit or "",
                    "pii": col.pii,
                    "sensitive": col.sensitive,
                    "domain": table.domain,
                    "is_primary_key": col.column_name in pk_set,
                    "is_time_column": col.column_name in time_set,
                    "is_foreign_key": col.column_name in fk_set,  # NEW
                })

        logger.info(f"Creating {len(all_columns)} Column nodes...")

        def create_columns(tx: ManagedTransaction, columns: list[dict[str, Any]]) -> None:
            tx.run("""
                UNWIND $columns AS c
                CREATE (col:Column {table_name: c.table_name, column_name: c.column_name})
                SET col.name = c.column_name,
                    col.data_type = c.data_type,
                    col.business_name = c.business_name,
                    col.description = c.description,
                    col.semantics = c.semantics,
                    col.unit = c.unit,
                    col.pii = c.pii,
                    col.sensitive = c.sensitive
                WITH col, c
                MATCH (t:Table {domain: c.domain, table_name: c.table_name})
                MERGE (t)-[r:HAS_COLUMN]->(col)
                SET r.primary_key = c.is_primary_key,
                    r.time_column = c.is_time_column,
                    r.foreign_key = c.is_foreign_key
            """, columns=columns)

        for i in range(0, len(all_columns), self.BATCH_SIZE):
            batch = all_columns[i:i + self.BATCH_SIZE]
            session.execute_write(create_columns, batch)

        logger.info("Column nodes and HAS_COLUMN relationships created")

    def _create_concept_nodes(self, session: Any, metadata: DomainMetadata) -> None:
        """Create Concept nodes."""
        logger.info(f"Creating {len(metadata.concepts)} Concept nodes...")

        def create_concepts(tx: ManagedTransaction, concepts: list[dict[str, Any]]) -> None:
            tx.run("""
                UNWIND $concepts AS c
                MERGE (concept:Concept {name: c.name})
                SET concept.synonyms = c.synonyms
            """, concepts=concepts)

        concept_data = [
            {"name": c.name, "synonyms": c.synonyms}
            for c in metadata.concepts.values()
        ]

        for i in range(0, len(concept_data), self.BATCH_SIZE):
            batch = concept_data[i:i + self.BATCH_SIZE]
            session.execute_write(create_concepts, batch)

        logger.info("Concept nodes created")

    def _create_metric_nodes(self, session: Any, metadata: DomainMetadata) -> None:
        """Create Metric nodes."""
        logger.info(f"Creating {len(metadata.metrics)} Metric nodes...")

        def create_metrics(tx: ManagedTransaction, metrics: list[dict[str, Any]]) -> None:
            tx.run("""
                UNWIND $metrics AS m
                MERGE (metric:Metric {name: m.name})
                SET metric.business_name = m.business_name,
                    metric.description = m.description,
                    metric.expression = m.expression,
                    metric.base_table = m.base_table,
                    metric.grain = m.grain,
                    metric.unit = m.unit,
                    metric.tags = m.tags
            """, metrics=metrics)

        metric_data = [
            {
                "name": m.name,
                "business_name": m.business_name,
                "description": m.description,
                "expression": m.expression,
                "base_table": m.base_table,
                "grain": m.grain,
                "unit": m.unit or "",
                "tags": m.tags
            }
            for m in metadata.metrics
        ]

        for i in range(0, len(metric_data), self.BATCH_SIZE):
            batch = metric_data[i:i + self.BATCH_SIZE]
            session.execute_write(create_metrics, batch)

        logger.info("Metric nodes created")

    def _create_join_relationships(self, session: Any, metadata: DomainMetadata) -> None:
        """Create JOIN relationships between tables."""
        logger.info(f"Creating {len(metadata.joins)} JOIN relationships...")

        def create_joins(tx: ManagedTransaction, joins: list[dict[str, Any]], domain: str) -> None:
            tx.run("""
                UNWIND $joins AS j
                MATCH (from_table:Table {domain: $domain, table_name: j.from_table})
                MATCH (to_table:Table {domain: $domain, table_name: j.to_table})
                MERGE (from_table)-[r:JOIN]->(to_table)
                SET r.join_type = j.join_type,
                    r.on = j.on,
                    r.description = j.description
            """, joins=joins, domain=domain)

        join_data = [
            {
                "from_table": j.from_table,
                "to_table": j.to_table,
                "join_type": j.join_type,
                "on": j.on,
                "description": j.description
            }
            for j in metadata.joins
        ]

        for i in range(0, len(join_data), self.BATCH_SIZE):
            batch = join_data[i:i + self.BATCH_SIZE]
            session.execute_write(create_joins, batch, metadata.domain)

        logger.info("JOIN relationships created")

    def _create_fk_relationships(self, session: Any, metadata: DomainMetadata) -> None:
        """Create FK relationships based on foreign_keys in tables."""
        fk_data = []

        for table in metadata.tables:
            for fk in table.foreign_keys:
                # Parse references_table to get just the table name
                # e.g., "lakehouse.lh_vnfilm_v2.bank" -> "bank"
                ref_table = fk.references_table
                if "." in ref_table:
                    ref_table = ref_table.split(".")[-1]

                fk_data.append({
                    "from_table": table.table_name,
                    "to_table": ref_table,
                    "column": fk.column,
                    "references_column": fk.references_column,
                    "relation": fk.relation,
                    "description": fk.description,
                    "domain": table.domain
                })

        logger.info(f"Creating {len(fk_data)} FK relationships...")

        def create_fks(tx: ManagedTransaction, fks: list[dict[str, Any]]) -> None:
            tx.run("""
                UNWIND $fks AS fk
                MATCH (from_table:Table {domain: fk.domain, table_name: fk.from_table})
                MATCH (to_table:Table {table_name: fk.to_table})
                MERGE (from_table)-[r:FK]->(to_table)
                SET r.column = fk.column,
                    r.references_column = fk.references_column,
                    r.relation = fk.relation,
                    r.description = fk.description
            """, fks=fks)

        for i in range(0, len(fk_data), self.BATCH_SIZE):
            batch = fk_data[i:i + self.BATCH_SIZE]
            session.execute_write(create_fks, batch)

        logger.info("FK relationships created")

    def _create_table_concept_relationships(self, session: Any, metadata: DomainMetadata) -> None:
        """Create HAS_CONCEPT relationships from tables to concepts."""
        rel_data = []

        for table in metadata.tables:
            for concept in table.concepts:
                rel_data.append({
                    "table_name": table.table_name,
                    "domain": table.domain,
                    "concept_name": concept.name
                })

        logger.info(f"Creating {len(rel_data)} HAS_CONCEPT relationships...")

        def create_has_concept(tx: ManagedTransaction, rels: list[dict[str, Any]]) -> None:
            tx.run("""
                UNWIND $rels AS r
                MATCH (t:Table {domain: r.domain, table_name: r.table_name})
                MATCH (c:Concept {name: r.concept_name})
                MERGE (t)-[rel:HAS_CONCEPT]->(c)
                SET rel.source = 'table'
            """, rels=rels)

        for i in range(0, len(rel_data), self.BATCH_SIZE):
            batch = rel_data[i:i + self.BATCH_SIZE]
            session.execute_write(create_has_concept, batch)

        logger.info("HAS_CONCEPT relationships created")

    def _create_column_semantic_relationships(self, session: Any, metadata: DomainMetadata) -> None:
        """Create HAS_SEMANTIC relationships from columns to concepts."""
        rel_data = []

        # Get concept names for matching
        concept_names = set(metadata.concepts.keys())

        for table in metadata.tables:
            for col in table.columns:
                for semantic in col.semantics:
                    if semantic in concept_names:
                        rel_data.append({
                            "table_name": col.table_name,
                            "column_name": col.column_name,
                            "concept_name": semantic
                        })

        logger.info(f"Creating {len(rel_data)} HAS_SEMANTIC relationships...")

        def create_has_semantic(tx: ManagedTransaction, rels: list[dict[str, Any]]) -> None:
            tx.run("""
                UNWIND $rels AS r
                MATCH (col:Column {table_name: r.table_name, column_name: r.column_name})
                MATCH (c:Concept {name: r.concept_name})
                MERGE (col)-[:HAS_SEMANTIC]->(c)
            """, rels=rels)

        for i in range(0, len(rel_data), self.BATCH_SIZE):
            batch = rel_data[i:i + self.BATCH_SIZE]
            session.execute_write(create_has_semantic, batch)

        logger.info("HAS_SEMANTIC relationships created")

    def _create_metric_relationships(self, session: Any, metadata: DomainMetadata) -> None:
        """Create METRIC_BASE_TABLE and HAS_METRIC relationships."""
        metric_data = []

        # Build a set of valid table names
        table_names = {t.table_name for t in metadata.tables}

        for metric in metadata.metrics:
            if metric.base_table in table_names:
                # Find the domain for the base table
                domain = next(
                    (t.domain for t in metadata.tables if t.table_name == metric.base_table),
                    metadata.domain
                )
                metric_data.append({
                    "metric_name": metric.name,
                    "base_table": metric.base_table,
                    "domain": domain
                })

        logger.info(f"Creating {len(metric_data)} metric relationships...")

        def create_metric_rels(tx: ManagedTransaction, metrics: list[dict[str, Any]]) -> None:
            tx.run("""
                UNWIND $metrics AS m
                MATCH (metric:Metric {name: m.metric_name})
                MATCH (t:Table {domain: m.domain, table_name: m.base_table})
                MERGE (metric)-[:METRIC_BASE_TABLE]->(t)
                MERGE (t)-[:HAS_METRIC]->(metric)
            """, metrics=metrics)

        for i in range(0, len(metric_data), self.BATCH_SIZE):
            batch = metric_data[i:i + self.BATCH_SIZE]
            session.execute_write(create_metric_rels, batch)

        logger.info("Metric relationships created")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build Neo4j knowledge graph from YAML metadata"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="vnfilm_ticketing",
        help="Domain name (default: vnfilm_ticketing)"
    )
    parser.add_argument(
        "--metadata-root",
        type=str,
        default="metadata/domains",
        help="Root path to metadata domains (default: metadata/domains)"
    )

    args = parser.parse_args()

    # Get Neo4j connection details from environment
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

    if not neo4j_password or neo4j_password == "password":
        logger.warning(
            "NEO4J_PASSWORD not set or using default. "
            "Set the NEO4J_PASSWORD environment variable."
        )

    # Resolve metadata root path
    metadata_root = Path(args.metadata_root)
    if not metadata_root.is_absolute():
        metadata_root = Path.cwd() / metadata_root

    # Load metadata
    loader = MetadataLoader(metadata_root, args.domain)
    metadata = loader.load()

    # Build graph
    builder = Neo4jGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
    try:
        builder.build_graph(metadata)
    finally:
        builder.close()

    logger.info("Done! You can now explore the graph in Neo4j Browser.")
    logger.info("Example queries:")
    logger.info("  MATCH (t:Table)-[r:JOIN]->(t2:Table) RETURN t,r,t2 LIMIT 50")
    logger.info("  MATCH (t:Table)-[:HAS_COLUMN]->(c:Column) RETURN t,c LIMIT 50")
    logger.info("  MATCH (m:Metric)-[:METRIC_BASE_TABLE]->(t:Table) RETURN m,t LIMIT 50")


if __name__ == "__main__":
    main()
