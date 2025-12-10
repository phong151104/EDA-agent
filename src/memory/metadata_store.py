"""
Metadata Store module.

Stores schema information and business rules in PostgreSQL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Column, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from config import config

logger = logging.getLogger(__name__)


# === SQLAlchemy Models ===

class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class TableMetadata(Base):
    """Metadata for database tables."""
    
    __tablename__ = "table_metadata"
    
    id = Column(Integer, primary_key=True)
    domain = Column(String(100), nullable=False)
    catalog = Column(String(100))
    schema_name = Column(String(100))
    table_name = Column(String(100), nullable=False, unique=True)
    business_name = Column(String(200))
    description = Column(Text)
    grain = Column(String(200))
    tags = Column(JSONB, default=list)
    
    columns = relationship("ColumnMetadata", back_populates="table")


class ColumnMetadata(Base):
    """Metadata for table columns."""
    
    __tablename__ = "column_metadata"
    
    id = Column(Integer, primary_key=True)
    table_id = Column(Integer, ForeignKey("table_metadata.id"), nullable=False)
    column_name = Column(String(100), nullable=False)
    data_type = Column(String(50))
    business_name = Column(String(200))
    description = Column(Text)
    semantics = Column(JSONB, default=list)
    unit = Column(String(50))
    is_pii = Column(Integer, default=0)
    
    table = relationship("TableMetadata", back_populates="columns")


class BusinessRule(Base):
    """Business rules and constraints."""
    
    __tablename__ = "business_rules"
    
    id = Column(Integer, primary_key=True)
    domain = Column(String(100), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    rule_type = Column(String(50))  # "constraint", "validation", "calculation"
    expression = Column(Text)
    affected_tables = Column(JSONB, default=list)


class MetricDefinition(Base):
    """Business metric definitions."""
    
    __tablename__ = "metric_definitions"
    
    id = Column(Integer, primary_key=True)
    domain = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False, unique=True)
    business_name = Column(String(200))
    description = Column(Text)
    expression = Column(Text, nullable=False)
    base_table = Column(String(100))
    unit = Column(String(50))
    tags = Column(JSONB, default=list)


# === Metadata Store Service ===

@dataclass
class TableInfo:
    """Table information container."""
    
    name: str
    business_name: str = ""
    description: str = ""
    columns: list[dict[str, Any]] = field(default_factory=list)


class MetadataStore:
    """
    Metadata store for schema and business rules.
    
    Provides methods for querying and updating metadata
    stored in PostgreSQL.
    """
    
    def __init__(self):
        """Initialize metadata store."""
        self._engine = None
        self._async_engine = None
        self._session_factory = None
    
    @property
    def engine(self):
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(
                config.postgres.connection_string,
                echo=False,
            )
        return self._engine
    
    @property
    def async_engine(self):
        """Get or create async SQLAlchemy engine."""
        if self._async_engine is None:
            self._async_engine = create_async_engine(
                config.postgres.async_connection_string,
                echo=False,
            )
        return self._async_engine
    
    async def get_table_metadata(
        self,
        table_name: str,
    ) -> TableInfo | None:
        """
        Get metadata for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            TableInfo or None if not found
        """
        # TODO: Implement actual database query
        # Placeholder for now
        logger.debug(f"Getting metadata for table: {table_name}")
        return None
    
    async def search_tables(
        self,
        query: str,
        domain: str | None = None,
    ) -> list[TableInfo]:
        """
        Search for tables matching a query.
        
        Args:
            query: Search query
            domain: Optional domain filter
            
        Returns:
            List of matching tables
        """
        # TODO: Implement search with full-text or semantic matching
        logger.debug(f"Searching tables for: {query}")
        return []
    
    async def get_business_rules(
        self,
        table_names: list[str] | None = None,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get business rules for tables.
        
        Args:
            table_names: Optional filter by table names
            domain: Optional domain filter
            
        Returns:
            List of business rules
        """
        # TODO: Implement query
        logger.debug(f"Getting business rules for tables: {table_names}")
        return []
    
    async def get_metric_definition(
        self,
        metric_name: str,
    ) -> dict[str, Any] | None:
        """
        Get definition for a business metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Metric definition or None
        """
        # TODO: Implement query
        logger.debug(f"Getting metric definition: {metric_name}")
        return None
    
    async def validate_column_exists(
        self,
        table_name: str,
        column_name: str,
    ) -> bool:
        """
        Check if a column exists in a table.
        
        Args:
            table_name: Table name
            column_name: Column name
            
        Returns:
            True if column exists
        """
        # TODO: Implement validation
        logger.debug(f"Validating column: {table_name}.{column_name}")
        return True
    
    async def init_db(self) -> None:
        """Initialize database schema."""
        Base.metadata.create_all(self.engine)
        logger.info("Initialized metadata store database")
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
        if self._async_engine:
            await self._async_engine.dispose()
