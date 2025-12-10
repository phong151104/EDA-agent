"""Context Fusion Layer module."""

from .context_builder import ContextBuilder, EnrichedContext, build_context, build_session
from .models import (
    AnalyzedQuery,
    ColumnNode,
    ConceptNode,
    ExtractedEntity,
    JoinEdge,
    MetricNode,
    QueryIntent,
    SubGraph,
    TableNode,
)
from .query_rewriter import QueryRewriter, quick_analyze
from .schema_retriever import SchemaRetriever
from .session_context import EDASession, cache_session, get_cached_session
from .vector_index import Neo4jVectorIndex

__all__ = [
    # Models
    "QueryIntent",
    "ExtractedEntity",
    "AnalyzedQuery",
    "TableNode",
    "ColumnNode",
    "JoinEdge",
    "MetricNode",
    "ConceptNode",
    "SubGraph",
    # Query Rewriter
    "QueryRewriter",
    "quick_analyze",
    # Schema Retriever
    "SchemaRetriever",
    # Vector Index
    "Neo4jVectorIndex",
    # Context Builder
    "ContextBuilder",
    "EnrichedContext",
    "build_context",
    "build_session",
    # Session
    "EDASession",
    "cache_session",
    "get_cached_session",
]

