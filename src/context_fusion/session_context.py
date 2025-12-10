"""
EDA Session Context - Minimal Version.

A lightweight wrapper around EnrichedContext for session management.
Designed to integrate with existing architecture without major changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime

from .models import AnalyzedQuery, SubGraph

logger = logging.getLogger(__name__)


@dataclass  
class EDASession:
    """
    Lightweight session container for EDA pipeline.
    
    Simply wraps the existing SubGraph and AnalyzedQuery,
    adding minimal session management without complexity.
    
    Usage:
        # Create once at start of pipeline
        session = EDASession.create(query, analyzed_query, sub_graph)
        
        # Pass to agents - they use existing SubGraph directly
        planner.process(session.sub_graph)
        sql_gen.process(session.sub_graph)
        
        # Track state across agents
        session.set_state("plan", planner_result)
        session.set_state("sql", sql_result)
    """
    
    # Core data (reuse existing models)
    original_query: str = ""
    analyzed_query: Optional[AnalyzedQuery] = None
    sub_graph: Optional[SubGraph] = None
    prompt_context: str = ""
    
    # Session metadata
    session_id: str = ""
    domain: str = "vnfilm_ticketing"
    created_at: datetime = field(default_factory=datetime.now)
    
    # Agent execution state (for pipeline tracking)
    _state: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    @classmethod
    def create(
        cls,
        query: str,
        analyzed_query: AnalyzedQuery,
        sub_graph: SubGraph,
        prompt_context: str = "",
        domain: str = "vnfilm_ticketing",
    ) -> "EDASession":
        """Create a new session from existing analysis results."""
        return cls(
            original_query=query,
            analyzed_query=analyzed_query,
            sub_graph=sub_graph,
            prompt_context=prompt_context or sub_graph.to_prompt_context(),
            domain=domain,
        )
    
    # =========================================================================
    # State Management (for passing data between agents)
    # =========================================================================
    
    def set_state(self, key: str, value: Any) -> None:
        """Store data for use by downstream agents."""
        self._state[key] = value
        logger.debug(f"Session state set: {key}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Retrieve data stored by upstream agents."""
        return self._state.get(key, default)
    
    def has_state(self, key: str) -> bool:
        """Check if state key exists."""
        return key in self._state
    
    # =========================================================================
    # Convenience accessors (delegate to SubGraph)
    # =========================================================================
    
    @property
    def tables(self):
        """Get tables from SubGraph."""
        return self.sub_graph.tables if self.sub_graph else []
    
    @property
    def columns(self):
        """Get columns from SubGraph."""
        return self.sub_graph.columns if self.sub_graph else []
    
    @property
    def joins(self):
        """Get joins from SubGraph."""
        return self.sub_graph.joins if self.sub_graph else []
    
    @property
    def table_names(self):
        """Get list of table names."""
        return self.sub_graph.get_table_names() if self.sub_graph else []
    
    # =========================================================================
    # Serialization (uses existing SubGraph.to_dict())
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "domain": self.domain,
            "created_at": self.created_at.isoformat(),
            "prompt_context": self.prompt_context,
            "sub_graph": self.sub_graph.to_dict() if self.sub_graph else None,
            "state": self._state,
        }
    
    def summary(self) -> str:
        """Get a quick summary."""
        return (
            f"EDASession({self.session_id}, "
            f"tables={len(self.tables)}, "
            f"columns={len(self.columns)}, "
            f"joins={len(self.joins)})"
        )
    
    def __repr__(self) -> str:
        return self.summary()


# =============================================================================
# Simple in-memory cache (no Redis/File complexity)
# =============================================================================

_session_cache: Dict[str, EDASession] = {}


def cache_session(session: EDASession) -> None:
    """Cache session for reuse."""
    _session_cache[session.session_id] = session
    logger.info(f"Session cached: {session.session_id}")


def get_cached_session(session_id: str) -> Optional[EDASession]:
    """Get cached session by ID."""
    return _session_cache.get(session_id)


def clear_session_cache() -> None:
    """Clear all cached sessions."""
    _session_cache.clear()
    logger.info("Session cache cleared")
