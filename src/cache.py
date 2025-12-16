"""
Redis Session Cache for EDA Agent.

Stores session data, step results, and SQL cache for fast retrieval.

Usage:
    from src.cache import SessionCache
    
    cache = SessionCache(session_id="abc123")
    cache.save_step_data("s1", rows)
    data = cache.get_step_data("s1")
"""

from __future__ import annotations

import json
import hashlib
import logging
from typing import Any, Optional
from datetime import timedelta

import redis

logger = logging.getLogger(__name__)


class RedisCache:
    """Low-level Redis connection wrapper."""
    
    _pool: redis.ConnectionPool | None = None
    
    @classmethod
    def get_client(cls) -> redis.Redis:
        """Get Redis client with connection pooling."""
        import os
        
        if cls._pool is None:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            db = int(os.getenv("REDIS_DB", "0"))
            password = os.getenv("REDIS_PASSWORD", None)
            
            cls._pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                max_connections=10,
            )
            logger.info(f"[Redis] Connection pool created: {host}:{port}")
        
        return redis.Redis(connection_pool=cls._pool)
    
    @classmethod
    def ping(cls) -> bool:
        """Check if Redis is available."""
        try:
            return cls.get_client().ping()
        except redis.ConnectionError:
            return False


class SessionCache:
    """Session-scoped cache for EDA Agent workflow."""
    
    # Default TTLs
    DEFAULT_TTL = 3600  # 1 hour
    DATA_TTL = 1800     # 30 minutes for large data
    SQL_CACHE_TTL = 300 # 5 minutes for SQL query cache
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.redis = RedisCache.get_client()
        self._prefix = f"session:{session_id}"
    
    # =========================================================================
    # Generic Key-Value Operations
    # =========================================================================
    
    def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
        """Set any key-value pair."""
        try:
            full_key = f"{self._prefix}:{key}"
            data = json.dumps(value, ensure_ascii=False, default=str)
            self.redis.setex(full_key, ttl, data)
            logger.debug(f"[Cache] SET {full_key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"[Cache] SET error: {e}")
            return False
    
    def get(self, key: str) -> Any | None:
        """Get value by key."""
        try:
            full_key = f"{self._prefix}:{key}"
            data = self.redis.get(full_key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"[Cache] GET error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key."""
        full_key = f"{self._prefix}:{key}"
        return bool(self.redis.delete(full_key))
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        full_key = f"{self._prefix}:{key}"
        return bool(self.redis.exists(full_key))
    
    # =========================================================================
    # Step Results (SQL/Python execution)
    # =========================================================================
    
    def save_step_result(
        self,
        step_id: str,
        sql: str | None = None,
        data: list[dict] | None = None,
        code: str | None = None,
        output: str | None = None,
        images: list[str] | None = None,
        # New: enriched metadata fields
        column_metadata: dict | None = None,
        tables_used: list[str] | None = None,
        context_text: str | None = None,
    ) -> bool:
        """Save complete step result with optional enriched metadata."""
        result = {
            "sql": sql,
            "data": data,
            "code": code,
            "output": output,
            "images": images or [],
            # Enriched metadata for Code Agent
            "column_metadata": column_metadata,
            "tables_used": tables_used or [],
            "context_text": context_text,
        }
        return self.set(f"step:{step_id}", result, ttl=self.DATA_TTL)
    
    def get_step_result(self, step_id: str) -> dict | None:
        """Get complete step result."""
        return self.get(f"step:{step_id}")
    
    def get_step_data(self, step_id: str) -> list[dict] | None:
        """Get only the data (rows) from a step."""
        result = self.get_step_result(step_id)
        return result.get("data") if result else None
    
    def get_multiple_step_data(self, step_ids: list[str]) -> dict[str, list]:
        """Get data from multiple steps at once."""
        dataframes = {}
        for step_id in step_ids:
            data = self.get_step_data(step_id)
            if data:
                dataframes[step_id] = data
        return dataframes
    
    def get_enriched_step_data(self, step_ids: list[str]) -> dict[str, Any]:
        """Get data with context from multiple steps for Code Agent.
        
        Returns:
            {
                "dataframes": {"s1": [...], "s2": [...]},
                "context": "## Column Descriptions:\n- col1: desc...",
                "tables_used": ["orders", "users"]
            }
        """
        dataframes = {}
        all_context = []
        all_tables = []
        
        for step_id in step_ids:
            result = self.get_step_result(step_id)
            if not result:
                continue
            
            if result.get("data"):
                dataframes[step_id] = result["data"]
            
            if result.get("context_text"):
                all_context.append(f"### Step {step_id}:\n{result['context_text']}")
            
            if result.get("tables_used"):
                all_tables.extend(result["tables_used"])
        
        return {
            "dataframes": dataframes,
            "context": "\n\n".join(all_context) if all_context else "",
            "tables_used": list(set(all_tables)),
        }
    
    # =========================================================================
    # Plan & SubGraph
    # =========================================================================
    
    def save_plan(self, plan: dict) -> bool:
        """Save analysis plan."""
        return self.set("plan", plan)
    
    def get_plan(self) -> dict | None:
        """Get analysis plan."""
        return self.get("plan")
    
    def save_subgraph(self, subgraph: dict) -> bool:
        """Save SubGraph (tables, columns, joins)."""
        return self.set("subgraph", subgraph)
    
    def get_subgraph(self) -> dict | None:
        """Get SubGraph."""
        return self.get("subgraph")
    
    def save_analyzed_query(self, analyzed: dict) -> bool:
        """Save query analysis result."""
        return self.set("analyzed_query", analyzed)
    
    def get_analyzed_query(self) -> dict | None:
        """Get query analysis result."""
        return self.get("analyzed_query")
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def get_all_keys(self) -> list[str]:
        """Get all keys for this session."""
        pattern = f"{self._prefix}:*"
        return [k.replace(f"{self._prefix}:", "") for k in self.redis.keys(pattern)]
    
    def clear_session(self) -> int:
        """Clear all data for this session."""
        pattern = f"{self._prefix}:*"
        keys = self.redis.keys(pattern)
        if keys:
            return self.redis.delete(*keys)
        return 0
    
    def get_session_summary(self) -> dict:
        """Get summary of session data."""
        keys = self.get_all_keys()
        return {
            "session_id": self.session_id,
            "key_count": len(keys),
            "keys": keys,
            "has_plan": self.exists("plan"),
            "has_subgraph": self.exists("subgraph"),
            "steps": [k for k in keys if k.startswith("step:")],
        }


class SQLCache:
    """Global SQL query result cache (shared across sessions)."""
    
    TTL = 300  # 5 minutes
    
    def __init__(self):
        self.redis = RedisCache.get_client()
    
    def _hash_query(self, sql: str) -> str:
        """Create hash of SQL query."""
        normalized = " ".join(sql.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def get(self, sql: str) -> list[dict] | None:
        """Get cached SQL result."""
        try:
            key = f"sql_cache:{self._hash_query(sql)}"
            data = self.redis.get(key)
            if data:
                logger.info(f"[SQLCache] HIT: {sql[:50]}...")
                return json.loads(data)
            return None
        except Exception:
            return None
    
    def set(self, sql: str, rows: list[dict]) -> bool:
        """Cache SQL result."""
        try:
            key = f"sql_cache:{self._hash_query(sql)}"
            data = json.dumps(rows, ensure_ascii=False, default=str)
            self.redis.setex(key, self.TTL, data)
            logger.info(f"[SQLCache] SET: {sql[:50]}... ({len(rows)} rows)")
            return True
        except Exception as e:
            logger.error(f"[SQLCache] SET error: {e}")
            return False


# Quick access functions
def get_session_cache(session_id: str) -> SessionCache:
    """Get session cache instance."""
    return SessionCache(session_id)


def get_sql_cache() -> SQLCache:
    """Get SQL cache instance."""
    return SQLCache()
