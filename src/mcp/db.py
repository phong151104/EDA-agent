"""
PostgreSQL Database Connection.

Provides async connection pool for SQL query execution.
Configuration is loaded from environment variables.
"""

import os
import logging
from typing import Any
from contextlib import asynccontextmanager

import asyncpg
from dotenv import load_dotenv

# Load .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Database:
    """
    Async PostgreSQL connection pool manager.
    
    Usage:
        await Database.connect()
        result = await Database.execute("SELECT * FROM table")
        await Database.disconnect()
    """
    
    _pool: asyncpg.Pool | None = None
    
    @classmethod
    async def connect(cls) -> None:
        """Create connection pool."""
        if cls._pool is not None:
            return
        
        config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", ""),
            "database": os.getenv("POSTGRES_DB", "postgres"),
        }
        
        logger.info(f"[DB] Connecting to PostgreSQL at {config['host']}:{config['port']}")
        
        cls._pool = await asyncpg.create_pool(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
            min_size=2,
            max_size=10,
        )
        
        logger.info("[DB] Connection pool created")
    
    @classmethod
    async def disconnect(cls) -> None:
        """Close connection pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            logger.info("[DB] Connection pool closed")
    
    @classmethod
    async def execute(
        cls,
        query: str,
        *args,
        timeout: float = 30.0,
    ) -> list[dict[str, Any]]:
        """
        Execute a SELECT query and return results as list of dicts.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            List of row dictionaries
        """
        if cls._pool is None:
            await cls.connect()
        
        async with cls._pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, *args, timeout=timeout)
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"[DB] Query error: {e}")
                raise
    
    @classmethod
    async def execute_raw(
        cls,
        query: str,
        *args,
    ) -> str:
        """
        Execute a query and return results as formatted string.
        
        Useful for DDL or non-SELECT queries.
        """
        if cls._pool is None:
            await cls.connect()
        
        async with cls._pool.acquire() as conn:
            result = await conn.execute(query, *args)
            return result
    
    @classmethod
    @asynccontextmanager
    async def transaction(cls):
        """Context manager for database transactions."""
        if cls._pool is None:
            await cls.connect()
        
        async with cls._pool.acquire() as conn:
            async with conn.transaction():
                yield conn
