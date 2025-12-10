"""
EDA Agent Configuration Module.

Centralized configuration using Pydantic Settings.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""
    
    model_config = SettingsConfigDict(env_prefix="OPENAI_")
    
    api_key: str = Field(default="", description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="Default model to use")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")


class Neo4jConfig(BaseSettings):
    """Neo4j database configuration."""
    
    model_config = SettingsConfigDict(env_prefix="NEO4J_")
    
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    user: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(default="", description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")


class PostgresConfig(BaseSettings):
    """PostgreSQL database configuration."""
    
    model_config = SettingsConfigDict(env_prefix="POSTGRES_")
    
    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    user: str = Field(default="eda_agent", description="PostgreSQL username")
    password: str = Field(default="", description="PostgreSQL password")
    db: str = Field(default="eda_agent", description="PostgreSQL database name")
    
    @property
    def connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
    
    @property
    def async_connection_string(self) -> str:
        """Get async SQLAlchemy connection string."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class APIConfig(BaseSettings):
    """API server configuration."""
    
    model_config = SettingsConfigDict(env_prefix="API_")
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")


class MCPConfig(BaseSettings):
    """MCP Server configuration."""
    
    model_config = SettingsConfigDict(env_prefix="MCP_")
    
    host: str = Field(default="localhost", description="MCP host")
    port: int = Field(default=3000, description="MCP port")


class SandboxConfig(BaseSettings):
    """Code execution sandbox configuration."""
    
    model_config = SettingsConfigDict(env_prefix="E2B_")
    
    api_key: str = Field(default="", description="E2B API key")
    timeout: int = Field(default=30, description="Execution timeout in seconds")


class VectorIndexConfig(BaseSettings):
    """Vector index configuration."""
    
    dimensions: int = Field(default=1536, description="Embedding dimensions")
    similarity_function: Literal["cosine", "euclidean"] = Field(
        default="cosine", description="Similarity function"
    )
    labels: list[str] = Field(
        default=["Table", "Column", "Metric", "Concept"],
        description="Node labels to index"
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Sub-configs
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: Literal["json", "console"] = Field(
        default="json", description="Logging format"
    )
    
    # Agent settings
    max_debate_iterations: int = Field(
        default=3, description="Max Planner-Critic debate iterations"
    )
    max_code_retries: int = Field(
        default=3, description="Max code execution retries"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global config instance
config = get_settings()
