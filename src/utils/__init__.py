"""
Utilities module.

Provides common utilities for logging, LLM client, and prompts.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from config import config


def setup_logging() -> None:
    """Configure structured logging."""
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if config.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Also configure standard logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def get_logger(name: str) -> Any:
    """Get a structured logger."""
    return structlog.get_logger(name)


# LLM utilities

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for text.
    
    Simple estimation based on average characters per token.
    """
    # Rough estimate: ~4 characters per token for English
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
