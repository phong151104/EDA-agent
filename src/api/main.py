"""
FastAPI Application.

Main API entry point with AG-UI streaming support.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import config
from src.graph import EDAWorkflowRunner
from src.protocols.ag_ui import AGUIStreamHandler

from .routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info("Starting EDA Agent API...")
    
    # Initialize workflow runner
    app.state.workflow_runner = EDAWorkflowRunner()
    
    yield
    
    logger.info("Shutting down EDA Agent API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="EDA Agent API",
        description="Multi-Agent Exploratory Data Analysis System",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    return app


# Create app instance
app = create_app()


def main() -> None:
    """Run the application."""
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
