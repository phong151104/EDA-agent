"""
API Routes.

Defines all API endpoints for the EDA Agent.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.graph import EDAWorkflowRunner, create_initial_state
from src.protocols.ag_ui import AGUIStreamHandler

logger = logging.getLogger(__name__)

router = APIRouter()


# === Request/Response Models ===

class AnalyzeRequest(BaseModel):
    """Request to analyze a question."""
    
    question: str = Field(..., description="Question to analyze")
    domain: str | None = Field(None, description="Optional domain filter")
    stream: bool = Field(True, description="Whether to stream response")


class AnalyzeResponse(BaseModel):
    """Non-streaming analyze response."""
    
    session_id: str
    question: str
    summary: str
    hypotheses: list[dict[str, Any]]
    insights: list[dict[str, Any]]
    final_report: dict[str, Any] | None


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str


# === Endpoints ===

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@router.post("/analyze")
async def analyze(
    request: Request,
    body: AnalyzeRequest,
):
    """
    Analyze a question using the EDA workflow.
    
    If stream=True, returns Server-Sent Events stream.
    Otherwise, returns complete result.
    """
    runner: EDAWorkflowRunner = request.app.state.workflow_runner
    
    if body.stream:
        return StreamingResponse(
            _stream_analysis(runner, body.question),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response
        result = await runner.run(body.question)
        
        return AnalyzeResponse(
            session_id=result.get("session_id", ""),
            question=body.question,
            summary=result.get("analysis_summary", ""),
            hypotheses=result.get("hypotheses", []),
            insights=result.get("insights", []),
            final_report=result.get("final_report"),
        )


async def _stream_analysis(
    runner: EDAWorkflowRunner,
    question: str,
):
    """Stream analysis results as SSE."""
    stream_handler = AGUIStreamHandler()
    
    await stream_handler.start_run()
    
    try:
        async for event in runner.stream(question):
            # Convert workflow events to AG-UI events
            node_name = list(event.keys())[0] if event else ""
            node_data = event.get(node_name, {})
            
            await stream_handler.start_step(node_name)
            
            # Stream any messages
            if "messages" in node_data:
                for msg in node_data["messages"]:
                    await stream_handler.start_message()
                    await stream_handler.stream_content(str(msg.content))
                    await stream_handler.end_message()
            
            await stream_handler.end_step(node_name)
        
        await stream_handler.end_run()
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        await stream_handler.end_run(result={"error": str(e)})
    
    async for sse_event in stream_handler.stream_events():
        yield sse_event


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Get session details."""
    # TODO: Implement session retrieval from storage
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions/{session_id}/report")
async def get_report(session_id: str) -> dict[str, Any]:
    """Get final report for a session."""
    # TODO: Implement report retrieval
    raise HTTPException(status_code=404, detail="Report not found")
