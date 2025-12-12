"""
FastAPI Backend for EDA Agent.

Exposes the EDA agent flow to the frontend via REST API and SSE.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session storage (in-memory for demo)
sessions: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan handler."""
    logger.info("EDA Agent API starting...")
    yield
    logger.info("EDA Agent API shutting down...")


app = FastAPI(
    title="EDA Agent API",
    description="API for EDA Multi-Agent System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    status: str
    plan: dict | None = None
    message: str | None = None


class ApproveRequest(BaseModel):
    session_id: str
    approved_hypotheses: list[str]
    approved_steps: list[str]


class RejectRequest(BaseModel):
    session_id: str
    feedback: str | None = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "EDA Agent API"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Start analysis flow with a question.
    
    Returns plan for user approval.
    """
    from src.graph.state import create_initial_state
    from src.graph.nodes import context_fusion_node, planner_node, critic_node
    
    session_id = request.session_id or str(uuid.uuid4())
    
    logger.info(f"[API] New chat - Session: {session_id}")
    logger.info(f"[API] Question: {request.question}")
    
    try:
        # Create initial state
        state = create_initial_state(request.question)
        state["max_debate_iterations"] = 3
        
        # Phase 1: Context Fusion
        logger.info("[API] Running Context Fusion...")
        context_result = await context_fusion_node(state)
        state.update(context_result)
        
        sub_graph = state.get("sub_graph", {})
        logger.info(f"[API] SubGraph: {len(sub_graph.get('tables', []))} tables")
        
        # Phase 2: Planner → Critic loop (max 3 iterations)
        iteration = 0
        max_iterations = state.get("max_debate_iterations", 3)
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[API] Planning iteration {iteration}/{max_iterations}...")
            
            # Planner
            planner_result = await planner_node(state)
            state.update(planner_result)
            
            # Critic
            critic_result = await critic_node(state)
            state.update(critic_result)
            
            if state.get("plan_approved"):
                logger.info("[API] Plan approved by Critic!")
                break
        
        # Extract plan for frontend
        plan = state.get("current_plan", {})
        hypotheses = plan.get("hypotheses", [])
        steps = plan.get("steps", [])
        
        # Format for frontend
        formatted_plan = {
            "version": plan.get("version", 1),
            "hypotheses": [
                {
                    "id": h.get("id", f"h{i+1}"),
                    "statement": h.get("statement", h.get("title", "")),
                    "rationale": h.get("rationale", ""),
                    "priority": h.get("priority", i + 1),
                }
                for i, h in enumerate(hypotheses)
            ],
            "steps": [
                {
                    "id": s.get("id", f"s{i+1}"),
                    "hypothesis_id": s.get("hypothesis_id", ""),
                    "description": s.get("description", ""),
                    "action_type": s.get("action_type", "query"),
                }
                for i, s in enumerate(steps)
            ],
        }
        
        # Store session
        sessions[session_id] = {
            "state": state,
            "plan": formatted_plan,
            "question": request.question,
            "status": "waiting_approval",
        }
        
        return ChatResponse(
            session_id=session_id,
            status="waiting_approval",
            plan=formatted_plan,
            message=f"Đã tạo plan với {len(hypotheses)} hypotheses và {len(steps)} steps. Vui lòng review.",
        )
        
    except Exception as e:
        logger.exception("[API] Error in chat")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan/approve")
async def approve_plan(request: ApproveRequest):
    """
    Approve plan and execute Code Agent.
    """
    from src.graph.nodes import code_agent_node
    
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    logger.info(f"[API] Approving plan - Session: {request.session_id}")
    logger.info(f"[API] Approved: {len(request.approved_hypotheses)} hypotheses, {len(request.approved_steps)} steps")
    
    state = session["state"]
    plan = state.get("current_plan", {})
    
    # Filter plan based on approved items
    plan["hypotheses"] = [
        h for h in plan.get("hypotheses", [])
        if h.get("id") in request.approved_hypotheses
    ]
    plan["steps"] = [
        s for s in plan.get("steps", [])
        if s.get("id") in request.approved_steps
    ]
    
    state["current_plan"] = plan
    state["plan_approved"] = True
    
    try:
        # Phase 3: Code Agent
        logger.info("[API] Running Code Agent...")
        code_result = await code_agent_node(state)
        state.update(code_result)
        
        generated_code = state.get("generated_code", [])
        execution_results = state.get("execution_results", {})
        
        # Format results
        results = []
        for code in generated_code:
            step_id = code.get("step_id", "")
            exec_result = execution_results.get(step_id, {})
            
            results.append({
                "step_id": step_id,
                "hypothesis_id": code.get("hypothesis_id", ""),
                "language": code.get("language", ""),
                "code": code.get("code", ""),
                "description": code.get("description", ""),
                "status": exec_result.get("status", "unknown"),
                "output": exec_result.get("output", {}),
                "execution_time_ms": exec_result.get("execution_time_ms", 0),
            })
        
        # Update session
        session["status"] = "completed"
        session["results"] = results
        
        return {
            "status": "completed",
            "results": results,
            "summary": {
                "total_steps": len(generated_code),
                "successful": sum(1 for r in results if r["status"] == "success"),
                "failed": sum(1 for r in results if r["status"] == "error"),
            },
        }
        
    except Exception as e:
        logger.exception("[API] Error in approve")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan/reject")
async def reject_plan(request: RejectRequest):
    """
    Reject plan and optionally provide feedback.
    """
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    logger.info(f"[API] Rejecting plan - Session: {request.session_id}")
    
    # Clear session
    del sessions[request.session_id]
    
    return {
        "status": "rejected",
        "message": "Plan rejected. Start a new conversation.",
    }


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session status and results."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "status": session.get("status"),
        "question": session.get("question"),
        "plan": session.get("plan"),
        "results": session.get("results"),
    }


# ============================================================================
# AG-UI Protocol Endpoints
# ============================================================================

class RunRequest(BaseModel):
    """AG-UI run request."""
    question: str
    thread_id: str | None = None


class ToolResultRequest(BaseModel):
    """AG-UI tool result request."""
    tool_call_id: str
    approved: bool = False
    approved_hypotheses: list[str] = []
    approved_steps: list[str] = []


@app.post("/agui/run")
async def agui_run(request: RunRequest):
    """
    AG-UI Protocol: Start agent run with SSE streaming.
    
    Returns SSE stream of AG-UI events.
    """
    from api.stream_handler import stream_handler
    
    logger.info(f"[AG-UI] Starting run - Question: {request.question}")
    
    async def event_generator():
        try:
            async for event in stream_handler.stream_agent_run(
                question=request.question,
                session_id=request.thread_id,
            ):
                yield event
        except Exception as e:
            logger.exception("[AG-UI] Error in run")
            yield f"data: {{\"type\": \"RUN_ERROR\", \"message\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/agui/tool_result")
async def agui_tool_result(request: ToolResultRequest):
    """
    AG-UI Protocol: Submit tool result (human-in-the-loop response).
    
    Called when user approves/rejects the plan.
    Returns SSE stream of remaining execution events.
    """
    from api.stream_handler import stream_handler
    
    logger.info(f"[AG-UI] Tool result - ID: {request.tool_call_id}, Approved: {request.approved}")
    
    async def event_generator():
        try:
            async for event in stream_handler.handle_tool_result(
                tool_call_id=request.tool_call_id,
                result={
                    "approved": request.approved,
                    "approved_hypotheses": request.approved_hypotheses,
                    "approved_steps": request.approved_steps,
                },
            ):
                yield event
        except Exception as e:
            logger.exception("[AG-UI] Error in tool_result")
            yield f"data: {{\"type\": \"RUN_ERROR\", \"message\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Legacy SSE endpoint (kept for compatibility)
@app.get("/stream/{session_id}")
async def stream_events(session_id: str):
    """
    Legacy SSE stream for real-time updates.
    
    Kept for backward compatibility with REST API.
    """
    async def event_generator():
        yield f"data: {json.dumps({'type': 'status', 'status': 'connected'})}\n\n"
        
        for i in range(60):
            await asyncio.sleep(1)
            
            session = sessions.get(session_id)
            if session:
                yield f"data: {json.dumps({'type': 'heartbeat', 'status': session.get('status')})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
                break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

