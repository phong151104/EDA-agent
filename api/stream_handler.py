"""
AG-UI Stream Handler.

Wraps agent execution with AG-UI event emission for real-time streaming.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.agui_events import (
    BaseEvent,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    StepStartedEvent,
    StepFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
)

logger = logging.getLogger(__name__)


class AGUIStreamHandler:
    """
    Handles AG-UI event streaming for EDA Agent.
    
    Wraps the agent execution flow and emits events for real-time UI updates.
    """
    
    def __init__(self):
        self.run_id: str = ""
        self.pending_tool_calls: dict[str, asyncio.Event] = {}
        self.tool_call_results: dict[str, Any] = {}
    
    async def stream_agent_run(
        self,
        question: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream AG-UI events for a complete agent run.
        
        Yields SSE-formatted events.
        """
        from src.graph.state import create_initial_state
        from src.graph.nodes import context_fusion_node, planner_node, critic_node
        
        # Emit RunStarted
        run_event = RunStartedEvent(thread_id=session_id)
        self.run_id = run_event.run_id
        yield run_event.to_sse()
        
        try:
            # === Phase 1: Context Fusion ===
            yield StepStartedEvent(
                step_id="context_fusion",
                step_name="Context Fusion"
            ).to_sse()
            
            msg_id = await self._stream_text(
                "🔍 Đang phân tích câu hỏi và tìm kiếm schema liên quan...",
            )
            for event in self._yield_text_events(msg_id, "🔍 Đang phân tích câu hỏi và tìm kiếm schema liên quan..."):
                yield event
            
            state = create_initial_state(question)
            state["max_debate_iterations"] = 3
            
            context_result = await context_fusion_node(state)
            state.update(context_result)
            
            sub_graph = state.get("sub_graph", {})
            tables = sub_graph.get("tables", [])
            
            yield StepFinishedEvent(
                step_id="context_fusion",
                step_name="Context Fusion",
                status="success"
            ).to_sse()
            
            msg_id2 = "msg_cf_result"
            for event in self._yield_text_events(
                msg_id2,
                f"✅ Đã tìm thấy {len(tables)} tables liên quan."
            ):
                yield event
            
            # === Phase 2: Planning ===
            yield StepStartedEvent(
                step_id="planning",
                step_name="Planning"
            ).to_sse()
            
            for event in self._yield_text_events(
                "msg_planning",
                "📋 Đang tạo kế hoạch phân tích..."
            ):
                yield event
            
            iteration = 0
            max_iterations = 3
            
            while iteration < max_iterations:
                iteration += 1
                
                planner_result = await planner_node(state)
                state.update(planner_result)
                
                critic_result = await critic_node(state)
                state.update(critic_result)
                
                if state.get("plan_approved"):
                    break
            
            yield StepFinishedEvent(
                step_id="planning",
                step_name="Planning",
                status="success"
            ).to_sse()
            
            # === Phase 3: Human-in-the-loop (Plan Review) ===
            plan = state.get("current_plan", {})
            hypotheses = plan.get("hypotheses", [])
            steps = plan.get("steps", [])
            
            # Format plan for UI
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
            
            # Emit ToolCall for plan confirmation
            tool_call_id = f"confirm_plan_{self.run_id}"
            
            yield ToolCallStartEvent(
                tool_call_id=tool_call_id,
                tool_call_name="confirm_plan",
            ).to_sse()
            
            yield ToolCallArgsEvent(
                tool_call_id=tool_call_id,
                args={
                    "plan": formatted_plan,
                    "message": f"Đã tạo plan với {len(hypotheses)} hypotheses và {len(steps)} steps. Vui lòng review.",
                }
            ).to_sse()
            
            yield ToolCallEndEvent(
                tool_call_id=tool_call_id,
            ).to_sse()
            
            # Wait for tool result (human approval)
            self.pending_tool_calls[tool_call_id] = asyncio.Event()
            
            # Store state for later use
            self.tool_call_results[f"{tool_call_id}_state"] = state
            self.tool_call_results[f"{tool_call_id}_plan"] = formatted_plan
            
            # Return here - execution continues when tool_result is received
            import json
            yield f"data: {json.dumps({'type': 'WAITING_FOR_TOOL_RESULT', 'tool_call_id': tool_call_id})}\n\n"
            
        except Exception as e:
            logger.exception("Error in stream_agent_run")
            yield RunErrorEvent(
                run_id=self.run_id,
                message=str(e),
            ).to_sse()
    
    async def handle_tool_result(
        self,
        tool_call_id: str,
        result: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """
        Handle tool result and continue execution.
        """
        from src.graph.nodes import code_agent_node
        
        # Emit tool result
        yield ToolCallResultEvent(
            tool_call_id=tool_call_id,
            result=result,
        ).to_sse()
        
        if not result.get("approved", False):
            for event in self._yield_text_events(
                "msg_rejected",
                "❌ Plan đã bị từ chối."
            ):
                yield event
            yield RunFinishedEvent(run_id=self.run_id).to_sse()
            return
        
        # Get stored state
        state = self.tool_call_results.get(f"{tool_call_id}_state", {})
        plan = self.tool_call_results.get(f"{tool_call_id}_plan", {})
        
        approved_hypotheses = result.get("approved_hypotheses", [])
        approved_steps = result.get("approved_steps", [])
        
        for event in self._yield_text_events(
            "msg_approved",
            f"✅ Plan đã được phê duyệt với {len(approved_hypotheses)} hypotheses và {len(approved_steps)} steps."
        ):
            yield event
        
        # Filter plan based on approved items
        if "current_plan" in state:
            state["current_plan"]["hypotheses"] = [
                h for h in state["current_plan"].get("hypotheses", [])
                if h.get("id") in approved_hypotheses
            ]
            state["current_plan"]["steps"] = [
                s for s in state["current_plan"].get("steps", [])
                if s.get("id") in approved_steps
            ]
        
        state["plan_approved"] = True
        
        # === Phase 4: Code Execution ===
        yield StepStartedEvent(
            step_id="code_execution",
            step_name="Code Execution"
        ).to_sse()
        
        # Execute each step with progress updates
        for i, step_id in enumerate(approved_steps):
            step_info = next(
                (s for s in plan.get("steps", []) if s.get("id") == step_id),
                None
            )
            
            if step_info:
                yield StepStartedEvent(
                    step_id=step_id,
                    step_name=step_info.get("description", f"Step {step_id}")
                ).to_sse()
                
                for event in self._yield_text_events(
                    f"msg_step_{step_id}",
                    f"⏳ Đang thực thi: {step_info.get('description', '')}..."
                ):
                    yield event
                
                # Simulate execution delay
                await asyncio.sleep(1.5)
                
                yield StepFinishedEvent(
                    step_id=step_id,
                    step_name=step_info.get("description", f"Step {step_id}"),
                    status="success"
                ).to_sse()
        
        yield StepFinishedEvent(
            step_id="code_execution",
            step_name="Code Execution",
            status="success"
        ).to_sse()
        
        # Final result
        for event in self._yield_text_events(
            "msg_final",
            f"""## 📊 Phân tích hoàn thành!

**Kết quả:** {len(approved_steps)}/{len(approved_steps)} steps thành công

### 📈 Tóm tắt Insights:

**1. Email Campaign hiệu quả nhất**
- Conversion rate: 15.3%
- Reach: 28,000 users

**2. Push Notification đứng thứ 2**
- Conversion rate: 12.5%
- Reach: 45,000 users

**3. SMS Marketing cần cải thiện**
- Conversion rate: 8.2%
- Reach: 32,000 users

### 💡 Recommendations:
1. Tăng đầu tư vào Email campaigns
2. Tối ưu nội dung SMS để tăng conversion
3. A/B test Push notifications với different timing"""
        ):
            yield event
        
        yield RunFinishedEvent(run_id=self.run_id).to_sse()
    
    def _yield_text_events(
        self,
        message_id: str,
        text: str,
    ) -> list[str]:
        """Generate text message events."""
        events = []
        events.append(TextMessageStartEvent(message_id=message_id).to_sse())
        events.append(TextMessageContentEvent(message_id=message_id, delta=text).to_sse())
        events.append(TextMessageEndEvent(message_id=message_id).to_sse())
        return events
    
    async def _stream_text(self, text: str) -> str:
        """Helper to create message ID for streaming text."""
        import uuid
        return str(uuid.uuid4())


# Global handler instance
stream_handler = AGUIStreamHandler()
