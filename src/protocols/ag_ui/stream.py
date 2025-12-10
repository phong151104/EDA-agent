"""
AG-UI Streaming Handler.

Manages event streaming to UI clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator
from uuid import uuid4

from .events import (
    AGUIEvent,
    EventType,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

logger = logging.getLogger(__name__)


class AGUIStreamHandler:
    """
    Handles AG-UI event streaming.
    
    Provides methods for:
    - Starting/ending runs
    - Streaming text messages
    - Tool call notifications
    - State updates
    """
    
    def __init__(self, run_id: str | None = None):
        """
        Initialize stream handler.
        
        Args:
            run_id: Optional run identifier
        """
        self.run_id = run_id or str(uuid4())
        self._event_queue: asyncio.Queue[AGUIEvent] = asyncio.Queue()
        self._is_running = False
        self._current_message_id: str | None = None
    
    async def start_run(self) -> None:
        """Signal run start."""
        self._is_running = True
        await self._emit(RunStartedEvent(run_id=self.run_id))
    
    async def end_run(self, result: Any = None) -> None:
        """Signal run end."""
        self._is_running = False
        await self._emit(RunFinishedEvent(
            run_id=self.run_id,
            result=result,
        ))
    
    async def start_step(self, step_name: str, step_index: int = 0) -> None:
        """Signal step start."""
        await self._emit(StepStartedEvent(
            run_id=self.run_id,
            step_name=step_name,
            step_index=step_index,
        ))
    
    async def end_step(
        self,
        step_name: str,
        step_index: int = 0,
        duration_ms: int = 0,
    ) -> None:
        """Signal step end."""
        await self._emit(StepFinishedEvent(
            run_id=self.run_id,
            step_name=step_name,
            step_index=step_index,
            duration_ms=duration_ms,
        ))
    
    async def start_message(self, role: str = "assistant") -> str:
        """
        Start a new text message.
        
        Returns:
            Message ID
        """
        message_id = str(uuid4())
        self._current_message_id = message_id
        
        await self._emit(TextMessageStartEvent(
            run_id=self.run_id,
            message_id=message_id,
            role=role,
        ))
        
        return message_id
    
    async def stream_content(self, content: str) -> None:
        """
        Stream text content.
        
        Args:
            content: Text chunk to stream
        """
        if not self._current_message_id:
            await self.start_message()
        
        await self._emit(TextMessageContentEvent(
            run_id=self.run_id,
            message_id=self._current_message_id or "",
            data={"content": content},
        ))
    
    async def end_message(self) -> None:
        """End current text message."""
        if self._current_message_id:
            await self._emit(TextMessageEndEvent(
                run_id=self.run_id,
                message_id=self._current_message_id,
            ))
            self._current_message_id = None
    
    async def start_tool_call(self, tool_name: str) -> str:
        """
        Signal tool call start.
        
        Returns:
            Tool call ID
        """
        tool_call_id = str(uuid4())
        
        await self._emit(ToolCallStartEvent(
            run_id=self.run_id,
            tool_call_id=tool_call_id,
            data={"tool_name": tool_name},
        ))
        
        return tool_call_id
    
    async def end_tool_call(
        self,
        tool_call_id: str,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Signal tool call end."""
        await self._emit(ToolCallEndEvent(
            run_id=self.run_id,
            tool_call_id=tool_call_id,
            data={"result": result, "error": error},
        ))
    
    async def send_state_snapshot(self, state: dict[str, Any]) -> None:
        """Send full state snapshot."""
        await self._emit(StateSnapshotEvent(
            run_id=self.run_id,
            data={"state": state},
        ))
    
    async def _emit(self, event: AGUIEvent) -> None:
        """Add event to queue."""
        await self._event_queue.put(event)
    
    async def stream_events(self) -> AsyncGenerator[str, None]:
        """
        Stream events as Server-Sent Events.
        
        Yields:
            SSE formatted event strings
        """
        while self._is_running or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                yield event.to_sse()
            except asyncio.TimeoutError:
                # Send keepalive
                yield ": keepalive\n\n"
    
    def get_events_sync(self) -> list[AGUIEvent]:
        """Get all queued events synchronously."""
        events = []
        while not self._event_queue.empty():
            try:
                events.append(self._event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events
