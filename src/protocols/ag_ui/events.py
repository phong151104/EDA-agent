"""
AG-UI Protocol Events.

Implements Agent-User Interaction protocol event types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class EventType(str, Enum):
    """AG-UI event types."""
    
    # Lifecycle events
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_ERROR = "run_error"
    
    # Step events
    STEP_STARTED = "step_started"
    STEP_FINISHED = "step_finished"
    
    # Message events
    TEXT_MESSAGE_START = "text_message_start"
    TEXT_MESSAGE_CONTENT = "text_message_content"
    TEXT_MESSAGE_END = "text_message_end"
    
    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_ARGS = "tool_call_args"
    TOOL_CALL_END = "tool_call_end"
    
    # State events
    STATE_SNAPSHOT = "state_snapshot"
    STATE_DELTA = "state_delta"
    
    # Custom events
    CUSTOM = "custom"


@dataclass
class AGUIEvent:
    """
    AG-UI Protocol event.
    
    Base event for streaming UI updates.
    """
    
    type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    run_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        import json
        
        event_data = {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "runId": self.run_id,
            **self.data,
        }
        
        return f"data: {json.dumps(event_data)}\n\n"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "runId": self.run_id,
            **self.data,
        }


# === Specific Event Types ===

@dataclass
class RunStartedEvent(AGUIEvent):
    """Run started event."""
    
    type: EventType = EventType.RUN_STARTED
    
    def __post_init__(self):
        if not self.run_id:
            self.run_id = str(uuid4())


@dataclass
class RunFinishedEvent(AGUIEvent):
    """Run finished event."""
    
    type: EventType = EventType.RUN_FINISHED
    result: Any = None


@dataclass
class TextMessageStartEvent(AGUIEvent):
    """Text message started."""
    
    type: EventType = EventType.TEXT_MESSAGE_START
    message_id: str = field(default_factory=lambda: str(uuid4()))
    role: str = "assistant"


@dataclass
class TextMessageContentEvent(AGUIEvent):
    """Text message content chunk."""
    
    type: EventType = EventType.TEXT_MESSAGE_CONTENT
    message_id: str = ""
    content: str = ""


@dataclass
class TextMessageEndEvent(AGUIEvent):
    """Text message ended."""
    
    type: EventType = EventType.TEXT_MESSAGE_END
    message_id: str = ""


@dataclass
class ToolCallStartEvent(AGUIEvent):
    """Tool call started."""
    
    type: EventType = EventType.TOOL_CALL_START
    tool_call_id: str = field(default_factory=lambda: str(uuid4()))
    tool_name: str = ""


@dataclass
class ToolCallEndEvent(AGUIEvent):
    """Tool call ended."""
    
    type: EventType = EventType.TOOL_CALL_END
    tool_call_id: str = ""
    result: Any = None
    error: str | None = None


@dataclass
class StateSnapshotEvent(AGUIEvent):
    """Full state snapshot."""
    
    type: EventType = EventType.STATE_SNAPSHOT
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateDeltaEvent(AGUIEvent):
    """State delta update."""
    
    type: EventType = EventType.STATE_DELTA
    path: str = ""
    operation: str = "set"  # set, append, delete
    value: Any = None


@dataclass
class StepStartedEvent(AGUIEvent):
    """Workflow step started."""
    
    type: EventType = EventType.STEP_STARTED
    step_name: str = ""
    step_index: int = 0


@dataclass
class StepFinishedEvent(AGUIEvent):
    """Workflow step finished."""
    
    type: EventType = EventType.STEP_FINISHED
    step_name: str = ""
    step_index: int = 0
    duration_ms: int = 0
