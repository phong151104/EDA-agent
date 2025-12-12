"""
AG-UI Protocol Event Types.

Follows the AG-UI specification for agent-user interaction events.
https://docs.ag-ui.com/concepts/events
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any
import json
import uuid


class EventType(str, Enum):
    """AG-UI Event Types."""
    
    # Lifecycle Events
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"
    
    # Text Message Events
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    
    # Tool Call Events
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    
    # State Events
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"


@dataclass
class BaseEvent:
    """Base event with common properties."""
    
    type: EventType
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_sse(self) -> str:
        """Convert to SSE format."""
        data = asdict(self)
        data["type"] = self.type.value
        return f"data: {json.dumps(data)}\n\n"


@dataclass
class RunStartedEvent(BaseEvent):
    """Emitted when agent run starts."""
    
    type: EventType = field(default=EventType.RUN_STARTED)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str | None = None


@dataclass
class RunFinishedEvent(BaseEvent):
    """Emitted when agent run completes successfully."""
    
    type: EventType = field(default=EventType.RUN_FINISHED)
    run_id: str = ""


@dataclass
class RunErrorEvent(BaseEvent):
    """Emitted when agent run encounters an error."""
    
    type: EventType = field(default=EventType.RUN_ERROR)
    run_id: str = ""
    message: str = ""
    code: str | None = None


@dataclass
class StepStartedEvent(BaseEvent):
    """Emitted when a step begins."""
    
    type: EventType = field(default=EventType.STEP_STARTED)
    step_id: str = ""
    step_name: str = ""


@dataclass
class StepFinishedEvent(BaseEvent):
    """Emitted when a step completes."""
    
    type: EventType = field(default=EventType.STEP_FINISHED)
    step_id: str = ""
    step_name: str = ""
    status: str = "success"  # success, error


@dataclass
class TextMessageStartEvent(BaseEvent):
    """Emitted when text message streaming starts."""
    
    type: EventType = field(default=EventType.TEXT_MESSAGE_START)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "assistant"


@dataclass
class TextMessageContentEvent(BaseEvent):
    """Emitted for each chunk of streaming text."""
    
    type: EventType = field(default=EventType.TEXT_MESSAGE_CONTENT)
    message_id: str = ""
    delta: str = ""


@dataclass
class TextMessageEndEvent(BaseEvent):
    """Emitted when text message streaming ends."""
    
    type: EventType = field(default=EventType.TEXT_MESSAGE_END)
    message_id: str = ""


@dataclass
class ToolCallStartEvent(BaseEvent):
    """Emitted when a tool call begins (triggers human-in-the-loop)."""
    
    type: EventType = field(default=EventType.TOOL_CALL_START)
    tool_call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_call_name: str = ""
    parent_message_id: str | None = None


@dataclass
class ToolCallArgsEvent(BaseEvent):
    """Emitted with tool call arguments."""
    
    type: EventType = field(default=EventType.TOOL_CALL_ARGS)
    tool_call_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallEndEvent(BaseEvent):
    """Emitted when tool call definition ends (waiting for result)."""
    
    type: EventType = field(default=EventType.TOOL_CALL_END)
    tool_call_id: str = ""


@dataclass
class ToolCallResultEvent(BaseEvent):
    """Emitted when tool call result is received."""
    
    type: EventType = field(default=EventType.TOOL_CALL_RESULT)
    tool_call_id: str = ""
    result: Any = None


@dataclass
class StateSnapshotEvent(BaseEvent):
    """Emitted with full state snapshot."""
    
    type: EventType = field(default=EventType.STATE_SNAPSHOT)
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateDeltaEvent(BaseEvent):
    """Emitted with state changes."""
    
    type: EventType = field(default=EventType.STATE_DELTA)
    delta: dict[str, Any] = field(default_factory=dict)
