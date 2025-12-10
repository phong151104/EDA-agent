"""
A2A Protocol Message Types.

Implements Agent-to-Agent protocol message format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class MessageType(str, Enum):
    """A2A message types."""
    
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class TaskState(str, Enum):
    """A2A task states."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class A2AMessage:
    """
    A2A Protocol message.
    
    Follows the A2A specification for agent communication.
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    type: MessageType = MessageType.REQUEST
    sender: str = ""
    receiver: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    content: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    reply_to: str | None = None  # ID of message being replied to
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to A2A JSON format."""
        return {
            "jsonrpc": "2.0",
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "metadata": self.metadata,
            "replyTo": self.reply_to,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "A2AMessage":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            type=MessageType(data.get("type", "request")),
            sender=data.get("sender", ""),
            receiver=data.get("receiver", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            content=data.get("content"),
            metadata=data.get("metadata", {}),
            reply_to=data.get("replyTo"),
        )


@dataclass
class TaskRequest:
    """
    A2A Task request.
    
    Used to request an agent to perform a task.
    """
    
    task_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    input_data: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    priority: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "taskId": self.task_id,
            "name": self.name,
            "description": self.description,
            "inputData": self.input_data,
            "timeoutSeconds": self.timeout_seconds,
            "priority": self.priority,
        }


@dataclass
class TaskResponse:
    """
    A2A Task response.
    
    Contains the result of a task execution.
    """
    
    task_id: str
    state: TaskState
    output_data: Any = None
    error: str | None = None
    execution_time_ms: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "taskId": self.task_id,
            "state": self.state.value,
            "outputData": self.output_data,
            "error": self.error,
            "executionTimeMs": self.execution_time_ms,
        }


class A2AProtocol:
    """
    A2A Protocol handler.
    
    Manages message routing and task handling between agents.
    """
    
    def __init__(self):
        """Initialize protocol handler."""
        self._pending_tasks: dict[str, TaskRequest] = {}
        self._message_handlers: dict[str, Any] = {}
    
    def register_handler(
        self,
        agent_name: str,
        handler: Any,
    ) -> None:
        """
        Register a message handler for an agent.
        
        Args:
            agent_name: Name of the agent
            handler: Handler callable
        """
        self._message_handlers[agent_name] = handler
    
    async def send_message(
        self,
        message: A2AMessage,
    ) -> A2AMessage | None:
        """
        Send a message to an agent.
        
        Args:
            message: Message to send
            
        Returns:
            Response message if applicable
        """
        receiver = message.receiver
        
        if receiver not in self._message_handlers:
            return A2AMessage(
                type=MessageType.ERROR,
                sender="system",
                receiver=message.sender,
                content={"error": f"Unknown receiver: {receiver}"},
                reply_to=message.id,
            )
        
        handler = self._message_handlers[receiver]
        
        try:
            response = await handler(message)
            return response
        except Exception as e:
            return A2AMessage(
                type=MessageType.ERROR,
                sender=receiver,
                receiver=message.sender,
                content={"error": str(e)},
                reply_to=message.id,
            )
    
    async def submit_task(
        self,
        sender: str,
        receiver: str,
        task: TaskRequest,
    ) -> TaskResponse:
        """
        Submit a task to an agent.
        
        Args:
            sender: Sending agent
            receiver: Target agent
            task: Task to execute
            
        Returns:
            Task response
        """
        message = A2AMessage(
            type=MessageType.REQUEST,
            sender=sender,
            receiver=receiver,
            content=task.to_dict(),
        )
        
        self._pending_tasks[task.task_id] = task
        
        response = await self.send_message(message)
        
        if response and response.type == MessageType.RESPONSE:
            del self._pending_tasks[task.task_id]
            return TaskResponse(
                task_id=task.task_id,
                state=TaskState.COMPLETED,
                output_data=response.content,
            )
        elif response and response.type == MessageType.ERROR:
            del self._pending_tasks[task.task_id]
            return TaskResponse(
                task_id=task.task_id,
                state=TaskState.FAILED,
                error=response.content.get("error", "Unknown error"),
            )
        else:
            return TaskResponse(
                task_id=task.task_id,
                state=TaskState.PENDING,
            )
