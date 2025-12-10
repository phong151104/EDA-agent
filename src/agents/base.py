"""
Base Agent class with A2A protocol support.

Provides common functionality for all agents in the EDA system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import config


class AgentRole(str, Enum):
    """Agent role identifiers."""
    
    PLANNER = "planner"
    CRITIC = "critic"
    CODE_AGENT = "code_agent"
    ANALYST = "analyst"


@dataclass
class AgentCard:
    """
    A2A Agent Card - Describes agent capabilities.
    
    Based on A2A protocol specification.
    """
    
    name: str
    description: str
    role: AgentRole
    capabilities: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to A2A-compatible dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "role": self.role.value,
            "capabilities": self.capabilities,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "version": self.version,
        }


@dataclass
class AgentMessage:
    """
    Message exchanged between agents.
    
    Follows A2A message format.
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    sender: str = ""
    receiver: str = ""
    content: Any = None
    message_type: str = "request"  # request, response, feedback
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "messageType": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all EDA agents.
    
    Provides:
    - LLM client management
    - A2A protocol message handling
    - Common execution patterns
    - Logging and tracing
    """
    
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ):
        """
        Initialize base agent.
        
        Args:
            model: LLM model to use (defaults to config)
            temperature: LLM temperature
            system_prompt: System prompt for the agent
        """
        self.model_name = model or config.openai.model
        self.temperature = temperature
        self._system_prompt = system_prompt
        self._llm: ChatOpenAI | None = None
        self._message_history: list[BaseMessage] = []
    
    @property
    @abstractmethod
    def agent_card(self) -> AgentCard:
        """Return the agent's capability card."""
        ...
    
    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return self.agent_card.role
    
    @property
    def name(self) -> str:
        """Get agent name."""
        return self.agent_card.name
    
    @property
    def llm(self) -> ChatOpenAI:
        """Get or create LLM client."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=config.openai.api_key,
            )
        return self._llm
    
    @property
    def system_prompt(self) -> str:
        """Get system prompt for this agent."""
        if self._system_prompt:
            return self._system_prompt
        return self._default_system_prompt()
    
    @abstractmethod
    def _default_system_prompt(self) -> str:
        """Return default system prompt for this agent type."""
        ...
    
    @abstractmethod
    async def process(self, input_data: InputT) -> OutputT:
        """
        Process input and produce output.
        
        This is the main entry point for agent execution.
        
        Args:
            input_data: Input data specific to this agent type
            
        Returns:
            Output data specific to this agent type
        """
        ...
    
    async def invoke_llm(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """
        Invoke the LLM with messages.
        
        Args:
            messages: Messages to send to LLM
            **kwargs: Additional LLM parameters
            
        Returns:
            AI response message
        """
        # Prepend system message if not present
        if messages and not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_prompt), *messages]
        
        response = await self.llm.ainvoke(messages, **kwargs)
        self._message_history.extend(messages)
        self._message_history.append(response)
        
        return response  # type: ignore
    
    def create_message(
        self,
        receiver: str,
        content: Any,
        message_type: str = "request",
        **metadata: Any,
    ) -> AgentMessage:
        """
        Create an A2A message to another agent.
        
        Args:
            receiver: Target agent name
            content: Message content
            message_type: Type of message
            **metadata: Additional metadata
            
        Returns:
            AgentMessage instance
        """
        return AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=message_type,
            metadata=metadata,
        )
    
    def clear_history(self) -> None:
        """Clear message history."""
        self._message_history.clear()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, role={self.role.value})"
