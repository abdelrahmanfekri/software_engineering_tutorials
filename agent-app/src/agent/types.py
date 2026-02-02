"""Shared type definitions for the agent application."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    """Message role in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in a conversation."""
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ToolCall(BaseModel):
    """Represents a tool call made by the agent."""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_call_id: str


class ToolResult(BaseModel):
    """Result from a tool execution."""
    tool_call_id: str
    tool_name: str
    output: Any
    error: Optional[str] = None
    success: bool = True


class AgentStep(BaseModel):
    """A single step in the agent's reasoning process."""
    thought: str
    action: Optional[ToolCall] = None
    observation: Optional[ToolResult] = None


class AgentResponse(BaseModel):
    """Final response from the agent."""
    response: str
    steps: List[AgentStep] = Field(default_factory=list)
    total_iterations: int = 0
    total_tokens: int = 0
