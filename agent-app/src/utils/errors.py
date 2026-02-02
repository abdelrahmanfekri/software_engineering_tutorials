"""Custom exception classes for the agent application."""


class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails."""
    pass


class MaxIterationsError(AgentError):
    """Raised when agent exceeds maximum iterations."""
    pass


class LLMError(AgentError):
    """Raised when LLM call fails."""
    pass


class ValidationError(AgentError):
    """Raised when input validation fails."""
    pass


class MemoryError(AgentError):
    """Raised when memory operations fail."""
    pass
