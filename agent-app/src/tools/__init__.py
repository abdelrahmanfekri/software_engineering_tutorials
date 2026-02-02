"""Tools package initialization."""

from .base import BaseTool, ToolSchema
from .registry import ToolRegistry
from .web_search.tavily import WebSearchTool
from .code_execution.sandbox import CodeExecutionTool
from .database.query import DatabaseQueryTool

__all__ = [
    "BaseTool",
    "ToolSchema",
    "ToolRegistry",
    "WebSearchTool",
    "CodeExecutionTool",
    "DatabaseQueryTool",
]


def create_default_registry() -> ToolRegistry:
    """Create and populate a registry with default tools.
    
    Returns:
        ToolRegistry with all default tools registered
    """
    registry = ToolRegistry()
    
    # Register default tools
    registry.register(WebSearchTool())
    registry.register(CodeExecutionTool())
    registry.register(DatabaseQueryTool())
    
    return registry
