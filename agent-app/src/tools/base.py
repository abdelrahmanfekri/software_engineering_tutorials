"""Base tool interface for all tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel


class ToolSchema(BaseModel):
    """Schema definition for a tool."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Return the tool's schema definition.
        
        Returns:
            ToolSchema with name, description, and parameters
        """
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Dict containing the tool execution result
            
        Raises:
            ToolExecutionError: If execution fails
        """
        pass
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters against schema.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation - override for custom validation
        schema = self.get_schema()
        required_params = schema.parameters.get("required", [])
        return all(param in kwargs for param in required_params)
