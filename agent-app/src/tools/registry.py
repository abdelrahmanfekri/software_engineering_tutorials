"""Tool registry for managing and discovering tools."""

from typing import Dict, List, Optional
import structlog
from .base import BaseTool, ToolSchema

logger = structlog.get_logger()


class ToolRegistry:
    """Central registry for all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
        """
        schema = tool.get_schema()
        tool_name = schema.name
        
        if tool_name in self._tools:
            logger.warning(f"Tool {tool_name} already registered, overwriting")
        
        self._tools[tool_name] = tool
        logger.info(f"Registered tool: {tool_name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_all_schemas(self) -> List[ToolSchema]:
        """Get schemas for all registered tools.
        
        Returns:
            List of ToolSchema objects
        """
        return [tool.get_schema() for tool in self._tools.values()]
    
    def get_schemas_for_llm(self) -> List[Dict]:
        """Get tool schemas formatted for LLM tool calling.
        
        Returns:
            List of tool schemas in Anthropic format
        """
        schemas = []
        for tool in self._tools.values():
            schema = tool.get_schema()
            schemas.append({
                "name": schema.name,
                "description": schema.description,
                "input_schema": schema.parameters
            })
        return schemas
    
    async def execute_tool(self, name: str, **kwargs) -> Dict:
        """Execute a tool by name.
        
        Args:
            name: Tool name
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found in registry")
        
        logger.info(f"Executing tool: {name}", parameters=kwargs)
        result = await tool.execute(**kwargs)
        logger.info(f"Tool execution completed: {name}")
        
        return result
