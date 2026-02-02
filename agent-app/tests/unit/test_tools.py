"""Unit tests for tools."""

import pytest
from src.tools import WebSearchTool, CodeExecutionTool, DatabaseQueryTool
from src.tools.registry import ToolRegistry


class TestWebSearchTool:
    """Tests for web search tool."""
    
    @pytest.mark.asyncio
    async def test_web_search_execution(self):
        """Test web search tool execution."""
        tool = WebSearchTool()
        result = await tool.execute(query="test query", max_results=3)
        
        assert "query" in result
        assert result["query"] == "test query"
        assert "results" in result
        assert len(result["results"]) <= 3
    
    def test_web_search_schema(self):
        """Test web search tool schema."""
        tool = WebSearchTool()
        schema = tool.get_schema()
        
        assert schema.name == "web_search"
        assert "query" in schema.parameters["properties"]
        assert "query" in schema.parameters["required"]


class TestCodeExecutionTool:
    """Tests for code execution tool."""
    
    @pytest.mark.asyncio
    async def test_simple_calculation(self):
        """Test executing simple calculation."""
        tool = CodeExecutionTool()
        code = """
result = 2 + 2
print(f"Result: {result}")
"""
        result = await tool.execute(code=code)
        
        assert result["success"] is True
        assert "4" in result["stdout"]
    
    @pytest.mark.asyncio
    async def test_code_with_error(self):
        """Test code execution with error."""
        tool = CodeExecutionTool()
        code = "print(undefined_variable)"
        
        result = await tool.execute(code=code)
        
        assert result["success"] is False
        assert "error" in result


class TestDatabaseQueryTool:
    """Tests for database query tool."""
    
    @pytest.mark.asyncio
    async def test_query_users(self):
        """Test querying users table."""
        tool = DatabaseQueryTool()
        result = await tool.execute(
            query_description="Get all users",
            table="users"
        )
        
        assert "results" in result
        assert result["table"] == "users"
        assert len(result["results"]) > 0


class TestToolRegistry:
    """Tests for tool registry."""
    
    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = WebSearchTool()
        
        registry.register(tool)
        
        assert "web_search" in registry.list_tools()
        assert registry.get_tool("web_search") is not None
    
    def test_get_schemas_for_llm(self):
        """Test getting schemas formatted for LLM."""
        registry = ToolRegistry()
        registry.register(WebSearchTool())
        registry.register(CodeExecutionTool())
        
        schemas = registry.get_schemas_for_llm()
        
        assert len(schemas) == 2
        assert all("name" in s for s in schemas)
        assert all("description" in s for s in schemas)
        assert all("input_schema" in s for s in schemas)
