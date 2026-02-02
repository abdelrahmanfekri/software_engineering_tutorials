"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import Generator

from src.llm import LLMClient
from src.tools import create_default_registry
from src.memory import ConversationMemory
from src.agent import Agent


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    # In real tests, you'd use a proper mock
    # For now, this is a placeholder
    return LLMClient()


@pytest.fixture
def tool_registry():
    """Create a tool registry with default tools."""
    return create_default_registry()


@pytest.fixture
def conversation_memory():
    """Create a fresh conversation memory."""
    return ConversationMemory(max_messages=10)


@pytest.fixture
def agent(mock_llm_client, tool_registry, conversation_memory):
    """Create an agent instance for testing."""
    return Agent(
        llm_client=mock_llm_client,
        tool_registry=tool_registry,
        memory=conversation_memory,
        max_iterations=5,
    )
