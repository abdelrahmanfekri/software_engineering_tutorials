"""Unit tests for conversation memory."""

import pytest
from src.memory import ConversationMemory
from src.agent.types import MessageRole


class TestConversationMemory:
    """Tests for conversation memory."""
    
    def test_add_message(self):
        """Test adding messages to memory."""
        memory = ConversationMemory(max_messages=10)
        
        memory.add_message(MessageRole.USER, "Hello")
        memory.add_message(MessageRole.ASSISTANT, "Hi there!")
        
        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[1].role == MessageRole.ASSISTANT
    
    def test_max_messages_limit(self):
        """Test that memory respects max messages limit."""
        memory = ConversationMemory(max_messages=3)
        
        for i in range(5):
            memory.add_message(MessageRole.USER, f"Message {i}")
        
        messages = memory.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "Message 2"  # Oldest kept
    
    def test_clear_memory(self):
        """Test clearing memory."""
        memory = ConversationMemory()
        
        memory.add_message(MessageRole.USER, "Hello")
        memory.clear()
        
        messages = memory.get_messages()
        assert len(messages) == 0
    
    def test_get_formatted_history(self):
        """Test getting formatted conversation history."""
        memory = ConversationMemory()
        
        memory.add_message(MessageRole.USER, "Hello")
        memory.add_message(MessageRole.ASSISTANT, "Hi!")
        
        history = memory.get_formatted_history()
        assert "user: Hello" in history
        assert "assistant: Hi!" in history
    
    def test_context_size(self):
        """Test context size calculation."""
        memory = ConversationMemory()
        
        memory.add_message(MessageRole.USER, "Hello")  # 5 chars
        memory.add_message(MessageRole.ASSISTANT, "Hi")  # 2 chars
        
        size = memory.get_context_size()
        assert size == 7
