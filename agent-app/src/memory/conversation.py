"""Conversation memory management."""

from typing import List, Optional
from collections import deque
import structlog
from ..agent.types import Message, MessageRole

logger = structlog.get_logger()


class ConversationMemory:
    """Manages conversation history with context window limits."""
    
    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self.messages: deque = deque(maxlen=max_messages)
    
    def add_message(self, role: MessageRole, content: str, metadata: dict = None):
        """Add a message to conversation history.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
        """
        message = Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        logger.debug(f"Added {role.value} message to memory")
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get conversation messages.
        
        Args:
            limit: Optional limit on number of messages to return
            
        Returns:
            List of messages
        """
        if limit:
            return list(self.messages)[-limit:]
        return list(self.messages)
    
    def get_formatted_history(self) -> str:
        """Get conversation history as formatted string.
        
        Returns:
            Formatted conversation history
        """
        history = []
        for msg in self.messages:
            history.append(f"{msg.role.value}: {msg.content}")
        return "\n".join(history)
    
    def clear(self):
        """Clear all messages from memory."""
        self.messages.clear()
        logger.info("Cleared conversation memory")
    
    def get_context_size(self) -> int:
        """Estimate context size in characters.
        
        Returns:
            Approximate character count
        """
        return sum(len(msg.content) for msg in self.messages)
    
    def truncate_to_fit(self, max_chars: int):
        """Truncate old messages to fit within character limit.
        
        Args:
            max_chars: Maximum characters to keep
        """
        while self.get_context_size() > max_chars and len(self.messages) > 1:
            # Remove oldest message (but keep system message if it's first)
            if self.messages[0].role != MessageRole.SYSTEM:
                self.messages.popleft()
            elif len(self.messages) > 1:
                # Remove second message if first is system
                del self.messages[1]
        
        logger.info(f"Truncated memory to {len(self.messages)} messages")
