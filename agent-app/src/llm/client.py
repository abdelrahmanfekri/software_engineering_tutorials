"""LLM client wrapper for Anthropic Claude."""

from typing import List, Dict, Any, Optional
import anthropic
import structlog
from ..utils.config import settings
from ..utils.errors import LLMError
from ..agent.types import Message, MessageRole

logger = structlog.get_logger()


class LLMClient:
    """Wrapper for Anthropic Claude API."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.model_name
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.total_tokens = 0
    
    async def generate(
        self,
        messages: List[Message],
        system_prompt: str = None,
        tools: List[Dict] = None,
        max_tokens: int = None,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate a response from Claude.
        
        Args:
            messages: Conversation history
            system_prompt: System prompt
            tools: Available tools for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Convert messages to Anthropic format
            formatted_messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
                if msg.role != MessageRole.SYSTEM
            ]
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "max_tokens": max_tokens or settings.max_tokens,
                "temperature": temperature,
                "messages": formatted_messages,
            }
            
            if system_prompt:
                request_params["system"] = system_prompt
            
            if tools:
                request_params["tools"] = tools
            
            logger.info("Calling Claude API", model=self.model)
            
            # Make API call
            response = self.client.messages.create(**request_params)
            
            # Track token usage
            self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
            
            logger.info(
                "Claude API response received",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            
            # Parse response
            result = {
                "content": [],
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            }
            
            # Extract content blocks
            for block in response.content:
                if block.type == "text":
                    result["content"].append({
                        "type": "text",
                        "text": block.text
                    })
                elif block.type == "tool_use":
                    result["content"].append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
            
            return result
            
        except anthropic.APIError as e:
            logger.error("Claude API error", error=str(e))
            raise LLMError(f"LLM API call failed: {str(e)}")
        except Exception as e:
            logger.error("Unexpected error in LLM client", error=str(e))
            raise LLMError(f"Unexpected LLM error: {str(e)}")
    
    def get_total_tokens(self) -> int:
        """Get total tokens used in this session.
        
        Returns:
            Total token count
        """
        return self.total_tokens
    
    def reset_token_count(self):
        """Reset the token counter."""
        self.total_tokens = 0
