#!/usr/bin/env python3
"""CLI script to run the agent interactively."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.llm import LLMClient
from src.tools import create_default_registry
from src.memory import ConversationMemory
from src.utils.config import settings
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer()
    ]
)

logger = structlog.get_logger()


async def main():
    """Run the agent in interactive mode."""
    print("=" * 60)
    print("Agent CLI - Interactive Mode")
    print("=" * 60)
    print(f"Model: {settings.model_name}")
    print(f"Max iterations: {settings.max_iterations}")
    print("\nType 'exit' or 'quit' to end the session")
    print("Type 'reset' to clear conversation history")
    print("=" * 60)
    print()
    
    # Initialize components
    llm_client = LLMClient()
    tool_registry = create_default_registry()
    memory = ConversationMemory()
    
    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
        memory=memory,
    )
    
    print(f"Available tools: {', '.join(tool_registry.list_tools())}")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == "reset":
                agent.reset()
                print("\n[Conversation reset]\n")
                continue
            
            # Run agent
            print("\nAgent: ", end="", flush=True)
            
            try:
                result = await agent.run(user_input)
                
                print(result.response)
                print()
                
                # Show debug info
                if result.steps:
                    print(f"\n[Debug: {result.total_iterations} iterations, "
                          f"{result.total_tokens} tokens, "
                          f"{len(result.steps)} steps]")
                print()
                
            except Exception as e:
                print(f"\n[Error: {str(e)}]\n")
                logger.error("Agent error", error=str(e))
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        
        except Exception as e:
            logger.error("Unexpected error", error=str(e))
            print(f"\n[Unexpected error: {str(e)}]\n")


if __name__ == "__main__":
    asyncio.run(main())
