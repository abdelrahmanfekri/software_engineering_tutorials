#!/usr/bin/env python3
"""Example usage of the agent application."""

import asyncio
from src.agent import Agent
from src.llm import LLMClient
from src.tools import create_default_registry
from src.memory import ConversationMemory


async def example_basic_conversation():
    """Example: Basic conversation without tools."""
    print("=" * 60)
    print("Example 1: Basic Conversation")
    print("=" * 60)
    
    llm = LLMClient()
    tools = create_default_registry()
    memory = ConversationMemory()
    agent = Agent(llm, tools, memory)
    
    response = await agent.run("Hello! What can you help me with?")
    print(f"Agent: {response.response}\n")


async def example_web_search():
    """Example: Using web search tool."""
    print("=" * 60)
    print("Example 2: Web Search")
    print("=" * 60)
    
    llm = LLMClient()
    tools = create_default_registry()
    memory = ConversationMemory()
    agent = Agent(llm, tools, memory)
    
    response = await agent.run("Search for recent developments in AI")
    print(f"Agent: {response.response}")
    print(f"Steps taken: {len(response.steps)}")
    print(f"Iterations: {response.total_iterations}\n")


async def example_code_execution():
    """Example: Using code execution tool."""
    print("=" * 60)
    print("Example 3: Code Execution")
    print("=" * 60)
    
    llm = LLMClient()
    tools = create_default_registry()
    memory = ConversationMemory()
    agent = Agent(llm, tools, memory)
    
    response = await agent.run("Calculate the factorial of 10")
    print(f"Agent: {response.response}")
    
    # Show tool usage
    for i, step in enumerate(response.steps, 1):
        if step.action:
            print(f"\nStep {i}:")
            print(f"  Tool: {step.action.tool_name}")
            print(f"  Result: {step.observation.output if step.observation else 'N/A'}")
    print()


async def example_database_query():
    """Example: Using database query tool."""
    print("=" * 60)
    print("Example 4: Database Query")
    print("=" * 60)
    
    llm = LLMClient()
    tools = create_default_registry()
    memory = ConversationMemory()
    agent = Agent(llm, tools, memory)
    
    response = await agent.run("Show me all users in the database")
    print(f"Agent: {response.response}\n")


async def example_multi_step():
    """Example: Multi-step reasoning with multiple tools."""
    print("=" * 60)
    print("Example 5: Multi-Step Reasoning")
    print("=" * 60)
    
    llm = LLMClient()
    tools = create_default_registry()
    memory = ConversationMemory()
    agent = Agent(llm, tools, memory)
    
    response = await agent.run(
        "Search for Python tutorials, then calculate how many results we found"
    )
    print(f"Agent: {response.response}")
    print(f"\nReasoning steps:")
    for i, step in enumerate(response.steps, 1):
        print(f"{i}. {step.thought}")
    print()


async def example_conversation_context():
    """Example: Multi-turn conversation with context."""
    print("=" * 60)
    print("Example 6: Conversation Context")
    print("=" * 60)
    
    llm = LLMClient()
    tools = create_default_registry()
    memory = ConversationMemory()
    agent = Agent(llm, tools, memory)
    
    # First message
    response1 = await agent.run("My favorite number is 42")
    print(f"User: My favorite number is 42")
    print(f"Agent: {response1.response}\n")
    
    # Second message - references context
    response2 = await agent.run("Can you calculate its factorial?")
    print(f"User: Can you calculate its factorial?")
    print(f"Agent: {response2.response}\n")


async def main():
    """Run all examples."""
    examples = [
        example_basic_conversation,
        example_web_search,
        example_code_execution,
        example_database_query,
        example_multi_step,
        example_conversation_context,
    ]
    
    for example in examples:
        try:
            await example()
            await asyncio.sleep(1)  # Brief pause between examples
        except Exception as e:
            print(f"Error in example: {e}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Agent App - Example Usage")
    print("=" * 60 + "\n")
    
    asyncio.run(main())
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)
