"""Core agent implementation with ReAct pattern."""

from typing import List, Optional
import structlog
from .types import Message, MessageRole, AgentResponse, AgentStep, ToolCall, ToolResult
from .prompts import SYSTEM_PROMPT, format_tool_descriptions
from ..llm.client import LLMClient
from ..tools.registry import ToolRegistry
from ..memory.conversation import ConversationMemory
from ..utils.errors import MaxIterationsError, ToolExecutionError
from ..utils.config import settings

logger = structlog.get_logger()


class Agent:
    """Main agent orchestrator using ReAct pattern."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        memory: ConversationMemory,
        max_iterations: int = None,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.memory = memory
        self.max_iterations = max_iterations or settings.max_iterations
        self.steps: List[AgentStep] = []
    
    async def run(self, user_input: str) -> AgentResponse:
        """Run the agent with user input.
        
        Args:
            user_input: User's question or request
            
        Returns:
            AgentResponse with final answer and reasoning steps
        """
        logger.info("Agent started", input=user_input)
        
        # Add user message to memory
        self.memory.add_message(MessageRole.USER, user_input)
        self.steps = []
        
        # Get tool schemas for LLM
        tool_schemas = self.tools.get_schemas_for_llm()
        
        # Prepare system prompt with tool descriptions
        tool_descriptions = format_tool_descriptions(tool_schemas)
        system_prompt = f"{SYSTEM_PROMPT}\n\nAvailable Tools:\n{tool_descriptions}"
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{self.max_iterations}")
            
            try:
                # Get messages for LLM
                messages = self.memory.get_messages()
                
                # Call LLM
                response = await self.llm.generate(
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=tool_schemas if tool_schemas else None,
                )
                
                # Check if LLM wants to use tools
                has_tool_use = any(
                    block["type"] == "tool_use" 
                    for block in response["content"]
                )
                
                if not has_tool_use:
                    # No tool use - this is the final answer
                    final_text = ""
                    for block in response["content"]:
                        if block["type"] == "text":
                            final_text += block["text"]
                    
                    # Add assistant response to memory
                    self.memory.add_message(MessageRole.ASSISTANT, final_text)
                    
                    logger.info("Agent completed successfully")
                    
                    return AgentResponse(
                        response=final_text,
                        steps=self.steps,
                        total_iterations=iteration,
                        total_tokens=self.llm.get_total_tokens(),
                    )
                
                # Process tool calls
                tool_results = []
                assistant_content_blocks = []
                
                for block in response["content"]:
                    if block["type"] == "text":
                        # Record thinking
                        step = AgentStep(thought=block["text"])
                        self.steps.append(step)
                        assistant_content_blocks.append(block)
                    
                    elif block["type"] == "tool_use":
                        tool_call = ToolCall(
                            tool_name=block["name"],
                            tool_input=block["input"],
                            tool_call_id=block["id"],
                        )
                        
                        # Execute tool
                        try:
                            logger.info(f"Executing tool: {block['name']}")
                            result = await self.tools.execute_tool(
                                block["name"],
                                **block["input"]
                            )
                            
                            tool_result = ToolResult(
                                tool_call_id=block["id"],
                                tool_name=block["name"],
                                output=result,
                                success=True,
                            )
                            
                            # Record step with action and observation
                            step = AgentStep(
                                thought=f"Using {block['name']}",
                                action=tool_call,
                                observation=tool_result,
                            )
                            self.steps.append(step)
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": str(result),
                            })
                            
                        except ToolExecutionError as e:
                            logger.error(f"Tool execution failed: {e}")
                            tool_result = ToolResult(
                                tool_call_id=block["id"],
                                tool_name=block["name"],
                                output=None,
                                error=str(e),
                                success=False,
                            )
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": f"Error: {str(e)}",
                                "is_error": True,
                            })
                        
                        assistant_content_blocks.append(block)
                
                # Add assistant message with tool use to memory
                # This is important for maintaining conversation context
                assistant_msg_content = ""
                for block in assistant_content_blocks:
                    if block["type"] == "text":
                        assistant_msg_content += block["text"] + "\n"
                    elif block["type"] == "tool_use":
                        assistant_msg_content += f"[Using tool: {block['name']}]\n"
                
                self.memory.add_message(MessageRole.ASSISTANT, assistant_msg_content.strip())
                
                # Add tool results to memory as user messages (this is how Claude expects it)
                tool_results_str = "\n".join([
                    f"Tool {r.get('tool_use_id', 'unknown')}: {r['content']}"
                    for r in tool_results
                ])
                self.memory.add_message(MessageRole.USER, f"[Tool Results]\n{tool_results_str}")
                
            except Exception as e:
                logger.error(f"Error in agent iteration: {e}")
                raise
        
        # Max iterations reached
        raise MaxIterationsError(
            f"Agent exceeded maximum iterations ({self.max_iterations})"
        )
    
    def reset(self):
        """Reset the agent state."""
        self.steps = []
        self.memory.clear()
        self.llm.reset_token_count()
        logger.info("Agent reset")
