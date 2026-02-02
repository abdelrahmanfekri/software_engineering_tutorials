"""Chat API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import structlog

from ...agent import Agent
from ...llm import LLMClient
from ...tools import create_default_registry
from ...memory import ConversationMemory
from ...utils.errors import AgentError, MaxIterationsError

logger = structlog.get_logger()

router = APIRouter(prefix="/chat", tags=["chat"])

# Global instances (in production, use dependency injection)
llm_client = None
tool_registry = None
agent = None


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    session_id: str
    steps: List[dict] = []
    total_iterations: int = 0
    total_tokens: int = 0


# Store conversations by session (in production, use Redis/DB)
conversations = {}


def get_or_create_agent(session_id: str) -> Agent:
    """Get or create an agent for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Agent instance
    """
    global llm_client, tool_registry
    
    if session_id not in conversations:
        # Create new conversation
        memory = ConversationMemory(max_messages=50)
        
        if llm_client is None:
            llm_client = LLMClient()
        
        if tool_registry is None:
            tool_registry = create_default_registry()
        
        agent_instance = Agent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            memory=memory,
        )
        
        conversations[session_id] = agent_instance
        logger.info(f"Created new agent for session: {session_id}")
    
    return conversations[session_id]


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Send a message to the agent and get a response.
    
    Args:
        request: Chat request with message
        
    Returns:
        Chat response with agent's answer
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{len(conversations)}"
        
        # Get or create agent
        agent = get_or_create_agent(session_id)
        
        logger.info(f"Processing message for session: {session_id}")
        
        # Run agent
        result = await agent.run(request.message)
        
        # Convert steps to dict
        steps = [
            {
                "thought": step.thought,
                "action": step.action.model_dump() if step.action else None,
                "observation": step.observation.model_dump() if step.observation else None,
            }
            for step in result.steps
        ]
        
        return ChatResponse(
            response=result.response,
            session_id=session_id,
            steps=steps,
            total_iterations=result.total_iterations,
            total_tokens=result.total_tokens,
        )
        
    except MaxIterationsError as e:
        logger.error(f"Max iterations exceeded: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except AgentError as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset a conversation session.
    
    Args:
        session_id: Session to reset
        
    Returns:
        Success message
    """
    if session_id in conversations:
        conversations[session_id].reset()
        del conversations[session_id]
        logger.info(f"Reset session: {session_id}")
        return {"message": f"Session {session_id} reset successfully"}
    
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions")
async def list_sessions():
    """List all active sessions.
    
    Returns:
        List of session IDs
    """
    return {"sessions": list(conversations.keys())}
