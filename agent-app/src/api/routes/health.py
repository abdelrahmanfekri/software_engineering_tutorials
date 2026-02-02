"""Health check endpoints."""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "agent-app"
    }


@router.get("/")
async def root():
    """Root endpoint.
    
    Returns:
        Welcome message
    """
    return {
        "message": "Agent App API",
        "version": "0.1.0",
        "docs": "/docs"
    }
