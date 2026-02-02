from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # LLM Configuration
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    
    # Application
    environment: str = "development"
    log_level: str = "INFO"
    max_iterations: int = 10
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Database
    database_url: str
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["http://localhost:3000"]
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60
    
    # Tools
    web_search_enabled: bool = True
    code_execution_enabled: bool = True
    database_query_enabled: bool = True
    
    # Observability
    enable_tracing: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
