# Agent App - Modern LLM Agent Application

A production-ready LLM agent application built with Claude, featuring tool calling, web search, code execution, and database queries.

## Features

- 🤖 **Intelligent Agent**: ReAct pattern-based reasoning with Claude
- 🛠️ **Tool Ecosystem**: Extensible tool system with web search, code execution, and database queries
- 💾 **Memory Management**: Conversation history with context window handling
- 🚀 **FastAPI Backend**: RESTful API with async support
- 📊 **Observability**: Structured logging and tracing
- 🧪 **Tested**: Unit and integration tests
- 🐳 **Docker Ready**: Containerized deployment

## Architecture

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│      FastAPI Application        │
│   (API Routes & Middleware)     │
└──────────┬──────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│        Agent Core                │
│   (ReAct Loop Orchestration)     │
└──────────┬───────────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐    ┌────────────┐
│  LLM   │    │   Tools    │
│ Client │    │  Registry  │
└────────┘    └─────┬──────┘
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │  Web   │ │  Code  │ │Database│
    │ Search │ │  Exec  │ │ Query  │
    └────────┘ └────────┘ └────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Anthropic API key

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd agent-app
```

2. Install dependencies:
```bash
pip install -e .
# or with poetry
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Running the Application

#### CLI Mode (Interactive)

```bash
python scripts/run_agent.py
```

This launches an interactive CLI where you can chat with the agent:

```
You: What's 123 * 456?
Agent: Let me calculate that for you...
[Uses execute_code tool]
The result is 56,088.
```

#### API Mode

```bash
# Start the FastAPI server
python -m src.api.main

# Or with uvicorn directly
uvicorn src.api.main:app --reload
```

Then visit `http://localhost:8000/docs` for the interactive API documentation.

### Making API Requests

```bash
# Send a message
curl -X POST "http://localhost:8000/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Search for recent AI news"}'

# Reset a session
curl -X POST "http://localhost:8000/chat/reset/session_0"

# List active sessions
curl "http://localhost:8000/chat/sessions"
```

## Project Structure

```
agent-app/
├── src/
│   ├── agent/          # Core agent logic (ReAct loop)
│   ├── tools/          # Tool implementations
│   ├── llm/            # LLM client wrapper
│   ├── memory/         # Conversation memory
│   ├── api/            # FastAPI application
│   └── utils/          # Utilities and config
├── tests/              # Unit and integration tests
├── scripts/            # CLI and utility scripts
├── config/             # Configuration files
└── docs/               # Documentation
```

## Available Tools

### 1. Web Search
Search the web for current information.
```python
result = await agent.run("What's the latest news about AI?")
```

### 2. Code Execution
Execute Python code for calculations and data processing.
```python
result = await agent.run("Calculate the first 10 Fibonacci numbers")
```

### 3. Database Query
Query database tables with natural language.
```python
result = await agent.run("Show me all users in the database")
```

## Adding Custom Tools

1. Create a new tool class:

```python
from src.tools.base import BaseTool, ToolSchema

class MyCustomTool(BaseTool):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="my_tool",
            description="Does something useful",
            parameters={
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                },
                "required": ["param"]
            }
        )
    
    async def execute(self, param: str, **kwargs):
        # Your tool logic here
        return {"result": f"Processed: {param}"}
```

2. Register it:

```python
from src.tools import ToolRegistry
from my_module import MyCustomTool

registry = ToolRegistry()
registry.register(MyCustomTool())
```

## Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/unit/test_tools.py

# Verbose mode
pytest -v
```

## Configuration

Environment variables (`.env`):

```bash
# Required
ANTHROPIC_API_KEY=your-key-here

# Optional
MODEL_NAME=claude-sonnet-4-20250514
MAX_ITERATIONS=10
LOG_LEVEL=INFO
```

## Development

### Code Style

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Project Workflow

1. Create a feature branch
2. Write tests first (TDD)
3. Implement the feature
4. Ensure tests pass
5. Format and lint code
6. Submit PR

## Production Considerations

### Security
- Add authentication middleware
- Validate all tool inputs
- Sandbox code execution properly
- Rate limit API endpoints
- Use secrets management (not .env files)

### Scalability
- Use Redis for session storage
- Implement connection pooling for database
- Add caching layer for LLM responses
- Use message queue for async tool execution

### Monitoring
- Add Prometheus metrics
- Integrate with APM (e.g., DataDog, New Relic)
- Set up alerting for errors
- Track token usage and costs

## Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
- Ensure you're in the project root
- Run `pip install -e .`

**"API key not found"**
- Check `.env` file exists
- Verify `ANTHROPIC_API_KEY` is set

**"Max iterations exceeded"**
- Increase `MAX_ITERATIONS` in config
- Check if agent is stuck in a loop

## License

MIT

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## Support

- Documentation: [docs/](docs/)
- Issues: GitHub Issues
- Discussions: GitHub Discussions
