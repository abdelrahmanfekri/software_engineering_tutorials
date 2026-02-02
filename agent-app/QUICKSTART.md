# Quick Start Guide

Get up and running with the Agent App in 5 minutes.

## 1. Prerequisites

- Python 3.10 or higher
- Anthropic API key ([get one here](https://console.anthropic.com/))

## 2. Installation

```bash
# Clone the repository
cd agent-app

# Install dependencies
pip install -r requirements.txt
```

## 3. Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
ANTHROPIC_API_KEY=your-actual-api-key-here
```

## 4. Run the CLI Agent

Test the agent in interactive mode:

```bash
python scripts/run_agent.py
```

Try these commands:
- `What's 25 * 48?` - Uses code execution tool
- `Search for Python news` - Uses web search tool
- `Show me users in the database` - Uses database query tool

## 5. Run the API Server

Start the FastAPI server:

```bash
python -m src.api.main
```

Visit http://localhost:8000/docs to see the interactive API documentation.

## 6. Make API Requests

### Using curl:

```bash
curl -X POST "http://localhost:8000/chat/message" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Calculate the square root of 144"
  }'
```

### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/chat/message",
    json={"message": "What can you help me with?"}
)

print(response.json()["response"])
```

### Using the examples:

```bash
python examples/usage_examples.py
```

## 7. Run Tests

Verify everything works:

```bash
pytest tests/ -v
```

## Next Steps

- Read [Architecture Documentation](docs/architecture.md)
- Add your own custom tools
- Deploy with Docker: `docker-compose up`
- Explore the [full README](README.md)

## Common Issues

**"No module named 'src'"**
```bash
# Make sure you're in the project root directory
cd agent-app
python -m src.api.main
```

**"API key not found"**
```bash
# Check your .env file exists and has the correct key
cat .env | grep ANTHROPIC_API_KEY
```

**"Connection refused"**
```bash
# Make sure the API server is running
python -m src.api.main
```

## Getting Help

- Check the [README](README.md) for detailed documentation
- Review [architecture docs](docs/architecture.md)
- Open an issue on GitHub
