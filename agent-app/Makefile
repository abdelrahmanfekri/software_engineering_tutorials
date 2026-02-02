.PHONY: help install test run api clean lint format docker-build

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make run          - Run CLI agent"
	@echo "  make api          - Run FastAPI server"
	@echo "  make clean        - Remove cache files"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make docker-build - Build Docker image"

install:
	pip install -e .
	pip install pytest pytest-asyncio pytest-cov black ruff mypy

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

run:
	python scripts/run_agent.py

api:
	python -m src.api.main

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/

docker-build:
	docker build -t agent-app:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env agent-app:latest
