.PHONY: install serve test lint typecheck fmt clean

# Install all dependencies (including dev)
install:
	uv sync --group dev

# Start the AIO gateway (FastAPI + Telegram bot)
serve:
	bash scripts/start.sh

# Run tests
test:
	uv run pytest -v

# Run tests with coverage
test-cov:
	uv run pytest --cov=aio --cov-report=term-missing

# Lint check
lint:
	uv run ruff check src/ tests/

# Type check
typecheck:
	uv run mypy src/

# Auto-format code
fmt:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

# Clean up caches
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
