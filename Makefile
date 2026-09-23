.PHONY: dev migrate makemigrations test install sync eval-routing

# Development server (port from .env PORT, default 8090)
dev:
	uv run python -c "import uvicorn; from app.core.config import settings; uvicorn.run('app.main:app', host='0.0.0.0', port=settings.port, reload=True)"

# Run migrations
migrate:
	uv run alembic upgrade head

# Create new migration (usage: make makemigrations m="migration message")
makemigrations:
	uv run alembic revision --autogenerate -m "$(m)"

# Run tests
test:
	uv run pytest

# Install dependencies
install:
	uv sync

# Sync dependencies (same as install)
sync:
	uv sync

# FAQ vs documents routing eval against the real LLM, in its own <db>_eval
# database (usage: make eval-routing runs=5)
eval-routing:
	uv run python scripts/eval_routing.py --runs $(or $(runs),1)
