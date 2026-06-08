.PHONY: dev migrate makemigrations test install sync

# Development server
dev:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8090

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
