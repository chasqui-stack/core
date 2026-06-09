# Chasqui Core

FastAPI + LangGraph backend — the heart of [Chasqui](https://github.com/chasqui-stack/chasqui), the open-source stack for building WhatsApp AI agents.

Handles the canonical `/ingest` entry point, the agent orchestrator (LangGraph), conversation memory, FAQ-RAG (pgvector), the pluggable **tool registry**, and admin authentication.

## Stack

FastAPI · SQLModel · PostgreSQL + pgvector · Alembic · JWT · LangChain · LangGraph · `uv`.

## Local dev

```bash
cp .env.example .env     # configure DB + JWT + LLM key
uv sync
make migrate
make dev                 # http://localhost:8090  (/docs)
```

## Testing

```bash
make test                # = uv run pytest
```

DB-backed tests run against a dedicated **`<postgres_db>_test`** database (e.g. `chasqui_test`), **auto-created on first run** from the same connection settings as the app — your dev database is never touched. No extra setup: the only requirement is that your Postgres user can `CREATE DATABASE`.

Isolation follows the SQLAlchemy-recommended transactional-rollback pattern: each test runs inside an outer transaction (app `commit()`s become savepoints) that is rolled back at the end, so **nothing is ever written** to the test database and tests can't contaminate each other. See `tests/conftest.py`.

To point tests at another Postgres (e.g. CI), export `POSTGRES_HOST` / `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` — real env vars take priority over `.env`.

## Architecture

This service speaks only the **canonical message contract** — it never knows about WhatsApp. See the parent's [`docs/ARCHITECTURE.md`](https://github.com/chasqui-stack/chasqui/blob/main/docs/ARCHITECTURE.md) (§5 contract, §6 domain model, §8 tool registry, §10 BSUID).

## License

[Apache-2.0](./LICENSE).
