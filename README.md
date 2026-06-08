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

## Architecture

This service speaks only the **canonical message contract** — it never knows about WhatsApp. See the parent's [`docs/ARCHITECTURE.md`](https://github.com/chasqui-stack/chasqui/blob/main/docs/ARCHITECTURE.md) (§5 contract, §6 domain model, §8 tool registry, §10 BSUID).

## License

[Apache-2.0](./LICENSE).
