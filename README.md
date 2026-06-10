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

## The agent

`POST /ingest` runs a real LangGraph turn (LangChain v1 `create_agent`):
DB-editable system prompt (`agent_config`, Sprint 5 admin UI) + conversation
history + long-term memories (pgvector) + the current message — multimodal
(image/audio content blocks) when the configured model supports it.

**Swappable LLM** — a `.env` change, never code (`app/core/llm.py`):

```bash
LLM_PROVIDER=google     LLM_MODEL=gemini-2.5-flash      # default (multimodal)
LLM_PROVIDER=anthropic  LLM_MODEL=claude-sonnet-4-6
LLM_PROVIDER=openai     LLM_MODEL=gpt-5-mini
LLM_PROVIDER=openrouter LLM_MODEL=vendor/model
LLM_PROVIDER=ollama     LLM_MODEL=llama3.3              # local, no key
```

Per-model vision/audio support is auto-detected (`app/core/llm_capabilities.py`);
unknown models degrade to text-only with a warning (override with
`LLM_SUPPORTS_VISION` / `LLM_SUPPORTS_AUDIO`).

**Embeddings are swappable too** (`app/core/embeddings.py`, via
`init_embeddings()`): `EMBEDDING_PROVIDER` google/openai/ollama — all pinned
to **768 dims** (the `Vector(768)` schema contract; switching providers
requires re-embedding). Rationale: parent repo
`docs/design/adr-001-embeddings-provider-dims.md`.

## Tool modules (the extension point)

Drop a self-contained package under `app/modules/` exposing a module-level
`module` attribute — it is discovered at startup and its tools reach the
agent **without touching core code**:

```python
# app/modules/my_feature/__init__.py
from langchain.tools import tool

@tool
def my_tool(query: str) -> str:
    """Cuándo y cómo debe usarla el modelo (el docstring es el manual)."""
    return "..."

class MyModule:
    name = "my_feature"
    def register_tools(self):
        return [my_tool]

module = MyModule()
```

Tools access the DB session / contact / conversation / config through
`runtime: ToolRuntime[TurnContext]` (`app/services/agent_context.py`).
Disable any tool at runtime via `agent_config.enabled_tools`
(`{"my_tool": false}`); tool exceptions become error `ToolMessage`s — the
graph never crashes. Shipped examples: `faq` (stub until Sprint 4 RAG),
`handoff` (human handoff + lead capture), `memory` (silent fact saving).
Full reference module: parent repo `docs/design/module-example-commercial-locations.md`.

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
