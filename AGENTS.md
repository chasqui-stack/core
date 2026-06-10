# AGENTS.md — Chasqui Core

The **heart** of Chasqui: all business logic, the agent orchestrator, memory, RAG, the tool registry, and admin auth. Part of the [`chasqui-stack`](https://github.com/chasqui-stack/chasqui) stack — read the parent's [`docs/ARCHITECTURE.md`](https://github.com/chasqui-stack/chasqui/blob/main/docs/ARCHITECTURE.md) first.

## Stack & structure

- FastAPI · SQLModel · PostgreSQL + **pgvector** · Alembic · JWT · LangChain · LangGraph · `uv`.
- `app/core` (config, security, deps) · `app/models` · `app/schemas` · `app/controllers` · `app/services` · `app/modules/` (tool modules).

## Key design rules (see ARCHITECTURE)

- **Canonical entry point:** `POST /ingest` (§5, §6). The core speaks the canonical contract only — it never knows about WhatsApp or any channel.
- **Admin-only auth (§4):** JWT for admins. **No organizations, no member roles, no self-serve signup.** End users (contacts) never authenticate.
- **BSUID-first identity (§10):** `contacts.external_id` = WhatsApp BSUID; `wa_id` is optional/secondary.
- **Single conversation per contact** + `messages` history + long-term `memories` (pgvector).
- **Tool Registry (§8):** tools are LangChain `@tool`/`StructuredTool` with Pydantic `args_schema`, injected via `ToolNode`/`bind_tools`, accessed through `ToolRuntime[TurnContext]`. Add capabilities as self-contained **modules** under `app/modules/` (a package exposing a module-level `module` attribute — auto-discovered at startup) — never by editing the core. Runtime enable/disable per tool lives in `agent_config.enabled_tools`; tool exceptions become error `ToolMessage`s (`app/services/agent_middleware.py`). Modules can also ship **tables** (`register_models()` — `registry.discover()` runs in `alembic/env.py` + `tests/conftest.py`), **admin routes** (`register_admin_routes()` → JWT-protected `/admin/modules/<name>`) and **config knobs** (`config_schema()` ↔ `agent_config.tool_config[config_key]`; keep schemas FLAT — str/int/float/bool only — so the admin auto-renders them as forms). Reference implementation: `app/modules/faq/` (RAG over `faq_entries`).
- **Admin panel API:** `GET/PUT /admin/config` (singleton; PUT validates `enabled_tools` names against the registry and `tool_config` values against each module's `config_schema()` — 422 on violations), `GET /admin/tools` (registry listing + JSON Schemas), `GET /admin/contacts[/{id}[/messages|/memories]]` (read-only inspection — never serialize embeddings or media payloads).
- **Vector search is dim-aware (ADR-001):** always build cosine expressions with `app/core/vector_search.py` (never raw `.cosine_distance()` in queries) — at >2000 dims the halfvec expression must match the index or Postgres seq-scans silently. Memory is dedup'd/correctable via `save_memory`/`update_memory`/`forget_memory` (`app/modules/memory`).
- **The agent turn (`app/services/orchestrator.py`):** LangChain v1 `create_agent` — DB-editable system prompt (`agent_config` singleton) + history + pgvector memories + current message (multimodal content blocks gated by `app/core/llm_capabilities.py`). The LLM is swappable via `.env` only (`app/core/llm.py` / `init_chat_model`): google · anthropic · openai · openrouter · ollama.
- **Media:** the gateway inlines media as base64 `data:` URIs in canonical `media_url` (Meta URLs expire). Current turn only — never persisted (history is text-only; `media_id` stays in message `metadata`).
- **English-only code (i18n posture):** code, comments, logs, API error details AND every LLM-facing string (tool docstrings, tool returns, prompt fragments) are English — the system-prompt rule *"always reply in the user's language"* is what localizes the agent. The one user-facing literal, `FALLBACK_REPLY`, is `.env`-configurable (set it in your users' language). Admin-panel i18n is a frontend concern (Sprint 5+).

## Dev

```bash
uv sync && make migrate && make dev   # serves /docs
make test
```
Every schema change = a SQLModel model + an Alembic migration.

DB-backed tests use an auto-created `<postgres_db>_test` database with transactional-rollback isolation (savepoints; nothing ever commits) — see `tests/conftest.py` and the README's Testing section. Don't add per-test TRUNCATE/cleanup logic.

## Planning

PRPs and the sprint plan live in the **parent repo** (`../PRPs`, `../docs`). Write a PRP before non-trivial features.

## Don't

- Add multi-tenant orgs/members or end-user auth.
- Return non-serializable data from tools.
- Break the canonical contract or commit secrets (`.env`, `.kamal/secrets`).
