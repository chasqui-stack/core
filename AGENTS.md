# AGENTS.md — Chasqui Core

The **heart** of Chasqui: all business logic, the agent orchestrator, memory, RAG, the tool registry, and admin auth. Part of the [`chasqui-stack`](https://github.com/chasqui-stack/chasqui) stack — read the parent's [`docs/ARCHITECTURE.md`](https://github.com/chasqui-stack/chasqui/blob/main/docs/ARCHITECTURE.md) first.

## Stack & structure

- FastAPI · SQLModel · PostgreSQL + **pgvector** · Alembic · JWT · LangChain · LangGraph · `uv`.
- `app/core` (config, security, deps) · `app/models` · `app/schemas` · `app/controllers` · `app/services` · `app/modules/` (tool modules).

## Key design rules (see ARCHITECTURE)

- **Canonical entry point:** `POST /ingest` (§5, §6). The core speaks the canonical contract only — it never knows about WhatsApp or any channel.
- **Admin-only auth (§4):** JWT for admins. **No organizations, no member roles, no self-serve signup.** End users (contacts) never authenticate.
- **BSUID-first identity (§10):** `contacts.external_id` = WhatsApp BSUID; `wa_id` is optional/secondary.
- **Single conversation per contact** + `messages` history + long-term `memories` (pgvector). `conversations.mode` (`"agent" | "human"`, ADR-004) is checked FIRST by ingest: human mode persists the inbound, runs NO agent turn and returns an empty `messages` list (silence on every channel).
- **Inbound coalescing (ADR-008, `app/services/coalesce_worker.py`):** with `INBOUND_DEBOUNCE_SECONDS > 0` (default 5), `/ingest` persists the inbound as pending (`messages.processed_at` NULL), arms `conversations.debounce_due_at`, and acks empty — **no inline turn**. A Postgres worker (`FOR UPDATE SKIP LOCKED`, no broker) folds the burst into ONE `orchestrator.run_coalesced_turn` and dispatches via the send seam below — so **`CHANNEL_<CH>_SEND_URL` is required** when debounce > 0. `=0` = legacy synchronous reply in the `/ingest` body. A per-identity advisory lock (`ingest_service.acquire_turn_lock`, Etapa 1) serializes turns in both modes.
- **Canonical outbound seam (ADR-004, `app/services/channel_send.py`):** operator replies go through each gateway's `POST /send` (mirror of `/ingest`, same `INTERNAL_API_KEY`); the core resolves `CHANNEL_<CHANNEL>_SEND_URL` from `.env` — one var per channel, never a channel SDK in the core. Outbound media (`image`/`document`/`audio`) travels as a base64 `data:` URI in `media_url` (the mirror of inbound) + `filename` for documents; the core size-caps per type (5/25/16 MB) and also uploads it to the bucket post-send (log-and-NULL) so the timeline renders it. Gateway error `code`s (`WINDOW_EXPIRED`, `NO_WA_ID`, …) pass through to the admin. Handoff notifications (`app/services/notify_service.py`): optional `NOTIFY_WEBHOOK_URL` + optional SMTP (stdlib `smtplib`, any relay — Brevo/Mailgun/SES), both best-effort, fired as background tasks.
- **Tool Registry (§8):** tools are LangChain `@tool`/`StructuredTool` with Pydantic `args_schema`, injected via `ToolNode`/`bind_tools`, accessed through `ToolRuntime[TurnContext]`. Add capabilities as self-contained **modules** under `app/modules/` (a package exposing a module-level `module` attribute — auto-discovered at startup) — never by editing the core. Runtime enable/disable per tool lives in `agent_config.enabled_tools`; tool exceptions become error `ToolMessage`s (`app/services/agent_middleware.py`). Modules can also ship **tables** (`register_models()` — `registry.discover()` runs in `alembic/env.py` + `tests/conftest.py`), **admin routes** (`register_admin_routes()` → JWT-protected `/admin/modules/<name>`) and **config knobs** (`config_schema()` ↔ `agent_config.tool_config[config_key]`; keep schemas FLAT — str/int/float/bool only — so the admin auto-renders them as forms). Reference implementation: `app/modules/faq/` (RAG over `faq_entries`).
- **Admin panel API:** `GET/PUT /admin/config` (singleton; PUT validates `enabled_tools` names against the registry and `tool_config` values against each module's `config_schema()` — 422 on violations), `GET /admin/tools` (registry listing + JSON Schemas), `GET /admin/contacts[/{id}[/messages|/memories]]` (inspection — never serialize embeddings or media payloads; messages expose a `has_media` boolean; list carries `mode`/handoff/`last_inbound_at` + `?mode=` filter, attention-first order), `PUT /admin/contacts/{id}/mode` + `POST /admin/contacts/{id}/messages` (inbox writes — 409 in agent mode, send-then-persist, `meta.sent_by`), `GET /admin/media/{message_id}` (presigned URL as JSON for stored media), `GET /admin/modules/handoff/leads` (module-contributed).
- **Vector search is dim-aware (ADR-001):** always build cosine expressions with `app/core/vector_search.py` (never raw `.cosine_distance()` in queries) — at >2000 dims the halfvec expression must match the index or Postgres seq-scans silently. Memory is dedup'd/correctable via `save_memory`/`update_memory`/`forget_memory` (`app/modules/memory`).
- **The agent turn (`app/services/orchestrator.py`):** LangChain v1 `create_agent` — DB-editable system prompt (`agent_config` singleton) + history + pgvector memories + current message (multimodal content blocks gated by `app/core/llm_capabilities.py`). The LLM is swappable via `.env` only (`app/core/llm.py` / `init_chat_model`): google · anthropic · openai · openrouter · ollama.
- **Media (ADR-003):** the gateway inlines media as base64 `data:` URIs in canonical `media_url` (Meta URLs expire). With storage configured (`STORAGE_*` env, S3-compatible via boto3 — `app/core/storage.py`), the core uploads on persist (key `media/<contact_id>/<message_id>.<ext>` in `messages.media_url`) and `GET /admin/media/{message_id}` serves a presigned URL as JSON (never a redirect — `<img src>` can't send JWT). Storage unset = not persisted (degraded, not broken); upload failures never break the turn (log + NULL, the embeddings pattern). LLM history stays text-only either way; `media_id` stays in message `metadata`.
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
