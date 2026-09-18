"""
Tool Registry — the extension point for per-company differentiating logic.

A tool module is a self-contained unit that contributes one or more LangChain
tools (and, optionally, its own DB models, admin routes, and per-project
config). The orchestrator discovers registered modules at startup and feeds
their tools to the agent. See ARCHITECTURE.md §8.

Modules are auto-discovered at startup: every package under `app/modules/`
that exposes a module-level `module` attribute implementing the protocol is
registered. Dropping a new folder = the agent gains its tools — the core
stays untouched.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolModule(Protocol):
    """Contract every tool module implements."""

    name: str

    def register_tools(self) -> list[Any]:
        """Return the LangChain tools this module contributes."""
        ...

    # Optional hooks (implement as needed):
    # def register_models(self) -> list[type]: ...          # SQLModel tables + Alembic
    # def register_admin_routes(self, router) -> None: ...  # admin CRUD/config UI backing
    # def config_schema(self) -> type | None: ...           # per-project settings (admin auto-forms)
    # config_key: str = <name>   # key inside agent_config.tool_config (faq uses "faq_search").
    #                            # Keep config_schema() FLAT (str/int/float/bool fields only) —
    #                            # that's what the admin SchemaForm renders.
    # async def system_prompt_fragment(self, context, query) -> str | None: ...
    #     # One block appended to the system prompt each turn (ADR-012).
    #     # `context` is the turn's TurnContext (session, contact_id,
    #     # conversation_id, config); `query` is the inbound text. Return None
    #     # to stay silent. English only, like every LLM-facing string.


_MODULES: list[ToolModule] = []


def register_module(module: ToolModule) -> None:
    """Register a tool module (call from the module's setup)."""
    _MODULES.append(module)


def get_modules() -> list[ToolModule]:
    """All registered modules."""
    return list(_MODULES)


def get_tools() -> list[Any]:
    """Flattened list of every registered module's tools (for bind_tools/ToolNode)."""
    tools: list[Any] = []
    for module in _MODULES:
        tools.extend(module.register_tools())
    return tools


def get_models() -> list[type]:
    """Flattened list of every module's SQLModel tables.

    Importing the module package is what actually lands the tables in
    `SQLModel.metadata` (so call `discover()` before using the metadata —
    conftest and alembic/env.py do); this hook makes the contract explicit
    and feeds future tooling (`chasqui generate module`).
    """
    models: list[type] = []
    for module in _MODULES:
        register = getattr(module, "register_models", None)
        if register is not None:
            models.extend(register())
    return models


async def get_prompt_fragments(context: Any, query: str) -> list[str]:
    """Collect each module's optional system-prompt contribution (ADR-012).

    A module opts in with `async def system_prompt_fragment(context, query)`;
    `None`/empty means it has nothing to say this turn. Failures are isolated:
    a broken fragment is logged and skipped — it never breaks the turn.
    """
    discover()  # idempotent — fragments may be collected before the agent is built
    fragments: list[str] = []
    for module in _MODULES:
        hook = getattr(module, "system_prompt_fragment", None)
        if hook is None:
            continue
        try:
            fragment = await hook(context, query)
        except Exception:
            logger.exception(
                "system_prompt_fragment failed for module '%s' — skipped", module.name
            )
            continue
        if fragment:
            fragments.append(fragment)
    return fragments


def mount_admin_routes(router: Any) -> None:
    """Give every module a chance to mount its admin endpoints.

    Each module gets its own sub-router under `/admin/modules/<module.name>`
    (auth is enforced by the parent router in app/main.py). A module opts in
    by implementing `register_admin_routes(router)`.
    """
    from fastapi import APIRouter

    for module in _MODULES:
        register = getattr(module, "register_admin_routes", None)
        if register is None:
            continue
        sub_router = APIRouter(prefix=f"/{module.name}", tags=[f"admin-{module.name}"])
        register(sub_router)
        router.include_router(sub_router)
        logger.info("Admin routes mounted for module '%s'", module.name)


def discover() -> list[ToolModule]:
    """Scan app/modules/ packages and register every exposed `module`.

    Idempotent (safe to call from startup and tests). A package opts in by
    defining `module = MyModule()` at its top level.
    """
    import app.modules as pkg

    registered_names = {m.name for m in _MODULES}
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name == "registry":
            continue
        imported = importlib.import_module(f"app.modules.{info.name}")
        candidate = getattr(imported, "module", None)
        if candidate is None or not isinstance(candidate, ToolModule):
            continue
        if candidate.name in registered_names:
            continue
        register_module(candidate)
        registered_names.add(candidate.name)
        logger.info(
            "Tool module '%s' registered: %s",
            candidate.name,
            [t.name for t in candidate.register_tools()],
        )
    return get_modules()
