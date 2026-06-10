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
    # def config_schema(self) -> type | None: ...           # per-project settings


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
