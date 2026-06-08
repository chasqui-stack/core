"""
Tool Registry — the extension point for per-company differentiating logic.

A tool module is a self-contained unit that contributes one or more LangChain
tools (and, optionally, its own DB models, admin routes, and per-project
config). The orchestrator discovers registered modules at startup and feeds
their tools to the agent. See ARCHITECTURE.md §8.

This is the Sprint-0 scaffold: the protocol and an in-memory registry. The
orchestrator wiring (bind_tools / ToolNode) lands with the agent.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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
