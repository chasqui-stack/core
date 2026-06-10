"""Agent middleware (ARCHITECTURE §8) — runtime tool filtering + error safety.

LangChain v1 middleware hooks:
- `awrap_model_call`  → drop disabled tools before the model sees them.
- `awrap_tool_call`   → a tool exception becomes a ToolMessage, never a crash.

Only the async variants are implemented: the orchestrator always `ainvoke`s.
"""

import logging

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain_core.messages import ToolMessage

from app.services.agent_config_service import tool_enabled

logger = logging.getLogger(__name__)


class ToolFilterMiddleware(AgentMiddleware):
    """Only offer the model the tools enabled in agent_config (admin-editable)."""

    async def awrap_model_call(self, request: ModelRequest, handler):
        config = request.runtime.context.config
        allowed = [t for t in request.tools if tool_enabled(config, t.name)]
        if len(allowed) != len(request.tools):
            dropped = [t.name for t in request.tools if t not in allowed]
            logger.debug("Tools disabled for this turn: %s", dropped)
        return await handler(request.override(tools=allowed))


class ToolErrorMiddleware(AgentMiddleware):
    """Convert tool exceptions into an error ToolMessage so the agent recovers."""

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        try:
            return await handler(request)
        except Exception as exc:
            tool_name = request.tool_call.get("name", "?")
            logger.exception("Tool '%s' failed", tool_name)
            return ToolMessage(
                content=(
                    f"The tool '{tool_name}' failed: {exc}. "
                    "Apologize briefly and carry on without it."
                ),
                tool_call_id=request.tool_call["id"],
                status="error",
            )
