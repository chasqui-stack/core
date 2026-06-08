"""
Tool modules — the Chasqui extension point (see ARCHITECTURE §8).

Each tool module lives under `app/modules/<name>/` and implements the
`ToolModule` protocol from `app.modules.registry`. The orchestrator (later
sprint) discovers registered modules and feeds their LangChain tools to the
agent. Add capabilities as modules — never by editing the core.
"""
