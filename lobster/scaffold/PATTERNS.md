# Scaffold Production Patterns

Verified against: packages/lobster-metabolomics/ (v1.0.11)
Date: 2026-02-28

## pyproject.toml
- Build system: setuptools
- Entry point group: "lobster.agents"
- Entry point value: "lobster.agents.{domain}.{agent_name}:AGENT_CONFIG"
- State entry point group: "lobster.states"
- Namespace config: packages.find = {where=["."], include=["lobster*"], namespaces=true}
- Python requires: ">=3.12,<3.14"
- Core dependency: lobster-ai~=1.0.0

## __init__.py
- State classes always importable (unconditional)
- Agent factory + config + prompts in try/except (graceful degradation)
- {AGENT_NAME}_AVAILABLE flag
- __all__ with all public exports

## agent.py
- AGENT_CONFIG at line 14 (after docstring + single import)
- Only import before AGENT_CONFIG: `from lobster.config.agent_registry import AgentRegistryConfig`
- Heavy imports after AGENT_CONFIG (lines 26+)
- Factory returns: create_react_agent(model=, tools=, prompt=, name=, state_schema=)
- Uses `prompt=` parameter (NOT state_modifier=)
- Lazy prompt import inside factory body

## shared_tools.py
- Factory function: create_shared_tools(data_manager, *services, force_platform_type=None)
- Returns: List[Callable]
- Tools defined as @tool inside factory (closure over data_manager + services)
- Each tool calls: data_manager.log_tool_usage(tool_name=, parameters=, description=, ir=ir)
- AQUADIF metadata assigned after @tool: tool.metadata = {"categories": [...], "provenance": bool}
- AQUADIF tags assigned: tool.tags = [...]

## state.py
- Inherits: langgraph.prebuilt.chat_agent_executor.AgentState
- Required field: next: str = ""
- Domain-specific fields with defaults

## config.py
- @dataclass for platform/domain config
- Dict registry: PLATFORM_CONFIGS
- Detection function: detect_platform_type(adata)
- Getter: get_platform_config(platform_type)

## prompts.py
- Factory function: create_{agent_name}_prompt() -> str
- XML sections: <Identity_And_Role>, <Your_Environment>, <Your_Responsibilities>, etc.
