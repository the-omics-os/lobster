#!/usr/bin/env python3
"""
Render the FULL agent LLM context: system prompt + tool schemas.

Shows EXACTLY what any agent model sees on every turn —
the system prompt AND the OpenAI-format tool definitions.

Usage:
    python scripts/agent_prompt_concat.py data_expert_agent        # full output
    python scripts/agent_prompt_concat.py transcriptomics_expert   # any agent
    python scripts/agent_prompt_concat.py data_expert_agent --prompt-only
    python scripts/agent_prompt_concat.py data_expert_agent --tools-only
    python scripts/agent_prompt_concat.py data_expert_agent --json
    python scripts/agent_prompt_concat.py --list                   # list all agents
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

# Add repo root so `lobster` package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Tool schema extraction
# ---------------------------------------------------------------------------


def _convert_tool_to_schema(t) -> dict:
    """Convert a LangChain @tool to its OpenAI function-calling schema."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return convert_to_openai_tool(t)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4)."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Agent introspection
# ---------------------------------------------------------------------------


def list_all_agents():
    """List all discovered agents with metadata."""
    from lobster.core.component_registry import component_registry

    all_agents = component_registry.list_agents()

    # Build child set for hierarchy display
    child_names = set()
    for cfg in all_agents.values():
        if cfg.child_agents:
            child_names.update(cfg.child_agents)

    print(f"Discovered {len(all_agents)} agents:\n")
    for name, cfg in sorted(all_agents.items()):
        marker = "  (child)" if name in child_names else ""
        children = f" → [{', '.join(cfg.child_agents)}]" if cfg.child_agents else ""
        print(f"  {cfg.display_name} ({name}){children}{marker}")
        print(f"    Factory: {cfg.factory_function}")
        print()


def extract_agent_prompt_and_tools(agent_name: str, workspace_path: Path):
    """Extract the system prompt and tools for a given agent.

    Creates the agent via its factory function (same path as graph.py)
    and introspects the compiled Pregel graph to extract prompt and tools.

    Returns:
        (system_prompt, tools, agent_config)
    """
    from lobster.core.component_registry import component_registry
    from lobster.core.data_manager_v2 import DataManagerV2
    from lobster.config.agent_registry import import_agent_factory
    from lobster.config.settings import get_settings

    agent_config = component_registry.get_agent(agent_name)
    if agent_config is None:
        print(f"Error: Agent '{agent_name}' not found.", file=sys.stderr)
        print("Use --list to see available agents.", file=sys.stderr)
        sys.exit(1)

    data_manager = DataManagerV2(workspace_path=workspace_path)

    # Import the factory function
    try:
        factory_function = import_agent_factory(agent_config.factory_function)
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        print(f"Error: Could not import factory for '{agent_name}': {e}", file=sys.stderr)
        sys.exit(1)

    # Build kwargs matching factory signature
    factory_kwargs = {
        "data_manager": data_manager,
        "callback_handler": None,
        "agent_name": agent_config.name,
    }

    sig = inspect.signature(factory_function)
    if "delegation_tools" in sig.parameters:
        factory_kwargs["delegation_tools"] = None
    if "subscription_tier" in sig.parameters:
        factory_kwargs["subscription_tier"] = "free"
    if "provider_override" in sig.parameters:
        factory_kwargs["provider_override"] = None
    if "model_override" in sig.parameters:
        factory_kwargs["model_override"] = None
    if "workspace_path" in sig.parameters:
        factory_kwargs["workspace_path"] = workspace_path
    if "store" in sig.parameters:
        factory_kwargs["store"] = None

    # Create the agent
    try:
        agent = factory_function(**factory_kwargs)
    except Exception as e:
        print(f"Error: Factory execution failed for '{agent_name}': {e}", file=sys.stderr)
        sys.exit(1)

    # Extract system prompt from the agent's prompt attribute or nodes
    system_prompt = _extract_system_prompt(agent, agent_name)

    # Extract tools from the agent's tool node
    tools = _extract_tools(agent, agent_name)

    return system_prompt, tools, agent_config


def _extract_system_prompt(agent, agent_name: str) -> str:
    """Extract system prompt from a compiled ReAct agent.

    LangGraph's create_react_agent stores the prompt in different places
    depending on version. We try multiple extraction paths.
    """
    # Path 1: Check agent.nodes for the "agent" node which contains the prompt
    try:
        agent_node = agent.nodes.get("agent")
        if agent_node:
            # The ReAct agent stores prompt in the bound runnable
            runnable = getattr(agent_node, "runnable", None) or getattr(agent_node, "bound", None)
            if runnable:
                # Try to find prompt in the runnable chain
                prompt = _find_prompt_in_runnable(runnable)
                if prompt:
                    return prompt
    except Exception:
        pass

    # Path 2: Try to get it from the agent's config
    try:
        config = getattr(agent, "config", {}) or {}
        if "prompt" in config:
            return config["prompt"]
    except Exception:
        pass

    # Path 3: Try direct attribute
    try:
        if hasattr(agent, "prompt"):
            p = agent.prompt
            if isinstance(p, str):
                return p
    except Exception:
        pass

    # Path 4: Invoke the prompt factory directly (agent-specific)
    return _invoke_prompt_factory(agent_name)


def _find_prompt_in_runnable(runnable, depth=0) -> str | None:
    """Recursively search a LangChain runnable chain for a system prompt string."""
    if depth > 10:
        return None

    # Check for direct prompt attribute
    if hasattr(runnable, "prompt") and isinstance(runnable.prompt, str):
        return runnable.prompt

    # Check bound kwargs (common in RunnableBinding)
    if hasattr(runnable, "kwargs"):
        kwargs = runnable.kwargs
        if isinstance(kwargs, dict) and "prompt" in kwargs:
            p = kwargs["prompt"]
            if isinstance(p, str):
                return p

    # Check first/middle/last in RunnableSequence
    for attr in ("first", "middle", "last", "bound"):
        child = getattr(runnable, attr, None)
        if child is not None:
            if isinstance(child, list):
                for c in child:
                    result = _find_prompt_in_runnable(c, depth + 1)
                    if result:
                        return result
            else:
                result = _find_prompt_in_runnable(child, depth + 1)
                if result:
                    return result

    # Check steps in RunnableParallel or similar
    if hasattr(runnable, "steps"):
        for step in (runnable.steps if isinstance(runnable.steps, list) else runnable.steps.values()):
            result = _find_prompt_in_runnable(step, depth + 1)
            if result:
                return result

    return None


def _invoke_prompt_factory(agent_name: str) -> str:
    """Fall back to directly calling the agent's prompt factory."""
    # Map known agents to their prompt factories
    prompt_factories = {
        "data_expert_agent": "lobster.agents.data_expert.prompts:create_data_expert_prompt",
        "research_agent": "lobster.agents.research.prompts:create_research_agent_prompt",
    }

    # Try known factory
    factory_path = prompt_factories.get(agent_name)
    if factory_path:
        module_path, func_name = factory_path.rsplit(":", 1)
        try:
            module = __import__(module_path, fromlist=[func_name])
            factory = getattr(module, func_name)
            return factory()
        except Exception:
            pass

    # Try to discover prompt module from agent package
    try:
        from lobster.core.component_registry import component_registry
        agent_config = component_registry.get_agent(agent_name)
        if agent_config:
            # e.g., "lobster.agents.data_expert.data_expert.data_expert"
            # → module "lobster.agents.data_expert.prompts"
            parts = agent_config.factory_function.rsplit(".", 2)
            if len(parts) >= 2:
                prompt_module_path = parts[0] + ".prompts"
                try:
                    prompt_module = __import__(prompt_module_path, fromlist=["create_" + agent_name.replace("_agent", "") + "_prompt"])
                    # Look for create_*_prompt functions
                    for attr_name in dir(prompt_module):
                        if attr_name.startswith("create_") and attr_name.endswith("_prompt"):
                            factory = getattr(prompt_module, attr_name)
                            if callable(factory):
                                sig = inspect.signature(factory)
                                if len(sig.parameters) == 0:
                                    return factory()
                except (ImportError, ModuleNotFoundError):
                    pass
    except Exception:
        pass

    return f"[Could not extract system prompt for '{agent_name}'. Manual inspection needed.]"


def _extract_tools(agent, agent_name: str) -> list:
    """Extract tools from a compiled ReAct agent (Pregel graph).

    LangGraph's create_react_agent wraps tools in a ToolNode inside a
    compiled StateGraph. We need to traverse the node structure to find them.
    """
    # Strategy 1: Navigate Pregel nodes → "tools" node → ToolNode → tools list
    try:
        for node_name, node in agent.nodes.items():
            # Skip non-tool nodes
            if node_name in ("__start__", "__end__"):
                continue

            # ToolNode stores tools directly
            node_obj = node
            # Unwrap ChannelWrite/ChannelRead wrappers
            while hasattr(node_obj, "bound"):
                node_obj = node_obj.bound
            while hasattr(node_obj, "runnable"):
                node_obj = node_obj.runnable

            # Check for tools attribute (ToolNode pattern)
            if hasattr(node_obj, "tools") and isinstance(node_obj.tools, (list, tuple)):
                tools = list(node_obj.tools)
                if tools and hasattr(tools[0], "name"):
                    return tools

            # Check tools_by_name (alternative ToolNode pattern)
            if hasattr(node_obj, "tools_by_name") and isinstance(node_obj.tools_by_name, dict):
                tools = list(node_obj.tools_by_name.values())
                if tools:
                    return tools
    except Exception:
        pass

    # Strategy 2: Look for tools in the "agent" node's bound model
    try:
        agent_node = agent.nodes.get("agent")
        if agent_node:
            node_obj = agent_node
            while hasattr(node_obj, "bound"):
                node_obj = node_obj.bound

            # ReAct agent may bind tools to the model
            if hasattr(node_obj, "kwargs") and "tools" in (node_obj.kwargs or {}):
                return list(node_obj.kwargs["tools"])
    except Exception:
        pass

    # Strategy 3: Walk all node attributes recursively looking for tool lists
    try:
        for node_name, node in agent.nodes.items():
            tools = _find_tools_recursive(node, depth=0)
            if tools:
                return tools
    except Exception:
        pass

    return []


def _find_tools_recursive(obj, depth=0, visited=None) -> list | None:
    """Recursively search an object tree for a list of LangChain tools."""
    if depth > 8:
        return None
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    # Check if this object has a tools attribute that looks like tool list
    for attr_name in ("tools", "tools_by_name"):
        val = getattr(obj, attr_name, None)
        if val is None:
            continue
        if isinstance(val, dict):
            tools = list(val.values())
        elif isinstance(val, (list, tuple)):
            tools = list(val)
        else:
            continue
        if tools and hasattr(tools[0], "name") and hasattr(tools[0], "description"):
            return tools

    # Recurse into common wrapper attributes
    for attr_name in ("bound", "runnable", "first", "last", "middle", "func"):
        child = getattr(obj, attr_name, None)
        if child is not None and child is not obj:
            if isinstance(child, (list, tuple)):
                for c in child:
                    result = _find_tools_recursive(c, depth + 1, visited)
                    if result:
                        return result
            else:
                result = _find_tools_recursive(child, depth + 1, visited)
                if result:
                    return result

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Show the full agent LLM context (system prompt + tool schemas)"
    )
    parser.add_argument(
        "agent_name",
        nargs="?",
        help="Agent name (e.g., data_expert_agent, transcriptomics_expert)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available agents"
    )
    parser.add_argument(
        "--prompt-only", action="store_true", help="Show system prompt only"
    )
    parser.add_argument(
        "--tools-only", action="store_true", help="Show tool schemas only"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON (for piping)"
    )
    args = parser.parse_args()

    if args.list:
        list_all_agents()
        return

    if not args.agent_name:
        parser.error("agent_name is required (use --list to see available agents)")

    # Setup
    workspace_path = Path("/tmp/lobster_agent_prompt_debug")
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Extract agent context
    system_prompt, tools, agent_config = extract_agent_prompt_and_tools(
        args.agent_name, workspace_path
    )

    # Build tool schemas
    tool_schemas = [_convert_tool_to_schema(t) for t in tools]
    tools_json = json.dumps(tool_schemas, indent=2)

    # --- JSON mode ---
    if args.json:
        output = {
            "agent_name": args.agent_name,
            "display_name": agent_config.display_name,
            "system_prompt": system_prompt,
            "tools": tool_schemas,
            "stats": {
                "system_prompt_chars": len(system_prompt),
                "system_prompt_tokens_est": estimate_tokens(system_prompt),
                "tools_chars": len(tools_json),
                "tools_tokens_est": estimate_tokens(tools_json),
                "total_tokens_est": estimate_tokens(system_prompt) + estimate_tokens(tools_json),
                "tool_count": len(tool_schemas),
            },
        }
        print(json.dumps(output, indent=2))
        return

    # --- Header ---
    print("=" * 80)
    print(f"AGENT: {agent_config.display_name} ({args.agent_name})")
    print("=" * 80)
    print(f"Description: {agent_config.description}")
    if agent_config.child_agents:
        print(f"Child agents: {', '.join(agent_config.child_agents)}")
    print(f"Factory: {agent_config.factory_function}")
    print(f"Tier: {agent_config.tier_requirement}")
    print()

    # --- System prompt ---
    if not args.tools_only:
        print("=" * 80)
        print("SECTION 1: SYSTEM PROMPT")
        print("=" * 80)
        print(system_prompt)
        print()

    # --- Tool schemas ---
    if not args.prompt_only:
        print("=" * 80)
        print("SECTION 2: TOOL DEFINITIONS")
        print("  (OpenAI function-calling format)")
        print("=" * 80)
        for i, schema in enumerate(tool_schemas, 1):
            fn = schema.get("function", schema)
            name = fn.get("name", "?")
            desc_preview = (fn.get("description") or "")[:120]
            params = fn.get("parameters", {})
            param_names = list(params.get("properties", {}).keys())

            # AQUADIF metadata
            aquadif = ""
            if i <= len(tools):
                tool_meta = getattr(tools[i - 1], "metadata", None)
                if tool_meta and isinstance(tool_meta, dict):
                    cats = tool_meta.get("categories", [])
                    aquadif = f"  [AQUADIF: {', '.join(cats)}]"

            print(f"\n--- Tool {i}/{len(tool_schemas)}: {name}{aquadif} ---")
            print(f"    Params: {', '.join(param_names) if param_names else '(none)'}")
            print(f"    Desc:   {desc_preview}{'...' if len(fn.get('description', '')) > 120 else ''}")
            print()
            print(json.dumps(schema, indent=2))

    # --- Summary ---
    prompt_tokens = estimate_tokens(system_prompt)
    tools_tokens = estimate_tokens(tools_json)
    total_tokens = prompt_tokens + tools_tokens

    print()
    print("=" * 80)
    print("CONTEXT BUDGET SUMMARY")
    print("=" * 80)
    print(f"System prompt:    {len(system_prompt):>8,} chars  ~{prompt_tokens:>6,} tokens")
    print(f"Tool definitions: {len(tools_json):>8,} chars  ~{tools_tokens:>6,} tokens")
    print(f"{'─' * 50}")
    print(f"Total baseline:   {len(system_prompt) + len(tools_json):>8,} chars  ~{total_tokens:>6,} tokens")
    print()
    print(f"Tools: {len(tool_schemas)} total")

    # Per-tool breakdown
    print()
    print("Per-tool token cost:")
    tool_costs = []
    for i, schema in enumerate(tool_schemas):
        fn = schema.get("function", schema)
        name = fn.get("name", "?")
        desc_chars = len(fn.get("description", ""))
        schema_chars = len(json.dumps(schema))
        tokens = schema_chars // 4
        tool_costs.append((name, desc_chars, schema_chars, tokens))

    # Sort by tokens descending
    tool_costs.sort(key=lambda x: x[3], reverse=True)
    for name, desc_chars, schema_chars, tokens in tool_costs:
        pct = tokens / tools_tokens * 100 if tools_tokens > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:42s}  ~{tokens:>5} tok ({pct:4.1f}%)  {bar}")

    # Context window comparison
    print()
    print("Context window utilization (baseline only):")
    windows = {
        "Claude Sonnet (200K)": 200_000,
        "Qwen3 14B (40K)": 40_000,
        "Qwen3 8B (32K)": 32_768,
    }
    for model_name, window in windows.items():
        pct = (total_tokens / window) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {model_name:.<30} {bar} {pct:5.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
