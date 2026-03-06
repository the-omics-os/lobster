#!/usr/bin/env python3
"""
Render the FULL supervisor LLM context: system prompt + tool schemas.

This shows EXACTLY what the supervisor model sees on every turn —
the system prompt from create_supervisor_prompt() AND the OpenAI-format
tool definitions that LangGraph serializes from all bound tools.

Usage:
    python scripts/supervisor_prompt_concat.py               # full output
    python scripts/supervisor_prompt_concat.py --prompt-only  # system prompt only
    python scripts/supervisor_prompt_concat.py --tools-only   # tool schemas only
    python scripts/supervisor_prompt_concat.py --json         # raw JSON (pipeable)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root so `lobster` package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Tool schema extraction
# ---------------------------------------------------------------------------


def _convert_tool_to_schema(t) -> dict:
    """Convert a LangChain @tool to its OpenAI function-calling schema.

    This is the exact format LangGraph sends to the LLM provider.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return convert_to_openai_tool(t)


def _create_mock_handoff_tool(tool_name: str, description: str):
    """Create a mock handoff tool with the same schema as real delegation tools.

    Real handoff tools wrap agent.invoke() but their SCHEMA is always:
        handoff_to_<name>(task_description: str) -> str

    We mock them to avoid needing LLM API keys / agent creation.
    """
    from langchain_core.tools import tool

    @tool(tool_name, description=description)
    def mock_handoff(task_description: str) -> str:
        """Invoke a sub-agent with a task description.

        Args:
            task_description: Detailed description of what the agent should do,
                including all relevant context. Should be in task format starting
                with 'Your task is to ...'
        """
        return ""

    mock_handoff.metadata = {"categories": ["DELEGATE"], "provenance": False}
    mock_handoff.tags = ["DELEGATE"]
    return mock_handoff


def collect_supervisor_tools(data_manager, active_agents: list, agent_configs: dict):
    """Collect all tools the supervisor would have, without creating real agents.

    Returns the tools in the same order as _build_supervisor_tools() in graph.py:
    1. Handoff tools (one per supervisor-accessible agent)
    2. Workspace tools (list_modalities, get_content, delete)
    3. Todo tools (write_todos, read_todos)
    4. execute_custom_code
    5. retrieve_agent_result (always included — store is default-on)
    """
    tools = []

    # 1. Handoff tools — use agent's own handoff_tool_description for routing signal.
    #    Small models rely on tool descriptions for selection.
    for agent_name in active_agents:
        cfg = agent_configs.get(agent_name)
        if cfg and cfg.handoff_tool_name:
            desc = cfg.handoff_tool_description or (
                f"Delegate task to {cfg.display_name}."
            )
            tools.append(
                _create_mock_handoff_tool(cfg.handoff_tool_name, desc)
            )

    # 2. Workspace tools
    from lobster.tools.workspace_tool import (
        create_delete_from_workspace_tool,
        create_get_content_from_workspace_tool,
        create_list_modalities_tool,
    )

    tools.append(create_list_modalities_tool(data_manager))
    tools.append(create_get_content_from_workspace_tool(data_manager))
    tools.append(create_delete_from_workspace_tool(data_manager))

    # 3. Todo tools
    from lobster.tools.todo_tools import create_todo_tools

    write_todos, read_todos = create_todo_tools()
    tools.append(write_todos)
    tools.append(read_todos)

    # 4. Custom code execution
    from lobster.services.execution.custom_code_execution_service import (
        CustomCodeExecutionService,
    )
    from lobster.tools.custom_code_tool import create_execute_custom_code_tool

    code_service = CustomCodeExecutionService(data_manager)
    tools.append(
        create_execute_custom_code_tool(
            data_manager=data_manager,
            custom_code_service=code_service,
            agent_name="supervisor",
        )
    )

    # 5. Store retrieval tool (store is always created in production)
    from langgraph.store.memory import InMemoryStore

    from lobster.tools.store_tools import create_retrieve_agent_result_tool

    store = InMemoryStore()
    tools.append(create_retrieve_agent_result_tool(store))

    return tools


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4). Good enough for context budget sense."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Show the full supervisor LLM context"
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

    # --- Setup (minimal — no LLM keys needed) ---
    workspace_path = Path("/tmp/lobster_supervisor_prompt_debug")
    workspace_path.mkdir(parents=True, exist_ok=True)

    from lobster.config.supervisor_config import SupervisorConfig
    from lobster.core.data_manager_v2 import DataManagerV2

    data_manager = DataManagerV2(workspace_path=workspace_path)
    config = SupervisorConfig.from_env()

    # --- Discover agents (same path as graph.py) ---
    from lobster.core.component_registry import component_registry

    all_agents = component_registry.list_agents()

    # Determine supervisor-accessible agents (exclude children)
    child_agent_names = set()
    for agent_config in all_agents.values():
        if agent_config.child_agents:
            child_agent_names.update(agent_config.child_agents)

    supervisor_accessible = []
    for name, cfg in all_agents.items():
        if cfg.supervisor_accessible is None:
            is_accessible = name not in child_agent_names
        else:
            is_accessible = cfg.supervisor_accessible
        if is_accessible:
            supervisor_accessible.append(name)

    # --- Build system prompt ---
    from lobster.agents.supervisor import create_supervisor_prompt

    system_prompt = create_supervisor_prompt(
        data_manager=data_manager,
        config=config,
        active_agents=supervisor_accessible,
    )

    # --- Build tool schemas ---
    tools = collect_supervisor_tools(data_manager, supervisor_accessible, all_agents)
    tool_schemas = [_convert_tool_to_schema(t) for t in tools]
    tools_json = json.dumps(tool_schemas, indent=2)

    # --- JSON mode: output structured data and exit ---
    if args.json:
        output = {
            "system_prompt": system_prompt,
            "tools": tool_schemas,
            "stats": {
                "system_prompt_chars": len(system_prompt),
                "system_prompt_tokens_est": estimate_tokens(system_prompt),
                "tools_chars": len(tools_json),
                "tools_tokens_est": estimate_tokens(tools_json),
                "total_tokens_est": estimate_tokens(system_prompt)
                + estimate_tokens(tools_json),
                "tool_count": len(tool_schemas),
                "agent_count": len(supervisor_accessible),
            },
        }
        print(json.dumps(output, indent=2))
        return

    # --- Config summary ---
    if not args.tools_only:
        print("=" * 80)
        print("SUPERVISOR CONFIG")
        print("=" * 80)
        print(f"Mode:                        {config.get_prompt_mode()}")
        print(f"Workflow guidance level:      {config.workflow_guidance_level}")
        print(
            f"Ask clarification questions:  {config.ask_clarification_questions} (max {config.max_clarification_questions})"
        )
        print(
            f"Require download confirm:     {config.require_download_confirmation}"
        )
        print(f"Require metadata preview:     {config.require_metadata_preview}")
        print(f"Summarize expert output:      {config.summarize_expert_output}")
        print(f"Auto suggest next steps:      {config.auto_suggest_next_steps}")
        print(f"Verbose delegation:           {config.verbose_delegation}")
        print(f"Include data context:         {config.include_data_context}")
        print(f"Include system info:          {config.include_system_info}")
        print(f"Include memory stats:         {config.include_memory_stats}")
        print(f"Enable todo planning:         {config.enable_todo_planning}")
        print(f"Auto discover agents:         {config.auto_discover_agents}")
        print(f"Delegation strategy:          {config.delegation_strategy}")
        print(f"Error handling:               {config.error_handling}")
        print()
        print(f"Supervisor-accessible agents ({len(supervisor_accessible)}):")
        for name in supervisor_accessible:
            cfg = all_agents[name]
            children = (
                f" -> [{', '.join(cfg.child_agents)}]" if cfg.child_agents else ""
            )
            print(f"  - {cfg.display_name} ({name}){children}")
        print()
        print(
            f"Child-only agents ({len(child_agent_names)}): {', '.join(sorted(child_agent_names))}"
        )
        print("=" * 80)
        print()

    # --- System prompt ---
    if not args.tools_only:
        print("=" * 80)
        print("SECTION 1: SYSTEM PROMPT")
        print("  (what LangGraph passes as the system message)")
        print("=" * 80)
        print(system_prompt)
        print()

    # --- Tool schemas ---
    if not args.prompt_only:
        print("=" * 80)
        print("SECTION 2: TOOL DEFINITIONS")
        print(
            "  (OpenAI function-calling format — serialized into every API request)"
        )
        print("=" * 80)
        for i, schema in enumerate(tool_schemas, 1):
            fn = schema.get("function", schema)
            name = fn.get("name", "?")
            desc_preview = (fn.get("description") or "")[:120]
            params = fn.get("parameters", {})
            param_names = list(params.get("properties", {}).keys())

            # AQUADIF metadata from the original tool
            aquadif = ""
            tool_meta = getattr(tools[i - 1], "metadata", None)
            if tool_meta and isinstance(tool_meta, dict):
                cats = tool_meta.get("categories", [])
                aquadif = f"  [AQUADIF: {', '.join(cats)}]"

            print(f"\n--- Tool {i}/{len(tool_schemas)}: {name}{aquadif} ---")
            print(
                f"    Params: {', '.join(param_names) if param_names else '(none)'}"
            )
            print(
                f"    Desc:   {desc_preview}{'...' if len(fn.get('description', '')) > 120 else ''}"
            )
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
    print(
        f"System prompt:    {len(system_prompt):>8,} chars  ~{prompt_tokens:>6,} tokens"
    )
    print(
        f"Tool definitions: {len(tools_json):>8,} chars  ~{tools_tokens:>6,} tokens"
    )
    print(f"{'─' * 50}")
    print(
        f"Total baseline:   {len(system_prompt) + len(tools_json):>8,} chars  ~{total_tokens:>6,} tokens"
    )
    print()

    n_handoff = sum(
        1
        for t in tools
        if "DELEGATE" in (getattr(t, "tags", None) or [])
    )
    n_utility = len(tool_schemas) - n_handoff
    print(f"Tools: {len(tool_schemas)} total ({n_handoff} handoff + {n_utility} utility)")
    print(
        f"Agents: {len(supervisor_accessible)} supervisor-accessible, "
        f"{len(child_agent_names)} child-only, {len(all_agents)} total"
    )
    print()

    # Context window comparison
    windows = {
        "Claude Sonnet (200K)": 200_000,
        "Claude Haiku (200K)": 200_000,
        "GPT-4o (128K)": 128_000,
        "Llama 3 8B (8K)": 8_192,
    }
    print("Context window utilization (baseline only, before any messages):")
    for model_name, window in windows.items():
        pct = (total_tokens / window) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {model_name:.<30} {bar} {pct:5.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
