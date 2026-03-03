"""
Context management for Lobster AI's LangGraph multi-agent system.

Provides pre_model_hook factories and context budget resolution to prevent
context overflow and reduce token waste across all LLM providers.

Architecture:
    - resolve_context_budget() calculates usable token budget
    - create_supervisor_pre_model_hook() returns a hook for create_react_agent
    - The hook uses trim_messages + llm_input_messages bypass (non-destructive)

See: kevin_notes/CONTEXT_ENGINEERING_REPORT.md
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.utils import trim_messages

logger = logging.getLogger(__name__)

# Default context budget if provider info unavailable
DEFAULT_CONTEXT_WINDOW = 128_000

# Reserve this fraction for model output generation
OUTPUT_RESERVE_FRACTION = 0.15

# Absolute minimum budget (prevents degenerate trimming)
MINIMUM_BUDGET = 4096


def approximate_token_count(text: str) -> int:
    """Approximate token count using chars/4 heuristic.

    Slightly overestimates (safe direction for budget math).
    Zero dependencies, works for all providers.
    """
    if not text:
        return 0
    return int(len(text) / 4.0 + 3.0)


def _message_list_token_counter(messages: list[BaseMessage]) -> int:
    """Token counter compatible with trim_messages (receives list of messages).

    trim_messages calls token_counter([msg1, msg2, ...]) and expects a single
    int back representing total tokens for the batch.
    """
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += approximate_token_count(content)
    return total


def measure_tool_schema_tokens(tools: Sequence[Any]) -> int:
    """Measure approximate token cost of tool schemas.

    Tool schemas are sent with every LLM call but NOT counted by
    trim_messages. We subtract them from the context budget to
    prevent the 85%-guess problem identified in the research report.
    """
    total = 0
    for t in tools:
        try:
            schema_str = ""
            if hasattr(t, "name"):
                schema_str += t.name + " "
            if hasattr(t, "description"):
                schema_str += (t.description or "") + " "
            if hasattr(t, "get_input_schema"):
                schema_str += json.dumps(
                    t.get_input_schema().model_json_schema(), separators=(",", ":")
                )
            elif hasattr(t, "args_schema") and t.args_schema:
                schema_str += json.dumps(
                    t.args_schema.model_json_schema(), separators=(",", ":")
                )
            total += approximate_token_count(schema_str)
        except Exception:
            total += 200  # Conservative fallback per tool
    return total


def resolve_context_budget(
    context_window: Optional[int] = None,
    tools: Optional[Sequence[Any]] = None,
) -> int:
    """Resolve the usable context budget for messages. Creates the max_tokens var

    budget = context_window - output_reserve - tool_schema_tokens

    Args:
        context_window: Max tokens for the model. None -> DEFAULT_CONTEXT_WINDOW.
        tools: Tool objects to measure schema overhead.

    Returns:
        Usable token budget for messages (minimum MINIMUM_BUDGET).
    """
    if context_window is None:
        context_window = DEFAULT_CONTEXT_WINDOW

    output_reserve = int(context_window * OUTPUT_RESERVE_FRACTION)
    tool_tokens = measure_tool_schema_tokens(tools or [])
    budget = context_window - output_reserve - tool_tokens

    logger.debug(
        f"Context budget: {budget} tokens "
        f"(window={context_window}, reserve={output_reserve}, schemas={tool_tokens})"
    )

    return max(budget, MINIMUM_BUDGET)


def resolve_context_window(
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    workspace_path=None,
) -> Optional[int]:
    """Best-effort resolution of context window from provider config.

    Uses the same ConfigResolver + ProviderRegistry path as create_llm.
    Returns None if resolution fails (caller should use DEFAULT_CONTEXT_WINDOW).
    """
    try:
        from lobster.core.config_resolver import ConfigResolver

        resolver = ConfigResolver.get_instance(workspace_path)
        provider_name, model_id = resolver.resolve_provider()

        # CLI overrides take precedence
        if provider_override:
            provider_name = provider_override
        if model_override:
            model_id = model_override

        from lobster.config.providers import get_provider

        provider = get_provider(provider_name)
        if provider:
            model_info = provider.get_model_info(model_id)
            if model_info and model_info.context_window:
                return model_info.context_window
    except Exception as e:
        logger.debug(f"Could not resolve context window: {e}")
    return None


def _fix_orphaned_tool_messages(messages: list) -> list:
    """Remove orphaned ToolMessages from the start of a trimmed message list.

    When trim_messages drops an AIMessage(tool_calls) but keeps the
    subsequent ToolMessage, the result is an invalid chat history that
    causes LLM API rejections. This function strips leading ToolMessages
    that have no corresponding AIMessage with matching tool_call_id.

    Only processes the front of the list (after system messages) since
    strategy="last" trims from the front.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    if not messages:
        return messages

    # Collect tool_call_ids present in AIMessages
    ai_tool_call_ids = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls or []:
                if isinstance(tc, dict) and "id" in tc:
                    ai_tool_call_ids.add(tc["id"])

    # Strip orphaned ToolMessages from the front (after system messages)
    result = []
    past_system = False
    stripping = True
    for msg in messages:
        if not past_system and isinstance(msg, SystemMessage):
            result.append(msg)
            continue
        past_system = True

        if stripping and isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id not in ai_tool_call_ids:
                logger.debug(
                    f"Stripped orphaned ToolMessage (tool_call_id={tool_call_id})"
                )
                continue
        stripping = False
        result.append(msg)

    return result


def _build_store_key_index(store) -> dict[str, str]:
    """Read all stored agent results and build a key→agent_name index.

    Returns empty dict if store is None or search fails (fail-open).
    """
    if store is None:
        return {}
    try:
        from lobster.tools.store_tools import AGENT_RESULTS_NAMESPACE

        items = store.search(AGENT_RESULTS_NAMESPACE)
        return {item.key: item.value.get("agent", "unknown") for item in items}
    except Exception:
        return {}


def _build_key_index_text(store_keys: dict[str, str]) -> str:
    """Build compact text listing available store keys.

    Returns empty string if no keys exist.
    Designed to be appended to the existing SystemMessage content
    (avoids multi-SystemMessage compatibility issues across providers).
    """
    if not store_keys:
        return ""

    lines = [f"- {key} ({agent})" for key, agent in store_keys.items()]
    return (
        "\n\n<Available Stored Results>\n"
        "Use retrieve_agent_result(store_key) to access full data.\n"
        + "\n".join(lines)
        + "\n</Available Stored Results>"
    )


def create_supervisor_pre_model_hook(max_tokens: int) -> callable:
    """Create a pre_model_hook for the supervisor agent.

    On each ReAct iteration:
    1. Reads InMemoryStore to build a store_keys index (survives trimming)
    2. Trims messages with strategy="last", include_system=True, end_on="ai"
    3. Injects a compact key index message so the LLM always knows which
       store keys are available for retrieve_agent_result
    4. Returns llm_input_messages (non-destructive) + store_keys state update

    The store parameter is auto-injected by LangGraph's Runtime when the
    supervisor is compiled with store=store.

    Args:
        max_tokens: Maximum token budget for messages.

    Returns:
        Callable suitable for create_react_agent(pre_model_hook=...).
    """
    _max_tokens = max_tokens

    def pre_model_hook(state: dict, *, store=None) -> dict:
        messages = state.get("messages", [])

        # Build store key index from InMemoryStore
        store_keys = _build_store_key_index(store)

        if not messages:
            return {"llm_input_messages": messages, "store_keys": store_keys}

        trimmed = trim_messages(
            messages,
            max_tokens=_max_tokens,
            token_counter=_message_list_token_counter,
            strategy="last",
            include_system=True,
            allow_partial=False,
        )

        # Fix orphaned ToolMessages that lost their AIMessage partner
        trimmed = _fix_orphaned_tool_messages(trimmed)

        if len(trimmed) < len(messages):
            logger.debug(
                f"pre_model_hook trimmed {len(messages)} -> {len(trimmed)} messages "
                f"(budget: {_max_tokens} tokens)"
            )

        # Append key index to existing SystemMessage (single system message
        # for maximum provider compatibility — Anthropic, OpenAI, Ollama, Gemini)
        key_index_text = _build_key_index_text(store_keys)
        if key_index_text:
            trimmed = list(trimmed)
            for i, msg in enumerate(trimmed):
                if isinstance(msg, SystemMessage):
                    trimmed[i] = SystemMessage(content=msg.content + key_index_text)
                    break

        return {"llm_input_messages": trimmed, "store_keys": store_keys}

    return pre_model_hook
