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

# Reserve this fraction for model output + safety margin against counting drift
OUTPUT_RESERVE_FRACTION = 0.20

# Absolute minimum budget (prevents degenerate trimming)
# Must be large enough for system prompt + a few conversation turns
MINIMUM_BUDGET = 4096

# Warn when tool schemas + system prompt exceed this fraction of context window
BASELINE_WARNING_THRESHOLD = 0.50


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
    system_prompt: Optional[str] = None,
) -> int:
    """Resolve the usable context budget for messages.

    budget = context_window * 0.80 - tool_schema_tokens - system_prompt_tokens

    Args:
        context_window: Max tokens for the model. None -> DEFAULT_CONTEXT_WINDOW.
        tools: Tool objects to measure schema overhead.
        system_prompt: System prompt text to subtract from budget.

    Returns:
        Usable token budget for messages (minimum MINIMUM_BUDGET).
    """
    if context_window is None:
        context_window = DEFAULT_CONTEXT_WINDOW

    input_ceiling = int(context_window * (1.0 - OUTPUT_RESERVE_FRACTION))
    tool_tokens = measure_tool_schema_tokens(tools or [])
    prompt_tokens = approximate_token_count(system_prompt) if system_prompt else 0
    fixed_overhead = tool_tokens + prompt_tokens
    raw_budget = input_ceiling - fixed_overhead

    # Warn when baseline overhead consumes too much of the window.
    baseline_fraction = fixed_overhead / context_window if context_window > 0 else 0
    if baseline_fraction > BASELINE_WARNING_THRESHOLD:
        logger.warning(
            f"Fixed overhead consumes {baseline_fraction:.0%} of context window "
            f"(tools={tool_tokens}, prompt={prompt_tokens}, total={fixed_overhead}/{context_window}). "
            f"Model may struggle with complex conversations. "
            f"Consider reducing tool description sizes or using a larger model."
        )

    if raw_budget < MINIMUM_BUDGET:
        logger.warning(
            f"Context budget {raw_budget} < minimum {MINIMUM_BUDGET}. "
            f"Clamping to {MINIMUM_BUDGET} but context overflow is likely "
            f"(window={context_window}, overhead={fixed_overhead}, ceiling={input_ceiling})."
        )

    budget = max(raw_budget, MINIMUM_BUDGET)

    logger.debug(
        f"Context budget: {budget} tokens "
        f"(window={context_window}, ceiling={input_ceiling}, tools={tool_tokens}, "
        f"prompt={prompt_tokens})"
    )

    return budget


def resolve_context_window(
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    workspace_path=None,
) -> Optional[int]:
    """Best-effort resolution of context window from provider config.

    Priority:
    1. Workspace model_context_windows override
    2. Global model_context_windows override
    3. Provider catalog (get_model_info — never returns None)

    Returns None only if resolution completely fails (caller should use
    DEFAULT_CONTEXT_WINDOW).
    """
    from lobster.config.providers import get_provider

    provider_name: Optional[str] = provider_override
    model_id: Optional[str] = model_override

    # When overrides are incomplete, resolve missing values from config.
    # resolve_provider() returns (provider_name, decision_source) — the second
    # element is NOT a model ID, so we only use it for the provider name.
    if not provider_name or not model_id:
        try:
            from lobster.core.config_resolver import ConfigResolver

            resolver = ConfigResolver.get_instance(workspace_path)
            resolved_provider, _source = resolver.resolve_provider()

            if not provider_name:
                provider_name = resolved_provider
            if not model_id:
                # ConfigResolver doesn't resolve model — provider default is used
                model_id = None
        except Exception as e:
            logger.warning(f"Could not resolve provider from config: {e}")

    # Check config-level context window overrides (highest priority)
    if model_id:
        try:
            from lobster.config.global_config import GlobalProviderConfig
            from lobster.config.workspace_config import WorkspaceProviderConfig

            # Workspace override (highest priority)
            if workspace_path:
                ws_config = WorkspaceProviderConfig.load(workspace_path)
                if model_id in ws_config.model_context_windows:
                    return ws_config.model_context_windows[model_id]

            # Global override
            global_config = GlobalProviderConfig.load()
            if model_id in global_config.model_context_windows:
                return global_config.model_context_windows[model_id]
        except Exception as e:
            logger.debug(f"Could not check config context window overrides: {e}")

    if not provider_name:
        logger.warning("No provider available for context window resolution")
        return None

    try:
        provider = get_provider(provider_name)
        if provider:
            model_info = provider.get_model_info(model_id)
            if model_info and model_info.context_window:
                return model_info.context_window
    except Exception as e:
        logger.warning(
            f"Could not resolve context window for {provider_name}/{model_id}: {e}"
        )
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


def _wrap_tool_results(messages: list) -> list:
    """Wrap ToolMessage content in <tool_data> markers (spotlighting defense).

    Ensures the LLM treats tool output as DATA, not instructions.
    Any directive-like text embedded in tool results (e.g. from GEO metadata,
    PubMed abstracts, or custom code output) stays inert.

    Handles string content and multi-part list content. Skips messages
    already wrapped or with empty/None content.
    """
    from langchain_core.messages import ToolMessage

    if not messages:
        return messages

    result = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            result.append(msg)
            continue

        content = msg.content

        # Skip None/empty content
        if not content:
            result.append(msg)
            continue

        if isinstance(content, str):
            # Skip if already wrapped
            if content.lstrip().startswith("<tool_data>"):
                result.append(msg)
                continue
            wrapped_content = f"<tool_data>{content}</tool_data>"
        elif isinstance(content, list):
            # Multi-part content: wrap each text part
            wrapped_content = []
            for part in content:
                if isinstance(part, str):
                    if part.lstrip().startswith("<tool_data>"):
                        wrapped_content.append(part)
                    else:
                        wrapped_content.append(f"<tool_data>{part}</tool_data>")
                elif isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if text.lstrip().startswith("<tool_data>"):
                        wrapped_content.append(part)
                    else:
                        wrapped_content.append(
                            {**part, "text": f"<tool_data>{text}</tool_data>"}
                        )
                else:
                    # Non-text parts (images, etc.) pass through unchanged
                    wrapped_content.append(part)
        else:
            result.append(msg)
            continue

        # Create new ToolMessage preserving tool_call_id and name
        wrapped_msg = ToolMessage(
            content=wrapped_content,
            tool_call_id=msg.tool_call_id,
            name=getattr(msg, "name", None),
        )
        result.append(wrapped_msg)

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

        # Wrap tool results BEFORE trimming so wrapper overhead is counted
        wrapped = _wrap_tool_results(messages)

        # Compute post-trim injection overhead and subtract from budget
        key_index_text = _build_key_index_text(store_keys)
        key_index_overhead = approximate_token_count(key_index_text) if key_index_text else 0
        effective_budget = max(_max_tokens - key_index_overhead, 0)

        trimmed = trim_messages(
            wrapped,
            max_tokens=effective_budget,
            token_counter=_message_list_token_counter,
            strategy="last",
            include_system=True,
            allow_partial=False,
        )

        # Fix orphaned ToolMessages that lost their AIMessage partner
        trimmed = _fix_orphaned_tool_messages(trimmed)

        compaction_metadata: dict[str, int] | None = None
        if len(trimmed) < len(wrapped):
            compaction_metadata = {
                "before_count": len(messages),
                "after_count": len(trimmed),
                "budget_tokens": _max_tokens,
            }
            logger.debug(
                f"pre_model_hook trimmed {len(messages)} -> {len(trimmed)} messages "
                f"(budget: {_max_tokens}, effective: {effective_budget})"
            )

        # Append key index to existing SystemMessage (single system message
        # for maximum provider compatibility — Anthropic, OpenAI, Ollama, Gemini)
        if key_index_text:
            trimmed = list(trimmed)
            for i, msg in enumerate(trimmed):
                if isinstance(msg, SystemMessage):
                    trimmed[i] = SystemMessage(content=msg.content + key_index_text)
                    break

        result = {"llm_input_messages": trimmed, "store_keys": store_keys}
        if compaction_metadata is not None:
            result["context_compaction"] = compaction_metadata
        return result

    return pre_model_hook
