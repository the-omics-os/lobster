"""
Store-backed tools for Lobster AI context management.

Provides:
- retrieve_agent_result: supervisor tool for on-demand detail retrieval
- store_delegation_result: helper for dual-write in delegation tools

Uses InMemoryStore with ("agent_results",) namespace.
Per-session isolation is implicit (InMemoryStore created per AgentClient).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Maximum characters returned from a single retrieval
MAX_RETRIEVAL_CHARS = 20_000

# Namespace for agent results in InMemoryStore
AGENT_RESULTS_NAMESPACE = ("agent_results",)


def store_delegation_result(
    store: Optional[Any],
    agent_name: str,
    content: str,
) -> Optional[str]:
    """Store a delegation result in InMemoryStore (dual-write helper).

    Called from delegation tool wrappers after agent.invoke() completes.
    Returns the store_key for retrieval, or None if store is unavailable.

    Args:
        store: InMemoryStore instance (or None for graceful degradation).
        agent_name: Name of the agent that produced the result.
        content: Full result content string.

    Returns:
        Unique store_key, or None if store is None.
    """
    if store is None:
        return None

    store_key = f"{agent_name}_{uuid.uuid4().hex[:8]}"
    try:
        store.put(
            AGENT_RESULTS_NAMESPACE,
            store_key,
            {"content": content, "agent": agent_name},
        )
        return store_key
    except Exception as e:
        logger.warning(f"Failed to store result for {agent_name}: {e}")
        return None


def create_retrieve_agent_result_tool(store: Any):
    """Create a tool for the supervisor to retrieve stored agent results.

    The tool captures the store instance via closure (consistent with how
    all Lobster tools capture data_manager).

    Args:
        store: InMemoryStore instance.

    Returns:
        A @tool function with AQUADIF UTILITY metadata.
    """
    _store = store

    @tool
    def retrieve_agent_result(store_key: str) -> str:
        """Retrieve detailed results from a previous sub-agent analysis.

        Use when you need specific data points (exact p-values, gene lists,
        full tables) beyond the summary in the delegation response.
        The store_key is in [store_key=...] at the end of each delegation result.

        Args:
            store_key: The unique key from the sub-agent's response.
        """
        if _store is None:
            return "Store not available. Cannot retrieve detailed results."

        try:
            item = _store.get(AGENT_RESULTS_NAMESPACE, store_key)
            if item and item.value:
                content = item.value.get("content", "")
                agent = item.value.get("agent", "unknown")

                if len(content) > MAX_RETRIEVAL_CHARS:
                    return (
                        f"[Retrieved from {agent}, truncated to {MAX_RETRIEVAL_CHARS:,} chars]\n\n"
                        f"{content[:MAX_RETRIEVAL_CHARS]}\n\n"
                        f"[...truncated, full result is {len(content):,} chars]"
                    )
                return f"[Retrieved from {agent}]\n\n{content}"
            return f"No results found for store_key='{store_key}'."
        except Exception as e:
            logger.warning(f"Failed to retrieve store_key='{store_key}': {e}")
            return f"Error retrieving results: {e}"

    retrieve_agent_result.metadata = {"categories": ["UTILITY"], "provenance": False}
    retrieve_agent_result.tags = ["UTILITY"]

    return retrieve_agent_result
