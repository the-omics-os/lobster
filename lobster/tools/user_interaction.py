"""HITL user interaction tool for the supervisor agent.

Provides the ``create_ask_user_tool`` factory that creates an ``ask_user`` tool
with optional LLM closure for hybrid component selection.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Optional

from langchain_core.tools import tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def create_ask_user_tool(llm: Optional["BaseChatModel"] = None):
    """Create an ask_user tool with optional LLM for component selection.

    Args:
        llm: Optional LLM used by the component mapper for ambiguous
             questions. When None, falls back to rule-based selection.

    Returns:
        A LangChain tool with AQUADIF UTILITY metadata.
    """

    @tool
    def ask_user(question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Ask the user a clarification question. Renders an interactive component."""
        from langgraph.types import interrupt

        from lobster.services.interaction.component_mapper import map_question

        selection = map_question(question, context, llm=llm)
        response = interrupt(selection.model_dump())
        return json.dumps(response)

    ask_user.metadata = {"categories": ["UTILITY"], "provenance": False}
    ask_user.tags = ["UTILITY"]
    return ask_user


# Backward-compatible module-level instance (no LLM).
ask_user = create_ask_user_tool()
