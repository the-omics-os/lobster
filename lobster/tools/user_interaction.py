"""HITL user interaction tool for the supervisor agent.

Provides the `ask_user` tool that pauses graph execution via LangGraph's
``interrupt()`` primitive and requests structured input from the user.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_core.tools import tool


@tool
def ask_user(question: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Ask the user a clarification question.

    Pauses the current workflow and renders an interactive component
    (confirmation, selection, text input, etc.) based on the question
    and context provided.

    Args:
        question: Natural language question for the user.
        context: Relevant data — option lists, cluster info, threshold
                 ranges, etc. Determines which UI component is rendered.

    Returns:
        JSON string with the user's structured response.
    """
    from langgraph.types import interrupt

    from lobster.services.interaction.component_mapper import map_question

    selection = map_question(question, context)
    response = interrupt(selection.model_dump())
    return json.dumps(response)


# AQUADIF metadata
ask_user.metadata = {"categories": ["UTILITY"], "provenance": False}
ask_user.tags = ["UTILITY"]
