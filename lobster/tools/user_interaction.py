"""HITL user interaction tool for the supervisor agent.

Provides the ``create_ask_user_tool`` factory that creates an ``ask_user`` tool
with optional LLM closure for hybrid component selection.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Annotated, Any, Dict, Optional

from langchain_core.tools import InjectedToolCallId, tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# Resume-value key set by the run API (``lobster.core.runs.DECISION_ENVELOPE``).
_ENVELOPE = "__lobster_decision__"


def _decision_ir(question: str, decision: Dict[str, Any]):
    """AQUADIF step for one answered decision: a comment-only notebook cell."""
    from lobster.core.provenance.analysis_ir import AnalysisStep

    params = {
        "question": question,
        "answer": decision.get("answer"),
        "comment": decision.get("comment"),
        "source": decision.get("source"),
        "decision_id": decision.get("decision_id"),
        "request_version": decision.get("request_version"),
    }
    # ``repr`` keeps the scientist's text verbatim on one comment line.
    header = "# Decision recorded during the analysis (source: {{ source | repr }}"
    lines = []
    channel = decision.get("channel")
    if channel:
        # ``channel`` is ``mcp:<host name>``; the host's claim is all there is.
        params["channel"] = channel
        header += ", as reported by the MCP host {{ channel.split(':', 1)[-1] | repr }}"
        lines = ["# Lobster cannot verify who answered through an MCP host."]
    template = "\n".join(
        [
            header + ")",
            *lines,
            "# Question: {{ question | repr }}",
            "# Answer: {{ answer | repr }}",
            "# Comment: {{ comment | repr }}",
        ]
    )
    return AnalysisStep(
        operation="lobster.decision",
        tool_name="ask_user",
        description="Decision recorded during the analysis",
        library="lobster",
        code_template=template,
        imports=[],
        parameters=params,
        parameter_schema={},
        replayable=False,
    )


def create_ask_user_tool(
    llm: Optional["BaseChatModel"] = None, data_manager: Any = None
):
    """Create an ask_user tool with optional LLM for component selection.

    Args:
        llm: Optional LLM used by the component mapper for ambiguous
             questions. When None, falls back to rule-based selection.
        data_manager: Records each answered decision in the provenance
             (question, answer, comment verbatim, source as told, and the channel when
             the answer came through one).

    Returns:
        A LangChain tool with AQUADIF UTILITY metadata.
    """

    @tool
    async def ask_user(
        question: str,
        context: Optional[Dict[str, Any]] = None,
        tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None,
    ) -> str:
        """Pause execution and ask the user a question. Use ONLY when user input
        is genuinely required (ambiguous intent, approval gate, or domain choice).
        The question might come from a sub-agent that needs user input and asks you to provide this to them

        Pass structured context for automatic component selection:
          - {"options": ["A", "B", "C"]}        → dropdown select
          - {"clusters": [{id, size, markers}]}  → cell-type annotation panel
          - {"min": 0, "max": 1, "threshold": 0.05} → threshold slider
          - (no context or confirm phrasing)     → yes/no or free-text

        Args:
            question: Clear, specific question for the user.
            context: Structured hints for component selection (see above).

        Returns:
            JSON string with the user's response."""
        from langgraph.types import interrupt

        from lobster.services.interaction.component_mapper import map_question

        selection = await asyncio.to_thread(map_question, question, context, llm=llm)
        # ``ask_id`` lets the run API keep one decision id when this question
        # surfaces again; it is stripped before anything is rendered.
        response = interrupt({**selection.model_dump(), "ask_id": tool_call_id})
        if not (isinstance(response, dict) and _ENVELOPE in response):
            return json.dumps(response)  # bare answer from an older caller
        decision = response[_ENVELOPE]
        if data_manager is not None:
            data_manager.log_tool_usage(
                tool_name="ask_user",
                parameters={"question": question},
                description="Decision recorded during the analysis",
                ir=_decision_ir(question, decision),
            )
        if decision.get("comment"):
            return json.dumps(
                {"answer": decision.get("answer"), "comment": decision["comment"]}
            )
        return json.dumps(decision.get("answer"))

    ask_user.metadata = {"categories": ["UTILITY"], "provenance": True}
    ask_user.tags = ["UTILITY"]

    from lobster.tools._async_compat import enable_sync_fallback

    return enable_sync_fallback(ask_user)


# Backward-compatible module-level instance (no LLM).
ask_user = create_ask_user_tool()
