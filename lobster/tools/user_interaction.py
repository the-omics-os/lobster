"""HITL user interaction tools for the supervisor agent.

``create_ask_user_tool`` builds ``ask_user``: it turns the supervisor's
question into decision fields (``component_mapper``), pauses the graph with a
DecisionRequest, and checks the interface's answer before returning it.

``create_dashboard_schema_tool`` builds ``get_dashboard_schema``: it hands the
supervisor the instructions for a multi-section dashboard only when needed, so
the ask_user docstring stays short.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Optional

from langchain_core.tools import InjectedToolCallId, tool
from pydantic import ValidationError

from lobster.services.interaction.component_schemas import (
    VIEWS,
    AnswerRejected,
    AnswerStale,
    DecisionAnswer,
    DecisionRequest,
    check_answer,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def create_ask_user_tool(llm: Optional["BaseChatModel"] = None):
    """Create an ask_user tool with optional LLM for formatting questions.

    Args:
        llm: Optional LLM used by the component mapper to format ambiguous
             questions as fields. When None, rules and a free-text fallback
             are used.

    Returns:
        A LangChain tool with AQUADIF UTILITY metadata.
    """

    @tool
    async def ask_user(
        question: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Pause and ask the user. Use ONLY when user input is genuinely required
        (ambiguous intent, an approval gate, or a scientific choice only they can make).
        The question might come from a sub-agent that needs user input.
        Ask ONE question per turn. For several linked decisions on one screen,
        call get_dashboard_schema first.

        context = {"fields": [field, ...]}. Several fields are answered on one screen.
        Every field:
          key       name the answer comes back under, e.g. "source"
          type      "choice" | "number" | "text" | "record"
          question  what to ask (may be left out when there is only one field)
          details   evidence or your recommendation; shown, never applied
          default   a pre-filled answer, shaped exactly like the answer
          count     {"min", "max"}: how many values; the answer becomes a list
          for_each  [item, ...]: ask the same field once per item; answer keyed by id
          view      how to draw it (see Views); never changes the answer
        choice:  options: [item, ...]. An option may carry "fields", shown only
                 when it is picked, e.g. a reason for "exclude".
        number:  min, max (value bounds), integer (true/false), unit
        text:    only when the answer can't be listed or counted
        record:  fields: [field, ...] answered together
        item:    {"id", "label", "details", "unavailable"} or a plain string.
                 id is a stable identity (sample ID, accession), never a row
                 number; label is display only; unavailable = the reason it
                 can't be picked (it is shown greyed out).

        Example: keep or exclude each flagged sample, with a reason to exclude:
          {"fields": [{"key": "qc", "type": "choice",
             "for_each": [{"id": "S02", "details": "low library size"}, "S06"],
             "options": ["retain",
                         {"id": "exclude", "fields": [{"key": "reason", "type": "text"}]}]}]}

        Args:
            question: Clear, specific question for the user.
            context: The fields to ask (see above). Omit for a single free-text answer.

        Returns:
            JSON {"decision_id", "action", "answer"}. action is "submit", "defer"
            or "cancel"; only "submit" has an answer, and it mirrors the fields:
              {"qc": {"S02": {"id": "exclude", "fields": {"reason": "..."}},
                      "S06": {"id": "retain"}}}
            "invalid_context": nothing was shown; fix context using "problems"
            and ask again. "rejected" / "stale": see "problems"."""
        from langgraph.types import interrupt

        from lobster.services.interaction.component_mapper import map_question

        try:
            asked = await asyncio.to_thread(map_question, question, context, llm=llm)
        except ValidationError as error:
            return json.dumps(
                {
                    "action": "invalid_context",
                    "problems": _problems(error),
                    "note": "Nothing was shown to the user. "
                    "Fix context and call ask_user again.",
                }
            )

        # LangGraph re-runs this function on resume, so the id must come from
        # the tool call, which stays the same, not from a fresh random value.
        request = DecisionRequest(
            decision_id=f"dec-{tool_call_id}",
            question=question,
            view=asked.view,
            fields=asked.fields,
        )

        raw = interrupt(request.to_wire())

        try:
            result = check_answer(request, DecisionAnswer.model_validate(raw))
        except AnswerStale as error:
            result = _refused(request, "stale", error.problems)
        except AnswerRejected as error:
            result = _refused(request, "rejected", error.problems)
        except ValidationError as error:
            result = _refused(request, "rejected", _problems(error))
        return json.dumps(result)

    ask_user.description += _views_section()
    ask_user.metadata = {"categories": ["UTILITY"], "provenance": False}
    ask_user.tags = ["UTILITY"]

    from lobster.tools._async_compat import enable_sync_fallback

    return enable_sync_fallback(ask_user)


_DASHBOARD_SCHEMA = """\
A dashboard asks several linked decisions on one screen. The user moves between
sections, and changing a section marks the sections that depend on it as stale
until they are checked again.

Call ask_user once, with every section filled in now:
  context = {"view": "dashboard", "fields": [section, ...]}

Rules:
- At least 2 sections. Each section is one field with a "key" and a "question".
- Order sections upstream first.
- A section may be:
    choice                    options; add count {"min", "max"} for several picks
    choice + for_each         the same choice per item; needs "view": "plate_map"
    number                    add "view": "slider" when min and max are known
    text                      only when the answer can't be listed or counted
  Not allowed in a dashboard: record, options with "fields", for_each without plate_map.
- depends_on: the keys of earlier sections whose change should make this section
  stale, e.g. "depends_on": ["source"].
- Use ids exactly as they appear in the data; never invent ids. Put evidence and
  your recommendations in "details" (shown, never applied) or "default"
  (pre-filled). Show an ineligible option with "unavailable": "<reason>"
  instead of leaving it out.

Example:
  {"view": "dashboard", "fields": [
    {"key": "genome", "type": "choice", "question": "Reference genome?",
     "options": [{"id": "GRCh38", "details": "matches the annotation"},
                 {"id": "hg19", "unavailable": "annotation is GRCh38-only"}]},
    {"key": "groups", "type": "choice", "question": "Group of each sample?",
     "view": "plate_map", "depends_on": ["genome"],
     "for_each": ["P1", "P2", "P3"], "options": ["case", "control"],
     "default": {"P1": {"id": "case"}, "P2": {"id": "case"}, "P3": {"id": "control"}}},
    {"key": "min_reads", "type": "number", "question": "Minimum reads per sample?",
     "view": "slider", "min": 0, "max": 50000, "integer": true, "default": 10000,
     "depends_on": ["groups"]}]}

The answer has one entry per section key:
  {"genome": {"id": "GRCh38"},
   "groups": {"P1": {"id": "case"}, "P2": {"id": "case"}, "P3": {"id": "control"}},
   "min_reads": 10000}"""


def create_dashboard_schema_tool():
    """Create the get_dashboard_schema tool.

    Returns:
        A LangChain tool with AQUADIF UTILITY metadata.
    """

    @tool
    def get_dashboard_schema() -> str:
        """Get the instructions for a dashboard: several linked decisions on one
        screen, where changing an upstream decision marks downstream ones stale.
        Call this before ask_user when the user must make decisions that depend
        on each other (e.g. data source, then sample groups, then QC).

        Returns:
            How to fill ask_user's context for a dashboard, with an example."""
        return _DASHBOARD_SCHEMA

    get_dashboard_schema.metadata = {"categories": ["UTILITY"], "provenance": False}
    get_dashboard_schema.tags = ["UTILITY"]
    return get_dashboard_schema


def _views_section() -> str:
    """The docstring's list of views, generated from the VIEWS registry."""
    lines = []
    for name, entry in VIEWS.items():
        needs = f" with {', '.join(entry['needs'])}" if entry["needs"] else ""
        lines.append(
            f'  - "{name}" ({entry["applies_to"]}{needs}): {entry["use_when"]}'
        )
        if entry.get("example"):
            lines.append(f"      e.g. {json.dumps(entry['example'])}")
    return '\n\nViews (leave "view" out for the plain control):\n' + "\n".join(lines)


def _problems(error: ValidationError) -> List[str]:
    """One readable line per pydantic error, e.g. 'fields.0.number: ...'."""
    return [
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors()
    ]


def _refused(
    request: DecisionRequest, action: str, problems: List[str]
) -> Dict[str, Any]:
    """The tool result for an answer that didn't fit its request."""
    return {"decision_id": request.decision_id, "action": action, "problems": problems}


# Backward-compatible module-level instance (no LLM).
ask_user = create_ask_user_tool()
