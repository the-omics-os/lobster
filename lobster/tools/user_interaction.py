"""HITL user interaction tools for the supervisor agent.

``create_ask_user_tool`` builds ``ask_user``: it turns the supervisor's
question into decision fields (``component_mapper``), pauses the graph with a
DecisionRequest, and checks the interface's answer before returning it. A
refused answer brings the same question back (next version, answers kept,
problems shown) inside the same tool call.

``create_dashboard_schema_tool`` builds ``get_dashboard_schema``: it hands the
supervisor the instructions for a multi-section dashboard only when needed, so
the ask_user docstring stays short.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Optional, Union

from langchain_core.tools import InjectedToolCallId, tool
from pydantic import ValidationError

from lobster.services.interaction.component_schemas import (
    CHECKS,
    VIEWS,
    AnswerMisrouted,
    AnswerRejected,
    AnswerStale,
    DecisionAnswer,
    DecisionRequest,
    Problem,
    check_answer,
    check_request,
    fingerprint,
    reask,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# After this many refused answers in a row the supervisor is told instead, so
# an interface (or a user) that keeps sending the same bad answer can't loop.
_MAX_REJECTED = 5

_FREE_TEXT_NOTE = (
    "The user's terminal could not show this question, so they answered in "
    "their own words. This text was not checked against the question's rules. "
    "Restate how you understood it to the user and confirm before acting."
)
_REJECTED_NOTE = (
    "The user's answers kept failing these checks; nothing was decided. "
    "Tell the user what failed and how to answer, or ask differently."
)

# --- Request cache ---------------------------------------------------------
#
# LangGraph pauses ask_user by raising inside interrupt() and, on resume,
# runs the whole function again; earlier interrupt() calls then return their
# recorded answers. Without a cache that replay would rebuild the question
# (the LLM mapper path may build a different one) and re-read the data files,
# so an answer could be judged against a question the user never saw.
#
# So each decision keeps the requests shown so far, version by version:
# shown[i] is version i + 1, and every version but the last was refused. The
# replay walks that list instead of judging old answers again.
#
# Memory lifecycle: process-local; values are plain pydantic models (no
# closures, no data_manager, graph or store references); at most _MAX_CACHED
# entries, oldest evicted; an entry is removed when its tool call returns
# (any return path), and kept only while the call is paused. An evicted or
# lost entry (restart, other process) just means the question is rebuilt.
_MAX_CACHED = 256
_REQUESTS: "OrderedDict[str, List[DecisionRequest]]" = OrderedDict()
_REQUESTS_LOCK = threading.Lock()


def _cache_key(decision_id: str, question: str, context: Any, workspace: Any) -> str:
    """decision_id plus the call's inputs and workspace, so a reused tool-call
    id can't pick up another question or another workspace's fingerprints."""
    inputs = json.dumps(
        [question, context, str(workspace)], sort_keys=True, default=str
    )
    return f"{decision_id}:{hashlib.sha256(inputs.encode()).hexdigest()[:16]}"


def _recall(key: str) -> Optional[List[DecisionRequest]]:
    with _REQUESTS_LOCK:
        shown = _REQUESTS.get(key)
        if shown is not None:
            _REQUESTS.move_to_end(key)
        return shown


def _remember(key: str, shown: List[DecisionRequest]) -> None:
    with _REQUESTS_LOCK:
        _REQUESTS[key] = shown
        _REQUESTS.move_to_end(key)
        while len(_REQUESTS) > _MAX_CACHED:
            _REQUESTS.popitem(last=False)


def _forget(key: str) -> None:
    with _REQUESTS_LOCK:
        _REQUESTS.pop(key, None)


def create_ask_user_tool(
    llm: Optional["BaseChatModel"] = None,
    workspace_path: Optional[Union[str, Path]] = None,
):
    """Create an ask_user tool with optional LLM for formatting questions.

    Args:
        llm: Optional LLM used by the component mapper to format ambiguous
             questions as fields. When None, rules and a free-text fallback
             are used.
        workspace_path: The session workspace. ``context.data_files`` are
             resolved inside it; without it, a question naming data files is
             refused.

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

        question: one short line. Explanations, evidence and recommendations
        go in "details".

        context = {"fields": [field, ...], "checks": [check, ...],
                   "data_files": [path, ...]}. Several fields are answered on
        one screen. Every field:
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
                 number; label is display only. Show an ineligible option or
                 item with "unavailable": "<reason>"; never leave it out.
        data_files: workspace-relative paths of the files the decision is
                 about; if one changes before the user answers, they are
                 asked again.
        When items are split into groups to compare, also ask, as its own
        choice field, which group is compared with which ("B relative to A"
        or "A relative to B").

        Example: keep or exclude each flagged sample, with a reason to exclude:
          {"fields": [{"key": "qc", "type": "choice",
             "for_each": [{"id": "S02", "details": "low library size"}, "S06"],
             "options": ["retain",
                         {"id": "exclude", "fields": [{"key": "reason", "type": "text"}]}]}]}

        Args:
            question: One short line; put explanations in "details".
            context: The fields to ask (see above). Omit for a single free-text answer.

        Returns:
            JSON {"decision_id", "action", "answer", "attempts"}. An answer that
            breaks the question or its checks is asked again automatically.
              "submit": "answer" mirrors the fields:
                {"qc": {"S02": {"id": "exclude", "fields": {"reason": "..."}},
                        "S06": {"id": "retain"}}}
              "defer" / "cancel": nothing was decided; don't continue with a
                default or a guess. If the comment says the terminal can't show
                a shape, don't ask with the same shape again.
              "free_text": "text" is the user's own words, not checked; follow "note".
              "invalid_context": nothing was shown; fix context using "problems"
                and ask again.
              "rejected": the answers kept failing the checks; see "problems"."""
        from langgraph.errors import GraphInterrupt

        decision_id = f"dec-{tool_call_id}"
        key = _cache_key(decision_id, question, context, workspace_path)
        shown = _recall(key)
        if shown is None:
            built = await asyncio.to_thread(
                _build_request, question, context, decision_id, llm, workspace_path
            )
            if not isinstance(built, DecisionRequest):
                return json.dumps(
                    {
                        "action": "invalid_context",
                        "problems": [problem.model_dump() for problem in built],
                        "note": "Nothing was shown to the user. "
                        "Fix context and call ask_user again.",
                    }
                )
            shown = [built]
            _remember(key, shown)

        paused = False
        try:
            result = await _ask_until_settled(shown, workspace_path)
        except GraphInterrupt:
            paused = True  # the replay after the resume needs the entry
            raise
        finally:
            if not paused:
                _forget(key)
        return json.dumps(result)

    ask_user.description += _views_section() + _checks_section()
    ask_user.metadata = {"categories": ["UTILITY"], "provenance": False}
    ask_user.tags = ["UTILITY"]

    from lobster.tools._async_compat import enable_sync_fallback

    return enable_sync_fallback(ask_user)


def _build_request(
    question: str,
    context: Optional[Dict[str, Any]],
    decision_id: str,
    llm: Optional["BaseChatModel"],
    workspace: Optional[Union[str, Path]],
) -> Union[DecisionRequest, List[Problem]]:
    """The first version of the request, or the problems that stop it.

    Request-time checks run here, once, on the supervisor's context; data
    files are fingerprinted here, once.
    """
    from lobster.services.interaction.component_mapper import map_question

    try:
        asked = map_question(question, context, llm=llm)
    except ValidationError as error:
        return problems_from(error)
    problems = check_request(asked)
    if problems:
        return problems
    snapshot = None
    if asked.data_files:
        if workspace is None:
            return [
                Problem(
                    path="data_files",
                    message="no workspace is open, so data files can't be "
                    "checked; leave data_files out",
                )
            ]
        try:
            snapshot = fingerprint(asked.data_files, workspace)
        except ValueError as error:
            return [Problem(path="data_files", message=str(error))]
    return DecisionRequest(
        decision_id=decision_id,
        question=question,
        view=asked.view,
        fields=asked.fields,
        checks=asked.checks,
        snapshot=snapshot,
    )


async def _ask_until_settled(
    shown: List[DecisionRequest], workspace: Optional[Union[str, Path]]
) -> Dict[str, Any]:
    """Ask, judge the answer, and ask again until it is settled.

    ``shown`` is the cached list of versions; on a replay, answers to every
    version but the last were refused before, so they are skipped.
    """
    from langgraph.errors import GraphInterrupt
    from langgraph.types import interrupt

    attempt = 0
    while True:
        request = shown[attempt]
        try:
            raw = interrupt(request.to_wire())
        except GraphInterrupt:
            # Paused at this version: later ones belong to an abandoned run.
            del shown[attempt + 1 :]
            raise
        attempt += 1
        if attempt < len(shown):
            continue
        outcome = await _judge(request, raw, workspace)
        if isinstance(outcome, dict):
            return {**outcome, "attempts": attempt}
        if attempt >= _MAX_REJECTED:
            return {
                "decision_id": request.decision_id,
                "action": "rejected",
                "problems": [problem.model_dump() for problem in outcome.problems],
                "attempts": attempt,
                "note": _REJECTED_NOTE,
            }
        shown.append(outcome)


async def _judge(
    request: DecisionRequest, raw: Any, workspace: Optional[Union[str, Path]]
) -> Union[Dict[str, Any], DecisionRequest]:
    """The tool result for this answer, or the next version of the request."""
    try:
        answer = DecisionAnswer.model_validate(raw)
    except ValidationError as error:
        return reask(request, problems_from(error), None)
    ours = answer.responds_to.decision_id == request.decision_id
    if (
        ours
        and answer.action in ("defer", "cancel")
        and answer.action in request.actions
    ):
        # Nothing is applied, so the version answered doesn't matter.
        result = {
            "decision_id": request.decision_id,
            "answer_id": answer.answer_id,
            "action": answer.action,
        }
        if answer.comment:
            result["comment"] = answer.comment
        return result
    current = None
    if (
        ours
        and answer.action == "submit"
        and answer.responds_to.request_version == request.request_version
        and request.snapshot is not None
        and workspace is not None
    ):
        paths = [entry.path for entry in request.snapshot.files]
        current = await asyncio.to_thread(fingerprint, paths, workspace, True)
    try:
        result = check_answer(request, answer, current)
    except AnswerMisrouted as error:
        return reask(request, error.problems, None)
    except AnswerStale as error:
        return reask(request, error.problems, answer.answer, snapshot=error.snapshot)
    except AnswerRejected as error:
        return reask(request, error.problems, answer.answer)
    if result["action"] == "free_text":
        result.update(checked=False, note=_FREE_TEXT_NOTE)
    return result


_DASHBOARD_SCHEMA = """\
A dashboard asks several linked decisions on one screen. The user moves between
sections; changing a section marks the sections that depend on it as stale
until they are checked again. Before anything is sent, the user confirms every
answer on a review screen.

Call ask_user once, with every section filled in now:
  context = {"view": "dashboard", "fields": [section, ...],
             "checks": [check, ...], "data_files": [path, ...]}

Rules:
- At least 2 sections. Each section is one field with a "key" and a "question".
- Order sections upstream first.
- A section may be:
    choice                    options; add count {"min", "max"} for several picks
    choice + for_each         the same choice per item, on a plate:
                              add "view": "plate_map"
    choice + for_each (list)  one row per item, one option per row, without
                              plate_map; an option may ask for one reason:
                              "fields": [{"key": "reason", "type": "text"}]
    number                    add "view": "slider" when min and max are known
    text                      only when the answer can't be listed or counted
  Not allowed in a dashboard: record, options with other fields, count on a
  list section.
- A comparison needs its direction asked explicitly: a choice of which group is
  compared with which, never assumed from the group names.
- depends_on: the keys of earlier sections whose change should make this section
  stale, e.g. "depends_on": ["source"].
- checks: rules Python enforces on the whole answer (see ask_user's Checks).
  For a QC review whose exclusions must not empty a group, use each_option_kept
  with "minus" listing every QC option that takes an item out of its group:
  exclude and defer (a deferred item does not count toward its group).
- data_files: the workspace files the decision is about, e.g. the counts table.
- Use ids exactly as they appear in the data; never invent ids. Put evidence and
  your recommendations in "details" (shown, never applied) or "default"
  (pre-filled). Show an ineligible option or item with "unavailable": "<reason>"
  instead of leaving it out.

Example:
  {"view": "dashboard", "data_files": ["data/counts.csv"], "fields": [
    {"key": "source", "type": "choice", "question": "Which count table?",
     "options": [{"id": "SRC-A", "details": "raw counts"},
                 {"id": "SRC-B", "unavailable": "already normalized"}]},
    {"key": "groups", "type": "choice", "question": "Group of each sample?",
     "view": "plate_map", "depends_on": ["source"],
     "for_each": ["P1", "P2", "P3", "P4"], "options": ["case", "control"]},
    {"key": "direction", "type": "choice", "question": "Compare which with which?",
     "depends_on": ["groups"],
     "options": [{"id": "case_vs_control", "label": "case compared with control"},
                 {"id": "control_vs_case", "label": "control compared with case"}]},
    {"key": "qc", "type": "choice", "question": "Keep each flagged sample?",
     "depends_on": ["groups"],
     "for_each": [{"id": "P2", "details": "low library size"}, "P3"],
     "options": ["retain", "defer",
                 {"id": "exclude", "fields": [{"key": "reason", "type": "text"}]}]}],
   "checks": [
    {"check": "each_option_kept", "field": "groups", "options": ["case", "control"],
     "minus": {"field": "qc", "options": ["exclude", "defer"]}},
    {"check": "reason_required", "field": "qc", "options": ["exclude"]},
    {"check": "no_default_for", "field": "qc", "options": ["exclude"]}]}

The answer has one entry per section key:
  {"source": {"id": "SRC-A"},
   "groups": {"P1": {"id": "case"}, "P2": {"id": "case"},
              "P3": {"id": "control"}, "P4": {"id": "control"}},
   "direction": {"id": "case_vs_control"},
   "qc": {"P2": {"id": "exclude", "fields": {"reason": "low library size"}},
          "P3": {"id": "retain"}}}"""


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


def _checks_section() -> str:
    """The docstring's list of checks, generated from the CHECKS registry."""
    lines = [f'  - "{name}": {entry["use_when"]}' for name, entry in CHECKS.items()]
    return (
        '\n\nChecks (context "checks": [{"check": name, ...params}]; fields are '
        "top-level keys; an answer that breaks one is asked again):\n"
        + "\n".join(lines)
    )


def problems_from(error: ValidationError) -> List[Problem]:
    """One Problem per pydantic error, its path the error's location."""
    problems = []
    for item in error.errors():
        path = ".".join(str(part) for part in item["loc"]) or "answer"
        message = item["msg"].removeprefix("Value error, ")
        problems.append(Problem(path=path, message=message))
    return problems


# Backward-compatible module-level instance (no LLM, no workspace).
ask_user = create_ask_user_tool()
