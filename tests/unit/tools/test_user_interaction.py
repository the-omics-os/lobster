"""Tests for the ask_user and get_dashboard_schema HITL tools.

Most questions run through a real checkpointed LangGraph (no LLM): the graph
pauses in ask_user, the test answers as the interface would, and the graph
resumes. The factory tests patch interrupt() and call the tool directly.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Send

from lobster.tools.user_interaction import (
    ask_user,
    create_ask_user_tool,
    create_dashboard_schema_tool,
)

SOURCE = {
    "fields": [
        {
            "key": "source",
            "type": "choice",
            "options": [
                {"id": "SRC-A", "label": "Author matrix"},
                "SRC-B",
                {"id": "SRC-C", "unavailable": "Normalized; build unknown"},
            ],
        }
    ]
}
CONFIG = {"configurable": {"thread_id": "t"}}


def _graph(*calls, tool=None):
    """A graph whose agent makes the given ask_user calls once, then stops.

    Each call is (tool_call_id, question, context). Like the supervisor's
    create_react_agent (v2), every tool call runs as its own task (Send), so
    two calls in one message pause together.
    """

    def agent(state):
        if any(m.type == "tool" for m in state["messages"]):
            return {"messages": [AIMessage(content="done")]}
        tool_calls = [
            {"name": "ask_user", "id": cid, "args": {"question": q, "context": ctx}}
            for cid, q, ctx in calls
        ]
        return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([tool or create_ask_user_tool()]))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=InMemorySaver())


def _route(state):
    calls = state["messages"][-1].tool_calls
    if not calls:
        return END
    return [
        Send(
            "tools",
            {"__type": "tool_call_with_context", "tool_call": c, "state": state},
        )
        for c in calls
    ]


def _run(graph, graph_input):
    """Stream until the graph stops; return the pending interrupts."""
    for _ in graph.stream(graph_input, CONFIG, stream_mode="updates"):
        pass
    return [i for task in graph.get_state(CONFIG).tasks for i in task.interrupts]


def _tool_results(graph):
    return {
        m.tool_call_id: json.loads(m.content)
        for m in graph.get_state(CONFIG).values["messages"]
        if m.type == "tool"
    }


def _reply(request, answer=None, action="submit", **extra):
    """An answer envelope for a request, as the interface would build it."""
    reply = {
        "responds_to": {
            "decision_id": request["decision_id"],
            "request_version": request["request_version"],
        },
        "action": action,
        **extra,
    }
    if answer is not None:
        reply["answer"] = answer
    return reply


def _ask(context, question="Which count source?", tool=None):
    """Start one ask_user call; return the graph and its pending interrupt."""
    graph = _graph(("call_1", question, context), tool=tool)
    (pending,) = _run(graph, {"messages": []})
    return graph, pending


def _answer_pending(graph, pending, answer=None, action="submit", **extra):
    """Answer the pending question; return the next pending one, or None."""
    reply = _reply(pending.value, answer, action=action, **extra)
    after = _run(graph, Command(resume={pending.id: reply}))
    assert len(after) <= 1
    return after[0] if after else None


def _ask_and_answer(context, make_reply):
    """Run one ask_user call; answer the pause with make_reply(request)."""
    graph = _graph(("call_1", "Which count source?", context))
    (pending,) = _run(graph, {"messages": []})
    _run(graph, Command(resume={pending.id: make_reply(pending.value)}))
    return pending.value, _tool_results(graph)["call_1"]


# --- Tool surface ---


def _invoke(tool, question, context=None, call_id="call_1"):
    """Call the tool as the supervisor's ToolNode does; return the result JSON."""
    message = tool.invoke(
        {
            "type": "tool_call",
            "name": "ask_user",
            "id": call_id,
            "args": {"question": question, "context": context},
        }
    )
    return json.loads(message.content)


def _submit(answer):
    """A patched interrupt() that submits ``answer`` to whatever it is asked."""
    return lambda request: _reply(request, answer)


def test_ask_user_has_aquadif_metadata():
    """Module-level ask_user (backward compat) has AQUADIF metadata."""
    assert hasattr(ask_user, "metadata")
    assert ask_user.metadata["categories"] == ["UTILITY"]
    assert ask_user.metadata["provenance"] is False
    assert ask_user.tags == ["UTILITY"]


def test_ask_user_calls_interrupt_with_component_selection():
    """ask_user should map the question and call interrupt() with a DecisionRequest."""
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.side_effect = _submit({"answer": {"id": "yes"}})
        result = _invoke(ask_user, "Proceed?")

    mock_interrupt.assert_called_once()
    call_arg = mock_interrupt.call_args[0][0]
    assert isinstance(call_arg, dict)
    assert call_arg["schema"] == "lobster.ask/1"
    assert call_arg["decision_id"] == "dec-call_1"
    assert call_arg["fields"][0]["type"] == "choice"

    # The result is the checked answer, as JSON.
    assert result["action"] == "submit"
    assert result["answer"] == {"answer": {"id": "yes"}}


def test_ask_user_with_options_context():
    """ask_user with options context should produce a choice field."""
    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.side_effect = _submit({"answer": {"id": "CPM"}})
        _invoke(
            ask_user,
            "Which normalization?",
            {"options": ["DESeq2", "CPM", "TMM"]},
        )

    field = mock_interrupt.call_args[0][0]["fields"][0]
    assert field["type"] == "choice"
    assert [option["id"] for option in field["options"]] == ["DESeq2", "CPM", "TMM"]


# --- Factory pattern tests ---


def test_create_ask_user_tool_returns_tool_with_metadata():
    """Factory-created tool has AQUADIF metadata."""
    tool = create_ask_user_tool(llm=None)
    assert hasattr(tool, "metadata")
    assert tool.metadata["categories"] == ["UTILITY"]
    assert tool.metadata["provenance"] is False
    assert tool.tags == ["UTILITY"]


def _text_question():
    from lobster.services.interaction.component_schemas import AskUserContext

    return AskUserContext(fields=[{"key": "answer", "type": "text"}])


def test_create_ask_user_tool_with_llm_passes_llm_to_mapper():
    """Factory-created tool passes LLM to map_question."""
    mock_llm = MagicMock()
    tool = create_ask_user_tool(llm=mock_llm)

    with (
        patch("langgraph.types.interrupt") as mock_interrupt,
        patch("lobster.services.interaction.component_mapper.map_question") as mock_map,
    ):
        mock_map.return_value = _text_question()
        mock_interrupt.side_effect = _submit({"answer": "hello"})
        _invoke(tool, "How to proceed?")

    mock_map.assert_called_once()
    _, kwargs = mock_map.call_args
    assert kwargs.get("llm") is mock_llm


def test_create_ask_user_tool_calls_interrupt_with_selection():
    """Factory-created tool calls interrupt() with a DecisionRequest dict."""
    tool = create_ask_user_tool(llm=None)

    with patch("langgraph.types.interrupt") as mock_interrupt:
        mock_interrupt.side_effect = _submit({"answer": {"id": "yes"}})
        _invoke(tool, "Continue?")

    mock_interrupt.assert_called_once()
    call_arg = mock_interrupt.call_args[0][0]
    assert isinstance(call_arg, dict)
    assert call_arg["schema"] == "lobster.ask/1"
    assert "fields" in call_arg


def test_create_ask_user_tool_no_llm_backward_compat():
    """create_ask_user_tool(llm=None) works like the old standalone ask_user."""
    tool = create_ask_user_tool(llm=None)

    with (
        patch("langgraph.types.interrupt") as mock_interrupt,
        patch("lobster.services.interaction.component_mapper.map_question") as mock_map,
    ):
        mock_map.return_value = _text_question()
        mock_interrupt.side_effect = _submit({"answer": "ok"})
        _invoke(tool, "ok?")

    _, kwargs = mock_map.call_args
    assert kwargs.get("llm") is None


def test_tools_have_aquadif_metadata():
    for tool in (ask_user, create_ask_user_tool(), create_dashboard_schema_tool()):
        assert tool.metadata == {"categories": ["UTILITY"], "provenance": False}
        assert tool.tags == ["UTILITY"]


def test_llm_sees_only_question_and_context():
    properties = ask_user.tool_call_schema.model_json_schema()["properties"]
    assert list(properties) == ["question", "context"]


def test_docstring_lists_views_from_the_registry():
    description = create_ask_user_tool().description
    for view in ('"slider" (number with min, max)', '"plate_map"', '"dashboard"'):
        assert view in description
    assert "get_dashboard_schema" in description


# --- One question, through a real graph ---


def test_question_pauses_with_a_decision_request():
    graph = _graph(("call_1", "Which count source?", SOURCE))
    (pending,) = _run(graph, {"messages": []})
    request = pending.value
    assert request["schema"] == "lobster.ask/1"
    assert request["decision_id"] == "dec-call_1"
    assert request["request_version"] == 1
    assert request["question"] == "Which count source?"
    options = request["fields"][0]["options"]
    assert options[2] == {
        "id": "SRC-C",
        "label": "SRC-C",
        "unavailable": "Normalized; build unknown",
    }


def test_submitted_answer_reaches_the_supervisor():
    """ask_user runs twice (pause, then resume); the stable decision id matches."""
    request, result = _ask_and_answer(
        SOURCE, lambda r: _reply(r, {"source": {"id": "SRC-A"}}, comment="raw counts")
    )
    assert result["decision_id"] == request["decision_id"] == "dec-call_1"
    assert result["action"] == "submit"
    assert result["answer"] == {"source": {"id": "SRC-A"}}
    assert result["comment"] == "raw counts"


def test_answer_that_breaks_the_question_is_asked_again():
    graph, first = _ask(SOURCE)
    second = _answer_pending(graph, first, {"source": {"id": "SRC-C"}})
    assert second.id == first.id  # same pause, asked again
    assert second.value["request_version"] == 2
    assert second.value["problems"] == [
        {
            "path": "source",
            "message": "option 'SRC-C' is unavailable (Normalized; build unknown)",
        }
    ]
    assert "call_1" not in _tool_results(graph)


def test_answer_to_an_old_question_version_is_asked_again():
    graph, first = _ask(SOURCE)
    second = _answer_pending(graph, first, {"source": {"id": "SRC-C"}})
    old = _reply(first.value, {"source": {"id": "SRC-B"}})  # still says version 1
    (third,) = _run(graph, Command(resume={second.id: old}))
    assert third.value["request_version"] == 3
    assert third.value["problems"][0]["path"] == "responds_to.request_version"
    assert third.value["fields"][0]["default"] == {"id": "SRC-B"}
    assert _answer_pending(graph, third, {"source": {"id": "SRC-B"}}) is None
    result = _tool_results(graph)["call_1"]
    assert (result["answer"], result["attempts"]) == ({"source": {"id": "SRC-B"}}, 3)


def test_old_style_reply_is_asked_again_not_passed_through():
    graph, first = _ask(SOURCE)
    (second,) = _run(
        graph, Command(resume={first.id: {"selected": "SRC-A", "index": 0}})
    )
    assert second.value["request_version"] == 2
    assert {p["path"] for p in second.value["problems"]} >= {"responds_to", "action"}
    assert "default" not in second.value["fields"][0]


def test_defer_resumes_with_no_answer():
    _, result = _ask_and_answer(SOURCE, lambda r: _reply(r, action="defer"))
    assert result == {
        "decision_id": "dec-call_1",
        "answer_id": result["answer_id"],
        "action": "defer",
        "attempts": 1,
    }


def test_invalid_context_never_pauses():
    bad = {"fields": [{"key": "cut", "type": "number", "view": "slider", "min": 0}]}
    graph = _graph(("call_1", "Cutoff?", bad))
    assert _run(graph, {"messages": []}) == []
    result = _tool_results(graph)["call_1"]
    assert result["action"] == "invalid_context"
    assert any("view 'slider' needs max" in p["message"] for p in result["problems"])


def test_hint_style_context_still_works():
    graph = _graph(("call_1", "Which method?", {"options": ["DESeq2", "edgeR"]}))
    (pending,) = _run(graph, {"messages": []})
    field = pending.value["fields"][0]
    assert field["type"] == "choice"
    assert [o["id"] for o in field["options"]] == ["DESeq2", "edgeR"]


def test_two_pending_decisions_answered_out_of_order():
    """Each answer reaches its own question, whatever order they arrive in."""
    groups = {
        "fields": [
            {
                "key": "treated",
                "type": "choice",
                "count": {"min": 1},
                "options": ["S05", "S06", "S07"],
            }
        ]
    }
    graph = _graph(("call_a", "Which source?", SOURCE), ("call_b", "Treated?", groups))
    pending = {p.value["decision_id"]: p for p in _run(graph, {"messages": []})}
    assert sorted(pending) == ["dec-call_a", "dec-call_b"]
    a, b = pending["dec-call_a"], pending["dec-call_b"]
    resume = {
        b.id: _reply(b.value, {"treated": [{"id": "S07"}, {"id": "S05"}]}),
        a.id: _reply(a.value, {"source": {"id": "SRC-B"}}),
    }
    _run(graph, Command(resume=resume))
    results = _tool_results(graph)
    assert results["call_a"]["answer"] == {"source": {"id": "SRC-B"}}
    assert results["call_b"]["answer"] == {"treated": [{"id": "S07"}, {"id": "S05"}]}


# --- Dashboard ---

DASHBOARD = {
    "view": "dashboard",
    "fields": [
        {**SOURCE["fields"][0], "question": "Count source?"},
        {
            "key": "groups",
            "type": "choice",
            "question": "Group of each sample?",
            "view": "plate_map",
            "depends_on": ["source"],
            "for_each": ["S01", "S05"],
            "options": ["control", "treated"],
        },
        {
            "key": "mito",
            "type": "number",
            "question": "Mito cutoff?",
            "view": "slider",
            "min": 0,
            "max": 0.5,
            "depends_on": ["groups"],
        },
    ],
}


def test_dashboard_schema_tool_explains_depends_on():
    text = create_dashboard_schema_tool().invoke({})
    assert '"view": "dashboard"' in text
    assert "depends_on" in text


def test_dashboard_question_pauses_and_answers():
    request, result = _ask_and_answer(
        DASHBOARD,
        lambda r: _reply(
            r,
            {
                "source": {"id": "SRC-A"},
                "groups": {"S01": {"id": "control"}, "S05": {"id": "treated"}},
                "mito": 0.2,
            },
        ),
    )
    assert request["view"] == "dashboard"
    assert request["fields"][1]["depends_on"] == ["source"]
    assert result["answer"]["groups"] == {
        "S01": {"id": "control"},
        "S05": {"id": "treated"},
    }


def test_dashboard_section_it_cannot_draw_is_refused():
    record = {
        "key": "range",
        "type": "record",
        "question": "Range?",
        "fields": [{"key": "lo", "type": "number"}],
    }
    bad = {**DASHBOARD, "fields": [*DASHBOARD["fields"], record]}
    graph = _graph(("call_1", "Set up the comparison", bad))
    assert _run(graph, {"messages": []}) == []
    result = _tool_results(graph)["call_1"]
    assert result["action"] == "invalid_context"
    assert any(
        "a record can't be a section" in p["message"] for p in result["problems"]
    )


# --- Re-ask loop (real checkpointed graph, no LLM) ---

GROUPS = {
    "key": "groups",
    "type": "choice",
    "question": "Group of each sample?",
    "for_each": ["S01", "S02", "S03", "S04"],
    "options": ["control", "treated"],
}
QC = {
    "key": "qc",
    "type": "choice",
    "question": "Review flagged samples",
    "for_each": ["S01", "S03"],
    "options": [
        "retain",
        {"id": "exclude", "fields": [{"key": "reason", "type": "text"}]},
        "defer",
    ],
}
KEPT = {
    "check": "each_option_kept",
    "field": "groups",
    "options": ["control", "treated"],
    "minus": {"field": "qc", "options": ["exclude", "defer"]},
}
REVIEW = {"fields": [GROUPS, QC], "checks": [KEPT]}
GOOD_GROUPS = {
    "S01": {"id": "control"},
    "S02": {"id": "control"},
    "S03": {"id": "treated"},
    "S04": {"id": "treated"},
}


def test_rejected_answer_is_asked_again_prefilled():
    """Bad rows are dropped, good rows kept; a valid answer then needs 2 attempts."""
    graph, first = _ask(REVIEW, "Review groups and QC")
    submitted = {
        "groups": {**GOOD_GROUPS, "S02": {"id": "placebo"}},
        "qc": {"S01": {"id": "exclude"}, "S03": {"id": "retain"}},
    }
    second = _answer_pending(graph, first, submitted)
    request = second.value
    assert (request["decision_id"], request["request_version"]) == ("dec-call_1", 2)
    assert [p["path"] for p in request["problems"]] == ["groups.S02", "qc.S01.fields"]
    groups, qc = request["fields"]
    assert groups["default"] == {k: v for k, v in GOOD_GROUPS.items() if k != "S02"}
    assert qc["default"] == {"S03": {"id": "retain"}}

    fixed = {
        "groups": GOOD_GROUPS,
        "qc": {
            "S01": {"id": "exclude", "fields": {"reason": "low library size"}},
            "S03": {"id": "retain"},
        },
    }
    assert _answer_pending(graph, second, fixed) is None
    result = _tool_results(graph)["call_1"]
    assert (result["action"], result["attempts"]) == ("submit", 2)
    assert result["answer"]["qc"]["S01"]["fields"] == {"reason": "low library size"}


def test_emptied_group_is_asked_again_with_answers_kept():
    graph, first = _ask(REVIEW, "Review groups and QC")
    submitted = {
        "groups": GOOD_GROUPS,
        "qc": {
            "S01": {"id": "exclude", "fields": {"reason": "degraded"}},
            "S03": {"id": "retain"},
        },
    }
    # Moving S02 to treated leaves S01 as the only control, and QC excludes it.
    emptied = {
        **submitted,
        "groups": {**GOOD_GROUPS, "S02": {"id": "treated"}},
    }
    second = _answer_pending(graph, first, emptied)
    message = "'control' has no items left after removing S01 (qc: exclude)"
    assert second.value["problems"] == [
        {"path": "groups", "message": message},
        {"path": "qc", "message": message},
    ]
    assert second.value["fields"][0]["default"] == emptied["groups"]
    assert second.value["fields"][1]["default"] == emptied["qc"]
    assert _answer_pending(graph, second, submitted) is None
    assert _tool_results(graph)["call_1"]["attempts"] == 2


def test_answer_for_another_decision_is_not_consumed():
    graph, first = _ask(SOURCE)
    misrouted = _reply(first.value, {"source": {"id": "SRC-B"}})
    misrouted["responds_to"]["decision_id"] = "dec-someone-else"
    (second,) = _run(graph, Command(resume={first.id: misrouted}))
    assert second.value["request_version"] == 2
    assert second.value["problems"][0]["path"] == "responds_to.decision_id"
    assert "default" not in second.value["fields"][0]  # no pre-fill
    assert "call_1" not in _tool_results(graph)


@pytest.mark.parametrize("action", ["defer", "cancel"])
def test_defer_and_cancel_end_the_call_even_for_an_old_version(action):
    graph, first = _ask(SOURCE)
    second = _answer_pending(graph, first, {"source": {"id": "SRC-C"}})
    old = _reply(first.value, action=action)  # version 1; current is 2
    assert _run(graph, Command(resume={second.id: old})) == []
    result = _tool_results(graph)["call_1"]
    assert (result["action"], result["attempts"]) == (action, 2)
    assert "answer" not in result


def test_revise_is_asked_again():
    graph, first = _ask(SOURCE)
    second = _answer_pending(graph, first, action="revise")
    assert second.value["request_version"] == 2
    assert second.value["problems"][0]["path"] == "action"


def test_five_rejections_in_a_row_return_rejected():
    graph, pending = _ask(SOURCE)
    for attempt in range(1, 5):
        pending = _answer_pending(graph, pending, {"source": {"id": "SRC-C"}})
        assert pending.value["request_version"] == attempt + 1
    assert _answer_pending(graph, pending, {"source": {"id": "SRC-C"}}) is None
    result = _tool_results(graph)["call_1"]
    assert (result["action"], result["attempts"]) == ("rejected", 5)
    assert result["problems"][0]["path"] == "source"
    assert "nothing was decided" in result["note"]


def test_resume_with_a_non_pending_interrupt_id():
    """LangGraph hands a dict whose key is not pending to the pending pause as
    its value; that is no answer, so the question is asked again."""
    graph, first = _ask(SOURCE)
    reply = _reply(first.value, {"source": {"id": "SRC-A"}})
    (second,) = _run(graph, Command(resume={"not-a-pending-id": reply}))
    assert second.id == first.id
    assert second.value["request_version"] == 2
    assert "call_1" not in _tool_results(graph)


def test_free_text_answer_is_marked_unchecked_and_not_reasked():
    graph, first = _ask(REVIEW, "Review groups and QC")
    after = _answer_pending(
        graph, first, action="free_text", text="S01 S02 control, rest treated"
    )
    assert after is None
    result = _tool_results(graph)["call_1"]
    assert result["action"] == "free_text"
    assert result["text"] == "S01 S02 control, rest treated"
    assert result["checked"] is False
    assert "confirm" in result["note"]
    assert result["attempts"] == 1


def test_hint_option_marked_unavailable_stays_unavailable():
    hint = {"options": [{"id": "SRC-C", "unavailable": "normalized"}, "SRC-A"]}
    graph, first = _ask(hint, "Which source?")
    assert first.value["fields"][0]["options"][0]["unavailable"] == "normalized"
    second = _answer_pending(graph, first, {"answer": {"id": "SRC-C"}})
    assert "unavailable" in second.value["problems"][0]["message"]


# --- Request-time checks (once, in ask_user) ---


def _invalid(context):
    graph = _graph(("call_1", "Review", context))
    assert _run(graph, {"messages": []}) == []  # never pauses
    result = _tool_results(graph)["call_1"]
    assert result["action"] == "invalid_context"
    return result["problems"]


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"check": "nope"}, "unknown check"),
        ({**KEPT, "field": "missing"}, "no field 'missing'"),
        ({**KEPT, "options": ["control", "placebo"]}, "unknown option 'placebo'"),
        ({**KEPT, "colour": "red"}, "unknown parameter"),
        ({"check": "each_option_kept", "field": "note", "options": ["x"]}, "per-item"),
    ],
)
def test_bad_check_specs_never_pause(spec, message):
    note = {"key": "note", "type": "text", "question": "Note?"}
    context = {"fields": [GROUPS, QC, note], "checks": [spec]}
    problems = _invalid(context)
    assert any(message in p["message"] for p in problems)


def test_reason_required_without_a_reason_field_never_pauses():
    qc = {**QC, "options": ["retain", "exclude"]}
    spec = {"check": "reason_required", "field": "qc", "options": ["exclude"]}
    (problem,) = _invalid({"fields": [qc], "checks": [spec]})
    assert problem["path"] == "qc"


def test_no_default_for_a_listed_item_never_pauses():
    groups = {**GROUPS, "default": {"S04": {"id": "treated"}}}
    spec = {"check": "no_default_for", "field": "groups", "items": ["S04"]}
    (problem,) = _invalid({"fields": [groups], "checks": [spec]})
    assert "pre-fills S04" in problem["message"]


def test_no_default_for_a_listed_option_never_pauses():
    qc = {**QC, "default": {"S03": {"id": "exclude", "fields": {"reason": "warned"}}}}
    spec = {"check": "no_default_for", "field": "qc", "options": ["exclude"]}
    (problem,) = _invalid({"fields": [qc], "checks": [spec]})
    assert "'exclude'" in problem["message"]


def test_reask_prefilled_with_the_users_own_answer_does_not_trip_no_default_for():
    """A14: request-time checks run once, on the supervisor's context."""
    spec = {"check": "no_default_for", "field": "groups", "items": ["S04"]}
    context = {"fields": [GROUPS], "checks": [spec, {**KEPT, "minus": None}]}
    graph, first = _ask(context, "Groups?")
    all_treated = {s: {"id": "treated"} for s in ["S01", "S02", "S03", "S04"]}
    second = _answer_pending(graph, first, {"groups": all_treated})
    assert second is not None  # re-asked, not invalid_context
    assert second.value["fields"][0]["default"]["S04"] == {"id": "treated"}
    assert _answer_pending(graph, second, {"groups": GOOD_GROUPS}) is None
    assert _tool_results(graph)["call_1"]["action"] == "submit"


# --- Request cache ---


class _ChangingLLM:
    """A fake model that builds a different question on every call."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        from lobster.services.interaction.component_schemas import AskUserContext

        self.calls += 1
        return AskUserContext(fields=[{"key": f"answer_{self.calls}", "type": "text"}])


def test_replay_reuses_the_cached_request():
    """The LLM mapper runs once; the resumed replay checks the answer against
    the question the user saw, so it is accepted and nothing loops."""
    llm = _ChangingLLM()
    graph, first = _ask(
        None, "How should we handle these values?", tool=create_ask_user_tool(llm=llm)
    )
    assert first.value["fields"][0]["key"] == "answer_1"
    assert _answer_pending(graph, first, {"answer_1": "drop them"}) is None
    assert llm.calls == 1
    result = _tool_results(graph)["call_1"]
    assert (result["action"], result["answer"], result["attempts"]) == (
        "submit",
        {"answer_1": "drop them"},
        1,
    )


def test_replay_after_a_rejection_still_uses_the_first_build():
    llm = _ChangingLLM()
    tool = create_ask_user_tool(llm=llm)
    graph, first = _ask(None, "How should we handle these values?", tool=tool)
    second = _answer_pending(graph, first, {"answer_1": "   "})
    assert second.value["fields"][0]["key"] == "answer_1"
    assert _answer_pending(graph, second, {"answer_1": "keep"}) is None
    assert llm.calls == 1
    assert _tool_results(graph)["call_1"]["attempts"] == 2


def _cached_keys():
    from lobster.tools import user_interaction

    return [k for k in user_interaction._REQUESTS if k.startswith("dec-call_cache")]


def test_cache_is_kept_while_paused_and_removed_when_the_tool_returns():
    graph = _graph(("call_cache", "Which count source?", SOURCE))
    (first,) = _run(graph, {"messages": []})
    assert len(_cached_keys()) == 1  # pause keeps it for the replay
    second = _answer_pending(graph, first, {"source": {"id": "SRC-C"}})
    assert len(_cached_keys()) == 1
    assert _answer_pending(graph, second, {"source": {"id": "SRC-A"}}) is None
    assert _cached_keys() == []


def test_cache_is_removed_on_invalid_context_and_errors():
    bad = {"fields": [{"key": "cut", "type": "number", "view": "slider", "min": 0}]}
    graph = _graph(("call_cache", "Cutoff?", bad))
    _run(graph, {"messages": []})
    assert _cached_keys() == []


def test_cache_is_bounded():
    from lobster.tools import user_interaction

    saved = dict(user_interaction._REQUESTS)
    try:
        user_interaction._REQUESTS.clear()
        for n in range(user_interaction._MAX_CACHED + 3):
            user_interaction._remember(f"dec-{n}", [])
        assert len(user_interaction._REQUESTS) == user_interaction._MAX_CACHED
        assert "dec-0" not in user_interaction._REQUESTS
        assert f"dec-{user_interaction._MAX_CACHED + 2}" in user_interaction._REQUESTS
    finally:
        user_interaction._REQUESTS.clear()
        user_interaction._REQUESTS.update(saved)


# --- data_files fingerprints ---


def _data_workspace(tmp_path):
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "counts.csv").write_text("gene,S01\nG1,5\n")
    return tmp_path


def _ask_about_file(tmp_path, **context):
    workspace = _data_workspace(tmp_path)
    tool = create_ask_user_tool(workspace_path=workspace)
    graph, first = _ask(
        {**SOURCE, "data_files": ["data/counts.csv"], **context}, tool=tool
    )
    return workspace, graph, first


def test_data_files_are_fingerprinted_when_asking(tmp_path):
    import hashlib

    _, _, first = _ask_about_file(tmp_path)
    assert first.value["snapshot"] == {
        "files": [
            {
                "path": "data/counts.csv",
                "sha256": hashlib.sha256(b"gene,S01\nG1,5\n").hexdigest(),
            }
        ]
    }


def test_unchanged_data_is_accepted(tmp_path):
    _, graph, first = _ask_about_file(tmp_path)
    assert _answer_pending(graph, first, {"source": {"id": "SRC-A"}}) is None
    assert _tool_results(graph)["call_1"]["action"] == "submit"


def test_data_edited_before_submit_is_asked_again_with_answers_kept(tmp_path):
    workspace, graph, first = _ask_about_file(tmp_path)
    (workspace / "data" / "counts.csv").write_text("gene,S01\nG1,6\n")
    second = _answer_pending(graph, first, {"source": {"id": "SRC-A"}})
    assert second.value["problems"] == [
        {
            "path": "data_files",
            "message": "the data changed while you were reviewing it: data/counts.csv",
        }
    ]
    assert second.value["fields"][0]["default"] == {"id": "SRC-A"}
    assert second.value["snapshot"] != first.value["snapshot"]  # fresh fingerprints
    assert _answer_pending(graph, second, {"source": {"id": "SRC-A"}}) is None
    assert _tool_results(graph)["call_1"]["attempts"] == 2


def test_deleted_data_file_is_stale(tmp_path):
    workspace, graph, first = _ask_about_file(tmp_path)
    (workspace / "data" / "counts.csv").unlink()
    second = _answer_pending(graph, first, {"source": {"id": "SRC-A"}})
    assert "data/counts.csv" in second.value["problems"][0]["message"]
    third = _answer_pending(graph, second, {"source": {"id": "SRC-A"}})
    assert third is not None  # still gone: still stale


@pytest.mark.parametrize("path", ["../outside.csv", "/etc/hostname", "data/none.csv"])
def test_untrusted_or_missing_data_file_is_invalid_context(tmp_path, path):
    workspace = _data_workspace(tmp_path / "ws")
    (tmp_path / "outside.csv").write_text("x")
    tool = create_ask_user_tool(workspace_path=workspace)
    graph = _graph(("call_1", "Source?", {**SOURCE, "data_files": [path]}), tool=tool)
    assert _run(graph, {"messages": []}) == []
    (problem,) = _tool_results(graph)["call_1"]["problems"]
    assert problem["path"] == "data_files"


def test_data_files_without_a_workspace_is_invalid_context():
    graph = _graph(("call_1", "Source?", {**SOURCE, "data_files": ["data/a.csv"]}))
    assert _run(graph, {"messages": []}) == []
    assert _tool_results(graph)["call_1"]["action"] == "invalid_context"


def test_replay_reuses_the_cached_fingerprints(tmp_path):
    """Hashing happens at ask time and once per new answer, never on replay."""
    from lobster.tools import user_interaction

    real = user_interaction.fingerprint
    calls = []

    def counting(paths, workspace, missing_ok=False):
        calls.append(missing_ok)
        return real(paths, workspace, missing_ok=missing_ok)

    with patch.object(user_interaction, "fingerprint", counting):
        _, graph, first = _ask_about_file(tmp_path)
        second = _answer_pending(graph, first, {"source": {"id": "SRC-C"}})
        assert _answer_pending(graph, second, {"source": {"id": "SRC-A"}}) is None
    # ask time, answer 1, answer 2 — not again for answer 1 on the second replay
    assert calls == [False, True, True]


# --- Tool description ---


def test_docstring_lists_every_check_from_the_registry():
    from lobster.services.interaction.component_schemas import CHECKS

    description = create_ask_user_tool().description
    for name, entry in CHECKS.items():
        assert f'"{name}"' in description
        assert entry["use_when"] in description
    for phrase in ('"unavailable"', "data_files", "nothing was decided", "same shape"):
        assert phrase in description


def test_dashboard_schema_covers_lists_checks_direction_and_review():
    text = create_dashboard_schema_tool().invoke({})
    for phrase in (
        "each_option_kept",
        '"minus": {"field": "qc", "options": ["exclude", "defer"]}',
        "review screen",
        "data_files",
        "direction",
        "reason",
    ):
        assert phrase in text
    assert "for_each without plate_map" not in text
