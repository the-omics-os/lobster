"""Tests for the ask_user and get_dashboard_schema HITL tools.

Most questions run through a real checkpointed LangGraph (no LLM): the graph
pauses in ask_user, the test answers as the interface would, and the graph
resumes. The factory tests patch interrupt() and call the tool directly.
"""

import json
import re
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
    # dec-<tool-call id>-<8 random hex>, with or without a graph
    assert re.fullmatch(r"dec-call_1-[0-9a-f]{8}", call_arg["decision_id"])
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
    # dec-<tool-call id>-<8 random hex>: a reused tool-call id stays distinct
    assert re.fullmatch(r"dec-call_1-[0-9a-f]{8}", request["decision_id"])
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
    assert result["decision_id"] == request["decision_id"]
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
        "decision_id": result["decision_id"],
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
    pending = {
        p.value["decision_id"].rsplit("-", 1)[0]: p
        for p in _run(graph, {"messages": []})
    }
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
    assert request["decision_id"].startswith("dec-call_1-")
    assert request["request_version"] == 2
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
    # v4 2.4: and it must not just ask the same question again.
    assert (
        "Don't ask the same question again in this turn; tell the user what "
        "failed first." in result["note"]
    )
    # v4 2.3: why it came back, once each.
    assert len(result["reasked_because"]) == 1
    assert "SRC-C" in result["reasked_because"][0]


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
    problems = _invalid({"fields": [qc], "checks": [spec]})
    # The check and the removal rule both name 'exclude'.
    assert {p["path"] for p in problems} == {"qc"} and len(problems) == 2


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


def test_plate_map_with_a_count_never_pauses():
    """Finding 4a: the plate answers one option per well, so a count could
    only end in five refused answers."""
    plate = {**GROUPS, "view": "plate_map", "count": {"min": 1}}
    (problem,) = _invalid({"fields": [plate]})
    assert "'plate_map' takes one option per item" in problem["message"]


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


def test_bare_id_default_pauses_instead_of_invalid_context():
    """U-17: the supervisor's shorthand default ("SRC-A", {"S01": "retain"})
    is read as the option; the paused request carries the full form."""
    source = {**SOURCE["fields"][0], "question": "Source?", "default": "SRC-A"}
    qc = {**QC, "default": {"S01": "retain"}}
    graph, pending = _ask({"fields": [source, qc]}, "Source and QC?")
    defaults = [field["default"] for field in pending.value["fields"]]
    assert defaults == [{"id": "SRC-A"}, {"S01": {"id": "retain"}}]
    answer = {
        "source": {"id": "SRC-A"},
        "qc": {"S01": {"id": "retain"}, "S03": {"id": "defer"}},
    }
    assert _answer_pending(graph, pending, answer) is None
    assert _tool_results(graph)["call_1"]["action"] == "submit"


# --- Replay: the question is built and each answer judged once ---


class _ChangingLLM:
    """A fake model that builds a different question on every call."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt, config=None):
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


def test_replay_in_an_async_graph_still_uses_the_first_build():
    """The same replay when the graph runs async (ainvoke/astream): ask_user's
    steps are awaited there instead of waited on, and still run once."""
    import asyncio

    llm = _ChangingLLM()
    tool = create_ask_user_tool(llm=llm)
    graph = _graph(("call_1", "How should we handle these values?", None), tool=tool)

    def run(graph_input):
        async def go():
            await graph.ainvoke(graph_input, CONFIG)
            state = await graph.aget_state(CONFIG)
            return [i for task in state.tasks for i in task.interrupts]

        return asyncio.run(go())

    def answer(pending, value):
        return run(Command(resume={pending.id: _reply(pending.value, value)}))

    (first,) = run({"messages": []})
    (second,) = answer(first, {"answer_1": "   "})
    assert second.value["fields"][0]["key"] == "answer_1"
    assert answer(second, {"answer_1": "keep"}) == []
    assert llm.calls == 1
    assert _tool_results(graph)["call_1"]["attempts"] == 2


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


def test_data_file_turned_into_a_link_out_is_stale_not_an_error(tmp_path):
    """Finding 9: an answer-time fingerprint failure is a data change; the
    user's answer is kept and they are asked again."""
    workspace, graph, first = _ask_about_file(tmp_path / "ws")
    counts = workspace / "data" / "counts.csv"
    original = counts.read_text()
    (tmp_path / "outside.csv").write_text(original)
    counts.unlink()
    counts.symlink_to(tmp_path / "outside.csv")
    second = _answer_pending(graph, first, {"source": {"id": "SRC-A"}})
    assert second.value["problems"] == [
        {
            "path": "data_files",
            "message": "the data changed while you were reviewing it: data/counts.csv",
        }
    ]
    assert second.value["fields"][0]["default"] == {"id": "SRC-A"}
    counts.unlink()
    counts.write_text(original)  # back as it was when asked
    assert _answer_pending(graph, second, {"source": {"id": "SRC-A"}}) is None
    assert _tool_results(graph)["call_1"]["attempts"] == 2


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


def test_docstring_asks_for_the_direction_of_a_group_comparison():
    """U-19: a question that sets up groups to compare also asks the direction."""
    description = create_ask_user_tool().description
    assert "which group is compared with which" in description
    assert "B relative to A" in description


def test_docstring_keeps_ruled_out_options_in_the_list():
    """Live T1/T3: a ruled-out source was dropped instead of shown unavailable."""
    description = " ".join(create_ask_user_tool().description.split())
    assert 'stays in the list with "unavailable": "<reason>"' in description
    assert 'never drop it or mention it only in "details"' in description


def test_docstring_asks_a_grouping_as_one_field_over_every_item():
    """Live T4: groups asked one field per sample, so no check could see them."""
    description = " ".join(create_ask_user_tool().description.split())
    assert "ask ONE choice field with for_each over every item" in description
    assert "never one field per item" in description
    assert "checks only see groups asked as one field" in description


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


# --- v3 findings: reused call ids, overrides ---


def test_reused_tool_call_ids_get_their_own_decisions():
    """Some providers number tool calls (call_0, call_1, ...) and can repeat
    one in a message. The random part of the id tells the calls apart, so
    each answer still reaches its own question."""
    graph = _graph(
        ("call_1", "Which source?", SOURCE), ("call_1", "Which other source?", SOURCE)
    )
    first, second = _run(graph, {"messages": []})
    ids = {first.value["decision_id"], second.value["decision_id"]}
    assert len(ids) == 2
    assert all(re.fullmatch(r"dec-call_1-[0-9a-f]{8}", i) for i in ids)
    by_question = {p.value["question"]: p for p in (first, second)}
    a, b = by_question["Which source?"], by_question["Which other source?"]
    resume = {
        a.id: _reply(a.value, {"source": {"id": "SRC-A"}}),
        b.id: _reply(b.value, {"source": {"id": "SRC-B"}}),
    }
    assert _run(graph, Command(resume=resume)) == []
    answers = sorted(
        json.loads(m.content)["answer"]["source"]["id"]
        for m in graph.get_state(CONFIG).values["messages"]
        if m.type == "tool"
    )
    assert answers == ["SRC-A", "SRC-B"]


def test_decision_id_is_the_same_on_replay():
    """The id is minted once, in the build task; a resume replays it as is."""
    graph, first = _ask(SOURCE)
    second = _answer_pending(graph, first, {"source": {"id": "SRC-C"}})
    assert second.value["decision_id"] == first.value["decision_id"]


def test_decision_id_does_not_come_from_langgraph():
    """Lobster mints the id itself (random), so the same call in the same
    graph position gets a new id each run: nothing is derived from the
    graph's task or interrupt id."""
    _, first = _ask(SOURCE)
    _, again = _ask(SOURCE)
    assert first.value["decision_id"] != again.value["decision_id"]
    assert first.value["decision_id"] not in (first.id, again.id)


PROPOSED_GROUPS = {
    "fields": [
        {
            "key": "groups",
            "type": "choice",
            "question": "Group of each sample?",
            "for_each": ["S01", "S02", "S08"],
            "options": ["control", "treated"],
            "default": {"S01": {"id": "control"}, "S02": {"id": "treated"}},
        }
    ]
}


def test_picks_changed_from_the_default_are_reported_as_overrides():
    """S01 was proposed as control and answered treated: an override. S08
    had no proposal, so answering it is not one."""
    answer = {
        "groups": {
            "S01": {"id": "treated"},
            "S02": {"id": "treated"},
            "S08": {"id": "control"},
        }
    }
    _, result = _ask_and_answer(
        PROPOSED_GROUPS, lambda request: _reply(request, answer)
    )
    assert result["changed_from_default"] == {
        "groups": {"S01": {"proposed": "control", "answered": "treated"}}
    }


def test_no_overrides_means_no_changed_from_default():
    answer = {
        "groups": {
            "S01": {"id": "control"},
            "S02": {"id": "treated"},
            "S08": {"id": "control"},
        }
    }
    _, result = _ask_and_answer(
        PROPOSED_GROUPS, lambda request: _reply(request, answer)
    )
    assert "changed_from_default" not in result


def test_overrides_compare_with_the_first_proposal_not_a_reask():
    """A re-ask's defaults are the user's own answers; the proposal is v1's."""
    graph, first = _ask(PROPOSED_GROUPS, question="Groups?")
    full = {"groups": {k: {"id": "treated"} for k in ("S01", "S02", "S08")}}
    second = _answer_pending(graph, first, {"groups": {"S01": {"id": "treated"}}})
    assert second is not None  # S02 and S08 missing: asked again
    assert _answer_pending(graph, second, full) is None
    result = _tool_results(graph)["call_1"]
    assert result["changed_from_default"] == {
        "groups": {"S01": {"proposed": "control", "answered": "treated"}}
    }


# --- items_from: the samples read from a workspace table ---

PLATE_LAYOUT = "sample,well,condition\nP1,D1,case\nP2,D2,?\n"


def _plate_from_table(tmp_path, **spec):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "layout.csv").write_text(PLATE_LAYOUT)
    items_from = {
        "file": "data/layout.csv",
        "id": "sample",
        "default": "condition",
        "position": "well",
        **spec,
    }
    context = {
        "fields": [
            {
                "key": "groups",
                "type": "choice",
                "view": "plate_map",
                "options": ["case", "control"],
                "items_from": items_from,
            }
        ]
    }
    return create_ask_user_tool(workspace_path=tmp_path), context


def test_items_from_is_read_when_the_question_is_built(tmp_path):
    import hashlib

    tool, context = _plate_from_table(tmp_path)
    _, pending = _ask(context, question="Group of each sample?", tool=tool)
    field = pending.value["fields"][0]
    assert [(item["id"], item["position"]) for item in field["for_each"]] == [
        ("P1", "D1"),
        ("P2", "D2"),
    ]
    assert field["default"] == {"P1": {"id": "case"}}  # "?" proposes nothing
    assert field["items_from"]["file"] == "data/layout.csv"  # kept to re-read it
    assert pending.value["snapshot"]["files"] == [
        {
            "path": "data/layout.csv",
            "sha256": hashlib.sha256(PLATE_LAYOUT.encode()).hexdigest(),
        }
    ]


def test_items_from_with_a_missing_column_is_invalid_context(tmp_path):
    tool, context = _plate_from_table(tmp_path, position="plate_well")
    graph = _graph(("call_1", "Group?", context), tool=tool)
    assert _run(graph, {"messages": []}) == []  # never pauses
    result = _tool_results(graph)["call_1"]
    assert result["action"] == "invalid_context"
    assert result["problems"][0]["path"] == "fields.0.items_from"
    assert "no column 'plate_well'" in result["problems"][0]["message"]


def test_the_supervisor_is_told_when_to_use_the_plate():
    from lobster.tools.user_interaction import create_dashboard_schema_tool

    text = create_ask_user_tool().description
    for said in (
        'Use view "plate_map" to put samples into groups, one group',
        "Ask over ALL the samples, not",
        "prefer items_from",
        "items that aren't samples (clusters, features,",
        "more than one pick per item",
    ):
        assert said in text, said
    schema = create_dashboard_schema_tool().invoke({})
    assert "Ask over all the samples" in schema and "items_from" in schema


def test_a_changed_table_is_read_again_on_the_reask(tmp_path):
    """Swapped wells, a vanished and a new sample: the re-ask shows the table
    as it is now, keeps the answers for samples still there, and the old
    layout is no longer accepted."""
    tool, context = _plate_from_table(tmp_path)
    graph, pending = _ask(context, question="Group of each sample?", tool=tool)
    (tmp_path / "data" / "layout.csv").write_text(
        "sample,well,condition\nP2,D1,?\nP1,D2,case\nP3,D3,control\n"
    )
    answer = {"groups": {"P1": {"id": "case"}, "P2": {"id": "control"}}}
    again = _answer_pending(graph, pending, answer)
    field = again.value["fields"][0]
    assert [(i["id"], i["position"]) for i in field["for_each"]] == [
        ("P2", "D1"),
        ("P1", "D2"),
        ("P3", "D3"),
    ]
    assert field["default"] == answer["groups"]  # kept; P3 left unresolved
    messages = [p["message"] for p in again.value["problems"]]
    assert "new in data/layout.csv, not answered yet: P3" in messages
    assert any("changed" in m for m in messages)

    (tmp_path / "data" / "layout.csv").write_text(
        "sample,well,condition\nP2,D1,?\nP3,D3,control\n"
    )
    third = _answer_pending(
        graph, again, {"groups": {**answer["groups"], "P3": {"id": "case"}}}
    )
    messages = [p["message"] for p in third.value["problems"]]
    assert "no longer in data/layout.csv: P1" in messages
    assert [i["id"] for i in third.value["fields"][0]["for_each"]] == ["P2", "P3"]

    final = {"groups": {"P2": {"id": "control"}, "P3": {"id": "case"}}}
    assert _answer_pending(graph, third, final) is None
    assert _tool_results(graph)["call_1"]["answer"] == final


def test_a_table_that_breaks_keeps_the_answer_stale(tmp_path):
    tool, context = _plate_from_table(tmp_path)
    graph, pending = _ask(context, question="Group?", tool=tool)
    (tmp_path / "data" / "layout.csv").write_text("sample,condition\nP1,case\n")
    answer = {"groups": {"P1": {"id": "case"}, "P2": {"id": "control"}}}
    again = _answer_pending(graph, pending, answer)
    messages = [p["message"] for p in again.value["problems"]]
    assert any(m.startswith("couldn't read data/layout.csv again: ") for m in messages)
    # Still the old layout, and an answer to it is not accepted.
    assert _answer_pending(graph, again, answer) is not None


def test_context_sent_as_json_text_is_read():
    graph = _graph(("call_1", "Which count source?", json.dumps(SOURCE)))
    (pending,) = _run(graph, {"messages": []})
    assert pending.value["fields"][0]["key"] == "source"


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", '"fields"'])
def test_context_text_that_is_not_a_json_object_is_refused(text):
    graph = _graph(("call_1", "Which count source?", text))
    assert _run(graph, {"messages": []}) == []
    result = _tool_results(graph)["call_1"]
    assert result["action"] == "invalid_context"
    assert result["problems"][0]["path"] == "context"


def _per_item(default=None, checks=None, options=None):
    field = {
        "key": "groups",
        "type": "choice",
        "for_each": ["S07", "S13"],
        "options": options
        or [
            "vehicle",
            "treated",
            {"id": "leave_out", "fields": [{"key": "reason", "type": "text"}]},
        ],
    }
    if default:
        field["default"] = default
    return {"fields": [field], "checks": checks or []}


def _settle(context, answer):
    graph, pending = _ask(context, question="Group?")
    _answer_pending(graph, pending, answer)
    return _tool_results(graph)["call_1"]


ANSWERED = {"groups": {"S07": {"id": "vehicle"}, "S13": {"id": "vehicle"}}}


def test_a_per_item_question_with_nothing_pre_filled_gets_the_note():
    result = _settle(_per_item(), ANSWERED)
    assert result["note"] == (
        "When samples are being put into groups, ask over all of them; never "
        "pre-fill exclusions or QC decisions."
    )


def test_a_qc_review_gets_no_note():
    """v4 1.3: a keep/exclude review is not a grouping; the note made the
    supervisor pre-fill exclusions next time."""
    qc = [
        "retain",
        "defer",
        {"id": "exclude", "fields": [{"key": "reason", "type": "text"}]},
    ]
    answered = {"groups": {"S07": {"id": "retain"}, "S13": {"id": "retain"}}}
    assert "note" not in _settle(_per_item(options=qc), answered)


def test_no_note_when_items_are_pre_filled_or_left_open_on_purpose():
    assert "note" not in _settle(_per_item(default={"S07": "vehicle"}), ANSWERED)
    check = [{"check": "no_default_for", "field": "groups", "items": ["S07", "S13"]}]
    assert "note" not in _settle(_per_item(checks=check), ANSWERED)


# --- the data folder is watched when data_files is left out ---


def _folder_workspace(tmp_path):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "layout.csv").write_text("sample,well\nS1,A1\n")
    (tmp_path / "data" / "counts.csv").write_text("gene,S1\nG1,5\n")
    return tmp_path


def test_without_data_files_the_data_folder_is_watched(tmp_path):
    workspace = _folder_workspace(tmp_path)
    tool = create_ask_user_tool(workspace_path=workspace)
    graph, pending = _ask(SOURCE, tool=tool)
    paths = [entry["path"] for entry in pending.value["snapshot"]["files"]]
    assert paths == ["data/counts.csv", "data/layout.csv"]

    (workspace / "data" / "layout.csv").write_text("sample,well\nS1,D1\n")
    (workspace / "data" / "new.csv").write_text("added during the question\n")
    again = _answer_pending(graph, pending, {"source": {"id": "SRC-A"}})
    assert [p["message"] for p in again.value["problems"]] == [
        "the data changed while you were reviewing it: data/layout.csv"
    ]  # the new file is not a change
    assert _answer_pending(graph, again, {"source": {"id": "SRC-A"}}) is None
    assert _tool_results(graph)["call_1"]["action"] == "submit"


def test_named_data_files_are_the_only_ones_watched(tmp_path):
    workspace = _folder_workspace(tmp_path)
    tool = create_ask_user_tool(workspace_path=workspace)
    _, pending = _ask({**SOURCE, "data_files": ["data/counts.csv"]}, tool=tool)
    assert [e["path"] for e in pending.value["snapshot"]["files"]] == [
        "data/counts.csv"
    ]


def test_without_a_data_folder_nothing_is_watched(tmp_path):
    tool = create_ask_user_tool(workspace_path=tmp_path)
    _, pending = _ask(SOURCE, tool=tool)
    assert pending.value.get("snapshot") is None


def test_the_result_says_why_the_question_came_back(tmp_path):
    """v4 2.3: the supervisor can explain a re-ask."""
    workspace = _folder_workspace(tmp_path)
    tool = create_ask_user_tool(workspace_path=workspace)
    graph, pending = _ask(SOURCE, tool=tool)
    (workspace / "data" / "layout.csv").write_text("sample,well\nS1,D1\n")
    again = _answer_pending(graph, pending, {"source": {"id": "SRC-A"}})
    _answer_pending(graph, again, {"source": {"id": "SRC-A"}})
    result = _tool_results(graph)["call_1"]
    assert result["reasked_because"] == [
        "the data changed while you were reviewing it: data/layout.csv"
    ]


def test_an_answer_at_the_first_attempt_has_no_reasked_because():
    graph, pending = _ask(SOURCE)
    _answer_pending(graph, pending, {"source": {"id": "SRC-A"}})
    assert "reasked_because" not in _tool_results(graph)["call_1"]


def test_the_supervisor_is_told_to_pre_fill_earlier_answers_when_it_asks_again():
    """v4 2.5: it re-asked showing an item the user had dropped as undecided."""
    assert (
        "When you ask again yourself, put the user's earlier"
        in create_ask_user_tool().description
    )


def test_docstring_states_the_removes_mark():
    """The removal rule is stated, not hidden in spelling (review: word rules)."""
    description = " ".join(create_ask_user_tool().description.split())
    assert '"removes": true and carries a required reason field' in description
    assert '"removes": false on an option that only sounds like one' in description
    assert "overrides something" not in description


def test_a_marked_option_decides_whether_a_field_is_a_grouping():
    from lobster.services.interaction.component_schemas import ChoiceField
    from lobster.tools.user_interaction import _is_group_field

    def field(*options):
        return ChoiceField.model_validate(
            {"key": "g", "type": "choice", "for_each": ["S01"], "options": options}
        )

    # Two groups, one of them named like a removal but marked as not one.
    assert _is_group_field(field("control", {"id": "dropout_cells", "removes": False}))
    # A group named innocently but marked as a removal is not a group.
    assert not _is_group_field(field("control", {"id": "batch_2", "removes": True}))
