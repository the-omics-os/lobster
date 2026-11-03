"""Tests for the ask_user and get_dashboard_schema HITL tools.

Most questions run through a real checkpointed LangGraph (no LLM): the graph
pauses in ask_user, the test answers as the interface would, and the graph
resumes. The factory tests patch interrupt() and call the tool directly.
"""

import json
from unittest.mock import MagicMock, patch

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


def _graph(*calls):
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
    builder.add_node("tools", ToolNode([create_ask_user_tool()]))
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


def test_answer_that_breaks_the_question_is_rejected():
    _, result = _ask_and_answer(
        SOURCE, lambda r: _reply(r, {"source": {"id": "SRC-C"}})
    )
    assert result["action"] == "rejected"
    assert result["problems"] == [
        "source: option 'SRC-C' is unavailable (Normalized; build unknown)"
    ]


def test_answer_to_an_old_question_version_is_stale():
    def old_version(request):
        reply = _reply(request, {"source": {"id": "SRC-A"}})
        reply["responds_to"]["request_version"] = 0 + 2
        return reply

    _, result = _ask_and_answer(SOURCE, old_version)
    assert result["action"] == "stale"


def test_old_style_reply_is_rejected_not_passed_through():
    _, result = _ask_and_answer(SOURCE, lambda r: {"selected": "SRC-A", "index": 0})
    assert result["action"] == "rejected"


def test_defer_resumes_with_no_answer():
    _, result = _ask_and_answer(SOURCE, lambda r: _reply(r, action="defer"))
    assert result == {
        "decision_id": "dec-call_1",
        "answer_id": result["answer_id"],
        "action": "defer",
    }


def test_invalid_context_never_pauses():
    bad = {"fields": [{"key": "cut", "type": "number", "view": "slider", "min": 0}]}
    graph = _graph(("call_1", "Cutoff?", bad))
    assert _run(graph, {"messages": []}) == []
    result = _tool_results(graph)["call_1"]
    assert result["action"] == "invalid_context"
    assert any("view 'slider' needs max" in p for p in result["problems"])


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
    assert any("a record can't be a section" in p for p in result["problems"])
