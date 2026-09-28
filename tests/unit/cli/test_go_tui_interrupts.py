"""Interrupt identity and parallel resume regressions for the Go launcher."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from lobster.cli_internal import go_tui_launcher
from lobster.core.client import AgentClient


class _Bridge:
    def __init__(self, events=()):
        self.events = list(events)
        self.renders = []

    def send(self, msg_type, payload=None, msg_id=""):
        if msg_type == "component_render":
            self.renders.append((msg_id, payload))

    def recv_event(self, timeout=None):
        return self.events.pop(0) if self.events else None


@pytest.mark.parametrize(
    "response_type", ["component_response", "confirm_response", "select_response"]
)
@pytest.mark.parametrize("bad_id", [None, "other-question"])
def test_interrupt_ignores_unmatched_response(response_type, bad_id):
    wrong = {
        "type": response_type,
        "payload": {"data": {"action": "cancel"}, "confirm": False, "value": "wrong"},
    }
    if bad_id is not None:
        wrong["id"] = bad_id
    bridge = _Bridge(
        [
            wrong,
            {
                "type": "component_response",
                "id": "question-1",
                "payload": {"id": "other-question", "data": {"answer": "wrong"}},
            },
            {
                "type": "component_response",
                "id": "question-1",
                "payload": {"id": "question-1", "data": {"answer": "right"}},
            },
        ]
    )
    assert go_tui_launcher._handle_interrupt(
        bridge, {"interrupt_id": "question-1", "data": {"component": "text_input"}}
    ) == {"answer": "right"}


def test_interrupt_requires_original_id():
    bridge = _Bridge()
    with pytest.raises(ValueError, match="interrupt ID"):
        go_tui_launcher._handle_interrupt(bridge, {"data": {}})
    assert bridge.renders == []


@pytest.mark.parametrize("reverse_display", [False, True])
def test_parallel_questions_resume_by_id(tmp_path, reverse_display):
    """Real parallel pauses survive one-at-a-time answers in either display order."""
    answers = {}

    def ask(question):
        answer = interrupt({"component": "text_input", "data": {"question": question}})
        answers[question] = answer
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("first", lambda state: ask("First?"))
    builder.add_node("second", lambda state: ask("Second?"))
    builder.add_node(
        "supervisor", lambda state: {"messages": [AIMessage(content="Finished")]}
    )
    builder.add_edge(START, "first")
    builder.add_edge(START, "second")
    builder.add_edge(["first", "second"], "supervisor")
    builder.add_edge("supervisor", END)
    outer = StateGraph(MessagesState)
    outer.add_node("supervisor", builder.compile())
    outer.add_edge(START, "supervisor")
    outer.add_edge("supervisor", END)
    graph = outer.compile(checkpointer=InMemorySaver())
    data_manager = Mock(profile_timings_enabled=False)
    data_manager.has_data.return_value = False
    with patch(
        "lobster.core.client.create_bioinformatics_graph", return_value=(graph, Mock())
    ):
        client = AgentClient(data_manager=data_manager, workspace_path=tmp_path)
    client._save_session_json = Mock()
    graph.stream = Mock(wraps=graph.stream)
    original_query = client.query
    original_ids = {}

    def query(*args, **kwargs):
        events = list(original_query(*args, **kwargs))
        questions = [event for event in events if event["type"] == "interrupt"]
        assert len(questions) == 2
        original_ids.update(
            (event["interrupt_id"], event["data"]["data"]["question"])
            for event in questions
        )
        yield from (event for event in events if event["type"] != "interrupt")
        yield from reversed(questions) if reverse_display else questions

    client.query = query

    class AnswerBridge(_Bridge):
        def send(self, msg_type, payload=None, msg_id=""):
            super().send(msg_type, payload, msg_id)
            if msg_type == "component_render":
                assert original_ids[msg_id] == payload["data"]["question"]
                self.events.append(
                    {
                        "type": "component_response",
                        "id": msg_id,
                        "payload": {
                            "id": msg_id,
                            "data": {"answer": payload["data"]["question"]},
                        },
                    }
                )

    bridge = AnswerBridge()
    go_tui_launcher._handle_user_query(bridge, client, "hello")
    assert answers == {"First?": {"answer": "First?"}, "Second?": {"answer": "Second?"}}
    assert len(bridge.renders) == 2
    assert {msg_id for msg_id, _ in bridge.renders} == set(original_ids)
    commands = [call.args[0] for call in graph.stream.call_args_list]
    resumes = [command.resume for command in commands if isinstance(command, Command)]
    assert len(resumes) == 2
    assert all(len(response) == 1 for response in resumes)
    assert {key: value for response in resumes for key, value in response.items()} == {
        key: {"answer": question} for key, question in original_ids.items()
    }
    assert not graph.get_state({"configurable": {"thread_id": client.session_id}}).next
