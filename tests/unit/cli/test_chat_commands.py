from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console

from lobster.cli_internal.commands.heavy import chat_commands
from lobster.core.client import AgentClient


class _QueryClient:
    def __init__(self):
        self.answers = []

    def query(self, text, stream=False):
        question = {
            "interrupt_id": "question-1",
            "data": {"component": "text_input", "fallback_prompt": "Name?"},
        }
        if stream:
            return iter(
                [
                    {"type": "content_delta", "delta": "Before question"},
                    {"type": "interrupt", **question},
                ]
            )
        return {"success": False, "interrupts": [question]}

    def resume_from_interrupt(self, response, stream=True):
        self.answers.append(response)
        yield {"type": "complete", "success": True, "response": "Finished"}


@pytest.mark.parametrize("stream", [True, False])
def test_classic_chat_prompts_and_resumes(stream):
    client = _QueryClient()
    console = Console(record=True)
    with patch("builtins.input", return_value="Alice") as prompt:
        if stream:
            result = chat_commands._display_streaming_response(client, "hello", console)
        else:
            result = chat_commands._query_classic(client, "hello")

    prompt.assert_called_once_with("\nName?: ")
    assert client.answers == [{"question-1": {"answer": "Alice"}}]
    assert result["success"] is True
    assert result["response"] == "Finished"

    if stream:
        assert "Before question" in console.export_text()


@pytest.mark.parametrize("stream", [True, False])
def test_classic_chat_cancellation_does_not_resume(stream):
    client = _QueryClient()
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        if stream:
            result = chat_commands._display_streaming_response(
                client, "hello", Console()
            )
            assert result["error"] == "Interrupted by user"
        else:
            with pytest.raises(KeyboardInterrupt):
                chat_commands._query_classic(client, "hello")
    assert client.answers == []


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("questions", ["none", "one", "sequential", "parallel"])
def test_classic_chat_with_checkpointed_graph(tmp_path, stream, questions):
    """Exercise real client resume Commands and checkpoint IDs without an LLM."""
    answers = []

    def ask(question):
        answer = interrupt({"component": "text_input", "fallback_prompt": question})
        answers.append(answer)
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node(
        "supervisor", lambda state: {"messages": [AIMessage(content="Finished")]}
    )
    if questions == "none":
        builder.add_edge(START, "supervisor")
    else:
        builder.add_node("first", lambda state: ask("First?"))
        builder.add_edge(START, "first")
        if questions in ("sequential", "parallel"):
            builder.add_node("second", lambda state: ask("Second?"))
            if questions == "parallel":
                builder.add_edge(START, "second")
                builder.add_edge(["first", "second"], "supervisor")
            else:
                builder.add_edge("first", "second")
                builder.add_edge("second", "supervisor")
        else:
            builder.add_edge("first", "supervisor")
    builder.add_edge("supervisor", END)
    # Production also wraps the supervisor graph in an outer StateGraph.
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
    client._save_session_json = Mock(return_value=None)
    graph.stream = Mock(wraps=graph.stream)
    expected_responses = {}

    def answer_question(prompt):
        state = graph.get_state({"configurable": {"thread_id": client.session_id}})
        expected_responses.update(
            (intr.id, {"answer": intr.value["fallback_prompt"]})
            for task in state.tasks
            for intr in task.interrupts
        )
        return prompt.strip().removesuffix(":")

    with patch("builtins.input", side_effect=answer_question) as prompt:
        if stream:
            result = chat_commands._display_streaming_response(
                client, "hello", Console()
            )
        else:
            result = chat_commands._query_classic(client, "hello")

    count = {"none": 0, "one": 1, "sequential": 2, "parallel": 2}[questions]
    assert result["success"] is True, result
    assert result["response"] == "Finished"
    assert prompt.call_count == count
    assert (
        sorted(answer["answer"] for answer in answers) == ["First?", "Second?"][:count]
    )
    commands = [
        call.args[0] if call.args else call.kwargs["input"]
        for call in graph.stream.call_args_list
    ]
    resumes = [command.resume for command in commands if isinstance(command, Command)]
    assert {
        key: value for response in resumes for key, value in response.items()
    } == expected_responses
    if questions == "parallel":
        assert len(resumes) == 1
        assert len(resumes[0]) == 2
    assert not graph.get_state({"configurable": {"thread_id": client.session_id}}).next
