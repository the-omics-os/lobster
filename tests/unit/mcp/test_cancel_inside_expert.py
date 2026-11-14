"""A cancel stops a run at the next tool or model call, even inside an expert.

The client checks the run's cancel event between the supervisor's stream
events, and an expert's whole work arrives as one of those: before, a cancel
waited for the expert to finish. The server now raises at the next step
boundary through a callback, shows phase cancelling until the run ends, and
puts the conversation back as it was before the turn so the next can start.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from lobster.cli_internal import mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    Act,
    Hold,
    ScriptedClient,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)


def test_a_cancel_stops_an_expert_at_its_next_step():
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    from lobster.agents.graph import _create_agent_tool

    class Fake(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    cancel = threading.Event()
    ran = []

    def step(n):
        @tool(f"step_{n}")
        def run_step() -> str:
            """One analysis step."""
            ran.append(n)
            if n == 2:
                cancel.set()
            return f"step {n} done"

        return run_step

    def calls(name, n):
        return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": n}])

    expert = create_react_agent(
        Fake(responses=[calls(f"step_{n}", f"s{n}") for n in (1, 2, 3, 4)]),
        [step(n) for n in (1, 2, 3, 4)],
    )
    handoff = _create_agent_tool("expert", expert, "handoff_to_expert", "Delegate.")
    supervisor = create_react_agent(
        Fake(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "handoff_to_expert",
                            "args": {"task_description": "Your task is to work."},
                            "id": "h1",
                        }
                    ],
                ),
                AIMessage("done"),
            ]
        ),
        [handoff],
    )

    class Client:
        callbacks: list = []
        messages: list = []

    client = Client()
    with mcp_stdio._stop_on_cancel(client, cancel) as stop:
        assert len(client.callbacks) == 1
        with pytest.raises(mcp_stdio.RunCancelled):
            for _event in supervisor.stream(
                {"messages": [("user", "go")]},
                {"callbacks": client.callbacks},
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                pass
    assert ran == [1, 2]  # not the rest of the expert's steps
    assert stop.tripped and client.callbacks == []


def test_a_cancel_stops_a_run_whose_tool_waits_for_another():
    # One worker tool runs at a time: the second tool of a message waits for
    # the first. A cancel still stops the run before its next step.
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    from lobster.agents.graph import _create_agent_tool, _one_tool_at_a_time

    class Fake(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    cancel, holding, release = threading.Event(), threading.Event(), threading.Event()
    ran = []

    @tool("hold")
    def hold() -> str:
        """A long step."""
        holding.set()
        release.wait(10)
        ran.append("hold")
        return "held"

    @tool("queued")
    def queued() -> str:
        """A step that waits its turn."""
        ran.append("queued")
        return "queued"

    @tool("after")
    def after() -> str:
        """A later step."""
        ran.append("after")
        return "after"

    def calls(*names):
        return AIMessage(
            content="",
            tool_calls=[{"name": n, "args": {}, "id": n} for n in names],
        )

    expert = create_react_agent(
        Fake(responses=[calls("hold", "queued"), calls("after"), AIMessage("ok")]),
        [hold, queued, after],
    )
    _one_tool_at_a_time(expert)
    handoff = _create_agent_tool("expert", expert, "handoff_to_expert", "Delegate.")
    supervisor = create_react_agent(
        Fake(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "handoff_to_expert",
                            "args": {"task_description": "Your task is to work."},
                            "id": "h1",
                        }
                    ],
                ),
                AIMessage("done"),
            ]
        ),
        [handoff],
    )

    class Client:
        callbacks: list = []
        messages: list = []

    client, outcome = Client(), []

    def run():
        with mcp_stdio._stop_on_cancel(client, cancel):
            try:
                for _event in supervisor.stream(
                    {"messages": [("user", "go")]},
                    {"callbacks": client.callbacks},
                    stream_mode=["messages", "updates"],
                    subgraphs=True,
                ):
                    pass
            except mcp_stdio.RunCancelled:
                outcome.append("cancelled")

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert holding.wait(10)
    cancel.set()
    release.set()
    worker.join(30)
    assert not worker.is_alive(), "the run did not stop"
    assert outcome == ["cancelled"]
    # The waiting tool runs after the first, or is stopped before it starts.
    assert ran in (["hold"], ["hold", "queued"])


def test_a_turn_cancelled_in_a_real_graph_leaves_every_tool_call_answered():
    """One completed turn through a real graph with a checkpointer, then a
    turn cancelled inside the expert: the thread is as it was after turn 1,
    every tool call has its result, and a next turn runs."""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    from lobster.agents.graph import _create_agent_tool

    class Fake(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    cancel = threading.Event()

    @tool
    def step() -> str:
        """One analysis step."""
        if cancel_on_step.is_set():
            cancel.set()
        return "step done"

    cancel_on_step = threading.Event()

    def handoff(n):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "handoff_to_expert",
                    "args": {"task_description": "Your task is to work."},
                    "id": f"h{n}",
                }
            ],
        )

    def calls_step(n):
        return AIMessage(content="", tool_calls=[{"name": "step", "args": {}, "id": n}])

    expert = create_react_agent(
        Fake(responses=[calls_step("s1"), AIMessage("expert done"), calls_step("s2")]),
        [step],
    )
    graph = create_react_agent(
        Fake(
            responses=[
                handoff(1),
                AIMessage("Turn one answer."),
                handoff(2),
                AIMessage("Turn three answer."),
            ]
        ),
        [_create_agent_tool("expert", expert, "handoff_to_expert", "Delegate.")],
        checkpointer=InMemorySaver(),
    )

    class Client:
        session_id = "thread"

        def __init__(self):
            self.graph, self.callbacks, self.messages = graph, [], []
            self.saved = 0

        def _save_session_json(self):
            self.saved += 1

        def turn(self, text):
            self.messages.append(HumanMessage(text))
            config = {**mcp_stdio._thread_config(self), "callbacks": self.callbacks}
            for _event in self.graph.stream(
                {"messages": self.messages},
                config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                pass
            self.messages.append(AIMessage(self.thread()[-1].content))

        def thread(self):
            return mcp_stdio._thread_messages(self)

    client = Client()
    client.turn("first")
    after_first = client.thread()
    assert after_first[-1].content == "Turn one answer."
    assert len(client.messages) < len(after_first)  # the lists differ in length

    cancel_on_step.set()
    with mcp_stdio._stop_on_cancel(client, cancel) as stop:
        with pytest.raises(mcp_stdio.RunCancelled):
            client.turn("second")
    assert stop.tripped
    mcp_stdio._forget_cancelled_turn(client, stop)

    thread = client.thread()
    assert [m.id for m in thread] == [m.id for m in after_first]
    answered = {m.tool_call_id for m in thread if isinstance(m, ToolMessage)}
    for message in thread:
        for call in getattr(message, "tool_calls", None) or ():
            assert call["id"] in answered, call
    assert len(client.messages) == 2 and client.saved == 1
    client.turn("third")
    assert client.thread()[-1].content == "Turn three answer."


class Boundary:
    """Scripted step: the next tool call starts, as the real graph reports it
    to the client's callbacks."""


class Thread:
    """The graph thread: more messages than the client's list (tool calls and
    their results), and the updates made to it."""

    def __init__(self):
        self.messages = ["question", "hand-off call", "hand-off result", "answer"]
        self.updates = []

    def get_state(self, config):
        assert config == {"configurable": {"thread_id": "session_test"}}
        return SimpleNamespace(values={"messages": list(self.messages)})

    def update_state(self, config, values):
        self.updates.append(values["messages"])


class CallbackClient(ScriptedClient):
    """A client with the conversation and callbacks the real one has."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callbacks = []
        self.messages = ["earlier question", "earlier answer"]
        self.graph = Thread()
        self.saved = 0

    def _save_session_json(self):
        self.saved += 1

    def _play(self, events, cancel_event):
        self.messages.append("this question")
        self.graph.messages += ["this question", "a call without its result"]
        for event in events:
            if isinstance(event, Boundary):
                try:
                    for handler in self.callbacks:
                        handler.on_tool_start({"name": "next_step"}, "")
                except Exception as exc:  # the real client yields it as an error
                    yield {"type": "error", "error": str(exc), "error_type": "X"}
                    return
                continue
            yield from super()._play([event], cancel_event)


@pytest.fixture
def callback_engine(engine, monkeypatch):
    opened, script = engine
    from lobster.cli_internal.commands.heavy import session_infra

    made = session_infra.init_client_or_raise_startup_diagnostic

    def init(*args, **kwargs):
        plain = made(*args, **kwargs)
        client = CallbackClient(
            plain.data_manager.workspace, script["events"], loads_input=True
        )
        client.data_manager = plain.data_manager
        opened[-1]["client"] = client
        return client

    monkeypatch.setattr(session_infra, "init_client_or_raise_startup_diagnostic", init)
    return opened, script


def test_a_cancel_mid_step_ends_the_run_at_the_next_step(
    callback_engine, tmp_path, data
):
    opened, script = callback_engine
    hold = Hold("load_dataset")
    script["events"] = [hold, Boundary(), Act("cluster"), COMPLETE]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    assert hold.entered.wait(10)
    call(server, "lobster_cancel", {"run_id": run["run_id"]})
    stopping = call(server, "lobster_status", {"run_id": run["run_id"]})
    assert stopping["status"] == "running" and stopping["phase"] == "cancelling"
    hold.release.set()
    done = wait_terminal(server, run["run_id"])
    assert done["status"] == "cancelled", done
    client = opened[-1]["client"]
    tools = [a["type"] for a in client.data_manager.provenance.activities]
    assert "cluster" not in tools
    # The conversation is as before the turn, the whole thread included.
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    assert client.messages == ["earlier question", "earlier answer"]
    [update] = client.graph.updates
    assert update[0].id == REMOVE_ALL_MESSAGES
    assert update[1:] == ["question", "hand-off call", "hand-off result", "answer"]
    assert client.saved == 1
    assert client.callbacks == []


def test_a_turn_that_ends_on_its_own_keeps_its_conversation(
    callback_engine, tmp_path, data
):
    opened, script = callback_engine
    script["events"] = [Boundary(), COMPLETE]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    assert wait_terminal(server, run["run_id"])["status"] == "completed"
    client = opened[-1]["client"]
    assert client.graph.updates == [] and client.callbacks == []
