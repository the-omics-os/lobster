"""What one session sends to the model is append-only across turns.

A provider prompt cache only reuses a request prefix that is byte-identical to an
earlier one. This builds the real graph (supervisor plus every installed expert)
behind a real ``AgentClient``, replaces each agent's model with a recording fake,
runs three turns in one session and checks, for every agent:

- its system message is identical on every request;
- its bound tools (name, description, argument schema) are identical on every request;
- each request starts with the previous request of the same conversation, unchanged;
- no message is sent twice, so nothing grows on every turn by repetition.

No network and no real model: the fake answers from a script.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict, deque
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

EXPERT = "transcriptomics_expert"
HANDOFF = f"handoff_to_{EXPERT}"


class RecordingModel(BaseChatModel):
    """Records each request (messages and bound tools) and replies from a script."""

    agent: str
    requests: Any  # shared list of (agent, messages, tools)
    script: Any  # shared dict: agent -> deque of replies
    tools: list = []

    @property
    def _llm_type(self) -> str:
        return "recording"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(
            update={"tools": [convert_to_openai_tool(t) for t in tools]}
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append((self.agent, list(messages), self.tools))
        queue = self.script[self.agent]
        reply = queue.popleft() if queue else AIMessage(f"{self.agent} done")
        return ChatResult(generations=[ChatGeneration(message=reply.model_copy())])


def _handoff(call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": HANDOFF,
                "args": {"task_description": "Your task is to summarise the data."},
                "id": call_id,
            }
        ],
    )


def _render(message: BaseMessage) -> str:
    """The parts of a message a provider sends, as stable JSON."""
    return json.dumps(
        {
            "type": message.type,
            "content": message.content,
            "tool_calls": getattr(message, "tool_calls", None),
            "tool_call_id": getattr(message, "tool_call_id", None),
        },
        sort_keys=True,
        default=str,
    )


def _split(messages):
    system = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    return [_render(m) for m in system], [_render(m) for m in rest]


def _route_create_llm(monkeypatch, create_llm):
    """Point every module's own `create_llm` name at one function."""
    from lobster.agents import graph as graph_module
    from lobster.config import llm_factory
    from lobster.core.component_registry import component_registry

    agents = component_registry.list_agents()
    if EXPERT not in agents:
        pytest.skip(f"{EXPERT} is not installed")
    # Import every factory first so each module's own `create_llm` name exists.
    for config in agents.values():
        module_name = config.factory_function.rsplit(".", 1)[0]
        importlib.import_module(module_name)

    def is_create_llm(value):
        # A module imported while an earlier test mocked the factory can still
        # hold that test's mock, so any mock under this name is replaced too.
        return isinstance(value, Mock) or (
            getattr(value, "__module__", None) == llm_factory.__name__
        )

    for name, module in tuple(sys.modules.items()):
        if name.startswith("lobster.") and isinstance(module, ModuleType):
            if is_create_llm(getattr(module, "create_llm", None)):
                monkeypatch.setattr(module, "create_llm", create_llm)
    monkeypatch.setattr(graph_module, "create_llm", create_llm)


@pytest.fixture
def session(tmp_path):
    from lobster.core.client import AgentClient

    requests: list = []
    script: dict = defaultdict(deque)

    def create_llm(agent_name, *args, **kwargs):
        return RecordingModel(agent=agent_name, requests=requests, script=script)

    # Undo these patches before the provider mock is undone, so its own cleanup
    # finds and restores every alias instead of a later undo bringing back a
    # mock that outlives its test.
    with pytest.MonkeyPatch.context() as patches:
        _route_create_llm(patches, create_llm)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        client = AgentClient(
            workspace_path=workspace, session_id="append-only", interactive=False
        )
        client._save_session_json = lambda: None
        assert EXPERT in client.graph_metadata.supervisor_accessible_agents
        yield client, requests, script


def test_three_turns_send_an_append_only_prompt_to_every_agent(session):
    client, requests, script = session
    supervisor_name = "supervisor"

    # Turn 1 hands off to the expert, turn 2 answers directly, turn 3 hands off again.
    script["supervisor"].extend(
        [
            _handoff("call-1"),
            AIMessage("answer one"),
            AIMessage("answer two"),
            _handoff("call-3"),
            AIMessage("answer three"),
        ]
    )
    script[EXPERT].extend(
        [AIMessage("expert result one"), AIMessage("expert result two")]
    )

    for turn, text in enumerate(
        ["first question", "second question", "third question"]
    ):
        result = client.query(text)
        assert result.get("success") is True, (turn, result)

    by_agent: dict = defaultdict(list)
    for agent, messages, tools in requests:
        by_agent[agent].append((messages, tools))
    assert len(by_agent[supervisor_name]) == 5
    assert len(by_agent[EXPERT]) == 2

    for agent, calls in by_agent.items():
        first_system, _ = _split(calls[0][0])
        first_tools = json.dumps(calls[0][1], sort_keys=True, default=str)
        assert first_system and first_tools != "[]", agent
        for index, (messages, tools) in enumerate(calls[1:], start=1):
            system, _ = _split(messages)
            # (a) the system prompt never changes within the session
            assert system == first_system, (
                f"{agent} request {index}: system prompt changed\n"
                f"before: {first_system[0][-400:]}\nafter:  {system[0][-400:] if system else None}"
            )
            # (b) the tool list never changes within the session
            assert (
                json.dumps(tools, sort_keys=True, default=str) == first_tools
            ), f"{agent} request {index}: bound tools changed"

    # (c) every supervisor request extends the previous one; (d) nothing is sent twice
    supervisor = [_split(messages)[1] for messages, _ in by_agent[supervisor_name]]
    for index in range(1, len(supervisor)):
        before, after = supervisor[index - 1], supervisor[index]
        assert after[: len(before)] == before, (
            f"supervisor request {index} edits or drops earlier messages:\n"
            + "\n".join(
                f"  [{i}] {b!s:.160} -> {a!s:.160}"
                for i, (b, a) in enumerate(zip(before, after))
                if a != b
            )
        )
    last = supervisor[-1]
    repeated = {m: last.count(m) for m in last if last.count(m) > 1}
    assert not repeated, f"messages sent more than once: {repeated}"
    # (e) the third turn still carries the first question and its answer
    contents = [json.loads(m)["content"] for m in last]
    assert "first question" in contents and "answer one" in contents, contents

    # Each hand-off starts a fresh expert conversation; within it the same rules hold.
    for messages, _ in by_agent[EXPERT]:
        _, rest = _split(messages)
        assert len(rest) == 1 and "summarise the data" in rest[0]


def _supervisor_requests(requests):
    return [
        _split(messages)[1] for agent, messages, _ in requests if agent == "supervisor"
    ]


def _system_prompts(requests):
    by_agent: dict = defaultdict(list)
    for agent, messages, _ in requests:
        by_agent[agent].append(_split(messages)[0])
    return by_agent


def _assert_supervisor_append_only(requests):
    """Each supervisor request (system prompt included) extends the previous one."""
    calls = [
        [_render(m) for m in messages]
        for agent, messages, _ in requests
        if agent == "supervisor"
    ]
    for index in range(1, len(calls)):
        before, after = calls[index - 1], calls[index]
        assert after[: len(before)] == before, (
            f"supervisor request {index} edits or drops earlier messages:\n"
            + "\n".join(
                f"  [{i}] {b!s:.200} -> {a!s:.200}"
                for i, (b, a) in enumerate(zip(before, after))
                if a != b
            )
        )
    last = calls[-1]
    repeated = {m: last.count(m) for m in last if last.count(m) > 1}
    assert not repeated, f"messages sent more than once: {repeated}"
    return [json.loads(m)["content"] for m in last if json.loads(m)["type"] != "system"]


@pytest.mark.no_auto_config
def test_graph_rebuilt_for_each_run_keeps_an_append_only_prompt(tmp_path, monkeypatch):
    """A host that binds a fresh inference run before each turn rebuilds the graph.

    Every model comes from the host's factory, through the real model factory.
    """
    from lobster.config import llm_factory
    from lobster.config.inference_context import (
        INFERENCE_BINDING_VERSION,
        InferenceContext,
    )
    from lobster.core.client import AgentClient

    _route_create_llm(monkeypatch, llm_factory.create_llm)
    requests: list = []
    script: dict = defaultdict(deque)
    components: list = []

    class Factory:
        capability_version = INFERENCE_BINDING_VERSION

        def create(self, *, component, context):
            components.append(component)
            return RecordingModel(
                agent=component, requests=requests, script=script, cache=False
            )

    context = InferenceContext(
        "bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "synthetic-binding",
        "version-a",
        200_000,
        4096,
    )
    workspace = tmp_path / "bound-workspace"
    workspace.mkdir()
    client = AgentClient(
        workspace_path=workspace,
        session_id="append-only-bound",
        interactive=False,
        inference_context=context,
        model_factory=Factory(),
    )
    client._save_session_json = lambda: None
    checkpointer = client.checkpointer

    script["supervisor"].extend(
        [
            _handoff("call-1"),
            AIMessage("answer one"),
            AIMessage("answer two"),
            _handoff("call-3"),
            AIMessage("answer three"),
        ]
    )
    script[EXPERT].extend(
        [AIMessage("expert result one"), AIMessage("expert result two")]
    )
    graphs = []
    for text in ["first question", "second question", "third question"]:
        client.bind_inference_run(inference_context=context, model_factory=Factory())
        graphs.append(client.graph)
        assert client.query(text).get("success") is True, text
    assert len({id(graph) for graph in graphs}) == 3
    assert client.checkpointer is checkpointer
    assert {"supervisor", EXPERT} <= set(components)

    assert sum(agent == "supervisor" for agent, _, _ in requests) == 5
    for agent, prompts in _system_prompts(requests).items():
        assert all(p == prompts[0] for p in prompts), f"{agent} system prompt changed"
    contents = _assert_supervisor_append_only(requests)
    assert contents[0] == "first question" and "answer one" in contents, contents
    for agent, messages, _ in requests:
        if agent == EXPERT:
            _, rest = _split(messages)
            assert len(rest) == 1 and "summarise the data" in rest[0]


def test_duplicate_left_by_an_earlier_version_does_not_grow(session):
    """A thread that already holds a duplicated answer gains no new duplicates."""
    from langchain_core.messages import HumanMessage

    client, requests, script = session
    script["supervisor"].extend(
        [
            AIMessage("answer one"),
            AIMessage("answer two"),
            AIMessage("answer three"),
            AIMessage("answer four"),
        ]
    )
    assert client.query("first question").get("success") is True
    # The earlier version sent the whole local history on the same thread: the
    # id-less local copy of "answer one" was appended to the checkpoint again.
    client.messages.append(HumanMessage(content="second question"))
    config = {"configurable": {"thread_id": client.session_id}}
    assert client._run_query({"messages": client.messages}, config)["success"]
    seeded = [m.content for m in client.graph.get_state(config).values["messages"]]
    assert seeded.count("answer one") == 2, seeded

    for text in ["third question", "fourth question"]:
        assert client.query(text).get("success") is True

    supervisor = _supervisor_requests(requests)
    for index in range(2, len(supervisor)):
        assert supervisor[index][: len(supervisor[index - 1])] == supervisor[index - 1]
    last = [json.loads(m)["content"] for m in supervisor[-1]]
    assert last.count("answer one") == 2, last
    assert {c: last.count(c) for c in last if c != "answer one"} == {
        c: 1 for c in last if c != "answer one"
    }, last
    assert last[-1] == "fourth question"


def test_data_loaded_between_turns_keeps_every_system_prompt(session):
    import anndata
    import numpy as np

    client, requests, script = session
    script["supervisor"].extend(
        [
            _handoff("call-1"),
            AIMessage("answer one"),
            _handoff("call-2"),
            AIMessage("answer two"),
        ]
    )
    script[EXPERT].extend(
        [AIMessage("expert result one"), AIMessage("expert result two")]
    )

    assert client.query("first question").get("success") is True
    client.data_manager.store_modality(
        "tiny_counts", anndata.AnnData(np.ones((3, 2), dtype=np.float32))
    )
    assert "tiny_counts" in client.data_manager.list_modalities()
    assert client.query("second question").get("success") is True

    prompts = _system_prompts(requests)
    assert len(prompts["supervisor"]) == 4 and len(prompts[EXPERT]) == 2
    for agent, rendered in prompts.items():
        for index, prompt in enumerate(rendered[1:], start=1):
            assert (
                prompt == rendered[0]
            ), f"{agent} request {index}: system prompt changed after a data load"
    _assert_supervisor_append_only(requests)


def test_streamed_turns_send_each_earlier_message_once(session):
    client, requests, script = session
    script["supervisor"].extend([AIMessage("answer one"), AIMessage("answer two")])

    for text in ["first question", "second question"]:
        events = list(client.query(text, stream=True))
        assert events and events[-1]["type"] == "complete", events[-1:]

    last = _supervisor_requests(requests)[-1]
    assert [json.loads(m)["content"] for m in last] == [
        "first question",
        "answer one",
        "second question",
    ]


def test_restored_session_sends_its_history_once(session, tmp_path):
    client, requests, script = session
    saved = tmp_path / "saved_session.json"
    saved.write_text(
        json.dumps(
            {
                "session_id": "restored",
                "conversation": [
                    {"role": "user", "content": "old question"},
                    {"role": "assistant", "content": "old answer"},
                ],
            }
        )
    )
    client.load_session(saved)
    script["supervisor"].extend([AIMessage("answer one"), AIMessage("answer two")])

    for text in ["first question", "second question"]:
        assert client.query(text).get("success") is True

    first, second = _supervisor_requests(requests)
    assert [json.loads(m)["content"] for m in first] == [
        "old question",
        "old answer",
        "first question",
    ]
    assert [json.loads(m)["content"] for m in second] == [
        "old question",
        "old answer",
        "first question",
        "answer one",
        "second question",
    ]


def _contents(rendered):
    return [json.loads(m)["content"] for m in rendered]


def _restore(client, tmp_path):
    saved = tmp_path / "saved_session.json"
    saved.write_text(
        json.dumps(
            {
                "session_id": "restored",
                "conversation": [
                    {"role": "user", "content": "old question"},
                    {"role": "assistant", "content": "old answer"},
                ],
            }
        )
    )
    client.load_session(saved)


def test_command_after_a_restore_keeps_the_restored_history(session, tmp_path):
    from lobster.cli_internal.commands.heavy.session_infra import (
        _add_command_to_history,
    )

    client, requests, script = session
    _restore(client, tmp_path)
    assert _add_command_to_history(client, "/status", "status ok")
    script["supervisor"].extend([AIMessage("answer one"), AIMessage("answer two")])

    for text in ["first question", "second question"]:
        assert client.query(text).get("success") is True

    first, second = _supervisor_requests(requests)
    assert _contents(first) == [
        "old question",
        "old answer",
        "Command: /status",
        "Command Result: status ok",
        "first question",
    ]
    assert second == first + second[len(first) :]
    assert _contents(second[len(first) :]) == ["answer one", "second question"]


def test_unreadable_thread_sends_the_whole_history_once(session, tmp_path, caplog):
    import logging

    client, requests, script = session
    _restore(client, tmp_path)

    def unreadable(*args, **kwargs):
        raise RuntimeError("state store unavailable")

    client.graph.get_state = unreadable
    script["supervisor"].append(AIMessage("answer one"))
    with caplog.at_level(logging.WARNING, logger="lobster.core.client"):
        assert client.query("first question").get("success") is True

    (first,) = _supervisor_requests(requests)
    assert _contents(first) == ["old question", "old answer", "first question"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("thread" in r.getMessage() for r in warnings), caplog.text


def _two_turns_with_handoff(client, script):
    """Turn 1 hands off to the expert, so the thread holds a tool call and its
    result that the client's own list does not."""
    script["supervisor"].extend(
        [_handoff("call-1"), AIMessage("answer one"), _handoff("call-2")]
    )
    script[EXPERT].extend(
        [AIMessage("expert result one"), AIMessage("expert result two")]
    )
    assert client.query("first question").get("success") is True


def _assert_turn_three_follows_turn_one(requests, script, client):
    script["supervisor"].append(AIMessage("answer three"))
    assert client.query("third question").get("success") is True
    supervisor = _supervisor_requests(requests)
    turn_one_last, turn_three = supervisor[1], supervisor[-1]
    assert turn_three[: len(turn_one_last)] == turn_one_last
    assert _contents(turn_three[len(turn_one_last) :]) == [
        "answer one",
        "third question",
    ]
    assert len(set(turn_three)) == len(turn_three), turn_three


def test_cancelled_streamed_turn_is_rolled_back_whole(session):
    import threading

    client, requests, script = session
    _two_turns_with_handoff(client, script)

    # The client checks the event at the first streamed event, after the
    # supervisor's work in this turn has run but before it is checkpointed.
    cancel = threading.Event()
    cancel.set()
    events = list(client.query("second question", stream=True, cancel_event=cancel))
    assert sum(agent == EXPERT for agent, _, _ in requests) == 2
    assert not any(e["type"] == "complete" for e in events), events
    assert [m.content for m in client.messages] == ["first question", "answer one"]

    _assert_turn_three_follows_turn_one(requests, script, client)


def test_turn_forgotten_after_a_cancel_is_not_sent_again(session):
    from lobster.cli_internal.mcp_stdio import (
        _forget_cancelled_turn,
        _Stop,
        _thread_messages,
    )

    client, requests, script = session
    _two_turns_with_handoff(client, script)
    stop = _Stop(len(client.messages), _thread_messages(client))
    script["supervisor"].append(AIMessage("answer two"))
    assert client.query("second question").get("success") is True
    _forget_cancelled_turn(client, stop)

    _assert_turn_three_follows_turn_one(requests, script, client)
    last = _contents(_supervisor_requests(requests)[-1])
    assert "second question" not in last and "answer two" not in last


def test_reset_forgets_the_conversation_for_the_model(session):
    from lobster.cli_internal.commands.heavy.session_infra import (
        _add_command_to_history,
    )

    client, requests, script = session
    script["supervisor"].extend([AIMessage("answer one"), AIMessage("answer two")])
    assert client.query("first question").get("success") is True
    client.reset()
    assert _add_command_to_history(client, "/reset", "Conversation reset")
    assert client.query("second question").get("success") is True

    assert _contents(_supervisor_requests(requests)[-1]) == [
        "Command: /reset",
        "Command Result: Conversation reset",
        "second question",
    ]


@pytest.fixture
def new_process(tmp_path):
    """Makes a fresh ``AgentClient`` on one workspace, as each ``lobster query``
    process does; all of them record into one request list."""
    from lobster.core.client import AgentClient

    requests: list = []
    script: dict = defaultdict(deque)

    def create_llm(agent_name, *args, **kwargs):
        return RecordingModel(agent=agent_name, requests=requests, script=script)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.MonkeyPatch.context() as patches:
        _route_create_llm(patches, create_llm)

        def make():
            return AgentClient(
                workspace_path=workspace, session_id="per-turn", interactive=False
            )

        yield make, workspace, requests, script


def _turn_with_a_tool_call_then_a_new_process(new_process):
    make, workspace, requests, script = new_process
    script["supervisor"].extend([_handoff("call-1"), AIMessage("answer one")])
    script[EXPERT].append(AIMessage("expert result one"))
    first = make()
    assert first.query("first question").get("success") is True
    saved = workspace / "session_per-turn.json"
    assert saved.exists()

    second = make()
    second.load_session(saved)
    script["supervisor"].append(AIMessage("answer two"))
    assert second.query("second question").get("success") is True
    return saved, _supervisor_requests(requests)


def test_continued_session_extends_the_previous_turn_with_tool_history(
    new_process, monkeypatch
):
    """With the opt-in, a turn run in a new process sends the earlier turn as the
    model received it, tool call and tool result included, so the provider's
    prompt cache can reuse it."""
    monkeypatch.setenv("LOBSTER_SESSION_TOOL_HISTORY", "1")
    saved, supervisor = _turn_with_a_tool_call_then_a_new_process(new_process)

    turn_one_last, turn_two = supervisor[1], supervisor[2]
    assert turn_two[: len(turn_one_last)] == turn_one_last, (
        "the continued turn rewrites the earlier one:\n"
        f"before: {turn_one_last}\nafter:  {turn_two}"
    )
    assert _contents(turn_two[len(turn_one_last) :]) == [
        "answer one",
        "second question",
    ]
    assert "messages" in json.loads(saved.read_text())


def test_without_the_opt_in_the_session_file_keeps_no_tool_results(
    new_process, monkeypatch
):
    """Tool results can hold data values, so they are saved only on request."""
    monkeypatch.delenv("LOBSTER_SESSION_TOOL_HISTORY", raising=False)
    saved, supervisor = _turn_with_a_tool_call_then_a_new_process(new_process)

    data = json.loads(saved.read_text())
    assert "messages" not in data
    assert "expert result one" not in saved.read_text()
    assert _contents(supervisor[2]) == [
        "first question",
        "answer one",
        "second question",
    ]


def test_unreadable_tool_history_falls_back_to_the_conversation(
    new_process, monkeypatch, caplog
):
    import logging

    make, workspace, requests, script = new_process
    saved = workspace / "saved.json"
    saved.write_text(
        json.dumps(
            {
                "session_id": "restored",
                "conversation": [
                    {"role": "user", "content": "old question"},
                    {"role": "assistant", "content": "old answer"},
                ],
                "messages": [{"type": "not-a-message"}],
            }
        )
    )
    client = make()
    with caplog.at_level(logging.WARNING, logger="lobster.core.client"):
        client.load_session(saved)
    script["supervisor"].append(AIMessage("answer one"))
    assert client.query("first question").get("success") is True

    (first,) = _supervisor_requests(requests)
    assert _contents(first) == ["old question", "old answer", "first question"]
    assert any(
        "tool history" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    ), caplog.text


def test_tool_history_json_cannot_hold_is_left_out_whole(new_process, monkeypatch):
    """Bytes (a redacted reasoning block) would come back as other text, so the
    tool history is left out and the next process uses the conversation."""
    import stat

    from langchain_core.messages import HumanMessage

    make, workspace, _requests, _script = new_process
    monkeypatch.setenv("LOBSTER_SESSION_TOOL_HISTORY", "1")
    client = make()
    client._replace_thread_messages(
        [
            HumanMessage("q", id="h1"),
            AIMessage("a", id="a1", additional_kwargs={"blob": b"\x00\xff"}),
        ]
    )
    client.messages = [HumanMessage("q"), AIMessage("a")]

    saved = client._save_session_json()
    data = json.loads(saved.read_text())
    assert "messages" not in data
    assert [m["content"] for m in data["conversation"]] == ["q", "a"]

    client._replace_thread_messages(
        [HumanMessage("q", id="h1"), AIMessage("a", id="a1")]
    )
    saved = client._save_session_json()
    assert "messages" in json.loads(saved.read_text())
    if sys.platform != "win32":
        assert stat.S_IMODE(saved.stat().st_mode) == 0o600
