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


@pytest.fixture
def session(tmp_path, monkeypatch):
    from lobster.agents import graph as graph_module
    from lobster.config import llm_factory
    from lobster.core.client import AgentClient
    from lobster.core.component_registry import component_registry

    agents = component_registry.list_agents()
    if EXPERT not in agents:
        pytest.skip(f"{EXPERT} is not installed")
    # Import every factory first so each module's own `create_llm` name exists.
    for config in agents.values():
        module_name = config.factory_function.rsplit(".", 1)[0]
        importlib.import_module(module_name)

    requests: list = []
    script: dict = defaultdict(deque)

    def create_llm(agent_name, *args, **kwargs):
        return RecordingModel(agent=agent_name, requests=requests, script=script)

    create_llm_names = {llm_factory.create_llm, graph_module.create_llm}
    for name, module in tuple(sys.modules.items()):
        if name.startswith("lobster.") and isinstance(module, ModuleType):
            if getattr(module, "create_llm", None) in create_llm_names:
                monkeypatch.setattr(module, "create_llm", create_llm)
    monkeypatch.setattr(graph_module, "create_llm", create_llm)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = AgentClient(
        workspace_path=workspace, session_id="append-only", interactive=False
    )
    client._save_session_json = lambda: None
    assert EXPERT in client.graph_metadata.supervisor_accessible_agents
    return client, requests, script


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

    # Each hand-off starts a fresh expert conversation; within it the same rules hold.
    for messages, _ in by_agent[EXPERT]:
        _, rest = _split(messages)
        assert len(rest) == 1 and "summarise the data" in rest[0]


def _supervisor_requests(requests):
    return [
        _split(messages)[1] for agent, messages, _ in requests if agent == "supervisor"
    ]


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
