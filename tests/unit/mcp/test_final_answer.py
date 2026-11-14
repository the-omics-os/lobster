"""A turn whose model sent no final answer is not reported as a normal one."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lobster.cli_internal import mcp_record, mcp_stdio
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    ScriptedClient,
    data,
    engine,
    run_engine,
)

COMMENTARY = "Now running filter_and_normalize with the same thresholds."


class FakeGraph:
    def __init__(self, messages):
        self.messages = messages
        self.configs = []

    def get_state(self, config):
        self.configs.append(config)
        return SimpleNamespace(values={"messages": self.messages})


def session(messages):
    return SimpleNamespace(session_id="session_test", graph=FakeGraph(messages))


def tool_call():
    return AIMessage(
        content=COMMENTARY, tool_calls=[{"name": "t", "args": {}, "id": "1"}]
    )


# reply_of ----------------------------------------------------------------------


def test_the_last_message_is_the_answer_and_the_text_before_it_is_notes():
    client = session(
        [HumanMessage("go"), tool_call(), AIMessage("Cells: 2,700 to 2,612.")]
    )
    reply = mcp_record.reply_of(client, f"{COMMENTARY}\n\nCells: 2,700 to 2,612.")
    assert reply == mcp_record.Reply(
        "Cells: 2,700 to 2,612.", COMMENTARY, True, len(COMMENTARY)
    )
    assert client.graph.configs == [{"configurable": {"thread_id": "session_test"}}]


@pytest.mark.parametrize(
    "last",
    [
        AIMessage(""),  # an empty reply, as a provider can send
        tool_call(),  # a tool call that no result followed
        ToolMessage("done", tool_call_id="1"),
        HumanMessage("go"),
    ],
    ids=["empty", "tool_call", "tool_result", "human"],
)
def test_a_turn_that_does_not_end_on_text_has_no_final_answer(last):
    client = session([HumanMessage("go"), tool_call(), last])
    reply = mcp_record.reply_of(client, COMMENTARY)
    assert reply == mcp_record.Reply(
        mcp_record.NO_FINAL_ANSWER_SUMMARY, COMMENTARY, False, len(COMMENTARY)
    )


def test_text_blocks_of_the_last_message_are_its_answer():
    last = AIMessage([{"type": "text", "text": "Done. "}, {"type": "tool_use"}])
    reply = mcp_record.reply_of(session([last]), "Done.")
    assert (reply.text, reply.final) == ("Done.", True)


@pytest.mark.parametrize(
    "response, final", [("Done.", True), ("", False), ("  \n", False), (None, False)]
)
def test_without_a_readable_state_the_response_decides(response, final):
    reply = mcp_record.reply_of(SimpleNamespace(), response)
    assert reply.final is final
    assert reply.text == ("Done." if final else mcp_record.NO_FINAL_ANSWER_SUMMARY)


# a completed turn -----------------------------------------------------------


def test_commentary_is_not_the_summary(engine, monkeypatch, tmp_path, data):
    _opened, script = engine
    script["events"] = [{**COMPLETE, "response": COMMENTARY}]
    monkeypatch.setattr(
        ScriptedClient,
        "graph",
        FakeGraph([HumanMessage("go"), AIMessage("")]),
        raising=False,
    )
    status = run_engine(tmp_path, data)
    assert status["status"] == "completed"
    assert status["summary"] == mcp_record.NO_FINAL_ANSWER_SUMMARY
    assert COMMENTARY not in json.dumps(status)
    assert status["warnings"][0]["code"] == "no_final_answer"
    record = json.loads(Path(status["result_path"]).read_text())
    assert record["response"] == mcp_record.NO_FINAL_ANSWER_SUMMARY
    assert record["notes"] == COMMENTARY


def test_an_empty_response_is_a_turn_without_a_final_answer(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [{**COMPLETE, "response": ""}]
    status = run_engine(tmp_path, data)
    assert status["summary"] == mcp_record.NO_FINAL_ANSWER_SUMMARY
    assert "no_final_answer" in [w["code"] for w in status["warnings"]]
    assert "notes" not in json.loads(Path(status["result_path"]).read_text())


def test_a_turn_with_a_final_answer_has_no_warning(engine, monkeypatch, tmp_path, data):
    _opened, script = engine
    script["events"] = [{**COMPLETE, "response": f"{COMMENTARY}\n\nKept 2,612 cells."}]
    graph = FakeGraph([HumanMessage("go"), tool_call(), AIMessage("Kept 2,612 cells.")])
    monkeypatch.setattr(ScriptedClient, "graph", graph, raising=False)
    status = run_engine(tmp_path, data)
    assert status["summary"] == "Kept 2,612 cells."
    assert "no_final_answer" not in [w["code"] for w in status["warnings"]]
    record = json.loads(Path(status["result_path"]).read_text())
    assert record["notes"] == COMMENTARY
    assert "notes" not in status


def test_the_fixed_summary_keeps_the_forbidden_words_out():
    text = mcp_record.NO_FINAL_ANSWER_SUMMARY + mcp_record.NO_FINAL_ANSWER_WARNING
    for word in ("verified", "reproducible", "sandbox", "proves", "guarantee"):
        assert word not in text.lower()


def test_the_code_is_in_the_schema_enum():
    schema = json.loads(
        (
            Path(mcp_stdio.__file__).parents[1]
            / "schemas"
            / "lobster.result.v1.schema.json"
        ).read_text()
    )
    codes = schema["properties"]["warnings"]["items"]["properties"]["code"]["enum"]
    assert list(codes) == list(mcp_record.WARNING_CODES)
    assert "notes" in schema["properties"]
