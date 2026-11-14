"""``notes``: the model's commentary before its final message. Long commentary
keeps its end (where a report sent before a short closing line is), and the
full length is always stated."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from lobster.cli_internal import mcp_contract, mcp_record
from tests.unit.mcp.test_final_answer import FakeGraph, session
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    ScriptedClient,
    data,
    engine,
    run_engine,
)

POINTER = "The summary above reports every figure."
REPORT = "REPORT: cells 2700 -> 2612"


def chatter(lines: int) -> str:
    return "\n".join(f"Step {n}: running the next tool." for n in range(lines))


def reply_with_long_commentary(lines: int = 200):
    earlier = f"{chatter(lines)}\n{REPORT}"
    client = session([HumanMessage("go"), AIMessage(earlier), AIMessage(POINTER)])
    return earlier, mcp_record.reply_of(client, f"{earlier}\n\n{POINTER}")


def test_long_commentary_keeps_its_end_and_a_mark_that_the_start_is_gone():
    earlier, reply = reply_with_long_commentary()
    assert reply.final and reply.text == POINTER  # never replaced by earlier text
    assert len(earlier) > mcp_record.NOTES_MAX_CHARS
    assert REPORT in reply.notes
    assert reply.notes.startswith(mcp_record.NOTES_CUT)
    kept = reply.notes[len(mcp_record.NOTES_CUT) :]
    assert len(kept) <= mcp_record.NOTES_MAX_CHARS
    assert kept.startswith("Step ") and earlier.endswith(kept)  # cut at a line start
    assert reply.notes_chars == len(earlier)


def test_commentary_that_fits_is_kept_whole_with_its_length():
    reply = mcp_record.reply_of(
        session([HumanMessage("go"), AIMessage("Working."), AIMessage("Done.")]),
        "Working.\n\nDone.",
    )
    assert (reply.notes, reply.notes_chars) == ("Working.", len("Working."))


def test_one_very_long_line_is_still_cut_to_the_limit():
    earlier = "x" * 10_000
    client = session([HumanMessage("go"), AIMessage(earlier), AIMessage("Done.")])
    reply = mcp_record.reply_of(client, f"{earlier}\n\nDone.")
    assert reply.notes.startswith(mcp_record.NOTES_CUT)
    assert len(reply.notes) <= mcp_record.NOTES_MAX_CHARS + len(mcp_record.NOTES_CUT)
    assert reply.notes_chars == 10_000


def test_a_last_line_longer_than_the_limit_is_cut_inside_not_dropped():
    long_line = "R" * 5000
    kept = mcp_record.notes_tail(f"{chatter(50)}\n{long_line}\nDone.")
    assert kept.startswith(mcp_record.NOTES_CUT) and kept.endswith(
        "R" * 100 + "\nDone."
    )
    body = kept[len(mcp_record.NOTES_CUT) :]
    assert len(body) == mcp_record.NOTES_MAX_CHARS


def test_a_partial_line_goes_only_when_enough_is_left():
    limit = mcp_record.NOTES_MAX_CHARS
    # the line cut in two is short: the tail starts at the next line
    text = "a" * 50 + "\n" + "b" * (limit - 10)
    assert mcp_record.notes_tail(text) == mcp_record.NOTES_CUT + "b" * (limit - 10)


def test_a_turn_without_a_final_answer_keeps_the_end_of_its_text_too():
    text = f"{chatter(200)}\n{REPORT}"
    reply = mcp_record.reply_of(session([HumanMessage("go"), AIMessage("")]), text)
    assert not reply.final and REPORT in reply.notes
    assert reply.notes_chars == len(text)


def test_the_record_and_the_status_carry_the_full_length(
    engine, monkeypatch, tmp_path, data
):
    _opened, script = engine
    earlier = f"{chatter(300)}\n{REPORT}"
    script["events"] = [{**COMPLETE, "response": f"{earlier}\n\n{POINTER}"}]
    graph = FakeGraph([HumanMessage("go"), AIMessage(earlier), AIMessage(POINTER)])
    monkeypatch.setattr(ScriptedClient, "graph", graph, raising=False)
    status = run_engine(tmp_path, data)
    assert status["summary"] == POINTER
    assert "notes" not in status  # the status carries only how long they are
    assert status["notes_chars"] == len(earlier)
    record = json.loads(Path(status["result_path"]).read_text())
    assert record["notes_chars"] == len(earlier) and REPORT in record["notes"]


def test_a_turn_with_no_commentary_has_neither_field(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [COMPLETE]
    status = run_engine(tmp_path, data)
    assert "notes_chars" not in status
    assert "notes_chars" not in json.loads(Path(status["result_path"]).read_text())


def test_a_first_run_that_loaded_nothing_keeps_the_end_of_what_it_said(
    engine, tmp_path, data
):
    _opened, script = engine
    script["loads_input"] = False
    said = f"{chatter(300)}\nThat file is not expression data."
    script["events"] = [{**COMPLETE, "response": said}]
    status = run_engine(tmp_path, data)
    assert status["error"]["code"] == "input_not_loaded"
    assert status["notes"].endswith("That file is not expression data.")
    assert status["notes_chars"] == len(said)


def test_the_schema_has_the_length_as_an_optional_integer():
    schema = json.loads(
        (
            Path(mcp_record.__file__).parents[1]
            / "schemas/lobster.result.v1.schema.json"
        ).read_text()
    )
    assert schema["properties"]["notes_chars"]["type"] == "integer"
    assert schema["properties"]["notes_chars"]["minimum"] == 0
    assert "notes_chars" not in schema.get("required", [])
    assert "end of" in schema["properties"]["notes"]["description"]


def test_the_texts_say_what_notes_are_and_ask_for_the_whole_reply_in_one_message():
    status_text = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_status"][
        "description"
    ]
    assert "notes_chars" in status_text and "commentary" in status_text
    sentence = (
        "Put the whole reply in your last message; do not refer to text you "
        "sent earlier."
    )
    assert mcp_record._QUESTION_LEAD.endswith(sentence)
    assert sentence in mcp_record._CONTINUE_LEAD
