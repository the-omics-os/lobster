"""The answers a host gave during a turn are listed in the turn's record and
status, as the host reported them."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from lobster.cli_internal import mcp_decision
from tests.unit.mcp.test_answer import (  # noqa: F401  (fixtures)
    SLIDER,
    SLIDER_PROMPT,
    answer_args,
    call,
    interrupt,
    server,
    start,
    wait_waiting,
)
from tests.unit.mcp.test_publish import scan  # noqa: F401
from tests.unit.mcp.test_run import COMPLETE, data, engine, wait_terminal  # noqa: F401

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "lobster/schemas/lobster.result.v1.schema.json"
    ).read_text()
)
NOTE = "Answers were reported by the host; Lobster cannot check who gave them."


def at_version(server, run_id, version, timeout=30):
    """The waiting status of the question at *version*, once it is shown."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = call(server, "lobster_status", {"run_id": run_id})
        decision = status.get("decision") or {}
        if status["status"] == "needs_decision" and (
            decision.get("request_version") == version
        ):
            return status
        time.sleep(0.02)
    raise AssertionError(f"question {version} never shown: {status}")


def record_of(done):
    record = json.loads(Path(done["result_path"]).read_text())
    Draft202012Validator(SCHEMA).validate(record)
    return record


def test_an_answer_is_listed_as_reported(engine, server, tmp_path, data):
    _opened, script = engine
    script.update(events=[SLIDER], resume_events=[COMPLETE])
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    call(
        server,
        "lobster_answer",
        answer_args(run_id, waiting, 0.1, "model", comment="per lab SOP"),
    )
    done = wait_terminal(server, run_id)
    expected = [
        {
            "question": "Which mitochondrial threshold?",
            "answer": {"value": 0.1},
            "source": "model",
            "channel": "mcp:claude-code",
            "comment": "per lab SOP",
        }
    ]
    assert done["decisions"] == expected
    assert record_of(done)["decisions"] == expected
    assert done["decisions_note"] == NOTE
    assert "decisions_note" not in record_of(done)
    # Only the load of the input is a step; the answer is not.
    assert [s["tool"] for s in record_of(done).get("steps", [])] == ["load_dataset"]


def test_the_source_is_the_one_the_host_gave(engine, server, tmp_path, data):
    _opened, script = engine
    script.update(events=[SLIDER], resume_events=[COMPLETE])
    run_id = start(server, tmp_path, data)
    call(
        server,
        "lobster_answer",
        answer_args(run_id, wait_waiting(server, run_id), 0.2, "human"),
    )
    done = wait_terminal(server, run_id)
    (entry,) = done["decisions"]
    assert entry["source"] == "human" and "comment" not in entry


def test_a_refused_answer_is_not_listed(engine, server, tmp_path, data):
    _opened, script = engine
    script.update(events=[SLIDER], resume_events=[COMPLETE])
    run_id = start(server, tmp_path, data)
    waiting = wait_waiting(server, run_id)
    refused = call(server, "lobster_answer", answer_args(run_id, waiting, 7.0))
    assert refused["error_text"].startswith("bad_answer:")
    call(server, "lobster_answer", answer_args(run_id, waiting, 0.3))
    (entry,) = wait_terminal(server, run_id)["decisions"]
    assert entry["answer"] == {"value": 0.3}


def test_a_run_with_no_question_has_no_decisions_and_no_note(
    engine, server, tmp_path, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    done = wait_terminal(server, start(server, tmp_path, data))
    assert "decisions" not in done and "decisions_note" not in done
    assert "decisions" not in record_of(done)


def test_a_long_question_is_cut_and_the_count_is_bounded(
    engine, server, tmp_path, data, monkeypatch
):
    monkeypatch.setattr(mcp_decision, "MAX_RECORDED", 2)
    opened, script = engine
    long_question = interrupt(
        "threshold_slider",
        {**SLIDER["data"]["data"], "label": "Q" * 3000},
        SLIDER_PROMPT,
    )
    more = [
        interrupt(
            "threshold_slider",
            SLIDER["data"]["data"],
            SLIDER_PROMPT,
            interrupt_id=f"lg_{n}",
        )
        for n in range(2, 5)
    ]
    script.update(
        events=[long_question],
        resume_events=[[more[0]], [more[1]], [more[2]], [COMPLETE]],
    )
    run_id = start(server, tmp_path, data)
    for version in (1, 2, 3, 4):
        waiting = at_version(server, run_id, version)
        call(server, "lobster_answer", answer_args(run_id, waiting, 0.1))
    done = wait_terminal(server, run_id)
    assert len(done["decisions"]) == 2 and done["decisions_total"] == 4
    assert record_of(done)["decisions_total"] == 4
    question = done["decisions"][0]["question"]
    assert len(question) <= mcp_decision.RECORDED_QUESTION_MAX_CHARS
    assert question.endswith(mcp_decision.CUT)


def test_the_schema_allows_decisions_only_as_optional():
    assert "decisions" not in SCHEMA.get("required", [])
    item = SCHEMA["properties"]["decisions"]["items"]
    assert set(item["required"]) == {"question", "answer", "source", "channel"}
    assert "comment" in item["properties"]


# the model behind a run ---------------------------------------------------


def test_a_run_on_a_replay_script_says_so(engine, server, tmp_path, data, monkeypatch):
    from lobster.config.providers.replay_provider import REPLAY_ENV

    monkeypatch.setenv(REPLAY_ENV, "/some/script.json")
    _opened, script = engine
    script["events"] = [COMPLETE]
    done = wait_terminal(server, start(server, tmp_path, data))
    assert done["model_source"] == "replay"
    assert record_of(done)["request"]["model_source"] == "replay"


def test_a_run_on_a_real_model_says_nothing_of_it(
    engine, server, tmp_path, data, monkeypatch
):
    from lobster.config.providers.replay_provider import REPLAY_ENV

    monkeypatch.delenv(REPLAY_ENV, raising=False)
    _opened, script = engine
    script["events"] = [COMPLETE]
    done = wait_terminal(server, start(server, tmp_path, data))
    assert "model_source" not in done
    assert "model_source" not in record_of(done)["request"]
    monkeypatch.setenv(REPLAY_ENV, "")  # set but empty: the factory ignores it
    done = wait_terminal(server, start(server, tmp_path, data, "out2"))
    assert "model_source" not in done


def test_the_schema_allows_model_source_only_as_optional():
    request = SCHEMA["properties"]["request"]
    assert "model_source" not in request["required"]
    assert request["properties"]["model_source"]["enum"] == ["replay"]
