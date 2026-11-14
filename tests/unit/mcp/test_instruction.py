"""The host's free text: the ``question`` of ``lobster_run`` and the
``instruction`` of ``lobster_continue``.

It is bounded, composed as one JSON-quoted line, and recorded: a comment-only
notebook cell (non-replayable), the ``request`` in ``result.json`` and
``question_recorded`` in the reply.
"""

from __future__ import annotations

import ast
import json
import re
import threading
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_contract, mcp_record, mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, RunRequest, ThreadRunner
from lobster.core.runs import new_run_id
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    Gate,
    Hold,
    call,
    data,
    engine,
    wait_terminal,
)

HOSTILE = [
    "line one\nimport os; os.system('id')\nline three",
    "a\r\nb c d\x0be\x85f\x1cg",
    'it\'s "quoted" and a backslash \\ at the end\\',
    "#!/bin/sh\nrm -rf /",
    "{{ 7*7 }}{% raw %} {input_path} {0} {}",
    "'); import os #",
    "café ☃ \U0001f600",
]


INVISIBLE = (
    "instruction must not contain invisible format characters (zero-width, "
    "direction marks) or the line and paragraph separators U+2028 and U+2029"
)

# normalising -------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("use a 10 % mito cutoff", "use a 10 % mito cutoff"),
        ("  use a\n 10 %\t mito  ", "use a 10 % mito"),
        ("a\r\nb c d\x85e\x0bf\x1cg", "a b c d e f g"),
        ("x" * 500, "x" * 500),
        ("x" * 250 + "   \n  " + "y" * 249, "x" * 250 + " " + "y" * 249),
    ],
)
def test_an_instruction_is_stripped_and_its_whitespace_collapsed(raw, expected):
    assert mcp_record.normalise_instruction(raw) == expected


@pytest.mark.parametrize(
    "raw, message",
    [
        ("x" * 4001, "instruction_too_long: 4000 characters at most"),
        (
            "x" * 2000 + "  " + "y" * 2000,
            "instruction_too_long: 4000 characters at most",
        ),
        ("", "instruction must be a non-empty string when given"),
        (" \n\t ", "instruction must be a non-empty string when given"),
        (7, "instruction must be a non-empty string when given"),
        (["a"], "instruction must be a non-empty string when given"),
        ("a\x00b", "instruction must not contain control characters"),
        ("a\x1bb", "instruction must not contain control characters"),
        ("a\x7fb", "instruction must not contain control characters"),
        ("a\x9bb", "instruction must not contain control characters"),
        # Format characters (category Cf) and the line and paragraph separators
        # read as a space or as nothing to a person and as text to a model.
        ("a\u200bb", INVISIBLE),  # zero-width space
        ("a\u200db", INVISIBLE),  # zero-width joiner
        ("a\u202eb", INVISIBLE),  # right-to-left override
        ("a\u2066b", INVISIBLE),  # left-to-right isolate
        ("a\ufeffb", INVISIBLE),  # byte order mark
        ("a\U000e0041b", INVISIBLE),  # tag letter
        ("a\u2028b", INVISIBLE),  # line separator
        ("a\u2029b", INVISIBLE),  # paragraph separator
    ],
)
def test_a_bad_instruction_is_refused_with_its_reason(raw, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        mcp_record.normalise_instruction(raw)


# the prompt -------------------------------------------------------------------


@pytest.mark.parametrize("raw", HOSTILE)
def test_what_the_question_holds_cannot_leave_its_line_or_be_formatted(raw):
    text = mcp_record.normalise_question(raw.replace("\x1b", ""))
    plain = mcp_record.compose_question_prompt("q", "/in/{x}.h5ad").split("\n")
    prompt = mcp_record.compose_question_prompt(text, "/in/{x}.h5ad").split("\n")
    assert len(prompt) == len(plain)  # the text added no line
    assert json.loads(prompt[plain.index('{"question": "q"}')]) == {"question": text}
    assert "Input file: /in/{x}.h5ad" in prompt  # the path is not formatted


# the notebook cell ------------------------------------------------------------


def test_the_cell_is_two_comment_lines_in_the_decision_wording():
    ir = mcp_record.instruction_ir("use a 10 % mito cutoff", "mcp:claude-code")
    assert ir.render() == (
        "# Instruction from the MCP host 'claude-code', as reported. Lobster cannot "
        "verify who wrote it.\n"
        "# Instruction: 'use a 10 % mito cutoff'"
    )
    assert ir.replayable is False
    assert ir.parameters == {
        "instruction": "use a 10 % mito cutoff",
        "channel": "mcp:claude-code",
    }


@pytest.mark.parametrize("raw", HOSTILE + ["#!payload", "x\x1by"])
@pytest.mark.parametrize(
    "channel",
    ["mcp:claude-code", "mcp:x'); import os #\nimport os", "no colon\nimport os"],
)
def test_a_hostile_instruction_stays_two_comment_lines(raw, channel):
    rendered = mcp_record.instruction_ir(raw, channel).render()
    lines = rendered.split("\n")
    assert len(lines) == 2 and rendered.splitlines() == lines
    for line in lines:
        assert line.startswith("#"), line
    assert ast.parse(rendered).body == []


def test_the_question_cell_is_two_comment_lines_in_the_same_wording():
    ir = mcp_record.question_ir("which datasets?", "mcp:claude-code")
    assert ir.render() == (
        "# Question from the MCP host 'claude-code', as reported. Lobster cannot "
        "verify who wrote it.\n"
        "# Question: 'which datasets?'"
    )
    assert ir.replayable is False
    for raw in HOSTILE:
        rendered = mcp_record.question_ir(raw, "mcp:x\nimport os").render()
        assert ast.parse(rendered).body == [] and len(rendered.split("\n")) == 2


def test_the_cells_reach_a_real_notebook():
    import tempfile

    from lobster.core.runtime.data_manager import DataManagerV2

    with tempfile.TemporaryDirectory() as tmp:
        dm = DataManagerV2(workspace_path=Path(tmp) / "ws")
        assert mcp_record.record_question(dm, "which datasets?", "mcp:claude-code")
        assert mcp_record.record_instruction(
            dm, "use a 10 % mito cutoff", "mcp:claude-code"
        )
        notebook = json.loads(Path(dm.export_notebook("nb")).read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"]]
    assert (
        "# Question from the MCP host 'claude-code', as reported. Lobster cannot "
        "verify who wrote it.\n# Question: 'which datasets?'"
    ) in sources
    assert (
        "# Instruction from the MCP host 'claude-code', as reported. Lobster cannot "
        "verify who wrote it.\n# Instruction: 'use a 10 % mito cutoff'"
    ) in sources


# through lobster_run ----------------------------------------------------------


def initialized(server, host="claude-code"):
    reply = server.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "clientInfo": {"name": host},
                },
            }
        )
    )
    assert "result" in reply


def run_args(data, out, question="  which\n datasets exist? "):
    return {"question": question, "input_path": str(data), "output_dir": str(out)}


def test_the_run_request_carries_the_normalised_question_and_the_channel(
    tmp_path, data
):
    gate = Gate()
    gate.release.set()
    server = McpServer(ThreadRunner(gate))
    initialized(server)
    started = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    assert set(started) == {"status", "run_id", "output_dir", "question_recorded"}
    wait_terminal(server, started["run_id"])
    assert gate.calls[0].question == "which datasets exist?"
    assert gate.calls[0].instruction is None
    assert gate.calls[0].channel == "mcp:claude-code"


def test_a_refused_question_starts_no_run(tmp_path, data):
    gate = Gate()
    server = McpServer(ThreadRunner(gate))
    refused = call(server, "lobster_run", run_args(data, tmp_path / "out", "a\x00b"))
    assert refused == {"error_text": "question must not contain control characters"}
    assert gate.calls == [] and not (tmp_path / "out").exists()


# in the engine ----------------------------------------------------------------


def asked(tmp_path, data, text="which datasets exist?"):
    request = RunRequest(
        new_run_id(),
        str(data),
        str(tmp_path / "out"),
        text,
        channel="mcp:claude-code",
    )
    return mcp_stdio.run_engine(request, threading.Event())


def test_it_is_recorded_before_the_run_starts_and_is_not_a_step(engine, tmp_path, data):
    opened, script = engine
    hold = Hold("assess_data_quality")
    hold.release.set()
    script["events"] = [hold, COMPLETE]
    status = asked(tmp_path, data)
    activities = opened[0]["client"].data_manager.provenance.activities
    assert [a["type"] for a in activities] == [
        "host_question",
        "load_dataset",
        "assess_data_quality",
    ]
    recorded = activities[0]
    assert recorded["parameters"] == {
        "question": "which datasets exist?",
        "channel": "mcp:claude-code",
    }
    assert recorded["ir"].replayable is False
    on_disk = json.loads(Path(status["result_path"]).read_text())
    assert [s["tool"] for s in on_disk["steps"]] == [
        "load_dataset",
        "assess_data_quality",
    ]
    assert on_disk["request"] == {
        "question": "which datasets exist?",
        "input_path": str(data),
        "plan": False,  # run_engine directly: no lobster_run default
        "instruction": None,
        "channel": "mcp:claude-code",
        "turn": 1,
    }
    assert opened[0]["client"].prompts == [
        mcp_record.compose_question_prompt("which datasets exist?", str(data))
    ]


def test_the_step_count_starts_after_the_question(engine, tmp_path, data):
    _opened, script = engine
    hold = Hold("assess_data_quality")
    script["events"] = [hold, COMPLETE]
    server = McpServer(ThreadRunner())
    started = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    assert hold.entered.wait(10)
    live = call(server, "lobster_status", {"run_id": started["run_id"]})
    assert live["steps_done"] == 2 and live["step"] == "assess_data_quality"
    hold.release.set()
    wait_terminal(server, started["run_id"])


def test_a_question_that_cannot_be_recorded_stops_the_run_before_it_starts(
    engine, tmp_path, data
):
    opened, script = engine
    script["events"] = [COMPLETE]
    script["activities"] = None  # a session with no provenance tracker
    status = asked(tmp_path, data)
    assert status["status"] == "failed"
    assert status["error"]["code"] == "question_not_recorded"
    assert opened[0]["client"].prompts == []  # the model never saw it
