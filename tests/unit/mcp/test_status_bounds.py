"""A status line stays inside ``STATUS_MAX_CHARS`` whatever the engine reports."""

from __future__ import annotations

import json

from lobster.cli_internal import mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)


def status_text(server: McpServer, run_id: str) -> str:
    """The text block of a ``lobster_status`` reply, as the host receives it."""
    line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lobster_status", "arguments": {"run_id": run_id}},
        }
    )
    return server.handle_line(line)["result"]["content"][0]["text"]


def test_a_huge_provider_error_is_cut_in_the_status_and_whole_on_stderr(
    engine, tmp_path, data, capsys
):
    _opened, script = engine
    script["events"] = [
        {"type": "error", "error": "E" * 300_000, "error_type": "provider_error"}
    ]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    wait_terminal(server, run["run_id"])

    text = status_text(server, run["run_id"])
    assert len(text) < mcp_stdio.STATUS_MAX_CHARS
    status = json.loads(text)
    assert status["status"] == "failed"
    message = status["error"]["message"]
    assert len(message) <= 1000 and message.endswith("...") and "\n" not in message
    assert status["error"]["code"] == "provider_error"
    assert len(status["summary"]) <= 300
    assert "E" * 300_000 in capsys.readouterr().err


def test_a_multi_line_error_message_is_one_line(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [
        {"type": "error", "error": "first\nsecond\n  third", "error_type": "x"}
    ]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    failed = wait_terminal(server, run["run_id"])
    assert failed["error"]["message"] == "first second third"


def test_a_huge_error_code_is_cut_too(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [{"type": "error", "error": "x", "error_type": "C" * 50_000}]
    server = McpServer(ThreadRunner())
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    wait_terminal(server, run["run_id"])
    text = status_text(server, run["run_id"])
    assert len(text) < 5000


def test_no_string_of_a_completed_status_is_unbounded():
    huge = "M" * 2_000_000
    status = {
        "status": "completed",
        "summary": "ok",
        "result_path": "/out/result.json",
        "files": [{"path": "/out/a.h5ad", "category": "data", "size_bytes": 1}],
        "modality_details": [
            {"name": huge, "stage": huge, "parent": huge, "file": "f"}
        ],
        "steps": [{"tool": huge, "parameters": {huge: huge}}],
        "warnings": [{"code": "metrics_unavailable", "message": huge}],
        "result": {"session_id": huge},
    }
    bounded = mcp_stdio._bounded(status)
    assert len(json.dumps(bounded)) < mcp_stdio.STATUS_MAX_CHARS
    assert bounded["files"] == status["files"]
    assert bounded["result_path"] == "/out/result.json"
    assert bounded["warnings"][0]["code"] == "metrics_unavailable"
