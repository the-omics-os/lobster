"""Every tool reply stays one JSON text block whose first key, ``headline``, is
one line a host can show: no brace, at most 120 characters, the rest unchanged."""

from __future__ import annotations

import json

import pytest

from lobster.cli_internal import mcp_contract
from lobster.cli_internal.mcp_headline import HEADLINE_MAX_CHARS, elapsed, headline
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner

LONG = "Approve {the} plan? " + "x" * 300

# One reply per status a run reports, and the other reply shapes.
PAYLOADS = {
    "starting": {
        "status": "running",
        "run_id": "r1",
        "phase": "starting",
        "elapsed_s": 2,
    },
    "running": {
        "status": "running",
        "run_id": "r1",
        "phase": "running",
        "step": "filter_and_normalize",
        "steps_done": 3,
        "elapsed_s": 252,
        "download": {"file": "GSE1.h5ad", "mb_done": 3, "mb_total": 9},
    },
    "cancelling": {"status": "running", "run_id": "r1", "phase": "cancelling"},
    "timed_out": {
        "run_id": "r1",
        "status": "running",
        "elapsed_s": 50,
        "timed_out": True,
    },
    "needs_decision": {
        "status": "needs_decision",
        "phase": "waiting",
        "decision": {"decision_id": "d1", "request_version": 1, "question": LONG},
    },
    "completed": {
        "status": "completed",
        "elapsed_s": 3725,
        "files": [
            {"path": "/o/result.json", "category": "result"},
            {"path": "/o/umap.html", "category": "plots"},
        ],
        "warnings": [{"code": "no_final_answer", "message": "m"}],
    },
    "failed": {
        "status": "failed",
        "summary": "provider_key_rejected: no",
        "error": {"code": "provider_key_rejected", "message": "no"},
    },
    "cancelled": {"status": "cancelled", "summary": "s", "cancel_reason": "cancel"},
    "nan": {"status": "running", "elapsed_s": float("nan")},
    "answered": {
        "run_id": "r1",
        "outcome": "answered",
        "source": "human",
        "answer": "go",
    },
    "cancel": {"run_id": "r1", "outcome": "cancel_requested"},
    "other": {"results": [{"slug": "a"}]},
}


def reply_of(name: str, payload: dict) -> dict:
    server = McpServer(ThreadRunner(lambda request, cancel_event: {}))
    server._tools[name] = lambda args: dict(payload)
    server._lobster_wait = lambda args, request_id=None: dict(payload)
    line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": {}},
        }
    )
    try:
        return server.handle_line(line)["result"]
    finally:
        server.runner.stop(5)


@pytest.mark.parametrize("tool", [t["name"] for t in mcp_contract.TOOLS])
@pytest.mark.parametrize("kind", sorted(PAYLOADS))
def test_every_reply_is_one_json_block_led_by_its_headline(tool, kind):
    payload = PAYLOADS[kind]
    if kind == "nan":  # NaN is not JSON: the reply is the one-line error
        assert reply_of(tool, payload)["isError"] is True
        assert headline(tool, payload) == "⏳ running"
        return
    result = reply_of(tool, payload)
    assert set(result) == {"content"}
    assert len(result["content"]) == 1 and result["content"][0]["type"] == "text"
    reply = json.loads(result["content"][0]["text"])
    assert next(iter(reply)) == "headline"
    line = reply.pop("headline")
    assert line and "\n" not in line and "{" not in line and "}" not in line
    assert len(line) <= HEADLINE_MAX_CHARS
    assert reply == payload


def test_headlines_name_what_the_user_needs():
    by = {kind: headline("lobster_wait", p) for kind, p in PAYLOADS.items()}
    assert by["starting"] == "⏳ starting · 2s"
    assert (
        by["running"]
        == "⏳ running · filter_and_normalize · 3 steps · ↓ GSE1.h5ad · 4m12s"
    )
    assert by["cancelling"] == "⏳ cancelling"
    assert by["timed_out"] == "⏳ running · 50s"
    assert by["needs_decision"].startswith("? needs decision · Approve (the) plan? xxx")
    assert by["needs_decision"].endswith("…")
    assert by["completed"] == "✓ completed · 1h02m · 2 files · 1 plot · 1 warning"
    assert by["failed"] == "✗ failed · provider_key_rejected"
    assert by["cancelled"] == "■ cancelled · cancel"
    assert by["answered"] == "↳ answered (human)"
    assert by["cancel"] == "■ cancel · cancel_requested"
    assert headline("lobster_docs", PAYLOADS["other"]) == "ℹ docs"


def test_elapsed_reads_seconds_minutes_and_hours():
    assert [elapsed(v) for v in (0, 59, 60, 252, 3600, 3725)] == [
        "0s",
        "59s",
        "1m00s",
        "4m12s",
        "1h00m",
        "1h02m",
    ]
    assert elapsed(None) == elapsed(True) == elapsed(float("inf")) == ""
