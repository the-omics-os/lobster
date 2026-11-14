"""``lobster_continue`` across two real server processes, on replay scripts.

The first process runs a QC recipe and a follow-up turn; the second one is
started afterwards and continues from ``output_dir`` alone. The engine, the run
API and the session restore are the real ones; only the model is a script.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lobster.cli_internal.mcp_record import NOTICE_LEAD as NOTICE
from tests.unit.mcp.replay_input import make_input, script
from tests.unit.mcp.transcriptomics import quality_service

quality_service()

ENTRY = (
    "import json, os, sys\n"
    "from lobster.cli_internal.mcp_stdio import run_mcp_stdio\n"
    "log = os.environ.get('MCP_TEST_MESSAGES')\n"
    "if log:  # what the scripted model is sent, one JSON line per call\n"
    "    from lobster.config.providers.replay_provider import ReplayChatModel\n"
    "    generate = ReplayChatModel._generate\n"
    "    def spy(self, messages, *a, **k):\n"
    "        with open(log, 'a') as f:\n"
    "            f.write(json.dumps([[m.type, str(m.content)] for m in messages]) + '\\n')\n"
    "        return generate(self, messages, *a, **k)\n"
    "    ReplayChatModel._generate = spy\n"
    "sys.exit(run_mcp_stdio())\n"
)
DEADLINE_S = 300  # a loaded machine needs minutes for the first start-up


class Host:
    """One server process on one replay script."""

    def __init__(self, script: Path, cwd: Path, messages: Optional[Path] = None):
        env = dict(os.environ)
        env.update(
            LOBSTER_TEST_REPLAY_FIXTURE=str(script),
            LOBSTER_LLM_PROVIDER="ollama-cloud",
            OLLAMA_API_KEY="replay-dummy",
            PYTHONDONTWRITEBYTECODE="1",
        )
        if messages is not None:
            env["MCP_TEST_MESSAGES"] = str(messages)
        cwd.mkdir()
        self.proc = subprocess.Popen(  # nosec B603
            [sys.executable, "-c", ENTRY],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            cwd=cwd,
        )
        self.n = 0

    def call(self, name: str, **arguments):
        self.n += 1
        message = {"jsonrpc": "2.0", "id": self.n, "method": "tools/call"}
        message["params"] = {"name": name, "arguments": arguments}
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        result = json.loads(self.proc.stdout.readline())["result"]
        text = result["content"][0]["text"]
        return text if result.get("isError") else json.loads(text)

    def finished(self, run_id: str) -> dict:
        deadline = time.monotonic() + DEADLINE_S
        while time.monotonic() < deadline:
            status = self.call("lobster_status", run_id=run_id)
            if status["status"] != "running":
                return status
            time.sleep(0.5)
        raise AssertionError(f"run {run_id} did not finish")

    def close(self) -> int:
        self.proc.stdin.close()
        try:
            return self.proc.wait(timeout=60)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()


def tools_recorded(workspace: Path) -> list:
    (log,) = workspace.glob(".lobster/sessions/*/provenance.jsonl")
    lines = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return [line.get("type") or line.get("tool_name") for line in lines]


def notebook_text(path: Path) -> str:
    cells = json.loads(path.read_text())["cells"]
    return "\n".join("".join(c["source"]) for c in cells)


@pytest.mark.slow
def test_a_session_survives_a_restart_without_doubling_its_history(tmp_path):
    data, out = tmp_path / "input.h5ad", tmp_path / "out"
    make_input(data)
    first = Host(
        script("continue_first_process.json", tmp_path, data), tmp_path / "cwd1"
    )
    try:
        run = first.call(
            "lobster_run",
            input_path=str(data),
            output_dir=str(out),
            analysis="scrna_qc",
            instruction="use the tool defaults",
        )
        assert first.finished(run["run_id"])["status"] == "completed"
        second_turn = first.call(
            "lobster_continue", run_id=run["run_id"], instruction="summarise it"
        )
        done = first.finished(second_turn["run_id"])
        assert done["status"] == "completed", done
        assert (done["continues"], done["turn"]) == (run["run_id"], 2)
        assert Path(done["result_path"]) == out / "result_2.json"
    finally:
        assert first.close() == 0
    workspace = out / ".lobster_workspace"
    before = tools_recorded(workspace)

    # A new process: the server remembers nothing, the folder holds the session.
    second = Host(
        script("continue_second_process.json", tmp_path, data), tmp_path / "cwd2"
    )
    try:
        refused = second.call(
            "lobster_continue", run_id=run["run_id"], instruction="again"
        )
        assert refused.startswith("unknown_run:") and "output_dir" in refused
        third = second.call(
            "lobster_continue", output_dir=str(out), instruction="and once more"
        )
        assert (third["continues"], third["turn"]) == (done["run_id"], 3)
        final = second.finished(third["run_id"])
        assert final["status"] == "completed", final
        assert final["summary"] == "Third turn answer, after a restart."
        assert Path(final["notebook_path"]) == out / "lobster_scrna_qc_3.ipynb"
    finally:
        assert second.close() == 0

    # Restoring read the history once: only the new instruction was added.
    after = tools_recorded(workspace)
    assert after[: len(before)] == before
    assert after[len(before) :] == ["host_instruction"]
    assert before.count("assess_data_quality") == 1
    assert before.count("filter_and_normalize") == 1
    notebook = notebook_text(out / "lobster_scrna_qc_3.ipynb")
    assert notebook.count(": assess_data_quality") == 1  # one step heading
    assert notebook.count(": filter_and_normalize") == 1
    assert notebook.count(": host_instruction") == 3
    record = json.loads((out / "result_3.json").read_text())
    assert record["continued_from"] == done["run_id"]
    assert [
        json.loads((out / n).read_text())["request"]["turn"]
        for n in ("result.json", "result_2.json", "result_3.json")
    ] == [1, 2, 3]


NEW_DATA = "input_quality_assessed_filtered_normalized_quality_assessed"


def sent_to_the_model(log: Path) -> list:
    """Every message the scripted model received, as ``(type, text)``."""
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    return [tuple(message) for call in calls for message in call]


@pytest.mark.slow
def test_a_restart_tells_the_next_turn_which_modalities_did_not_come_back(tmp_path):
    data, out = tmp_path / "input.h5ad", tmp_path / "out"
    make_input(data)
    log_one, log_two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    first = Host(
        script("unsaved_first_process.json", tmp_path, data), tmp_path / "c1", log_one
    )
    try:
        run = first.call(
            "lobster_run",
            input_path=str(data),
            output_dir=str(out),
            analysis="scrna_qc",
            instruction="use the tool defaults",
        )
        assert first.finished(run["run_id"])["status"] == "completed"
        second = first.call(
            "lobster_continue", run_id=run["run_id"], instruction="assess again"
        )
        done = first.finished(second["run_id"])
        assert done["status"] == "completed", done
    finally:
        assert first.close() == 0
    # The second turn made a modality and saved no data file of it.
    assert "no_data_file_saved" in [w["code"] for w in done["warnings"]]
    made = [m["name"] for m in done["modality_details"]]
    assert NEW_DATA in made
    assert not [m for m in sent_to_the_model(log_one) if NOTICE in m[1]]

    restarted = Host(
        script("unsaved_second_process.json", tmp_path, data), tmp_path / "c2", log_two
    )
    try:
        third = restarted.call(
            "lobster_continue", output_dir=str(out), instruction="and once more"
        )
        final = restarted.finished(third["run_id"])
        assert final["status"] == "completed", final
    finally:
        assert restarted.close() == 0
    restored = [m["name"] for m in final["modality_details"]]
    gone = [name for name in made if name not in restored]
    assert NEW_DATA in gone and gone != made  # one file came back, the rest did not
    (warning,) = [
        w for w in final["warnings"] if w["code"] == "modalities_not_restored"
    ]
    assert all(name in warning["message"] for name in gone)
    record = json.loads((out / "result_3.json").read_text())
    assert record["request"]["not_restored"] == gone
    assert warning in record["warnings"]
    # The model was told, in server words, after the instruction block.
    received = [text for kind, text in sent_to_the_model(log_two) if kind == "human"]
    (prompt,) = {t for t in received if "and once more" in t}
    lines = prompt.split("\n")
    notice = [ln for ln in lines if ln.startswith(NOTICE)]
    assert len(notice) == 1 and lines[-1] == notice[0]
    assert json.dumps(gone) in notice[0]
    assert (
        lines[lines.index(json.dumps({"instruction": "and once more"})) + 1]
        != notice[0]
    )
    # It is not the host's instruction in the notebook or the record.
    assert NOTICE not in notebook_text(out / "lobster_scrna_qc_3.ipynb")
    assert record["request"]["instruction"] == "and once more"
