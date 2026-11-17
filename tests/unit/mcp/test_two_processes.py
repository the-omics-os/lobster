"""Two real server processes, one ``output_dir``: the second is refused, and
takes the folder over once the first has closed. The model is a replay script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_lock, mcp_record
from tests.unit.mcp.test_continue_process import Host, make_input, script


def lock_of(out: Path) -> Path:
    return out / mcp_record.WORKSPACE_DIR / mcp_lock.LOCK_NAME


@pytest.mark.slow
def test_a_second_process_is_refused_until_the_first_has_closed(tmp_path):
    data, out = tmp_path / "input.h5ad", tmp_path / "out"
    make_input(data)
    args = {"question": "q", "input_path": str(data), "output_dir": str(out)}
    args["plan"] = False  # the script hands off to the analysis expert at once
    first = Host(script("continue_first_process.json", tmp_path, data), tmp_path / "c1")
    second = Host(
        script("continue_second_process.json", tmp_path, data), tmp_path / "c2"
    )
    try:
        run = first.call("lobster_run", **args)
        assert run["status"] == "running"  # the folder is claimed before this reply
        assert lock_of(out).read_text().strip() == str(first.proc.pid)

        refused = second.call("lobster_run", **args)
        assert isinstance(refused, str)
        assert refused.startswith("output_dir_in_use:")
        assert "another Lobster process" in refused and "\n" not in refused
        busy = second.call("lobster_continue", output_dir=str(out), instruction="go")
        assert isinstance(busy, str) and busy.startswith("output_dir_in_use:")
        assert "another Lobster process" in busy

        assert first.finished(run["run_id"])["status"] == "completed"
        # The idle session of the first process still holds the folder.
        for tool, extra in (("lobster_run", args), ("lobster_continue", {})):
            reply = second.call(
                tool,
                **(extra or {"output_dir": str(out), "instruction": "go"}),
            )
            assert isinstance(reply, str) and "another Lobster process" in reply
        before = (out / "result.json").read_bytes()
    finally:
        assert first.close() == 0
    assert not lock_of(out).exists()  # closing released the folder

    try:
        turn = second.call("lobster_continue", output_dir=str(out), instruction="go on")
        assert turn["turn"] == 2, turn
        assert lock_of(out).read_text().strip() == str(second.proc.pid)
        assert second.finished(turn["run_id"])["status"] == "completed"
    finally:
        assert second.close() == 0
    assert (out / "result.json").read_bytes() == before
    assert json.loads((out / "result_2.json").read_text())["request"]["turn"] == 2
    assert not lock_of(out).exists()
