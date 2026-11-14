"""A first run whose session holds no data read from ``input_path`` fails: the
model may have stopped before loading, or the file could not be read."""

from __future__ import annotations

import json

import pytest

from tests.unit.mcp.test_continue_process import Host, make_input, script


def first_run(tmp_path, name, readable=True):
    data, out = tmp_path / "input.h5ad", tmp_path / "out"
    if readable:
        make_input(data)
    else:
        data.write_bytes(b"this is not an h5ad file")
    host = Host(script(name, tmp_path, data), tmp_path / "cwd")
    try:
        run = host.call(
            "lobster_run",
            input_path=str(data),
            output_dir=str(out),
            analysis="scrna_qc",
        )
        done = host.finished(run["run_id"])
    finally:
        assert host.close() == 0
    return data, out, done


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, readable, said",
    [
        ("input_unreadable_process.json", False, "I could not read the file"),
        ("input_skipped_process.json", True, "The QC for this file looks fine."),
    ],
    ids=["the file could not be read", "the model never loaded it"],
)
def test_a_first_run_that_loaded_no_data_failed(tmp_path, name, readable, said):
    data, out, done = first_run(tmp_path, name, readable)
    assert done["status"] == "failed", done
    assert done["error"]["code"] == "input_not_loaded"
    assert done["error"]["retryable"] is False
    assert f"The file at {data} was not loaded" in done["summary"]
    assert "nothing was analysed" in done["summary"]
    assert said in done["notes"]  # what the model said is not the summary
    assert said not in done["summary"]
    # Nothing is published: no result, notebook or data file at the top level.
    assert sorted(p.name for p in out.iterdir()) == [".lobster_workspace"]
    assert "result_path" not in done and "files" not in done
