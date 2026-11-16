"""A run that branches still replays: ``lobster replay --verify`` gives MATCH.

A real ``lobster mcp --stdio`` process runs one turn on the replay model and
real data, then the notebook it exported is re-run and compared with the file
the session saved. Two everyday QC shapes branch: the quality check and the
filter both start from the raw data, or the filtered data is checked again
after it was saved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.mcp.replay_input import make_full_rank_input, make_input
from tests.unit.mcp.test_cancel_real_run import Server, ends
from tests.unit.mcp.transcriptomics import quality_service

pytestmark = pytest.mark.no_auto_config

quality_service()


def call(name, **args):
    return {"content": "", "tool_calls": [{"name": name, "args": args}]}


def handoff(expert):
    return call(f"handoff_to_{expert}", task_description="Your task is to work.")


ASSESS = {"min_genes": 200, "max_genes": 5000, "max_mt_pct": 20.0}
FILTER = {
    "min_genes_per_cell": 200,
    "max_genes_per_cell": 5000,
    "min_cells_per_gene": 3,
    "max_mito_percent": 20.0,
    "normalization_method": "log1p",
    "target_sum": 10000,
}
SHAPES = {
    "assess and filter both from the raw data": [
        call("assess_data_quality", modality_name="raw", **ASSESS),
        call("filter_and_normalize", modality_name="raw", **FILTER),
    ],
    "the filtered data assessed again after its save": [
        call("assess_data_quality", modality_name="raw", **ASSESS),
        call("filter_and_normalize", modality_name="raw_quality_assessed", **FILTER),
        call(
            "assess_data_quality",
            modality_name="raw_quality_assessed_filtered_normalized",
            **ASSESS,
        ),
    ],
}


def verify(tmp_path, steps, make=make_input):
    """One turn on the replay model running *steps* on the input *make*
    writes, then ``replay --verify`` of the notebook it exported:
    (exit code, report)."""
    data, out = tmp_path / "raw.h5ad", tmp_path / "out"
    make(data)
    responses = [
        handoff("data_expert_agent"),
        call(
            "load_modality",
            modality_name="raw",
            file_path=str(data),
            adapter="transcriptomics_single_cell",
        ),
        {"content": "Loaded."},
        handoff("transcriptomics_expert"),
        *steps,
        {"content": "Done."},
        {"content": "QC done."},
    ]
    server = Server(tmp_path, responses)
    try:
        run = server.tool(
            "lobster_run",
            input_path=str(data),
            output_dir=str(out),
            question="Run quality control.",
            plan=False,
        )
        done = ends(server, run["run_id"])
        assert done["status"] == "completed", (done, server.stderr())
    finally:
        server.proc.stdin.close()
    assert server.proc.wait(timeout=30) == 0, server.stderr()
    server.err.close()

    replay = subprocess.run(  # nosec B603
        [sys.executable, "-m", "lobster", "replay", "--verify"]
        + [done["notebook_path"], "--input", str(data)]
        + ["-w", str(out / ".lobster_workspace"), "--json"],
        cwd=tmp_path,
        env=server.env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return replay.returncode, json.loads(replay.stdout)


@pytest.mark.slow
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_branched_run_replays_to_a_match(tmp_path, shape):
    code, report = verify(tmp_path, SHAPES[shape])
    assert report["verdict"] == "MATCH", json.dumps(report, indent=1)[:4000]
    assert code == 0


FILTERED = "raw_quality_assessed_filtered_normalized"
THREE_SAVES = [
    call("assess_data_quality", modality_name="raw", **ASSESS),
    call("filter_and_normalize", modality_name="raw_quality_assessed", **FILTER),
    call("cluster_cells", modality_name=FILTERED, resolution=0.5),
    call("find_marker_genes", modality_name=f"{FILTERED}_clustered", groupby="leiden"),
]


@pytest.mark.slow
def test_a_run_that_saved_three_files_replays_to_a_match(tmp_path):
    # The run saves the filtered, the clustered and the marker data; the
    # notebook writes each where it was made, so all three are compared. The
    # input is full rank: the clustering's PCA is then unique (see below).
    code, report = verify(tmp_path, THREE_SAVES, make=make_full_rank_input)
    (slot,) = report["report"]["slots"]
    assert slot["not_compared"] == [], json.dumps(report, indent=1)[:4000]
    assert not any("not produced" in note for note in report["notes"])
    assert report["verdict"] == "MATCH", json.dumps(report, indent=1)[:4000]
    assert code == 0


@pytest.mark.slow
def test_rank_deficient_pca_is_never_matched_over_a_difference(tmp_path):
    # make_input's counts are rank 1, fewer than the 30 PCA components, so
    # ARPACK restarts at random and the replay's PCA may differ from the
    # session's in the last bits, or by chance agree. Either way the verdict
    # follows the values: MISMATCH when any differs, MATCH only when none does.
    _code, report = verify(tmp_path, THREE_SAVES)
    (slot,) = report["report"]["slots"]
    assert slot["not_compared"] == [], json.dumps(report, indent=1)[:4000]
    assert report["verdict"] in ("MATCH", "MISMATCH"), report
    assert (report["verdict"] == "MISMATCH") == bool(slot["diffs"]), report
