"""The numbers and warnings of a completed MCP run, read from a real session.

The assess and filter tools run on a small synthetic AnnData (the pattern of
``test_filter_reports_qc_flags.py``); the collector then reads the data manager
the way ``run_engine`` does. Nothing is taken from model text.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lobster.cli_internal import mcp_record
from lobster.cli_internal.mcp_record import WARNING_CODES, Turn, collect
from tests.unit.mcp.transcriptomics import quality_service

quality_service()

from lobster.agents.transcriptomics.shared_tools import (  # noqa: E402
    create_shared_tools,
)
from lobster.core.runtime.data_manager import DataManagerV2  # noqa: E402
from lobster.services.quality.preprocessing_service import (  # noqa: E402
    PreprocessingService,
)
from lobster.services.quality.quality_service import QualityService  # noqa: E402

USAGE = {  # what the run API puts in a result's token_usage for the run
    "latest_cost_usd": 0.01,  # the last call, not the run: never read
    "session_total_usd": 9.0,  # the session, not the run: never read
    "run_total_tokens": 1234,
    "run_cost_usd": 0.25,
    "run_cost_complete": True,
    "run_unpriced_invocations": 0,
    "run_missing_usage_invocations": 0,
}
PLOT = {"path": "/w/plots/f.json", "category": "plots", "size_bytes": 3}


@pytest.fixture
def counts():
    """128 cells x 2601 genes: cell 65 is high-ribosomal only; the rest pass QC."""
    x = np.zeros((128, 2601), dtype=np.float64)
    x[:, :240] = 1
    x[65, 1] = 1000  # RPS3: ribosomal share far above 50 %
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(128)])
    var = pd.DataFrame(
        {"source_label": ["synthetic"] * 2601},
        index=["MT-ND1", "RPS3", "ACTB"] + [f"GENE{i}" for i in range(2598)],
    )
    return ad.AnnData(sparse.csr_matrix(x), obs=obs, var=var)


def session(tmp_path, counts):
    dm = DataManagerV2(workspace_path=tmp_path / "workspace")
    dm.store_modality("input", counts.copy())
    tools = {
        t.name: t
        for t in create_shared_tools(dm, QualityService(), PreprocessingService())
    }
    return dm, SimpleNamespace(data_manager=dm), tools


def run_qc(dm, tools, assess=None, **filter_args):
    """Assess then filter on the real tools, as the recipe does; returns the
    turn and the result the run API would hand over."""
    turn = Turn.begin(dm)
    tools["assess_data_quality"].invoke({"modality_name": "input", **(assess or {})})
    tools["filter_and_normalize"].invoke(
        {"modality_name": "input_quality_assessed", **filter_args}
    )
    result = {"token_usage": dict(USAGE), "files_written": files_since(dm, turn)}
    return turn, result


def files_since(dm, turn, extra=()):
    return [*dm.files_written_since(turn.started_at), *extra]


def gather(client, turn, result, **overrides):
    args = {
        "exported": True,
        "open_question": False,
        **overrides,
    }
    return collect(client, turn, result, **args)


def codes(sections):
    return [w["code"] for w in sections["warnings"]]


def test_a_clean_run_has_numbers_and_no_warnings(tmp_path, counts):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50)
    result["files_written"].append(PLOT)
    sections = gather(client, turn, result)

    assert sections["warnings"] == []
    metrics = sections["metrics"]
    assert (metrics["cells_before"], metrics["cells_after"]) == (128, 127)
    assert (metrics["genes_before"], metrics["genes_after"]) == (2601, 240)
    qc = metrics["qc"]
    assert qc["thresholds_assessed"]["max_ribo_pct"] == 50.0
    assert qc["thresholds_assessed"]["min_genes"] == 200.0
    assert qc["thresholds_applied"] == {
        "min_genes_per_cell": 200,
        "max_genes_per_cell": 5000,
        "max_mito_percent": 20.0,
        "max_ribo_percent": 50.0,
    }
    assert qc["flagged_kept"] == 0
    assert metrics["tokens"] == {
        "total": 1234,
        "cost_usd": 0.25,
        "cost_complete": True,
        "usage_complete": True,
        "missing_usage_invocations": 0,
        "scope": "turn",
    }
    json.dumps(sections, allow_nan=False)


def test_modalities_name_their_stage_parent_shape_and_saved_file(tmp_path, counts):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50)
    modalities = {
        m["name"]: m for m in gather(client, turn, result)["modality_details"]
    }
    last = "input_quality_assessed_filtered_normalized"
    assert set(modalities) == {"input", "input_quality_assessed", last}
    assert modalities["input"]["parent"] is None
    assert modalities["input"]["stage"] == "raw"
    assert modalities[last]["parent"] == "input_quality_assessed"
    assert modalities[last]["stage"] == "filtered_normalized"
    assert (modalities[last]["n_obs"], modalities[last]["n_vars"]) == (127, 240)
    assert modalities[last]["file"].endswith(f"data/{last}.h5ad")
    assert "file" not in modalities["input"]  # only a saved modality has one


def test_the_steps_are_this_turn_only_with_short_values(tmp_path, counts):
    dm, client, tools = session(tmp_path, counts)
    dm.log_tool_usage("earlier", {"x": 1})  # before the turn
    turn, result = run_qc(dm, tools)
    dm.log_tool_usage("ask_user", {"question": "?"})  # a decision is no step
    dm.log_tool_usage("wordy", {"text": "x" * 500, "n": 3})
    steps = gather(client, turn, result)["steps"]
    assert [s["tool"] for s in steps] == [
        "assess_data_quality",
        "save_dataset",
        "filter_and_normalize",
        "wordy",
    ]
    assert steps[0]["agent"] == "transcriptomics_expert"
    assert steps[0]["parameters"]["modality_name"] == "input"
    wordy = steps[-1]["parameters"]["text"]
    assert len(wordy) == mcp_record.STEP_VALUE_MAX_CHARS and wordy.endswith("(cut)")
    assert all(isinstance(s["seq"], int) for s in steps)


def test_steps_are_cut_at_the_cap_and_the_total_is_kept(tmp_path, counts):
    dm, client, _tools = session(tmp_path, counts)
    turn = Turn.begin(dm)
    for i in range(mcp_record.MAX_STEPS + 1):
        dm.log_tool_usage("tool", {"i": i})
    sections = gather(client, turn, {})
    assert len(sections["steps"]) == mcp_record.MAX_STEPS
    assert sections["steps_total"] == mcp_record.MAX_STEPS + 1
    again = gather(client, Turn.begin(dm), {})
    assert again["steps"] == [] and "steps_total" not in again


def test_a_later_turn_reports_qc_only_when_it_did_qc(tmp_path, counts):
    dm, client, tools = session(tmp_path, counts)
    run_qc(dm, tools, max_ribo_percent=50)
    # A turn that only clusters: it did no assessment, so it reports none.
    clustering = Turn.begin(dm, 2)
    dm.log_tool_usage("cluster_cells", {"resolution": 0.5})
    later = gather(client, clustering, {})
    assert "qc" not in later["metrics"]
    assert "qc_flagged_cells_kept" not in codes(later)
    # A turn that assesses again reports its own thresholds.
    again = Turn.begin(dm, 2)
    tools["assess_data_quality"].invoke({"modality_name": "input", "max_mt_pct": 12})
    qc = gather(client, again, {})["metrics"]["qc"]
    assert qc["thresholds_assessed"]["max_mt_pct"] == 12.0
    # A first run that did no QC step reports none either.
    assert "qc" not in gather(client, Turn.begin(dm, 1), {})["metrics"]


# warnings: one per code --------------------------------------------------------


def test_qc_flagged_cells_kept_and_thresholds_differ_on_the_default_recipe(
    tmp_path, counts
):
    """Assess with its default ribosomal ceiling (50), filter without one."""
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools)
    result["files_written"].append(PLOT)
    sections = gather(client, turn, result)

    assert codes(sections) == ["qc_flagged_cells_kept", "qc_thresholds_differ"]
    flagged, differ = sections["warnings"]
    assert flagged["message"] == (
        "1 cells flagged by QC (qc_pass=False) are in the final data."
    )
    assert differ["message"] == (
        "max_ribo_pct: assessed at 50, the filter did not apply it."
    )
    qc = sections["metrics"]["qc"]
    assert qc["flagged_kept"] == 1
    assert qc["thresholds_applied"]["max_ribo_percent"] is None  # '' is null


def test_a_value_the_filter_changed_is_a_difference(tmp_path, counts):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50, max_mito_percent=10)
    result["files_written"].append(PLOT)
    sections = gather(client, turn, result)
    assert codes(sections) == ["qc_thresholds_differ"]
    assert sections["warnings"][0]["message"] == (
        "max_mt_pct: assessed at 20, the filter used 10."
    )


def test_a_flag_without_recorded_thresholds_is_not_recorded(tmp_path, counts):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50)
    result["files_written"].append(PLOT)
    for adata in dm.modalities.values():
        adata.uns.pop("qc_thresholds", None)
        adata.uns.pop("qc_threshold_names", None)
    sections = gather(client, turn, result)
    assert codes(sections) == ["qc_thresholds_not_recorded"]
    assert sections["metrics"]["qc"]["thresholds_assessed"] is None
    assert "no record of the thresholds" in sections["warnings"][0]["message"]


@pytest.mark.parametrize(
    "values", [[200.0, 5000.0], [float("nan")] * 5, [float("inf")] * 5]
)
def test_unreadable_thresholds_are_said_so_and_never_reach_the_json(
    tmp_path, counts, values
):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50)
    result["files_written"].append(PLOT)
    for adata in dm.modalities.values():
        if "qc_thresholds" in adata.uns:
            adata.uns["qc_thresholds"] = np.array(values)
    sections = gather(client, turn, result)
    assert codes(sections) == ["qc_thresholds_not_recorded"]
    assert "unreadable" in sections["warnings"][0]["message"]
    json.dumps(sections, allow_nan=False)


def test_the_thresholds_are_read_from_a_parent_when_the_newest_has_none(
    tmp_path, counts
):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50)
    result["files_written"].append(PLOT)
    leaf = dm.modalities["input_quality_assessed_filtered_normalized"]
    leaf.uns.pop("qc_thresholds")
    leaf.uns.pop("qc_threshold_names")
    sections = gather(client, turn, result)
    assert sections["warnings"] == []
    assert sections["metrics"]["qc"]["thresholds_assessed"]["max_ribo_pct"] == 50.0


@pytest.mark.parametrize(
    "usage, fragment",
    [
        ({"run_missing_usage_invocations": 2}, "usage_complete=False"),
        ({"run_cost_complete": False}, "cost_complete=False"),
    ],
)
def test_usage_incomplete(tmp_path, counts, usage, fragment):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50)
    result["files_written"].append(PLOT)
    result["token_usage"] = {**USAGE, **usage}
    sections = gather(client, turn, result)
    assert codes(sections) == ["usage_incomplete"]
    assert fragment in sections["warnings"][0]["message"]
    assert sections["metrics"]["tokens"]["missing_usage_invocations"] == (
        usage.get("run_missing_usage_invocations", 0)
    )


def test_a_usage_without_the_run_cost_keys_is_not_a_warning(tmp_path, counts):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50)
    result["files_written"].append(PLOT)
    result["token_usage"] = {"input": 10, "output": 5}  # an older engine
    sections = gather(client, turn, result)
    assert sections["warnings"] == []
    tokens = sections["metrics"]["tokens"]
    assert tokens["total"] is None and "missing_usage_invocations" not in tokens
    del result["token_usage"]
    assert "tokens" not in gather(client, turn, result)["metrics"]


def test_no_data_file_saved(tmp_path, counts):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50, save_result=False)
    result["files_written"].append(PLOT)
    assert [f for f in result["files_written"] if f["category"] == "data"] == []
    sections = gather(client, turn, result)
    assert codes(sections) == ["no_data_file_saved"]
    assert "file" not in sections["modality_details"][-1]


def test_a_question_run_is_not_told_to_draw_qc_plots(tmp_path, counts):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50)
    assert codes(gather(client, turn, result)) == []


def test_notebook_not_exported_and_open_question(tmp_path, counts):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50)
    result["files_written"].append(PLOT)
    sections = gather(client, turn, result, exported=False, open_question=True)
    assert codes(sections) == ["notebook_not_exported", "open_question"]


def test_a_section_that_cannot_be_read_is_one_warning_and_the_rest_stays(
    tmp_path, counts, monkeypatch
):
    dm, client, tools = session(tmp_path, counts)
    turn, result = run_qc(dm, tools, max_ribo_percent=50)
    result["files_written"].append(PLOT)

    def broken():
        raise RuntimeError("spilled")

    monkeypatch.setattr(dm, "list_modalities_with_lineage", broken)
    sections = gather(client, turn, result)
    assert codes(sections) == ["metrics_unavailable"]
    assert sections["warnings"][0]["message"] == "Could not read: modalities."
    assert (
        "modality_details" not in sections and "cells_before" not in sections["metrics"]
    )
    assert [s["tool"] for s in sections["steps"]][0] == "assess_data_quality"
    assert sections["metrics"]["tokens"]["total"] == 1234


def test_a_session_without_provenance_names_the_steps_and_the_files_error(
    tmp_path, counts
):
    turn = Turn.begin(SimpleNamespace())  # no tracker
    assert turn.baseline is None
    client = SimpleNamespace(
        data_manager=SimpleNamespace(
            list_modalities_with_lineage=lambda: [],
            modalities={},
        )
    )
    sections = gather(client, turn, {"files_written": None, "files_written_error": "x"})
    assert codes(sections) == ["metrics_unavailable"]
    assert sections["warnings"][0]["message"] == "Could not read: steps, files."
    assert "steps" not in sections and "modality_details" not in sections


def test_without_a_data_manager_the_run_is_still_described(tmp_path):
    sections = collect(
        SimpleNamespace(),
        Turn.begin(None),
        {},
        exported=True,
        open_question=False,
    )
    assert codes(sections) == ["metrics_unavailable"]
    assert sections["metrics"] == {}


def test_every_code_is_in_the_closed_list():
    assert len(WARNING_CODES) == len(set(WARNING_CODES)) == 12
    assert "plot_view_failed" in WARNING_CODES  # reserved for the plot views


def test_the_filter_prints_the_flags_it_carries(tmp_path, counts):
    dm, _client, tools = session(tmp_path, counts)
    tools["assess_data_quality"].invoke({"modality_name": "input"})
    response = tools["filter_and_normalize"].invoke(
        {"modality_name": "input_quality_assessed", "save_result": False}
    )
    quoted = "**QC flags carried into this result:**"
    paragraph = next(p for p in response.split("\n\n") if p.startswith(quoted))
    assert paragraph.endswith("they are not fitted to this dataset.")
