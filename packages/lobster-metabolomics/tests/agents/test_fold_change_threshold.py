"""run_metabolomics_statistics applies fold_change_threshold: a feature is up
when log2FC > log2(threshold) and down when log2FC < -log2(threshold). It used
to count at |log2FC| > 1 (2-fold) whatever the threshold. The record carries
the threshold, so the exported notebook computes the same fold changes.

The service's log2FC is log2(mean(group) / mean(reference)) on raw intensities
(log_space=False) and mean(group) - mean(reference) on log2 data (log_space=True,
what the tool uses). The fixture scales every reference sample by a known fold,
so both definitions give exactly log2(fold).
"""

from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.metabolomics.shared_tools import create_shared_tools
from lobster.services.analysis.metabolomics_analysis_service import (
    MetabolomicsAnalysisError,
    MetabolomicsAnalysisService,
)
from lobster.testing.provenance_replay import RecordingDataManager, replay_ir

FOLDS = {
    "U12": 1.2,
    "U14": 1.4,
    "U16": 1.6,
    "U25": 2.5,
    "D12": 1 / 1.2,
    "D14": 1 / 1.4,
    "D16": 1 / 1.6,
    "D25": 1 / 2.5,
    "N": 1.0,
}
# (threshold, up, down), known by construction
EXPECTED = [
    (1.5, {"U16", "U25"}, {"D16", "D25"}),
    (2.0, {"U25"}, {"D25"}),
]


def _raw():
    reference = np.outer([0.8, 1.0, 1.2], 100 + 10 * np.arange(len(FOLDS)))
    x = np.vstack([reference, reference * np.array(list(FOLDS.values()))])
    obs = pd.DataFrame(
        {"condition": ["ctrl"] * 3 + ["treat"] * 3}, index=[f"S{i}" for i in range(6)]
    )
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=list(FOLDS)))


def _log2():
    adata = _raw()
    adata.X = np.log2(adata.X)
    return adata


def _expected(adata, threshold, log_space):
    """Up and down features from numpy on the group means, not the service."""
    x = np.asarray(adata.X, dtype=float)
    treat = (adata.obs["condition"] == "treat").to_numpy()
    a, b = x[treat].mean(axis=0), x[~treat].mean(axis=0)
    log2_fc = a - b if log_space else np.log2(a / b)
    names = np.asarray(adata.var_names)
    cut = np.log2(threshold)
    return set(names[log2_fc > cut]), set(names[log2_fc < -cut])


@pytest.mark.parametrize("threshold,up,down", EXPECTED)
def test_the_fixture_has_the_known_answer(threshold, up, down):
    assert _expected(_raw(), threshold, False) == (up, down)
    assert _expected(_log2(), threshold, True) == (up, down)


@pytest.mark.parametrize("threshold,up,down", EXPECTED)
@pytest.mark.parametrize("log_space", [False, True])
def test_the_service_counts_at_the_threshold(threshold, up, down, log_space):
    adata = _log2() if log_space else _raw()

    _, stats, ir = MetabolomicsAnalysisService().calculate_fold_changes(
        adata,
        group_column="condition",
        log_space=log_space,
        fold_change_threshold=threshold,
    )

    assert (stats["n_upregulated"], stats["n_downregulated"]) == (len(up), len(down))
    assert ir.parameters["fold_change_threshold"] == threshold


def test_the_service_default_stays_two_fold():
    _, stats, _ = MetabolomicsAnalysisService().calculate_fold_changes(
        _raw(), group_column="condition", log_space=False
    )

    assert (stats["n_upregulated"], stats["n_downregulated"]) == (1, 1)


def _tool(tmp_path, adata):
    dm = RecordingDataManager({"met": adata}, workspace_path=tmp_path / "ws")
    tools = create_shared_tools(
        dm, MagicMock(), MagicMock(), MetabolomicsAnalysisService(), MagicMock()
    )
    return dm, {t.name: t for t in tools}["run_metabolomics_statistics"]


@pytest.mark.parametrize("threshold,up,down", EXPECTED)
def test_the_tool_applies_and_records_the_threshold(tmp_path, threshold, up, down):
    adata = _log2()
    dm, tool = _tool(tmp_path, adata)

    reply = tool.invoke(
        {
            "modality_name": "met",
            "group_column": "condition",
            "fold_change_threshold": threshold,
        }
    )

    cut = np.log2(threshold)
    assert f"- Upregulated (log2FC > {cut:.3g}): {len(up)}\n" in reply, reply
    assert f"- Downregulated (log2FC < -{cut:.3g}): {len(down)}\n" in reply, reply
    assert "not applied" not in reply
    (call,) = dm.calls_for("run_metabolomics_statistics")
    assert call.parameters["fold_change_threshold"] == threshold
    assert call.ir.parameters["fold_change_threshold"] == threshold

    # The notebook step computes the fold changes the session stored.
    scope = replay_ir(call.ir, adata=adata.copy())
    replayed = scope["adata_fc"]
    stored = dm.get_modality("met_statistics")
    np.testing.assert_allclose(
        replayed.var["log2_fold_change"], stored.var["log2_fold_change"]
    )
    assert "p_value" in replayed.var
    names = np.asarray(replayed.var_names)
    log2_fc = replayed.var["log2_fold_change"].to_numpy()
    assert (set(names[log2_fc > cut]), set(names[log2_fc < -cut])) == (up, down)
    assert (scope["stats"]["n_upregulated"], scope["stats"]["n_downregulated"]) == (
        len(up),
        len(down),
    )


@pytest.mark.parametrize("threshold", [1.0, 0.5, -2.0, float("nan"), float("inf")])
def test_a_threshold_that_is_not_a_fold_change_is_refused(tmp_path, threshold):
    dm, tool = _tool(tmp_path, _log2())

    reply = tool.invoke(
        {
            "modality_name": "met",
            "group_column": "condition",
            "fold_change_threshold": threshold,
        }
    )

    assert "fold_change_threshold must be a finite number greater than 1" in reply
    assert dm.calls == []
    assert dm.list_modalities() == ["met"]
    with pytest.raises(MetabolomicsAnalysisError, match="greater than 1"):
        MetabolomicsAnalysisService().calculate_fold_changes(
            _raw(), group_column="condition", fold_change_threshold=threshold
        )
