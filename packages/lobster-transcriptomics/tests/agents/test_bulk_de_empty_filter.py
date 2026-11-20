"""Empty marker-style filters must not masquerade as a biological DE result.

The filters still run for wilcoxon / t_test on a pseudobulk modality; direct
bulk turns them off and deseq2 runs pyDESeq2 (test_bulk_de_pydeseq2.py).
"""

from importlib import import_module
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService
from lobster.testing.provenance_replay import factory_tools

MODULE = "lobster.agents.transcriptomics.de_analysis_expert"
METHODS = ["wilcoxon", "t_test"]


def counts(kind):
    if kind == "bulk_all_expressed":
        rng = np.random.default_rng(7)
        values = rng.negative_binomial(5, 0.05, size=(500, 6))
        values[:25, 3:] *= 6
        values = values.T.astype(np.float32)
    else:
        values = np.array(
            [[2, 1000], [50, 1010], [1000, 1020], [1, 1000], [0, 1010], [1, 1020]],
            dtype=np.float32,
        )
    return ad.AnnData(
        X=values,
        obs=pd.DataFrame(
            {"condition": ["ctrl"] * 3 + ["treat"] * 3},
            index=[f"sample_{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(values.shape[1])]),
    )


def invoke(tmp_path, monkeypatch, method, kind, save):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    original = counts(kind)
    original.uns["is_pseudobulk"] = True
    dm.store_modality("bulk", original)
    original = dm.get_modality("bulk").copy()
    model_tools = factory_tools(de_analysis_expert, MODULE, dm)
    actual_service = BulkRNASeqService.run_differential_expression_analysis
    calls = []

    def science(service, *args, **kwargs):
        output = actual_service(service, *args, **kwargs)
        calls.append(output)
        return output

    monkeypatch.setattr(
        BulkRNASeqService, "run_differential_expression_analysis", science
    )
    store = Mock(wraps=dm.store_modality)
    save_call = Mock(wraps=dm.save_modality)
    log = Mock(wraps=dm.log_tool_usage)
    monkeypatch.setattr(dm, "store_modality", store)
    monkeypatch.setattr(dm, "save_modality", save_call)
    monkeypatch.setattr(dm, "log_tool_usage", log)
    args = {
        "modality_name": "bulk",
        "groupby": "condition",
        "group1": "ctrl",
        "group2": "treat",
        "method": method,
        "save_result": save,
    }
    response = model_tools["run_differential_expression"].invoke(args)
    assert len(calls) == 1
    return dm, original, response, calls[0], store, save_call, log


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("save", [False, True])
def test_all_bulk_genes_filtered_is_clear_refusal_without_result_side_effects(
    tmp_path, monkeypatch, method, save
):
    dm, original, response, (_, stats, _), store, save_call, log = invoke(
        tmp_path, monkeypatch, method, "bulk_all_expressed", save
    )
    assert (
        stats["pre_filter_gene_count"] == 500 and stats["post_filter_gene_count"] == 0
    )
    assert "DE refused" in response
    assert "500 -> 0" in response
    for name, value in stats["filtering_params"].items():
        assert f"{name}={value}" in response
    assert "not a zero-DE biological result" in response
    assert "run_de_with_formula" in response
    assert "marker" in response
    assert "No raw counts layer" not in response
    assert "Analysis Complete" not in response and "Genes tested: 0" not in response
    store.assert_not_called()
    save_call.assert_not_called()
    log.assert_not_called()
    assert dm.list_modalities() == ["bulk"]
    assert not list(dm.data_dir.glob("*.h5ad"))
    np.testing.assert_array_equal(dm.get_modality("bulk").X, original.X)
    pd.testing.assert_frame_equal(dm.get_modality("bulk").obs, original.obs)
    pd.testing.assert_frame_equal(dm.get_modality("bulk").var, original.var)
    assert dm.get_modality("bulk").uns == original.uns


@pytest.mark.parametrize("method", METHODS)
def test_surviving_genes_with_no_significant_hits_still_complete(
    tmp_path, monkeypatch, method
):
    dm, original, response, (output, stats, _), store, save_call, log = invoke(
        tmp_path, monkeypatch, method, "survivor", False
    )
    assert output.n_vars == 1 and stats["post_filter_gene_count"] == 1
    assert stats["n_significant_genes"] == 0
    assert "DE refused" not in response
    assert "Analysis Complete" in response
    assert "Genes tested: 1" in response and "Significant genes" in response
    store.assert_called_once()
    save_call.assert_not_called()
    log.assert_called_once()
    assert dm.get_modality("bulk_de_ctrl_vs_treat").n_vars == 1
    assert "de_current_results_key" in dm.get_modality("bulk_de_ctrl_vs_treat").uns
    np.testing.assert_array_equal(dm.get_modality("bulk").X, original.X)


def test_real_sources_are_the_active_checkout():
    root = Path(__file__).resolve().parents[4]
    assert (
        Path(import_module(MODULE).__file__).resolve()
        == root
        / "packages/lobster-transcriptomics/lobster/agents/transcriptomics/de_analysis_expert.py"
    )
    assert (
        Path(import_module(BulkRNASeqService.__module__).__file__).resolve()
        == root / "lobster/services/analysis/bulk_rnaseq_service.py"
    )
    assert (
        Path(import_module(DataManagerV2.__module__).__file__).resolve()
        == root / "lobster/core/runtime/data_manager.py"
    )
