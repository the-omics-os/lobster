"""Wilcoxon / t_test on a pseudobulk modality test every gene.

The single-cell marker filters (out-group expression cap, min fraction, min fold
change) once removed every gene expressed across the samples; they are off on
every bulk path now (test_bulk_rank_no_marker_filter.py), so a result is never
empty and the reply counts every gene tested.
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
def test_genes_expressed_in_every_sample_are_all_tested(
    tmp_path, monkeypatch, method, save
):
    dm, original, response, (output, stats, _), store, save_call, log = invoke(
        tmp_path, monkeypatch, method, "bulk_all_expressed", save
    )
    assert stats["pre_filter_gene_count"] == stats["post_filter_gene_count"] == 500
    assert output.n_vars == 500
    assert "DE refused" not in response
    assert "Analysis Complete" in response and "Genes tested: 500" in response
    store.assert_called_once()
    assert save_call.call_count == (1 if save else 0)
    de_logs = [
        c for c in log.call_args_list if c.kwargs.get("tool_name") != "save_dataset"
    ]
    assert len(de_logs) == 1  # a save also logs save_dataset
    assert dm.get_modality("bulk_de_ctrl_vs_treat").n_vars == 500
    np.testing.assert_array_equal(dm.get_modality("bulk").X, original.X)


@pytest.mark.parametrize("method", METHODS)
def test_genes_with_no_significant_hits_still_complete(tmp_path, monkeypatch, method):
    dm, original, response, (output, stats, _), store, save_call, log = invoke(
        tmp_path, monkeypatch, method, "survivor", False
    )
    assert output.n_vars == 2 and stats["post_filter_gene_count"] == 2
    assert stats["n_significant_genes"] == 0
    assert "DE refused" not in response
    assert "Analysis Complete" in response
    assert "Genes tested: 2" in response and "Significant genes" in response
    store.assert_called_once()
    save_call.assert_not_called()
    log.assert_called_once()
    assert dm.get_modality("bulk_de_ctrl_vs_treat").n_vars == 2
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
