"""Wilcoxon and t-test DE on bulk samples apply no single-cell marker filter.

rqea's c2 cohort: 12 samples (6 ctrl, 6 treat), 2,000 genes, the first 50 planted
4x up in treat. On pseudobulk, and on bulk after prepare_de_design, the service's
marker post-filter (out-group expression <= 50% of group2 samples) kept 2 genes:
the reply said "Genes tested: 2, Significant: 0" while the stored table had all
2,000 and, for the t-test, 18 significant. Every bulk path now tests every gene,
and the reply counts are the stored table's. Single-cell markers are unchanged.
"""

import re

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

DE = "lobster.agents.transcriptomics.de_analysis_expert"
PLANTED = {f"GENE{i:04d}" for i in range(50)}
SAMPLES = [f"C{i}" for i in range(1, 7)] + [f"T{i}" for i in range(1, 7)]
GENES = [f"GENE{i:04d}" for i in range(2000)]


def _c2():
    """rqea's fixture.py case 2: the second table drawn from the seed-7 generator."""
    rng = np.random.default_rng(7)
    treat = np.array([s.startswith("T") for s in SAMPLES])
    for _ in range(2):
        mu = rng.gamma(2.0, 50.0, size=len(GENES))[:, None] * np.ones(len(SAMPLES))
        mu[:50, treat] *= 4
        x = rng.negative_binomial(5, 5 / (5 + mu))
    obs = pd.DataFrame(
        {"condition": ["ctrl"] * 6 + ["treat"] * 6, "batch": ["b1", "b2"] * 6},
        index=SAMPLES,
    )
    adata = ad.AnnData(X=x.T.astype(np.float32), obs=obs, var=pd.DataFrame(index=GENES))
    adata.layers["counts"] = adata.X.copy()
    return adata


def _number(reply, label):
    found = re.search(rf"{re.escape(label)}[^:]*: ([\d,]+)", reply)
    assert found, (label, reply)
    return int(found.group(1).replace(",", ""))


@pytest.mark.parametrize("method", ["wilcoxon", "t_test"])
@pytest.mark.parametrize("path", ["plain", "after_design", "pseudobulk"])
def test_bulk_rank_and_t_test_report_every_gene_tested(
    tmp_path, monkeypatch, path, method
):
    monkeypatch.chdir(tmp_path)  # BulkRNASeqService writes ./bulk_results
    adata = _c2()
    if path == "pseudobulk":
        adata.uns["is_pseudobulk"] = True
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("bulk", adata)
    tools = factory_tools(de_analysis_expert, DE, dm)
    if path == "after_design":
        tools["prepare_de_design"].invoke(
            {
                "modality_name": "bulk",
                "formula": "~condition",
                "contrast": ["condition", "treat", "ctrl"],
            }
        )
    reply = tools["run_differential_expression"].invoke(
        {
            "modality_name": "bulk",
            "groupby": "condition",
            "group1": "ctrl",
            "group2": "treat",
            "method": method,
            "save_result": False,
        }
    )
    result_name = re.search(r"New modality created\*\*: '([^']+)'", reply).group(1)
    result = dm.get_modality(result_name)
    stored = result.uns[result.uns["de_current_results_key"]]
    table = pd.DataFrame(stored["results_df"] if "results_df" in stored else stored)

    assert _number(reply, "Genes tested") == 2000 == result.n_vars == len(table)
    significant = set(table.index[table["padj"] < 0.05])
    assert _number(reply, "Significant genes") == len(significant)
    assert PLANTED <= set(result.var_names)
    if method == "t_test":
        assert len(significant & PLANTED) >= 17, len(significant & PLANTED)


def test_single_cell_markers_keep_their_filters(tmp_path):
    """find_marker_genes (cells, clusters) still reports its specificity filters."""
    from lobster.agents.transcriptomics.transcriptomics_expert import (
        transcriptomics_expert,
    )

    rng = np.random.default_rng(0)
    x = rng.poisson(1.0, size=(120, 60)).astype(np.float32)
    x[:60, :10] += 6  # cluster A markers
    adata = ad.AnnData(
        X=np.log1p(x),
        obs=pd.DataFrame(
            {"cluster": pd.Categorical(["A"] * 60 + ["B"] * 60)},
            index=[f"cell{i}" for i in range(120)],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(60)]),
    )
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("cells", adata)
    tools = factory_tools(
        transcriptomics_expert,
        "lobster.agents.transcriptomics.transcriptomics_expert",
        dm,
    )
    reply = tools["find_marker_genes"].invoke(
        {"modality_name": "cells", "groupby": "cluster", "save_result": False}
    )
    assert "Positive log2FC > 1.5" in reply, reply
    assert "In-group expression fraction > 0.25" in reply
    assert "Out-group expression fraction < 0.5" in reply


PREFILTER = "no gene has a value above 1.0 in at least 2 samples"
FOLD_CHANGE = "none has a finite fold change"


def _no_gene_case(case):
    """12 samples x 2,000 genes that leave no gene to test, for one cause."""
    adata = _c2()
    rng = np.random.default_rng(3)
    if case == "fractional":  # every value <= 1: the expression pre-filter
        adata.X = adata.X / 1e4
    elif case == "all_zero":
        adata.X = np.zeros_like(adata.X)
    else:  # signed: ctrl ~N(-3), treat ~N(+3); passes the pre-filter, log2 of a
        # negative mean is not a number
        adata.X = np.vstack(
            [rng.normal(-3, 1, (6, 2000)), rng.normal(3, 1, (6, 2000))]
        ).astype(np.float32)
    adata.layers["counts"] = adata.X.copy()
    return adata


def _invoke(tmp_path, monkeypatch, adata, method, save, pseudobulk):
    monkeypatch.chdir(tmp_path)
    if pseudobulk:
        adata.uns["is_pseudobulk"] = True
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("bulk", adata)
    tools = factory_tools(de_analysis_expert, DE, dm)
    reply = tools["run_differential_expression"].invoke(
        {
            "modality_name": "bulk",
            "groupby": "condition",
            "group1": "ctrl",
            "group2": "treat",
            "method": method,
            "save_result": save,
        }
    )
    return dm, reply


@pytest.mark.parametrize("pseudobulk", [False, True], ids=["plain", "pseudobulk"])
@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("method", ["wilcoxon", "t_test"])
@pytest.mark.parametrize("case", ["fractional", "all_zero", "signed"])
def test_no_gene_tested_is_refused_with_its_cause(
    tmp_path, monkeypatch, case, method, save, pseudobulk
):
    """No gene left to test is no DE result: refused, naming the step that
    dropped them (the expression pre-filter, or a fold change that is not a
    number), with nothing stored or saved."""
    dm, reply = _invoke(
        tmp_path, monkeypatch, _no_gene_case(case), method, save, pseudobulk
    )
    assert reply.startswith("**DE refused: no gene was tested.**"), reply
    assert "Of 2,000 genes" in reply
    if case == "signed":
        assert FOLD_CHANGE in reply and PREFILTER not in reply, reply
    else:
        assert PREFILTER in reply and FOLD_CHANGE not in reply, reply
    assert "Complete" not in reply
    assert dm.list_modalities() == ["bulk"]
    assert not list(dm.data_dir.glob("*.h5ad"))


@pytest.mark.parametrize("pseudobulk", [False, True], ids=["plain", "pseudobulk"])
@pytest.mark.parametrize("method", ["wilcoxon", "t_test"])
def test_a_gene_off_in_group1_and_on_in_group2_is_tested(
    tmp_path, monkeypatch, method, pseudobulk
):
    """Genes at 0 in every ctrl sample and high in treat are the clearest DE;
    no filter on group1 expression may drop them."""
    rng = np.random.default_rng(11)
    x = rng.negative_binomial(5, 5 / 105, size=(12, 20)).astype(np.float32)
    x[:6, :10] = 0
    x[6:, :10] = rng.poisson(200, size=(6, 10))
    adata = ad.AnnData(
        X=x,
        obs=pd.DataFrame({"condition": ["ctrl"] * 6 + ["treat"] * 6}, index=SAMPLES),
        var=pd.DataFrame(index=[f"GENE{i:04d}" for i in range(20)]),
    )
    adata.layers["counts"] = adata.X.copy()
    dm, reply = _invoke(tmp_path, monkeypatch, adata, method, False, pseudobulk)
    result_name = re.search(r"New modality created\*\*: '([^']+)'", reply).group(1)
    result = dm.get_modality(result_name)
    table = pd.DataFrame(result.uns[result.uns["de_current_results_key"]])
    off_on = {f"GENE{i:04d}" for i in range(10)}
    assert _number(reply, "Genes tested") == 20 == result.n_vars
    assert off_on <= set(result.var_names)
    assert set(table.index[table["padj"] < 0.05]) == off_on
    assert _number(reply, "Significant genes") == 10
