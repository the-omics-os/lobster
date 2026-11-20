"""Simple bulk DE on an imported (not pseudobulk) modality runs pyDESeq2 on its counts.

The design is rqea's seeded L55/L69 cohort: 12 samples (6 ctrl, 6 treat, two
batches), 2,000 genes, the first 50 planted 4x up in treat. Before, both simple
tools ran a t-test behind single-cell marker filters that removed every gene.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools, replay_ir

DE = "lobster.agents.transcriptomics.de_analysis_expert"
PLANTED = {f"GENE{i:04d}" for i in range(50)}
SAMPLES = [f"C{i}" for i in range(1, 7)] + [f"T{i}" for i in range(1, 7)]
GENES = [f"GENE{i:04d}" for i in range(2000)]
OBS = pd.DataFrame(
    {"condition": ["ctrl"] * 6 + ["treat"] * 6, "batch": ["b1", "b2"] * 6},
    index=SAMPLES,
)
TOOLS = ["run_bulk_de_direct", "run_differential_expression"]


def _counts():
    """Genes x samples, as rqea's e2e.py generates them."""
    rng = np.random.default_rng(7)
    mu = rng.gamma(2.0, 50.0, size=len(GENES))[:, None] * np.ones(len(SAMPLES))
    treat = np.array([s.startswith("T") for s in SAMPLES])
    mu[:50, treat] *= 4
    return pd.DataFrame(
        rng.negative_binomial(5, 5 / (5 + mu)), index=GENES, columns=SAMPLES
    )


def _bulk():
    adata = ad.AnnData(
        X=_counts().T.to_numpy().astype(np.float32),
        obs=OBS.copy(),
        var=pd.DataFrame(index=GENES),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


def _dm(tmp_path, monkeypatch, adata=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)  # BulkRNASeqService writes ./bulk_results
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    if adata is not None:
        dm.store_modality("bulk", adata)
    return dm


def _simple_de(dm, tool, **extra):
    args = {"modality_name": "bulk", "group1": "ctrl", "group2": "treat"}
    args["group_key" if tool == "run_bulk_de_direct" else "groupby"] = "condition"
    tools = factory_tools(de_analysis_expert, DE, dm)
    return tools[tool].invoke({**args, "save_result": False, **extra})


def _stored(dm):
    adata = dm.get_modality("bulk")
    return pd.DataFrame(adata.uns[adata.uns["de_current_results_key"]]["results_df"])


def _assert_planted_found(results):
    significant = set(results.index[results["padj"] < 0.05])
    found = significant & PLANTED
    assert len(found) >= 45, (len(found), len(significant))
    assert len(significant - found) <= 10


@pytest.mark.parametrize("tool", TOOLS)
def test_simple_bulk_de_tests_every_gene_and_finds_the_planted_ones(
    tmp_path, monkeypatch, tool
):
    dm = _dm(tmp_path, monkeypatch, _bulk())
    reply = _simple_de(dm, tool)
    assert "Genes tested: 2,000" in reply, reply
    assert "pyDESeq2" in reply
    assert "pseudobulk" not in reply.lower()
    _assert_planted_found(_stored(dm))


@pytest.mark.parametrize("tool", TOOLS)
def test_simple_de_equals_the_formula_tool_with_condition_alone(
    tmp_path, monkeypatch, tool
):
    simple = _dm(tmp_path / "simple", monkeypatch, _bulk())
    _simple_de(simple, tool)
    formula = _dm(tmp_path / "formula", monkeypatch, _bulk())
    factory_tools(de_analysis_expert, DE, formula)["run_de_with_formula"].invoke(
        {
            "pseudobulk_modality": "bulk",
            "formula": "~condition",
            "contrast": ["condition", "treat", "ctrl"],
            "save_result": False,
        }
    )
    expected = _stored(formula)
    actual = _stored(simple)
    for column in ("log2FoldChange", "pvalue", "padj"):
        np.testing.assert_allclose(
            actual[column].to_numpy(), expected[column].to_numpy(), rtol=1e-6
        )


@pytest.mark.parametrize("tool", TOOLS)
def test_the_exported_cells_reproduce_the_stored_results(tmp_path, monkeypatch, tool):
    dm = _dm(tmp_path, monkeypatch, _bulk())
    original = dm.get_modality("bulk").copy()
    _simple_de(dm, tool)
    activity = [a for a in dm.provenance.activities if a["type"] == tool][-1]
    replayed = replay_ir(AnalysisStep.from_dict(activity["ir"]), adata=original)
    pd.testing.assert_frame_equal(pd.DataFrame(replayed["results_df"]), _stored(dm))
    # The cells store the result in the object the save cell writes, as the tool
    # stored it: replay compares that object with the session's saved file.
    stored, written = dm.get_modality("bulk").uns, replayed["adata"].uns
    key = stored["de_current_results_key"]
    assert written["de_current_results_key"] == key
    pd.testing.assert_frame_equal(written[key]["results_df"], stored[key]["results_df"])
    assert written[key]["parameters"] == stored[key]["parameters"]


@pytest.mark.parametrize("tool", TOOLS)
def test_the_saved_file_carries_the_modality_name(tmp_path, monkeypatch, tool):
    """Replay pairs a saved file with its modality by name: <modality>.h5ad."""
    dm = _dm(tmp_path, monkeypatch, _bulk())
    reply = _simple_de(dm, tool, save_result=True)
    assert "Complete" in reply and "error" not in reply.lower(), reply
    assert "bulk.h5ad" in reply
    assert sorted(p.name for p in dm.data_dir.glob("*.h5ad")) == ["bulk.h5ad"]
    assert (dm.exports_dir / "bulk_de_results.csv").is_file()
    saved = ad.read_h5ad(dm.data_dir / "bulk.h5ad")
    assert saved.uns["de_current_results_key"] in saved.uns


@pytest.mark.parametrize("tool", TOOLS)
def test_normalized_bulk_without_counts_is_refused(tmp_path, monkeypatch, tool):
    adata = _bulk()
    del adata.layers["counts"]
    adata.X = np.log1p(adata.X / adata.X.sum(axis=1, keepdims=True) * 1e4)
    dm = _dm(tmp_path, monkeypatch, adata)
    reply = _simple_de(dm, tool)
    assert "Raw counts required" in reply, reply
    assert "pseudobulk" not in reply.lower()
    assert "de_current_results_key" not in dm.get_modality("bulk").uns


@pytest.mark.parametrize("tool", TOOLS)
def test_the_reply_names_a_batch_column_the_simple_design_left_out(
    tmp_path, monkeypatch, tool
):
    dm = _dm(tmp_path, monkeypatch, _bulk())
    reply = _simple_de(dm, tool)
    assert (
        "Not modelled: batch. For a design with it, use run_de_with_formula" in reply
    ), reply


@pytest.mark.parametrize("method", ["wilcoxon", "t_test"])
def test_rank_and_t_tests_on_bulk_keep_every_gene(tmp_path, monkeypatch, method):
    dm = _dm(tmp_path, monkeypatch, _bulk())
    reply = _simple_de(dm, "run_differential_expression", method=method)
    assert "Genes tested: 2,000" in reply, reply


@pytest.mark.parametrize(
    "reader",
    ["filter_de_results", "run_gsea_analysis", "extract_and_export_de_results"],
)
def test_downstream_readers_use_a_direct_bulk_result(tmp_path, monkeypatch, reader):
    dm = _dm(tmp_path, monkeypatch, _bulk())
    _simple_de(dm, "run_bulk_de_direct")
    expected = _stored(dm)
    args = {"modality_name": "bulk"}
    captured = {}
    if reader == "run_gsea_analysis":
        from lobster.services.analysis.pathway_enrichment_service import (
            PathwayEnrichmentService,
        )

        def capture(self, ranked_genes, **kwargs):
            captured["ranked"] = ranked_genes.copy()
            raise RuntimeError("offline routing boundary")

        monkeypatch.setattr(
            PathwayEnrichmentService, "gene_set_enrichment_analysis", capture
        )
    elif reader == "extract_and_export_de_results":
        args.update(output_path=str(dm.workspace_path / "de.csv"), shrink_lfc=False)
    else:
        args.update(save_result=False)
    reply = factory_tools(de_analysis_expert, DE, dm)[reader].invoke(args)
    if reader == "run_gsea_analysis":
        assert "offline routing boundary" in reply, reply
        assert set(captured["ranked"]["gene"]) == set(expected.index)
    elif reader == "filter_de_results":
        assert "DE Results Filtered" in reply, reply
    else:
        exported = pd.read_csv(dm.workspace_path / "de.csv", index_col=0)
        assert set(exported.index) == set(
            expected.index[expected["padj"] < 0.05]
        ), reply
