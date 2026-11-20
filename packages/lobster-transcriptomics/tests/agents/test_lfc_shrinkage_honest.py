"""LFC shrinkage is reported as applied only when pyDESeq2 actually ran it.

pyDESeq2 0.6 names a coefficient ``condition[T.treat]``; a coefficient built as
``condition_treat_vs_ctrl`` made every shrink fail silently while the reply and
the stats said "Yes". Shrinkage is applied when the contrast is one model
coefficient (its second level is the reference); otherwise it is not applied,
with the reason, and the fold changes keep their sign.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService
from lobster.testing.provenance_replay import factory_tools

pytest.importorskip("pydeseq2")

DE = "lobster.agents.transcriptomics.de_analysis_expert"
SAMPLES = [f"C{i}" for i in range(1, 5)] + [f"T{i}" for i in range(1, 5)]
GENES = [f"GENE{i:04d}" for i in range(300)]
FORWARD = ["condition", "treat", "ctrl"]  # ctrl is the reference level
REVERSED = ["condition", "ctrl", "treat"]


def _bulk():
    rng = np.random.default_rng(11)
    mu = rng.gamma(2.0, 50.0, size=len(GENES))[None, :] * np.ones((len(SAMPLES), 1))
    mu[4:, :30] *= 4
    counts = rng.negative_binomial(5, 5 / (5 + mu)).astype(np.float32)
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"condition": ["ctrl"] * 4 + ["treat"] * 4}, index=SAMPLES),
        var=pd.DataFrame(index=GENES),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


def _run(tmp_path, monkeypatch, contrast, shrink):
    monkeypatch.chdir(tmp_path)
    return BulkRNASeqService(
        results_dir=tmp_path / "bulk_results"
    ).run_pydeseq2_from_pseudobulk(
        pseudobulk_adata=_bulk(),
        formula="~condition",
        contrast=contrast,
        count_layer="counts",
        alpha=0.05,
        shrink_lfc=shrink,
        n_cpus=1,
    )


def test_shrinkage_runs_and_changes_the_fold_changes(tmp_path, monkeypatch):
    raw, raw_stats = _run(tmp_path, monkeypatch, FORWARD, shrink=False)
    shrunk, stats = _run(tmp_path, monkeypatch, FORWARD, shrink=True)
    assert stats["lfc_shrinkage_applied"] is True, stats.get("lfc_shrinkage_note")
    assert raw_stats["lfc_shrinkage_applied"] is False
    lfc, raw_lfc = shrunk["log2FoldChange"], raw["log2FoldChange"]
    assert not np.allclose(lfc, raw_lfc)
    # apeGLM pulls the fold changes towards zero and leaves the p-values alone
    assert np.abs(lfc).median() < np.abs(raw_lfc).median()
    np.testing.assert_allclose(shrunk["pvalue"], raw["pvalue"])


def test_a_contrast_that_is_not_one_coefficient_is_not_shrunk(tmp_path, monkeypatch):
    raw, _ = _run(tmp_path, monkeypatch, REVERSED, shrink=False)
    results, stats = _run(tmp_path, monkeypatch, REVERSED, shrink=True)
    assert stats["lfc_shrinkage_applied"] is False
    assert "reference" in stats["lfc_shrinkage_note"], stats
    np.testing.assert_allclose(results["log2FoldChange"], raw["log2FoldChange"])
    assert (results["log2FoldChange"].iloc[:30] < 0).mean() > 0.9  # ctrl vs treat


def _tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("bulk", _bulk())
    return dm, factory_tools(de_analysis_expert, DE, dm)


def test_the_reply_says_shrinkage_was_applied(tmp_path, monkeypatch):
    _, tools = _tools(tmp_path, monkeypatch)
    reply = tools["run_bulk_de_direct"].invoke(
        {"modality_name": "bulk", "group_key": "condition", "group1": "ctrl"}
        | {"group2": "treat", "save_result": False}
    )
    assert "- LFC shrinkage: applied" in reply, reply


def test_the_reply_and_the_export_say_why_shrinkage_was_not_applied(
    tmp_path, monkeypatch
):
    dm, tools = _tools(tmp_path, monkeypatch)
    reply = tools["run_differential_expression"].invoke(
        {"modality_name": "bulk", "groupby": "condition", "group1": "treat"}
        | {"group2": "ctrl", "method": "deseq2", "save_result": False}
    )
    assert "- LFC shrinkage: not applied:" in reply, reply
    assert "reference" in reply
    exported = tools["extract_and_export_de_results"].invoke(
        {"modality_name": "bulk", "output_path": str(dm.workspace_path / "de.csv")}
    )
    assert "**LFC Shrinkage:** Not applied" in exported, exported
    assert "already applied" not in exported


def test_the_ir_template_builds_the_coefficient_as_the_service_does(tmp_path):
    ir = BulkRNASeqService(results_dir=tmp_path / "bulk_results")._create_pydeseq2_ir(
        formula="~condition", contrast=FORWARD, shrink_lfc=True
    )
    code = ir.render()
    assert "_vs_" not in code
    assert "contrast_vector" in code and "LFC.columns" in code
