"""The DE result tools read a wilcoxon or t-test result, as they read DESeq2's.

The bulk service stored its table as ``results_df.to_dict()``, while every reader
(filter_de_results, export_de_results, run_gsea_analysis and their exported
code) expects ``{"results_df": DataFrame}``, the shape the DESeq2 path stores. So
after a wilcoxon or t-test DE each of them refused: "Invalid DE results format".
"""

import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.testing.provenance_replay import factory_tools

from .test_bulk_rank_no_marker_filter import DE, _c2
from .test_de_genes_tested_line import _run

METHODS = ["wilcoxon", "t_test", "deseq2"]


def _analysed(tmp_path, monkeypatch, method):
    dm, reply = _run(tmp_path, monkeypatch, _c2(), method, pseudobulk=True)
    assert "Complete" in reply, reply
    (result,) = [
        m
        for m in dm.list_modalities()
        if "de_current_results_key" in dm.get_modality(m).uns
    ]
    return dm, result, factory_tools(de_analysis_expert, DE, dm)


@pytest.mark.parametrize("method", METHODS)
def test_the_stored_result_holds_its_table(tmp_path, monkeypatch, method):
    dm, result, _ = _analysed(tmp_path, monkeypatch, method)

    uns = dm.get_modality(result).uns
    table = uns[uns["de_current_results_key"]]["results_df"]
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 2000


@pytest.mark.parametrize("method", METHODS)
def test_filter_de_results_reads_it(tmp_path, monkeypatch, method):
    _, result, tools = _analysed(tmp_path, monkeypatch, method)

    reply = tools["filter_de_results"].invoke(
        {"modality_name": result, "padj_threshold": 0.05, "lfc_threshold": 0.0}
    )

    assert "Invalid DE results format" not in reply, reply
    assert "Error" not in reply, reply


@pytest.mark.parametrize("method", METHODS)
def test_export_de_results_reads_it(tmp_path, monkeypatch, method):
    dm, result, tools = _analysed(tmp_path, monkeypatch, method)

    reply = tools["export_de_results"].invoke(
        {"modality_name": result, "output_path": "de.csv"}
    )

    assert "Invalid DE results format" not in reply, reply
    assert len(pd.read_csv(dm.exports_dir / "de.csv", index_col=0)) == 2000


@pytest.mark.parametrize("method", METHODS)
def test_run_gsea_analysis_reads_it(tmp_path, monkeypatch, method):
    _, result, tools = _analysed(tmp_path, monkeypatch, method)
    ranked = []

    def gsea(self, adata, ranked_genes, **_):
        ranked.append(ranked_genes)
        raise RuntimeError("stop after ranking")

    monkeypatch.setattr(
        "lobster.services.analysis.pathway_enrichment_service."
        "PathwayEnrichmentService.gene_set_enrichment_analysis",
        gsea,
    )

    reply = tools["run_gsea_analysis"].invoke({"modality_name": result})

    assert "Invalid DE results format" not in reply, reply
    assert len(ranked) == 1 and len(ranked[0]) == 2000, reply
