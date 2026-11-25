"""A DE reply never shows a filter as a threshold when the filter did not run.

filter_de_results applies its base-mean filter only when the table has a
base-mean column, and extract_and_export_de_results its padj and |LFC| filters
only when the table has those columns; the replies listed every threshold
either way.
"""

import anndata as ad
import numpy as np
import pandas as pd

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.testing.provenance_replay import RecordingDataManager, factory_tools

MODULE = "lobster.agents.transcriptomics.de_analysis_expert"
KEY = "de_results_treat_vs_ctrl"


def _tools(tmp_path, monkeypatch, table):
    monkeypatch.chdir(tmp_path)
    adata = ad.AnnData(np.zeros((len(table), 2)))
    adata.uns[KEY] = {"results_df": table}
    adata.uns["de_current_results_key"] = KEY
    dm = RecordingDataManager({"de": adata}, workspace_path=tmp_path / "ws")
    dm.exports_dir = dm.workspace_path / "exports"
    return factory_tools(de_analysis_expert, MODULE, dm)


def _table(**columns):
    index = ["GENE1", "GENE2", "GENE3"]
    full = {
        "padj": [0.01, 0.01, 0.2],
        "log2FoldChange": [2.0, -2.0, 0.1],
        "baseMean": [5.0, 50.0, 50.0],
    }
    return pd.DataFrame(
        {k: v for k, v in full.items() if columns.get(k, True)}, index=index
    )


def test_the_base_mean_filter_is_named_unapplied_without_a_base_mean_column(
    tmp_path, monkeypatch
):
    tools = _tools(tmp_path, monkeypatch, _table(baseMean=False))

    reply = tools["filter_de_results"].invoke({"modality_name": "de"})

    assert "- Base mean > 10.0 (not applied: no base-mean column)" in reply
    assert "- Passing filters: 2 " in reply


def test_an_applied_base_mean_filter_is_shown_as_a_threshold(tmp_path, monkeypatch):
    tools = _tools(tmp_path, monkeypatch, _table())

    reply = tools["filter_de_results"].invoke({"modality_name": "de"})

    assert "- Base mean > 10.0\n" in reply
    assert "- Passing filters: 1 " in reply


def _exported(tmp_path, monkeypatch, table, **arguments):
    tools = _tools(tmp_path, monkeypatch, table)
    return tools["extract_and_export_de_results"].invoke(
        {"modality_name": "de", "shrink_lfc": False, **arguments}
    )


def test_the_padj_filter_is_named_unapplied_without_a_padj_column(
    tmp_path, monkeypatch
):
    reply = _exported(tmp_path, monkeypatch, _table(padj=False), padj_threshold=0.05)

    assert "- padj threshold: 0.05 (not applied: no padj column)" in reply
    assert "- Genes after filtering: 3" in reply


def test_the_lfc_filter_is_named_unapplied_without_a_fold_change_column(
    tmp_path, monkeypatch
):
    reply = _exported(
        tmp_path, monkeypatch, _table(log2FoldChange=False), lfc_threshold=1.0
    )

    assert "- |LFC| threshold: 1.0 (not applied: no log2FoldChange column)" in reply
    assert "- Genes after filtering: 2" in reply


def test_applied_export_filters_are_shown_as_thresholds(tmp_path, monkeypatch):
    reply = _exported(
        tmp_path, monkeypatch, _table(), padj_threshold=0.05, lfc_threshold=1.0
    )

    assert "- padj threshold: 0.05\n" in reply
    assert "- |LFC| threshold: 1.0\n" in reply
    assert "- Genes after filtering: 2" in reply


def _enrichment_reply(tmp_path, monkeypatch, padj, lfc=1.0, genes=None):
    from lobster.core.provenance.analysis_ir import AnalysisStep
    from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService

    received = []

    def enrichment(self, gene_list, analysis_type="GO", **_):
        received.append(list(gene_list))
        stats = {
            "n_genes_input": len(gene_list),
            "enrichment_database": "GO",
            "n_terms_total": 0,
            "n_significant_terms": 0,
            "top_terms": [],
        }
        ir = AnalysisStep(
            operation="enrichment",
            tool_name="run_pathway_enrichment",
            description="",
            library="gseapy",
            code_template="",
            imports=[],
            parameters={},
            parameter_schema={},
            input_entities=[],
            output_entities=[],
        )
        return pd.DataFrame(), stats, ir

    monkeypatch.setattr(BulkRNASeqService, "run_pathway_enrichment", enrichment)
    if genes is None:
        genes = [f"GENE{i}" for i in range(len(padj))]
    table = pd.DataFrame({"padj": padj, "log2FoldChange": lfc}, index=genes)
    tools = _tools(tmp_path, monkeypatch, table)
    reply = tools["run_pathway_enrichment"].invoke(
        {"gene_list": [], "modality_name": "de"}
    )
    return reply, received[0]


def test_enrichment_from_a_de_result_names_the_threshold_it_took_genes_at(
    tmp_path, monkeypatch
):
    reply, genes = _enrichment_reply(tmp_path, monkeypatch, [0.01, 0.04, 0.2])

    assert genes == ["GENE0", "GENE1"]
    assert f"- Genes from DE result '{KEY}': padj < 0.05, 2 genes\n" in reply, reply


def test_enrichment_names_the_genes_it_left_out_beyond_500(tmp_path, monkeypatch):
    reply, genes = _enrichment_reply(tmp_path, monkeypatch, [0.01] * 600)

    assert len(genes) == 500
    assert (
        f"- Genes from DE result '{KEY}': padj < 0.05, 600 genes (the 500 with the "
        "smallest adjusted p-value were used)\n" in reply
    ), reply


def test_enrichment_beyond_500_takes_the_smallest_padj_ties_by_larger_lfc(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(7)
    n = 700
    # Few distinct values, so padj ties and |LFC| ties both reach the 500th slot
    padj = rng.choice([1e-8, 1e-4, 0.001, 0.01, 0.03], n)
    padj[rng.choice(n, 60, replace=False)] = np.nan
    padj[rng.choice(n, 40, replace=False)] = 0.5
    lfc = rng.choice([-3.0, -1.0, 0.5, 1.0, 2.0, 3.0], n)
    # var order is a shuffle, so table order says nothing about padj
    genes = np.array([f"G{i:04d}" for i in rng.permutation(n)])

    significant = (padj < 0.05) & ~np.isnan(padj)
    s_padj, s_lfc, s_genes = padj[significant], lfc[significant], genes[significant]
    order = np.lexsort((s_genes, -np.abs(s_lfc), s_padj))
    expected = s_genes[order][:500].tolist()
    # The cut falls inside a padj tie, so the |LFC| rule decides some slots
    assert s_padj[order][499] == s_padj[order][500]
    assert significant.sum() > 500

    reply, received = _enrichment_reply(
        tmp_path, monkeypatch, padj, lfc=lfc, genes=list(genes)
    )

    assert set(received) == set(expected)
    assert received == expected
    assert (
        f"padj < 0.05, {significant.sum():,} genes (the 500 with the smallest "
        "adjusted p-value were used)" in reply
    ), reply


def test_enrichment_of_a_given_gene_list_names_no_de_threshold(tmp_path, monkeypatch):
    from lobster.core.provenance.analysis_ir import AnalysisStep
    from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService

    ir = AnalysisStep(
        operation="enrichment",
        tool_name="run_pathway_enrichment",
        description="",
        library="gseapy",
        code_template="",
        imports=[],
        parameters={},
        parameter_schema={},
        input_entities=[],
        output_entities=[],
    )
    stats = {
        "n_genes_input": 2,
        "enrichment_database": "GO",
        "n_terms_total": 0,
        "n_significant_terms": 0,
        "top_terms": [],
    }
    monkeypatch.setattr(
        BulkRNASeqService,
        "run_pathway_enrichment",
        lambda self, gene_list, analysis_type="GO", **_: (pd.DataFrame(), stats, ir),
    )
    tools = _tools(tmp_path, monkeypatch, _table())

    reply = tools["run_pathway_enrichment"].invoke({"gene_list": ["A", "B"]})

    assert "Genes from DE result" not in reply
