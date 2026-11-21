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
