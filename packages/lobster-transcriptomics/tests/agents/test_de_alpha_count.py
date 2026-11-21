"""The significant-gene count in a two-group DE reply is taken at the alpha the
reply names.

A Wilcoxon or t-test reply said "Significant genes (padj < 0.01): N", but N and
the up/down counts were taken at 0.05: alpha never reached the service. DESeq2
already passed it on; its rows pin that.
"""

import re

import pandas as pd
import pytest

from lobster.core.runtime.data_manager import DataManagerV2

from .test_bulk_de_pydeseq2 import _bulk, _simple_de
from .test_bulk_rank_no_marker_filter import _c2, _number


def _run(tmp_path, monkeypatch, adata, tool, method, alpha):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)  # BulkRNASeqService writes ./bulk_results
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("bulk", adata)
    reply = _simple_de(dm, tool, method=method, alpha=alpha)
    found = re.search(r"New modality created\*\*: '([^']+)'", reply)
    result = dm.get_modality(found.group(1) if found else "bulk")
    stored = result.uns[result.uns["de_current_results_key"]]
    table = pd.DataFrame(stored["results_df"] if "results_df" in stored else stored)
    return reply, table


# 6 v 6 Wilcoxon has no padj below 0.19 here, so none below 0.01 or 0.05
@pytest.mark.parametrize(
    "method, alpha", [("t_test", 0.01), ("t_test", 0.2), ("wilcoxon", 0.2)]
)
def test_a_rank_or_t_test_reply_counts_at_its_alpha(
    tmp_path, monkeypatch, method, alpha
):
    reply, table = _run(
        tmp_path, monkeypatch, _c2(), "run_differential_expression", method, alpha
    )

    significant = table["padj"] < alpha
    assert significant.sum() != (table["padj"] < 0.05).sum()  # alpha matters here
    assert f"(padj < {alpha})" in reply
    assert _number(reply, "Significant genes") == significant.sum()
    up = significant & (table["log2FoldChange"] > 0)
    down = significant & (table["log2FoldChange"] < 0)
    assert _number(reply, "Upregulated in treat") == up.sum()
    assert _number(reply, "Downregulated in treat") == down.sum()


@pytest.mark.parametrize("alpha", [0.01, 0.2])
@pytest.mark.parametrize("tool", ["run_differential_expression", "run_bulk_de_direct"])
def test_a_deseq2_reply_counts_at_its_alpha(tmp_path, monkeypatch, tool, alpha):
    """run_bulk_de_direct runs DESeq2 only."""
    reply, table = _run(tmp_path, monkeypatch, _bulk(), tool, "deseq2", alpha)

    assert (table["padj"] < alpha).sum() != (table["padj"] < 0.05).sum()

    significant = table["padj"] < alpha
    assert f"alpha={alpha}" in reply
    assert _number(reply, "Significant genes") == significant.sum()
    assert (
        _number(reply, "Upregulated")
        == (significant & (table["log2FoldChange"] > 0)).sum()
    )


def test_a_gene_whose_padj_is_the_alpha_is_not_counted(tmp_path, monkeypatch):
    _, first = _run(
        tmp_path / "a", monkeypatch, _c2(), "run_differential_expression", "t_test", 0.2
    )
    alpha = float(first["padj"].sort_values().iloc[10])

    reply, table = _run(
        tmp_path / "b",
        monkeypatch,
        _c2(),
        "run_differential_expression",
        "t_test",
        alpha,
    )

    assert (table["padj"] == alpha).any()
    assert _number(reply, "Significant genes") == (table["padj"] < alpha).sum()
