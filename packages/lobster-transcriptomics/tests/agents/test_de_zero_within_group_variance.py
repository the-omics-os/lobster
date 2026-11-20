"""A gene with the same value within each group, and a different value between
them, no longer tops a t-test result.

Its within-group variance is zero, so the t-test gives t = ±inf and p = 0: the
gene ranked first with an adjusted p-value of 0. It is now kept with its fold
change, its p-value and adjusted p-value are not a number, it is flagged in a
column of its own, it is left out of the correction, and the reply names it.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService

from .test_bulk_rank_no_marker_filter import _c2
from .test_de_genes_tested_line import _run
from .test_de_without_pvalue import FILTERS, _analysed_and_exported

LINE = (
    "{n} have the same value within each group: the t-test gives {them} no "
    "usable p-value; check them with DESeq2 on counts."
)


def _split(adata, n):
    """The last ``n`` genes hold 10 in every ctrl sample and 40 in every treat one."""
    treat = (adata.obs["condition"] == "treat").to_numpy()
    adata.X[:, adata.n_vars - n :] = np.where(treat, 40.0, 10.0)[:, None]
    return adata


def _analysed(tmp_path, monkeypatch, adata, method="t_test"):
    monkeypatch.chdir(tmp_path)
    return BulkRNASeqService(
        results_dir=str(tmp_path / "results")
    ).run_differential_expression_analysis(
        adata=adata,
        groupby="condition",
        group1="ctrl",
        group2="treat",
        method=method,
        **FILTERS,
    )


def test_a_gene_with_no_variance_within_either_group_is_kept_and_flagged(
    tmp_path, monkeypatch
):
    out, stats, _ = _analysed(tmp_path, monkeypatch, _split(_c2(), 3))

    split = list(out.var_names[-3:])
    assert out.n_vars == 2000
    flagged = out.var["zero_within_group_variance"]
    assert list(flagged[flagged].index) == split
    rows = out.var.loc[split]
    np.testing.assert_allclose(rows["log2FoldChange"], np.log2(41 / 11), rtol=1e-6)
    assert rows["pvalue"].isna().all()
    assert rows["padj"].isna().all()
    assert not set(split) & set(stats["significant_genes_list"])
    assert stats["n_zero_within_group_variance"] == 3


def test_the_other_genes_are_corrected_without_the_flagged_ones(tmp_path, monkeypatch):
    out, _, _ = _analysed(tmp_path, monkeypatch, _split(_c2(), 3))

    others = out.var[~out.var["zero_within_group_variance"]]
    np.testing.assert_allclose(
        others["padj"], multipletests(others["pvalue"], method="fdr_bh")[1]
    )


def test_a_gene_with_the_same_value_in_every_sample_is_not_flagged(
    tmp_path, monkeypatch
):
    adata = _c2()
    adata.X[:, -2:] = 5.0  # equal means: no fold change and no p-value

    out, stats, _ = _analysed(tmp_path, monkeypatch, adata)

    assert stats["n_zero_within_group_variance"] == 0
    assert not out.var["zero_within_group_variance"].any()


def test_the_exported_code_keeps_and_flags_them_too(tmp_path, monkeypatch):
    out, exported = _analysed_and_exported(
        tmp_path, monkeypatch, _split(_c2(), 3), "t_test"
    )

    split = list(out.var_names[-3:])
    assert exported.loc[split, "zero_within_group_variance"].all()
    assert exported.loc[split, "padj"].isna().all()
    pd.testing.assert_series_equal(
        exported.loc[out.var_names, "padj"], out.var["padj"], check_names=False
    )


@pytest.mark.parametrize(
    "n, text", [(3, ("3 genes", "them")), (1, ("1 gene", "it"))], ids=["3", "1"]
)
def test_the_reply_names_them(tmp_path, monkeypatch, n, text):
    _, reply = _run(tmp_path, monkeypatch, _split(_c2(), n), "t_test", False)

    expected = LINE.format(n=text[0], them=text[1])
    if n == 1:
        expected = expected.replace(" have ", " has ").replace("check them", "check it")
    assert expected in reply, reply


@pytest.mark.parametrize("method", ["t_test", "wilcoxon"])
def test_the_reply_says_nothing_when_no_gene_is_split_so(tmp_path, monkeypatch, method):
    _, reply = _run(tmp_path, monkeypatch, _c2(), method, False)

    assert "the same value within each group" not in reply
