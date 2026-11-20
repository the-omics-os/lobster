"""The bulk DE statistic has the sign of the fold change; no standard error is
made up.

The fold change is log2 of group2 over group1, but the t-test statistic was
group1 against group2, so the two had opposite signs. The Wilcoxon and
deseq2_like statistics were copies of the fold change. Every method wrote an
lfcSE of 1.0 that no test estimated; it is now not a number. Under Wilcoxon a
gene with one value in every sample got p = 1 from an all-tied rank test (its
z is 0 / 0); like the t-test, it now has no p-value.
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from .test_bulk_rank_no_marker_filter import PLANTED, _c2
from .test_de_zero_within_group_variance import _analysed

METHODS = ["t_test", "wilcoxon", "deseq2_like"]


def _groups(adata, gene):
    x = adata[:, gene].X.ravel().astype(np.float64)
    treat = (adata.obs["condition"] == "treat").to_numpy()
    return x[~treat], x[treat]


@pytest.mark.parametrize("method", METHODS)
def test_the_planted_genes_up_in_treat_have_a_positive_statistic(
    tmp_path, monkeypatch, method
):
    out, _, _ = _analysed(tmp_path, monkeypatch, _c2(), method)

    planted = out.var.loc[sorted(PLANTED)]
    assert (planted["log2FoldChange"] > 0).all()
    assert (planted["stat"] > 0).all()


@pytest.mark.parametrize("method", ["t_test", "deseq2_like"])
def test_the_t_statistic_is_treat_against_ctrl(tmp_path, monkeypatch, method):
    adata = _c2()
    out, _, _ = _analysed(tmp_path, monkeypatch, adata, method)

    ctrl, treat = _groups(adata, "GENE0100")
    expected = scipy_stats.ttest_ind(treat, ctrl).statistic
    np.testing.assert_allclose(out.var.loc["GENE0100", "stat"], expected, rtol=1e-6)


def test_the_wilcoxon_statistic_is_the_rank_sum_z(tmp_path, monkeypatch):
    adata = _c2()
    out, _, _ = _analysed(tmp_path, monkeypatch, adata, "wilcoxon")

    ctrl, treat = _groups(adata, "GENE0100")
    expected = scipy_stats.ranksums(treat, ctrl).statistic
    np.testing.assert_allclose(out.var.loc["GENE0100", "stat"], expected, rtol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_no_standard_error_is_made_up_and_every_gene_is_reported(
    tmp_path, monkeypatch, method
):
    out, _, _ = _analysed(tmp_path, monkeypatch, _c2(), method)

    assert out.n_vars == 2000
    assert out.var["lfcSE"].isna().all()


def test_a_wilcoxon_gene_with_one_value_everywhere_has_no_p_value(
    tmp_path, monkeypatch
):
    adata = _c2()
    adata.X[:, -1] = 5.0

    out, stats, _ = _analysed(tmp_path, monkeypatch, adata, "wilcoxon")
    _, t_test, _ = _analysed(tmp_path, monkeypatch, adata, "t_test")

    assert "GENE1999" not in out.var_names
    assert stats["n_genes_without_pvalue"] == 1
    assert stats["n_genes_without_pvalue"] == t_test["n_genes_without_pvalue"]


def test_a_wilcoxon_gene_split_between_the_groups_keeps_its_p_value(
    tmp_path, monkeypatch
):
    adata = _c2()
    treat = (adata.obs["condition"] == "treat").to_numpy()
    adata.X[:, -1] = np.where(treat, 40.0, 10.0)

    out, _, _ = _analysed(tmp_path, monkeypatch, adata, "wilcoxon")

    assert np.isfinite(out.var.loc["GENE1999", "pvalue"])
