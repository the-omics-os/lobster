"""The bulk Wilcoxon p-value is exact at small groups, and tie-corrected above.

``ranksums`` used a normal approximation without tie correction: with 3 vs 3
samples a complete separation got p = 0.0495, below 0.05, where an exact test
cannot go below 0.10. The test is now an exact permutation Mann-Whitney (ties
handled) while the two groups can be split at most 20,000 ways, and the
tie-corrected normal approximation above that. The statistic is the
tie-corrected z, signed like the fold change.
"""

import itertools
import math

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from .test_de_zero_within_group_variance import _analysed


def _cohort(n_ctrl, n_treat, n_genes=40, seed=0):
    """Counts with ties; GENE0000 is lower in every ctrl sample than any treat one."""
    rng = np.random.default_rng(seed)
    x = rng.negative_binomial(5, 0.1, size=(n_ctrl + n_treat, n_genes)).astype(
        np.float64
    )
    x[:n_ctrl, 0] = np.arange(n_ctrl) + 10
    x[n_ctrl:, 0] = np.arange(n_treat) + 100
    obs = pd.DataFrame(
        {"condition": ["ctrl"] * n_ctrl + ["treat"] * n_treat},
        index=[f"S{i}" for i in range(n_ctrl + n_treat)],
    )
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    return ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=genes))


def _pvalues(tmp_path, monkeypatch, adata):
    out, _, _ = _analysed(tmp_path, monkeypatch, adata, "wilcoxon")
    return out.var["pvalue"]


@pytest.mark.parametrize("n, expected", [(3, 0.100), (4, 2 / 70)])
def test_a_complete_separation_gets_the_exact_p_value(
    tmp_path, monkeypatch, n, expected
):
    p = _pvalues(tmp_path, monkeypatch, _cohort(n, n))

    assert p["GENE0000"] == pytest.approx(expected, rel=1e-9)


def test_a_tied_gene_gets_the_exact_p_value_with_its_ties(tmp_path, monkeypatch):
    adata = _cohort(6, 6)
    p = _pvalues(tmp_path, monkeypatch, adata)

    for gene in p.index[1:6]:
        ranks = scipy_stats.rankdata(adata[:, gene].X.ravel())
        observed = abs(ranks[6:].sum() - 39)
        sums = [ranks[list(c)].sum() for c in itertools.combinations(range(12), 6)]
        exact = np.mean([abs(s - 39) >= observed - 1e-9 for s in sums])
        assert p[gene] == pytest.approx(exact, rel=1e-9), gene


@pytest.mark.parametrize("n_ctrl, n_treat", [(8, 8), (8, 9), (9, 9)])
def test_the_exact_test_runs_while_the_groups_split_20000_ways_or_fewer(
    tmp_path, monkeypatch, n_ctrl, n_treat
):
    adata = _cohort(n_ctrl, n_treat)
    p = _pvalues(tmp_path, monkeypatch, adata)

    x = adata.X
    treat, ctrl = x[n_ctrl:], x[:n_ctrl]
    if math.comb(n_ctrl + n_treat, n_ctrl) <= 20_000:  # 8 v 8: 12,870
        expected = scipy_stats.permutation_test(
            (treat, ctrl),
            lambda a, b, axis: scipy_stats.mannwhitneyu(a, b, axis=axis).statistic,
            permutation_type="independent",
            n_resamples=np.inf,
            vectorized=True,
            axis=0,
        ).pvalue
    else:  # 8 v 9: 24,310; 9 v 9: 48,620
        expected = scipy_stats.mannwhitneyu(
            treat, ctrl, method="asymptotic", axis=0
        ).pvalue
    reported = list(p.index)
    columns = [adata.var_names.get_loc(g) for g in reported]
    np.testing.assert_allclose(p.to_numpy(), expected[columns], rtol=1e-9)


def test_the_statistic_is_the_tie_corrected_z_signed_like_the_fold_change(
    tmp_path, monkeypatch
):
    adata = _cohort(6, 6)
    out, _, _ = _analysed(tmp_path, monkeypatch, adata, "wilcoxon")

    for gene in ["GENE0000", "GENE0003"]:
        x = adata[:, gene].X.ravel()
        ranks = scipy_stats.rankdata(x)
        u = ranks[6:].sum() - 6 * 7 / 2
        ties = np.unique(x, return_counts=True)[1]
        sigma = math.sqrt(36 / 12 * (13 - (ties**3 - ties).sum() / (12 * 11)))
        assert out.var.loc[gene, "stat"] == pytest.approx((u - 18) / sigma)
    assert out.var.loc["GENE0000", "stat"] > 0
