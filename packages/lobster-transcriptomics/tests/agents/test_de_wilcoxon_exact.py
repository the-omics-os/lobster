"""The bulk Wilcoxon p-value is exact at small groups, and tie-corrected above.

``ranksums`` used a normal approximation without tie correction: with 3 vs 3
samples a complete separation got p = 0.0495, below 0.05, where an exact test
cannot go below 0.10. The test is now an exact permutation Mann-Whitney (ties
handled) while the two groups can be split at most 20,000 ways, and the
tie-corrected normal approximation above that. The statistic is the
tie-corrected z, signed like the fold change, and the approximate p-value is
the one of that z (no continuity correction).
"""

import itertools
import math
import tracemalloc

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from lobster.services.analysis import mann_whitney as mann_whitney_module
from lobster.services.analysis.mann_whitney import mann_whitney

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
    else:  # 8 v 9: 24,310; 9 v 9: 48,620; no continuity correction, as the z
        expected = scipy_stats.mannwhitneyu(
            treat, ctrl, method="asymptotic", use_continuity=False, axis=0
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


@pytest.mark.parametrize("n", [6, 8])
def test_the_exact_test_holds_a_bounded_block_of_splits_in_memory(n):
    """Every split times every gene at once took 12 GB at 8 v 8 with 20,000 genes."""
    rng = np.random.default_rng(1)
    x = rng.negative_binomial(5, 0.1, size=(2 * n, 2_000)).astype(np.float64)

    tracemalloc.start()
    try:
        mann_whitney(x[n:], x[:n])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < 128 * 2**20, f"{peak / 2**20:.0f} MiB"


def test_every_gene_gets_its_own_p_value_across_blocks(monkeypatch):
    """Blocks of 3 genes over 10: each p equals the gene tested alone."""
    from lobster.services.analysis import mann_whitney as module

    rng = np.random.default_rng(2)
    x = rng.normal(size=(8, 10))
    alone = [mann_whitney(x[4:, [g]], x[:4, [g]])[1][0] for g in range(10)]
    monkeypatch.setattr(module, "_BLOCK", 3 * math.comb(8, 4))

    _, p = mann_whitney(x[4:], x[:4])

    assert p.tolist() == alone


def test_a_gene_with_a_missing_value_gets_no_p_value():
    x = np.arange(12, dtype=np.float64).reshape(6, 2)
    x[0, 1] = np.nan

    z, p = mann_whitney(x[3:], x[:3])

    assert p[0] == pytest.approx(0.1)
    assert np.isnan(p[1]) and np.isnan(z[1])


@pytest.mark.parametrize("n", [3, 12], ids=["exact", "normal"])
def test_a_gene_tied_in_every_sample_gets_no_p_value(n):
    """Every split gives the same U, so the samples tell the groups apart in no
    way: no p-value on either path (the normal path's z is 0 / 0)."""
    x = np.full((2 * n, 1), 5.0)

    z, p = mann_whitney(x[:n], x[n:])

    assert np.isnan(z[0]) and np.isnan(p[0])


def test_a_loop_over_proteins_of_one_group_size_builds_the_splits_once():
    """Proteomics tests one protein at a time: 5,000 proteins at 8 v 8 took 24 s
    when each call listed the 12,870 splits again."""
    mann_whitney_module._splits.cache_clear()
    x = np.random.default_rng(2).normal(size=(16, 50))

    for j in range(50):
        mann_whitney(x[:8, j : j + 1], x[8:, j : j + 1])

    assert mann_whitney_module._splits.cache_info().misses == 1


def test_a_group_of_one_against_thousands_builds_no_sample_by_sample_matrix():
    """1 v 4,999 has only 5,000 splits, but its split matrix is 5,000 x 5,000
    (200 MB), kept for the next call."""
    x = np.random.default_rng(3).normal(size=(5_000, 1))
    mann_whitney_module._splits.cache_clear()

    tracemalloc.start()
    try:
        mann_whitney(x[:1], x[1:])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < 32 * 2**20, f"{peak / 2**20:.0f} MiB"


@pytest.mark.parametrize("one_first", [True, False])
def test_the_split_matrices_kept_between_calls_hold_at_most_32_mib(one_first):
    """A Cloud worker keeps them for its lifetime: four 1 v ~2,045 matrices, each
    near 2**22 values, kept 128 MiB. The group of 1 may be the second one."""
    mann_whitney_module._splits.cache_clear()
    mann_whitney_module._members.cache_clear()
    rng = np.random.default_rng(5)

    tracemalloc.start()
    try:
        for n in range(2_045, 2_049):
            x = rng.normal(size=(n, 1))
            one, rest = x[:1], x[1:]
            mann_whitney(one, rest) if one_first else mann_whitney(rest, one)
        del x, one, rest
        kept = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()

    assert kept < 32 * 2**20, f"{kept / 2**20:.0f} MiB"


@pytest.mark.parametrize("n_x, n_y", [(2, 6), (6, 2), (3, 5), (5, 3), (4, 4)])
def test_the_exact_p_value_is_scipys_whichever_group_is_larger(n_x, n_y):
    """The splits are listed by the smaller group; a split of the larger one is
    the complement."""
    from scipy import stats

    x = np.random.default_rng(7).permutation(np.arange(8.0 * 40)).reshape(8, 40)
    mann_whitney_module._splits.cache_clear()
    mann_whitney_module._members.cache_clear()

    _, p = mann_whitney(x[:n_x], x[n_x:])

    expected = stats.mannwhitneyu(
        x[:n_x], x[n_x:], alternative="two-sided", method="exact", axis=0
    ).pvalue
    np.testing.assert_allclose(p, expected, rtol=1e-12)


@pytest.mark.parametrize("n, kept", [(1_024, 1), (1_025, 0)])
def test_a_split_matrix_is_kept_while_it_holds_2_20_values(n, kept):
    """1 v 1,023: 1,024 splits x 1,024 samples is 2**20 values (8 MiB), still
    kept; four of them hold 32 MiB."""
    mann_whitney_module._splits.cache_clear()
    x = np.random.default_rng(6).normal(size=(n, 1))

    mann_whitney(x[:1], x[1:])

    assert mann_whitney_module._splits.cache_info().currsize == kept


@pytest.mark.parametrize("n, exact", [(2_048, True), (2_049, False)])
def test_the_exact_test_runs_while_the_split_matrix_holds_2_22_values(
    monkeypatch, n, exact
):
    """1 v 2,047: 2,048 splits x 2,048 samples is 2**22 values, still exact."""
    called = []
    real = mann_whitney_module._exact_p
    monkeypatch.setattr(
        mann_whitney_module,
        "_exact_p",
        lambda *args: called.append(1) or real(*args),
    )
    x = np.random.default_rng(4).normal(size=(n, 1))

    mann_whitney(x[:1], x[1:])

    assert bool(called) is exact


@pytest.mark.parametrize("n_small, n_rest", [(1, 2_100), (2, 199), (2, 400)])
def test_a_group_of_one_or_two_gets_the_exact_p_value_beyond_the_split_matrix(
    n_small, n_rest
):
    """A gene detected only in the small group, the rest tied at 0. The normal
    approximation of the tie-corrected z gave 2v199 p = 2e-45 and 1v2100 p = 0,
    where no split is more extreme than the observed one: p = 2 / C(n, k)."""
    x = np.full((n_small, 1), 5.0)
    y = np.zeros((n_rest, 1))

    for a, b in ((x, y), (y, x)):
        _, p = mann_whitney(a, b)
        assert p[0] == pytest.approx(2 / math.comb(n_small + n_rest, n_small))


@pytest.mark.parametrize("n_small, n_rest", [(1, 2_100), (2, 250)])
def test_a_group_of_one_or_two_without_ties_gets_scipys_exact_p_value(n_small, n_rest):
    from scipy import stats

    rng = np.random.default_rng(8)
    x = rng.permutation(np.arange(float(n_small + n_rest) * 6)).reshape(-1, 6)
    x = x[: n_small + n_rest]

    _, p = mann_whitney(x[:n_small], x[n_small:])

    expected = stats.mannwhitneyu(
        x[:n_small], x[n_small:], alternative="two-sided", method="exact", axis=0
    ).pvalue
    np.testing.assert_allclose(p, expected, rtol=1e-9)


def _detected_in_four(n_ctrl, n_treat):
    """GENE0001 is 0 in every sample but four treat ones: 16 of 20 tie."""
    adata = _cohort(n_ctrl, n_treat)
    adata.X[:, 1] = 0.0
    adata.X[n_ctrl : n_ctrl + 4, 1] = [3.0, 4.0, 5.0, 6.0]
    return adata


def test_a_tie_heavy_gene_on_the_normal_path_is_flagged_and_counted(
    tmp_path, monkeypatch
):
    """10 v 10 has 184,756 splits: the normal approximation, which for this gene
    gives p = 0.031 where the exact p-value is 0.087."""
    out, stats, _ = _analysed(
        tmp_path, monkeypatch, _detected_in_four(10, 10), "wilcoxon"
    )

    flagged = out.var["approximate_with_ties"]
    assert list(flagged[flagged].index) == ["GENE0001"]
    assert stats["n_approximate_with_ties"] == 1


def test_an_exact_test_flags_no_gene(tmp_path, monkeypatch):
    out, stats, _ = _analysed(
        tmp_path, monkeypatch, _detected_in_four(8, 8), "wilcoxon"
    )

    assert not out.var["approximate_with_ties"].any()
    assert stats["n_approximate_with_ties"] == 0


@pytest.mark.parametrize(
    "n, line",
    [
        (0, ""),
        (
            1,
            "\n- Of the genes tested, 1 has many tied values (such as zeros): its "
            "Wilcoxon p-value is the normal approximation, which can be far smaller "
            "than the exact p-value; check it with DESeq2 on counts.",
        ),
        (
            3,
            "\n- Of the genes tested, 3 have many tied values (such as zeros): their "
            "Wilcoxon p-values are the normal approximation, which can be far "
            "smaller than the exact p-values; check them with DESeq2 on counts.",
        ),
    ],
)
def test_the_reply_names_the_tie_heavy_genes(n, line):
    from lobster.agents.transcriptomics.de_analysis_expert import _tied_text

    assert _tied_text({"n_approximate_with_ties": n}) == line
