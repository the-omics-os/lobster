"""A protein with one value in each group, different between them, no longer tops
a t-test result.

Its within-group variance is zero, so a t-test (Student, Welch, or the one an
ANOVA or Kruskal-Wallis pairwise comparison reduces to) gives t = inf and p = 0:
the protein was significant with an adjusted p-value of 0. It is now kept with
its fold change, its p-value is not a number, it is flagged, the correction
leaves it out, and the reply names it.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.proteomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.analysis.proteomics_differential_service import (
    ProteomicsDifferentialService,
)
from lobster.testing.provenance_replay import factory_tools

DE = "lobster.agents.proteomics.de_analysis_expert"
T_TESTS = ["t_test", "welch_t_test", "anova"]
# the advice after "...gives it no p-value"; limma_like borrows variance from
# 10 proteins tested or more
ADVICE = (
    "; check {it} with limma_like, which borrows variance from the other "
    "proteins, or with Mann-Whitney\n"
)
FEW_ADVICE = (
    "; check {it} with Mann-Whitney (fewer than 10 proteins tested: limma_like "
    "has no variance to borrow)\n"
)
LIMMA_FEW_ADVICE = (
    " (fewer than 10 proteins tested: no variance to borrow); check {it} with "
    "Mann-Whitney\n"
)


def _proteins(split):
    """12 samples (6 a, 6 b), 50 proteins, P0-P4 higher in b; with ``split``, P49
    holds 5.0 in every a sample and 9.0 in every b one."""
    rng = np.random.default_rng(0)
    x = rng.normal(10, 1, (12, 50))
    x[6:, :5] += 4
    if split:
        x[:6, 49], x[6:, 49] = 5.0, 9.0
    obs = pd.DataFrame(
        {"group": ["a"] * 6 + ["b"] * 6}, index=[f"s{i}" for i in range(12)]
    )
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=[f"P{i}" for i in range(50)]))


def _run(adata, method):
    out, stats, _ = ProteomicsDifferentialService().perform_differential_expression(
        adata, group_column="group", test_method=method
    )
    de = out.uns["differential_expression"]
    return {r["protein"]: r for r in de["all_results"]}, de, stats


@pytest.mark.parametrize("method", T_TESTS)
def test_a_split_protein_is_kept_flagged_and_left_out_of_the_correction(method):
    with_split, de, stats = _run(_proteins(split=True), method)
    without, _, stats_without = _run(_proteins(split=False)[:, :49].copy(), method)

    split = with_split.pop("P49")
    assert np.isnan(split["p_value"]) and np.isnan(split["p_adjusted"])
    assert split["zero_within_group_variance"] is True
    assert np.isfinite(split["log2_fold_change"]) and split["log2_fold_change"] != 0
    assert "P49" not in {r["protein"] for r in de["significant_results"]}
    for protein, result in without.items():
        assert with_split[protein]["p_adjusted"] == pytest.approx(result["p_adjusted"])
        assert with_split[protein]["zero_within_group_variance"] is False
    assert stats["n_zero_within_group_variance"] == 1
    assert stats["n_tests_without_pvalue"] == 0  # tested; the line above explains it
    assert stats["n_significant_proteins"] == stats_without["n_significant_proteins"]


def _reply(tmp_path, adata, method):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("prot", adata)
    return factory_tools(de_analysis_expert, DE, dm)[
        "find_differential_proteins"
    ].invoke(
        {
            "modality_name": "prot",
            "group_column": "group",
            "platform_type": "mass_spec",
            "method": method,
        }
    )


def _few(adata):
    """8 proteins: too few for limma_like to borrow variance (it needs 10)."""
    return adata[:, 42:].copy()


def test_limma_like_without_a_prior_flags_a_split_protein_and_drops_a_flat_one():
    adata = _few(_proteins(split=True))
    adata.X[:, -2] = 3.0  # one value in every sample

    results, _, stats = _run(adata, "limma_like")

    assert np.isnan(results["P49"]["p_value"])
    assert results["P49"]["zero_within_group_variance"] is True
    assert np.isnan(results["P48"]["p_value"])
    assert np.isnan(results["P48"]["statistic"])  # 0 / 0
    assert results["P48"]["zero_within_group_variance"] is False
    assert stats["n_zero_within_group_variance"] == 1
    assert stats["n_tests_without_pvalue"] == 1


def test_limma_like_with_a_prior_gives_a_split_protein_its_p_value():
    results, _, stats = _run(_proteins(split=True), "limma_like")

    assert np.isfinite(results["P49"]["p_value"])
    assert results["P49"]["zero_within_group_variance"] is False
    assert stats["n_zero_within_group_variance"] == 0


@pytest.mark.parametrize(
    "method, adata, test, advice",
    [
        ("welch_t_test", _proteins(split=True), "Welch t-test", ADVICE),
        ("t_test", _few(_proteins(split=True)), "t-test", FEW_ADVICE),
        (
            "limma_like",
            _few(_proteins(split=True)),
            "limma-like test",
            LIMMA_FEW_ADVICE,
        ),
    ],
    ids=["welch", "t_test_few_proteins", "limma_without_prior"],
)
def test_the_reply_names_the_test_that_ran_and_advises_for_it(
    tmp_path, method, adata, test, advice
):
    reply = _reply(tmp_path, adata, method)

    line = (
        "- Of the proteins tested, 1 has the same value in every sample of each "
        f"group (different between the groups): the {test} gives it no p-value"
    )
    assert line + advice.format(it="it") in reply, reply


def test_mann_whitney_gives_a_split_protein_its_p_value():
    with_split, _, stats = _run(_proteins(split=True), "mann_whitney")

    assert np.isfinite(with_split["P49"]["p_value"])
    assert stats["n_zero_within_group_variance"] == 0


@pytest.mark.parametrize("value", [5.0, 5.1], ids=["integer", "non_integer"])
def test_one_value_in_both_groups_is_not_split(value):
    adata = _proteins(split=False)
    adata = adata[:10].copy()  # 6 a and 4 b: float means of one value can differ
    adata.X[:, 49] = value

    with_flat, _, stats = _run(adata, "t_test")

    assert with_flat["P49"]["zero_within_group_variance"] is False
    assert stats["n_zero_within_group_variance"] == 0


@pytest.mark.parametrize("split", [True, False])
def test_the_reply_names_a_split_protein(tmp_path, split):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("prot", _proteins(split))
    reply = factory_tools(de_analysis_expert, DE, dm)[
        "find_differential_proteins"
    ].invoke(
        {
            "modality_name": "prot",
            "group_column": "group",
            "platform_type": "affinity",
            "method": "t_test",
        }
    )

    line = (
        "- Of the proteins tested, 1 has the same value in every sample of each "
        "group (different between the groups): the t-test gives it no p-value"
    ) + ADVICE.format(it="it")
    assert (line in reply) is split, reply
    assert ("the same value in every sample of each group" in reply) is split


def test_a_split_protein_has_no_statistic():
    with_split, _, _ = _run(_proteins(split=True), "t_test")

    assert np.isnan(with_split["P49"]["statistic"])  # not +-inf


@pytest.mark.parametrize("group", ["a", "b"])
def test_one_value_in_one_group_only_keeps_its_p_value(group):
    adata = _proteins(split=False)
    adata.X[(adata.obs["group"] == group).to_numpy(), 49] = 7.0

    with_flat, _, stats = _run(adata, "t_test")

    assert np.isfinite(with_flat["P49"]["p_value"])
    assert stats["n_zero_within_group_variance"] == 0


def test_the_reply_counts_two_split_proteins(tmp_path):
    adata = _proteins(split=True)
    adata.X[:6, 48], adata.X[6:, 48] = 3.0, 8.0
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("prot", adata)
    reply = factory_tools(de_analysis_expert, DE, dm)[
        "find_differential_proteins"
    ].invoke(
        {
            "modality_name": "prot",
            "group_column": "group",
            "platform_type": "affinity",
            "method": "t_test",
        }
    )

    assert (
        "- Of the proteins tested, 2 have the same value in every sample of each "
        "group (different between the groups): the t-test gives them no p-value"
        + ADVICE.format(it="them")
        in reply
    ), reply


@pytest.mark.parametrize(
    "n, method, advice",
    [
        (10, "t_test", ADVICE),
        (9, "t_test", FEW_ADVICE),
        (9, "limma_like", LIMMA_FEW_ADVICE),
    ],
)
def test_limma_like_borrows_from_10_proteins_tested(tmp_path, n, method, advice):
    adata = _proteins(split=True)[:, 50 - n :].copy()

    reply = _reply(tmp_path, adata, method)

    assert advice.format(it="it") in reply, reply


def test_limma_like_with_10_proteins_gives_a_split_protein_its_p_value():
    results, _, stats = _run(_proteins(split=True)[:, 40:].copy(), "limma_like")

    assert np.isfinite(results["P49"]["p_value"])
    assert stats["n_zero_within_group_variance"] == 0


def _three_groups(split_in):
    """a, b and c of 4 samples; c has a value for P45-P49 only, so a comparison
    with c tests 5 proteins. P49 is split between a and ``split_in``."""
    adata = _proteins(split=False)
    adata.obs["group"] = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    adata.X[8:, :45] = np.nan
    other = (adata.obs["group"] == split_in).to_numpy()
    adata.X[:4, 49], adata.X[other, 49] = 5.0, 9.0
    return adata


@pytest.mark.parametrize(
    "split_in, advice",
    [("b", ADVICE), ("c", FEW_ADVICE)],
    ids=["50_tested", "5_tested"],
)
def test_the_advice_follows_the_comparison_of_the_split_protein(
    tmp_path, split_in, advice
):
    reply = _reply(tmp_path, _three_groups(split_in), "t_test")

    assert "- Of the proteins tested, 1 has the same value" in reply, reply
    assert advice.format(it="it") in reply, reply
