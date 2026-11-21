"""A Mann-Whitney test of proteins with tied values gives the exact p-value.

scipy's ``mannwhitneyu`` with its default method falls back to a normal
approximation whenever a value ties, even at 3 v 3: a complete separation with
a tie inside a group got p = 0.077, where an exact test cannot go below 0.10.
Ties are common in proteomics (values imputed at the detection limit, spectral
counts). Both services now use the exact test over every split of the samples,
which handles ties, while there are at most 20,000 splits.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from lobster.services.analysis.proteomics_analysis_service import (
    ProteomicsAnalysisService,
)
from lobster.services.analysis.proteomics_differential_service import (
    ProteomicsDifferentialService,
)

pytestmark = pytest.mark.unit

# (group a, group b, exact two-sided p over the 20 splits of 3 v 3)
TIED = {
    "P0": ([1, 1, 2], [4, 5, 6], 0.1),  # a tie inside a group, complete separation
    "P1": ([1, 2, 3], [3, 4, 5], 0.2),  # one value tied across the groups
    "P2": ([1, 1, 2], [2, 3, 3], 0.2),
}


def _proteins():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 1, (6, 12))
    for j, (a, b, _) in enumerate(TIED.values()):
        x[:3, j], x[3:, j] = a, b
    obs = pd.DataFrame(
        {"group": ["a"] * 3 + ["b"] * 3}, index=[f"s{i}" for i in range(6)]
    )
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=[f"P{i}" for i in range(12)]))


def test_the_rows_are_ones_scipys_default_approximates():
    for a, b, exact in TIED.values():
        default = scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        assert default != pytest.approx(exact)


def test_differential_expression_gives_tied_proteins_the_exact_p_value():
    out, _, _ = ProteomicsDifferentialService().perform_differential_expression(
        _proteins(), group_column="group", test_method="mann_whitney"
    )
    results = {
        r["protein"]: r for r in out.uns["differential_expression"]["all_results"]
    }

    for protein, (a, b, exact) in TIED.items():
        assert results[protein]["p_value"] == pytest.approx(exact, abs=1e-12)
        # the statistic stays U of the first group
        u = scipy_stats.mannwhitneyu(a, b, alternative="two-sided").statistic
        assert results[protein]["statistic"] == u


def test_statistical_testing_gives_tied_proteins_the_exact_p_value():
    out, _, _ = ProteomicsAnalysisService().perform_statistical_testing(
        _proteins(), group_column="group", test_method="mann_whitney"
    )
    results = {r["protein_index"]: r for r in out.uns["statistical_tests"]["results"]}

    for index, (a, b, exact) in enumerate(TIED.values()):
        assert results[index]["p_value"] == pytest.approx(exact, abs=1e-12)
        u = scipy_stats.mannwhitneyu(a, b, alternative="two-sided").statistic
        assert results[index]["statistic"] == u
