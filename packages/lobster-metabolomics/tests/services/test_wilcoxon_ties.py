"""A Wilcoxon (Mann-Whitney) test of metabolites with tied values gives the
exact p-value.

scipy's ``mannwhitneyu`` with its default method falls back to a normal
approximation whenever a value ties, even at 3 v 3: a complete separation with
a tie inside a group got p = 0.077, where an exact test cannot go below 0.10.
Ties are common in metabolomics (values imputed at the detection limit). The
test now uses the exact test over every split of the samples, which handles
ties, as proteomics and bulk RNA-seq do.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from lobster.services.analysis.metabolomics_analysis_service import (
    MetabolomicsAnalysisService,
)

pytestmark = pytest.mark.unit

# (group a, group b, exact two-sided p over the 20 splits of 3 v 3)
TIED = {
    "M0": ([1, 1, 2], [4, 5, 6], 0.1),  # a tie inside a group, complete separation
    "M1": ([1, 2, 3], [3, 4, 5], 0.2),  # one value tied across the groups
    "M2": ([1, 1, 2], [2, 3, 3], 0.2),
}


def _metabolites(flat=False):
    rng = np.random.default_rng(0)
    x = rng.normal(10, 1, (6, 12))
    for j, (a, b, _) in enumerate(TIED.values()):
        x[:3, j], x[3:, j] = a, b
    if flat:
        x[:, 11] = 2.0
    obs = pd.DataFrame(
        {"group": ["a"] * 3 + ["b"] * 3}, index=[f"s{i}" for i in range(6)]
    )
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=[f"M{i}" for i in range(12)]))


def _run(adata):
    return MetabolomicsAnalysisService().run_univariate_statistics(
        adata, group_column="group", method="wilcoxon"
    )


def test_the_rows_are_ones_scipys_default_approximates():
    for a, b, exact in TIED.values():
        default = scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        assert default != pytest.approx(exact)


def test_tied_metabolites_get_the_exact_p_value():
    out, _, _ = _run(_metabolites())

    for name, (a, b, exact) in TIED.items():
        assert out.var.loc[name, "p_value"] == pytest.approx(exact, abs=1e-12)
        # the statistic stays U of the first group
        u = scipy_stats.mannwhitneyu(a, b, alternative="two-sided").statistic
        assert out.var.loc[name, "test_statistic"] == u


def test_a_flat_metabolite_gets_no_p_value_on_any_scipy_version():
    out, stats, _ = _run(_metabolites(flat=True))

    assert np.isnan(out.var.loc["M11", "p_value"])
    assert np.isnan(out.var.loc["M11", "fdr"])
    assert stats["n_tested"] == 11
