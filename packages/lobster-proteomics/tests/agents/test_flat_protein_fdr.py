"""A protein with the same value in every sample no longer wipes out a result.

A t-test, a Mann-Whitney test or a correlation gives no p-value (NaN) for such a
protein. Benjamini-Hochberg over a list holding one NaN made every adjusted
p-value NaN, so no protein was significant; Holm and Bonferroni counted the NaN
among the tests. The correction now runs over the finite p-values only, and the
reply counts the tests without one. A linear time trend gave the flat protein
p = 0 (a perfect fit of slope 0); it now gives p = 1, as the polynomial one does.
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
TIME = np.repeat([0.0, 1, 2, 3], 3)


def _proteins(flat):
    """12 samples (6 a, 6 b; 4 time points), 50 proteins: P0-P4 higher in b and
    rising with time; with ``flat``, P49 holds 2.0 in every sample."""
    rng = np.random.default_rng(0)
    x = rng.normal(10, 1, (12, 50))
    x[6:, :5] += 4
    x[:, :5] += 1.5 * TIME[:, None]
    if flat:
        x[:, 49] = 2.0
    obs = pd.DataFrame(
        {"group": ["a"] * 6 + ["b"] * 6, "time": TIME},
        index=[f"s{i}" for i in range(12)],
    )
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=[f"P{i}" for i in range(50)]))


def _results(adata, uns_key):
    return {r["protein"]: r for r in adata.uns[uns_key]["results"]}


@pytest.mark.parametrize("fdr", ["benjamini_hochberg", "holm", "bonferroni"])
@pytest.mark.parametrize("method", ["t_test", "mann_whitney"])
def test_a_flat_protein_changes_nothing_for_the_others(method, fdr):
    def run(adata):
        out, stats, _ = ProteomicsDifferentialService().perform_differential_expression(
            adata,
            group_column="group",
            test_method=method,
            fdr_method=fdr,
        )
        return {
            r["protein"]: r for r in out.uns["differential_expression"]["all_results"]
        }, stats

    with_flat, stats = run(_proteins(flat=True))
    without, stats_without = run(_proteins(flat=False)[:, :49].copy())

    assert np.isnan(with_flat.pop("P49")["p_adjusted"])
    assert with_flat.keys() == without.keys()
    for protein, result in without.items():
        assert with_flat[protein]["p_adjusted"] == pytest.approx(result["p_adjusted"])
    assert stats["n_significant_proteins"] == stats_without["n_significant_proteins"]
    assert stats["n_tests_without_pvalue"] == 1
    assert stats_without["n_tests_without_pvalue"] == 0


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_a_flat_protein_changes_no_correlation_result(method):
    def run(adata):
        out, stats, _ = ProteomicsDifferentialService().perform_correlation_analysis(
            adata, target_column="time", correlation_method=method
        )
        return _results(out, "correlation_analysis"), stats

    with_flat, stats = run(_proteins(flat=True))
    without, stats_without = run(_proteins(flat=False)[:, :49].copy())

    assert np.isnan(with_flat.pop("P49")["p_adjusted"])
    for protein, result in without.items():
        assert with_flat[protein]["p_adjusted"] == pytest.approx(result["p_adjusted"])
    for key in (
        "n_significant_results",
        "median_abs_correlation",
        "max_abs_correlation",
    ):
        assert stats[key] == pytest.approx(stats_without[key]), key
    assert stats["n_tests_without_pvalue"] == 1


def test_a_flat_protein_has_no_linear_time_trend():
    out, _, _ = ProteomicsDifferentialService().perform_time_course_analysis(
        _proteins(flat=True), time_column="time", test_method="linear_trend"
    )

    flat = _results(out, "time_course_analysis")["P49"]
    assert flat["p_value"] == 1.0
    significant = out.uns["time_course_analysis"]["significant_results"]
    assert "P49" not in {r["protein"] for r in significant}


def _tools(tmp_path, flat):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("prot", _proteins(flat))
    return factory_tools(de_analysis_expert, DE, dm)


@pytest.mark.parametrize("flat", [True, False])
def test_the_de_reply_counts_the_tests_without_a_p_value(tmp_path, flat):
    reply = _tools(tmp_path, flat)["find_differential_proteins"].invoke(
        {
            "modality_name": "prot",
            "group_column": "group",
            "platform_type": "affinity",
            "method": "t_test",
        }
    )

    line = (
        "- Tests without a p-value: 1 (a protein with the same value in every "
        "sample compared); left out of the FDR correction\n"
    )
    assert (line in reply) is flat, reply
    assert ("Tests without a p-value" in reply) is flat, reply
    assert "- Significant proteins: 0\n" not in reply, reply


@pytest.mark.parametrize("flat", [True, False])
def test_the_correlation_reply_counts_the_tests_without_a_p_value(tmp_path, flat):
    reply = _tools(tmp_path, flat)["run_correlation_analysis"].invoke(
        {"modality_name": "prot", "target_column": "time"}
    )

    line = (
        "- Tests without a p-value: 1 (a protein, or the target, with the same "
        "value in every sample); left out of the FDR correction\n"
    )
    assert (line in reply) is flat, reply
    assert "- Significant correlations: 0\n" not in reply, reply
    assert "nan" not in reply, reply


@pytest.fixture
def statsmodels_0_14(monkeypatch):
    """The corrections as statsmodels 0.14 has them: an empty list divides by zero."""
    import statsmodels.stats.multitest as multitest

    for name in ("fdrcorrection", "multipletests"):
        real = getattr(multitest, name)

        def refuse_empty(pvals, *args, _real=real, **kwargs):
            if len(pvals) == 0:
                raise ZeroDivisionError("float division by zero")
            return _real(pvals, *args, **kwargs)

        monkeypatch.setattr(multitest, name, refuse_empty)


def test_a_flat_target_gives_no_p_value_and_no_error(statsmodels_0_14):
    adata = _proteins(flat=False)
    adata.obs["dose"] = 1.0

    out, stats, _ = ProteomicsDifferentialService().perform_correlation_analysis(
        adata, target_column="dose"
    )

    assert stats["n_tests_without_pvalue"] == 50
    assert stats["n_significant_results"] == 0
    assert stats["median_abs_correlation"] == 0.0
    assert stats["max_abs_correlation"] == 0.0
    results = out.uns["correlation_analysis"]["results"]
    assert all(np.isnan(r["p_adjusted"]) for r in results)
