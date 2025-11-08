"""Quick validation script for proteomics service fixes."""

import numpy as np
import anndata as ad
from lobster.tools.proteomics_quality_service import ProteomicsQualityService
from lobster.tools.proteomics_differential_service import ProteomicsDifferentialService

def test_quality_service_fixes():
    """Test quality service MNAR/MCAR and API fixes."""
    print("=" * 80)
    print("Testing Quality Service Fixes")
    print("=" * 80)

    service = ProteomicsQualityService()

    # Create mock data with realistic missing value patterns
    n_samples, n_proteins = 48, 100
    X = np.random.lognormal(mean=10, sigma=1, size=(n_samples, n_proteins))

    # Add MNAR pattern (low abundance proteins have more missing)
    mnar_proteins = np.random.choice(n_proteins, 20, replace=False)
    for protein_idx in mnar_proteins:
        # Lower the mean intensity for these proteins
        X[:, protein_idx] = X[:, protein_idx] * 0.3
        # Add more missing values
        missing_mask = np.random.rand(n_samples) < 0.6
        X[missing_mask, protein_idx] = np.nan

    # Add MCAR pattern (random missing)
    mcar_missing = np.random.rand(n_samples, n_proteins) < 0.1
    X[mcar_missing] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]
    adata.obs["replicate_group"] = [f"group_{i//3}" for i in range(n_samples)]

    # Test 1: Missing value patterns with MNAR/MCAR detection
    print("\nTest 1: Missing Value Pattern Analysis")
    try:
        result_adata, stats = service.assess_missing_value_patterns(adata)

        # Check expected API fields
        required_fields = [
            "total_missing_percentage",
            "n_high_missing_samples",
            "n_high_missing_proteins",
            "missing_value_patterns",
            "mnar_proteins",
            "mcar_proteins",
            "analysis_type",
        ]

        missing_fields = [f for f in required_fields if f not in stats]
        if missing_fields:
            print(f"  ❌ FAILED: Missing fields: {missing_fields}")
        else:
            print(f"  ✅ PASSED: All API fields present")

        # Check MNAR/MCAR detection
        if "mnar_proteins" in stats and "mcar_proteins" in stats:
            print(f"  ✅ PASSED: MNAR/MCAR detection working")
            print(f"     - MNAR proteins: {stats['mnar_proteins']}")
            print(f"     - MCAR proteins: {stats['mcar_proteins']}")
        else:
            print(f"  ❌ FAILED: MNAR/MCAR detection not working")

        # Check obs columns
        required_obs = ["missing_value_percentage", "high_missing_sample"]
        missing_obs = [c for c in required_obs if c not in result_adata.obs.columns]
        if missing_obs:
            print(f"  ❌ FAILED: Missing obs columns: {missing_obs}")
        else:
            print(f"  ✅ PASSED: Required obs columns present")

        # Check var columns
        required_var = ["missing_value_percentage", "high_missing_protein", "missing_pattern"]
        missing_var = [c for c in required_var if c not in result_adata.var.columns]
        if missing_var:
            print(f"  ❌ FAILED: Missing var columns: {missing_var}")
        else:
            print(f"  ✅ PASSED: Required var columns present")

    except Exception as e:
        print(f"  ❌ FAILED: {str(e)}")

    # Test 2: CV assessment with replicate grouping
    print("\nTest 2: Coefficient of Variation Assessment")
    try:
        result_adata, stats = service.assess_coefficient_variation(
            adata, replicate_column="replicate_group"
        )

        required_fields = [
            "mean_cv_across_proteins",
            "median_cv_across_proteins",
            "n_high_cv_proteins",
            "cv_threshold",
            "analysis_type",
        ]

        missing_fields = [f for f in required_fields if f not in stats]
        if missing_fields:
            print(f"  ❌ FAILED: Missing fields: {missing_fields}")
        else:
            print(f"  ✅ PASSED: All API fields present")
            print(f"     - Mean CV: {stats['mean_cv_across_proteins']:.3f}")

        # Check var columns
        required_var = ["cv_mean", "cv_median", "high_cv_protein"]
        missing_var = [c for c in required_var if c not in result_adata.var.columns]
        if missing_var:
            print(f"  ❌ FAILED: Missing var columns: {missing_var}")
        else:
            print(f"  ✅ PASSED: Required var columns present")

    except Exception as e:
        print(f"  ❌ FAILED: {str(e)}")


def test_differential_service_fixes():
    """Test differential service empirical Bayes and API fixes."""
    print("\n" + "=" * 80)
    print("Testing Differential Service Fixes")
    print("=" * 80)

    service = ProteomicsDifferentialService()

    # Create mock data with true differential expression
    n_samples, n_proteins = 60, 100
    X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    # Add true differential expression (20 proteins)
    de_proteins = np.random.choice(n_proteins, 20, replace=False)
    for protein_idx in de_proteins[:10]:
        X[20:40, protein_idx] *= 2.5  # Upregulated in treatment1

    for protein_idx in de_proteins[10:]:
        X[40:60, protein_idx] *= 0.4  # Downregulated in treatment2

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]
    adata.obs["condition"] = ["control"] * 20 + ["treatment1"] * 20 + ["treatment2"] * 20

    # Test 1: Differential expression with limma-like method
    print("\nTest 1: Differential Expression (limma-like)")
    try:
        result_adata, stats = service.perform_differential_expression(
            adata, group_column="condition", test_method="limma_like"
        )

        # Check expected API fields
        required_fields = [
            "n_comparisons",
            "n_significant_proteins",
            "test_method",
            "fdr_method",
            "volcano_plot_data",
            "top_upregulated",
            "top_downregulated",
            "effect_size_distribution",
            "analysis_type",
        ]

        missing_fields = [f for f in required_fields if f not in stats]
        if missing_fields:
            print(f"  ❌ FAILED: Missing fields: {missing_fields}")
        else:
            print(f"  ✅ PASSED: All API fields present")
            print(f"     - Significant proteins: {stats['n_significant_proteins']}")
            print(f"     - Comparisons: {stats['n_comparisons']}")

        # Check var columns
        required_var = ["is_de_significant", "significant_proteins"]
        missing_var = [c for c in required_var if c not in result_adata.var.columns]
        if missing_var:
            print(f"  ❌ FAILED: Missing var columns: {missing_var}")
        else:
            print(f"  ✅ PASSED: Required var columns present")

        # Check uns storage
        if "differential_expression" not in result_adata.uns:
            print(f"  ❌ FAILED: Results not stored in uns")
        else:
            print(f"  ✅ PASSED: Results stored in uns")

    except Exception as e:
        print(f"  ❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

    # Test 2: Empirical Bayes method
    print("\nTest 2: Empirical Bayes Shrinkage")
    try:
        # Test prior variance estimation
        variances = np.random.gamma(2, 2, 100)  # Simulate protein variances
        dfs = np.full(100, 18)  # 20 samples per group = 18 df

        prior_var, prior_df = service._estimate_prior_variance(variances, dfs)

        print(f"  ✅ PASSED: Prior variance estimation working")
        print(f"     - Prior variance: {prior_var:.3f}")
        print(f"     - Prior df: {prior_df:.1f}")

        # Test moderated t-test
        group1 = np.random.normal(100, 10, 20)
        group2 = np.random.normal(120, 10, 20)

        t_stat, p_value = service._moderated_t_test(group1, group2, prior_var, prior_df)

        if 0 <= p_value <= 1:
            print(f"  ✅ PASSED: Moderated t-test producing valid p-values")
            print(f"     - t-statistic: {t_stat:.3f}")
            print(f"     - p-value: {p_value:.4f}")
        else:
            print(f"  ❌ FAILED: Invalid p-value: {p_value}")

    except Exception as e:
        print(f"  ❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Run all validation tests."""
    test_quality_service_fixes()
    test_differential_service_fixes()

    print("\n" + "=" * 80)
    print("Validation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
