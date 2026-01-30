"""
Unit tests for ProteomicsSurvivalService.

Tests Cox proportional hazards regression and Kaplan-Meier survival analysis
for proteomics data.
"""

import numpy as np
import pandas as pd
import pytest
import anndata

from lobster.services.analysis.proteomics_survival_service import (
    ProteomicsSurvivalService,
    ProteomicsSurvivalError,
)


class TestProteomicsSurvivalService:
    """Test suite for ProteomicsSurvivalService."""

    @pytest.fixture
    def service(self):
        """Create a ProteomicsSurvivalService instance."""
        return ProteomicsSurvivalService()

    @pytest.fixture
    def sample_adata_with_survival(self):
        """Create sample AnnData with survival data."""
        np.random.seed(42)

        n_samples = 100
        n_proteins = 50

        # Generate expression data
        X = np.random.randn(n_samples, n_proteins) * 2 + 10

        # Make some proteins correlated with survival
        # High expression = better survival for protein 0
        survival_correlated = np.random.randn(n_samples)
        X[:, 0] = survival_correlated * 3 + 10

        # High expression = worse survival for protein 1
        X[:, 1] = -survival_correlated * 3 + 10

        # Generate survival times (days)
        # Base survival time
        base_time = np.random.exponential(scale=500, size=n_samples)

        # Modify based on survival_correlated (protective effect)
        pfs_days = base_time * np.exp(-survival_correlated * 0.3)
        pfs_days = np.clip(pfs_days, 1, 2000)

        # Generate censoring
        censoring_time = np.random.uniform(100, 1500, n_samples)
        pfs_event = (pfs_days <= censoring_time).astype(int)
        pfs_days = np.minimum(pfs_days, censoring_time)

        # Create AnnData
        obs = pd.DataFrame(
            {
                "PFS_days": pfs_days,
                "PFS_event": pfs_event,
                "age": np.random.randint(30, 70, n_samples),
                "stage": np.random.choice(["I", "II", "III"], n_samples),
            },
            index=[f"sample_{i}" for i in range(n_samples)],
        )

        var = pd.DataFrame(
            {"protein_name": [f"protein_{i}" for i in range(n_proteins)]},
            index=[f"protein_{i}" for i in range(n_proteins)],
        )

        adata = anndata.AnnData(X=X, obs=obs, var=var)
        return adata

    @pytest.fixture
    def small_adata_with_survival(self):
        """Create small AnnData for quick tests."""
        np.random.seed(42)

        n_samples = 30
        n_proteins = 10

        X = np.random.randn(n_samples, n_proteins) * 2 + 10

        obs = pd.DataFrame(
            {
                "PFS_days": np.random.exponential(scale=500, size=n_samples),
                "PFS_event": np.random.binomial(1, 0.7, n_samples),
            },
            index=[f"sample_{i}" for i in range(n_samples)],
        )

        var = pd.DataFrame(
            index=[f"protein_{i}" for i in range(n_proteins)],
        )

        return anndata.AnnData(X=X, obs=obs, var=var)

    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        # Check lifelines detection
        assert isinstance(service._lifelines_available, bool)

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_cox_regression_basic(self, service, sample_adata_with_survival):
        """Test basic Cox regression functionality."""
        adata_cox, stats, ir = service.perform_cox_regression(
            sample_adata_with_survival,
            duration_col="PFS_days",
            event_col="PFS_event",
            fdr_threshold=0.1,
            min_samples=20,
        )

        # Check return types
        assert isinstance(adata_cox, anndata.AnnData)
        assert isinstance(stats, dict)
        assert ir is not None

        # Check statistics
        assert "n_proteins_tested" in stats
        assert "n_significant_proteins" in stats
        assert stats["n_proteins_tested"] > 0

        # Check AnnData modifications
        assert "cox_hazard_ratio" in adata_cox.var.columns
        assert "cox_p_value" in adata_cox.var.columns
        assert "cox_fdr" in adata_cox.var.columns
        assert "cox_regression" in adata_cox.uns

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_cox_regression_with_covariates(self, service, sample_adata_with_survival):
        """Test Cox regression with covariates."""
        adata_cox, stats, ir = service.perform_cox_regression(
            sample_adata_with_survival,
            duration_col="PFS_days",
            event_col="PFS_event",
            covariates=["age"],  # Test with numeric covariate
            fdr_threshold=0.1,
        )

        assert isinstance(adata_cox, anndata.AnnData)
        assert stats["n_proteins_tested"] > 0

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_cox_regression_stores_results(self, service, sample_adata_with_survival):
        """Test that Cox results are properly stored."""
        adata_cox, stats, ir = service.perform_cox_regression(
            sample_adata_with_survival,
            duration_col="PFS_days",
            event_col="PFS_event",
        )

        # Check uns storage
        assert "cox_regression" in adata_cox.uns
        assert "results" in adata_cox.uns["cox_regression"]
        assert "parameters" in adata_cox.uns["cox_regression"]

        # Check results structure
        results = adata_cox.uns["cox_regression"]["results"]
        assert len(results) > 0

        first_result = results[0]
        assert "protein" in first_result
        assert "hazard_ratio" in first_result
        assert "p_value" in first_result

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_kaplan_meier_basic(self, service, sample_adata_with_survival):
        """Test basic Kaplan-Meier analysis."""
        adata_km, stats, ir = service.kaplan_meier_analysis(
            sample_adata_with_survival,
            duration_col="PFS_days",
            event_col="PFS_event",
            protein="protein_0",
            stratify_method="median",
        )

        # Check return types
        assert isinstance(adata_km, anndata.AnnData)
        assert isinstance(stats, dict)
        assert ir is not None

        # Check statistics
        assert "log_rank_p_value" in stats
        assert "median_survival_by_group" in stats
        assert "survival_curves" in stats

        # Check uns storage
        assert "kaplan_meier" in adata_km.uns
        assert "survival_curves" in adata_km.uns["kaplan_meier"]

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_kaplan_meier_tertile_stratification(self, service, sample_adata_with_survival):
        """Test Kaplan-Meier with tertile stratification."""
        adata_km, stats, ir = service.kaplan_meier_analysis(
            sample_adata_with_survival,
            duration_col="PFS_days",
            event_col="PFS_event",
            protein="protein_0",
            stratify_method="tertile",
        )

        assert stats["n_groups"] == 3
        assert len(stats["survival_curves"]) == 3

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_kaplan_meier_quartile_stratification(self, service, sample_adata_with_survival):
        """Test Kaplan-Meier with quartile stratification."""
        adata_km, stats, ir = service.kaplan_meier_analysis(
            sample_adata_with_survival,
            duration_col="PFS_days",
            event_col="PFS_event",
            protein="protein_0",
            stratify_method="quartile",
        )

        assert stats["n_groups"] == 4

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_batch_kaplan_meier(self, service, small_adata_with_survival):
        """Test batch Kaplan-Meier analysis."""
        adata_batch, stats, ir = service.batch_kaplan_meier(
            small_adata_with_survival,
            duration_col="PFS_days",
            event_col="PFS_event",
            proteins=["protein_0", "protein_1", "protein_2"],
            fdr_threshold=0.1,
        )

        assert isinstance(adata_batch, anndata.AnnData)
        assert "n_proteins_tested" in stats
        assert "batch_kaplan_meier" in adata_batch.uns

    def test_missing_duration_column(self, service, sample_adata_with_survival):
        """Test error handling for missing duration column."""
        with pytest.raises(ProteomicsSurvivalError, match="Duration column"):
            service.perform_cox_regression(
                sample_adata_with_survival,
                duration_col="nonexistent_column",
                event_col="PFS_event",
            )

    def test_missing_event_column(self, service, sample_adata_with_survival):
        """Test error handling for missing event column."""
        with pytest.raises(ProteomicsSurvivalError, match="Event column"):
            service.perform_cox_regression(
                sample_adata_with_survival,
                duration_col="PFS_days",
                event_col="nonexistent_column",
            )

    def test_missing_protein_in_km(self, service, sample_adata_with_survival):
        """Test error handling for missing protein in KM analysis."""
        with pytest.raises(ProteomicsSurvivalError, match="not found in var_names"):
            service.kaplan_meier_analysis(
                sample_adata_with_survival,
                duration_col="PFS_days",
                event_col="PFS_event",
                protein="nonexistent_protein",
            )

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_insufficient_samples(self, service):
        """Test error handling for insufficient samples."""
        # Create very small dataset
        X = np.random.randn(5, 10)
        obs = pd.DataFrame(
            {
                "PFS_days": [100, 200, 300, 400, 500],
                "PFS_event": [1, 1, 0, 1, 0],
            }
        )
        adata_small = anndata.AnnData(X=X, obs=obs)

        with pytest.raises(ProteomicsSurvivalError, match="Insufficient valid samples"):
            service.perform_cox_regression(
                adata_small,
                duration_col="PFS_days",
                event_col="PFS_event",
                min_samples=20,  # More than available
            )

    def test_benjamini_hochberg_correction(self, service):
        """Test FDR correction method."""
        p_values = [0.01, 0.03, 0.05, 0.10, 0.50]
        fdr = service._benjamini_hochberg(p_values)

        assert len(fdr) == len(p_values)
        # FDR values should be >= p-values
        assert all(f >= p for f, p in zip(fdr, p_values))
        # FDR values should be <= 1
        assert all(f <= 1.0 for f in fdr)

    def test_benjamini_hochberg_empty(self, service):
        """Test FDR correction with empty list."""
        fdr = service._benjamini_hochberg([])
        assert fdr == []

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_ir_generation(self, service, small_adata_with_survival):
        """Test that IR is properly generated."""
        _, _, ir = service.perform_cox_regression(
            small_adata_with_survival,
            duration_col="PFS_days",
            event_col="PFS_event",
        )

        # Check IR structure
        assert ir.operation == "proteomics.survival.perform_cox_regression"
        assert ir.tool_name == "perform_cox_regression"
        assert ir.library == "lobster.services.analysis.proteomics_survival_service"
        assert "duration_col" in ir.parameters
        assert "event_col" in ir.parameters

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_handles_missing_values_in_expression(self, service):
        """Test handling of missing values in expression data."""
        np.random.seed(42)

        X = np.random.randn(50, 10)
        # Add some NaN values
        X[5:10, 0] = np.nan
        X[15:20, 1] = np.nan

        obs = pd.DataFrame(
            {
                "PFS_days": np.random.exponential(scale=500, size=50),
                "PFS_event": np.random.binomial(1, 0.7, 50),
            }
        )

        adata = anndata.AnnData(X=X, obs=obs)

        # Should handle NaN gracefully
        adata_cox, stats, _ = service.perform_cox_regression(
            adata,
            duration_col="PFS_days",
            event_col="PFS_event",
        )

        assert stats["n_proteins_tested"] > 0

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_survival_curve_data_structure(self, service, sample_adata_with_survival):
        """Test survival curve data structure for plotting."""
        adata_km, stats, _ = service.kaplan_meier_analysis(
            sample_adata_with_survival,
            duration_col="PFS_days",
            event_col="PFS_event",
            protein="protein_0",
            stratify_method="median",
        )

        curves = stats["survival_curves"]
        assert len(curves) == 2  # High and Low groups

        for group, curve_data in curves.items():
            assert "timeline" in curve_data
            assert "survival_function" in curve_data
            assert "confidence_lower" in curve_data
            assert "confidence_upper" in curve_data
            assert "n_at_risk" in curve_data
            assert "n_samples" in curve_data
            assert "n_events" in curve_data

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_optimal_cutpoint_method(self, service, sample_adata_with_survival):
        """Test optimal cutpoint selection method."""
        adata_km, stats, _ = service.kaplan_meier_analysis(
            sample_adata_with_survival,
            duration_col="PFS_days",
            event_col="PFS_event",
            protein="protein_0",
            stratify_method="optimal",
        )

        assert stats["n_groups"] == 2
        assert "log_rank_p_value" in stats


class TestProteomicsSurvivalServiceEdgeCases:
    """Edge case tests for ProteomicsSurvivalService."""

    @pytest.fixture
    def service(self):
        return ProteomicsSurvivalService()

    def test_all_censored_data(self, service):
        """Test handling when all data is censored."""
        np.random.seed(42)

        X = np.random.randn(30, 10)
        obs = pd.DataFrame(
            {
                "PFS_days": np.random.exponential(scale=500, size=30),
                "PFS_event": np.zeros(30),  # All censored
            }
        )

        adata = anndata.AnnData(X=X, obs=obs)

        # Should still run but may have fewer converged models
        if service._lifelines_available:
            adata_cox, stats, _ = service.perform_cox_regression(
                adata,
                duration_col="PFS_days",
                event_col="PFS_event",
            )
            # Check it completes without error
            assert isinstance(stats, dict)

    def test_low_variance_proteins(self, service):
        """Test handling of low variance proteins."""
        np.random.seed(42)

        X = np.random.randn(50, 10)
        # Make one protein constant
        X[:, 0] = 10.0

        obs = pd.DataFrame(
            {
                "PFS_days": np.random.exponential(scale=500, size=50),
                "PFS_event": np.random.binomial(1, 0.7, 50),
            }
        )

        adata = anndata.AnnData(X=X, obs=obs)

        if service._lifelines_available:
            adata_cox, stats, _ = service.perform_cox_regression(
                adata,
                duration_col="PFS_days",
                event_col="PFS_event",
            )
            # Low variance proteins should be skipped
            assert "n_low_variance_skipped" in stats

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_integer_var_names_handling(self, service):
        """Test Cox regression with integer var_names (AnnData default without explicit var).

        This verifies the fix for KeyError when AnnData has RangeIndex as var_names,
        which is the default behavior when creating AnnData without specifying var.
        """
        np.random.seed(42)

        X = np.random.randn(30, 10)
        obs = pd.DataFrame({
            "PFS_days": np.random.exponential(scale=500, size=30),
            "PFS_event": np.random.binomial(1, 0.7, 30),
        })

        # Create AnnData WITHOUT explicit var - defaults to integer indices [0, 1, 2, ...]
        adata = anndata.AnnData(X=X, obs=obs)

        # Verify var_names are string representations of integers (AnnData converts to strings)
        assert adata.var_names.tolist() == [str(i) for i in range(10)]

        # Should NOT raise KeyError
        adata_cox, stats, ir = service.perform_cox_regression(
            adata,
            duration_col="PFS_days",
            event_col="PFS_event",
            min_samples=20,
        )

        # Verify results stored correctly
        assert "cox_hazard_ratio" in adata_cox.var.columns
        assert len(adata_cox.var["cox_hazard_ratio"]) == 10
        assert stats["n_proteins_tested"] > 0

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_cox_regression_parallel(self, service):
        """Test Cox regression with parallel execution (n_jobs > 1)."""
        np.random.seed(42)

        n_samples = 50
        n_proteins = 30

        X = np.random.randn(n_samples, n_proteins) * 2 + 10

        obs = pd.DataFrame(
            {
                "PFS_days": np.random.exponential(scale=500, size=n_samples),
                "PFS_event": np.random.binomial(1, 0.7, n_samples),
            },
            index=[f"sample_{i}" for i in range(n_samples)],
        )

        var = pd.DataFrame(
            index=[f"protein_{i}" for i in range(n_proteins)],
        )

        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Run with parallel execution
        adata_cox, stats, ir = service.perform_cox_regression(
            adata,
            duration_col="PFS_days",
            event_col="PFS_event",
            n_jobs=4,  # Use 4 parallel workers
        )

        # Verify results
        assert isinstance(adata_cox, anndata.AnnData)
        assert "n_proteins_tested" in stats
        assert stats["n_proteins_tested"] > 0
        assert "cox_hazard_ratio" in adata_cox.var.columns
        assert "cox_regression" in adata_cox.uns

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_cox_regression_parallel_matches_sequential(self, service):
        """Test that parallel and sequential execution produce consistent results."""
        np.random.seed(42)

        n_samples = 40
        n_proteins = 20

        X = np.random.randn(n_samples, n_proteins) * 2 + 10

        obs = pd.DataFrame(
            {
                "PFS_days": np.random.exponential(scale=500, size=n_samples),
                "PFS_event": np.random.binomial(1, 0.7, n_samples),
            },
            index=[f"sample_{i}" for i in range(n_samples)],
        )

        var = pd.DataFrame(
            index=[f"protein_{i}" for i in range(n_proteins)],
        )

        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Run sequential
        _, stats_seq, _ = service.perform_cox_regression(
            adata.copy(),
            duration_col="PFS_days",
            event_col="PFS_event",
            n_jobs=1,
        )

        # Run parallel
        _, stats_par, _ = service.perform_cox_regression(
            adata.copy(),
            duration_col="PFS_days",
            event_col="PFS_event",
            n_jobs=4,
        )

        # Results should match (same number of proteins tested, significant, etc.)
        assert stats_seq["n_proteins_tested"] == stats_par["n_proteins_tested"]
        assert stats_seq["n_proteins_converged"] == stats_par["n_proteins_converged"]
        assert stats_seq["n_low_variance_skipped"] == stats_par["n_low_variance_skipped"]

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_fit_single_protein_cox_helper(self, service):
        """Test the helper method directly."""
        np.random.seed(42)

        n_samples = 30
        duration = np.random.exponential(scale=500, size=n_samples)
        event = np.random.binomial(1, 0.7, n_samples)
        valid_mask = np.ones(n_samples, dtype=bool)
        protein_values = np.random.randn(n_samples) * 2 + 10

        result = service._fit_single_protein_cox(
            protein_index=0,
            protein_name="test_protein",
            protein_values=protein_values,
            duration=duration,
            event=event,
            valid_mask=valid_mask,
            covariate_data={},
            min_samples=20,
            penalizer=0.1,
        )

        # Should return a result dict
        assert result is not None
        assert "protein" in result
        assert result["protein"] == "test_protein"
        assert "hazard_ratio" in result
        assert "p_value" in result
        assert "converged" in result

    @pytest.mark.skipif(
        not ProteomicsSurvivalService()._lifelines_available,
        reason="lifelines not installed",
    )
    def test_fit_single_protein_cox_low_variance(self, service):
        """Test helper method skips low variance proteins."""
        np.random.seed(42)

        n_samples = 30
        duration = np.random.exponential(scale=500, size=n_samples)
        event = np.random.binomial(1, 0.7, n_samples)
        valid_mask = np.ones(n_samples, dtype=bool)
        # Constant protein values (zero variance)
        protein_values = np.full(n_samples, 10.0)

        result = service._fit_single_protein_cox(
            protein_index=0,
            protein_name="constant_protein",
            protein_values=protein_values,
            duration=duration,
            event=event,
            valid_mask=valid_mask,
            covariate_data={},
            min_samples=20,
            penalizer=0.1,
        )

        # Should return None for low variance
        assert result is None
