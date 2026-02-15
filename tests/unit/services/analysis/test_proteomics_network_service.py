"""
Unit tests for WGCNALiteService (proteomics network analysis).

Tests WGCNA-like co-expression module identification, module eigengene
calculation, and module-trait correlation analysis.
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.services.analysis.proteomics_network_service import (
    GREY_MODULE,
    WGCNA_COLORS,
    ProteomicsNetworkError,
    WGCNALiteService,
)


class TestWGCNALiteService:
    """Test suite for WGCNALiteService."""

    @pytest.fixture
    def service(self):
        """Create a WGCNALiteService instance."""
        return WGCNALiteService()

    @pytest.fixture
    def sample_adata_with_modules(self):
        """Create sample AnnData with correlated protein expression patterns."""
        np.random.seed(42)

        n_samples = 50
        n_proteins = 100

        # Create base expression
        X = np.random.randn(n_samples, n_proteins)

        # Create module 1 (proteins 0-29): correlated expression
        module1_signal = np.random.randn(n_samples)
        for i in range(30):
            X[:, i] = module1_signal + np.random.randn(n_samples) * 0.3

        # Create module 2 (proteins 30-59): different correlated expression
        module2_signal = np.random.randn(n_samples)
        for i in range(30, 60):
            X[:, i] = module2_signal + np.random.randn(n_samples) * 0.3

        # Create module 3 (proteins 60-79): another pattern
        module3_signal = np.random.randn(n_samples)
        for i in range(60, 80):
            X[:, i] = module3_signal + np.random.randn(n_samples) * 0.3

        # Proteins 80-99 remain uncorrelated (should be grey)

        # Create clinical traits
        obs = pd.DataFrame(
            {
                "PFS_days": np.random.exponential(scale=500, size=n_samples),
                "response": np.random.choice(["CR", "PR", "SD", "PD"], n_samples),
                "age": np.random.randint(30, 70, n_samples),
                "continuous_trait": module1_signal * 2
                + np.random.randn(n_samples) * 0.5,
            },
            index=[f"sample_{i}" for i in range(n_samples)],
        )

        var = pd.DataFrame(
            {"protein_name": [f"protein_{i}" for i in range(n_proteins)]},
            index=[f"protein_{i}" for i in range(n_proteins)],
        )

        return anndata.AnnData(X=X, obs=obs, var=var)

    @pytest.fixture
    def small_adata(self):
        """Create small AnnData for quick tests."""
        np.random.seed(42)

        n_samples = 30
        n_proteins = 50

        X = np.random.randn(n_samples, n_proteins)

        # Create one module
        signal = np.random.randn(n_samples)
        for i in range(20):
            X[:, i] = signal + np.random.randn(n_samples) * 0.3

        obs = pd.DataFrame(
            {"trait1": np.random.randn(n_samples)},
            index=[f"sample_{i}" for i in range(n_samples)],
        )

        return anndata.AnnData(X=X, obs=obs)

    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None

    def test_identify_modules_basic(self, service, sample_adata_with_modules):
        """Test basic module identification."""
        adata_modules, stats, ir = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
            distance_threshold=0.3,
            min_module_size=10,
        )

        # Check return types
        assert isinstance(adata_modules, anndata.AnnData)
        assert isinstance(stats, dict)
        assert ir is not None

        # Check statistics
        assert "n_modules" in stats
        assert "module_sizes" in stats
        assert stats["n_modules"] >= 1  # At least one module should be found

        # Check AnnData modifications
        assert "module" in adata_modules.var.columns
        assert "wgcna" in adata_modules.uns
        assert "module_eigengenes" in adata_modules.obsm

    def test_identify_modules_finds_expected_modules(
        self, service, sample_adata_with_modules
    ):
        """Test that module detection finds expected structure."""
        adata_modules, stats, _ = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
            distance_threshold=0.4,  # More lenient threshold
            min_module_size=15,
        )

        # Should find at least 2-3 modules given our data structure
        assert (
            stats["n_modules"] >= 2
        ), f"Expected >=2 modules, got {stats['n_modules']}"

        # Check module colors are from WGCNA palette
        module_colors = adata_modules.var["module"].unique()
        for color in module_colors:
            assert color in WGCNA_COLORS or color == GREY_MODULE

    def test_module_eigengenes_stored_correctly(
        self, service, sample_adata_with_modules
    ):
        """Test that module eigengenes are stored correctly."""
        adata_modules, stats, _ = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
        )

        # Check eigengenes are stored
        assert "module_eigengenes" in adata_modules.obsm

        # Check ME columns in obs
        me_cols = [c for c in adata_modules.obs.columns if c.startswith("ME_")]
        assert len(me_cols) > 0

        # Check eigengene shape matches samples
        assert adata_modules.obsm["module_eigengenes"].shape[0] == adata_modules.n_obs

    def test_correlate_modules_with_traits(self, service, sample_adata_with_modules):
        """Test module-trait correlation analysis."""
        # First identify modules
        adata_modules, _, _ = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
        )

        # Then correlate with traits
        adata_corr, stats, ir = service.correlate_modules_with_traits(
            adata_modules,
            traits=["PFS_days", "age", "continuous_trait"],
            correlation_method="pearson",
        )

        # Check return types
        assert isinstance(adata_corr, anndata.AnnData)
        assert isinstance(stats, dict)
        assert ir is not None

        # Check statistics
        assert "n_modules" in stats
        assert "n_traits" in stats
        assert "n_tests" in stats
        assert "n_significant_correlations" in stats

        # Check uns storage
        assert "module_trait_correlation" in adata_corr.uns
        assert "correlation_matrix" in adata_corr.uns["module_trait_correlation"]

    def test_correlate_finds_expected_correlation(
        self, service, sample_adata_with_modules
    ):
        """Test that module-trait correlation finds expected relationship."""
        # First identify modules
        adata_modules, _, _ = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
        )

        # Correlate with traits - continuous_trait was constructed to correlate with module1
        adata_corr, stats, _ = service.correlate_modules_with_traits(
            adata_modules,
            traits=["continuous_trait"],
            correlation_method="pearson",
        )

        # Should find at least some correlation
        results = adata_corr.uns["module_trait_correlation"]["results"]
        correlations = [abs(r["correlation"]) for r in results]

        # At least one module should have meaningful correlation
        assert max(correlations) > 0.3

    def test_get_module_proteins(self, service, sample_adata_with_modules):
        """Test retrieving proteins from a specific module."""
        adata_modules, _, _ = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
        )

        # Get the first non-grey module
        module_colors = [
            c for c in adata_modules.var["module"].unique() if c != GREY_MODULE
        ]

        if module_colors:
            proteins = service.get_module_proteins(adata_modules, module_colors[0])
            assert isinstance(proteins, list)
            assert len(proteins) > 0

    def test_get_module_summary(self, service, sample_adata_with_modules):
        """Test module summary statistics."""
        adata_modules, _, _ = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
        )

        summary = service.get_module_summary(adata_modules)

        assert "n_modules" in summary
        assert "module_sizes" in summary
        assert "total_assigned" in summary
        assert "total_unassigned" in summary
        assert "modules" in summary

    def test_calculate_module_membership(self, service, sample_adata_with_modules):
        """Test module membership (kME) calculation."""
        adata_modules, _, _ = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
        )

        adata_kme, stats, ir = service.calculate_module_membership(adata_modules)

        # Check kME columns added
        kme_cols = [c for c in adata_kme.var.columns if c.startswith("kME_")]
        assert len(kme_cols) > 0

        # Check hub protein identification
        if "is_hub" in adata_kme.var.columns:
            hub_count = adata_kme.var["is_hub"].sum()
            assert hub_count >= 0  # May be 0 if modules are small

    def test_spearman_correlation_method(self, service, small_adata):
        """Test module identification with Spearman correlation."""
        adata_modules, stats, _ = service.identify_modules(
            small_adata,
            n_top_variable=50,
            correlation_method="spearman",
        )

        assert "n_modules" in stats
        assert stats["correlation_method"] == "spearman"

    def test_soft_power_thresholding(self, service, small_adata):
        """Test module identification with soft power thresholding."""
        adata_modules, stats, _ = service.identify_modules(
            small_adata,
            n_top_variable=50,
            soft_power=6,
        )

        assert stats["soft_power"] == 6

    def test_module_merging(self, service, sample_adata_with_modules):
        """Test that similar modules can be merged."""
        # Run with strict threshold (more modules)
        adata_strict, stats_strict, _ = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
            merge_cut_height=0.1,  # Strict - less merging
        )

        # Run with lenient threshold (fewer modules)
        adata_lenient, stats_lenient, _ = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
            merge_cut_height=0.5,  # Lenient - more merging
        )

        # Lenient should have same or fewer modules
        assert stats_lenient["n_modules"] <= stats_strict["n_modules"]

    def test_min_module_size_enforcement(self, service, small_adata):
        """Test minimum module size enforcement."""
        adata_modules, stats, _ = service.identify_modules(
            small_adata,
            n_top_variable=50,
            min_module_size=10,
        )

        # Check all modules meet minimum size
        module_sizes = stats["module_sizes"]
        for module, size in module_sizes.items():
            assert size >= 10, f"Module {module} has size {size} < 10"

    def test_ir_generation(self, service, small_adata):
        """Test that IR is properly generated."""
        _, _, ir = service.identify_modules(small_adata, n_top_variable=50)

        # Check IR structure
        assert ir.operation == "proteomics.network.identify_modules"
        assert ir.tool_name == "identify_modules"
        assert ir.library == "lobster.services.analysis.proteomics_network_service"
        assert "n_top_variable" in ir.parameters
        assert "correlation_method" in ir.parameters

    def test_missing_module_eigengenes_error(self, service, small_adata):
        """Test error when correlating without running identify_modules first."""
        with pytest.raises(ProteomicsNetworkError, match="Module eigengenes not found"):
            service.correlate_modules_with_traits(small_adata, traits=["trait1"])

    def test_missing_module_assignments_error(self, service, small_adata):
        """Test error when getting proteins without running identify_modules first."""
        with pytest.raises(
            ProteomicsNetworkError, match="Module assignments not found"
        ):
            service.get_module_proteins(small_adata, "turquoise")

    def test_invalid_trait_handling(self, service, sample_adata_with_modules):
        """Test handling of invalid trait names."""
        adata_modules, _, _ = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
        )

        # Should warn but not fail with some valid traits
        adata_corr, stats, _ = service.correlate_modules_with_traits(
            adata_modules,
            traits=["nonexistent_trait", "PFS_days"],  # Mix of invalid and valid
        )

        assert stats["n_traits"] == 1  # Only PFS_days should be used

    def test_all_invalid_traits_error(self, service, sample_adata_with_modules):
        """Test error when all traits are invalid."""
        adata_modules, _, _ = service.identify_modules(
            sample_adata_with_modules,
            n_top_variable=100,
        )

        with pytest.raises(ProteomicsNetworkError, match="No valid traits"):
            service.correlate_modules_with_traits(
                adata_modules,
                traits=["nonexistent1", "nonexistent2"],
            )

    def test_handles_missing_values(self, service):
        """Test handling of missing values in expression data."""
        np.random.seed(42)

        X = np.random.randn(30, 50)
        # Add some NaN values
        X[5:10, 0:5] = np.nan
        X[15:20, 10:15] = np.nan

        obs = pd.DataFrame(
            {"trait": np.random.randn(30)},
            index=[f"sample_{i}" for i in range(30)],
        )

        adata = anndata.AnnData(X=X, obs=obs)

        # Should handle NaN gracefully
        adata_modules, stats, _ = service.identify_modules(
            adata,
            n_top_variable=50,
        )

        assert stats["n_modules"] >= 0  # Should complete

    def test_fdr_correction(self, service):
        """Test FDR correction method."""
        p_values = [0.01, 0.03, 0.05, 0.10, 0.50]
        fdr = service._benjamini_hochberg(p_values)

        assert len(fdr) == len(p_values)
        assert all(f >= p for f, p in zip(fdr, p_values))
        assert all(f <= 1.0 for f in fdr)

    def test_empty_fdr(self, service):
        """Test FDR correction with empty list."""
        fdr = service._benjamini_hochberg([])
        assert fdr == []

    def test_pick_soft_threshold_basic(self, service, sample_adata_with_modules):
        """Test basic soft power selection."""
        results, stats, ir = service.pick_soft_threshold(
            sample_adata_with_modules,
            n_top_variable=100,
        )

        assert "selected_power" in results
        assert 1 <= results["selected_power"] <= 20
        assert "power_table" in results
        assert isinstance(results["power_table"], pd.DataFrame)
        assert len(results["power_table"]) == 20
        assert ir is not None

    def test_pick_soft_threshold_custom_powers(
        self, service, sample_adata_with_modules
    ):
        """Test with custom power range."""
        results, stats, ir = service.pick_soft_threshold(
            sample_adata_with_modules,
            powers=[4, 6, 8, 10, 12],
            n_top_variable=100,
        )

        assert results["selected_power"] in [4, 6, 8, 10, 12]
        assert len(results["power_table"]) == 5

    def test_pick_soft_threshold_power_table_columns(
        self, service, sample_adata_with_modules
    ):
        """Test power table has expected columns."""
        results, _, _ = service.pick_soft_threshold(
            sample_adata_with_modules,
            n_top_variable=100,
        )

        expected_cols = [
            "power",
            "r_squared",
            "slope",
            "mean_connectivity",
            "median_connectivity",
            "truncated_r_squared",
        ]
        for col in expected_cols:
            assert col in results["power_table"].columns

    def test_pick_soft_threshold_spearman(self, service, sample_adata_with_modules):
        """Test soft power selection with Spearman correlation."""
        results, stats, _ = service.pick_soft_threshold(
            sample_adata_with_modules,
            correlation_method="spearman",
            n_top_variable=100,
        )

        assert "selected_power" in results
        assert stats["correlation_method"] == "spearman"

    def test_pick_soft_threshold_integration_with_identify_modules(
        self, service, sample_adata_with_modules
    ):
        """Test that selected power can be used in identify_modules."""
        # First pick soft threshold
        results, _, _ = service.pick_soft_threshold(
            sample_adata_with_modules,
            n_top_variable=100,
        )

        selected_power = results["selected_power"]

        # Then use it in identify_modules
        adata_modules, stats, _ = service.identify_modules(
            sample_adata_with_modules,
            soft_power=selected_power,
            n_top_variable=100,
        )

        assert stats["soft_power"] == selected_power
        assert "module" in adata_modules.var.columns


class TestWGCNALiteServiceEdgeCases:
    """Edge case tests for WGCNALiteService."""

    @pytest.fixture
    def service(self):
        return WGCNALiteService()

    def test_very_small_dataset(self, service):
        """Test with very small dataset."""
        X = np.random.randn(10, 20)
        obs = pd.DataFrame(index=[f"s_{i}" for i in range(10)])

        adata = anndata.AnnData(X=X, obs=obs)

        # Should still run
        adata_modules, stats, _ = service.identify_modules(
            adata,
            n_top_variable=20,
            min_module_size=3,  # Allow smaller modules
        )

        assert isinstance(stats, dict)

    def test_single_protein_per_sample(self, service):
        """Test with extreme dimensions."""
        X = np.random.randn(100, 10)
        obs = pd.DataFrame(index=[f"s_{i}" for i in range(100)])

        adata = anndata.AnnData(X=X, obs=obs)

        adata_modules, stats, _ = service.identify_modules(
            adata,
            n_top_variable=10,
            min_module_size=3,
        )

        assert isinstance(stats, dict)

    def test_constant_protein_values(self, service):
        """Test handling of constant protein values."""
        np.random.seed(42)

        X = np.random.randn(30, 50)
        # Make some proteins constant
        X[:, 0] = 10.0
        X[:, 1] = 5.0

        obs = pd.DataFrame(index=[f"s_{i}" for i in range(30)])

        adata = anndata.AnnData(X=X, obs=obs)

        # Should handle gracefully
        adata_modules, stats, _ = service.identify_modules(
            adata,
            n_top_variable=50,
        )

        assert isinstance(stats, dict)

    def test_n_top_variable_larger_than_available(self, service):
        """Test when n_top_variable exceeds available proteins."""
        X = np.random.randn(30, 20)
        obs = pd.DataFrame(index=[f"s_{i}" for i in range(30)])

        adata = anndata.AnnData(X=X, obs=obs)

        # Request 100 but only 20 available
        adata_modules, stats, _ = service.identify_modules(
            adata,
            n_top_variable=100,
            min_module_size=3,
        )

        # Should use all 20
        assert stats["n_proteins_analyzed"] == 20

    def test_wgcna_color_assignment(self, service):
        """Test that WGCNA colors are properly assigned."""
        np.random.seed(42)

        # Create data with many modules
        n_samples = 50
        n_proteins = 200
        X = np.random.randn(n_samples, n_proteins)

        # Create 5 distinct modules
        for module_idx in range(5):
            start = module_idx * 30
            end = start + 30
            signal = np.random.randn(n_samples)
            for i in range(start, min(end, n_proteins)):
                X[:, i] = signal + np.random.randn(n_samples) * 0.2

        obs = pd.DataFrame(index=[f"s_{i}" for i in range(n_samples)])
        adata = anndata.AnnData(X=X, obs=obs)

        adata_modules, stats, _ = service.identify_modules(
            adata,
            n_top_variable=200,
            min_module_size=15,
        )

        # All module colors should be from WGCNA palette
        colors_used = adata_modules.var["module"].unique()
        for color in colors_used:
            assert color in WGCNA_COLORS or color == GREY_MODULE

    def test_module_uns_storage(
        self,
        service,
    ):
        """Test comprehensive uns storage."""
        np.random.seed(42)

        X = np.random.randn(30, 50)
        signal = np.random.randn(30)
        for i in range(20):
            X[:, i] = signal + np.random.randn(30) * 0.3

        obs = pd.DataFrame(index=[f"s_{i}" for i in range(30)])
        adata = anndata.AnnData(X=X, obs=obs)

        adata_modules, _, _ = service.identify_modules(
            adata,
            n_top_variable=50,
            min_module_size=5,
        )

        # Check all expected uns fields
        wgcna_uns = adata_modules.uns["wgcna"]
        assert "modules" in wgcna_uns
        assert "module_sizes" in wgcna_uns
        assert "module_colors" in wgcna_uns
        assert "correlation_method" in wgcna_uns
        assert "proteins_used" in wgcna_uns
        assert "linkage_matrix" in wgcna_uns
