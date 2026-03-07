"""
Tests for FeatureSelectionService.

Covers:
- FSL-01: M&B probability calculation (selection_counts / n_rounds)
- FSL-02: Method-prefixed column names, full adata return
- FSL-03: RNG reproducibility with isolated per-round RNG
- get_selected_features auto-detection and error handling
"""

import numpy as np
import pytest
from anndata import AnnData

from lobster.services.ml.feature_selection_service import FeatureSelectionService


@pytest.fixture
def feature_selection_service():
    """Create a FeatureSelectionService instance."""
    return FeatureSelectionService()


@pytest.fixture
def sample_adata():
    """Create sample AnnData for feature selection tests."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    # Create data with some informative features
    X = np.random.randn(n_samples, n_features)

    # Make some features actually predictive of target
    target = np.random.choice([0, 1], n_samples)

    # Features 0-4 are predictive (correlated with target)
    for i in range(5):
        X[:, i] = X[:, i] + target * 2.0

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_features)]
    adata.obs["target"] = target

    return adata


class TestStabilitySelection:
    """Tests for stability_selection method."""

    def test_fsl01_probability_range(self, feature_selection_service, sample_adata):
        """FSL-01: Selection probability must be in [0, 1] range."""
        adata_result, stats, ir = feature_selection_service.stability_selection(
            sample_adata,
            target_column="target",
            n_features=10,
            n_rounds=5,
            method="random_forest",
            random_state=42,
        )

        # Check probability is stored
        assert "stability_probability" in adata_result.var.columns

        probs = adata_result.var["stability_probability"].values

        # FSL-01: Probabilities must be in [0, 1]
        assert probs.min() >= 0.0, "Probability below 0"
        assert probs.max() <= 1.0, "Probability above 1"

    def test_fsl01_probability_formula(self, feature_selection_service, sample_adata):
        """FSL-01: Probability = selection_counts / n_rounds (M&B formula)."""
        n_rounds = 10

        adata_result, stats, ir = feature_selection_service.stability_selection(
            sample_adata,
            target_column="target",
            n_features=10,
            n_rounds=n_rounds,
            method="random_forest",
            random_state=42,
        )

        probs = adata_result.var["stability_probability"].values

        # Probabilities must be multiples of 1/n_rounds
        # (since they come from counting selections across discrete rounds)
        valid_values = np.arange(0, n_rounds + 1) / n_rounds
        for prob in probs:
            assert np.any(np.isclose(prob, valid_values, atol=1e-10)), (
                f"Probability {prob} is not a valid multiple of 1/{n_rounds}"
            )

    def test_fsl02_method_prefixed_columns(
        self, feature_selection_service, sample_adata
    ):
        """FSL-02: Results stored in stability_* prefixed columns."""
        adata_result, stats, ir = feature_selection_service.stability_selection(
            sample_adata,
            target_column="target",
            n_features=10,
            n_rounds=5,
            method="random_forest",
            random_state=42,
        )

        # Method-prefixed columns must exist
        assert "stability_probability" in adata_result.var.columns
        assert "stability_mean_importance" in adata_result.var.columns
        assert "stability_selected" in adata_result.var.columns

        # Old column name must NOT exist
        assert "selected_feature" not in adata_result.var.columns

    def test_fsl02_full_adata_return(self, feature_selection_service, sample_adata):
        """FSL-02: Returns full adata with selection columns, not filtered."""
        n_original_features = sample_adata.shape[1]

        adata_result, stats, ir = feature_selection_service.stability_selection(
            sample_adata,
            target_column="target",
            n_features=10,
            n_rounds=5,
            method="random_forest",
            random_state=42,
        )

        # Must return full adata (not filtered)
        assert adata_result.shape[1] == n_original_features

        # Selection mask should be boolean
        assert adata_result.var["stability_selected"].dtype == bool

    def test_fsl03_rng_reproducibility(self, feature_selection_service, sample_adata):
        """FSL-03: Same random_state produces identical results."""
        result1, _, _ = feature_selection_service.stability_selection(
            sample_adata.copy(),
            target_column="target",
            n_features=10,
            n_rounds=5,
            method="random_forest",
            random_state=42,
        )

        result2, _, _ = feature_selection_service.stability_selection(
            sample_adata.copy(),
            target_column="target",
            n_features=10,
            n_rounds=5,
            method="random_forest",
            random_state=42,
        )

        # Probabilities must be identical
        np.testing.assert_array_equal(
            result1.var["stability_probability"].values,
            result2.var["stability_probability"].values,
            err_msg="RNG not reproducible",
        )

        # Selection must be identical
        np.testing.assert_array_equal(
            result1.var["stability_selected"].values,
            result2.var["stability_selected"].values,
            err_msg="Selection not reproducible",
        )

    def test_fsl03_isolated_rng(self, feature_selection_service, sample_adata):
        """FSL-03: Feature selection doesn't affect downstream RNG state."""
        # Set global RNG state
        np.random.seed(999)
        before_value = np.random.rand()

        # Reset and run stability selection
        np.random.seed(999)
        _ = feature_selection_service.stability_selection(
            sample_adata.copy(),
            target_column="target",
            n_features=10,
            n_rounds=5,
            method="random_forest",
            random_state=42,
        )

        # Check global RNG state is unchanged
        # If stability_selection used np.random directly, this would differ
        after_value = np.random.rand()

        assert before_value == after_value, (
            "Stability selection polluted global RNG state"
        )


class TestLassoSelection:
    """Tests for lasso_selection method."""

    def test_lasso_method_prefixed_columns(
        self, feature_selection_service, sample_adata
    ):
        """Lasso results stored in lasso_* prefixed columns."""
        adata_result, stats, ir = feature_selection_service.lasso_selection(
            sample_adata,
            target_column="target",
            alpha=0.1,
            random_state=42,
        )

        # Method-prefixed columns must exist
        assert "lasso_coefficient" in adata_result.var.columns
        assert "lasso_selected" in adata_result.var.columns

    def test_lasso_full_adata_return(self, feature_selection_service, sample_adata):
        """Lasso returns full adata with selection columns, not filtered."""
        n_original_features = sample_adata.shape[1]

        adata_result, stats, ir = feature_selection_service.lasso_selection(
            sample_adata,
            target_column="target",
            alpha=0.1,
            random_state=42,
        )

        # Must return full adata (not filtered)
        assert adata_result.shape[1] == n_original_features


class TestVarianceFilter:
    """Tests for variance_filter method."""

    def test_variance_method_prefixed_columns(
        self, feature_selection_service, sample_adata
    ):
        """Variance results stored in variance_* prefixed columns."""
        adata_result, stats, ir = feature_selection_service.variance_filter(
            sample_adata,
            percentile=20.0,
        )

        # Method-prefixed columns must exist
        assert "variance" in adata_result.var.columns
        assert "variance_selected" in adata_result.var.columns

    def test_variance_full_adata_return(self, feature_selection_service, sample_adata):
        """Variance filter returns full adata with selection columns."""
        n_original_features = sample_adata.shape[1]

        adata_result, stats, ir = feature_selection_service.variance_filter(
            sample_adata,
            percentile=20.0,
        )

        # Must return full adata (not filtered)
        assert adata_result.shape[1] == n_original_features


class TestGetSelectedFeatures:
    """Tests for get_selected_features utility method."""

    def test_auto_detect_single_column(self, feature_selection_service, sample_adata):
        """Auto-detects selection column when only one exists."""
        # Add single selection column
        sample_adata.var["stability_selected"] = np.random.choice(
            [True, False], sample_adata.shape[1]
        )
        n_expected = sample_adata.var["stability_selected"].sum()

        features = feature_selection_service.get_selected_features(sample_adata)

        assert len(features) == n_expected

    def test_explicit_column_parameter(self, feature_selection_service, sample_adata):
        """Explicit column parameter works correctly."""
        sample_adata.var["stability_selected"] = [True, False] * (
            sample_adata.shape[1] // 2
        )
        sample_adata.var["lasso_selected"] = [False, True] * (
            sample_adata.shape[1] // 2
        )

        stability_features = feature_selection_service.get_selected_features(
            sample_adata, selection_column="stability_selected"
        )
        lasso_features = feature_selection_service.get_selected_features(
            sample_adata, selection_column="lasso_selected"
        )

        # Should return different feature sets
        assert set(stability_features) != set(lasso_features)

    def test_error_no_selection_columns(self, feature_selection_service, sample_adata):
        """Raises clear error when no selection columns exist."""
        # No *_selected columns
        with pytest.raises(ValueError, match="No selection columns found"):
            feature_selection_service.get_selected_features(sample_adata)

    def test_error_multiple_columns_ambiguous(
        self, feature_selection_service, sample_adata
    ):
        """Raises error when multiple selection columns exist without explicit param."""
        sample_adata.var["stability_selected"] = True
        sample_adata.var["lasso_selected"] = True

        with pytest.raises(ValueError, match="Multiple selection columns found"):
            feature_selection_service.get_selected_features(sample_adata)

    def test_error_specified_column_not_found(
        self, feature_selection_service, sample_adata
    ):
        """Raises error when specified column doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            feature_selection_service.get_selected_features(
                sample_adata, selection_column="nonexistent_selected"
            )


class TestApplySelection:
    """Tests for apply_selection utility method."""

    def test_apply_selection_filters_correctly(
        self, feature_selection_service, sample_adata
    ):
        """apply_selection returns filtered adata with selected features only."""
        selected_features = ["gene_0", "gene_1", "gene_5"]

        adata_filtered, stats, ir = feature_selection_service.apply_selection(
            sample_adata, selected_features
        )

        # Filtered adata should have only selected features
        assert adata_filtered.shape[1] == len(selected_features)
        assert list(adata_filtered.var_names) == selected_features

        # Stats should reflect reduction
        assert stats["n_original_features"] == sample_adata.shape[1]
        assert stats["n_selected_features"] == len(selected_features)

    def test_apply_selection_error_missing_features(
        self, feature_selection_service, sample_adata
    ):
        """apply_selection raises error for features not in adata."""
        selected_features = ["gene_0", "nonexistent_gene"]

        with pytest.raises(ValueError, match="not found"):
            feature_selection_service.apply_selection(sample_adata, selected_features)
