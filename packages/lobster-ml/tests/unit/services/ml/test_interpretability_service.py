"""
Unit tests for InterpretabilityService multi-class SHAP storage.

Tests per-class SHAP layer storage, binary consistency, backward compatibility,
and aggregation layer discovery.
"""

import numpy as np
import pytest
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier

from lobster.services.ml.interpretability_service import InterpretabilityService

# Check if SHAP is available for conditional testing
try:
    import shap  # noqa: F401

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SHAP_AVAILABLE, reason="SHAP not installed"
)


@pytest.fixture
def service():
    """Create InterpretabilityService instance."""
    return InterpretabilityService()


@pytest.fixture
def sample_adata():
    """Create sample AnnData for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 50)
    obs = {"sample_id": [f"sample_{i}" for i in range(100)]}
    var = {"gene_name": [f"gene_{i}" for i in range(50)]}
    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def multiclass_model():
    """Create mock multi-class model with 3 classes."""
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.random.choice([0, 1, 2], size=100)

    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    model.fit(X, y)
    model.classes_ = np.array([0, 1, 2])

    return model


@pytest.fixture
def binary_model():
    """Create mock binary classification model."""
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.random.choice([0, 1], size=100)

    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    model.fit(X, y)
    model.classes_ = np.array([0, 1])

    return model


class TestMultiClassSHAP:
    """Test multi-class SHAP per-class layer storage."""

    def test_extract_shap_multiclass_per_class_layers(
        self, service, sample_adata, multiclass_model
    ):
        """Test that multi-class models store per-class SHAP layers."""
        adata_result, stats, ir = service.extract_shap_values(
            sample_adata, multiclass_model, model_type="tree", random_state=42
        )

        # Check per-class layers exist
        assert "shap_class_0" in adata_result.layers
        assert "shap_class_1" in adata_result.layers
        assert "shap_class_2" in adata_result.layers

        # Check shapes match
        n_samples, n_features = sample_adata.shape
        assert adata_result.layers["shap_class_0"].shape == (n_samples, n_features)
        assert adata_result.layers["shap_class_1"].shape == (n_samples, n_features)
        assert adata_result.layers["shap_class_2"].shape == (n_samples, n_features)

        # Check all values are absolute (non-negative)
        assert np.all(adata_result.layers["shap_class_0"] >= 0)
        assert np.all(adata_result.layers["shap_class_1"] >= 0)
        assert np.all(adata_result.layers["shap_class_2"] >= 0)

    def test_extract_shap_multiclass_aggregate_layer(
        self, service, sample_adata, multiclass_model
    ):
        """Test that aggregate layer exists for backward compatibility."""
        adata_result, stats, ir = service.extract_shap_values(
            sample_adata, multiclass_model, model_type="tree", random_state=42
        )

        # Check aggregate layer exists
        assert "shap_values" in adata_result.layers

        # Verify aggregate is mean absolute of per-class
        per_class = [
            adata_result.layers["shap_class_0"],
            adata_result.layers["shap_class_1"],
            adata_result.layers["shap_class_2"],
        ]
        expected_aggregate = np.mean(per_class, axis=0)

        np.testing.assert_array_almost_equal(
            adata_result.layers["shap_values"], expected_aggregate, decimal=5
        )

    def test_extract_shap_multiclass_class_mapping(
        self, service, sample_adata, multiclass_model
    ):
        """Test that class mapping is stored in uns."""
        adata_result, stats, ir = service.extract_shap_values(
            sample_adata, multiclass_model, model_type="tree", random_state=42
        )

        # Check class mapping exists
        assert "shap_class_mapping" in adata_result.uns
        mapping = adata_result.uns["shap_class_mapping"]

        # Check mapping structure
        assert len(mapping) == 3
        assert mapping == {0: "0", 1: "1", 2: "2"}

    def test_extract_shap_multiclass_metadata(
        self, service, sample_adata, multiclass_model
    ):
        """Test that metadata includes aggregation method and n_classes."""
        adata_result, stats, ir = service.extract_shap_values(
            sample_adata, multiclass_model, model_type="tree", random_state=42
        )

        # Check shap_analysis metadata
        assert "shap_analysis" in adata_result.uns
        shap_meta = adata_result.uns["shap_analysis"]

        assert shap_meta["n_classes"] == 3
        assert shap_meta["aggregation_method"] == "mean_absolute"
        assert shap_meta["model_type"] == "tree"

    def test_extract_shap_multiclass_stats(
        self, service, sample_adata, multiclass_model
    ):
        """Test that stats include per-class top features."""
        adata_result, stats, ir = service.extract_shap_values(
            sample_adata, multiclass_model, model_type="tree", random_state=42
        )

        # Check stats structure
        assert "per_class_top_features" in stats
        assert "class_mapping" in stats
        assert "aggregation_method" in stats
        assert "layers_created" in stats

        # Check per-class top features
        per_class_features = stats["per_class_top_features"]
        assert len(per_class_features) == 3
        assert "shap_class_0" in per_class_features
        assert "shap_class_1" in per_class_features
        assert "shap_class_2" in per_class_features

        # Check each class has 10 features
        for layer_name, features in per_class_features.items():
            assert len(features) == 10
            # Check feature structure
            for feat in features:
                assert "rank" in feat
                assert "feature" in feat
                assert "mean_shap" in feat
                assert feat["rank"] >= 1
                assert feat["mean_shap"] >= 0

        # Check layers_created
        assert "shap_values" in stats["layers_created"]
        assert "shap_class_0" in stats["layers_created"]
        assert "shap_class_1" in stats["layers_created"]
        assert "shap_class_2" in stats["layers_created"]

        # Check aggregation method
        assert stats["aggregation_method"] == "mean_absolute"


class TestBinaryClassification:
    """Test binary classification stores both classes for consistency."""

    def test_extract_shap_binary_both_classes_stored(
        self, service, sample_adata, binary_model
    ):
        """Test that binary models store both classes."""
        adata_result, stats, ir = service.extract_shap_values(
            sample_adata, binary_model, model_type="tree", random_state=42
        )

        # Check both classes exist
        assert "shap_class_0" in adata_result.layers
        assert "shap_class_1" in adata_result.layers

        # For binary, both should have same values
        np.testing.assert_array_almost_equal(
            adata_result.layers["shap_class_0"],
            adata_result.layers["shap_class_1"],
            decimal=5,
        )

    def test_extract_shap_binary_positive_class_idx(
        self, service, sample_adata, binary_model
    ):
        """Test that positive_class_idx is set for binary models."""
        adata_result, stats, ir = service.extract_shap_values(
            sample_adata, binary_model, model_type="tree", random_state=42
        )

        # Check positive_class_idx is set
        assert "shap_analysis" in adata_result.uns
        assert "positive_class_idx" in adata_result.uns["shap_analysis"]
        assert adata_result.uns["shap_analysis"]["positive_class_idx"] == 1

    def test_extract_shap_binary_per_class_features(
        self, service, sample_adata, binary_model
    ):
        """Test that binary models have per-class features for both classes."""
        adata_result, stats, ir = service.extract_shap_values(
            sample_adata, binary_model, model_type="tree", random_state=42
        )

        # Check per-class top features has 2 entries
        assert len(stats["per_class_top_features"]) == 2
        assert "shap_class_0" in stats["per_class_top_features"]
        assert "shap_class_1" in stats["per_class_top_features"]


class TestBackwardCompatibility:
    """Test backward compatibility with aggregate layer."""

    def test_aggregate_layer_always_exists(
        self, service, sample_adata, multiclass_model
    ):
        """Test that shap_values aggregate layer always exists."""
        adata_result, stats, ir = service.extract_shap_values(
            sample_adata, multiclass_model, model_type="tree", random_state=42
        )

        # Aggregate layer must exist
        assert "shap_values" in adata_result.layers

    def test_global_importance_calculated(
        self, service, sample_adata, multiclass_model
    ):
        """Test that global importance is calculated from aggregate."""
        adata_result, stats, ir = service.extract_shap_values(
            sample_adata, multiclass_model, model_type="tree", random_state=42
        )

        # Check var columns exist
        assert "shap_importance" in adata_result.var
        assert "shap_importance_std" in adata_result.var
        assert "shap_rank" in adata_result.var

        # Verify importance calculated from aggregate
        expected_importance = np.mean(adata_result.layers["shap_values"], axis=0)
        np.testing.assert_array_almost_equal(
            adata_result.var["shap_importance"].values, expected_importance, decimal=5
        )


class TestAggregationDiscovery:
    """Test per-class layer discovery in aggregation."""

    def test_aggregate_shap_discovers_per_class_layers(
        self, service, sample_adata, multiclass_model
    ):
        """Test that aggregate_shap_to_global discovers per-class layers."""
        # First extract SHAP values
        adata_result, _, _ = service.extract_shap_values(
            sample_adata, multiclass_model, model_type="tree", random_state=42
        )

        # Now aggregate
        adata_agg, stats_agg, ir_agg = service.aggregate_shap_to_global(
            adata_result, normalize=True
        )

        # Check that per-class importance columns were created
        assert "shap_class_0_importance" in adata_agg.var
        assert "shap_class_1_importance" in adata_agg.var
        assert "shap_class_2_importance" in adata_agg.var

        # Verify per-class importance calculated correctly
        for class_idx in range(3):
            layer_name = f"shap_class_{class_idx}"
            var_col = f"{layer_name}_importance"
            expected = np.mean(adata_result.layers[layer_name], axis=0)
            np.testing.assert_array_almost_equal(
                adata_agg.var[var_col].values, expected, decimal=5
            )

    def test_aggregate_shap_without_per_class_layers(self, service, sample_adata):
        """Test that aggregation works without per-class layers (backward compat)."""
        # Manually create adata with only aggregate layer
        adata = sample_adata.copy()
        adata.layers["shap_values"] = np.abs(np.random.randn(100, 50))

        # Should work without error
        adata_agg, stats_agg, ir_agg = service.aggregate_shap_to_global(
            adata, normalize=True
        )

        # Check global importance calculated
        assert "global_importance" in adata_agg.var
        assert "global_importance_pct" in adata_agg.var
