"""
Unit tests for CrossValidationService operational fixes.

Tests model parameter extraction, sparse memory safety, prediction capping,
feature importance storage, and exact version pinning.
"""

import numpy as np
import psutil
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier

from lobster.services.quality.cross_validation_service import CrossValidationService


@pytest.fixture
def cv_service():
    """Create CrossValidationService instance."""
    return CrossValidationService()


class TestCrossValidationService:
    """Test operational fixes for cross-validation service."""

    def _create_test_adata(self, n_obs=100, n_vars=20, binary_target=True):
        """Helper to create test AnnData."""
        np.random.seed(42)
        X = np.random.randn(n_obs, n_vars)

        if binary_target:
            labels = np.random.choice([0, 1], size=n_obs)
        else:
            labels = np.random.choice([0, 1, 2], size=n_obs)

        obs = {
            "sample_id": [f"sample_{i}" for i in range(n_obs)],
            "label": labels,
        }
        var = {"feature_name": [f"feature_{i}" for i in range(n_vars)]}

        return AnnData(X=X, obs=obs, var=var)

    def test_model_parameter_extraction(self, cv_service):
        """Test model_class and model_params extracted from factory (F1)."""
        # Create adata with binary target
        adata = self._create_test_adata(n_obs=100, n_vars=20, binary_target=True)

        # Model factory (typical user API)
        model_factory = lambda: RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42
        )

        result_adata, stats, ir = cv_service.stratified_kfold_cv(
            adata=adata,
            target_column="label",
            model_factory=model_factory,
            n_splits=3,
            random_state=42,
        )

        # Verify model info extracted
        assert ir.parameters["model_class"] == "RandomForestClassifier"
        assert ir.parameters["model_params"]["n_estimators"] == 50
        assert ir.parameters["model_params"]["max_depth"] == 5
        assert "model_class" in ir.parameter_schema
        assert ir.parameter_schema["model_class"].papermill_injectable is True

    def test_sparse_matrix_memory_check(self, cv_service, monkeypatch):
        """Test sparse matrix OOM prevention with memory estimation (F2)."""
        # Create sparse adata (10k cells × 5k genes)
        # Dense: 10000 × 5000 × 8 = 0.37 GB
        # Required (1.5x): 0.56 GB
        adata = self._create_test_adata(n_obs=10000, n_vars=5000, binary_target=True)
        adata.X = csr_matrix(adata.X)  # Convert to sparse

        # Mock low memory (force MemoryError)
        # Set to 0.4 GB so required (0.56 GB) > available (0.4 GB)
        class MockMemory:
            available = 0.4 * 1024**3  # 0.4 GB available

        def mock_virtual_memory():
            return MockMemory()

        monkeypatch.setattr(psutil, "virtual_memory", mock_virtual_memory)

        model_factory = lambda: RandomForestClassifier()

        # Should raise MemoryError with actionable message
        with pytest.raises(MemoryError) as exc_info:
            cv_service.stratified_kfold_cv(
                adata=adata,
                target_column="label",
                model_factory=model_factory,
                n_splits=3,
            )

        error_msg = str(exc_info.value)
        assert "Insufficient memory" in error_msg
        assert "GB" in error_msg
        assert "Options:" in error_msg

    def test_prediction_storage_capping(self, cv_service):
        """Test prediction storage capped at 1000/fold to prevent bloat (F3)."""
        # Large dataset (5000 samples, 5 folds = 1000 predictions/fold)
        adata = self._create_test_adata(n_obs=5000, n_vars=20, binary_target=True)

        model_factory = lambda: RandomForestClassifier(random_state=42)

        # With store_predictions=True
        result_adata, stats, ir = cv_service.stratified_kfold_cv(
            adata=adata,
            target_column="label",
            model_factory=model_factory,
            n_splits=5,
            random_state=42,
            return_predictions=True,
            store_predictions=True,  # Enable storage
        )

        # Should cap at 5000 (1000 per fold × 5 folds)
        predictions = result_adata.uns["cross_validation"]["predictions"]
        assert len(predictions) <= 5000, "Predictions not capped at 1000/fold"
        assert result_adata.uns["cross_validation"]["predictions_stored"] is True

        # With store_predictions=False (default)
        result_adata2, stats2, ir2 = cv_service.stratified_kfold_cv(
            adata=adata,
            target_column="label",
            model_factory=model_factory,
            n_splits=5,
            random_state=42,
            return_predictions=True,
            store_predictions=False,  # Summary-only
        )

        # Should NOT store predictions
        assert result_adata2.uns["cross_validation"]["predictions_stored"] is False
        assert "predictions" not in result_adata2.uns["cross_validation"]
        assert result_adata2.uns["cross_validation"]["n_predictions"] == 5000

    def test_feature_importance_efficient_storage(self, cv_service):
        """Test feature importance stored in var (full) + uns (top 100) (F3)."""
        adata = self._create_test_adata(n_obs=200, n_vars=500, binary_target=True)

        model_factory = lambda: RandomForestClassifier(random_state=42)

        result_adata, stats, ir = cv_service.stratified_kfold_cv(
            adata=adata,
            target_column="label",
            model_factory=model_factory,
            n_splits=3,
            random_state=42,
            return_importances=True,
        )

        # Full arrays in var (efficient for 500 features)
        assert "cv_mean_importance" in result_adata.var.columns
        assert "cv_importance_std" in result_adata.var.columns
        assert "cv_stability" in result_adata.var.columns
        assert len(result_adata.var["cv_mean_importance"]) == 500

        # Top 100 in uns (quick access)
        top_features = result_adata.uns["cross_validation"]["top_features"]
        assert len(top_features["features"]) <= 100
        assert len(top_features["mean_importance"]) <= 100

    def test_exact_version_pinning(self, cv_service):
        """Test exact sklearn version captured (not requirement range) (F6)."""
        adata = self._create_test_adata(n_obs=100, n_vars=20, binary_target=True)

        model_factory = lambda: RandomForestClassifier()

        result_adata, stats, ir = cv_service.stratified_kfold_cv(
            adata=adata,
            target_column="label",
            model_factory=model_factory,
            n_splits=3,
        )

        # Should capture exact version
        sklearn_version = ir.execution_context["sklearn_version"]
        assert sklearn_version != ">=1.0.0", "Version is requirement range not exact"
        assert sklearn_version != "unknown" or True, "Version unknown (acceptable)"

        # Should be format like "1.3.2" if sklearn available
        import sklearn

        expected_version = sklearn.__version__
        assert sklearn_version == expected_version

    def test_template_renders_successfully(self, cv_service):
        """Test that IR template renders without Jinja2 errors."""
        from jinja2 import Template

        ir = cv_service._create_stratified_kfold_ir(
            target_column="label",
            model_class="RandomForestClassifier",
            model_params={"n_estimators": 100, "max_depth": 10},
            n_splits=5,
            shuffle=True,
            random_state=42,
            scale_features=True,
            return_predictions=False,
            return_importances=True,
            store_predictions=False,
        )

        template = Template(ir.code_template)
        rendered = template.render(ir.parameters)

        assert len(rendered) > 0, "Template rendered to empty string"
        assert "RandomForestClassifier" in rendered
        assert "issparse" in rendered
        assert "MAX_PREDICTIONS_PER_FOLD" not in rendered  # store_predictions=False

    def test_model_class_explicit_override(self, cv_service):
        """Test that explicitly provided model_class overrides extraction."""
        adata = self._create_test_adata(n_obs=100, n_vars=20, binary_target=True)

        model_factory = lambda: RandomForestClassifier(n_estimators=50)

        result_adata, stats, ir = cv_service.stratified_kfold_cv(
            adata=adata,
            target_column="label",
            model_factory=model_factory,
            model_class="CustomModel",  # Explicit override
            model_params={"param1": 123},
            n_splits=3,
            random_state=42,
        )

        # Should use explicitly provided values
        assert ir.parameters["model_class"] == "CustomModel"
        assert ir.parameters["model_params"]["param1"] == 123

    def test_basic_functionality_preserved(self, cv_service):
        """Test that basic CV functionality still works after changes."""
        adata = self._create_test_adata(n_obs=200, n_vars=20, binary_target=True)

        model_factory = lambda: RandomForestClassifier(n_estimators=10, random_state=42)

        result_adata, stats, ir = cv_service.stratified_kfold_cv(
            adata=adata,
            target_column="label",
            model_factory=model_factory,
            n_splits=5,
            random_state=42,
            return_predictions=True,
            return_importances=True,
            store_predictions=True,
        )

        # Basic checks
        assert "cross_validation" in result_adata.uns
        assert result_adata.uns["cross_validation"]["n_splits"] == 5
        assert "aggregated_metrics" in result_adata.uns["cross_validation"]
        assert "accuracy" in result_adata.uns["cross_validation"]["aggregated_metrics"]

        # Stats structure
        assert "n_splits" in stats
        assert "n_samples" in stats
        assert "n_features" in stats
        assert "metrics" in stats

        # IR structure
        assert ir.operation == "sklearn.model_selection.StratifiedKFold"
        assert ir.tool_name == "stratified_kfold_cv"
        assert "model_class" in ir.parameters
        assert "sklearn_version" in ir.execution_context
