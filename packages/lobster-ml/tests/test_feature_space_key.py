"""
Unit tests for feature_space_key parameter across ML services.

Tests verify:
- Parameter exists in expected methods
- KeyError raised when key not found
- Correct feature extraction when key provided
- Backward compatibility (None uses adata.X)
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from lobster.services.analysis.survival_analysis_service import SurvivalAnalysisService
from lobster.services.ml.feature_selection_service import FeatureSelectionService
from lobster.services.ml.interpretability_service import InterpretabilityService
from lobster.services.quality.cross_validation_service import CrossValidationService


def _has_sksurv():
    """Check if scikit-survival is installed."""
    try:
        import sksurv

        return True
    except ImportError:
        return False


@pytest.fixture
def adata_with_factors():
    """
    AnnData with both X and obsm['X_mofa'] populated.

    - n_samples: 100
    - n_genes (X): 500
    - n_factors (obsm['X_mofa']): 15
    - obs columns: target (binary), time (continuous), event (binary)
    """
    n_samples, n_genes, n_factors = 100, 500, 15

    adata = AnnData(
        X=np.random.rand(n_samples, n_genes),
        obs=pd.DataFrame(
            {
                "target": np.random.randint(0, 2, n_samples),
                "time": np.random.rand(n_samples) * 100,
                "event": np.random.randint(0, 2, n_samples),
            },
            index=[f"sample_{i}" for i in range(n_samples)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )

    # Add integrated factors
    adata.obsm["X_mofa"] = np.random.rand(n_samples, n_factors)

    return adata


class TestFeatureSelectionService:
    """Tests for feature_space_key in FeatureSelectionService."""

    def test_validate_and_extract_features_with_key(self, adata_with_factors):
        """_validate_and_extract_features uses obsm when key provided."""
        service = FeatureSelectionService()

        X, feature_names = service._validate_and_extract_features(
            adata_with_factors, feature_space_key="X_mofa"
        )

        # Should have extracted factors (15) not genes (500)
        assert X.shape == (100, 15), f"Expected (100, 15), got {X.shape}"
        assert len(feature_names) == 15

    def test_validate_and_extract_features_backward_compatible(
        self, adata_with_factors
    ):
        """_validate_and_extract_features with None uses adata.X."""
        service = FeatureSelectionService()

        X, feature_names = service._validate_and_extract_features(
            adata_with_factors, feature_space_key=None
        )

        # Should have used genes (500)
        assert X.shape == (100, 500), f"Expected (100, 500), got {X.shape}"
        assert len(feature_names) == 500

    def test_validate_and_extract_features_keyerror(self, adata_with_factors):
        """KeyError raised when key not in obsm."""
        service = FeatureSelectionService()

        with pytest.raises(KeyError) as exc_info:
            service._validate_and_extract_features(
                adata_with_factors, feature_space_key="X_nonexistent"
            )

        error_msg = str(exc_info.value)
        # Error should mention the missing key
        assert "X_nonexistent" in error_msg
        # Error should list available keys
        assert "X_mofa" in error_msg or "Available keys" in error_msg

    def test_validate_and_extract_features_lists_available_keys(
        self, adata_with_factors
    ):
        """KeyError message lists available obsm keys."""
        service = FeatureSelectionService()

        # Add multiple obsm keys
        adata_with_factors.obsm["X_pca"] = np.random.rand(100, 20)
        adata_with_factors.obsm["X_umap"] = np.random.rand(100, 2)

        with pytest.raises(KeyError) as exc_info:
            service._validate_and_extract_features(
                adata_with_factors, feature_space_key="X_wrong"
            )

        error_msg = str(exc_info.value)
        # Should list available keys
        assert "X_mofa" in error_msg
        assert "X_pca" in error_msg


class TestSurvivalAnalysisService:
    """Tests for feature_space_key in SurvivalAnalysisService."""

    def test_validate_and_extract_features_exists(self):
        """Service has _validate_and_extract_features method."""
        service = SurvivalAnalysisService()
        assert hasattr(service, "_validate_and_extract_features")

    def test_validate_and_extract_features_with_key(self, adata_with_factors):
        """_validate_and_extract_features uses obsm when key provided."""
        service = SurvivalAnalysisService()

        X, feature_names = service._validate_and_extract_features(
            adata_with_factors, feature_space_key="X_mofa"
        )

        # Should have extracted factors (15)
        assert X.shape == (100, 15), f"Expected (100, 15), got {X.shape}"
        assert len(feature_names) == 15

    def test_validate_and_extract_features_keyerror(self, adata_with_factors):
        """KeyError raised when key not in obsm."""
        service = SurvivalAnalysisService()

        with pytest.raises(KeyError) as exc_info:
            service._validate_and_extract_features(
                adata_with_factors, feature_space_key="X_nonexistent"
            )

        error_msg = str(exc_info.value)
        assert "X_nonexistent" in error_msg


class TestCrossValidationService:
    """Tests for feature_space_key in CrossValidationService."""

    def test_validate_and_extract_features_exists(self):
        """Service has _validate_and_extract_features method."""
        service = CrossValidationService()
        assert hasattr(service, "_validate_and_extract_features")

    def test_validate_and_extract_features_with_key(self, adata_with_factors):
        """_validate_and_extract_features uses obsm when key provided."""
        service = CrossValidationService()

        X, feature_names = service._validate_and_extract_features(
            adata_with_factors, feature_space_key="X_mofa"
        )

        # Should have extracted factors (15)
        assert X.shape == (100, 15), f"Expected (100, 15), got {X.shape}"
        assert len(feature_names) == 15

    def test_validate_and_extract_features_backward_compatible(
        self, adata_with_factors
    ):
        """_validate_and_extract_features with None uses adata.X."""
        service = CrossValidationService()

        X, feature_names = service._validate_and_extract_features(
            adata_with_factors, feature_space_key=None
        )

        # Should have used genes (500)
        assert X.shape == (100, 500), f"Expected (100, 500), got {X.shape}"
        assert len(feature_names) == 500

    def test_validate_and_extract_features_keyerror(self, adata_with_factors):
        """KeyError raised when key not in obsm."""
        service = CrossValidationService()

        with pytest.raises(KeyError) as exc_info:
            service._validate_and_extract_features(
                adata_with_factors, feature_space_key="X_nonexistent"
            )

        error_msg = str(exc_info.value)
        assert "X_nonexistent" in error_msg


class TestInterpretabilityService:
    """Tests for feature_space_key in InterpretabilityService."""

    def test_validate_and_extract_features_exists(self):
        """Service has _validate_and_extract_features method."""
        service = InterpretabilityService()
        assert hasattr(service, "_validate_and_extract_features")

    def test_validate_and_extract_features_with_key(self, adata_with_factors):
        """_validate_and_extract_features uses obsm when key provided."""
        service = InterpretabilityService()

        X, feature_names = service._validate_and_extract_features(
            adata_with_factors, feature_space_key="X_mofa"
        )

        # Should have extracted factors (15)
        assert X.shape == (100, 15), f"Expected (100, 15), got {X.shape}"
        assert len(feature_names) == 15

    def test_validate_and_extract_features_backward_compatible(
        self, adata_with_factors
    ):
        """_validate_and_extract_features with None uses adata.X."""
        service = InterpretabilityService()

        X, feature_names = service._validate_and_extract_features(
            adata_with_factors, feature_space_key=None
        )

        # Should have used genes (500)
        assert X.shape == (100, 500), f"Expected (100, 500), got {X.shape}"
        assert len(feature_names) == 500

    def test_validate_and_extract_features_keyerror(self, adata_with_factors):
        """KeyError raised when key not in obsm."""
        service = InterpretabilityService()

        with pytest.raises(KeyError) as exc_info:
            service._validate_and_extract_features(
                adata_with_factors, feature_space_key="X_nonexistent"
            )

        error_msg = str(exc_info.value)
        assert "X_nonexistent" in error_msg


class TestConsistentErrorMessages:
    """Tests that all services provide consistent error messages."""

    def test_all_services_suggest_available_keys(self, adata_with_factors):
        """All services list available keys in KeyError."""
        services = [
            FeatureSelectionService(),
            SurvivalAnalysisService(),
            CrossValidationService(),
            InterpretabilityService(),
        ]

        for service in services:
            with pytest.raises(KeyError) as exc_info:
                service._validate_and_extract_features(
                    adata_with_factors, feature_space_key="X_missing"
                )

            error_msg = str(exc_info.value)
            # All should mention available keys
            assert "X_mofa" in error_msg or "available" in error_msg.lower(), (
                f"{service.__class__.__name__} error message doesn't list available keys"
            )

    def test_all_services_suggest_integration(self, adata_with_factors):
        """All services suggest running integration in KeyError."""
        # Remove factors to simulate pre-integration state
        adata_no_factors = adata_with_factors.copy()
        del adata_no_factors.obsm["X_mofa"]

        services = [
            FeatureSelectionService(),
            SurvivalAnalysisService(),
            CrossValidationService(),
            InterpretabilityService(),
        ]

        for service in services:
            with pytest.raises(KeyError) as exc_info:
                service._validate_and_extract_features(
                    adata_no_factors, feature_space_key="X_mofa"
                )

            error_msg = str(exc_info.value)
            # Should suggest integration
            assert "integration" in error_msg.lower() or "obsm" in error_msg.lower(), (
                f"{service.__class__.__name__} error message doesn't suggest integration"
            )
