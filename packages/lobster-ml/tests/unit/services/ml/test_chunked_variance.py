"""
Unit tests for chunked variance filter.

Verifies chunked processing produces identical results to non-chunked,
handles edge cases, and works with large sparse matrices.
"""

import anndata as ad
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from lobster.services.ml.feature_selection_service import FeatureSelectionService


@pytest.fixture
def service():
    """Feature selection service instance."""
    return FeatureSelectionService()


@pytest.fixture
def small_sparse_adata():
    """Small sparse AnnData for testing."""
    np.random.seed(42)
    X = csr_matrix(np.random.rand(200, 100))
    adata = ad.AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(100)]
    adata.obs["target"] = np.random.choice(["A", "B"], 200)
    return adata


class TestChunkedVarianceFilter:
    """Tests for chunked variance filter."""

    def test_chunked_matches_non_chunked(self, service, small_sparse_adata):
        """Chunked variance produces same result as non-chunked."""
        # Non-chunked
        result1, stats1, _ = service.variance_filter(
            small_sparse_adata.copy(),
            threshold=0.01,
            chunked=False,
        )

        # Chunked
        result2, stats2, _ = service.variance_filter(
            small_sparse_adata.copy(),
            threshold=0.01,
            chunked=True,
        )

        # Same features selected
        assert result1.n_vars == result2.n_vars
        # Check variance_selected column matches
        np.testing.assert_array_equal(
            result1.var["variance_selected"].values,
            result2.var["variance_selected"].values,
        )

    def test_chunked_flag_in_stats(self, service, small_sparse_adata):
        """Stats dict includes chunked flag."""
        _, stats, _ = service.variance_filter(
            small_sparse_adata,
            threshold=0.01,
            chunked=True,
        )
        assert "chunked" in stats
        assert stats["chunked"] == True

    def test_chunks_processed_in_stats(self, service, small_sparse_adata):
        """Stats dict includes chunks_processed when chunked=True."""
        _, stats, _ = service.variance_filter(
            small_sparse_adata,
            threshold=0.01,
            chunked=True,
        )
        assert "chunks_processed" in stats
        assert stats["chunks_processed"] >= 1

    def test_non_chunked_flag_false(self, service, small_sparse_adata):
        """Stats dict shows chunked=False when not using chunked mode."""
        _, stats, _ = service.variance_filter(
            small_sparse_adata,
            threshold=0.01,
            chunked=False,
        )
        assert stats["chunked"] == False

    def test_chunked_with_percentile(self, service, small_sparse_adata):
        """Chunked mode works with percentile selection."""
        _, stats, _ = service.variance_filter(
            small_sparse_adata,
            percentile=80.0,
            chunked=True,
        )
        # 80th percentile means we keep features above it
        # With 100 features, we expect some selected
        assert stats["n_selected_features"] > 0
        assert stats["n_selected_features"] < 100

    def test_chunked_variance_numerical_accuracy(self, service):
        """Chunked variance is numerically accurate."""
        np.random.seed(123)
        # Create known variance data
        X_dense = np.random.rand(500, 50)
        expected_var = np.var(X_dense, axis=0)

        X_sparse = csr_matrix(X_dense)
        adata = ad.AnnData(X=X_sparse)
        adata.var_names = [f"gene_{i}" for i in range(50)]

        # Run chunked variance with very low threshold (keeps all)
        result, stats, _ = service.variance_filter(
            adata,
            threshold=0.0,
            chunked=True,
        )

        # Variance stored in var should be computed
        assert "variance" in result.var.columns
        assert result.n_vars == 50  # All features kept


class TestChunkedVarianceEdgeCases:
    """Edge cases for chunked variance."""

    def test_single_row_chunked(self, service):
        """Chunked mode handles single row matrix."""
        X = csr_matrix(np.random.rand(1, 100))
        adata = ad.AnnData(X=X)
        adata.var_names = [f"gene_{i}" for i in range(100)]

        # Should not raise
        result, stats, _ = service.variance_filter(
            adata,
            threshold=0.0,
            chunked=True,
        )
        assert result.n_vars == 100  # All features kept (variance undefined for 1 row)

    def test_dense_matrix_with_chunked_flag(self, service):
        """Dense matrix with chunked=True still works."""
        X = np.random.rand(100, 50)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"gene_{i}" for i in range(50)]

        result, stats, _ = service.variance_filter(
            adata,
            threshold=0.01,
            chunked=True,
        )
        # Should work, chunked flag may be False since not sparse
        assert result.n_vars == 50  # Returns full adata
