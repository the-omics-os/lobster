"""
Unit tests for sparse matrix utilities.

Tests memory estimation accuracy, SparseConversionError, and check_sparse_conversion_safe.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from lobster.services.ml.sparse_utils import (
    SparseConversionError,
    check_sparse_conversion_safe,
    estimate_dense_memory_gb,
)


class TestEstimateDenseMemory:
    """Tests for estimate_dense_memory_gb function."""

    def test_small_matrix_estimation(self):
        """Small matrix returns correct GB estimate."""
        # 1000 x 500 x 8 bytes = 4MB = 0.00372529... GB
        X = csr_matrix((1000, 500))
        expected_gb = 1000 * 500 * 8 / (1024**3)
        result = estimate_dense_memory_gb(X)
        assert abs(result - expected_gb) < 1e-9

    def test_large_matrix_estimation(self):
        """Large matrix GB estimate is reasonable."""
        # 50000 x 20000 x 8 bytes = 8GB
        X = csr_matrix((50000, 20000))
        result = estimate_dense_memory_gb(X)
        expected_gb = 50000 * 20000 * 8 / (1024**3)
        assert abs(result - expected_gb) < 0.01

    def test_single_row_matrix(self):
        """Single row matrix returns correct estimate."""
        X = csr_matrix((1, 10000))
        expected_gb = 10000 * 8 / (1024**3)
        result = estimate_dense_memory_gb(X)
        assert abs(result - expected_gb) < 1e-9

    def test_single_column_matrix(self):
        """Single column matrix returns correct estimate."""
        X = csr_matrix((10000, 1))
        expected_gb = 10000 * 8 / (1024**3)
        result = estimate_dense_memory_gb(X)
        assert abs(result - expected_gb) < 1e-9


class TestSparseConversionError:
    """Tests for SparseConversionError exception."""

    def test_is_valueerror_subclass(self):
        """SparseConversionError is ValueError subclass."""
        assert issubclass(SparseConversionError, ValueError)

    def test_exception_attributes(self):
        """Exception stores required_gb, available_gb, shape."""
        exc = SparseConversionError(
            "test message",
            required_gb=8.0,
            available_gb=4.0,
            shape=(50000, 20000),
        )
        assert exc.required_gb == 8.0
        assert exc.available_gb == 4.0
        assert exc.shape == (50000, 20000)

    def test_exception_message(self):
        """Exception message includes actionable guidance."""
        exc = SparseConversionError(
            "Dataset too large for dense conversion.",
            required_gb=8.0,
            available_gb=4.0,
            shape=(50000, 20000),
        )
        assert "dense conversion" in str(exc).lower()


class TestCheckSparseConversionSafe:
    """Tests for check_sparse_conversion_safe function."""

    def test_dense_array_passes(self):
        """Dense numpy array passes without check."""
        X = np.random.rand(100, 50)
        # Should not raise
        check_sparse_conversion_safe(X)

    def test_small_sparse_passes(self):
        """Small sparse matrix passes check."""
        X = csr_matrix(np.random.rand(100, 50))
        # Should not raise
        check_sparse_conversion_safe(X)

    def test_overhead_multiplier_affects_threshold(self):
        """Higher overhead multiplier is more conservative."""
        X = csr_matrix((1000, 500))
        # Both should pass for small matrix
        check_sparse_conversion_safe(X, overhead_multiplier=1.0)
        check_sparse_conversion_safe(X, overhead_multiplier=2.0)

    def test_raises_for_huge_matrix(self, monkeypatch):
        """Raises SparseConversionError for matrix exceeding memory."""

        # Mock low available memory (1GB)
        class MockMemInfo:
            available = 1 * 1024**3  # 1 GB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockMemInfo())

        # Create matrix that needs 8GB dense
        X = csr_matrix((50000, 20000))
        with pytest.raises(SparseConversionError) as excinfo:
            check_sparse_conversion_safe(X)

        # Check error message has actionable guidance
        error_msg = str(excinfo.value)
        assert "chunked processing" in error_msg.lower() or "reduce features" in error_msg.lower()

    def test_error_includes_matrix_dimensions(self, monkeypatch):
        """Error message includes matrix dimensions."""

        class MockMemInfo:
            available = 1 * 1024**3

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockMemInfo())

        X = csr_matrix((50000, 20000))
        with pytest.raises(SparseConversionError) as excinfo:
            check_sparse_conversion_safe(X)

        error_msg = str(excinfo.value)
        assert "50,000" in error_msg or "50000" in error_msg

    def test_error_includes_memory_requirements(self, monkeypatch):
        """Error message includes GB requirements."""

        class MockMemInfo:
            available = 1 * 1024**3

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockMemInfo())

        X = csr_matrix((50000, 20000))
        with pytest.raises(SparseConversionError) as excinfo:
            check_sparse_conversion_safe(X)

        error_msg = str(excinfo.value)
        assert "GB" in error_msg


class TestLargeDatasetSimulation:
    """Simulates 50k+ cell dataset handling."""

    def test_50k_cell_memory_estimation(self):
        """50k cells x 20k genes estimated correctly."""
        # 50,000 x 20,000 x 8 bytes = 8 GB
        X = csr_matrix((50000, 20000))
        gb = estimate_dense_memory_gb(X)

        expected_gb = 50000 * 20000 * 8 / (1024**3)
        assert abs(gb - expected_gb) < 0.01
        assert gb > 7.0  # Should be ~7.45 GB

    def test_small_feature_subset_passes(self, monkeypatch):
        """After feature selection, 50k cells x 100 genes passes."""

        # Mock moderate memory (8GB available)
        class MockMemInfo:
            available = 8 * 1024**3

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockMemInfo())

        # 50k cells x 100 features = 40 MB (should pass)
        X = csr_matrix((50000, 100))
        check_sparse_conversion_safe(X)  # Should not raise

    def test_actionable_error_suggests_feature_reduction(self, monkeypatch):
        """Error message suggests feature reduction as solution."""

        class MockMemInfo:
            available = 4 * 1024**3  # 4 GB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockMemInfo())

        X = csr_matrix((50000, 20000))
        with pytest.raises(SparseConversionError) as excinfo:
            check_sparse_conversion_safe(X)

        error_msg = str(excinfo.value)
        assert (
            "reduce features" in error_msg.lower()
            or "feature selection" in error_msg.lower()
        )
