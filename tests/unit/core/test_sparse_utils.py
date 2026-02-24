"""Tests for lobster.core.sparse_utils — sparse-to-dense safety guards."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from lobster.core.sparse_utils import (
    SparseConversionError,
    check_sparse_conversion_safe,
    estimate_dense_memory_gb,
    safe_toarray,
)


# ---------------------------------------------------------------------------
# TestEstimateDenseMemory
# ---------------------------------------------------------------------------


class TestEstimateDenseMemory:
    def test_small_matrix(self):
        X = csr_matrix(np.zeros((100, 50)))
        gb = estimate_dense_memory_gb(X)
        # 100 * 50 * 8 bytes = 40_000 bytes
        expected = 40_000 / (1024**3)
        assert abs(gb - expected) < 1e-12

    def test_larger_matrix(self):
        X = csr_matrix((10_000, 5_000))
        gb = estimate_dense_memory_gb(X)
        expected = 10_000 * 5_000 * 8 / (1024**3)
        assert abs(gb - expected) < 1e-12

    def test_raises_on_dense_input(self):
        with pytest.raises(TypeError, match="sparse matrix"):
            estimate_dense_memory_gb(np.zeros((10, 10)))


# ---------------------------------------------------------------------------
# TestSparseConversionError
# ---------------------------------------------------------------------------


class TestSparseConversionError:
    def test_inherits_memory_error(self):
        err = SparseConversionError("msg", required_gb=10.0, available_gb=2.0, shape=(1000, 500))
        assert isinstance(err, MemoryError)
        assert isinstance(err, ValueError)

    def test_attributes(self):
        err = SparseConversionError("msg", required_gb=10.0, available_gb=2.0, shape=(1000, 500))
        assert err.required_gb == 10.0
        assert err.available_gb == 2.0
        assert err.shape == (1000, 500)

    def test_message(self):
        err = SparseConversionError("boom", required_gb=1.0, available_gb=0.5, shape=(10, 10))
        assert str(err) == "boom"


# ---------------------------------------------------------------------------
# TestCheckSparseConversionSafe
# ---------------------------------------------------------------------------


class TestCheckSparseConversionSafe:
    def test_dense_input_passes(self):
        """Dense arrays should always pass without checking memory."""
        X = np.zeros((100_000, 50_000))  # Would be huge if sparse
        # Should not raise — it's already dense, no conversion needed
        check_sparse_conversion_safe(X)

    def test_small_sparse_passes(self):
        """Small sparse matrices should pass on any machine."""
        X = csr_matrix(np.zeros((10, 5)))
        check_sparse_conversion_safe(X)

    def test_raises_when_memory_insufficient(self):
        """Mock psutil to simulate OOM condition."""
        X = csr_matrix((1000, 500))

        mock_mem = MagicMock()
        # Report only 1 KB available — any matrix will fail
        mock_mem.available = 1024

        with patch("lobster.core.sparse_utils.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_mem
            with pytest.raises(SparseConversionError) as exc_info:
                check_sparse_conversion_safe(X)

            err = exc_info.value
            assert err.shape == (1000, 500)
            assert err.available_gb < err.required_gb

    def test_custom_overhead_multiplier(self):
        """Higher overhead should make the check stricter."""
        X = csr_matrix((100, 50))

        mock_mem = MagicMock()
        # Set available memory to just barely enough for 1.0x but not 3.0x
        dense_bytes = 100 * 50 * 8
        mock_mem.available = int(dense_bytes * 2)  # 2x the dense size

        with patch("lobster.core.sparse_utils.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_mem

            # 1.5x overhead: needs 1.5 * dense ≈ 60KB, have 80KB → passes
            check_sparse_conversion_safe(X, overhead_multiplier=1.5)

            # 3.0x overhead: needs 3.0 * dense ≈ 120KB, have 80KB → fails
            with pytest.raises(SparseConversionError):
                check_sparse_conversion_safe(X, overhead_multiplier=3.0)


# ---------------------------------------------------------------------------
# TestSafeToarray
# ---------------------------------------------------------------------------


class TestSafeToarray:
    def test_dense_passthrough(self):
        """Dense input should be returned as-is (no copy by default)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = safe_toarray(X)
        assert isinstance(result, np.ndarray)
        # Should share memory (no copy)
        assert np.shares_memory(result, X)

    def test_dense_copy(self):
        """copy=True should return a new array."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = safe_toarray(X, copy=True)
        assert isinstance(result, np.ndarray)
        assert not np.shares_memory(result, X)
        np.testing.assert_array_equal(result, X)

    def test_sparse_conversion(self):
        """Small sparse matrix should be converted to dense."""
        dense = np.array([[1.0, 0.0], [0.0, 2.0]])
        X = csr_matrix(dense)
        result = safe_toarray(X)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, dense)

    def test_sparse_oom_raises(self):
        """Mock OOM condition — should raise SparseConversionError."""
        X = csr_matrix((1000, 500))

        mock_mem = MagicMock()
        mock_mem.available = 1024  # 1 KB

        with patch("lobster.core.sparse_utils.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_mem
            with pytest.raises(SparseConversionError):
                safe_toarray(X)

    def test_values_preserved(self):
        """Verify actual values survive round-trip."""
        rng = np.random.default_rng(42)
        dense = rng.random((50, 30))
        dense[dense < 0.7] = 0  # Make it sparse-ish
        X = csr_matrix(dense)
        result = safe_toarray(X)
        np.testing.assert_array_almost_equal(result, dense)


# ---------------------------------------------------------------------------
# TestBackwardCompatImport
# ---------------------------------------------------------------------------


class TestBackwardCompatImport:
    def test_ml_package_reexport(self):
        """The ML package shim should re-export all core symbols.

        Only runs when lobster-ml is installed (provides lobster.services.ml namespace).
        """
        try:
            from lobster.services.ml.sparse_utils import (
                SparseConversionError as MLError,
            )
            from lobster.services.ml.sparse_utils import (
                check_sparse_conversion_safe as ml_check,
            )
            from lobster.services.ml.sparse_utils import (
                estimate_dense_memory_gb as ml_estimate,
            )
            from lobster.services.ml.sparse_utils import safe_toarray as ml_safe
        except ModuleNotFoundError:
            pytest.skip("lobster-ml not installed — backward-compat shim not testable")

        # They should be the exact same objects
        assert MLError is SparseConversionError
        assert ml_check is check_sparse_conversion_safe
        assert ml_estimate is estimate_dense_memory_gb
        assert ml_safe is safe_toarray
