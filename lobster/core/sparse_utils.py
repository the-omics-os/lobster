"""
Sparse matrix utilities for memory-safe operations.

Provides memory estimation and safety checks for sparse-to-dense conversions.
All code performing .toarray() on potentially large matrices should use
``safe_toarray()`` or ``check_sparse_conversion_safe()`` first.
"""

from typing import Any, Union

import numpy as np
import psutil
from scipy.sparse import issparse, spmatrix

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class SparseConversionError(MemoryError, ValueError):
    """
    Raised when sparse-to-dense conversion would exceed available memory.

    Inherits from both MemoryError and ValueError for compatibility
    with memory-check and validation error handlers.

    Attributes:
        required_gb: Memory required (GB) including overhead
        available_gb: Memory available (GB) on system
        shape: Matrix dimensions (n_rows, n_cols)
    """

    def __init__(
        self, message: str, required_gb: float, available_gb: float, shape: tuple
    ):
        super().__init__(message)
        self.required_gb = required_gb
        self.available_gb = available_gb
        self.shape = shape


def estimate_dense_memory_gb(X: spmatrix) -> float:
    """
    Estimate memory required for dense conversion of sparse matrix.

    Args:
        X: Sparse matrix (scipy.sparse format)

    Returns:
        Estimated memory in GB (float64 assumption: 8 bytes per element)
    """
    if not issparse(X):
        raise TypeError("X must be a sparse matrix (scipy.sparse)")

    n_rows, n_cols = X.shape
    # float64 = 8 bytes per element
    bytes_required = n_rows * n_cols * 8
    gb_required = bytes_required / (1024**3)

    return gb_required


def _calculate_max_safe_cells(
    n_features: int, available_gb: float, overhead: float = 1.5
) -> int:
    """
    Calculate max cells that can safely fit with given features.

    Used in error messages for actionable guidance.

    Args:
        n_features: Number of features (columns)
        available_gb: Available system memory (GB)
        overhead: Memory overhead multiplier

    Returns:
        Maximum number of cells (rows) that can safely convert
    """
    max_cells = int((available_gb * (1024**3)) / (n_features * 8 * overhead))
    return max_cells


def check_sparse_conversion_safe(
    X: Union[np.ndarray, spmatrix], overhead_multiplier: float = 1.5
) -> None:
    """
    Check if sparse-to-dense conversion is memory-safe.

    Raises SparseConversionError if conversion would exceed available memory.
    Returns normally (None) if safe or if X is already dense.

    Args:
        X: Matrix to check (may be sparse or dense)
        overhead_multiplier: Memory overhead multiplier (default 1.5x)
            Accounts for intermediate copies during conversion

    Raises:
        SparseConversionError: If conversion would exceed available memory
    """
    # If already dense, nothing to check
    if not issparse(X):
        return

    # Get matrix dimensions
    n_rows, n_cols = X.shape

    # Estimate dense memory requirement
    dense_gb = estimate_dense_memory_gb(X)
    required_gb = dense_gb * overhead_multiplier

    # Get available system memory
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)

    # Check if safe
    if required_gb > available_gb:
        max_cells = _calculate_max_safe_cells(n_cols, available_gb, overhead_multiplier)

        error_msg = (
            f"Insufficient memory for dense conversion of {n_rows:,} x {n_cols:,} sparse matrix.\n"
            f"\n"
            f"  Matrix: {n_rows:,} x {n_cols:,} ({dense_gb:.1f} GB dense)\n"
            f"  Required: {required_gb:.1f} GB (with {overhead_multiplier}x overhead)\n"
            f"  Available: {available_gb:.1f} GB\n"
            f"\n"
            f"Options:\n"
            f"  1. Reduce features via filtering or selection first\n"
            f"  2. Subsample observations to <{max_cells:,}\n"
            f"  3. Use chunked processing if the operation supports it\n"
            f"  4. Increase available RAM"
        )

        raise SparseConversionError(
            error_msg,
            required_gb=required_gb,
            available_gb=available_gb,
            shape=(n_rows, n_cols),
        )

    # Safe - log for transparency
    logger.info(
        f"Sparse conversion safe: {n_rows:,} x {n_cols:,} "
        f"({dense_gb:.1f} GB dense, {required_gb:.1f} GB with overhead, "
        f"{available_gb:.1f} GB available)"
    )


def safe_toarray(
    X: Union[np.ndarray, spmatrix],
    overhead_multiplier: float = 1.5,
    copy: bool = False,
) -> np.ndarray:
    """
    Safely convert a matrix to dense ndarray with memory guard.

    Combines ``check_sparse_conversion_safe()`` and ``.toarray()`` in one call.
    If *X* is already dense, returns it directly (or a copy if *copy=True*).

    Args:
        X: Matrix (sparse or dense).
        overhead_multiplier: Memory overhead multiplier for sparse check.
        copy: If True, always return a new array (even for dense input).

    Returns:
        Dense numpy ndarray.

    Raises:
        SparseConversionError: If sparse conversion would exceed available memory.
    """
    if issparse(X):
        check_sparse_conversion_safe(X, overhead_multiplier=overhead_multiplier)
        return X.toarray()

    # Already dense
    return np.array(X, copy=copy)
