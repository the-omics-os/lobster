"""
Sparse matrix utilities for memory-safe operations.

Backward-compatibility shim — all implementations now live in
``lobster.core.sparse_utils``. This module re-exports them so existing
``from lobster.services.ml.sparse_utils import ...`` continues to work.
"""

from lobster.core.sparse_utils import (  # noqa: F401
    SparseConversionError,
    check_sparse_conversion_safe,
    estimate_dense_memory_gb,
    safe_toarray,
)

__all__ = [
    "SparseConversionError",
    "check_sparse_conversion_safe",
    "estimate_dense_memory_gb",
    "safe_toarray",
]
