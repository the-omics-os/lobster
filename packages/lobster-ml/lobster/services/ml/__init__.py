"""
Machine Learning services package.

Provides ML preparation, feature selection, cross-validation, interpretability,
and sparse matrix utilities.
"""

from lobster.services.ml.sparse_utils import (
    SparseConversionError,
    check_sparse_conversion_safe,
    estimate_dense_memory_gb,
)

__all__ = [
    "SparseConversionError",
    "check_sparse_conversion_safe",
    "estimate_dense_memory_gb",
]
