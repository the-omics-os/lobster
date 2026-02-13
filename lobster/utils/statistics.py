"""
Shared statistical utilities for Lobster services.

This module provides common statistical functions used across multiple services,
ensuring consistent implementations and reducing code duplication.
"""

from typing import List

import numpy as np


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """
    Apply Benjamini-Hochberg FDR correction to p-values.

    The Benjamini-Hochberg procedure controls the false discovery rate (FDR)
    by adjusting p-values to account for multiple testing. This is the standard
    method for proteomics and genomics analyses.

    Args:
        p_values: List of raw p-values to correct

    Returns:
        List of FDR-adjusted p-values (q-values), capped at 1.0

    Example:
        >>> p_values = [0.01, 0.03, 0.05, 0.10, 0.50]
        >>> fdr = benjamini_hochberg(p_values)
        >>> all(f >= p for f, p in zip(fdr, p_values))
        True

    References:
        Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
        rate: a practical and powerful approach to multiple testing.
        Journal of the Royal Statistical Society, Series B, 57(1), 289-300.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Get sorted indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Calculate FDR: q_i = p_i * n / rank_i
    fdr = np.zeros(n)
    for i in range(n):
        fdr[sorted_indices[i]] = sorted_p[i] * n / (i + 1)

    # Ensure monotonicity (going from largest to smallest rank)
    # q-values should be non-decreasing when sorted by p-value
    fdr_monotonic = np.zeros(n)
    inverse_sorted = np.argsort(sorted_indices)
    fdr_sorted = fdr[sorted_indices]

    running_min = 1.0
    for i in range(n - 1, -1, -1):
        running_min = min(running_min, fdr_sorted[i])
        fdr_monotonic[i] = running_min

    # Cap at 1.0 and return in original order
    fdr_final = np.minimum(fdr_monotonic[inverse_sorted], 1.0)

    return fdr_final.tolist()
