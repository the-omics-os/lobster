"""Mann-Whitney (Wilcoxon rank-sum) test over many columns at once.

The p-value is exact at small groups and handles ties; ``scipy.stats.ranksums``
uses a normal approximation without tie correction, and ``mannwhitneyu`` with
its default method uses a normal approximation whenever a value ties, even at
3 v 3.
"""

import math
from typing import Tuple

import numpy as np

# An exact test enumerates every split of the samples into the two groups:
# 12,870 for 8 v 8, 48,620 for 9 v 9
MANN_WHITNEY_EXACT_MAX_SPLITS = 20_000


def mann_whitney_u(x, y, axis):
    """Mann-Whitney U of ``x`` against ``y`` along ``axis``, ties at midranks."""
    from scipy import stats

    ranks = stats.rankdata(np.concatenate([x, y], axis=axis), axis=axis)
    n_x = x.shape[axis]
    return np.take(ranks, np.arange(n_x), axis=axis).sum(axis=axis) - (
        n_x * (n_x + 1) / 2
    )


def mann_whitney(x, y) -> Tuple[np.ndarray, np.ndarray]:
    """Per column (gene): the tie-corrected z of ``x`` against ``y``, and the
    two-sided p-value. The p-value is exact, a permutation test over every split
    (ties handled), while there are at most MANN_WHITNEY_EXACT_MAX_SPLITS of them;
    above, the tie-corrected normal approximation. A normal approximation without
    tie correction gave p = 0.0495 to a complete separation at 3 v 3, where an
    exact test cannot go below 0.10."""
    from scipy import stats

    n_x, n_y = len(x), len(y)
    n = n_x + n_y
    ranks = stats.rankdata(np.vstack([x, y]), axis=0)
    u = ranks[:n_x].sum(axis=0) - n_x * (n_x + 1) / 2
    # the permutation variance of U, which accounts for ties
    spread = ((ranks - (n + 1) / 2) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (u - n_x * n_y / 2) / np.sqrt(n_x * n_y * spread / (n * (n - 1)))
    if math.comb(n, n_x) <= MANN_WHITNEY_EXACT_MAX_SPLITS:
        p = stats.permutation_test(
            (x, y),
            mann_whitney_u,
            permutation_type="independent",
            n_resamples=np.inf,
            vectorized=True,
            axis=0,
        ).pvalue
    else:
        p = stats.mannwhitneyu(x, y, method="asymptotic", axis=0).pvalue
    return z, np.asarray(p, dtype=float)
