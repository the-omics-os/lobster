"""Mann-Whitney (Wilcoxon rank-sum) test over many columns at once.

The p-value is exact at small groups and handles ties; ``scipy.stats.ranksums``
uses a normal approximation without tie correction, and ``mannwhitneyu`` with
its default method uses a normal approximation whenever a value ties, even at
3 v 3.
"""

import itertools
import math
from typing import Tuple

import numpy as np

# An exact test enumerates every split of the samples into the two groups:
# 12,870 for 8 v 8, 48,620 for 9 v 9
MANN_WHITNEY_EXACT_MAX_SPLITS = 20_000

# U of every split is computed for this many (split, gene) pairs at a time
_BLOCK = 2**22


def _exact_p(ranks: np.ndarray, n_x: int, u: np.ndarray) -> np.ndarray:
    """Two-sided permutation p-value of ``u`` per column: twice the smaller tail
    over every split of the rows into ``n_x`` and the rest."""
    n, n_genes = ranks.shape
    members = np.array(list(itertools.combinations(range(n), n_x)))
    splits = np.zeros((len(members), n))
    splits[np.arange(len(members))[:, None], members] = 1
    p = np.empty(n_genes)
    step = max(1, _BLOCK // len(members))
    for start in range(0, n_genes, step):
        block = slice(start, start + step)
        every_u = splits @ ranks[:, block] - n_x * (n_x + 1) / 2
        # midranks are multiples of 1/2, and so are their sums: exact in floats
        below = (every_u <= u[block]).mean(axis=0)
        above = (every_u >= u[block]).mean(axis=0)
        p[block] = np.minimum(1.0, 2 * np.minimum(below, above))
    return p


def mann_whitney(x, y) -> Tuple[np.ndarray, np.ndarray]:
    """Per column (gene): the tie-corrected z of ``x`` against ``y``, and the
    two-sided p-value. The p-value is exact, over every split of the samples
    (ties handled), while there are at most MANN_WHITNEY_EXACT_MAX_SPLITS of them;
    above, the normal approximation of that z. A normal approximation without
    tie correction gave p = 0.0495 to a complete separation at 3 v 3, where an
    exact test cannot go below 0.10. A column with a missing value has neither."""
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
        p = _exact_p(ranks, n_x, u)
    else:
        p = 2 * stats.norm.sf(np.abs(z))
    p[np.isnan(u)] = np.nan
    return z, p
