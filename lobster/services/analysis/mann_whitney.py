"""Mann-Whitney (Wilcoxon rank-sum) test over many columns at once.

The p-value is exact at small groups and handles ties; ``scipy.stats.ranksums``
uses a normal approximation without tie correction, and ``mannwhitneyu`` with
its default method uses a normal approximation whenever a value ties, even at
3 v 3.
"""

import functools
import itertools
import math
from typing import Tuple

import numpy as np

# An exact test enumerates every split of the samples into the two groups:
# 12,870 for 8 v 8, 48,620 for 9 v 9
MANN_WHITNEY_EXACT_MAX_SPLITS = 20_000
# The exact test also needs its split matrix (splits x samples) to hold at most
# this many values: 1 v 4,999 has 5,000 splits but a 5,000 x 5,000 matrix
_EXACT_MAX_VALUES = 2**22
# A split matrix is kept for the next call while it holds at most this many
# values: the 4 kept then hold at most 32 MiB. A larger one has a group of 1 or
# 2 samples and is built again from the kept list of splits: 2 v 198 then takes
# 1.3 ms a protein instead of 0.9
_KEPT_MAX_VALUES = 2**20

# Below this share of the rank variance left by ties, the normal approximation
# of the tie-corrected z can give a p-value far below the exact one: 10 v 10
# with 16 samples tied gave 0.031 for an exact 0.087, 3 v 51 with the 51 tied
# 3e-13 for 8e-5. A group of at most four is flagged below the second: at
# 3 v 100 with 45 tied (0.92 left) the normal gave 0.047 for an exact 0.057
MANN_WHITNEY_TIE_FACTOR_MIN = 0.9
MANN_WHITNEY_SMALL_GROUP_TIE_FACTOR_MIN = 0.99

# U of every split is computed for this many (split, gene) pairs at a time
_BLOCK = 2**22


@functools.lru_cache(maxsize=4)
def _members(n: int, k: int) -> np.ndarray:
    """The samples of the smaller group, one row per split of ``n`` into ``k``
    and the rest: at most MANN_WHITNEY_EXACT_MAX_SPLITS x 8 values, kept for the
    next call."""
    members = np.array(list(itertools.combinations(range(n), k)))
    members.flags.writeable = False  # shared by every later call
    return members


def _split_matrix(n: int, n_x: int) -> np.ndarray:
    """One row per split of ``n`` samples into ``n_x`` and the rest: 1 where a
    sample is in the first group."""
    k = min(n_x, n - n_x)
    members = _members(n, k)
    # the rows list the smaller group, so a split of the larger one is its complement
    splits = np.full((len(members), n), float(k < n_x))
    splits[np.arange(len(members))[:, None], members] = float(k == n_x)
    return splits


@functools.lru_cache(maxsize=4)
def _splits(n: int, n_x: int) -> np.ndarray:
    """The split matrix, kept for the next call, which in a per-protein loop has
    the same group sizes."""
    splits = _split_matrix(n, n_x)
    splits.flags.writeable = False  # shared by every later call
    return splits


def _exact_p(ranks: np.ndarray, n_x: int, u: np.ndarray) -> np.ndarray:
    """Two-sided permutation p-value of ``u`` per column: twice the smaller tail
    over every split of the rows into ``n_x`` and the rest."""
    n, n_genes = ranks.shape
    if math.comb(n, n_x) * n <= _KEPT_MAX_VALUES:
        splits = _splits(n, n_x)
    else:
        splits = _split_matrix(n, n_x)
    p = np.empty(n_genes)
    step = max(1, _BLOCK // len(splits))
    for start in range(0, n_genes, step):
        block = slice(start, start + step)
        every_u = splits @ ranks[:, block] - n_x * (n_x + 1) / 2
        # midranks are multiples of 1/2, and so are their sums: exact in floats
        below = (every_u <= u[block]).mean(axis=0)
        above = (every_u >= u[block]).mean(axis=0)
        p[block] = np.minimum(1.0, 2 * np.minimum(below, above))
    return p


def _pairs_at_most(values: np.ndarray, limit: int) -> int:
    """How many pairs of the sorted ``values`` sum to at most ``limit``."""
    partners = np.searchsorted(values, limit - values, side="right").sum()
    return int(partners - np.count_nonzero(2 * values <= limit)) // 2


def _small_group_p(ranks: np.ndarray, small: slice) -> np.ndarray:
    """Two-sided exact p-value per column of the rank sum of the ``small`` rows,
    one or two, over every draw of that many rows: counted from the sorted
    ranks, with no split matrix. Ties are midranks, so twice a rank is an
    integer and every sum is exact."""
    n = len(ranks)
    k = small.stop - small.start
    total = n if k == 1 else n * (n - 1) // 2
    p = np.full(ranks.shape[1], np.nan)
    for j in np.flatnonzero(~np.isnan(ranks).any(axis=0)):
        doubled = np.rint(2 * ranks[:, j]).astype(np.int64)
        values, observed = np.sort(doubled), int(doubled[small].sum())
        if k == 1:
            below = np.count_nonzero(values <= observed)
            above = np.count_nonzero(values >= observed)
        else:
            below = _pairs_at_most(values, observed)
            above = _pairs_at_most(-values[::-1], -observed)
        p[j] = min(1.0, 2 * min(below, above) / total)
    return p


def _exact(n_x: int, n_y: int) -> bool:
    """mann_whitney's p-value is exact: by the split matrix, or counted for a
    group of one or two."""
    n = n_x + n_y
    n_splits = math.comb(n, n_x)
    return (
        n_splits <= MANN_WHITNEY_EXACT_MAX_SPLITS and n_splits * n <= _EXACT_MAX_VALUES
    ) or min(n_x, n_y) <= 2


def approximate_with_ties(x, y) -> np.ndarray:
    """Per column: mann_whitney(x, y) gives the normal approximation, and ties
    leave less than MANN_WHITNEY_TIE_FACTOR_MIN of the rank variance
    (MANN_WHITNEY_SMALL_GROUP_TIE_FACTOR_MIN when the smaller group has at most
    four samples)."""
    from scipy import stats

    n_x, n_y = len(x), len(y)
    n = n_x + n_y
    if _exact(n_x, n_y):
        return np.zeros(np.shape(x)[1], dtype=bool)
    ranks = stats.rankdata(np.vstack([x, y]), axis=0)
    spread = ((ranks - (n + 1) / 2) ** 2).sum(axis=0)
    # without ties the spread is n (n^2 - 1) / 12; a missing value is not flagged
    # one value in every sample has no p-value at all
    least = (
        MANN_WHITNEY_SMALL_GROUP_TIE_FACTOR_MIN
        if min(n_x, n_y) <= 4
        else MANN_WHITNEY_TIE_FACTOR_MIN
    )
    return (spread > 0) & (spread * 12 / (n * (n * n - 1)) < least)


def mann_whitney(x, y) -> Tuple[np.ndarray, np.ndarray]:
    """Per column (gene): the tie-corrected z of ``x`` against ``y``, and the
    two-sided p-value. The p-value is exact, over every split of the samples
    (ties handled), while there are at most MANN_WHITNEY_EXACT_MAX_SPLITS of them
    and their matrix fits _EXACT_MAX_VALUES, or the smaller group has one or two
    samples; otherwise, the normal approximation of that z. A normal approximation without
    tie correction gave p = 0.0495 to a complete separation at 3 v 3, where an
    exact test cannot go below 0.10. A column with a missing value has neither,
    and nor has a column of one value tied in every sample (z is 0 / 0)."""
    from scipy import stats

    n_x, n_y = len(x), len(y)
    n = n_x + n_y
    ranks = stats.rankdata(np.vstack([x, y]), axis=0)
    u = ranks[:n_x].sum(axis=0) - n_x * (n_x + 1) / 2
    # the permutation variance of U, which accounts for ties
    spread = ((ranks - (n + 1) / 2) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (u - n_x * n_y / 2) / np.sqrt(n_x * n_y * spread / (n * (n - 1)))
    n_splits = math.comb(n, n_x)
    if n_splits <= MANN_WHITNEY_EXACT_MAX_SPLITS and n_splits * n <= _EXACT_MAX_VALUES:
        p = _exact_p(ranks, n_x, u)
    elif min(n_x, n_y) <= 2:
        p = _small_group_p(ranks, slice(0, n_x) if n_x <= n_y else slice(n_x, n))
    else:
        p = 2 * stats.norm.sf(np.abs(z))
    p[np.isnan(z)] = np.nan
    return z, p
