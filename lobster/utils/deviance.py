"""
Deviance-based feature selection utilities for single-cell RNA-seq.

Implementation based on Townes et al. (2019):
"Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model"
"""

from typing import Union

import numpy as np
import scipy.sparse as spr

# Appended to every input-domain rejection. The point of naming the likely upstream
# cause is that the caller is almost never wrong about deviance -- they are wrong
# about which matrix they handed it.
_DOMAIN_HINT = (
    "calculate_deviance expects RAW COUNTS: non-negative and finite. Values "
    "outside that domain normally mean the matrix was normalized, "
    "log-transformed, scaled or batch-corrected upstream. Deviance computed on "
    "such input returns a finite number that looks valid and means nothing, so "
    "this is raised rather than warned. Pass the raw counts (e.g. "
    "``adata.raw.X``, or ``adata.layers['counts']``)."
)


def _reject_non_count_values(values: np.ndarray) -> None:
    """Raise ``ValueError`` if any stored value is negative or non-finite.

    ``O(len(values))`` time and one boolean temporary of the same length, so it is
    called with the *stored* values of a sparse matrix (``nnz``), never with an
    expanded dense matrix. The dense path checks reductions instead -- see
    ``calculate_deviance``.

    Structural and explicitly-stored zeros are legal: zero is a non-negative,
    finite count and contributes nothing to the deviance.
    """
    if values.size == 0:
        return
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(
            f"calculate_deviance received {int((~finite).sum())} non-finite "
            f"value(s) (NaN or +/-inf) out of {values.size} stored. " + _DOMAIN_HINT
        )
    smallest = float(values.min())
    if smallest < 0.0:
        raise ValueError(
            f"calculate_deviance received {int((values < 0).sum())} negative "
            f"value(s) out of {values.size} stored; minimum is {smallest!r}. "
            + _DOMAIN_HINT
        )


# Target element count per row block on the dense path. Several temporaries exist
# simultaneously per block (boolean mask, two index arrays, the gathered values,
# ``expected``, the ratio and its log), so the real peak is a multiple of this --
# measured ~256 MiB for a fully dense 4195x1000 block, not one 32 MiB array. A row
# wider than this cannot be split, so a matrix with very many genes exceeds it.
_DENSE_BLOCK_ELEMENTS = 4_194_304


def _null_probabilities(gene_totals: np.ndarray, total_counts: float) -> np.ndarray:
    """Multinomial null probability per gene. No floor, deliberately.

    A floor here would be a silent correctness bug, not a safety net. Because the
    input is validated non-negative, ``gene_totals[j] == 0`` implies every entry
    of gene ``j`` is zero, so gene ``j`` has no observed positive count and
    ``p_null[j]`` is never indexed by ``_accumulate_gene_deviance``. The floor
    could therefore only ever fire on genes that contribute nothing -- while
    corrupting the genes it did apply to, by replacing a legitimate share below
    ``1e-10`` (reachable: ~10M cells x 3000 counts gives a singleton gene a true
    share of 3.3e-11) and understating the deviance of exactly the ultra-rare
    genes that feature selection exists to surface.
    """
    if total_counts <= 0:
        # Degenerate input (empty or all-zero matrix): no gene carries signal, and
        # no entry is observed, so these values are never read.
        return np.zeros(gene_totals.shape, dtype=np.float64)
    return np.asarray(gene_totals / total_counts, dtype=np.float64)


def _accumulate_gene_deviance(
    values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    cell_totals: np.ndarray,
    p_null: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    """Sum ``2 * x * log(x / E[x])`` per gene over the given observed entries.

    ``values``/``rows``/``cols`` describe only strictly-positive observed counts,
    so ``log`` never sees zero or a negative.

    ``expected`` needs no floor: an observed positive count at ``(i, j)`` forces
    ``cell_totals[i] > 0`` and ``p_null[j] > 0`` on validated non-negative input,
    so the product is strictly positive for every entry actually indexed here.
    """
    expected = cell_totals[rows] * p_null[cols]
    terms = 2.0 * values * np.log(values / expected)
    # np.bincount ignores the weight dtype when the input is empty and returns
    # int64, so the float64 return contract needs asserting rather than assuming.
    return np.asarray(
        np.bincount(cols, weights=terms, minlength=n_genes), dtype=np.float64
    )


def calculate_deviance(count_matrix: Union[np.ndarray, spr.spmatrix]) -> np.ndarray:
    """
    Calculate binomial deviance from multinomial null model for feature selection.

    This method works on raw counts without normalization bias, providing a mathematically
    principled alternative to highly variable genes (HVG) methods.

    The deviance measures how much each gene deviates from the expected expression under
    a simple multinomial null model where all cells have the same gene expression proportions.

    Mathematical formula:
        D(gene) = 2 × Σ_cells [x_ij × log(x_ij / μ_ij)]

    Where:
        - x_ij = observed count for gene j in cell i
        - μ_ij = expected count under multinomial null = n_i × p_j
        - n_i = total UMI count for cell i
        - p_j = gene j's proportion of total counts across all cells

    Only strictly-positive observed counts contribute: ``x·log(x/μ) → 0`` as
    ``x → 0``, so zeros add exactly nothing to the sum. That makes the O(nnz)
    form below the *reference* implementation, not an approximation of a dense
    one.

    .. warning::
        Do not floor the observed matrix (e.g. ``X = np.maximum(X, 1e-10)``) to
        "avoid log(0)". Restrict to the nonzero entries instead. A floor makes
        every element positive, so a subsequent ``X > 0`` mask selects the whole
        matrix: the sparse path silently densifies by a factor of exactly
        ``1 / density`` (a 0.55%-dense droplet matrix becomes a ~180x working
        set, TiB-scale on real GEO inputs) and every zero contributes a small
        spurious negative term. This was a live defect and the mask-saturation
        regression test exists to keep it from returning.

    Memory: O(nnz + n_cells + n_genes) for sparse input -- the marginals and the
    output scale with the shape, not the stored values. A CSC or COO input is
    converted, which copies; an ndarray and a canonical CSR are not copied, and
    the input is never mutated. Dense input is processed in row blocks, which
    bounds the per-block temporaries but not the marginals, and a single row
    wider than ``_DENSE_BLOCK_ELEMENTS`` cannot be split.

    Input domain -- enforced, not assumed: entries must be **non-negative and
    finite**. Anything else raises ``ValueError``. Counts are non-negative and
    finite by definition of the assay, so a violation does not mean unusual data,
    it means this is not a count matrix -- normalized, log-transformed, scaled or
    batch-corrected input produces a finite deviance that looks valid and means
    nothing. Rejecting is also what lets the denominators carry no epsilon floor
    (see ``_null_probabilities``), so the two are one decision, not two.

    Warn-and-drop was considered and rejected: dropping entries silently changes
    the gene set returned, so two runs on near-identical inputs could produce
    different top-N sets with nothing in the return value to say so. A selection
    step whose output depends on undeclared filtering is not reproducible.

    Args:
        count_matrix: Cell × gene count matrix (raw counts, sparse or dense)
                     Shape: (n_cells, n_genes)

    Returns:
        np.ndarray: Deviance score for each gene (higher = more variable)
                   Shape: (n_genes,)

    Raises:
        ValueError: if any entry is negative or non-finite. The message names the
            offending statistic (count and minimum, or count of non-finite
            marginals) and states the expected domain.

    Example:
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc3k()
        >>> deviance_scores = calculate_deviance(adata.X)
        >>> # Select top 2000 genes
        >>> top_genes_idx = np.argsort(deviance_scores)[::-1][:2000]
        >>> adata.var['highly_deviant'] = False
        >>> adata.var.iloc[top_genes_idx, adata.var.columns.get_loc('highly_deviant')] = True

    Reference:
        Townes, F. W., Hicks, S. C., Aryee, M. J., & Irizarry, R. A. (2019).
        Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model.
        Genome Biology, 20(1), 295. https://doi.org/10.1186/s13059-019-1861-6
    """
    if spr.issparse(count_matrix):
        matrix = count_matrix.tocsr()
        # A non-canonical container may store several entries for the same
        # (cell, gene). They must be summed before scoring, because x*log(x) is
        # nonlinear: f(2) + f(3) != f(5). ``tocsr()`` returns an existing CSR
        # unchanged, so duplicates survive unless handled here. Copy first --
        # ``sum_duplicates()`` mutates in place and the caller's matrix must not
        # change.
        if not matrix.has_canonical_format:
            matrix = matrix.copy()
            matrix.sum_duplicates()
        n_cells, n_genes = matrix.shape

        entries = matrix.tocoo()
        # Totals come from the true matrix; a floored copy would perturb them.
        #
        # Accumulate them here rather than via ``matrix.sum(axis=...)``: scipy
        # reduces in the STORED dtype and casts afterwards, so a float32 matrix
        # loses precision and an int64 matrix can wrap to a negative total before
        # anything sees it. Passing ``dtype=`` to ``sum()`` does NOT help -- it
        # casts after the damage. Summing the entries in float64 is exact for any
        # count magnitude up to 2**53 and reuses the COO we already need.
        values = entries.data.astype(np.float64, copy=False)
        # Validate before the marginals, so a rejected matrix cannot first produce
        # a NaN total that some later check misattributes. Only the stored values
        # need checking: absent entries are zero, which is a legal count.
        _reject_non_count_values(values)
        cell_totals = np.asarray(
            np.bincount(entries.row, weights=values, minlength=n_cells),
            dtype=np.float64,
        )
        gene_totals = np.asarray(
            np.bincount(entries.col, weights=values, minlength=n_genes),
            dtype=np.float64,
        )
        p_null = _null_probabilities(gene_totals, float(gene_totals.sum()))

        # Sparse containers may hold explicitly-stored zeros.
        observed = entries.data > 0
        return _accumulate_gene_deviance(
            values[observed],
            entries.row[observed],
            entries.col[observed],
            cell_totals,
            p_null,
            n_genes,
        )

    X = np.asarray(count_matrix)
    n_cells, n_genes = X.shape
    cell_totals = X.sum(axis=1, dtype=np.float64)
    gene_totals = X.sum(axis=0, dtype=np.float64)

    # Dense validation via reductions, NOT via _reject_non_count_values: that would
    # allocate a boolean the size of the whole matrix, which is the very cost this
    # implementation exists to avoid.
    #
    # Non-finite values are caught in the marginals because every element belongs to
    # exactly one row and one column, so a NaN or inf anywhere makes both of its
    # marginals non-finite. That is O(n_cells + n_genes) instead of O(n_cells *
    # n_genes). It must run BEFORE the min() check, since min() of an array holding
    # NaN returns NaN and ``NaN < 0`` is False -- the negative check alone would
    # silently pass a NaN matrix.
    if not (np.isfinite(cell_totals).all() and np.isfinite(gene_totals).all()):
        raise ValueError(
            "calculate_deviance received non-finite value(s) (NaN or +/-inf): "
            f"{int((~np.isfinite(cell_totals)).sum())} of {n_cells} cell "
            f"total(s) and {int((~np.isfinite(gene_totals)).sum())} of "
            f"{n_genes} gene total(s) are non-finite. " + _DOMAIN_HINT
        )
    if X.size and float(X.min()) < 0.0:
        # X.min() is a reduction, so this adds no full-size temporary.
        raise ValueError(
            f"calculate_deviance received negative value(s); minimum is "
            f"{float(X.min())!r}. " + _DOMAIN_HINT
        )

    p_null = _null_probabilities(gene_totals, float(gene_totals.sum()))

    deviance_scores = np.zeros(n_genes, dtype=np.float64)
    block_rows = max(1, _DENSE_BLOCK_ELEMENTS // max(n_genes, 1))
    for start in range(0, n_cells, block_rows):
        block = X[start : start + block_rows]
        rows, cols = np.nonzero(block > 0)
        if rows.size == 0:
            continue
        deviance_scores += _accumulate_gene_deviance(
            block[rows, cols].astype(np.float64, copy=False),
            rows + start,
            cols,
            cell_totals,
            p_null,
            n_genes,
        )

    return deviance_scores
