"""
Deviance-based feature selection utilities for single-cell RNA-seq.

Implementation based on Townes et al. (2019):
"Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model"
"""

from typing import Union

import numpy as np
import scipy.sparse as spr

#: Target number of stored values per row block. Peak memory is driven by NONZEROS per
#: block, not rows, so a fixed row count is the wrong unit: the same number of rows can
#: have very different densities. Deriving the row count from the input's own density
#: keeps the bound stable across inputs.
#:
#: The target limits temporary memory while retaining efficient cache locality.
TARGET_BLOCK_NNZ = 20_000_000


def calculate_deviance(
    count_matrix: Union[np.ndarray, spr.spmatrix], chunk: Union[int, None] = None
) -> np.ndarray:
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

    Memory: O(nnz), never densifies. A sparse input is processed in row blocks sized from
    its own density, so peak memory depends on the configured block size rather than the
    full input size. The previous implementation densified sparse matrices, which could
    cause very large peak memory use.

    Args:
        count_matrix: Cell × gene count matrix (raw counts, sparse or dense)
                     Shape: (n_cells, n_genes)
        chunk: Rows (cells) per block. ``None`` (default) derives it from the matrix's
               density so each block holds about ``TARGET_BLOCK_NNZ`` stored values.
               Pass an explicit value only to force a specific bound — it changes peak
               memory and speed, never the result.

    Returns:
        np.ndarray: Deviance score for each gene (higher = more variable)
                   Shape: (n_genes,)

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
    # Work in CSR so row-block slicing is cheap. A dense input is converted TO sparse
    # rather than the reverse: the deviance only ever reads observed counts, so storing
    # the zeros buys nothing and costs everything.
    X = count_matrix if spr.issparse(count_matrix) else spr.csr_matrix(count_matrix)
    X = X.tocsr()

    # A non-canonical container may store SEVERAL entries for the same (cell, gene).
    # They must be summed before scoring, because x*log(x) is nonlinear: f(2) + f(3) is
    # not f(5). `tocsr()` returns an existing CSR unchanged, so duplicates survive it.
    # Copy first -- `sum_duplicates()` mutates in place and the caller's matrix must not
    # change. Nothing in this repo currently feeds duplicates here (every call site passes
    # `adata.X`), so this is a guard, not a fix for an observed failure.
    if not X.has_canonical_format:
        X = X.copy()
        X.sum_duplicates()

    n_cells, n_genes = X.shape

    # Size the row block from the actual density so the bound holds across matrices.
    # At least one row, so a single very wide row still makes progress rather than
    # looping forever on an empty slice.
    #
    # Computed before the marginals because they are accumulated block-wise too.
    if chunk is None:
        nnz_per_row = max(X.nnz / n_cells, 1.0) if n_cells else 1.0
        chunk = max(1, int(TARGET_BLOCK_NNZ / nnz_per_row))

    # Marginals, accumulated in float64 over the same row blocks.
    #
    # NOT `X.sum(axis=)`: scipy reduces in the STORED dtype and casts afterwards, so a
    # float32 matrix -- the AnnData/scanpy default -- loses precision that a later
    # `.astype(np.float64)` cannot recover, and an int64 matrix can wrap to a negative
    # total before anything sees it. Passing `dtype=` to `sum()` does not help either; it
    # casts after the damage. Accumulate in float64 before reducing to avoid
    # stored-dtype rounding or overflow.
    #
    # Blocked rather than one pass over the whole COO: a single pass would materialise
    # row/col/value arrays for every stored entry at once, which is precisely the
    # unbounded working set TARGET_BLOCK_NNZ exists to prevent.
    cell_totals = np.zeros(n_cells, dtype=np.float64)  # n_i
    gene_totals = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, n_cells, chunk):
        block = X[start : start + chunk].tocoo()
        if block.nnz == 0:
            continue
        values = block.data.astype(np.float64, copy=False)
        cell_totals[start : start + block.shape[0]] += np.bincount(
            block.row, weights=values, minlength=block.shape[0]
        )
        gene_totals += np.bincount(block.col, weights=values, minlength=n_genes)
    total_counts = gene_totals.sum()

    # Multinomial null probabilities: p_g = (sum of gene g) / (total counts).
    # Guard the division so an all-zero matrix returns zeros instead of NaN.
    if total_counts <= 0:
        return np.zeros(n_genes, dtype=np.float64)
    p_null = np.maximum(gene_totals / total_counts, 1e-10)

    # Accumulate per gene over row blocks.
    #
    # `expected` used to be built as `cell_totals @ p_null.reshape(1, -1)` -- a full
    # (cells x genes) outer product, so it was never smaller than X however X was stored.
    # Here it is evaluated only at the stored coordinates, which is all the deviance needs.
    deviance_scores = np.zeros(n_genes, dtype=np.float64)

    for start in range(0, n_cells, chunk):
        block = X[start : start + chunk].tocoo()
        if block.nnz == 0:
            continue

        x = block.data.astype(np.float64, copy=False)

        # Only strictly positive counts contribute. This is the mask the original
        # intended: it read `mask = X > 0`, but `X = np.maximum(X, 1e-10)` two lines
        # earlier had already made every element positive, so it selected 100% of a
        # densified matrix. Do NOT reintroduce a floor before this point.
        #
        # Filtering here is also required for correctness, not just speed. A sparse
        # matrix may hold EXPLICIT zeros (routine after filtering or arithmetic), and
        # `0 * log(0/E)` evaluates to `0 * -inf` = NaN rather than 0. Negative values --
        # which reach this function when log-transformed or scaled data is passed by
        # mistake -- would give `log(negative)` = NaN and inf. The old floor masked both
        # by clamping to 1e-10; dropping them is exact instead, since an unobserved count
        # contributes exactly zero to the deviance (x*log(x) -> 0 as x -> 0).
        positive = x > 0
        if not positive.all():
            x = x[positive]
            rows = block.row[positive]
            cols = block.col[positive]
        else:
            rows = block.row
            cols = block.col
        if x.size == 0:
            continue

        # Expected counts under null: E[x_ig] = n_i * p_g, at observed coordinates only.
        expected = np.maximum(cell_totals[start + rows] * p_null[cols], 1e-10)

        # Binomial deviance: 2 * x * log(x / E[x]), summed per gene.
        deviance_scores += np.bincount(
            cols, weights=2.0 * x * np.log(x / expected), minlength=n_genes
        )

    return deviance_scores
