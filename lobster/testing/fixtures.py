"""
Shared pytest fixtures for Lobster testing.

This module provides domain-agnostic factory functions for creating test
data and workspaces. These can be used directly in tests or wrapped as
pytest fixtures.

Functions:
    - create_test_workspace: Create a standard Lobster workspace structure
    - synthetic_single_cell_data: Generate scRNA-seq test data
    - synthetic_bulk_rnaseq_data: Generate bulk RNA-seq test data
    - synthetic_proteomics_data: Generate proteomics test data

Example:
    >>> from lobster.testing.fixtures import create_test_workspace, synthetic_single_cell_data
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     workspace = create_test_workspace(Path(tmp))
    ...     assert (workspace / 'data').exists()
    >>>
    >>> adata = synthetic_single_cell_data(n_cells=100, n_genes=200)
    >>> assert adata.shape == (100, 200)
"""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import anndata as ad
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "lobster.testing.fixtures requires anndata and pandas. "
        "Install with: pip install anndata pandas"
    ) from e


def create_test_workspace(base_path: Path) -> Path:
    """Create a standard Lobster workspace structure.

    Creates the following directory structure:
        base_path/
        ├── data/           # For H5AD files
        ├── exports/        # For exported notebooks and reports
        ├── cache/          # For cached data
        └── plots/          # For generated visualizations

    Args:
        base_path: Base directory for the workspace.

    Returns:
        Path to the workspace root (same as base_path).

    Raises:
        TypeError: If base_path is not a Path instance.

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     ws = create_test_workspace(Path(tmp))
        ...     assert (ws / 'data').exists()
    """
    if not isinstance(base_path, Path):
        raise TypeError(f"base_path must be Path, got {type(base_path).__name__}")

    # Create standard workspace directories
    directories = ["data", "exports", "cache", "plots"]
    for dirname in directories:
        (base_path / dirname).mkdir(parents=True, exist_ok=True)

    return base_path


def synthetic_single_cell_data(
    n_cells: int = 1000,
    n_genes: int = 2000,
    seed: int = 42,
    *,
    sparsity: float = 0.7,
    cell_types: Optional[list] = None,
    batches: Optional[list] = None,
) -> "ad.AnnData":
    """Generate realistic synthetic single-cell RNA-seq data.

    Creates an AnnData object with realistic scRNA-seq characteristics:
    - Negative binomial count distribution (simulates biological variation)
    - Configurable sparsity (default 70%, typical for scRNA-seq)
    - Cell type annotations and batch metadata
    - QC metrics (total_counts, n_genes_by_counts, pct_counts_mt)

    Args:
        n_cells: Number of cells (observations).
        n_genes: Number of genes (features).
        seed: Random seed for reproducibility.
        sparsity: Fraction of zero entries (default 0.7).
        cell_types: List of cell type labels (default: T_cell, B_cell, etc).
        batches: List of batch labels (default: Batch1, Batch2, Batch3).

    Returns:
        ad.AnnData: Synthetic scRNA-seq dataset with:
            - X: Sparse count matrix
            - obs: Cell metadata (cell_type, batch, QC metrics)
            - var: Gene metadata (gene_ids, chromosome, feature_types)

    Example:
        >>> adata = synthetic_single_cell_data(n_cells=100, n_genes=200)
        >>> assert adata.shape == (100, 200)
        >>> assert 'cell_type' in adata.obs.columns
        >>> assert adata.X.min() >= 0  # Non-negative counts
    """
    np.random.seed(seed)

    # Default cell types and batches
    if cell_types is None:
        cell_types = ["T_cell", "B_cell", "NK_cell", "Monocyte", "Dendritic_cell"]
    if batches is None:
        batches = ["Batch1", "Batch2", "Batch3"]

    # Generate count matrix with negative binomial distribution
    X = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(
        np.float32
    )

    # Add zeros to make it sparse
    zero_mask = np.random.random((n_cells, n_genes)) < sparsity
    X[zero_mask] = 0

    # Create names
    var_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    obs_names = [f"Cell_{i:08d}" for i in range(n_cells)]

    # Create AnnData object
    adata = ad.AnnData(
        X=X,
        var=pd.DataFrame(index=var_names),
        obs=pd.DataFrame(index=obs_names),
    )

    # Add gene metadata
    adata.var["gene_ids"] = [f"ENSG{i:011d}" for i in range(n_genes)]
    adata.var["feature_types"] = ["Gene Expression"] * n_genes
    adata.var["chromosome"] = np.random.choice(
        [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"],
        size=n_genes,
    )

    # Add cell metadata
    adata.obs["total_counts"] = np.array(X.sum(axis=1))
    adata.obs["n_genes_by_counts"] = np.array((X > 0).sum(axis=1))
    adata.obs["pct_counts_mt"] = np.random.uniform(0, 30, n_cells)
    adata.obs["pct_counts_ribo"] = np.random.uniform(0, 50, n_cells)

    # Add cell type and batch annotations
    adata.obs["cell_type"] = np.random.choice(cell_types, size=n_cells)
    adata.obs["batch"] = np.random.choice(batches, size=n_cells)

    return adata


def synthetic_bulk_rnaseq_data(
    n_samples: int = 24,
    n_genes: int = 2000,
    seed: int = 42,
    *,
    conditions: Optional[list] = None,
    batches: Optional[list] = None,
) -> "ad.AnnData":
    """Generate realistic synthetic bulk RNA-seq data.

    Creates an AnnData object with typical bulk RNA-seq experiment structure:
    - Higher counts than single-cell (negative binomial n=20)
    - Balanced experimental design (half treatment, half control by default)
    - Batch effects and biological covariates

    Args:
        n_samples: Number of samples (observations).
        n_genes: Number of genes (features).
        seed: Random seed for reproducibility.
        conditions: List of condition labels (default: Treatment, Control).
        batches: List of batch labels (default: Batch1, Batch2).

    Returns:
        ad.AnnData: Synthetic bulk RNA-seq dataset with:
            - X: Count matrix
            - obs: Sample metadata (condition, batch, sex, age)
            - var: Gene metadata (gene_ids, gene_name, biotype)

    Example:
        >>> adata = synthetic_bulk_rnaseq_data(n_samples=24, n_genes=2000)
        >>> assert adata.shape == (24, 2000)
        >>> assert 'condition' in adata.obs.columns
    """
    np.random.seed(seed)

    # Default conditions and batches
    if conditions is None:
        conditions = ["Treatment", "Control"]
    if batches is None:
        batches = ["Batch1", "Batch2"]

    # Generate count matrix with higher counts than single-cell
    X = np.random.negative_binomial(n=20, p=0.1, size=(n_samples, n_genes)).astype(
        np.float32
    )

    # Create names
    obs_names = [f"Sample_{i:02d}" for i in range(n_samples)]
    var_names = [f"Gene_{i:04d}" for i in range(n_genes)]

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
    )

    # Add gene metadata
    adata.var["gene_ids"] = [f"ENSG{i:011d}" for i in range(n_genes)]
    adata.var["gene_name"] = [f"GENE{i}" for i in range(n_genes)]
    adata.var["biotype"] = np.random.choice(
        ["protein_coding", "lncRNA", "miRNA", "pseudogene"],
        size=n_genes,
        p=[0.7, 0.15, 0.05, 0.1],
    )

    # Add sample metadata - balanced design
    half = n_samples // 2
    adata.obs["condition"] = [conditions[0]] * half + [conditions[1]] * (
        n_samples - half
    )

    # Add batch structure
    batch_size = n_samples // len(batches)
    batch_assignments = []
    for batch in batches:
        batch_assignments.extend([batch] * batch_size)
    # Handle remainder
    while len(batch_assignments) < n_samples:
        batch_assignments.append(batches[-1])
    adata.obs["batch"] = batch_assignments[:n_samples]

    # Add biological covariates
    adata.obs["sex"] = np.random.choice(["M", "F"], size=n_samples)
    adata.obs["age"] = np.random.randint(20, 80, size=n_samples)

    return adata


def synthetic_proteomics_data(
    n_samples: int = 48,
    n_proteins: int = 500,
    seed: int = 42,
    *,
    missing_rate: float = 0.2,
    conditions: Optional[list] = None,
    tissues: Optional[list] = None,
) -> "ad.AnnData":
    """Generate realistic synthetic proteomics data.

    Creates an AnnData object with typical proteomics characteristics:
    - Log-normal intensity distribution (mass spec output)
    - Configurable missing values (default 20%, common in proteomics)
    - Multi-condition experimental design

    Args:
        n_samples: Number of samples (observations).
        n_proteins: Number of proteins (features).
        seed: Random seed for reproducibility.
        missing_rate: Fraction of missing values (default 0.2).
        conditions: List of condition labels (default: Disease, Healthy, Control).
        tissues: List of tissue types (default: Brain, Liver, Kidney).

    Returns:
        ad.AnnData: Synthetic proteomics dataset with:
            - X: Intensity matrix with NaN values
            - obs: Sample metadata (condition, tissue, batch)
            - var: Protein metadata (protein_ids, protein_names, molecular_weight)

    Example:
        >>> adata = synthetic_proteomics_data(n_samples=48, n_proteins=500)
        >>> assert adata.shape == (48, 500)
        >>> assert np.isnan(adata.X).sum() > 0  # Has missing values
    """
    np.random.seed(seed)

    # Default conditions and tissues
    if conditions is None:
        conditions = ["Disease", "Healthy", "Control"]
    if tissues is None:
        tissues = ["Brain", "Liver", "Kidney"]

    # Generate intensity matrix with log-normal distribution
    X = np.random.lognormal(mean=10, sigma=2, size=(n_samples, n_proteins)).astype(
        np.float32
    )

    # Add missing values (common in proteomics)
    missing_mask = np.random.random((n_samples, n_proteins)) < missing_rate
    X[missing_mask] = np.nan

    # Create names
    obs_names = [f"Sample_{i:03d}" for i in range(n_samples)]
    var_names = [f"Protein_{i:03d}" for i in range(n_proteins)]

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
    )

    # Add protein metadata
    adata.var["protein_ids"] = [f"P{i:05d}" for i in range(n_proteins)]
    adata.var["protein_names"] = [f"PROT{i}" for i in range(n_proteins)]
    adata.var["molecular_weight"] = np.random.uniform(10, 200, n_proteins)

    # Add sample metadata - balanced design across conditions
    samples_per_condition = n_samples // len(conditions)
    condition_assignments = []
    for cond in conditions:
        condition_assignments.extend([cond] * samples_per_condition)
    # Handle remainder
    while len(condition_assignments) < n_samples:
        condition_assignments.append(conditions[-1])
    adata.obs["condition"] = condition_assignments[:n_samples]

    # Add tissue and batch info
    adata.obs["tissue"] = np.random.choice(tissues, size=n_samples)
    adata.obs["batch"] = np.random.choice(
        ["Batch1", "Batch2", "Batch3", "Batch4"],
        size=n_samples,
    )

    return adata
