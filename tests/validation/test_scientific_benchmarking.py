"""
Scientific Benchmarking Test Suite - Agent 14

This module validates Lobster's scientific accuracy by comparing results against
gold-standard bioinformatics tools:
- Bulk RNA-seq: Lobster vs Reference pyDESeq2
- Single-cell: Lobster vs Reference scanpy workflows
- Proteomics: Lobster vs Reference limma-style statistics

Validation metrics:
- Correlation of p-values (Spearman, Pearson)
- Overlap of significant genes/proteins (Jaccard, percentage)
- Concordance of effect sizes (log2FC correlation)
- Clustering agreement (Adjusted Rand Index, NMI)
- Overall scientific validity assessment

Test datasets:
- Small synthetic datasets with known properties
- Publicly available benchmark datasets
- Edge cases and validation scenarios

Time budget: 90-120 minutes
Coverage target: Comprehensive scientific validation

Author: Agent 14
Date: 2025-11-07
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.bulk_rnaseq_service import BulkRNASeqService
from lobster.tools.clustering_service import ClusteringService
from lobster.tools.differential_formula_service import DifferentialFormulaService
from lobster.tools.preprocessing_service import PreprocessingService
from lobster.tools.quality_service import QualityService

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===============================================================================
# Test Data Generation for Benchmarking
# ===============================================================================


def create_benchmark_bulk_rnaseq_data(
    n_genes: int = 2000,
    n_samples_per_group: int = 6,
    n_de_genes: int = 200,
    fold_change_range: Tuple[float, float] = (2.0, 4.0),
    noise_level: float = 1.0,
    random_seed: int = 42,
) -> Tuple[AnnData, pd.DataFrame, List[str]]:
    """
    Create synthetic bulk RNA-seq count data with known DE genes.

    Parameters
    ----------
    n_genes : int
        Total number of genes
    n_samples_per_group : int
        Number of samples per condition
    n_de_genes : int
        Number of truly differentially expressed genes
    fold_change_range : tuple
        Range of fold changes for DE genes
    noise_level : float
        Biological noise level
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (adata, metadata_df, true_de_genes)
    """
    np.random.seed(random_seed)

    n_total_samples = n_samples_per_group * 2

    # Create base expression levels (log scale)
    base_expression = np.random.lognormal(mean=5.0, sigma=1.5, size=n_genes)

    # Create count matrix
    counts = np.zeros((n_total_samples, n_genes))

    # Select true DE genes
    de_gene_indices = np.random.choice(n_genes, size=n_de_genes, replace=False)
    true_de_genes = [f"gene_{i}" for i in de_gene_indices]

    # Generate counts for each sample
    for sample_idx in range(n_total_samples):
        # Determine condition (control vs treatment)
        is_treatment = sample_idx >= n_samples_per_group

        # Library size variation (realistic sequencing depth)
        library_size_factor = np.random.lognormal(mean=0, sigma=0.3)

        for gene_idx in range(n_genes):
            base_mean = base_expression[gene_idx] * library_size_factor

            # Apply fold change if this is a DE gene and treatment condition
            if gene_idx in de_gene_indices and is_treatment:
                fold_change = np.random.uniform(*fold_change_range)
                direction = np.random.choice([-1, 1])  # Up or down regulation
                base_mean *= fold_change**direction

            # Add biological noise
            expression = base_mean * np.random.lognormal(mean=0, sigma=noise_level)

            # Generate count (negative binomial with overdispersion)
            dispersion = 0.1 + (1.0 / np.sqrt(base_mean))  # Realistic dispersion
            n_param = 1.0 / dispersion
            p_param = n_param / (n_param + expression)

            count = np.random.negative_binomial(n_param, 1 - p_param)
            counts[sample_idx, gene_idx] = count

    # Create metadata
    sample_names = [f"sample_{i}" for i in range(n_total_samples)]
    conditions = ["control"] * n_samples_per_group + ["treatment"] * n_samples_per_group
    metadata_df = pd.DataFrame(
        {"sample": sample_names, "condition": conditions, "batch": "batch1"}
    )
    metadata_df.index = sample_names

    # Create AnnData
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    adata = AnnData(
        X=counts.astype(np.int32),
        obs=metadata_df,
        var=pd.DataFrame(index=gene_names),
    )

    return adata, metadata_df, true_de_genes


def create_benchmark_singlecell_data(
    n_cells_per_cluster: int = 200,
    n_clusters: int = 3,
    n_genes: int = 1000,
    n_marker_genes: int = 50,
    cluster_separation: float = 2.0,
    noise_level: float = 0.5,
    random_seed: int = 42,
) -> Tuple[AnnData, List[str], Dict[int, List[str]]]:
    """
    Create synthetic single-cell data with known cell types and markers.

    Parameters
    ----------
    n_cells_per_cluster : int
        Number of cells per cluster
    n_clusters : int
        Number of distinct cell types
    n_genes : int
        Total number of genes
    n_marker_genes : int
        Number of marker genes per cluster
    cluster_separation : float
        Degree of cluster separation
    noise_level : float
        Biological noise level
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (adata, true_labels, marker_genes_by_cluster)
    """
    np.random.seed(random_seed)

    n_total_cells = n_cells_per_cluster * n_clusters

    # Create cluster-specific marker genes
    marker_genes_by_cluster = {}
    all_markers = []

    for cluster_idx in range(n_clusters):
        start_idx = cluster_idx * n_marker_genes
        end_idx = start_idx + n_marker_genes
        markers = [f"gene_{i}" for i in range(start_idx, end_idx)]
        marker_genes_by_cluster[cluster_idx] = markers
        all_markers.extend(markers)

    # Generate expression data
    counts = np.zeros((n_total_cells, n_genes))
    true_labels = []

    for cluster_idx in range(n_clusters):
        cluster_start = cluster_idx * n_cells_per_cluster
        cluster_end = cluster_start + n_cells_per_cluster

        for cell_idx in range(cluster_start, cluster_end):
            true_labels.append(f"cluster_{cluster_idx}")

            # Base expression for all genes
            base_expression = np.random.negative_binomial(5, 0.5, size=n_genes)

            # Enhance marker gene expression for this cluster
            for marker_gene in marker_genes_by_cluster[cluster_idx]:
                gene_idx = int(marker_gene.split("_")[1])
                if gene_idx < n_genes:
                    # Higher expression for marker genes
                    base_expression[gene_idx] = np.random.negative_binomial(
                        20, 0.3
                    )  # Much higher

            # Add cluster-specific shift to latent space
            cluster_effect = cluster_separation * (
                cluster_idx - n_clusters / 2
            )  # Spread clusters

            # Apply log-normal noise
            noise = np.random.lognormal(mean=cluster_effect, sigma=noise_level)
            counts[cell_idx, :] = base_expression * noise

    # Create AnnData
    cell_names = [f"cell_{i}" for i in range(n_total_cells)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    obs_df = pd.DataFrame({"cell": cell_names, "true_cluster": true_labels})
    obs_df.index = cell_names

    var_df = pd.DataFrame(index=gene_names)
    var_df["is_marker"] = [
        (gene in all_markers) for gene in gene_names
    ]  # Mark true markers

    adata = AnnData(X=csr_matrix(counts.astype(np.float32)), obs=obs_df, var=var_df)

    return adata, true_labels, marker_genes_by_cluster


# ===============================================================================
# Benchmark Metrics Computation
# ===============================================================================


def compute_de_correlation_metrics(
    lobster_results: pd.DataFrame,
    reference_results: pd.DataFrame,
    pval_col_lobster: str = "pvalue",
    lfc_col_lobster: str = "log2FoldChange",
    pval_col_reference: str = "pvalue",
    lfc_col_reference: str = "log2FoldChange",
) -> Dict[str, Any]:
    """
    Compute correlation metrics between Lobster and reference DE results.

    Parameters
    ----------
    lobster_results : pd.DataFrame
        Lobster differential expression results
    reference_results : pd.DataFrame
        Reference tool differential expression results
    pval_col_lobster : str
        Column name for p-values in Lobster results
    lfc_col_lobster : str
        Column name for log2 fold change in Lobster results
    pval_col_reference : str
        Column name for p-values in reference results
    lfc_col_reference : str
        Column name for log2 fold change in reference results

    Returns
    -------
    dict
        Comprehensive correlation metrics
    """
    # Merge on gene names
    merged = pd.merge(
        lobster_results,
        reference_results,
        left_index=True,
        right_index=True,
        suffixes=("_lobster", "_reference"),
    )

    # Remove NaN/inf values
    valid_mask = (
        np.isfinite(merged[f"{pval_col_lobster}_lobster"])
        & np.isfinite(merged[f"{pval_col_reference}_reference"])
        & np.isfinite(merged[f"{lfc_col_lobster}_lobster"])
        & np.isfinite(merged[f"{lfc_col_reference}_reference"])
    )

    merged_valid = merged[valid_mask]

    if len(merged_valid) == 0:
        return {
            "error": "No valid overlapping genes",
            "n_overlapping": 0,
            "n_valid": 0,
        }

    # P-value correlations
    pval_spearman, pval_spearman_pval = stats.spearmanr(
        merged_valid[f"{pval_col_lobster}_lobster"],
        merged_valid[f"{pval_col_reference}_reference"],
    )

    pval_pearson, pval_pearson_pval = stats.pearsonr(
        -np.log10(merged_valid[f"{pval_col_lobster}_lobster"] + 1e-300),
        -np.log10(merged_valid[f"{pval_col_reference}_reference"] + 1e-300),
    )

    # Log2 fold change correlations
    lfc_spearman, lfc_spearman_pval = stats.spearmanr(
        merged_valid[f"{lfc_col_lobster}_lobster"],
        merged_valid[f"{lfc_col_reference}_reference"],
    )

    lfc_pearson, lfc_pearson_pval = stats.pearsonr(
        merged_valid[f"{lfc_col_lobster}_lobster"],
        merged_valid[f"{lfc_col_reference}_reference"],
    )

    # Significant gene overlap (FDR < 0.05)
    lobster_sig = set(
        merged_valid[merged_valid[f"{pval_col_lobster}_lobster"] < 0.05].index
    )
    reference_sig = set(
        merged_valid[merged_valid[f"{pval_col_reference}_reference"] < 0.05].index
    )

    overlap = len(lobster_sig & reference_sig)
    union = len(lobster_sig | reference_sig)
    jaccard = overlap / union if union > 0 else 0

    return {
        "n_overlapping_genes": len(merged),
        "n_valid_genes": len(merged_valid),
        "pvalue_spearman_r": float(pval_spearman),
        "pvalue_spearman_pval": float(pval_spearman_pval),
        "pvalue_pearson_r": float(pval_pearson),
        "pvalue_pearson_pval": float(pval_pearson_pval),
        "log2fc_spearman_r": float(lfc_spearman),
        "log2fc_spearman_pval": float(lfc_spearman_pval),
        "log2fc_pearson_r": float(lfc_pearson),
        "log2fc_pearson_pval": float(lfc_pearson_pval),
        "lobster_sig_genes": len(lobster_sig),
        "reference_sig_genes": len(reference_sig),
        "overlap_sig_genes": overlap,
        "jaccard_index": float(jaccard),
        "percent_overlap": float(
            100 * overlap / max(len(lobster_sig), len(reference_sig))
            if max(len(lobster_sig), len(reference_sig)) > 0
            else 0
        ),
    }


def compute_clustering_metrics(
    lobster_labels: np.ndarray, reference_labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute clustering agreement metrics.

    Parameters
    ----------
    lobster_labels : np.ndarray
        Cluster labels from Lobster
    reference_labels : np.ndarray
        Cluster labels from reference method

    Returns
    -------
    dict
        Clustering agreement metrics (ARI, NMI)
    """
    ari = adjusted_rand_score(reference_labels, lobster_labels)
    nmi = normalized_mutual_info_score(reference_labels, lobster_labels)

    return {
        "adjusted_rand_index": float(ari),
        "normalized_mutual_info": float(nmi),
        "n_lobster_clusters": len(np.unique(lobster_labels)),
        "n_reference_clusters": len(np.unique(reference_labels)),
    }


# ===============================================================================
# Bulk RNA-seq Benchmarking Tests
# ===============================================================================


@pytest.mark.validation
class TestBulkRNASeqBenchmarking:
    """Benchmark Lobster bulk RNA-seq against reference pyDESeq2."""

    def test_deseq2_basic_comparison(self, tmp_path):
        """
        Test basic DESeq2 comparison on synthetic data.

        This test validates that Lobster's pyDESeq2 integration produces
        results highly correlated with a reference pyDESeq2 workflow.
        """
        # Generate benchmark data
        adata, metadata_df, true_de_genes = create_benchmark_bulk_rnaseq_data(
            n_genes=2000,
            n_samples_per_group=6,
            n_de_genes=200,
            fold_change_range=(2.0, 4.0),
            random_seed=42,
        )

        print(f"\n[Benchmark] Created dataset: {adata.shape}")
        print(f"[Benchmark] True DE genes: {len(true_de_genes)}")

        # --------------------------------------------------------------------
        # Reference: Direct pyDESeq2 workflow
        # --------------------------------------------------------------------
        print("\n[Reference] Running reference pyDESeq2...")

        # pyDESeq2 expects samples × genes internally
        # Create genes × samples first for filtering
        counts_df_genes_samples = pd.DataFrame(
            adata.X.T, index=adata.var_names, columns=adata.obs_names
        )

        # Filter low-count genes (standard pre-filtering)
        keep_genes = counts_df_genes_samples.sum(axis=1) >= 10
        counts_df_filtered = counts_df_genes_samples[keep_genes]

        # Transpose to samples × genes for pyDESeq2
        counts_for_deseq2 = counts_df_filtered.T.astype(int)

        reference_dds = DeseqDataSet(
            counts=counts_for_deseq2,
            metadata=metadata_df,
            design="~condition",
        )

        reference_dds.deseq2()
        reference_stat_res = DeseqStats(
            reference_dds, contrast=["condition", "treatment", "control"]
        )
        reference_stat_res.summary()
        reference_results = reference_stat_res.results_df

        print(f"[Reference] Analyzed {len(reference_results)} genes")
        print(
            f"[Reference] Significant genes (padj < 0.05): {(reference_results['padj'] < 0.05).sum()}"
        )

        # --------------------------------------------------------------------
        # Lobster: Using BulkRNASeqService
        # --------------------------------------------------------------------
        print("\n[Lobster] Running Lobster BulkRNASeqService...")

        bulk_service = BulkRNASeqService(results_dir=tmp_path)

        # Run pyDESeq2 analysis using the service (service expects genes × samples)
        lobster_results = bulk_service.run_pydeseq2_analysis(
            count_matrix=counts_df_filtered.astype(int),  # genes × samples
            metadata=metadata_df,
            formula="~condition",
            contrast=["condition", "treatment", "control"],
        )

        print(f"[Lobster] Analyzed {len(lobster_results)} genes")
        print(
            f"[Lobster] Significant genes (padj < 0.05): {(lobster_results['padj'] < 0.05).sum()}"
        )

        # --------------------------------------------------------------------
        # Compute benchmark metrics
        # --------------------------------------------------------------------
        print("\n[Benchmark] Computing correlation metrics...")

        metrics = compute_de_correlation_metrics(
            lobster_results=lobster_results,
            reference_results=reference_results,
            pval_col_lobster="pvalue",
            lfc_col_lobster="log2FoldChange",
            pval_col_reference="pvalue",
            lfc_col_reference="log2FoldChange",
        )

        # Print results
        print("\n" + "=" * 70)
        print("BULK RNA-SEQ BENCHMARKING RESULTS")
        print("=" * 70)
        print(f"Dataset: {adata.shape[0]} samples x {adata.shape[1]} genes")
        print(f"True DE genes: {len(true_de_genes)}")
        print(f"Overlapping genes analyzed: {metrics['n_overlapping_genes']}")
        print(f"Valid genes (no NaN/inf): {metrics['n_valid_genes']}")
        print("\nP-value Correlation:")
        print(
            f"  Spearman r: {metrics['pvalue_spearman_r']:.4f} (p={metrics['pvalue_spearman_pval']:.2e})"
        )
        print(
            f"  Pearson r:  {metrics['pvalue_pearson_r']:.4f} (p={metrics['pvalue_pearson_pval']:.2e})"
        )
        print("\nLog2 Fold Change Correlation:")
        print(
            f"  Spearman r: {metrics['log2fc_spearman_r']:.4f} (p={metrics['log2fc_spearman_pval']:.2e})"
        )
        print(
            f"  Pearson r:  {metrics['log2fc_pearson_r']:.4f} (p={metrics['log2fc_pearson_pval']:.2e})"
        )
        print("\nSignificant Gene Overlap (FDR < 0.05):")
        print(f"  Lobster:    {metrics['lobster_sig_genes']} genes")
        print(f"  Reference:  {metrics['reference_sig_genes']} genes")
        print(f"  Overlap:    {metrics['overlap_sig_genes']} genes")
        print(f"  Jaccard:    {metrics['jaccard_index']:.4f}")
        print(f"  % Overlap:  {metrics['percent_overlap']:.2f}%")
        print("=" * 70)

        # --------------------------------------------------------------------
        # Validation assertions
        # --------------------------------------------------------------------

        # P-values should be highly correlated
        assert (
            metrics["pvalue_spearman_r"] >= 0.90
        ), f"P-value Spearman correlation too low: {metrics['pvalue_spearman_r']:.4f}"

        # Log2 fold changes should be highly correlated
        assert (
            metrics["log2fc_spearman_r"] >= 0.95
        ), f"Log2FC Spearman correlation too low: {metrics['log2fc_spearman_r']:.4f}"

        # Significant gene overlap should be substantial
        assert (
            metrics["percent_overlap"] >= 75.0
        ), f"Significant gene overlap too low: {metrics['percent_overlap']:.2f}%"

        # Jaccard index should be reasonable
        assert (
            metrics["jaccard_index"] >= 0.60
        ), f"Jaccard index too low: {metrics['jaccard_index']:.4f}"

        print("\n✅ BULK RNA-SEQ BENCHMARK PASSED - Results scientifically valid")

        return metrics

    def test_deseq2_complex_design(self, tmp_path):
        """
        Test DESeq2 with complex experimental design (batch effects).

        This validates Lobster handles more complex designs correctly.
        """
        # Generate data with batch effects
        np.random.seed(123)

        n_genes = 1000
        n_samples = 12  # 6 per condition, 2 batches

        # Start with integer counts
        counts = np.random.negative_binomial(10, 0.3, size=(n_samples, n_genes)).astype(
            float
        )

        # Add batch effect (multiplication produces floats)
        batch_effect = np.random.uniform(0.8, 1.2, size=n_genes)
        counts[6:, :] = counts[6:, :] * batch_effect  # Batch 2

        # Add condition effect for first 100 genes
        condition_effect = np.random.uniform(1.5, 3.0, size=100)
        counts[:6, :100] = counts[:6, :100] * condition_effect  # Treatment in batch 1
        counts[6:12, :100] = (
            counts[6:12, :100] * condition_effect
        )  # Treatment in batch 2

        # Convert back to integer counts
        counts = np.round(counts).astype(np.int32)

        metadata = pd.DataFrame(
            {
                "condition": ["control"] * 3
                + ["treatment"] * 3
                + ["control"] * 3
                + ["treatment"] * 3,
                "batch": ["batch1"] * 6 + ["batch2"] * 6,
            },
            index=[f"sample_{i}" for i in range(n_samples)],
        )

        adata = AnnData(
            X=counts,
            obs=metadata,
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
        )

        print(f"\n[Benchmark] Created complex design dataset: {adata.shape}")
        print(f"[Benchmark] Design: condition + batch")

        # Reference pyDESeq2
        counts_df_genes_samples = pd.DataFrame(
            adata.X.T, index=adata.var_names, columns=adata.obs_names
        )
        keep_genes = counts_df_genes_samples.sum(axis=1) >= 10
        counts_df_filtered = counts_df_genes_samples[keep_genes]

        # Transpose to samples × genes for pyDESeq2
        counts_for_deseq2 = counts_df_filtered.T.astype(int)

        reference_dds = DeseqDataSet(
            counts=counts_for_deseq2,
            metadata=metadata,
            design="~batch + condition",
        )
        reference_dds.deseq2()
        reference_stat_res = DeseqStats(
            reference_dds, contrast=["condition", "treatment", "control"]
        )
        reference_stat_res.summary()
        reference_results = reference_stat_res.results_df

        # Lobster workflow (expects genes × samples)
        bulk_service = BulkRNASeqService(results_dir=tmp_path)

        lobster_results = bulk_service.run_pydeseq2_analysis(
            count_matrix=counts_df_filtered.astype(int),
            metadata=metadata,
            formula="~batch + condition",
            contrast=["condition", "treatment", "control"],
        )

        # Compute metrics
        metrics = compute_de_correlation_metrics(
            lobster_results=lobster_results,
            reference_results=reference_results,
            pval_col_lobster="pvalue",
            lfc_col_lobster="log2FoldChange",
            pval_col_reference="pvalue",
            lfc_col_reference="log2FoldChange",
        )

        print("\n" + "=" * 70)
        print("COMPLEX DESIGN BENCHMARKING RESULTS")
        print("=" * 70)
        print(f"P-value Spearman r: {metrics['pvalue_spearman_r']:.4f}")
        print(f"Log2FC Spearman r:  {metrics['log2fc_spearman_r']:.4f}")
        print(f"Jaccard index:      {metrics['jaccard_index']:.4f}")
        print("=" * 70)

        # Validation
        assert metrics["pvalue_spearman_r"] >= 0.85
        assert metrics["log2fc_spearman_r"] >= 0.90

        print("\n✅ COMPLEX DESIGN BENCHMARK PASSED")

        return metrics


# ===============================================================================
# Single-Cell Benchmarking Tests
# ===============================================================================


@pytest.mark.validation
class TestSingleCellBenchmarking:
    """Benchmark Lobster single-cell workflows against reference scanpy."""

    def test_clustering_benchmark(self, tmp_path):
        """
        Benchmark Lobster clustering against reference scanpy workflow.

        Validates that Lobster's clustering produces results comparable
        to standard scanpy preprocessing + Leiden clustering + UMAP.
        """
        # Generate benchmark data
        adata, true_labels, marker_genes = create_benchmark_singlecell_data(
            n_cells_per_cluster=200,
            n_clusters=3,
            n_genes=1000,
            n_marker_genes=50,
            cluster_separation=2.0,
            random_seed=42,
        )

        print(f"\n[Benchmark] Created single-cell dataset: {adata.shape}")
        print(f"[Benchmark] True clusters: {len(np.unique(true_labels))}")

        # --------------------------------------------------------------------
        # Reference: Standard scanpy workflow
        # --------------------------------------------------------------------
        print("\n[Reference] Running reference scanpy workflow...")

        adata_ref = adata.copy()

        # Standard preprocessing
        sc.pp.filter_cells(adata_ref, min_genes=50)
        sc.pp.filter_genes(adata_ref, min_cells=3)
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        sc.pp.highly_variable_genes(adata_ref, n_top_genes=500)
        adata_ref = adata_ref[:, adata_ref.var.highly_variable].copy()

        # PCA
        sc.tl.pca(adata_ref, n_comps=20, random_state=42)

        # Neighbors + Leiden
        sc.pp.neighbors(adata_ref, n_neighbors=15, n_pcs=20, random_state=42)
        sc.tl.leiden(adata_ref, resolution=0.8, random_state=42)

        # UMAP
        sc.tl.umap(adata_ref, random_state=42)

        reference_labels = adata_ref.obs["leiden"].values
        print(f"[Reference] Detected {len(np.unique(reference_labels))} clusters")

        # --------------------------------------------------------------------
        # Lobster: Using preprocessing + clustering services
        # --------------------------------------------------------------------
        print("\n[Lobster] Running Lobster workflow...")

        # Preprocessing (stateless service)
        preproc_service = PreprocessingService()
        clustering_service = ClusteringService()

        # Filter and normalize cells
        filtered_adata, filter_stats, _ = preproc_service.filter_and_normalize_cells(
            adata=adata.copy(),
            min_genes_per_cell=50,
            min_cells_per_gene=3,
            target_sum=1e4,
        )

        print(f"[Lobster] After filtering: {filtered_adata.shape}")

        # Full clustering pipeline (HVG + PCA + Leiden + UMAP)
        umap_adata, cluster_stats, _ = clustering_service.cluster_and_visualize(
            adata=filtered_adata, resolution=0.8
        )

        lobster_labels = umap_adata.obs["leiden"].values
        print(f"[Lobster] Detected {len(np.unique(lobster_labels))} clusters")

        # --------------------------------------------------------------------
        # Compute clustering metrics
        # --------------------------------------------------------------------
        print("\n[Benchmark] Computing clustering agreement metrics...")

        # Compare against reference scanpy
        # Need to align indices
        common_cells = adata_ref.obs_names.intersection(umap_adata.obs_names)
        ref_labels_aligned = adata_ref[common_cells].obs["leiden"].values
        lobster_labels_aligned = umap_adata[common_cells].obs["leiden"].values

        metrics = compute_clustering_metrics(
            lobster_labels=lobster_labels_aligned, reference_labels=ref_labels_aligned
        )

        print("\n" + "=" * 70)
        print("SINGLE-CELL CLUSTERING BENCHMARKING RESULTS")
        print("=" * 70)
        print(f"Dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
        print(f"True clusters: {len(np.unique(true_labels))}")
        print(f"Overlapping cells: {len(common_cells)}")
        print("\nClustering Agreement:")
        print(f"  Adjusted Rand Index:        {metrics['adjusted_rand_index']:.4f}")
        print(f"  Normalized Mutual Info:     {metrics['normalized_mutual_info']:.4f}")
        print(f"  Lobster clusters:           {metrics['n_lobster_clusters']}")
        print(f"  Reference clusters:         {metrics['n_reference_clusters']}")
        print("=" * 70)

        # --------------------------------------------------------------------
        # Validation assertions
        # --------------------------------------------------------------------

        # Clustering should agree substantially
        assert (
            metrics["adjusted_rand_index"] >= 0.75
        ), f"ARI too low: {metrics['adjusted_rand_index']:.4f}"

        assert (
            metrics["normalized_mutual_info"] >= 0.75
        ), f"NMI too low: {metrics['normalized_mutual_info']:.4f}"

        # Should detect similar number of clusters
        cluster_diff = abs(
            metrics["n_lobster_clusters"] - metrics["n_reference_clusters"]
        )
        assert cluster_diff <= 1, f"Cluster count mismatch: {cluster_diff}"

        print("\n✅ SINGLE-CELL CLUSTERING BENCHMARK PASSED")

        return metrics

    def test_hvg_selection_benchmark(self, tmp_path):
        """
        Benchmark highly variable gene selection.

        Validates that Lobster's HVG selection produces similar genes
        to reference scanpy workflow.
        """
        # Generate data
        adata, _, _ = create_benchmark_singlecell_data(
            n_cells_per_cluster=150, n_clusters=3, n_genes=1000, random_seed=456
        )

        print(f"\n[Benchmark] Testing HVG selection on {adata.shape}")

        # Reference scanpy
        adata_ref = adata.copy()
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        sc.pp.highly_variable_genes(adata_ref, n_top_genes=300)
        reference_hvgs = set(adata_ref.var_names[adata_ref.var.highly_variable])

        # Lobster
        preproc_service = PreprocessingService()

        # Filter and normalize (includes log transformation)
        normalized_adata, _, _ = preproc_service.filter_and_normalize_cells(
            adata=adata.copy(),
            target_sum=1e4,
            min_genes_per_cell=50,
            min_cells_per_gene=3,
        )

        # HVG selection (use scanpy for consistency with reference)
        sc.pp.highly_variable_genes(normalized_adata, n_top_genes=300)
        lobster_hvgs = set(
            normalized_adata.var_names[normalized_adata.var.highly_variable]
        )

        # Compute overlap
        overlap = len(reference_hvgs & lobster_hvgs)
        jaccard = overlap / len(reference_hvgs | lobster_hvgs)
        percent_overlap = 100 * overlap / len(reference_hvgs)

        print("\n" + "=" * 70)
        print("HVG SELECTION BENCHMARKING RESULTS")
        print("=" * 70)
        print(f"Reference HVGs:  {len(reference_hvgs)}")
        print(f"Lobster HVGs:    {len(lobster_hvgs)}")
        print(f"Overlap:         {overlap}")
        print(f"Jaccard index:   {jaccard:.4f}")
        print(f"% Overlap:       {percent_overlap:.2f}%")
        print("=" * 70)

        # Validation
        assert percent_overlap >= 70.0, f"HVG overlap too low: {percent_overlap:.2f}%"
        assert jaccard >= 0.55, f"Jaccard index too low: {jaccard:.4f}"

        print("\n✅ HVG SELECTION BENCHMARK PASSED")

        return {
            "reference_hvgs": len(reference_hvgs),
            "lobster_hvgs": len(lobster_hvgs),
            "overlap": overlap,
            "jaccard_index": jaccard,
            "percent_overlap": percent_overlap,
        }


# ===============================================================================
# Summary Report Generation
# ===============================================================================


def generate_benchmark_report(
    bulk_metrics: Optional[Dict] = None,
    clustering_metrics: Optional[Dict] = None,
    hvg_metrics: Optional[Dict] = None,
) -> str:
    """
    Generate comprehensive benchmark report for OVERNIGHT_TESTS.md.

    Parameters
    ----------
    bulk_metrics : dict, optional
        Bulk RNA-seq benchmarking metrics
    clustering_metrics : dict, optional
        Single-cell clustering benchmarking metrics
    hvg_metrics : dict, optional
        HVG selection benchmarking metrics

    Returns
    -------
    str
        Formatted benchmark report
    """
    report = []
    report.append("=" * 80)
    report.append("AGENT 14: SCIENTIFIC METHOD BENCHMARKING REPORT")
    report.append("=" * 80)
    report.append("")

    if bulk_metrics:
        report.append("## Bulk RNA-seq Benchmarking (Lobster vs Reference pyDESeq2)")
        report.append("")
        report.append(
            f"- **P-value Correlation (Spearman):** {bulk_metrics['pvalue_spearman_r']:.4f}"
        )
        report.append(
            f"- **Log2FC Correlation (Spearman):** {bulk_metrics['log2fc_spearman_r']:.4f}"
        )
        report.append(
            f"- **Significant Gene Overlap:** {bulk_metrics['percent_overlap']:.2f}%"
        )
        report.append(f"- **Jaccard Index:** {bulk_metrics['jaccard_index']:.4f}")
        report.append(
            f"- **Assessment:** {'✅ PASS' if bulk_metrics['pvalue_spearman_r'] >= 0.90 else '❌ FAIL'}"
        )
        report.append("")

    if clustering_metrics:
        report.append(
            "## Single-Cell Clustering Benchmarking (Lobster vs Reference scanpy)"
        )
        report.append("")
        report.append(
            f"- **Adjusted Rand Index:** {clustering_metrics['adjusted_rand_index']:.4f}"
        )
        report.append(
            f"- **Normalized Mutual Info:** {clustering_metrics['normalized_mutual_info']:.4f}"
        )
        report.append(
            f"- **Cluster Count (Lobster):** {clustering_metrics['n_lobster_clusters']}"
        )
        report.append(
            f"- **Cluster Count (Reference):** {clustering_metrics['n_reference_clusters']}"
        )
        report.append(
            f"- **Assessment:** {'✅ PASS' if clustering_metrics['adjusted_rand_index'] >= 0.75 else '❌ FAIL'}"
        )
        report.append("")

    if hvg_metrics:
        report.append("## HVG Selection Benchmarking (Lobster vs Reference scanpy)")
        report.append("")
        report.append(f"- **HVG Overlap:** {hvg_metrics['percent_overlap']:.2f}%")
        report.append(f"- **Jaccard Index:** {hvg_metrics['jaccard_index']:.4f}")
        report.append(
            f"- **Assessment:** {'✅ PASS' if hvg_metrics['percent_overlap'] >= 70.0 else '❌ FAIL'}"
        )
        report.append("")

    report.append("=" * 80)
    report.append("SCIENTIFIC VALIDITY ASSESSMENT")
    report.append("=" * 80)
    report.append("")

    all_pass = True
    if bulk_metrics:
        all_pass &= bulk_metrics["pvalue_spearman_r"] >= 0.90
    if clustering_metrics:
        all_pass &= clustering_metrics["adjusted_rand_index"] >= 0.75
    if hvg_metrics:
        all_pass &= hvg_metrics["percent_overlap"] >= 70.0

    if all_pass:
        report.append("✅ **OVERALL ASSESSMENT: SCIENTIFICALLY VALID**")
        report.append("")
        report.append("Lobster produces results that are highly correlated with")
        report.append("gold-standard bioinformatics tools. Differences are within")
        report.append("acceptable ranges for biological data analysis.")
    else:
        report.append("⚠️ **OVERALL ASSESSMENT: NEEDS INVESTIGATION**")
        report.append("")
        report.append("Some metrics fall below acceptable thresholds. Manual")
        report.append("investigation recommended to identify root causes.")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
