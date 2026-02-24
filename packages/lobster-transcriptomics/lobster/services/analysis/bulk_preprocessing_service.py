"""
Bulk RNA-seq preprocessing service.

Provides stateless methods for bulk RNA-seq quality assessment, gene filtering,
count normalization, and batch effect detection. All methods return the standard
3-tuple (AnnData, Dict, AnalysisStep) for provenance tracking and notebook export.

This service lives in the transcriptomics package (not core) to keep bulk
preprocessing self-contained alongside the transcriptomics expert agent.
"""

from typing import Any, Dict, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from lobster.core.analysis_ir import AnalysisStep
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class BulkPreprocessingError(Exception):
    """Base exception for bulk preprocessing operations."""

    pass


class BulkPreprocessingService:
    """
    Stateless service for bulk RNA-seq preprocessing.

    Handles sample-level QC, gene filtering, count normalization, and batch
    effect detection. Designed for count matrices where rows are samples and
    columns are genes (standard bulk RNA-seq layout in AnnData).
    """

    def __init__(self):
        """Initialize the bulk preprocessing service (stateless)."""
        logger.debug("Initializing stateless BulkPreprocessingService")

    # =========================================================================
    # BLK-03: Sample Quality Assessment
    # =========================================================================

    def assess_sample_quality(
        self,
        adata: anndata.AnnData,
        batch_key: Optional[str] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Assess sample quality via PCA outlier detection and correlation.

        Computes log-CPM, runs sample-level PCA, flags outlier samples (>3 SD
        from centroid), and computes pairwise Pearson correlation. If batch_key
        is provided, computes R-squared of batch vs PC1-3.

        Args:
            adata: AnnData (samples x genes) with raw counts
            batch_key: Optional column in .obs for batch R-squared computation

        Returns:
            Tuple of (AnnData with QC results, stats dict, AnalysisStep IR)

        Raises:
            BulkPreprocessingError: If quality assessment fails
        """
        try:
            logger.info("Starting bulk sample quality assessment")
            adata = adata.copy()

            # Get count matrix as dense
            X = adata.X
            if sparse.issparse(X):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float64)

            n_samples, n_genes = X.shape
            logger.info(f"Quality assessment: {n_samples} samples x {n_genes} genes")

            # Compute log-CPM
            library_sizes = X.sum(axis=1, keepdims=True)
            # Guard against zero library size
            library_sizes = np.where(library_sizes == 0, 1, library_sizes)
            log_cpm = np.log1p(X / library_sizes * 1e6)

            # Sample-level PCA (sklearn, not scanpy -- this is sample-level)
            n_components = min(n_samples, n_genes, 10)
            pca = PCA(n_components=n_components)
            pca_coords = pca.fit_transform(log_cpm)

            # Flag outlier samples (>3 SD from centroid)
            centroid = pca_coords.mean(axis=0)
            distances = np.sqrt(np.sum((pca_coords - centroid) ** 2, axis=1))
            dist_mean = distances.mean()
            dist_std = distances.std()
            threshold = dist_mean + 3 * dist_std if dist_std > 0 else dist_mean + 1
            is_outlier = distances > threshold

            adata.obs["is_outlier"] = is_outlier
            adata.obs["distance_from_centroid"] = distances

            # Pairwise Pearson correlation between samples
            corr_matrix = np.corrcoef(log_cpm)
            adata.uns["sample_correlation_matrix"] = corr_matrix
            median_corr = float(
                np.median(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
            )

            # Store PCA coordinates in obsm
            adata.obsm["X_pca_samples"] = pca_coords

            # Batch R-squared if batch_key provided
            batch_r_squared = None
            if batch_key is not None and batch_key in adata.obs.columns:
                batch_r_squared = self._compute_batch_r_squared(
                    pca_coords, adata.obs[batch_key].values, n_pcs=min(3, n_components)
                )
                adata.uns["batch_r_squared"] = batch_r_squared

            outlier_names = list(adata.obs_names[is_outlier])

            stats = {
                "analysis_type": "bulk_sample_quality",
                "n_samples": n_samples,
                "n_outliers": int(np.sum(is_outlier)),
                "outlier_names": outlier_names,
                "median_correlation": median_corr,
                "distance_threshold": float(threshold),
                "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
            }
            if batch_r_squared is not None:
                stats["batch_r_squared"] = batch_r_squared

            logger.info(
                f"Quality assessment complete: {np.sum(is_outlier)} outliers detected, "
                f"median correlation={median_corr:.3f}"
            )

            ir = self._create_assess_quality_ir(batch_key=batch_key)

            return adata, stats, ir

        except Exception as e:
            logger.exception(f"Error in sample quality assessment: {e}")
            raise BulkPreprocessingError(f"Sample quality assessment failed: {str(e)}")

    def _compute_batch_r_squared(
        self,
        pca_coords: np.ndarray,
        batch_labels: np.ndarray,
        n_pcs: int = 3,
    ) -> Dict[str, float]:
        """Compute R-squared of batch vs each PC using OLS."""
        result = {}
        # One-hot encode batch labels
        unique_batches = np.unique(batch_labels)
        if len(unique_batches) < 2:
            return {"PC1": 0.0}

        # Design matrix with batch indicators
        design = np.zeros((len(batch_labels), len(unique_batches)))
        for i, b in enumerate(unique_batches):
            design[batch_labels == b, i] = 1.0

        for pc_idx in range(min(n_pcs, pca_coords.shape[1])):
            y = pca_coords[:, pc_idx]
            # R-squared = 1 - SS_res / SS_tot
            ss_tot = np.sum((y - y.mean()) ** 2)
            if ss_tot == 0:
                result[f"PC{pc_idx + 1}"] = 0.0
                continue
            # OLS: beta = (X^T X)^{-1} X^T y
            beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            y_pred = design @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            r_sq = 1.0 - ss_res / ss_tot
            result[f"PC{pc_idx + 1}"] = float(r_sq)

        return result

    def _create_assess_quality_ir(
        self, batch_key: Optional[str] = None
    ) -> AnalysisStep:
        """Create AnalysisStep IR for sample quality assessment."""

        code_template = """
# Bulk RNA-seq sample quality assessment
import numpy as np
from sklearn.decomposition import PCA
from scipy import sparse

# Get counts as dense
X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X, dtype=np.float64)

# Compute log-CPM
library_sizes = X.sum(axis=1, keepdims=True)
library_sizes = np.where(library_sizes == 0, 1, library_sizes)
log_cpm = np.log1p(X / library_sizes * 1e6)

# Sample-level PCA
n_components = min(X.shape[0], X.shape[1], 10)
pca = PCA(n_components=n_components)
pca_coords = pca.fit_transform(log_cpm)

# Outlier detection (>3 SD from centroid)
centroid = pca_coords.mean(axis=0)
distances = np.sqrt(np.sum((pca_coords - centroid) ** 2, axis=1))
threshold = distances.mean() + 3 * distances.std()
adata.obs['is_outlier'] = distances > threshold
adata.obs['distance_from_centroid'] = distances

# Pairwise correlation
adata.uns['sample_correlation_matrix'] = np.corrcoef(log_cpm)

# Results stored in:
# - adata.obs['is_outlier']: Boolean outlier flag
# - adata.obs['distance_from_centroid']: PCA distance from centroid
# - adata.uns['sample_correlation_matrix']: Pairwise Pearson correlation
"""

        return AnalysisStep(
            operation="assess_sample_quality",
            tool_name="BulkPreprocessingService.assess_sample_quality",
            description="Bulk RNA-seq sample quality assessment with PCA outlier detection",
            library="sklearn + numpy",
            code_template=code_template,
            imports=[
                "import numpy as np",
                "from sklearn.decomposition import PCA",
                "from scipy import sparse",
            ],
            parameters={"batch_key": batch_key},
            parameter_schema={
                "batch_key": {
                    "type": "string",
                    "description": "Optional column for batch variance quantification",
                    "required": False,
                },
            },
            input_entities=["adata"],
            output_entities=["adata_qc"],
        )

    # =========================================================================
    # BLK-04: Gene Filtering
    # =========================================================================

    def filter_genes(
        self,
        adata: anndata.AnnData,
        min_counts: int = 10,
        min_samples: int = 3,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Filter lowly-expressed genes using bulk-appropriate criteria.

        Removes genes with total counts < min_counts across all samples AND
        genes expressed in fewer than min_samples samples (expression = count > 0).

        Args:
            adata: AnnData (samples x genes) with raw counts
            min_counts: Minimum total counts across all samples
            min_samples: Minimum number of samples with non-zero expression

        Returns:
            Tuple of (filtered AnnData, stats dict, AnalysisStep IR)

        Raises:
            BulkPreprocessingError: If gene filtering fails
        """
        try:
            logger.info(
                f"Filtering genes: min_counts={min_counts}, min_samples={min_samples}"
            )
            adata = adata.copy()

            n_genes_before = adata.n_vars

            # Get count matrix
            X = adata.X
            if sparse.issparse(X):
                total_counts = np.asarray(X.sum(axis=0)).flatten()
                n_expressing = np.asarray((X > 0).sum(axis=0)).flatten()
            else:
                X = np.asarray(X)
                total_counts = X.sum(axis=0)
                n_expressing = (X > 0).sum(axis=0)

            # Apply filters
            count_mask = total_counts >= min_counts
            sample_mask = n_expressing >= min_samples
            keep_mask = count_mask & sample_mask

            adata = adata[:, keep_mask].copy()
            n_genes_after = adata.n_vars
            n_filtered = n_genes_before - n_genes_after

            stats = {
                "analysis_type": "bulk_gene_filtering",
                "n_genes_before": n_genes_before,
                "n_genes_after": n_genes_after,
                "n_filtered": n_filtered,
                "filter_criteria": {
                    "min_counts": min_counts,
                    "min_samples": min_samples,
                },
                "n_failed_count_filter": int(np.sum(~count_mask)),
                "n_failed_sample_filter": int(np.sum(~sample_mask)),
            }

            logger.info(
                f"Gene filtering complete: {n_genes_before} -> {n_genes_after} "
                f"({n_filtered} removed)"
            )

            ir = self._create_filter_genes_ir(
                min_counts=min_counts, min_samples=min_samples
            )

            return adata, stats, ir

        except Exception as e:
            logger.exception(f"Error in gene filtering: {e}")
            raise BulkPreprocessingError(f"Gene filtering failed: {str(e)}")

    def _create_filter_genes_ir(
        self, min_counts: int, min_samples: int
    ) -> AnalysisStep:
        """Create AnalysisStep IR for gene filtering."""

        code_template = """
# Bulk RNA-seq gene filtering
import numpy as np
from scipy import sparse

X = adata.X
if sparse.issparse(X):
    total_counts = np.asarray(X.sum(axis=0)).flatten()
    n_expressing = np.asarray((X > 0).sum(axis=0)).flatten()
else:
    total_counts = np.asarray(X).sum(axis=0)
    n_expressing = (np.asarray(X) > 0).sum(axis=0)

# Filter genes by minimum counts and minimum expressing samples
keep = (total_counts >= {{ min_counts }}) & (n_expressing >= {{ min_samples }})
adata = adata[:, keep].copy()

print(f"Filtered to {adata.n_vars} genes (removed {int(np.sum(~keep))})")
"""

        return AnalysisStep(
            operation="filter_genes",
            tool_name="BulkPreprocessingService.filter_genes",
            description=f"Bulk gene filtering: min_counts={min_counts}, min_samples={min_samples}",
            library="numpy",
            code_template=code_template,
            imports=["import numpy as np", "from scipy import sparse"],
            parameters={
                "min_counts": min_counts,
                "min_samples": min_samples,
            },
            parameter_schema={
                "min_counts": {
                    "type": "integer",
                    "description": "Minimum total counts across all samples",
                    "default": 10,
                },
                "min_samples": {
                    "type": "integer",
                    "description": "Minimum samples with non-zero expression",
                    "default": 3,
                },
            },
            input_entities=["adata"],
            output_entities=["adata_filtered"],
        )

    # =========================================================================
    # BLK-05: Count Normalization
    # =========================================================================

    def normalize_counts(
        self,
        adata: anndata.AnnData,
        method: str = "deseq2",
        target_sum: Optional[float] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Normalize bulk RNA-seq counts.

        Supports DESeq2 median-of-ratios, VST, and CPM normalization.
        Raw counts are always preserved in adata.layers["counts"].

        Args:
            adata: AnnData (samples x genes) with raw counts
            method: Normalization method - "deseq2", "vst", or "cpm"
            target_sum: Target library size for CPM (default 1e6)

        Returns:
            Tuple of (normalized AnnData, stats dict, AnalysisStep IR)

        Raises:
            BulkPreprocessingError: If normalization fails
        """
        try:
            logger.info(f"Normalizing counts: method={method}")
            adata = adata.copy()

            # Store raw counts
            if sparse.issparse(adata.X):
                adata.layers["counts"] = adata.X.toarray().copy()
            else:
                adata.layers["counts"] = np.asarray(adata.X).copy()

            stats: Dict[str, Any] = {
                "analysis_type": "bulk_normalization",
                "method": method,
            }

            if method == "deseq2":
                adata, norm_stats = self._normalize_deseq2(adata)
                stats.update(norm_stats)
            elif method == "vst":
                adata, norm_stats = self._normalize_vst(adata)
                stats.update(norm_stats)
            elif method == "cpm":
                adata, norm_stats = self._normalize_cpm(adata, target_sum=target_sum)
                stats.update(norm_stats)
            else:
                raise BulkPreprocessingError(
                    f"Unknown normalization method '{method}'. "
                    "Use 'deseq2', 'vst', or 'cpm'."
                )

            logger.info(f"Normalization complete: method={method}")

            ir = self._create_normalize_ir(method=method, target_sum=target_sum)

            return adata, stats, ir

        except BulkPreprocessingError:
            raise
        except Exception as e:
            logger.exception(f"Error in normalization: {e}")
            raise BulkPreprocessingError(f"Normalization failed: {str(e)}")

    def _normalize_deseq2(
        self, adata: anndata.AnnData
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """Normalize using DESeq2 median-of-ratios via pyDESeq2."""
        try:
            from pydeseq2.dds import DeseqDataSet
        except ImportError:
            raise BulkPreprocessingError(
                "pyDESeq2 is required for DESeq2 normalization. "
                "Install with: pip install pydeseq2"
            )

        # pyDESeq2 needs counts as a DataFrame with integer values
        counts_df = pd.DataFrame(
            adata.layers["counts"].astype(int),
            index=adata.obs_names,
            columns=adata.var_names,
        )

        # Minimal metadata (pyDESeq2 requires it)
        metadata = pd.DataFrame(
            {"condition": ["A"] * adata.n_obs},
            index=adata.obs_names,
        )

        dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~1")
        dds.fit_size_factors()

        size_factors = dds.obs["size_factors"].values
        # Use pyDESeq2's precomputed normed_counts if available
        if "normed_counts" in dds.layers:
            adata.X = np.asarray(dds.layers["normed_counts"])
        else:
            adata.X = adata.layers["counts"] / size_factors[:, np.newaxis]

        stats = {
            "mean_size_factor": float(np.mean(size_factors)),
            "size_factor_range": [
                float(np.min(size_factors)),
                float(np.max(size_factors)),
            ],
        }

        return adata, stats

    def _normalize_vst(
        self, adata: anndata.AnnData
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """Variance-stabilizing transformation via pyDESeq2."""
        try:
            from pydeseq2.dds import DeseqDataSet
        except ImportError:
            raise BulkPreprocessingError(
                "pyDESeq2 is required for VST normalization. "
                "Install with: pip install pydeseq2"
            )

        counts_df = pd.DataFrame(
            adata.layers["counts"].astype(int),
            index=adata.obs_names,
            columns=adata.var_names,
        )

        metadata = pd.DataFrame(
            {"condition": ["A"] * adata.n_obs},
            index=adata.obs_names,
        )

        dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~1")
        dds.fit_size_factors()
        dds.fit_genewise_dispersions()
        dds.fit_dispersion_trend()
        dds.fit_MAP_dispersions()
        dds.vst()

        # VST values are in dds.layers["vst_counts"]
        if "vst_counts" in dds.layers:
            adata.X = np.asarray(dds.layers["vst_counts"])
        else:
            # Fallback: use log2 of size-factor normalized counts + 1
            logger.warning(
                "VST layer not found, using log2(normalized + 1) as fallback"
            )
            size_factors = dds.obs["size_factors"].values
            X_norm = adata.layers["counts"] / size_factors[:, np.newaxis]
            adata.X = np.log2(X_norm + 1)

        stats = {
            "vst_mean_range": [float(adata.X.min()), float(adata.X.max())],
        }

        return adata, stats

    def _normalize_cpm(
        self,
        adata: anndata.AnnData,
        target_sum: Optional[float] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """Counts-per-million normalization."""
        target = target_sum or 1e6

        X = adata.layers["counts"].astype(np.float64)
        library_sizes = X.sum(axis=1, keepdims=True)
        library_sizes = np.where(library_sizes == 0, 1, library_sizes)
        adata.X = X / library_sizes * target

        stats = {
            "target_sum": target,
            "library_size_range": [
                float(X.sum(axis=1).min()),
                float(X.sum(axis=1).max()),
            ],
        }

        return adata, stats

    def _create_normalize_ir(
        self, method: str, target_sum: Optional[float]
    ) -> AnalysisStep:
        """Create AnalysisStep IR for count normalization."""

        code_template = """
# Bulk RNA-seq count normalization
import numpy as np

# Preserve raw counts
adata.layers['counts'] = adata.X.copy()

{% if method == "deseq2" %}
from pydeseq2.dds import DeseqDataSet
import pandas as pd

counts_df = pd.DataFrame(adata.layers['counts'].astype(int),
                          index=adata.obs_names, columns=adata.var_names)
metadata = pd.DataFrame({'condition': ['A'] * adata.n_obs}, index=adata.obs_names)
dds = DeseqDataSet(counts=counts_df, metadata=metadata, design='~1')
dds.fit_size_factors()
size_factors = dds.obsm['size_factors']
adata.X = adata.layers['counts'] / size_factors[:, np.newaxis]
{% elif method == "vst" %}
from pydeseq2.dds import DeseqDataSet
import pandas as pd

counts_df = pd.DataFrame(adata.layers['counts'].astype(int),
                          index=adata.obs_names, columns=adata.var_names)
metadata = pd.DataFrame({'condition': ['A'] * adata.n_obs}, index=adata.obs_names)
dds = DeseqDataSet(counts=counts_df, metadata=metadata, design='~1')
dds.fit_size_factors()
dds.fit_genewise_dispersions()
dds.fit_dispersion_trend()
dds.fit_MAP_dispersions()
dds.vst()
adata.X = np.asarray(dds.layers['vst_counts'])
{% else %}
# CPM normalization
library_sizes = adata.layers['counts'].sum(axis=1, keepdims=True)
adata.X = adata.layers['counts'] / library_sizes * {{ target_sum or 1e6 }}
{% endif %}

# Results:
# - adata.X: Normalized expression values
# - adata.layers['counts']: Original raw counts preserved
"""

        return AnalysisStep(
            operation="normalize_counts",
            tool_name="BulkPreprocessingService.normalize_counts",
            description=f"Bulk count normalization using {method}",
            library="pydeseq2" if method in ("deseq2", "vst") else "numpy",
            code_template=code_template,
            imports=(
                [
                    "import numpy as np",
                    "import pandas as pd",
                    "from pydeseq2.dds import DeseqDataSet",
                ]
                if method in ("deseq2", "vst")
                else ["import numpy as np"]
            ),
            parameters={
                "method": method,
                "target_sum": target_sum,
            },
            parameter_schema={
                "method": {
                    "type": "string",
                    "description": "Normalization method",
                    "default": "deseq2",
                    "enum": ["deseq2", "vst", "cpm"],
                },
                "target_sum": {
                    "type": "number",
                    "description": "Target library size for CPM",
                    "default": 1000000,
                    "required": False,
                },
            },
            input_entities=["adata"],
            output_entities=["adata_normalized"],
        )

    # =========================================================================
    # BLK-06: Batch Effect Detection
    # =========================================================================

    def detect_batch_effects(
        self,
        adata: anndata.AnnData,
        batch_key: str,
        condition_key: Optional[str] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Detect batch effects by variance decomposition on top PCs.

        For each of the top 5 PCs, computes the proportion of variance
        explained by batch_key and optionally by condition_key. Recommends
        correction if batch variance exceeds condition variance.

        Args:
            adata: AnnData (samples x genes)
            batch_key: Column in .obs with batch labels
            condition_key: Optional column in .obs with biological condition

        Returns:
            Tuple of (AnnData with batch effect results, stats dict, AnalysisStep IR)

        Raises:
            BulkPreprocessingError: If batch effect detection fails
        """
        try:
            logger.info(
                f"Detecting batch effects: batch_key={batch_key}, "
                f"condition_key={condition_key}"
            )
            adata = adata.copy()

            # Validate keys
            if batch_key not in adata.obs.columns:
                raise BulkPreprocessingError(
                    f"Batch column '{batch_key}' not found in adata.obs. "
                    f"Available: {list(adata.obs.columns)}"
                )
            if condition_key is not None and condition_key not in adata.obs.columns:
                raise BulkPreprocessingError(
                    f"Condition column '{condition_key}' not found in adata.obs. "
                    f"Available: {list(adata.obs.columns)}"
                )

            # Compute PCA if not done
            X = adata.X
            if sparse.issparse(X):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float64)

            n_components = min(5, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_components)
            pca_coords = pca.fit_transform(X)

            # Compute variance explained by batch for each PC
            batch_labels = adata.obs[batch_key].values
            batch_variance_pcs = self._compute_variance_by_factor(
                pca_coords, batch_labels, n_components
            )

            # Compute variance explained by condition if provided
            condition_variance_pcs = None
            if condition_key is not None:
                condition_labels = adata.obs[condition_key].values
                condition_variance_pcs = self._compute_variance_by_factor(
                    pca_coords, condition_labels, n_components
                )

            # Overall batch variance (mean across top PCs)
            batch_var_mean = float(np.mean(list(batch_variance_pcs.values())))

            # Generate recommendation
            if condition_variance_pcs is not None:
                cond_var_mean = float(np.mean(list(condition_variance_pcs.values())))
                if batch_var_mean > cond_var_mean:
                    recommendation = (
                        f"Batch correction RECOMMENDED. Batch explains {batch_var_mean:.1%} "
                        f"of PC variance vs {cond_var_mean:.1%} for condition. "
                        "Consider ComBat or SVA before differential expression."
                    )
                else:
                    recommendation = (
                        f"Batch correction NOT strongly needed. Condition explains "
                        f"{cond_var_mean:.1%} of PC variance vs {batch_var_mean:.1%} for batch."
                    )
            else:
                if batch_var_mean > 0.3:
                    recommendation = (
                        f"Batch explains {batch_var_mean:.1%} of PC variance. "
                        "Consider batch correction."
                    )
                else:
                    recommendation = (
                        f"Batch explains only {batch_var_mean:.1%} of PC variance. "
                        "Batch effects appear modest."
                    )

            # Store results
            batch_effects = {
                "batch_variance_pcs": batch_variance_pcs,
                "batch_variance_mean": batch_var_mean,
                "recommendation": recommendation,
            }
            if condition_variance_pcs is not None:
                batch_effects["condition_variance_pcs"] = condition_variance_pcs
                batch_effects["condition_variance_mean"] = float(
                    np.mean(list(condition_variance_pcs.values()))
                )

            adata.uns["batch_effects"] = batch_effects

            stats = {
                "analysis_type": "batch_effect_detection",
                "batch_variance_pcs": batch_variance_pcs,
                "recommendation": recommendation,
            }
            if condition_variance_pcs is not None:
                stats["condition_variance_pcs"] = condition_variance_pcs

            logger.info(f"Batch effect detection complete: {recommendation}")

            ir = self._create_detect_batch_ir(
                batch_key=batch_key, condition_key=condition_key
            )

            return adata, stats, ir

        except BulkPreprocessingError:
            raise
        except Exception as e:
            logger.exception(f"Error in batch effect detection: {e}")
            raise BulkPreprocessingError(f"Batch effect detection failed: {str(e)}")

    def _compute_variance_by_factor(
        self,
        pca_coords: np.ndarray,
        labels: np.ndarray,
        n_pcs: int,
    ) -> Dict[str, float]:
        """Compute R-squared of factor labels vs each PC using OLS."""
        result = {}
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {f"PC{i + 1}": 0.0 for i in range(n_pcs)}

        # One-hot design matrix
        design = np.zeros((len(labels), len(unique_labels)))
        for i, label in enumerate(unique_labels):
            design[labels == label, i] = 1.0

        for pc_idx in range(n_pcs):
            y = pca_coords[:, pc_idx]
            ss_tot = np.sum((y - y.mean()) ** 2)
            if ss_tot == 0:
                result[f"PC{pc_idx + 1}"] = 0.0
                continue
            beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            y_pred = design @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            r_sq = max(0.0, 1.0 - ss_res / ss_tot)
            result[f"PC{pc_idx + 1}"] = float(r_sq)

        return result

    def _create_detect_batch_ir(
        self, batch_key: str, condition_key: Optional[str]
    ) -> AnalysisStep:
        """Create AnalysisStep IR for batch effect detection."""

        code_template = """
# Batch effect detection via variance decomposition
import numpy as np
from sklearn.decomposition import PCA

# PCA on expression data
n_components = min(5, adata.n_obs, adata.n_vars)
pca = PCA(n_components=n_components)
pca_coords = pca.fit_transform(adata.X)

# For each PC, compute R-squared with batch
batch = adata.obs['{{ batch_key }}'].values
unique_batches = np.unique(batch)
design = np.zeros((len(batch), len(unique_batches)))
for i, b in enumerate(unique_batches):
    design[batch == b, i] = 1.0

for pc_idx in range(n_components):
    y = pca_coords[:, pc_idx]
    ss_tot = np.sum((y - y.mean()) ** 2)
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    ss_res = np.sum((y - design @ beta) ** 2)
    r_sq = 1.0 - ss_res / ss_tot
    print(f"PC{pc_idx+1}: batch R-squared = {r_sq:.3f}")

# Results stored in adata.uns['batch_effects']
"""

        return AnalysisStep(
            operation="detect_batch_effects",
            tool_name="BulkPreprocessingService.detect_batch_effects",
            description=f"Batch effect detection by variance decomposition on batch_key={batch_key}",
            library="sklearn + numpy",
            code_template=code_template,
            imports=[
                "import numpy as np",
                "from sklearn.decomposition import PCA",
            ],
            parameters={
                "batch_key": batch_key,
                "condition_key": condition_key,
            },
            parameter_schema={
                "batch_key": {
                    "type": "string",
                    "description": "Column in .obs with batch labels",
                },
                "condition_key": {
                    "type": "string",
                    "description": "Optional column for condition variance comparison",
                    "required": False,
                },
            },
            input_entities=["adata"],
            output_entities=["adata_batch_detected"],
        )
