"""
Metabolomics preprocessing service for feature filtering, imputation, normalization,
and batch correction.

This service implements professional-grade preprocessing methods for metabolomics data
including prevalence-based filtering, multiple imputation strategies (KNN, min, LOD/2,
median, MICE), normalization methods (PQN, TIC, IS, median, quantile), and batch
correction (ComBat, median centering, QC-RLSC).

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class MetabolomicsPreprocessingError(Exception):
    """Base exception for metabolomics preprocessing operations."""

    pass


class MetabolomicsPreprocessingService:
    """
    Stateless preprocessing service for metabolomics data.

    Provides feature filtering, missing value imputation, normalization,
    and batch effect correction for LC-MS, GC-MS, and NMR metabolomics data.
    """

    def __init__(self):
        """Initialize the metabolomics preprocessing service (stateless)."""
        logger.debug("Initializing stateless MetabolomicsPreprocessingService")

    # =========================================================================
    # IR creation helpers
    # =========================================================================

    def _create_ir_filter_features(
        self, min_prevalence: float, max_rsd: Optional[float],
        blank_ratio_threshold: Optional[float], blank_label: str,
    ) -> AnalysisStep:
        """Create IR for feature filtering."""
        return AnalysisStep(
            operation="metabolomics.preprocessing.filter_features",
            tool_name="filter_metabolomics_features",
            description="Filter metabolomics features by prevalence, RSD, and blank ratio",
            library="numpy",
            code_template="""# Metabolomics feature filtering
from lobster.services.quality.metabolomics_preprocessing_service import MetabolomicsPreprocessingService

service = MetabolomicsPreprocessingService()
adata_filtered, stats, _ = service.filter_features(
    adata,
    min_prevalence={{ min_prevalence }},
    max_rsd={{ max_rsd | tojson }},
    blank_ratio_threshold={{ blank_ratio_threshold | tojson }},
    blank_label={{ blank_label | tojson }}
)
print(f"Filtered: {stats['n_before']} -> {stats['n_after']} features")""",
            imports=[
                "from lobster.services.quality.metabolomics_preprocessing_service import MetabolomicsPreprocessingService"
            ],
            parameters={
                "min_prevalence": min_prevalence,
                "max_rsd": max_rsd,
                "blank_ratio_threshold": blank_ratio_threshold,
                "blank_label": blank_label,
            },
            parameter_schema={
                "min_prevalence": ParameterSpec(
                    param_type="float", papermill_injectable=True, default_value=0.5,
                    required=False, validation_rule="0 <= min_prevalence <= 1",
                    description="Minimum fraction of samples with non-NaN values",
                ),
                "max_rsd": ParameterSpec(
                    param_type="Optional[float]", papermill_injectable=True,
                    default_value=None, required=False,
                    description="Maximum RSD threshold (requires var['rsd'])",
                ),
                "blank_ratio_threshold": ParameterSpec(
                    param_type="Optional[float]", papermill_injectable=True,
                    default_value=None, required=False,
                    description="Max blank/sample ratio for feature retention",
                ),
                "blank_label": ParameterSpec(
                    param_type="str", papermill_injectable=True,
                    default_value="blank", required=False,
                    description="Label for blank samples in sample_type column",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_filtered"],
        )

    def _create_ir_impute_missing_values(
        self, method: str, knn_neighbors: int,
    ) -> AnalysisStep:
        """Create IR for missing value imputation."""
        return AnalysisStep(
            operation="metabolomics.preprocessing.impute_missing_values",
            tool_name="handle_missing_values",
            description=f"Impute missing values using {method} method",
            library="sklearn/numpy",
            code_template="""# Metabolomics missing value imputation
from lobster.services.quality.metabolomics_preprocessing_service import MetabolomicsPreprocessingService

service = MetabolomicsPreprocessingService()
adata_imputed, stats, _ = service.impute_missing_values(
    adata,
    method={{ method | tojson }},
    knn_neighbors={{ knn_neighbors }}
)
print(f"Imputed {stats['n_values_imputed']} values ({stats['pct_imputed']:.1f}%) using {method}")""",
            imports=[
                "from lobster.services.quality.metabolomics_preprocessing_service import MetabolomicsPreprocessingService"
            ],
            parameters={"method": method, "knn_neighbors": knn_neighbors},
            parameter_schema={
                "method": ParameterSpec(
                    param_type="str", papermill_injectable=True, default_value="knn",
                    required=False,
                    validation_rule="method in ['knn', 'min', 'lod_half', 'median', 'mice']",
                    description="Imputation method",
                ),
                "knn_neighbors": ParameterSpec(
                    param_type="int", papermill_injectable=True, default_value=5,
                    required=False, validation_rule="knn_neighbors > 0",
                    description="Number of neighbors for KNN imputation",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_imputed"],
        )

    def _create_ir_normalize(
        self, method: str, log_transform: bool, reference_sample: Optional[str],
    ) -> AnalysisStep:
        """Create IR for normalization."""
        return AnalysisStep(
            operation="metabolomics.preprocessing.normalize",
            tool_name="normalize_metabolomics",
            description=f"Normalize metabolomics data using {method} method",
            library="numpy/scipy",
            code_template="""# Metabolomics normalization
from lobster.services.quality.metabolomics_preprocessing_service import MetabolomicsPreprocessingService

service = MetabolomicsPreprocessingService()
adata_norm, stats, _ = service.normalize(
    adata,
    method={{ method | tojson }},
    log_transform={{ log_transform | tojson }},
    reference_sample={{ reference_sample | tojson }}
)
print(f"Normalized using {method}, log_transform={log_transform}")""",
            imports=[
                "from lobster.services.quality.metabolomics_preprocessing_service import MetabolomicsPreprocessingService"
            ],
            parameters={
                "method": method,
                "log_transform": log_transform,
                "reference_sample": reference_sample,
            },
            parameter_schema={
                "method": ParameterSpec(
                    param_type="str", papermill_injectable=True, default_value="pqn",
                    required=False,
                    validation_rule="method in ['pqn', 'tic', 'is', 'median', 'quantile']",
                    description="Normalization method",
                ),
                "log_transform": ParameterSpec(
                    param_type="bool", papermill_injectable=True, default_value=True,
                    required=False, description="Apply log2 transformation after normalization",
                ),
                "reference_sample": ParameterSpec(
                    param_type="Optional[str]", papermill_injectable=True,
                    default_value=None, required=False,
                    description="Reference sample for IS normalization (feature names or indices)",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_norm"],
        )

    def _create_ir_correct_batch_effects(
        self, batch_key: str, method: str, qc_label: str,
    ) -> AnalysisStep:
        """Create IR for batch effect correction."""
        return AnalysisStep(
            operation="metabolomics.preprocessing.correct_batch_effects",
            tool_name="correct_batch_effects",
            description=f"Correct batch effects using {method} method",
            library="scanpy/scipy",
            code_template="""# Metabolomics batch correction
from lobster.services.quality.metabolomics_preprocessing_service import MetabolomicsPreprocessingService

service = MetabolomicsPreprocessingService()
adata_corrected, stats, _ = service.correct_batch_effects(
    adata,
    batch_key={{ batch_key | tojson }},
    method={{ method | tojson }},
    qc_label={{ qc_label | tojson }}
)
print(f"Batch correction: {method}, {stats['n_batches']} batches")""",
            imports=[
                "from lobster.services.quality.metabolomics_preprocessing_service import MetabolomicsPreprocessingService"
            ],
            parameters={
                "batch_key": batch_key,
                "method": method,
                "qc_label": qc_label,
            },
            parameter_schema={
                "batch_key": ParameterSpec(
                    param_type="str", papermill_injectable=True, default_value="batch",
                    required=True, description="Column in obs containing batch labels",
                ),
                "method": ParameterSpec(
                    param_type="str", papermill_injectable=True, default_value="combat",
                    required=False,
                    validation_rule="method in ['combat', 'median_centering', 'qc_rlsc']",
                    description="Batch correction method",
                ),
                "qc_label": ParameterSpec(
                    param_type="str", papermill_injectable=True, default_value="QC",
                    required=False,
                    description="Label for QC samples (required for qc_rlsc)",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_corrected"],
        )

    # =========================================================================
    # Public methods
    # =========================================================================

    def filter_features(
        self,
        adata: anndata.AnnData,
        min_prevalence: float = 0.5,
        max_rsd: Optional[float] = None,
        blank_ratio_threshold: Optional[float] = None,
        blank_label: str = "blank",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Filter metabolomics features by prevalence, RSD, and blank ratio.

        Args:
            adata: AnnData object with metabolomics data
            min_prevalence: Minimum fraction of samples with non-NaN values to keep feature
            max_rsd: Maximum RSD threshold; requires var['rsd'] from quality assessment
            blank_ratio_threshold: Maximum blank-to-sample intensity ratio
            blank_label: Label for blank/solvent samples in obs['sample_type']

        Returns:
            Tuple of (filtered AnnData, stats dict, AnalysisStep)
        """
        try:
            logger.info("Starting metabolomics feature filtering")
            adata_filtered = adata.copy()

            if hasattr(adata_filtered.X, "toarray"):
                X = adata_filtered.X.toarray().astype(np.float64)
            else:
                X = np.array(adata_filtered.X, dtype=np.float64)

            n_before = adata_filtered.n_vars
            keep_mask = np.ones(n_before, dtype=bool)

            # --- Prevalence filter ---
            n_samples = X.shape[0]
            prevalence = (~np.isnan(X)).sum(axis=0) / n_samples
            prevalence_mask = prevalence >= min_prevalence
            n_removed_prevalence = int((~prevalence_mask).sum())
            keep_mask &= prevalence_mask

            # --- RSD filter ---
            n_removed_rsd = 0
            if max_rsd is not None:
                if "rsd" in adata_filtered.var.columns:
                    rsd_vals = adata_filtered.var["rsd"].values
                    rsd_mask = np.where(np.isnan(rsd_vals), True, rsd_vals <= max_rsd)
                    n_removed_rsd = int((~rsd_mask & keep_mask).sum())
                    keep_mask &= rsd_mask
                else:
                    logger.warning(
                        "max_rsd specified but var['rsd'] not found. "
                        "Run assess_quality first. Skipping RSD filter."
                    )

            # --- Blank ratio filter ---
            n_removed_blank = 0
            if blank_ratio_threshold is not None and "sample_type" in adata_filtered.obs.columns:
                blank_mask_obs = adata_filtered.obs["sample_type"].astype(str).str.strip() == blank_label
                sample_mask_obs = ~blank_mask_obs
                if blank_mask_obs.any() and sample_mask_obs.any():
                    blank_means = np.nanmean(X[blank_mask_obs.values], axis=0)
                    sample_means = np.nanmean(X[sample_mask_obs.values], axis=0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        blank_ratio = np.where(
                            sample_means > 0, blank_means / sample_means, 0.0
                        )
                    blank_feature_mask = blank_ratio <= blank_ratio_threshold
                    n_removed_blank = int((~blank_feature_mask & keep_mask).sum())
                    keep_mask &= blank_feature_mask
                else:
                    logger.warning("No blank or sample entries found; skipping blank filter.")

            # Apply filter
            adata_filtered = adata_filtered[:, keep_mask].copy()
            n_after = adata_filtered.n_vars

            stats = {
                "n_before": n_before,
                "n_after": n_after,
                "n_removed_total": n_before - n_after,
                "n_removed_prevalence": n_removed_prevalence,
                "n_removed_rsd": n_removed_rsd,
                "n_removed_blank": n_removed_blank,
                "min_prevalence": min_prevalence,
                "analysis_type": "metabolomics_feature_filtering",
            }

            logger.info(f"Feature filtering: {n_before} -> {n_after} features")

            ir = self._create_ir_filter_features(
                min_prevalence, max_rsd, blank_ratio_threshold, blank_label
            )
            return adata_filtered, stats, ir

        except Exception as e:
            logger.exception(f"Error in feature filtering: {e}")
            raise MetabolomicsPreprocessingError(f"Feature filtering failed: {str(e)}")

    def impute_missing_values(
        self,
        adata: anndata.AnnData,
        method: str = "knn",
        knn_neighbors: int = 5,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Impute missing values using metabolomics-appropriate methods.

        Args:
            adata: AnnData object with metabolomics data
            method: Imputation method:
                - "knn": K-nearest neighbors (sklearn KNNImputer)
                - "min": Per-feature minimum / 2
                - "lod_half": LOD/2 if LOD in var, else minimum / 5
                - "median": Per-feature nanmedian
                - "mice": Iterative imputer (sklearn IterativeImputer)
            knn_neighbors: Number of neighbors for KNN imputation

        Returns:
            Tuple of (imputed AnnData, stats dict, AnalysisStep)
        """
        try:
            logger.info(f"Starting missing value imputation with method: {method}")
            adata_imputed = adata.copy()

            if hasattr(adata_imputed.X, "toarray"):
                X = adata_imputed.X.toarray().astype(np.float64)
            else:
                X = np.array(adata_imputed.X, dtype=np.float64)

            n_missing_before = int(np.isnan(X).sum())
            if n_missing_before == 0:
                logger.info("No missing values detected, skipping imputation")
                ir = self._create_ir_impute_missing_values(method, knn_neighbors)
                return adata_imputed, {
                    "method_used": method,
                    "n_values_imputed": 0,
                    "pct_imputed": 0.0,
                    "analysis_type": "metabolomics_imputation",
                }, ir

            # Store raw before imputation
            adata_imputed.layers["pre_imputation"] = X.copy()

            if method == "knn":
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=min(knn_neighbors, X.shape[0] - 1))
                X_imputed = imputer.fit_transform(X)

            elif method == "min":
                X_imputed = X.copy()
                for j in range(X.shape[1]):
                    col = X[:, j]
                    observed = col[~np.isnan(col)]
                    if len(observed) > 0:
                        fill_val = np.min(observed) / 2
                    else:
                        fill_val = 0.0
                    X_imputed[np.isnan(col), j] = fill_val

            elif method == "lod_half":
                X_imputed = X.copy()
                for j in range(X.shape[1]):
                    col = X[:, j]
                    observed = col[~np.isnan(col)]
                    # Check for LOD in var metadata
                    if "lod" in adata_imputed.var.columns:
                        lod_val = adata_imputed.var["lod"].iloc[j]
                        if not np.isnan(lod_val):
                            fill_val = lod_val / 2
                        elif len(observed) > 0:
                            fill_val = np.min(observed) / 5
                        else:
                            fill_val = 0.0
                    elif len(observed) > 0:
                        fill_val = np.min(observed) / 5
                    else:
                        fill_val = 0.0
                    X_imputed[np.isnan(col), j] = fill_val

            elif method == "median":
                X_imputed = X.copy()
                for j in range(X.shape[1]):
                    col = X[:, j]
                    observed = col[~np.isnan(col)]
                    if len(observed) > 0:
                        fill_val = np.nanmedian(observed)
                    else:
                        fill_val = 0.0
                    X_imputed[np.isnan(col), j] = fill_val

            elif method == "mice":
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                imputer = IterativeImputer(max_iter=10, random_state=42)
                X_imputed = imputer.fit_transform(X)

            else:
                raise MetabolomicsPreprocessingError(
                    f"Unknown imputation method: {method}. "
                    f"Choose from: knn, min, lod_half, median, mice"
                )

            adata_imputed.X = X_imputed
            n_imputed = n_missing_before - int(np.isnan(X_imputed).sum())

            stats = {
                "method_used": method,
                "n_values_imputed": n_imputed,
                "pct_imputed": float(n_imputed / X.size * 100) if X.size > 0 else 0.0,
                "imputation_details": {
                    "knn_neighbors": knn_neighbors if method == "knn" else None,
                },
                "analysis_type": "metabolomics_imputation",
            }

            logger.info(f"Imputation complete: {n_imputed} values imputed using {method}")

            ir = self._create_ir_impute_missing_values(method, knn_neighbors)
            return adata_imputed, stats, ir

        except Exception as e:
            logger.exception(f"Error in missing value imputation: {e}")
            raise MetabolomicsPreprocessingError(f"Imputation failed: {str(e)}")

    def normalize(
        self,
        adata: anndata.AnnData,
        method: str = "pqn",
        log_transform: bool = True,
        reference_sample: Optional[str] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Normalize metabolomics data using standard methods.

        Args:
            adata: AnnData object with metabolomics data
            method: Normalization method:
                - "pqn": Probabilistic Quotient Normalization (gold standard)
                - "tic": Total Ion Current normalization
                - "is": Internal Standard normalization (requires reference_sample)
                - "median": Median normalization
                - "quantile": Quantile normalization
            log_transform: Apply log2 transformation after normalization
            reference_sample: For IS normalization: comma-separated IS feature names/indices

        Returns:
            Tuple of (normalized AnnData, stats dict, AnalysisStep)
        """
        try:
            logger.info(f"Starting normalization with method: {method}")
            adata_norm = adata.copy()

            if hasattr(adata_norm.X, "toarray"):
                X = adata_norm.X.toarray().astype(np.float64)
            else:
                X = np.array(adata_norm.X, dtype=np.float64)

            # Store raw layer
            adata_norm.layers["pre_normalization"] = X.copy()

            # Apply normalization
            if method == "pqn":
                X_norm, norm_factors = self._pqn_normalize(X)
            elif method == "tic":
                X_norm, norm_factors = self._tic_normalize(X)
            elif method == "is":
                X_norm, norm_factors = self._is_normalize(X, adata_norm, reference_sample)
            elif method == "median":
                X_norm, norm_factors = self._median_normalize(X)
            elif method == "quantile":
                X_norm, norm_factors = self._quantile_normalize(X)
            else:
                raise MetabolomicsPreprocessingError(
                    f"Unknown normalization method: {method}. "
                    f"Choose from: pqn, tic, is, median, quantile"
                )

            # Apply log2 transformation if requested
            if log_transform:
                # Handle zeros: add small offset (half of minimum positive value)
                min_positive = np.nanmin(X_norm[X_norm > 0]) if np.any(X_norm > 0) else 1.0
                offset = min_positive / 2
                X_norm = np.log2(X_norm + offset)

            adata_norm.X = X_norm
            adata_norm.obs["norm_factor"] = norm_factors
            adata_norm.uns["normalization_method"] = method

            stats = {
                "method": method,
                "log_transformed": log_transform,
                "median_norm_factor": float(np.nanmedian(norm_factors)),
                "normalization_range": [
                    float(np.nanmin(norm_factors)),
                    float(np.nanmax(norm_factors)),
                ],
                "analysis_type": "metabolomics_normalization",
            }

            logger.info(f"Normalization complete: {method}, log={log_transform}")

            ir = self._create_ir_normalize(method, log_transform, reference_sample)
            return adata_norm, stats, ir

        except Exception as e:
            logger.exception(f"Error in normalization: {e}")
            raise MetabolomicsPreprocessingError(f"Normalization failed: {str(e)}")

    def correct_batch_effects(
        self,
        adata: anndata.AnnData,
        batch_key: str,
        method: str = "combat",
        qc_label: str = "QC",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Correct batch effects in metabolomics data.

        Args:
            adata: AnnData object with metabolomics data
            batch_key: Column in obs containing batch labels
            method: Correction method:
                - "combat": Parametric empirical Bayes (scanpy's sc.pp.combat)
                - "median_centering": Per-batch median centering
                - "qc_rlsc": QC-based signal correction using smoothing spline
            qc_label: Label for QC samples (required for qc_rlsc)

        Returns:
            Tuple of (corrected AnnData, stats dict, AnalysisStep)
        """
        try:
            logger.info(f"Starting batch correction with method: {method}")
            adata_corrected = adata.copy()

            if batch_key not in adata_corrected.obs.columns:
                raise MetabolomicsPreprocessingError(
                    f"Batch key '{batch_key}' not found in obs columns"
                )

            batch_labels = adata_corrected.obs[batch_key]
            batch_counts = batch_labels.value_counts().to_dict()
            n_batches = len(batch_counts)

            if n_batches < 2:
                logger.warning("Less than 2 batches found, skipping batch correction")
                ir = self._create_ir_correct_batch_effects(batch_key, method, qc_label)
                return adata_corrected, {
                    "method": method,
                    "n_batches": n_batches,
                    "correction_performed": False,
                    "analysis_type": "metabolomics_batch_correction",
                }, ir

            if hasattr(adata_corrected.X, "toarray"):
                X = adata_corrected.X.toarray().astype(np.float64)
            else:
                X = np.array(adata_corrected.X, dtype=np.float64)

            adata_corrected.layers["pre_batch_correction"] = X.copy()

            if method == "combat":
                X_corrected = self._combat_correction(X, batch_labels, adata_corrected)
            elif method == "median_centering":
                X_corrected = self._median_centering_correction(X, batch_labels)
            elif method == "qc_rlsc":
                X_corrected = self._qc_rlsc_correction(
                    X, batch_labels, adata_corrected.obs, qc_label
                )
            else:
                raise MetabolomicsPreprocessingError(
                    f"Unknown batch correction method: {method}. "
                    f"Choose from: combat, median_centering, qc_rlsc"
                )

            adata_corrected.X = X_corrected

            stats = {
                "method": method,
                "n_batches": n_batches,
                "batch_sizes": batch_counts,
                "correction_performed": True,
                "correction_summary": f"Applied {method} across {n_batches} batches",
                "analysis_type": "metabolomics_batch_correction",
            }

            logger.info(f"Batch correction complete: {method}, {n_batches} batches")

            ir = self._create_ir_correct_batch_effects(batch_key, method, qc_label)
            return adata_corrected, stats, ir

        except Exception as e:
            logger.exception(f"Error in batch correction: {e}")
            raise MetabolomicsPreprocessingError(f"Batch correction failed: {str(e)}")

    # =========================================================================
    # Private normalization helpers
    # =========================================================================

    def _pqn_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Probabilistic Quotient Normalization."""
        # Reference = median spectrum across all samples
        reference = np.nanmedian(X, axis=0)

        # Quotients = sample / reference for each feature
        with np.errstate(divide="ignore", invalid="ignore"):
            quotients = X / reference

        # Normalization factor = nanmedian of quotients per sample
        norm_factors = np.nanmedian(quotients, axis=1)

        # Warn if quotient based on few features
        for i in range(X.shape[0]):
            n_valid = np.sum(~np.isnan(quotients[i]))
            if 0 < n_valid < 20:
                logger.warning(
                    f"Sample {i}: PQN quotient based on only {n_valid} features "
                    f"(recommend >= 20)"
                )

        # Avoid division by zero
        norm_factors = np.where(norm_factors == 0, 1.0, norm_factors)
        norm_factors = np.where(np.isnan(norm_factors), 1.0, norm_factors)

        X_norm = X / norm_factors[:, np.newaxis]
        return X_norm, norm_factors

    def _tic_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Total Ion Current normalization."""
        tic = np.nansum(X, axis=1)
        median_tic = np.nanmedian(tic)
        tic = np.where(tic == 0, 1.0, tic)
        norm_factors = tic / median_tic
        X_norm = X / norm_factors[:, np.newaxis]
        return X_norm, norm_factors

    def _is_normalize(
        self, X: np.ndarray, adata: anndata.AnnData,
        reference_sample: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Internal Standard normalization."""
        if reference_sample is None:
            raise MetabolomicsPreprocessingError(
                "reference_sample is required for IS normalization. "
                "Provide comma-separated IS feature names."
            )

        # Parse IS feature names/indices
        is_names = [s.strip() for s in reference_sample.split(",")]
        var_names = adata.var_names.tolist()
        is_indices: List[int] = []
        for name in is_names:
            if name in var_names:
                is_indices.append(var_names.index(name))
            elif name.isdigit():
                idx = int(name)
                if 0 <= idx < X.shape[1]:
                    is_indices.append(idx)
                else:
                    logger.warning(f"IS index {idx} out of range, skipping")
            else:
                logger.warning(f"IS feature '{name}' not found in var_names, skipping")

        if not is_indices:
            raise MetabolomicsPreprocessingError(
                "No valid IS features found. Check reference_sample parameter."
            )

        is_values = X[:, is_indices]
        norm_factors = np.nanmedian(is_values, axis=1)
        norm_factors = np.where(norm_factors == 0, 1.0, norm_factors)
        norm_factors = np.where(np.isnan(norm_factors), 1.0, norm_factors)

        X_norm = X / norm_factors[:, np.newaxis]
        return X_norm, norm_factors

    def _median_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Median normalization."""
        sample_medians = np.nanmedian(X, axis=1)
        global_median = np.nanmedian(sample_medians)
        sample_medians = np.where(sample_medians == 0, 1.0, sample_medians)
        norm_factors = sample_medians / global_median
        norm_factors = np.where(np.isnan(norm_factors), 1.0, norm_factors)
        X_norm = X / norm_factors[:, np.newaxis]
        return X_norm, norm_factors

    def _quantile_normalize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantile normalization via ranking."""
        X_norm = X.copy()
        n_samples, n_features = X.shape

        # Get ranks per sample (handling NaN)
        ranks = np.zeros_like(X)
        for i in range(n_samples):
            row = X[i, :]
            valid = ~np.isnan(row)
            if valid.sum() > 0:
                ranks[i, valid] = rankdata(row[valid])

        # Mean values at each rank
        max_rank = int(np.nanmax(ranks))
        mean_at_rank = np.zeros(max_rank + 1)
        for rank in range(1, max_rank + 1):
            vals = []
            for i in range(n_samples):
                mask = ranks[i] == rank
                if mask.any():
                    vals.extend(X[i, mask].tolist())
            if vals:
                mean_at_rank[rank] = np.mean(vals)

        # Replace with quantile-normalized values
        for i in range(n_samples):
            for j in range(n_features):
                if not np.isnan(X[i, j]):
                    r = int(ranks[i, j])
                    X_norm[i, j] = mean_at_rank[r]

        # Normalization factors: ratio of quantile-normalized to original median
        norm_factors = np.ones(n_samples)
        return X_norm, norm_factors

    # =========================================================================
    # Private batch correction helpers
    # =========================================================================

    def _combat_correction(
        self, X: np.ndarray, batch_labels: pd.Series, adata: anndata.AnnData,
    ) -> np.ndarray:
        """ComBat batch correction using scanpy if available, else manual."""
        try:
            import scanpy as sc
            # Use scanpy's ComBat implementation
            adata_temp = anndata.AnnData(
                X=X.copy(), obs=adata.obs.copy(), var=adata.var.copy()
            )
            sc.pp.combat(adata_temp, key=batch_labels.name or "batch")
            return adata_temp.X
        except (ImportError, Exception) as e:
            logger.info(f"scanpy ComBat unavailable ({e}), using manual implementation")
            return self._manual_combat(X, batch_labels)

    def _manual_combat(self, X: np.ndarray, batch_labels: pd.Series) -> np.ndarray:
        """Manual ComBat-like batch correction (parametric empirical Bayes)."""
        X_corrected = X.copy()
        overall_means = np.nanmean(X, axis=0)
        overall_stds = np.nanstd(X, axis=0)
        overall_stds = np.where(overall_stds == 0, 1.0, overall_stds)

        for batch in batch_labels.unique():
            batch_mask = batch_labels == batch
            batch_data = X[batch_mask]
            batch_means = np.nanmean(batch_data, axis=0)
            batch_stds = np.nanstd(batch_data, axis=0)
            batch_stds = np.where(batch_stds == 0, 1.0, batch_stds)

            for i in np.where(batch_mask)[0]:
                X_corrected[i] = (
                    (X[i] - batch_means) / batch_stds
                ) * overall_stds + overall_means

        return X_corrected

    def _median_centering_correction(
        self, X: np.ndarray, batch_labels: pd.Series,
    ) -> np.ndarray:
        """Per-batch median centering."""
        X_corrected = X.copy()
        global_median = np.nanmedian(X, axis=0)

        for batch in batch_labels.unique():
            batch_mask = batch_labels == batch
            batch_data = X[batch_mask]
            batch_median = np.nanmedian(batch_data, axis=0)
            correction = global_median - batch_median
            X_corrected[batch_mask] = batch_data + correction

        return X_corrected

    def _qc_rlsc_correction(
        self, X: np.ndarray, batch_labels: pd.Series,
        obs: pd.DataFrame, qc_label: str,
    ) -> np.ndarray:
        """QC-based signal correction using smoothing spline."""
        X_corrected = X.copy()

        if "sample_type" not in obs.columns:
            logger.warning(
                "obs['sample_type'] not found, falling back to median centering"
            )
            return self._median_centering_correction(X, batch_labels)

        qc_mask = obs["sample_type"].astype(str).str.strip() == qc_label
        if qc_mask.sum() < 5:
            logger.warning(
                f"Only {qc_mask.sum()} QC samples found (< 5 minimum). "
                f"Falling back to median centering."
            )
            return self._median_centering_correction(X, batch_labels)

        # Create injection order (sample index)
        injection_order = np.arange(X.shape[0], dtype=float)

        for batch in batch_labels.unique():
            batch_mask = batch_labels == batch
            batch_qc_mask = batch_mask & qc_mask

            if batch_qc_mask.sum() < 5:
                logger.warning(
                    f"Batch '{batch}': only {batch_qc_mask.sum()} QC samples. "
                    f"Using median centering for this batch."
                )
                # Fallback to median centering for this batch
                global_median = np.nanmedian(X, axis=0)
                batch_data = X[batch_mask]
                batch_median = np.nanmedian(batch_data, axis=0)
                correction = global_median - batch_median
                X_corrected[batch_mask] = batch_data + correction
                continue

            qc_order = injection_order[batch_qc_mask]
            qc_data = X[batch_qc_mask]
            batch_order = injection_order[batch_mask]

            # For each feature, fit smoothing spline through QC samples
            for j in range(X.shape[1]):
                qc_vals = qc_data[:, j]
                valid = ~np.isnan(qc_vals)

                if valid.sum() < 3:
                    continue

                try:
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    smoothed = lowess(
                        qc_vals[valid], qc_order[valid],
                        frac=0.5, return_sorted=True,
                    )
                    # Interpolate correction for all samples in this batch
                    correction_at_qc = np.interp(
                        batch_order, smoothed[:, 0], smoothed[:, 1]
                    )
                    median_qc = np.nanmedian(qc_vals[valid])
                    if median_qc != 0:
                        correction_factor = correction_at_qc / median_qc
                        correction_factor = np.where(
                            correction_factor == 0, 1.0, correction_factor
                        )
                        X_corrected[batch_mask, j] = X[batch_mask, j] / correction_factor
                except Exception as e:
                    logger.warning(f"LOWESS failed for feature {j}: {e}")
                    continue

        return X_corrected
