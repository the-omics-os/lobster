"""
Metabolomics quality control service for comprehensive QC assessment.

This service implements quality control methods for metabolomics data including
RSD analysis, TIC distribution, QC sample evaluation, and missing value analysis.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

from typing import Any, Dict, Tuple

import anndata
import numpy as np

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class MetabolomicsQualityError(Exception):
    """Base exception for metabolomics quality control operations."""

    pass


class MetabolomicsQualityService:
    """
    Stateless quality control service for metabolomics data.

    Provides comprehensive quality assessment for LC-MS, GC-MS, and NMR
    metabolomics data, including RSD analysis, TIC distribution evaluation,
    QC sample reproducibility, and missing value profiling.
    """

    def __init__(self):
        """Initialize the metabolomics quality service (stateless)."""
        logger.debug("Initializing stateless MetabolomicsQualityService")

    def _create_ir_assess_quality(
        self, qc_label: str, rsd_threshold: float
    ) -> AnalysisStep:
        """Create IR for quality assessment."""
        return AnalysisStep(
            operation="metabolomics_qc.assess_quality",
            tool_name="assess_metabolomics_quality",
            description="Assess metabolomics data quality: RSD, TIC CV, QC sample metrics, missing values",
            library="numpy/scipy",
            code_template="""# Metabolomics quality assessment
from lobster.services.quality.metabolomics_quality_service import MetabolomicsQualityService

service = MetabolomicsQualityService()
adata_qc, stats, _ = service.assess_quality(
    adata,
    qc_label={{ qc_label | tojson }},
    rsd_threshold={{ rsd_threshold }}
)
print(f"RSD: {stats['median_rsd']:.1f}%, Missing: {stats['missing_pct']:.1f}%, TIC CV: {stats['tic_cv']:.1f}%")""",
            imports=[
                "from lobster.services.quality.metabolomics_quality_service import MetabolomicsQualityService"
            ],
            parameters={
                "qc_label": qc_label,
                "rsd_threshold": rsd_threshold,
            },
            parameter_schema={
                "qc_label": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="QC",
                    required=False,
                    description="Label for QC/pooled samples in sample_type column",
                ),
                "rsd_threshold": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=30.0,
                    required=False,
                    validation_rule="rsd_threshold > 0",
                    description="RSD threshold (%) for flagging high-variability features",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_qc"],
        )

    def assess_quality(
        self,
        adata: anndata.AnnData,
        qc_label: str = "QC",
        rsd_threshold: float = 30.0,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Assess metabolomics data quality including RSD, TIC, QC samples, and missing values.

        Args:
            adata: AnnData object with metabolomics data (samples x features)
            qc_label: Label for QC/pooled samples in sample_type column
            rsd_threshold: RSD threshold (%) for flagging high-variability features

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                AnnData with QC metrics in obs/var, quality stats dict, and IR

        Raises:
            MetabolomicsQualityError: If quality assessment fails
        """
        try:
            logger.info("Starting metabolomics quality assessment")

            # Create working copy
            adata_qc = adata.copy()

            # Get dense matrix for calculations
            if hasattr(adata_qc.X, "toarray"):
                X = adata_qc.X.toarray().astype(np.float64)
            else:
                X = np.array(adata_qc.X, dtype=np.float64)

            n_samples, n_features = X.shape
            logger.info(
                f"Input data shape: {n_samples} samples x {n_features} features"
            )

            # --- Per-feature RSD ---
            with np.errstate(divide="ignore", invalid="ignore"):
                feature_means = np.nanmean(X, axis=0)
                feature_stds = np.nanstd(X, axis=0, ddof=1)
                rsd = np.where(
                    (feature_means != 0) & ~np.isnan(feature_means),
                    (feature_stds / np.abs(feature_means)) * 100,
                    np.nan,
                )
            adata_qc.var["rsd"] = rsd
            adata_qc.var["high_rsd"] = rsd > rsd_threshold

            # --- Per-sample TIC (Total Ion Current) ---
            tic = np.nansum(X, axis=1)
            tic_mean = np.nanmean(tic)
            tic_cv = (np.nanstd(tic) / tic_mean * 100) if tic_mean > 0 else 0.0
            adata_qc.obs["tic"] = tic

            # --- Missing value analysis ---
            is_missing = np.isnan(X)
            total_missing = is_missing.sum()
            missing_pct = (total_missing / X.size * 100) if X.size > 0 else 0.0

            # Per-sample missing rate
            sample_missing_rate = is_missing.sum(axis=1) / n_features
            adata_qc.obs["missing_rate"] = sample_missing_rate
            adata_qc.obs["detected_features"] = n_features - is_missing.sum(axis=1)

            # Per-feature missing rate
            feature_missing_rate = is_missing.sum(axis=0) / n_samples
            adata_qc.var["missing_rate"] = feature_missing_rate
            adata_qc.var["detected_samples"] = n_samples - is_missing.sum(axis=0)

            # --- QC sample evaluation ---
            qc_stats: Dict[str, Any] = {}
            if "sample_type" in adata_qc.obs.columns:
                qc_mask = (
                    adata_qc.obs["sample_type"].astype(str).str.strip() == qc_label
                )
                if qc_mask.any():
                    X_qc = X[qc_mask.values]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        qc_means = np.nanmean(X_qc, axis=0)
                        qc_stds = np.nanstd(X_qc, axis=0, ddof=1)
                        qc_rsd = np.where(
                            (qc_means != 0) & ~np.isnan(qc_means),
                            (qc_stds / np.abs(qc_means)) * 100,
                            np.nan,
                        )
                    valid_qc_rsd = qc_rsd[~np.isnan(qc_rsd)]
                    qc_stats = {
                        "n_qc_samples": int(qc_mask.sum()),
                        "median_qc_rsd": (
                            float(np.nanmedian(valid_qc_rsd))
                            if len(valid_qc_rsd) > 0
                            else 0.0
                        ),
                        "features_below_threshold": (
                            int((valid_qc_rsd < rsd_threshold).sum())
                            if len(valid_qc_rsd) > 0
                            else 0
                        ),
                    }
                    adata_qc.var["qc_rsd"] = qc_rsd

            # --- Compile stats ---
            valid_rsd = rsd[~np.isnan(rsd)]
            stats: Dict[str, Any] = {
                "n_samples": n_samples,
                "n_features": n_features,
                "median_rsd": (
                    float(np.nanmedian(valid_rsd)) if len(valid_rsd) > 0 else 0.0
                ),
                "high_rsd_features": (
                    int((valid_rsd > rsd_threshold).sum()) if len(valid_rsd) > 0 else 0
                ),
                "missing_pct": float(missing_pct),
                "tic_cv": float(tic_cv),
                "qc_stats": qc_stats,
                "analysis_type": "metabolomics_quality_assessment",
            }
            # Flatten qc_stats into top level for easy access
            stats.update(qc_stats)

            logger.info(
                f"Quality assessment complete: median RSD={stats['median_rsd']:.1f}%, "
                f"missing={stats['missing_pct']:.1f}%, TIC CV={stats['tic_cv']:.1f}%"
            )

            ir = self._create_ir_assess_quality(qc_label, rsd_threshold)
            return adata_qc, stats, ir

        except Exception as e:
            logger.exception(f"Error in metabolomics quality assessment: {e}")
            raise MetabolomicsQualityError(f"Quality assessment failed: {str(e)}")
