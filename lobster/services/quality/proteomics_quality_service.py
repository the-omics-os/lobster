"""
Proteomics quality control service for comprehensive quality assessment and validation.

This service implements professional-grade quality control methods specifically designed for
proteomics data including missing value pattern analysis, contaminant detection, CV assessment,
dynamic range evaluation, and technical replicate validation.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsQualityError(Exception):
    """Base exception for proteomics quality control operations."""

    pass


class ProteomicsQualityService:
    """
    Advanced quality control service for proteomics data.

    This stateless service provides comprehensive quality assessment methods
    following best practices from proteomics analysis pipelines. Handles missing value
    pattern analysis, contaminant detection, coefficient of variation assessment,
    and technical replicate validation.
    """

    def __init__(self):
        """
        Initialize the proteomics quality service.

        This service is stateless and doesn't require a data manager instance.
        """
        logger.debug("Initializing stateless ProteomicsQualityService")

        # Define common contaminant patterns
        self.contaminant_patterns = {
            "keratin": ["KRT", "KERATIN", "KER_"],
            "common_contaminants": ["CON_", "CONTAM_", "CONT_"],
            "reverse_hits": ["REV_", "REVERSE_", "rev_"],
            "trypsin": ["TRYP_", "TRYPSIN"],
            "albumin": ["ALB_", "ALBUMIN", "BSA"],
            "immunoglobulin": ["IGG_", "IGH_", "IGL_", "IGK_"],
        }

        logger.debug("ProteomicsQualityService initialized successfully")

    def _create_ir_missing_value_patterns(
        self, sample_threshold: float, protein_threshold: float
    ) -> AnalysisStep:
        """Create IR for missing value pattern analysis."""
        return AnalysisStep(
            operation="proteomics.qc.assess_missing_value_patterns",
            tool_name="assess_missing_value_patterns",
            description="Analyze missing value patterns in proteomics data with MNAR/MCAR detection",
            library="lobster.services.quality.proteomics_quality_service",
            code_template="""# Missing value pattern analysis
from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService

service = ProteomicsQualityService()
adata_qc, stats, _ = service.assess_missing_value_patterns(
    adata,
    sample_threshold={{ sample_threshold }},
    protein_threshold={{ protein_threshold }}
)
print(f"Missing: {stats['total_missing_percentage']:.1f}%, MNAR: {stats['mnar_proteins']}, MCAR: {stats['mcar_proteins']}")""",
            imports=[
                "from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService"
            ],
            parameters={
                "sample_threshold": sample_threshold,
                "protein_threshold": protein_threshold,
            },
            parameter_schema={
                "sample_threshold": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.7,
                    required=False,
                    validation_rule="0 < sample_threshold <= 1",
                    description="Threshold for high missing value samples (fraction)",
                ),
                "protein_threshold": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.8,
                    required=False,
                    validation_rule="0 < protein_threshold <= 1",
                    description="Threshold for high missing value proteins (fraction)",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_qc"],
        )

    def _create_ir_coefficient_variation(
        self,
        replicate_column: Optional[str],
        cv_threshold: float,
        min_observations: int,
    ) -> AnalysisStep:
        """Create IR for coefficient of variation assessment."""
        return AnalysisStep(
            operation="proteomics.qc.assess_coefficient_variation",
            tool_name="assess_coefficient_variation",
            description="Assess coefficient of variation (CV) for proteins with optional replicate grouping",
            library="lobster.services.quality.proteomics_quality_service",
            code_template="""# Coefficient of variation assessment
from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService

service = ProteomicsQualityService()
adata_qc, stats, _ = service.assess_coefficient_variation(
    adata,
    replicate_column={{ replicate_column | tojson }},
    cv_threshold={{ cv_threshold }},
    min_observations={{ min_observations }}
)
print(f"Median CV: {stats['median_cv_across_proteins']:.3f}, High CV proteins: {stats['n_high_cv_proteins']}")""",
            imports=[
                "from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService"
            ],
            parameters={
                "replicate_column": replicate_column,
                "cv_threshold": cv_threshold,
                "min_observations": min_observations,
            },
            parameter_schema={
                "replicate_column": ParameterSpec(
                    param_type="Optional[str]",
                    papermill_injectable=True,
                    default_value=None,
                    required=False,
                    description="Column in obs containing replicate groups",
                ),
                "cv_threshold": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.2,
                    required=False,
                    validation_rule="cv_threshold > 0",
                    description="Threshold for high CV proteins (fractional, e.g., 0.2 = 20%)",
                ),
                "min_observations": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=3,
                    required=False,
                    validation_rule="min_observations >= 2",
                    description="Minimum observations required for CV calculation",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_qc"],
        )

    def _create_ir_detect_contaminants(
        self, protein_id_column: Optional[str], custom_patterns: Optional[Dict[str, List[str]]]
    ) -> AnalysisStep:
        """Create IR for contaminant detection."""
        return AnalysisStep(
            operation="proteomics.qc.detect_contaminants",
            tool_name="detect_contaminants",
            description="Detect contaminant proteins based on naming patterns (keratins, trypsin, etc.)",
            library="lobster.services.quality.proteomics_quality_service",
            code_template="""# Contaminant detection
from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService

service = ProteomicsQualityService()
adata_qc, stats, _ = service.detect_contaminants(
    adata,
    protein_id_column={{ protein_id_column | tojson }}
)
print(f"Contaminants: {stats['total_contaminants']} ({stats['contaminant_percentage']:.1f}%)")""",
            imports=[
                "from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService"
            ],
            parameters={
                "protein_id_column": protein_id_column,
                "custom_patterns": custom_patterns,
            },
            parameter_schema={
                "protein_id_column": ParameterSpec(
                    param_type="Optional[str]",
                    papermill_injectable=True,
                    default_value=None,
                    required=False,
                    description="Column in var containing protein IDs (uses index if None)",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_qc"],
        )

    def _create_ir_evaluate_dynamic_range(
        self, percentile_low: float, percentile_high: float
    ) -> AnalysisStep:
        """Create IR for dynamic range evaluation."""
        return AnalysisStep(
            operation="proteomics.qc.evaluate_dynamic_range",
            tool_name="evaluate_dynamic_range",
            description="Evaluate dynamic range of proteomics measurements",
            library="lobster.services.quality.proteomics_quality_service",
            code_template="""# Dynamic range evaluation
from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService

service = ProteomicsQualityService()
adata_qc, stats, _ = service.evaluate_dynamic_range(
    adata,
    percentile_low={{ percentile_low }},
    percentile_high={{ percentile_high }}
)
print(f"Median dynamic range: {stats['median_sample_dynamic_range']:.2f} log10 units")""",
            imports=[
                "from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService"
            ],
            parameters={
                "percentile_low": percentile_low,
                "percentile_high": percentile_high,
            },
            parameter_schema={
                "percentile_low": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=5.0,
                    required=False,
                    validation_rule="0 < percentile_low < 50",
                    description="Lower percentile for dynamic range calculation",
                ),
                "percentile_high": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=95.0,
                    required=False,
                    validation_rule="50 < percentile_high < 100",
                    description="Higher percentile for dynamic range calculation",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_qc"],
        )

    def _create_ir_detect_pca_outliers(
        self, n_components: int, outlier_threshold: float
    ) -> AnalysisStep:
        """Create IR for PCA outlier detection."""
        return AnalysisStep(
            operation="proteomics.qc.detect_pca_outliers",
            tool_name="detect_pca_outliers",
            description="Detect outlier samples using PCA analysis",
            library="lobster.services.quality.proteomics_quality_service",
            code_template="""# PCA outlier detection
from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService

service = ProteomicsQualityService()
adata_qc, stats, _ = service.detect_pca_outliers(
    adata,
    n_components={{ n_components }},
    outlier_threshold={{ outlier_threshold }}
)
print(f"Outliers: {stats['n_outliers_detected']} ({stats['outlier_percentage']:.1f}%)")""",
            imports=[
                "from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService"
            ],
            parameters={
                "n_components": n_components,
                "outlier_threshold": outlier_threshold,
            },
            parameter_schema={
                "n_components": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=50,
                    required=False,
                    validation_rule="n_components > 0",
                    description="Number of PCA components to compute",
                ),
                "outlier_threshold": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=3.0,
                    required=False,
                    validation_rule="outlier_threshold > 0",
                    description="Threshold in standard deviations for outlier detection",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_qc"],
        )

    def _create_ir_assess_technical_replicates(
        self, replicate_column: str, correlation_method: str
    ) -> AnalysisStep:
        """Create IR for technical replicate assessment."""
        return AnalysisStep(
            operation="proteomics.qc.assess_technical_replicates",
            tool_name="assess_technical_replicates",
            description="Assess technical replicate reproducibility and variation",
            library="lobster.services.quality.proteomics_quality_service",
            code_template="""# Technical replicate assessment
from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService

service = ProteomicsQualityService()
adata_qc, stats, _ = service.assess_technical_replicates(
    adata,
    replicate_column={{ replicate_column | tojson }},
    correlation_method={{ correlation_method | tojson }}
)
print(f"Replicate correlation: {stats['median_replicate_correlation']:.3f}, CV: {stats['median_replicate_cv']:.3f}")""",
            imports=[
                "from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService"
            ],
            parameters={
                "replicate_column": replicate_column,
                "correlation_method": correlation_method,
            },
            parameter_schema={
                "replicate_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="replicate_id",
                    required=True,
                    description="Column in obs identifying technical replicates",
                ),
                "correlation_method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="pearson",
                    required=False,
                    validation_rule="correlation_method in ['pearson', 'spearman']",
                    description="Method for correlation analysis",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_qc"],
        )

    def assess_missing_value_patterns(
        self,
        adata: anndata.AnnData,
        sample_threshold: float = 0.7,
        protein_threshold: float = 0.8,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Analyze missing value patterns in proteomics data with MNAR/MCAR detection.

        Args:
            adata: AnnData object with proteomics data
            sample_threshold: Threshold for high missing value samples
            protein_threshold: Threshold for high missing value proteins

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with QC metrics,
                analysis stats, and IR for notebook export

        Raises:
            ProteomicsQualityError: If assessment fails
        """
        try:
            logger.info("Starting missing value pattern analysis")

            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            X = adata_qc.X.copy()

            # Calculate missing value statistics
            is_missing = np.isnan(X)
            total_missing = is_missing.sum()
            total_values = X.size

            # Sample-level missing value analysis
            sample_missing_counts = is_missing.sum(axis=1)
            sample_missing_rates = sample_missing_counts / adata_qc.n_vars

            # Protein-level missing value analysis
            protein_missing_counts = is_missing.sum(axis=0)
            protein_missing_rates = protein_missing_counts / adata_qc.n_obs

            # Add QC metrics to observations (samples)
            adata_qc.obs["missing_value_percentage"] = sample_missing_rates * 100
            adata_qc.obs["missing_protein_count"] = sample_missing_counts
            adata_qc.obs["missing_protein_rate"] = sample_missing_rates
            adata_qc.obs["high_missing_sample"] = (
                sample_missing_rates > sample_threshold
            )
            adata_qc.obs["detected_protein_count"] = (
                adata_qc.n_vars - sample_missing_counts
            )

            # Add QC metrics to variables (proteins)
            adata_qc.var["missing_value_percentage"] = protein_missing_rates * 100
            adata_qc.var["missing_sample_count"] = protein_missing_counts
            adata_qc.var["missing_sample_rate"] = protein_missing_rates
            adata_qc.var["high_missing_protein"] = (
                protein_missing_rates > protein_threshold
            )
            adata_qc.var["detected_sample_count"] = (
                adata_qc.n_obs - protein_missing_counts
            )

            # Perform MNAR/MCAR detection
            mnar_mcar_analysis = self._detect_mnar_mcar_patterns(X, is_missing)
            adata_qc.var["missing_pattern"] = mnar_mcar_analysis["protein_patterns"]

            # Identify missing value patterns
            missing_patterns = self._identify_missing_patterns(is_missing)

            # Calculate missing value statistics
            missing_stats = {
                "total_missing_values": int(total_missing),
                "total_possible_values": int(total_values),
                "total_missing_percentage": float((total_missing / total_values) * 100),
                "overall_missing_rate": float(total_missing / total_values),
                "n_high_missing_samples": int(
                    (sample_missing_rates > sample_threshold).sum()
                ),
                "high_missing_samples": int(
                    (sample_missing_rates > sample_threshold).sum()
                ),
                "n_high_missing_proteins": int(
                    (protein_missing_rates > protein_threshold).sum()
                ),
                "high_missing_proteins": int(
                    (protein_missing_rates > protein_threshold).sum()
                ),
                "median_missing_rate_samples": float(np.median(sample_missing_rates)),
                "median_missing_rate_proteins": float(np.median(protein_missing_rates)),
                "sample_threshold": sample_threshold,
                "protein_threshold": protein_threshold,
                "missing_value_patterns": missing_patterns,
                "mnar_proteins": int(mnar_mcar_analysis["n_mnar"]),
                "mcar_proteins": int(mnar_mcar_analysis["n_mcar"]),
                "samples_processed": adata_qc.n_obs,
                "proteins_processed": adata_qc.n_vars,
                "analysis_type": "missing_value_assessment",
            }

            logger.info(
                f"Missing value analysis completed: {total_missing:,} missing values ({(total_missing/total_values)*100:.1f}%)"
            )
            logger.info(
                f"MNAR/MCAR: {mnar_mcar_analysis['n_mnar']} MNAR, {mnar_mcar_analysis['n_mcar']} MCAR proteins"
            )

            # Create IR for provenance tracking
            ir = self._create_ir_missing_value_patterns(sample_threshold, protein_threshold)
            return adata_qc, missing_stats, ir

        except Exception as e:
            logger.exception(f"Error in missing value pattern analysis: {e}")
            raise ProteomicsQualityError(
                f"Missing value pattern analysis failed: {str(e)}"
            )

    def assess_coefficient_variation(
        self,
        adata: anndata.AnnData,
        replicate_column: Optional[str] = None,
        cv_threshold: float = 0.2,
        min_observations: int = 3,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Assess coefficient of variation (CV) for proteins with optional replicate grouping.

        CV is calculated as std/mean and returned as fractional values (0.0 to 1.0),
        where 0.2 represents 20% variation.

        Args:
            adata: AnnData object with proteomics data
            replicate_column: Column in obs containing replicate groups (None for overall CV)
            cv_threshold: Threshold for high CV proteins (fractional, e.g., 0.2 = 20%)
            min_observations: Minimum observations required for CV calculation

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with CV metrics
                (fractional units), analysis stats, and IR for notebook export

        Raises:
            ProteomicsQualityError: If assessment fails
        """
        try:
            logger.info("Starting coefficient of variation assessment")

            # Validate replicate column if provided
            if (
                replicate_column is not None
                and replicate_column not in adata.obs.columns
            ):
                raise ProteomicsQualityError(
                    f"Replicate column '{replicate_column}' not found in obs"
                )

            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            X = adata_qc.X.copy()

            # Calculate protein-level CVs
            if replicate_column is not None:
                # Group-wise CV calculation
                replicate_groups = adata_qc.obs[replicate_column]
                unique_groups = replicate_groups.unique()

                protein_cv_means = []
                protein_cv_medians = []
                protein_cv_list_per_protein = []

                for i in range(adata_qc.n_vars):
                    group_cvs = []
                    for group in unique_groups:
                        group_mask = replicate_groups == group
                        protein_values = X[group_mask, i]

                        # Remove missing values
                        if hasattr(protein_values, "isnan"):
                            valid_values = protein_values[~np.isnan(protein_values)]
                        else:
                            valid_values = protein_values[protein_values > 0]

                        if len(valid_values) >= min_observations:
                            mean_val = np.mean(valid_values)
                            std_val = np.std(valid_values, ddof=1)
                            cv_val = (std_val / mean_val) if mean_val > 0 else np.inf
                            group_cvs.append(cv_val)

                    if len(group_cvs) > 0:
                        valid_cvs = [
                            cv
                            for cv in group_cvs
                            if not np.isnan(cv) and not np.isinf(cv)
                        ]
                        if valid_cvs:
                            protein_cv_means.append(np.mean(valid_cvs))
                            protein_cv_medians.append(np.median(valid_cvs))
                            protein_cv_list_per_protein.append(valid_cvs)
                        else:
                            protein_cv_means.append(np.nan)
                            protein_cv_medians.append(np.nan)
                            protein_cv_list_per_protein.append([])
                    else:
                        protein_cv_means.append(np.nan)
                        protein_cv_medians.append(np.nan)
                        protein_cv_list_per_protein.append([])

                # Add CV metrics to variables (proteins)
                adata_qc.var["cv_mean"] = protein_cv_means
                adata_qc.var["cv_median"] = protein_cv_medians
                adata_qc.var["high_cv_protein"] = (
                    np.array(protein_cv_means) > cv_threshold
                )

                # Calculate overall CV statistics
                valid_protein_cvs = [
                    cv
                    for cv in protein_cv_means
                    if not np.isnan(cv) and not np.isinf(cv)
                ]

                cv_stats = {
                    "mean_cv_across_proteins": (
                        float(np.mean(valid_protein_cvs))
                        if valid_protein_cvs
                        else np.nan
                    ),
                    "median_cv_across_proteins": (
                        float(np.median(valid_protein_cvs))
                        if valid_protein_cvs
                        else np.nan
                    ),
                    "n_high_cv_proteins": int(
                        np.sum(np.array(protein_cv_means) > cv_threshold)
                    ),
                    "cv_threshold": cv_threshold,
                    "min_observations": min_observations,
                    "replicate_column": replicate_column,
                    "samples_processed": adata_qc.n_obs,
                    "proteins_processed": adata_qc.n_vars,
                    "analysis_type": "coefficient_variation_assessment",
                }
            else:
                # Overall CV calculation (no replicate grouping)
                protein_cvs = []
                for i in range(adata_qc.n_vars):
                    protein_values = X[:, i]

                    # Remove missing values
                    if hasattr(protein_values, "isnan"):
                        valid_values = protein_values[~np.isnan(protein_values)]
                    else:
                        valid_values = protein_values[protein_values > 0]

                    if len(valid_values) >= min_observations:
                        mean_val = np.mean(valid_values)
                        std_val = np.std(valid_values, ddof=1)
                        cv_val = (std_val / mean_val) if mean_val > 0 else np.inf
                    else:
                        cv_val = np.nan

                    protein_cvs.append(cv_val)

                # Add CV metrics to variables
                adata_qc.var["cv_overall"] = protein_cvs
                adata_qc.var["cv_mean"] = protein_cvs  # Alias for consistency
                adata_qc.var["cv_median"] = protein_cvs  # Same as mean for overall
                adata_qc.var["high_cv_protein"] = np.array(protein_cvs) > cv_threshold

                valid_protein_cvs = [
                    cv for cv in protein_cvs if not np.isnan(cv) and not np.isinf(cv)
                ]

                cv_stats = {
                    "mean_cv_across_proteins": (
                        float(np.mean(valid_protein_cvs))
                        if valid_protein_cvs
                        else np.nan
                    ),
                    "median_cv_across_proteins": (
                        float(np.median(valid_protein_cvs))
                        if valid_protein_cvs
                        else np.nan
                    ),
                    "n_high_cv_proteins": int(
                        np.sum(np.array(protein_cvs) > cv_threshold)
                    ),
                    "cv_threshold": cv_threshold,
                    "min_observations": min_observations,
                    "samples_processed": adata_qc.n_obs,
                    "proteins_processed": adata_qc.n_vars,
                    "analysis_type": "coefficient_variation_assessment",
                }

            logger.info(
                f"CV assessment completed: mean protein CV = {cv_stats['mean_cv_across_proteins']:.3f}"
            )

            # Create IR for provenance tracking
            ir = self._create_ir_coefficient_variation(
                replicate_column, cv_threshold, min_observations
            )
            return adata_qc, cv_stats, ir

        except Exception as e:
            logger.exception(f"Error in coefficient of variation assessment: {e}")
            raise ProteomicsQualityError(
                f"Coefficient of variation assessment failed: {str(e)}"
            )

    def detect_contaminants(
        self,
        adata: anndata.AnnData,
        protein_id_column: str = None,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Detect contaminant proteins based on naming patterns.

        Args:
            adata: AnnData object with proteomics data
            protein_id_column: Column in var containing protein IDs (uses index if None)
            custom_patterns: Custom contaminant patterns to add to defaults

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with contaminant flags,
                analysis stats, and IR for notebook export

        Raises:
            ProteomicsQualityError: If detection fails
        """
        try:
            logger.info("Starting contaminant detection")

            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            # Get protein identifiers
            if protein_id_column and protein_id_column in adata_qc.var.columns:
                protein_ids = adata_qc.var[protein_id_column].astype(str)
            else:
                protein_ids = adata_qc.var_names.astype(str)

            # Combine default and custom patterns
            contaminant_patterns = self.contaminant_patterns.copy()
            if custom_patterns:
                contaminant_patterns.update(custom_patterns)

            # Initialize contaminant flags
            contaminant_flags = {}
            for contaminant_type in contaminant_patterns.keys():
                contaminant_flags[f"is_{contaminant_type}"] = np.zeros(
                    adata_qc.n_vars, dtype=bool
                )

            # Check each protein against contaminant patterns
            contaminant_counts = {}
            for contaminant_type, patterns in contaminant_patterns.items():
                count = 0
                for i, protein_id in enumerate(protein_ids):
                    protein_id_upper = protein_id.upper()
                    for pattern in patterns:
                        if pattern.upper() in protein_id_upper:
                            contaminant_flags[f"is_{contaminant_type}"][i] = True
                            count += 1
                            break
                contaminant_counts[contaminant_type] = count

            # Add contaminant flags to var
            for flag_name, flag_values in contaminant_flags.items():
                adata_qc.var[flag_name] = flag_values

            # Create overall contaminant flag
            overall_contaminant = np.zeros(adata_qc.n_vars, dtype=bool)
            for flag_values in contaminant_flags.values():
                overall_contaminant |= flag_values
            adata_qc.var["is_contaminant"] = overall_contaminant

            # Calculate contaminant statistics
            total_contaminants = overall_contaminant.sum()
            contaminant_percentage = (total_contaminants / adata_qc.n_vars) * 100

            contaminant_stats = {
                "total_contaminants": int(total_contaminants),
                "contaminant_percentage": float(contaminant_percentage),
                "contaminant_counts_by_type": contaminant_counts,
                "patterns_used": {k: len(v) for k, v in contaminant_patterns.items()},
                "samples_processed": adata_qc.n_obs,
                "proteins_processed": adata_qc.n_vars,
                "analysis_type": "contaminant_detection",
            }

            logger.info(
                f"Contaminant detection completed: {total_contaminants} contaminants ({contaminant_percentage:.1f}%)"
            )

            # Create IR for provenance tracking
            ir = self._create_ir_detect_contaminants(protein_id_column, custom_patterns)
            return adata_qc, contaminant_stats, ir

        except Exception as e:
            logger.exception(f"Error in contaminant detection: {e}")
            raise ProteomicsQualityError(f"Contaminant detection failed: {str(e)}")

    def evaluate_dynamic_range(
        self,
        adata: anndata.AnnData,
        percentile_low: float = 5.0,
        percentile_high: float = 95.0,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Evaluate dynamic range of proteomics measurements.

        Args:
            adata: AnnData object with proteomics data
            percentile_low: Lower percentile for dynamic range calculation
            percentile_high: Higher percentile for dynamic range calculation

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with dynamic range metrics,
                analysis stats, and IR for notebook export

        Raises:
            ProteomicsQualityError: If evaluation fails
        """
        try:
            logger.info("Starting dynamic range evaluation")

            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            X = adata_qc.X.copy()

            # Calculate dynamic range metrics per sample
            sample_dynamic_ranges = []
            sample_intensity_ranges = []
            sample_percentiles_low = []
            sample_percentiles_high = []

            for i in range(adata_qc.n_obs):
                sample_values = X[i, :]

                # Remove missing/zero values
                if hasattr(sample_values, "isnan"):
                    valid_values = sample_values[
                        ~np.isnan(sample_values) & (sample_values > 0)
                    ]
                else:
                    valid_values = sample_values[sample_values > 0]

                if len(valid_values) > 0:
                    p_low = np.percentile(valid_values, percentile_low)
                    p_high = np.percentile(valid_values, percentile_high)
                    dynamic_range = np.log10(p_high / p_low) if p_low > 0 else np.nan
                    intensity_range = p_high - p_low
                else:
                    p_low = np.nan
                    p_high = np.nan
                    dynamic_range = np.nan
                    intensity_range = np.nan

                sample_dynamic_ranges.append(dynamic_range)
                sample_intensity_ranges.append(intensity_range)
                sample_percentiles_low.append(p_low)
                sample_percentiles_high.append(p_high)

            # Calculate dynamic range metrics per protein
            protein_dynamic_ranges = []
            protein_intensity_ranges = []
            protein_percentiles_low = []
            protein_percentiles_high = []

            for j in range(adata_qc.n_vars):
                protein_values = X[:, j]

                # Remove missing/zero values
                if hasattr(protein_values, "isnan"):
                    valid_values = protein_values[
                        ~np.isnan(protein_values) & (protein_values > 0)
                    ]
                else:
                    valid_values = protein_values[protein_values > 0]

                if len(valid_values) > 0:
                    p_low = np.percentile(valid_values, percentile_low)
                    p_high = np.percentile(valid_values, percentile_high)
                    dynamic_range = np.log10(p_high / p_low) if p_low > 0 else np.nan
                    intensity_range = p_high - p_low
                else:
                    p_low = np.nan
                    p_high = np.nan
                    dynamic_range = np.nan
                    intensity_range = np.nan

                protein_dynamic_ranges.append(dynamic_range)
                protein_intensity_ranges.append(intensity_range)
                protein_percentiles_low.append(p_low)
                protein_percentiles_high.append(p_high)

            # Add dynamic range metrics to observations (samples)
            adata_qc.obs["dynamic_range_log10"] = sample_dynamic_ranges
            adata_qc.obs["intensity_range"] = sample_intensity_ranges
            adata_qc.obs[f"percentile_{percentile_low}"] = sample_percentiles_low
            adata_qc.obs[f"percentile_{percentile_high}"] = sample_percentiles_high

            # Add dynamic range metrics to variables (proteins)
            adata_qc.var["dynamic_range_log10"] = protein_dynamic_ranges
            adata_qc.var["intensity_range"] = protein_intensity_ranges
            adata_qc.var[f"percentile_{percentile_low}"] = protein_percentiles_low
            adata_qc.var[f"percentile_{percentile_high}"] = protein_percentiles_high

            # Calculate overall dynamic range statistics
            valid_sample_ranges = [
                dr for dr in sample_dynamic_ranges if not np.isnan(dr)
            ]
            valid_protein_ranges = [
                dr for dr in protein_dynamic_ranges if not np.isnan(dr)
            ]

            dynamic_range_stats = {
                "median_sample_dynamic_range": (
                    float(np.median(valid_sample_ranges))
                    if valid_sample_ranges
                    else np.nan
                ),
                "mean_sample_dynamic_range": (
                    float(np.mean(valid_sample_ranges))
                    if valid_sample_ranges
                    else np.nan
                ),
                "median_protein_dynamic_range": (
                    float(np.median(valid_protein_ranges))
                    if valid_protein_ranges
                    else np.nan
                ),
                "mean_protein_dynamic_range": (
                    float(np.mean(valid_protein_ranges))
                    if valid_protein_ranges
                    else np.nan
                ),
                "percentile_low": percentile_low,
                "percentile_high": percentile_high,
                "samples_processed": adata_qc.n_obs,
                "proteins_processed": adata_qc.n_vars,
                "analysis_type": "dynamic_range_evaluation",
            }

            logger.info(
                f"Dynamic range evaluation completed: median sample range = {dynamic_range_stats['median_sample_dynamic_range']:.2f} log10"
            )

            # Create IR for provenance tracking
            ir = self._create_ir_evaluate_dynamic_range(percentile_low, percentile_high)
            return adata_qc, dynamic_range_stats, ir

        except Exception as e:
            logger.exception(f"Error in dynamic range evaluation: {e}")
            raise ProteomicsQualityError(f"Dynamic range evaluation failed: {str(e)}")

    def detect_pca_outliers(
        self,
        adata: anndata.AnnData,
        n_components: int = 50,
        outlier_threshold: float = 3.0,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Detect outlier samples using PCA analysis.

        Args:
            adata: AnnData object with proteomics data
            n_components: Number of PCA components to compute
            outlier_threshold: Threshold in standard deviations for outlier detection

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with outlier flags,
                analysis stats, and IR for notebook export

        Raises:
            ProteomicsQualityError: If detection fails
        """
        try:
            logger.info("Starting PCA outlier detection")

            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            X = adata_qc.X.copy()

            # Prepare data for PCA (handle missing values)
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X)

            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)

            # Perform PCA
            n_components_actual = min(n_components, min(X_scaled.shape) - 1)
            pca = PCA(n_components=n_components_actual)
            X_pca = pca.fit_transform(X_scaled)

            # Store PCA results
            adata_qc.obsm["X_pca"] = X_pca
            adata_qc.uns["pca"] = {
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "explained_variance": pca.explained_variance_,
                "components": pca.components_,
            }

            # Detect outliers using Mahalanobis distance approximation
            # Calculate distance from center for each sample
            pc_distances = []
            for i in range(adata_qc.n_obs):
                # Calculate distance in PC space (first few components)
                pc_coords = X_pca[i, : min(10, n_components_actual)]  # Use first 10 PCs
                distance = np.sqrt(np.sum(pc_coords**2))
                pc_distances.append(distance)

            pc_distances = np.array(pc_distances)

            # Identify outliers based on distance threshold
            mean_distance = np.mean(pc_distances)
            std_distance = np.std(pc_distances)
            outlier_cutoff = mean_distance + outlier_threshold * std_distance

            is_outlier = pc_distances > outlier_cutoff

            # Add outlier information to observations
            adata_qc.obs["pca_distance"] = pc_distances
            adata_qc.obs["is_pca_outlier"] = is_outlier
            adata_qc.obs["outlier_score"] = (
                pc_distances - mean_distance
            ) / std_distance

            # Calculate PCA statistics
            variance_explained_top10 = pca.explained_variance_ratio_[:10].sum()
            n_outliers = is_outlier.sum()

            pca_stats = {
                "n_components_computed": n_components_actual,
                "variance_explained_top10": float(variance_explained_top10),
                "n_outliers_detected": int(n_outliers),
                "outlier_percentage": float((n_outliers / adata_qc.n_obs) * 100),
                "outlier_threshold": outlier_threshold,
                "mean_pca_distance": float(mean_distance),
                "std_pca_distance": float(std_distance),
                "samples_processed": adata_qc.n_obs,
                "proteins_processed": adata_qc.n_vars,
                "analysis_type": "pca_outlier_detection",
            }

            logger.info(
                f"PCA outlier detection completed: {n_outliers} outliers ({(n_outliers/adata_qc.n_obs)*100:.1f}%)"
            )

            # Create IR for provenance tracking
            ir = self._create_ir_detect_pca_outliers(n_components, outlier_threshold)
            return adata_qc, pca_stats, ir

        except Exception as e:
            logger.exception(f"Error in PCA outlier detection: {e}")
            raise ProteomicsQualityError(f"PCA outlier detection failed: {str(e)}")

    def assess_technical_replicates(
        self,
        adata: anndata.AnnData,
        replicate_column: str,
        correlation_method: str = "pearson",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Assess technical replicate reproducibility and variation.

        CV is calculated as std/mean in fractional units (0.0 to 1.0), consistent with
        assess_coefficient_variation(). A CV of 0.2 represents 20% variation.

        Args:
            adata: AnnData object with proteomics data
            replicate_column: Column in obs identifying technical replicates
            correlation_method: Method for correlation analysis ('pearson', 'spearman')

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with replicate metrics
                (CV in fractional units), analysis stats, and IR for notebook export

        Raises:
            ProteomicsQualityError: If assessment fails
        """
        try:
            logger.info("Starting technical replicate assessment")

            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            if replicate_column not in adata_qc.obs.columns:
                raise ProteomicsQualityError(
                    f"Replicate column '{replicate_column}' not found in obs"
                )

            X = adata_qc.X.copy()
            replicate_groups = adata_qc.obs[replicate_column]

            # Find replicate groups
            unique_groups = replicate_groups.unique()
            replicate_correlations = []
            replicate_cvs = []
            group_sizes = []

            for group in unique_groups:
                group_mask = replicate_groups == group
                group_samples = X[group_mask, :]
                group_size = group_samples.shape[0]
                group_sizes.append(group_size)

                if group_size < 2:
                    continue  # Need at least 2 replicates for correlation

                # Calculate pairwise correlations within group
                group_correlations = []
                for i in range(group_size):
                    for j in range(i + 1, group_size):
                        sample1 = group_samples[i, :]
                        sample2 = group_samples[j, :]

                        # Remove missing values for correlation
                        if hasattr(sample1, "isnan"):
                            valid_mask = ~(np.isnan(sample1) | np.isnan(sample2))
                        else:
                            valid_mask = (sample1 > 0) & (sample2 > 0)

                        if (
                            valid_mask.sum() > 10
                        ):  # Need at least 10 points for correlation
                            if correlation_method == "pearson":
                                corr, _ = stats.pearsonr(
                                    sample1[valid_mask], sample2[valid_mask]
                                )
                            else:  # spearman
                                corr, _ = stats.spearmanr(
                                    sample1[valid_mask], sample2[valid_mask]
                                )
                            group_correlations.append(corr)

                replicate_correlations.extend(group_correlations)

                # Calculate CV within replicate group for each protein
                group_cvs_per_protein = []
                for j in range(adata_qc.n_vars):
                    protein_values = group_samples[:, j]
                    if hasattr(protein_values, "isnan"):
                        valid_values = protein_values[~np.isnan(protein_values)]
                    else:
                        valid_values = protein_values[protein_values > 0]

                    if len(valid_values) >= 2:
                        mean_val = np.mean(valid_values)
                        std_val = np.std(valid_values)
                        cv_val = (std_val / mean_val) if mean_val > 0 else np.nan
                        group_cvs_per_protein.append(cv_val)

                replicate_cvs.extend(group_cvs_per_protein)

            # Calculate replicate statistics
            valid_correlations = [
                corr for corr in replicate_correlations if not np.isnan(corr)
            ]
            valid_cvs = [cv for cv in replicate_cvs if not np.isnan(cv)]

            replicate_stats = {
                "n_replicate_groups": len(unique_groups),
                "group_sizes": group_sizes,
                "median_replicate_correlation": (
                    float(np.median(valid_correlations))
                    if valid_correlations
                    else np.nan
                ),
                "mean_replicate_correlation": (
                    float(np.mean(valid_correlations)) if valid_correlations else np.nan
                ),
                "median_replicate_cv": (
                    float(np.median(valid_cvs)) if valid_cvs else np.nan
                ),
                "mean_replicate_cv": float(np.mean(valid_cvs)) if valid_cvs else np.nan,
                "correlation_method": correlation_method,
                "samples_processed": adata_qc.n_obs,
                "proteins_processed": adata_qc.n_vars,
                "analysis_type": "technical_replicate_assessment",
            }

            # Add replicate quality flags
            adata_qc.obs["replicate_group"] = replicate_groups
            adata_qc.obs["group_size"] = adata_qc.obs[replicate_column].map(
                replicate_groups.value_counts().to_dict()
            )

            logger.info(
                f"Technical replicate assessment completed: {len(unique_groups)} replicate groups"
            )

            # Create IR for provenance tracking
            ir = self._create_ir_assess_technical_replicates(replicate_column, correlation_method)
            return adata_qc, replicate_stats, ir

        except Exception as e:
            logger.exception(f"Error in technical replicate assessment: {e}")
            raise ProteomicsQualityError(
                f"Technical replicate assessment failed: {str(e)}"
            )

    # Helper methods
    def _detect_mnar_mcar_patterns(
        self, X: np.ndarray, is_missing: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect MNAR (Missing Not at Random) vs MCAR (Missing Completely at Random) patterns.

        MNAR detection is based on intensity: low-abundance proteins tend to have more missing values.
        MCAR detection is based on random distribution of missing values.

        Args:
            X: Data matrix with values
            is_missing: Boolean matrix indicating missing values

        Returns:
            Dictionary with pattern classification
        """
        n_samples, n_proteins = X.shape
        protein_patterns = []

        for protein_idx in range(n_proteins):
            protein_values = X[:, protein_idx]
            protein_missing = is_missing[:, protein_idx]

            # Calculate missing rate
            missing_rate = protein_missing.sum() / n_samples

            if missing_rate < 0.1:
                # Low missing rate - assume MCAR
                pattern = "MCAR"
            elif missing_rate > 0.7:
                # Very high missing rate - likely MNAR (low abundance)
                pattern = "MNAR"
            else:
                # Intermediate - test correlation with intensity
                # Get valid (non-missing) values
                valid_mask = ~protein_missing
                if valid_mask.sum() >= 3:
                    valid_values = protein_values[valid_mask]
                    # Calculate median intensity of valid values
                    np.median(valid_values)
                    mean_intensity = np.mean(valid_values)

                    # MNAR proteins typically have lower mean/median intensities
                    # and missing values correlate with low abundance
                    # Simple heuristic: if this protein is in bottom 30% of intensities, likely MNAR
                    all_valid_values = X[~is_missing]
                    if len(all_valid_values) > 0:
                        percentile_30 = np.percentile(all_valid_values, 30)
                        if mean_intensity < percentile_30:
                            pattern = "MNAR"
                        else:
                            pattern = "MCAR"
                    else:
                        pattern = "Unknown"
                else:
                    pattern = "Unknown"

            protein_patterns.append(pattern)

        # Count patterns
        n_mnar = sum(1 for p in protein_patterns if p == "MNAR")
        n_mcar = sum(1 for p in protein_patterns if p == "MCAR")
        n_unknown = sum(1 for p in protein_patterns if p == "Unknown")

        return {
            "protein_patterns": protein_patterns,
            "n_mnar": n_mnar,
            "n_mcar": n_mcar,
            "n_unknown": n_unknown,
        }

    def _identify_missing_patterns(self, is_missing: np.ndarray) -> Dict[str, Any]:
        """Identify common missing value patterns."""
        n_samples, n_proteins = is_missing.shape

        # Pattern 1: Completely missing proteins
        completely_missing_proteins = np.sum(is_missing, axis=0) == n_samples

        # Pattern 2: Completely missing samples
        completely_missing_samples = np.sum(is_missing, axis=1) == n_proteins

        # Pattern 3: Sample-wise patterns (samples with high missing rates)
        sample_missing_rates = is_missing.sum(axis=1) / n_proteins
        high_sample_missing = (sample_missing_rates > 0.5).sum()

        # Pattern 4: Protein-wise patterns (proteins with high missing rates)
        protein_missing_rates = is_missing.sum(axis=0) / n_samples
        high_protein_missing = (protein_missing_rates > 0.5).sum()

        # Pattern 5: Block missing patterns (consecutive missing)
        missing_blocks = 0
        for i in range(n_samples):
            missing_runs = []
            current_run = 0
            for j in range(n_proteins):
                if is_missing[i, j]:
                    current_run += 1
                else:
                    if current_run > 0:
                        missing_runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                missing_runs.append(current_run)

            # Count blocks of 10+ consecutive missing values
            missing_blocks += sum(1 for run in missing_runs if run >= 10)

        return {
            "sample_wise": int(high_sample_missing),
            "protein_wise": int(high_protein_missing),
            "total_patterns": int(high_sample_missing + high_protein_missing),
            "completely_missing_proteins": int(completely_missing_proteins.sum()),
            "completely_missing_samples": int(completely_missing_samples.sum()),
            "estimated_missing_blocks": missing_blocks,
            "random_missing_pattern": missing_blocks < n_samples * 0.1,
        }
