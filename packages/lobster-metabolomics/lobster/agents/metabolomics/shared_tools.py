"""
Shared tools for metabolomics analysis (LC-MS, GC-MS, NMR platforms).

This module provides 10 tools for the metabolomics expert agent, each wrapping
a stateless service method. Tools auto-detect platform type and use appropriate
defaults.

Following the same factory pattern as proteomics shared_tools.py.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from langchain_core.tools import tool

from lobster.agents.metabolomics.config import (
    MetabPlatformConfig,
    detect_platform_type,
    get_platform_config,
)
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.metabolomics_analysis_service import (
    MetabolomicsAnalysisService,
)
from lobster.services.annotation.metabolomics_annotation_service import (
    MetabolomicsAnnotationService,
)
from lobster.services.quality.metabolomics_preprocessing_service import (
    MetabolomicsPreprocessingService,
)
from lobster.services.quality.metabolomics_quality_service import (
    MetabolomicsQualityService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SHARED TOOL FACTORY
# =============================================================================


def create_shared_tools(
    data_manager: DataManagerV2,
    quality_service: MetabolomicsQualityService,
    preprocessing_service: MetabolomicsPreprocessingService,
    analysis_service: MetabolomicsAnalysisService,
    annotation_service: MetabolomicsAnnotationService,
    force_platform_type: Optional[str] = None,
) -> List[Callable]:
    """
    Create shared metabolomics tools with platform auto-detection.

    These tools are used by the metabolomics expert agent. Each tool auto-detects
    the platform type (LC-MS, GC-MS, NMR) and applies appropriate defaults.

    Args:
        data_manager: DataManagerV2 instance for modality management
        quality_service: MetabolomicsQualityService for QC operations
        preprocessing_service: MetabolomicsPreprocessingService for filtering/imputation/normalization
        analysis_service: MetabolomicsAnalysisService for statistics and multivariate analysis
        annotation_service: MetabolomicsAnnotationService for m/z annotation and lipid classification
        force_platform_type: Override auto-detection ("lc_ms", "gc_ms", or "nmr")

    Returns:
        List of 10 tool functions to be added to agent tools
    """
    _forced_platform_type = force_platform_type

    def _get_platform_for_modality(modality_name: str) -> MetabPlatformConfig:
        """Get platform config for a modality, using detection or forced type."""
        if _forced_platform_type:
            return get_platform_config(_forced_platform_type)

        try:
            adata = data_manager.get_modality(modality_name)
            detected_type = detect_platform_type(adata)
            return get_platform_config(detected_type)
        except ValueError:
            # Default to lc_ms if modality not found
            return get_platform_config("lc_ms")

    # =========================================================================
    # Tool 1: Quality Assessment
    # =========================================================================

    @tool
    def assess_metabolomics_quality(
        modality_name: str,
        qc_label: str = "QC",
        rsd_threshold: float = 30.0,
    ) -> str:
        """
        Run comprehensive quality assessment on metabolomics data.

        Computes per-feature RSD, TIC distribution, QC sample reproducibility,
        and missing value profiling. Flags high-variability features.

        Args:
            modality_name: Name of the metabolomics modality to assess
            qc_label: Label for QC/pooled samples in sample_type column
            rsd_threshold: RSD threshold (%) for flagging high-variability features

        Returns:
            str: Quality assessment report
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            adata_qc, stats, ir = quality_service.assess_quality(
                adata, qc_label=qc_label, rsd_threshold=rsd_threshold
            )

            result_name = f"{modality_name}_quality_assessed"
            data_manager.store_modality(
                name=result_name,
                adata=adata_qc,
                parent_name=modality_name,
                step_summary=f"Quality assessed: median RSD={stats['median_rsd']:.1f}%, missing={stats['missing_pct']:.1f}%",
            )

            data_manager.log_tool_usage(
                tool_name="assess_metabolomics_quality",
                parameters={
                    "modality_name": modality_name,
                    "qc_label": qc_label,
                    "rsd_threshold": rsd_threshold,
                },
                description="Assessed metabolomics data quality",
                ir=ir,
            )

            response = f"Quality assessment complete for '{modality_name}'.\n\n"
            response += "**Quality Metrics:**\n"
            response += f"- Samples: {stats['n_samples']}\n"
            response += f"- Features: {stats['n_features']}\n"
            response += f"- Median RSD: {stats['median_rsd']:.1f}%\n"
            response += f"- High RSD features (>{rsd_threshold}%): {stats['high_rsd_features']}\n"
            response += f"- Missing values: {stats['missing_pct']:.1f}%\n"
            response += f"- TIC CV: {stats['tic_cv']:.1f}%\n"

            if stats.get("qc_stats"):
                qc = stats["qc_stats"]
                response += f"\n**QC Sample Evaluation:**\n"
                response += f"- QC samples: {qc.get('n_qc_samples', 0)}\n"
                response += f"- Median QC RSD: {qc.get('median_qc_rsd', 0):.1f}%\n"
                response += f"- Features below threshold: {qc.get('features_below_threshold', 0)}\n"

            response += f"\n**New modality created**: '{result_name}'"
            return response

        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return f"Error assessing quality: {str(e)}"

    # =========================================================================
    # Tool 2: Feature Filtering
    # =========================================================================

    @tool
    def filter_metabolomics_features(
        modality_name: str,
        min_prevalence: float = 0.5,
        max_rsd: float = None,
        blank_ratio_threshold: float = None,
    ) -> str:
        """
        Filter metabolomics features by prevalence, RSD, and blank ratio.

        Removes features with too many missing values, excessive variability,
        or high blank-to-sample ratios (blank subtraction).

        Args:
            modality_name: Name of the metabolomics modality to filter
            min_prevalence: Minimum fraction of samples with non-NaN values (0-1)
            max_rsd: Maximum RSD threshold; requires prior quality assessment
            blank_ratio_threshold: Maximum blank/sample intensity ratio

        Returns:
            str: Filtering results with counts per criterion
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            adata_filtered, stats, ir = preprocessing_service.filter_features(
                adata,
                min_prevalence=min_prevalence,
                max_rsd=max_rsd,
                blank_ratio_threshold=blank_ratio_threshold,
            )

            result_name = f"{modality_name}_filtered"
            data_manager.store_modality(
                name=result_name,
                adata=adata_filtered,
                parent_name=modality_name,
                step_summary=f"Filtered: {stats['n_before']} -> {stats['n_after']} features",
            )

            data_manager.log_tool_usage(
                tool_name="filter_metabolomics_features",
                parameters={
                    "modality_name": modality_name,
                    "min_prevalence": min_prevalence,
                    "max_rsd": max_rsd,
                    "blank_ratio_threshold": blank_ratio_threshold,
                },
                description="Filtered metabolomics features",
                ir=ir,
            )

            response = f"Feature filtering complete for '{modality_name}'.\n\n"
            response += "**Filtering Results:**\n"
            response += f"- Features before: {stats['n_before']}\n"
            response += f"- Features after: {stats['n_after']}\n"
            response += f"- Removed (prevalence < {min_prevalence}): {stats['n_removed_prevalence']}\n"
            response += f"- Removed (high RSD): {stats['n_removed_rsd']}\n"
            response += f"- Removed (blank ratio): {stats['n_removed_blank']}\n"
            response += f"- Total removed: {stats['n_removed_total']}\n"
            response += f"\n**New modality created**: '{result_name}'"
            return response

        except Exception as e:
            logger.error(f"Error in feature filtering: {e}")
            return f"Error filtering features: {str(e)}"

    # =========================================================================
    # Tool 3: Missing Value Imputation
    # =========================================================================

    @tool
    def handle_missing_values(
        modality_name: str,
        method: str = "knn",
        knn_neighbors: int = 5,
    ) -> str:
        """
        Impute missing values in metabolomics data.

        Supports KNN, minimum/2, LOD/2, median, and MICE imputation methods.
        Stores pre-imputation data in a layer for comparison.

        Args:
            modality_name: Name of the metabolomics modality
            method: Imputation method: "knn", "min", "lod_half", "median", or "mice"
            knn_neighbors: Number of neighbors for KNN imputation

        Returns:
            str: Imputation results
        """
        valid_methods = ["knn", "min", "lod_half", "median", "mice"]
        if method not in valid_methods:
            return f"Invalid method '{method}'. Choose from: {valid_methods}"

        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            adata_imputed, stats, ir = preprocessing_service.impute_missing_values(
                adata, method=method, knn_neighbors=knn_neighbors
            )

            result_name = f"{modality_name}_imputed"
            data_manager.store_modality(
                name=result_name,
                adata=adata_imputed,
                parent_name=modality_name,
                step_summary=f"Imputed {stats['n_values_imputed']} values using {method}",
            )

            data_manager.log_tool_usage(
                tool_name="handle_missing_values",
                parameters={
                    "modality_name": modality_name,
                    "method": method,
                    "knn_neighbors": knn_neighbors,
                },
                description=f"Imputed missing values using {method}",
                ir=ir,
            )

            response = f"Missing value imputation complete for '{modality_name}'.\n\n"
            response += "**Imputation Results:**\n"
            response += f"- Method: {stats['method_used']}\n"
            response += f"- Values imputed: {stats['n_values_imputed']}\n"
            response += f"- Percentage imputed: {stats['pct_imputed']:.1f}%\n"
            response += f"\n**New modality created**: '{result_name}'"
            return response

        except Exception as e:
            logger.error(f"Error in missing value imputation: {e}")
            return f"Error imputing missing values: {str(e)}"

    # =========================================================================
    # Tool 4: Normalization
    # =========================================================================

    @tool
    def normalize_metabolomics(
        modality_name: str,
        method: str = "pqn",
        log_transform: bool = True,
        reference_sample: str = None,
    ) -> str:
        """
        Normalize metabolomics data using standard methods.

        Supports PQN (gold standard), TIC, Internal Standard, median, and quantile
        normalization. Optionally applies log2 transformation.

        Args:
            modality_name: Name of the metabolomics modality
            method: Normalization method: "pqn", "tic", "is", "median", or "quantile"
            log_transform: Apply log2 transformation after normalization
            reference_sample: For IS normalization: comma-separated IS feature names

        Returns:
            str: Normalization results
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Use platform defaults if not explicitly overridden
            platform_config = _get_platform_for_modality(modality_name)
            norm_defaults = platform_config.get_normalization_defaults()

            # Tool parameters take priority over platform defaults
            actual_method = method
            actual_log = log_transform

            adata_norm, stats, ir = preprocessing_service.normalize(
                adata,
                method=actual_method,
                log_transform=actual_log,
                reference_sample=reference_sample,
            )

            result_name = f"{modality_name}_normalized"
            data_manager.store_modality(
                name=result_name,
                adata=adata_norm,
                parent_name=modality_name,
                step_summary=f"Normalized using {actual_method}, log2={actual_log}",
            )

            data_manager.log_tool_usage(
                tool_name="normalize_metabolomics",
                parameters={
                    "modality_name": modality_name,
                    "method": actual_method,
                    "log_transform": actual_log,
                    "reference_sample": reference_sample,
                },
                description=f"Normalized metabolomics data using {actual_method}",
                ir=ir,
            )

            response = f"Normalization complete for '{modality_name}'.\n\n"
            response += "**Normalization Results:**\n"
            response += f"- Method: {stats['method']}\n"
            response += f"- Log2 transformed: {stats['log_transformed']}\n"
            response += f"- Normalization factor range: [{stats['normalization_range'][0]:.3f}, {stats['normalization_range'][1]:.3f}]\n"
            response += f"- Platform defaults: {norm_defaults['method']} (log={norm_defaults['log_transform']})\n"
            response += f"\n**New modality created**: '{result_name}'"
            return response

        except Exception as e:
            logger.error(f"Error in normalization: {e}")
            return f"Error normalizing data: {str(e)}"

    # =========================================================================
    # Tool 5: Batch Effect Correction
    # =========================================================================

    @tool
    def correct_batch_effects(
        modality_name: str,
        batch_key: str,
        method: str = "combat",
        qc_label: str = "QC",
    ) -> str:
        """
        Correct batch effects in metabolomics data.

        Supports ComBat (parametric empirical Bayes), median centering,
        and QC-RLSC (QC-based signal correction using LOWESS smoothing).

        Args:
            modality_name: Name of the metabolomics modality
            batch_key: Column in obs containing batch labels
            method: Correction method: "combat", "median_centering", or "qc_rlsc"
            qc_label: Label for QC samples (required for qc_rlsc)

        Returns:
            str: Batch correction results
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            adata_corrected, stats, ir = preprocessing_service.correct_batch_effects(
                adata, batch_key=batch_key, method=method, qc_label=qc_label
            )

            result_name = f"{modality_name}_batch_corrected"
            data_manager.store_modality(
                name=result_name,
                adata=adata_corrected,
                parent_name=modality_name,
                step_summary=f"Batch corrected using {method}, {stats['n_batches']} batches",
            )

            data_manager.log_tool_usage(
                tool_name="correct_batch_effects",
                parameters={
                    "modality_name": modality_name,
                    "batch_key": batch_key,
                    "method": method,
                    "qc_label": qc_label,
                },
                description=f"Corrected batch effects using {method}",
                ir=ir,
            )

            response = f"Batch correction complete for '{modality_name}'.\n\n"
            response += "**Batch Correction Results:**\n"
            response += f"- Method: {stats['method']}\n"
            response += f"- Number of batches: {stats['n_batches']}\n"
            if stats.get("batch_sizes"):
                response += f"- Batch sizes: {stats['batch_sizes']}\n"
            if stats.get("correction_summary"):
                response += f"- Summary: {stats['correction_summary']}\n"
            response += f"\n**New modality created**: '{result_name}'"
            return response

        except Exception as e:
            logger.error(f"Error in batch correction: {e}")
            return f"Error correcting batch effects: {str(e)}"

    # =========================================================================
    # Tool 6: Univariate Statistics + Fold Changes
    # =========================================================================

    @tool
    def run_metabolomics_statistics(
        modality_name: str,
        group_column: str,
        method: str = "auto",
        fdr_method: str = "fdr_bh",
        fold_change_threshold: float = 1.5,
    ) -> str:
        """
        Run univariate statistics with FDR correction and fold change analysis.

        Performs statistical testing (t-test/Wilcoxon/ANOVA/Kruskal-Wallis) with
        multiple testing correction, then computes log2 fold changes.

        Args:
            modality_name: Name of the metabolomics modality
            group_column: Column in obs containing group labels (e.g., "condition")
            method: Statistical test: "auto", "ttest", "wilcoxon", "anova", or "kruskal"
            fdr_method: FDR correction method (default: Benjamini-Hochberg)
            fold_change_threshold: Fold change threshold for reporting

        Returns:
            str: Statistical analysis results with significant features and fold changes
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Step 1: Run univariate statistics
            adata_stats, stats, stats_ir = analysis_service.run_univariate_statistics(
                adata, group_column=group_column, method=method, fdr_method=fdr_method
            )

            # Step 2: Calculate fold changes on the same data
            adata_fc, fc_stats, fc_ir = analysis_service.calculate_fold_changes(
                adata_stats, group_column=group_column
            )

            result_name = f"{modality_name}_statistics"
            data_manager.store_modality(
                name=result_name,
                adata=adata_fc,
                parent_name=modality_name,
                step_summary=(
                    f"Statistics: {stats['n_significant_fdr']} significant (FDR), "
                    f"FC: {fc_stats['n_upregulated']} up / {fc_stats['n_downregulated']} down"
                ),
            )

            data_manager.log_tool_usage(
                tool_name="run_metabolomics_statistics",
                parameters={
                    "modality_name": modality_name,
                    "group_column": group_column,
                    "method": method,
                    "fdr_method": fdr_method,
                    "fold_change_threshold": fold_change_threshold,
                },
                description="Ran univariate statistics with fold change analysis",
                ir=stats_ir,
            )

            response = f"Statistical analysis complete for '{modality_name}'.\n\n"
            response += "**Univariate Statistics:**\n"
            response += f"- Method: {stats['method_used']}\n"
            response += f"- Features tested: {stats['n_tested']}\n"
            response += f"- Significant (raw p < 0.05): {stats['n_significant_raw']}\n"
            response += f"- Significant (FDR < 0.05): {stats['n_significant_fdr']}\n"
            response += f"- Groups: {stats['groups']}\n"

            response += f"\n**Fold Change Summary ({fc_stats['comparison']} vs {fc_stats['reference']}):**\n"
            response += f"- Upregulated (log2FC > 1): {fc_stats['n_upregulated']}\n"
            response += (
                f"- Downregulated (log2FC < -1): {fc_stats['n_downregulated']}\n"
            )

            response += f"\n**New modality created**: '{result_name}'"
            return response

        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return f"Error running statistics: {str(e)}"

    # =========================================================================
    # Tool 7: Multivariate Analysis (PCA / PLS-DA / OPLS-DA)
    # =========================================================================

    @tool
    def run_multivariate_analysis(
        modality_name: str,
        method: str = "pca",
        n_components: int = 2,
        group_column: str = None,
        permutation_test: bool = True,
    ) -> str:
        """
        Run multivariate analysis: PCA, PLS-DA, or OPLS-DA.

        PCA for unsupervised overview, PLS-DA for supervised classification with
        VIP scores, OPLS-DA for separating predictive from orthogonal variation.

        Args:
            modality_name: Name of the metabolomics modality
            method: Analysis method: "pca", "plsda", or "oplsda"
            n_components: Number of components to compute
            group_column: Column in obs with group labels (required for plsda/oplsda)
            permutation_test: Run permutation test for model validation (plsda/oplsda)

        Returns:
            str: Method-specific results
        """
        valid_methods = ["pca", "plsda", "oplsda"]
        if method not in valid_methods:
            return f"Invalid method '{method}'. Choose from: {valid_methods}"

        if method in ("plsda", "oplsda") and not group_column:
            return f"group_column is required for {method.upper()}. Provide the column name containing group labels."

        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            if method == "pca":
                adata_result, stats, ir = analysis_service.run_pca(
                    adata, n_components=n_components
                )
            elif method == "plsda":
                adata_result, stats, ir = analysis_service.run_pls_da(
                    adata,
                    group_column=group_column,
                    n_components=n_components,
                    permutation_test=permutation_test,
                )
            elif method == "oplsda":
                adata_result, stats, ir = analysis_service.run_opls_da(
                    adata,
                    group_column=group_column,
                    n_orthogonal=1,
                    n_predictive=n_components,
                    permutation_test=permutation_test,
                )

            result_name = f"{modality_name}_{method}"
            data_manager.store_modality(
                name=result_name,
                adata=adata_result,
                parent_name=modality_name,
                step_summary=f"Multivariate analysis: {method.upper()}",
            )

            data_manager.log_tool_usage(
                tool_name="run_multivariate_analysis",
                parameters={
                    "modality_name": modality_name,
                    "method": method,
                    "n_components": n_components,
                    "group_column": group_column,
                    "permutation_test": permutation_test,
                },
                description=f"Ran {method.upper()} multivariate analysis",
                ir=ir,
            )

            # Build method-specific response
            response = f"Multivariate analysis ({method.upper()}) complete for '{modality_name}'.\n\n"

            if method == "pca":
                response += "**PCA Results:**\n"
                response += f"- Components: {stats['n_components']}\n"
                response += f"- Total variance explained: {stats['total_variance_explained']:.1f}%\n"
                var_explained = stats.get("variance_explained", [])
                for i, var in enumerate(var_explained[:5]):
                    response += f"  - PC{i + 1}: {var:.1f}%\n"

            elif method == "plsda":
                response += "**PLS-DA Results:**\n"
                response += f"- Components: {stats['n_components']}\n"
                response += f"- R2: {stats['r2']:.3f}\n"
                response += f"- Q2: {stats['q2']:.3f}\n"
                response += f"- VIP > 1 features: {stats['vip_gt_1_count']}\n"
                if stats.get("permutation_p_value") is not None:
                    response += (
                        f"- Permutation p-value: {stats['permutation_p_value']:.4f}\n"
                    )
                # Overfitting warning
                if stats["r2"] - stats["q2"] > 0.3:
                    response += (
                        "\n**WARNING**: R2-Q2 gap > 0.3, potential overfitting.\n"
                    )

            elif method == "oplsda":
                response += "**OPLS-DA Results:**\n"
                response += f"- R2: {stats['r2']:.3f}\n"
                response += f"- Q2: {stats['q2']:.3f}\n"
                response += f"- Orthogonal components: {stats['n_orthogonal']}\n"
                response += f"- Predictive components: {stats['n_predictive']}\n"
                if stats.get("permutation_p_value") is not None:
                    response += (
                        f"- Permutation p-value: {stats['permutation_p_value']:.4f}\n"
                    )
                if stats["r2"] - stats["q2"] > 0.3:
                    response += (
                        "\n**WARNING**: R2-Q2 gap > 0.3, potential overfitting.\n"
                    )

            response += f"\n**New modality created**: '{result_name}'"
            return response

        except Exception as e:
            logger.error(f"Error in multivariate analysis: {e}")
            return f"Error running {method.upper()}: {str(e)}"

    # =========================================================================
    # Tool 8: Metabolite Annotation
    # =========================================================================

    @tool
    def annotate_metabolites(
        modality_name: str,
        ppm_tolerance: float = 10.0,
        adducts: str = None,
        ion_mode: str = "positive",
    ) -> str:
        """
        Annotate metabolites by m/z matching against reference database.

        Matches observed m/z values against a bundled database of ~80 common
        metabolites with adduct correction and ppm tolerance matching.
        Assigns MSI level 2 (putative annotation) for m/z-only matches.

        Args:
            modality_name: Name of the metabolomics modality
            ppm_tolerance: Mass accuracy tolerance in ppm (default: 10)
            adducts: Comma-separated adduct list (default: mode-appropriate set)
            ion_mode: Ionization mode: "positive" or "negative"

        Returns:
            str: Annotation results with annotation rate and MSI levels
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Parse adducts string if provided
            adduct_list = None
            if adducts:
                adduct_list = [a.strip() for a in adducts.split(",")]

            adata_ann, stats, ir = annotation_service.annotate_by_mz(
                adata,
                ppm_tolerance=ppm_tolerance,
                adducts=adduct_list,
                ion_mode=ion_mode,
            )

            result_name = f"{modality_name}_annotated"
            data_manager.store_modality(
                name=result_name,
                adata=adata_ann,
                parent_name=modality_name,
                step_summary=f"Annotated: {stats['n_annotated']} features ({stats['annotation_rate_pct']:.1f}%)",
            )

            data_manager.log_tool_usage(
                tool_name="annotate_metabolites",
                parameters={
                    "modality_name": modality_name,
                    "ppm_tolerance": ppm_tolerance,
                    "adducts": adducts,
                    "ion_mode": ion_mode,
                },
                description="Annotated metabolites by m/z matching",
                ir=ir,
            )

            response = f"Metabolite annotation complete for '{modality_name}'.\n\n"
            response += "**Annotation Results:**\n"
            response += f"- Annotated: {stats['n_annotated']}\n"
            response += f"- Unannotated: {stats['n_unannotated']}\n"
            response += f"- Annotation rate: {stats['annotation_rate_pct']:.1f}%\n"
            response += (
                f"- Reference DB size: {stats['reference_db_size']} metabolites\n"
            )
            response += f"\n**MSI Level Distribution:**\n"
            for level, count in sorted(stats.get("msi_level_distribution", {}).items()):
                response += f"- {level}: {count}\n"
            response += f"\n**New modality created**: '{result_name}'"
            return response

        except Exception as e:
            logger.error(f"Error in metabolite annotation: {e}")
            return f"Error annotating metabolites: {str(e)}"

    # =========================================================================
    # Tool 9: Lipid Class Analysis
    # =========================================================================

    @tool
    def analyze_lipid_classes(
        modality_name: str,
    ) -> str:
        """
        Classify features by lipid class from annotations or m/z ranges.

        For annotated features: groups by annotation class. For unannotated
        features: uses m/z ranges for rough classification (MSI level 3).

        Args:
            modality_name: Name of the metabolomics modality (ideally after annotation)

        Returns:
            str: Lipid class distribution with feature counts
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            adata_lipid, stats, ir = annotation_service.classify_lipids(adata)

            result_name = f"{modality_name}_lipid_classified"
            data_manager.store_modality(
                name=result_name,
                adata=adata_lipid,
                parent_name=modality_name,
                step_summary=f"Lipid classification: {stats['n_classes']} classes identified",
            )

            data_manager.log_tool_usage(
                tool_name="analyze_lipid_classes",
                parameters={"modality_name": modality_name},
                description="Classified features by lipid class",
                ir=ir,
            )

            response = f"Lipid class analysis complete for '{modality_name}'.\n\n"
            response += "**Lipid Classes Found:**\n"
            response += f"- Number of classes: {stats['n_classes']}\n"
            response += f"- MSI level: {stats['msi_level']}\n"

            response += f"\n**Class Distribution:**\n"
            for cls, count in sorted(
                stats.get("class_counts", {}).items(), key=lambda x: -x[1]
            ):
                response += f"- {cls}: {count} features\n"

            response += f"\n**New modality created**: '{result_name}'"
            return response

        except Exception as e:
            logger.error(f"Error in lipid classification: {e}")
            return f"Error classifying lipids: {str(e)}"

    # =========================================================================
    # Tool 10: Pathway Enrichment
    # =========================================================================

    @tool
    def run_pathway_enrichment(
        modality_name: str,
        database: str = "KEGG_2021_Human",
        significance_threshold: float = 0.05,
    ) -> str:
        """
        Run pathway enrichment analysis on significant metabolites.

        Uses the core PathwayEnrichmentService to identify enriched metabolic
        pathways from significant or annotated metabolites.

        Args:
            modality_name: Name of the metabolomics modality
            database: Enrichment database (default: "KEGG_2021_Human")
            significance_threshold: FDR threshold for selecting significant metabolites

        Returns:
            str: Pathway enrichment results with top pathways
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Get the pathway enrichment service from core
            pathway_service = None
            try:
                from lobster.core.component_registry import component_registry

                pathway_service = component_registry.get_service("pathway_enrichment")
            except Exception:
                pass

            if pathway_service is None:
                try:
                    from lobster.services.analysis.pathway_enrichment_service import (
                        PathwayEnrichmentBridgeService,
                    )

                    pathway_service = PathwayEnrichmentBridgeService()
                except ImportError:
                    return (
                        "Pathway enrichment service not available. "
                        "Install lobster-ai with pathway support or ensure "
                        "PathwayEnrichmentBridgeService is accessible."
                    )

            # Extract significant metabolite names
            metabolite_names = []

            # Priority 1: Use FDR-significant annotated metabolites
            if "fdr" in adata.var.columns and "annotation_name" in adata.var.columns:
                sig_mask = adata.var["fdr"] < significance_threshold
                ann_mask = adata.var["annotation_name"].astype(str).str.strip().ne("")
                combined_mask = sig_mask & ann_mask
                metabolite_names = adata.var.loc[
                    combined_mask, "annotation_name"
                ].tolist()
                logger.info(
                    f"Using {len(metabolite_names)} FDR-significant annotated metabolites"
                )

            # Priority 2: Use all annotated metabolites
            if not metabolite_names and "annotation_name" in adata.var.columns:
                ann_mask = adata.var["annotation_name"].astype(str).str.strip().ne("")
                metabolite_names = adata.var.loc[ann_mask, "annotation_name"].tolist()
                logger.info(
                    f"Using all {len(metabolite_names)} annotated metabolites (no FDR data)"
                )

            # Priority 3: Use var_names as feature names
            if not metabolite_names:
                metabolite_names = adata.var_names.tolist()
                logger.info(
                    f"Using {len(metabolite_names)} feature names for pathway enrichment"
                )

            if not metabolite_names:
                return "No metabolite names available for pathway enrichment. Run annotation first."

            # Run enrichment using core service
            # The pathway service expects a gene list and database
            enrichment_results = pathway_service.overrepresentation_analysis(
                gene_list=metabolite_names,
                gene_sets=database,
            )

            # Store results in uns
            adata_enriched = adata.copy()
            adata_enriched.uns["pathway_enrichment"] = {
                "database": database,
                "n_metabolites_input": len(metabolite_names),
                "results": (
                    enrichment_results if isinstance(enrichment_results, dict) else {}
                ),
            }

            result_name = f"{modality_name}_pathway_enriched"
            data_manager.store_modality(
                name=result_name,
                adata=adata_enriched,
                parent_name=modality_name,
                step_summary=f"Pathway enrichment: {database}",
            )

            # Create IR for provenance
            ir = AnalysisStep(
                operation="metabolomics.analysis.pathway_enrichment",
                tool_name="run_pathway_enrichment",
                description=f"Pathway enrichment analysis using {database}",
                library="lobster.services.analysis",
                code_template="""# Pathway enrichment for metabolomics
from lobster.services.analysis.pathway_enrichment_service import PathwayEnrichmentBridgeService

service = PathwayEnrichmentBridgeService()
results = service.overrepresentation_analysis(
    gene_list={{ metabolite_names | tojson }},
    gene_sets={{ database | tojson }}
)""",
                imports=[
                    "from lobster.services.analysis.pathway_enrichment_service import PathwayEnrichmentBridgeService"
                ],
                parameters={
                    "database": database,
                    "significance_threshold": significance_threshold,
                    "n_metabolites": len(metabolite_names),
                },
                parameter_schema={
                    "database": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value="KEGG_2021_Human",
                        required=False,
                        description="Enrichment database",
                    ),
                    "significance_threshold": ParameterSpec(
                        param_type="float",
                        papermill_injectable=True,
                        default_value=0.05,
                        required=False,
                        description="FDR threshold for metabolite selection",
                    ),
                },
                input_entities=["adata"],
                output_entities=["adata_enriched"],
            )

            data_manager.log_tool_usage(
                tool_name="run_pathway_enrichment",
                parameters={
                    "modality_name": modality_name,
                    "database": database,
                    "significance_threshold": significance_threshold,
                },
                description="Ran pathway enrichment analysis",
                ir=ir,
            )

            # Format response
            response = f"Pathway enrichment complete for '{modality_name}'.\n\n"
            response += "**Enrichment Parameters:**\n"
            response += f"- Database: {database}\n"
            response += f"- Input metabolites: {len(metabolite_names)}\n"

            # Parse enrichment results for display
            if isinstance(enrichment_results, dict):
                sig_pathways = []
                for pathway_name, pathway_data in enrichment_results.items():
                    if isinstance(pathway_data, dict):
                        p_val = pathway_data.get(
                            "p_value", pathway_data.get("pval", 1.0)
                        )
                        if p_val < 0.05:
                            sig_pathways.append((pathway_name, p_val))

                sig_pathways.sort(key=lambda x: x[1])
                n_sig = len(sig_pathways)
                response += f"\n**Significant Pathways: {n_sig}**\n"
                for name, pval in sig_pathways[:5]:
                    response += f"- {name}: p={pval:.4f}\n"
                if n_sig > 5:
                    response += f"- ... and {n_sig - 5} more\n"
            else:
                response += "\n**Note**: Results stored in uns['pathway_enrichment']\n"

            response += f"\n**New modality created**: '{result_name}'"
            return response

        except Exception as e:
            logger.error(f"Error in pathway enrichment: {e}")
            return f"Error running pathway enrichment: {str(e)}"

    # =========================================================================
    # Return all 10 tools
    # =========================================================================

    return [
        assess_metabolomics_quality,
        filter_metabolomics_features,
        handle_missing_values,
        normalize_metabolomics,
        correct_batch_effects,
        run_metabolomics_statistics,
        run_multivariate_analysis,
        annotate_metabolites,
        analyze_lipid_classes,
        run_pathway_enrichment,
    ]
