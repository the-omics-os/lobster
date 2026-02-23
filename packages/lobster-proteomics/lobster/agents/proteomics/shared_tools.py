"""
Shared tools for proteomics analysis (mass spectrometry and affinity platforms).

This module provides tools shared by the proteomics parent agent and sub-agents.
Tools auto-detect platform type and use appropriate defaults.

Following the same factory pattern as transcriptomics shared_tools.py.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from langchain_core.tools import tool

from lobster.agents.proteomics.config import (
    PlatformConfig,
    detect_platform_type,
    get_platform_config,
)
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.proteomics_analysis_service import (
    ProteomicsAnalysisService,
)
from lobster.services.quality.proteomics_preprocessing_service import (
    ProteomicsPreprocessingService,
)
from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SHARED TOOL FACTORY
# =============================================================================


def create_shared_tools(
    data_manager: DataManagerV2,
    quality_service: ProteomicsQualityService,
    preprocessing_service: ProteomicsPreprocessingService,
    analysis_service: ProteomicsAnalysisService,
    force_platform_type: Optional[str] = None,
) -> List[Callable]:
    """
    Create shared proteomics tools with platform auto-detection.

    These tools are shared between the proteomics parent agent and sub-agents.
    Each tool auto-detects the platform type and applies appropriate defaults.

    Args:
        data_manager: DataManagerV2 instance for modality management
        quality_service: ProteomicsQualityService for QC operations
        preprocessing_service: ProteomicsPreprocessingService for normalization/imputation
        analysis_service: ProteomicsAnalysisService for PCA/clustering
        force_platform_type: Override auto-detection ("mass_spec" or "affinity")

    Returns:
        List of tool functions to be added to agent tools
    """
    # Analysis results storage (shared across tools)
    analysis_results: Dict[str, Any] = {"summary": "", "details": {}}

    # Store forced platform type for tools to access
    _forced_platform_type = force_platform_type

    def _get_platform_for_modality(modality_name: str) -> PlatformConfig:
        """Get platform config for a modality, using detection or forced type."""
        if _forced_platform_type:
            return get_platform_config(_forced_platform_type)

        try:
            adata = data_manager.get_modality(modality_name)
            detected_type = detect_platform_type(adata)
            if detected_type == "unknown":
                logger.warning(f"Platform type ambiguous for '{modality_name}'. Defaulting to mass_spec. Use platform_type parameter to override.")
                return get_platform_config("mass_spec")
            return get_platform_config(detected_type)
        except ValueError:
            # Default to mass_spec if modality not found
            return get_platform_config("mass_spec")

    # -------------------------
    # STATUS TOOL
    # -------------------------
    @tool
    def check_proteomics_status(modality_name: str = "") -> str:
        """
        Check status of proteomics modalities and detect platform type.

        Args:
            modality_name: Specific modality to check (empty for all proteomics modalities)

        Returns:
            str: Status report with platform detection and data characteristics
        """
        try:
            if modality_name == "":
                # Show all modalities with proteomics focus
                modalities = data_manager.list_modalities()
                proteomics_terms = [
                    "proteomics",
                    "protein",
                    "ms",
                    "mass_spec",
                    "olink",
                    "soma",
                    "affinity",
                    "panel",
                ]
                proteomics_modalities = [
                    m
                    for m in modalities
                    if any(term in m.lower() for term in proteomics_terms)
                ]

                if not proteomics_modalities:
                    response = f"No proteomics modalities found. Available modalities: {modalities}\n"
                    response += "Ask the data_expert to load proteomics data using appropriate adapter."
                    return response

                response = f"Proteomics modalities ({len(proteomics_modalities)}):\n\n"
                for mod_name in proteomics_modalities:
                    adata = data_manager.get_modality(mod_name)
                    platform_config = _get_platform_for_modality(mod_name)
                    metrics = data_manager.get_quality_metrics(mod_name)

                    response += f"**{mod_name}**\n"
                    response += f"- Platform: {platform_config.display_name}\n"
                    response += (
                        f"- Shape: {adata.n_obs} samples x {adata.n_vars} proteins\n"
                    )
                    if "missing_value_percentage" in metrics:
                        response += f"- Missing values: {metrics['missing_value_percentage']:.1f}%\n"
                    response += "\n"

                return response

            else:
                # Check specific modality
                try:
                    adata = data_manager.get_modality(modality_name)
                    platform_config = _get_platform_for_modality(modality_name)
                    metrics = data_manager.get_quality_metrics(modality_name)

                    # Calculate missing rate
                    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
                    missing_rate = np.isnan(X).sum() / X.size if X.size > 0 else 0

                    response = f"Proteomics modality '{modality_name}' status:\n\n"
                    response += "**Platform Detection:**\n"
                    response += f"- Detected platform: {platform_config.display_name}\n"
                    response += f"- Platform type: {platform_config.platform_type}\n\n"

                    response += "**Data Characteristics:**\n"
                    response += (
                        f"- Shape: {adata.n_obs} samples x {adata.n_vars} proteins\n"
                    )
                    response += f"- Missing values: {missing_rate * 100:.1f}%\n"
                    expected_range = platform_config.expected_missing_rate_range
                    response += f"- Expected range for platform: {expected_range[0] * 100:.0f}-{expected_range[1] * 100:.0f}%\n"

                    # Platform-specific metadata check
                    if platform_config.platform_type == "mass_spec":
                        ms_cols = [
                            "n_peptides",
                            "n_unique_peptides",
                            "sequence_coverage",
                        ]
                        present_cols = [
                            col for col in ms_cols if col in adata.var.columns
                        ]
                        if present_cols:
                            response += f"- MS metadata available: {present_cols}\n"
                    else:
                        affinity_cols = ["antibody_id", "panel_type", "plate_id"]
                        present_cols = [
                            col for col in affinity_cols if col in adata.var.columns
                        ]
                        if present_cols:
                            response += (
                                f"- Affinity metadata available: {present_cols}\n"
                            )

                    # Show key columns
                    obs_cols = list(adata.obs.columns)[:5]
                    var_cols = list(adata.var.columns)[:5]
                    response += "\n**Metadata:**\n"
                    response += f"- Sample columns: {obs_cols}...\n"
                    response += f"- Protein columns: {var_cols}...\n"

                    analysis_results["details"]["status"] = response
                    return response

                except ValueError:
                    return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        except Exception as e:
            logger.error(f"Error checking proteomics status: {e}")
            return f"Error checking status: {str(e)}"

    # -------------------------
    # QUALITY ASSESSMENT TOOL
    # -------------------------
    @tool
    def assess_proteomics_quality(
        modality_name: str,
        platform_type: str = "auto",
    ) -> str:
        """
        Run comprehensive quality assessment with platform-appropriate metrics.

        Args:
            modality_name: Name of the proteomics modality to assess
            platform_type: Platform type ("mass_spec", "affinity", or "auto" for detection)

        Returns:
            str: Quality assessment report with platform-specific metrics
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Determine platform
            if platform_type == "auto":
                platform_config = _get_platform_for_modality(modality_name)
            else:
                platform_config = get_platform_config(platform_type)

            # Run quality assessments using service (returns 3-tuples)
            processed_adata, missing_stats, missing_ir = (
                quality_service.assess_missing_value_patterns(
                    adata,
                    sample_threshold=platform_config.max_missing_per_sample,
                    protein_threshold=platform_config.max_missing_per_protein,
                )
            )

            cv_adata, cv_stats, cv_ir = quality_service.assess_coefficient_variation(
                processed_adata,
                cv_threshold=platform_config.cv_threshold / 100,  # Convert to decimal
            )

            contam_adata, contam_stats, contam_ir = quality_service.detect_contaminants(
                cv_adata
            )
            final_adata, range_stats, range_ir = quality_service.evaluate_dynamic_range(
                contam_adata
            )

            # Update the modality with quality assessment results
            assessed_name = f"{modality_name}_quality_assessed"
            data_manager.store_modality(
                name=assessed_name,
                adata=final_adata,
                parent_name=modality_name,
                step_summary=f"Quality assessed: {platform_config.display_name} data",
            )

            # Log with IR
            combined_stats = {
                **missing_stats,
                **cv_stats,
                **contam_stats,
                **range_stats,
            }
            data_manager.log_tool_usage(
                tool_name="assess_proteomics_quality",
                parameters={
                    "modality_name": modality_name,
                    "platform_type": platform_config.platform_type,
                },
                description=f"Quality assessment for {platform_config.display_name} data",
                ir=missing_ir,  # Use the first IR as representative
            )

            # Generate response
            response = f"Proteomics Quality Assessment for '{modality_name}':\n\n"
            response += f"**Platform:** {platform_config.display_name}\n\n"

            response += "**Dataset Characteristics:**\n"
            response += f"- Samples: {final_adata.n_obs}\n"
            response += f"- Proteins: {final_adata.n_vars}\n"

            # Missing values
            if "missing_value_percentage" in combined_stats:
                mv_pct = combined_stats["missing_value_percentage"]
                expected = platform_config.expected_missing_rate_range
                status = (
                    "OK"
                    if expected[0] * 100 <= mv_pct <= expected[1] * 100
                    else "CHECK"
                )
                response += f"- Missing values: {mv_pct:.1f}% [{status}] (expected: {expected[0] * 100:.0f}-{expected[1] * 100:.0f}%)\n"

            # CV metrics
            if "median_cv" in combined_stats:
                cv_val = combined_stats["median_cv"]
                cv_status = "OK" if cv_val <= platform_config.cv_threshold else "HIGH"
                response += f"- Median CV: {cv_val:.1f}% [{cv_status}] (threshold: {platform_config.cv_threshold}%)\n"

            # Platform-specific metrics
            if platform_config.platform_type == "mass_spec":
                if "contaminant_proteins" in combined_stats:
                    response += f"- Contaminant proteins: {combined_stats['contaminant_proteins']}\n"
                if "reverse_hits" in combined_stats:
                    response += (
                        f"- Reverse database hits: {combined_stats['reverse_hits']}\n"
                    )
            else:
                if "high_cv_proteins" in combined_stats:
                    response += (
                        f"- High CV proteins: {combined_stats['high_cv_proteins']}\n"
                    )

            # Dynamic range
            if "dynamic_range_log10" in combined_stats:
                response += f"- Dynamic range: {combined_stats['dynamic_range_log10']:.1f} log10 units\n"

            # Recommendations
            response += "\n**Platform-Specific Recommendations:**\n"
            if platform_config.platform_type == "mass_spec":
                if combined_stats.get("contaminant_proteins", 0) > 0:
                    response += (
                        "- Remove contaminant proteins before downstream analysis\n"
                    )
                if combined_stats.get("reverse_hits", 0) > 0:
                    response += "- Remove reverse database hits (search artifacts)\n"
                response += "- Consider peptide count requirements for reliable quantification\n"
            else:
                if combined_stats.get("missing_value_percentage", 0) > 30:
                    response += "- High missing values unusual for affinity - check assay quality\n"
                if combined_stats.get("median_cv", 0) > 30:
                    response += (
                        "- High CVs suggest technical issues - check sample handling\n"
                    )
                response += "- Check for plate effects and correct if needed\n"

            response += f"\n**New modality created**: '{assessed_name}'"

            analysis_results["details"]["quality_assessment"] = response
            return response

        except Exception as e:
            logger.error(f"Error in proteomics quality assessment: {e}")
            return f"Error in quality assessment: {str(e)}"

    # -------------------------
    # FILTER TOOL
    # -------------------------
    @tool
    def filter_proteomics_data(
        modality_name: str,
        platform_type: str = "auto",
        max_missing_per_sample: float = None,
        max_missing_per_protein: float = None,
        save_result: bool = True,
    ) -> str:
        """
        Filter proteomics data with platform-specific quality criteria.

        Args:
            modality_name: Name of the proteomics modality to filter
            platform_type: Platform type ("mass_spec", "affinity", or "auto")
            max_missing_per_sample: Override default missing threshold per sample
            max_missing_per_protein: Override default missing threshold per protein
            save_result: Whether to save the filtered modality

        Returns:
            str: Filtering report with statistics
        """
        try:
            adata = data_manager.get_modality(modality_name)
            original_shape = adata.shape
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Determine platform and defaults
            if platform_type == "auto":
                platform_config = _get_platform_for_modality(modality_name)
            else:
                platform_config = get_platform_config(platform_type)

            # Use platform defaults if not overridden
            max_sample = (
                max_missing_per_sample or platform_config.max_missing_per_sample
            )
            max_protein = (
                max_missing_per_protein or platform_config.max_missing_per_protein
            )

            # Create working copy
            adata_filtered = adata.copy()
            X = (
                adata_filtered.X.toarray()
                if hasattr(adata_filtered.X, "toarray")
                else adata_filtered.X
            )

            # Step 1: Filter based on missing values
            sample_missing_rate = np.isnan(X).sum(axis=1) / X.shape[1]
            protein_missing_rate = np.isnan(X).sum(axis=0) / X.shape[0]

            sample_filter = sample_missing_rate <= max_sample
            adata_filtered = adata_filtered[sample_filter, :].copy()

            # Recalculate after sample filtering
            X = (
                adata_filtered.X.toarray()
                if hasattr(adata_filtered.X, "toarray")
                else adata_filtered.X
            )
            protein_missing_rate = np.isnan(X).sum(axis=0) / X.shape[0]
            protein_filter = protein_missing_rate <= max_protein
            adata_filtered = adata_filtered[:, protein_filter].copy()

            # Step 2: Platform-specific filtering
            if platform_config.platform_type == "mass_spec":
                # MS: Filter by peptide count
                if "n_peptides" in adata_filtered.var.columns:
                    min_peptides = platform_config.platform_specific.get(
                        "min_peptides_per_protein", 2
                    )
                    peptide_filter = adata_filtered.var["n_peptides"] >= min_peptides
                    adata_filtered = adata_filtered[:, peptide_filter].copy()

                # MS: Remove contaminants
                if platform_config.platform_specific.get("remove_contaminants", True):
                    if "is_contaminant" in adata_filtered.var.columns:
                        adata_filtered = adata_filtered[
                            :, ~adata_filtered.var["is_contaminant"]
                        ].copy()

                # MS: Remove reverse hits
                if platform_config.platform_specific.get("remove_reverse_hits", True):
                    if "is_reverse" in adata_filtered.var.columns:
                        adata_filtered = adata_filtered[
                            :, ~adata_filtered.var["is_reverse"]
                        ].copy()
            else:
                # Affinity: Filter by CV (column added by external Olink/SomaScan metadata, not by Lobster tools)
                if "cv" in adata_filtered.var.columns:
                    cv_filter = adata_filtered.var["cv"] <= platform_config.cv_threshold
                    adata_filtered = adata_filtered[:, cv_filter].copy()

                # Affinity: Remove failed antibodies (column from assay QC metadata)
                if "antibody_quality" in adata_filtered.var.columns:
                    quality_filter = adata_filtered.var["antibody_quality"] != "failed"
                    adata_filtered = adata_filtered[:, quality_filter].copy()

            # Update modality
            filtered_name = f"{modality_name}_filtered"
            data_manager.store_modality(
                name=filtered_name,
                adata=adata_filtered,
                parent_name=modality_name,
                step_summary=f"Filtered {platform_config.display_name} data",
            )

            # Save if requested
            if save_result:
                save_path = f"{modality_name}_filtered.h5ad"
                data_manager.save_modality(filtered_name, save_path)

            # Create IR for provenance tracking
            ir = AnalysisStep(
                operation="proteomics.filtering.filter_data",
                tool_name="filter_proteomics_data",
                description=f"Filter proteomics data for {platform_config.display_name}",
                library="lobster.agents.proteomics.shared_tools",
                code_template="""# Proteomics data filtering
import numpy as np

# Filter based on missing values
X = adata.X.copy() if not hasattr(adata.X, 'toarray') else adata.X.toarray()
sample_missing_rate = np.isnan(X).sum(axis=1) / X.shape[1]
protein_missing_rate = np.isnan(X).sum(axis=0) / X.shape[0]

sample_filter = sample_missing_rate <= {{ max_missing_per_sample }}
adata_filtered = adata[sample_filter, :].copy()

X = adata_filtered.X.copy() if not hasattr(adata_filtered.X, 'toarray') else adata_filtered.X.toarray()
protein_missing_rate = np.isnan(X).sum(axis=0) / X.shape[0]
protein_filter = protein_missing_rate <= {{ max_missing_per_protein }}
adata_filtered = adata_filtered[:, protein_filter].copy()""",
                imports=["import numpy as np"],
                parameters={
                    "max_missing_per_sample": max_sample,
                    "max_missing_per_protein": max_protein,
                    "platform_type": platform_config.platform_type,
                },
                parameter_schema={
                    "max_missing_per_sample": ParameterSpec(
                        param_type="float",
                        papermill_injectable=True,
                        default_value=0.7,
                        required=False,
                        validation_rule="0 < max_missing_per_sample <= 1",
                        description="Maximum fraction of missing values per sample",
                    ),
                    "max_missing_per_protein": ParameterSpec(
                        param_type="float",
                        papermill_injectable=True,
                        default_value=0.8,
                        required=False,
                        validation_rule="0 < max_missing_per_protein <= 1",
                        description="Maximum fraction of missing values per protein",
                    ),
                },
                input_entities=["adata"],
                output_entities=["adata_filtered"],
            )

            # Log operation
            data_manager.log_tool_usage(
                tool_name="filter_proteomics_data",
                parameters={
                    "modality_name": modality_name,
                    "platform_type": platform_config.platform_type,
                    "max_missing_per_sample": max_sample,
                    "max_missing_per_protein": max_protein,
                },
                description=f"Filtered {platform_config.display_name} data",
                ir=ir,
            )

            # Generate summary
            samples_removed = original_shape[0] - adata_filtered.n_obs
            proteins_removed = original_shape[1] - adata_filtered.n_vars

            response = (
                f"Successfully filtered proteomics modality '{modality_name}'!\n\n"
            )
            response += f"**Platform:** {platform_config.display_name}\n\n"
            response += "**Filtering Results:**\n"
            response += f"- Original shape: {original_shape[0]} samples x {original_shape[1]} proteins\n"
            response += f"- Filtered shape: {adata_filtered.n_obs} samples x {adata_filtered.n_vars} proteins\n"
            response += f"- Samples removed: {samples_removed} ({samples_removed / original_shape[0] * 100:.1f}%)\n"
            response += f"- Proteins removed: {proteins_removed} ({proteins_removed / original_shape[1] * 100:.1f}%)\n\n"

            response += (
                f"**Filtering Parameters ({platform_config.display_name} defaults):**\n"
            )
            response += f"- Max missing per sample: {max_sample * 100:.0f}%\n"
            response += f"- Max missing per protein: {max_protein * 100:.0f}%\n"

            response += f"\n**New modality created**: '{filtered_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            analysis_results["details"]["filtering"] = response
            return response

        except Exception as e:
            logger.error(f"Error filtering proteomics data: {e}")
            return f"Error in filtering: {str(e)}"

    # -------------------------
    # NORMALIZE TOOL
    # -------------------------
    @tool
    def normalize_proteomics_data(
        modality_name: str,
        platform_type: str = "auto",
        normalization_method: str = None,
        log_transform: bool = None,
        handle_missing: str = None,
        save_result: bool = True,
    ) -> str:
        """
        Normalize proteomics data using platform-appropriate methods.

        Args:
            modality_name: Name of the proteomics modality to normalize
            platform_type: Platform type ("mass_spec", "affinity", or "auto")
            normalization_method: Override default method (median, quantile, vsn)
            log_transform: Override log transformation setting
            handle_missing: Override imputation method (keep, impute_knn, impute_min)
            save_result: Whether to save the normalized modality

        Returns:
            str: Normalization report with processing details
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Determine platform and defaults
            if platform_type == "auto":
                platform_config = _get_platform_for_modality(modality_name)
            else:
                platform_config = get_platform_config(platform_type)

            # Use platform defaults if not overridden
            norm_method = normalization_method or platform_config.default_normalization
            do_log = (
                log_transform
                if log_transform is not None
                else platform_config.log_transform
            )
            impute_method = handle_missing or platform_config.default_imputation

            # Step 1: Handle missing values if requested
            impute_stats = {}
            if impute_method == "impute_knn":
                processed_adata, impute_stats, impute_ir = (
                    preprocessing_service.impute_missing_values(adata, method="knn")
                )
            elif impute_method == "impute_min":
                processed_adata, impute_stats, impute_ir = (
                    preprocessing_service.impute_missing_values(
                        adata, method="min_prob"
                    )
                )
            else:
                processed_adata = adata.copy()
                impute_stats = {
                    "imputation_method": "none",
                    "imputation_applied": False,
                }

            # Step 2: Normalize
            normalized_adata, norm_stats, norm_ir = (
                preprocessing_service.normalize_intensities(
                    processed_adata,
                    method=norm_method,
                    log_transform=do_log,
                )
            )

            # Update modality
            normalized_name = f"{modality_name}_normalized"
            data_manager.store_modality(
                name=normalized_name,
                adata=normalized_adata,
                parent_name=modality_name,
                step_summary=f"Normalized using {norm_method}",
            )

            # Save if requested
            if save_result:
                save_path = f"{modality_name}_normalized.h5ad"
                data_manager.save_modality(normalized_name, save_path)

            # Log operation
            combined_stats = {**impute_stats, **norm_stats}
            data_manager.log_tool_usage(
                tool_name="normalize_proteomics_data",
                parameters={
                    "modality_name": modality_name,
                    "platform_type": platform_config.platform_type,
                    "normalization_method": norm_method,
                    "log_transform": do_log,
                    "handle_missing": impute_method,
                },
                description=f"Normalized {platform_config.display_name} data",
                ir=norm_ir,
            )

            # Generate response
            response = (
                f"Successfully normalized proteomics modality '{modality_name}'!\n\n"
            )
            response += f"**Platform:** {platform_config.display_name}\n\n"
            response += "**Normalization Settings:**\n"
            response += f"- Method: {norm_method}\n"
            response += f"- Log transformation: {do_log}\n"
            response += f"- Missing value handling: {impute_method}\n\n"

            response += "**Processing Details:**\n"
            if combined_stats.get("imputation_applied", False):
                response += f"- Imputation applied: {combined_stats.get('imputation_method', 'unknown')}\n"
                if "n_imputed_values" in combined_stats:
                    response += (
                        f"- Values imputed: {combined_stats['n_imputed_values']}\n"
                    )
            else:
                if platform_config.platform_type == "mass_spec":
                    response += "- Preserved missing values (MNAR pattern preserved)\n"
                else:
                    response += "- No imputation applied\n"

            if combined_stats.get("log_transform_applied", False):
                response += "- Log2 transformation applied\n"

            response += f"\n**New modality created**: '{normalized_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            analysis_results["details"]["normalization"] = response
            return response

        except Exception as e:
            logger.error(f"Error normalizing proteomics data: {e}")
            return f"Error in normalization: {str(e)}"

    # -------------------------
    # PATTERN ANALYSIS TOOL
    # -------------------------
    @tool
    def analyze_proteomics_patterns(
        modality_name: str,
        platform_type: str = "auto",
        analysis_type: str = "pca_clustering",
        n_components: int = None,
        n_clusters: int = 4,
        save_result: bool = True,
    ) -> str:
        """
        Perform pattern analysis with dimensionality reduction and clustering.

        Args:
            modality_name: Name of the proteomics modality to analyze
            platform_type: Platform type ("mass_spec", "affinity", or "auto")
            analysis_type: Type of analysis ("pca_clustering", "pca_only")
            n_components: Number of PCA components (uses platform default if None)
            n_clusters: Number of clusters for sample grouping
            save_result: Whether to save results

        Returns:
            str: Analysis report with PCA and clustering results
        """
        try:
            adata = data_manager.get_modality(modality_name).copy()
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Determine platform and defaults
            if platform_type == "auto":
                platform_config = _get_platform_for_modality(modality_name)
            else:
                platform_config = get_platform_config(platform_type)

            n_pcs = n_components or platform_config.default_n_pca_components

            # Perform dimensionality reduction
            pca_adata, pca_stats, pca_ir = (
                analysis_service.perform_dimensionality_reduction(
                    adata, method="pca", n_components=n_pcs
                )
            )

            # Perform clustering if requested
            if analysis_type == "pca_clustering":
                clustered_adata, cluster_stats, cluster_ir = (
                    analysis_service.perform_clustering_analysis(
                        pca_adata, clustering_method="kmeans", n_clusters=n_clusters
                    )
                )
                final_adata = clustered_adata
                combined_stats = {**pca_stats, **cluster_stats}
                ir = cluster_ir
            else:
                final_adata = pca_adata
                combined_stats = pca_stats
                ir = pca_ir

            # Update modality
            analyzed_name = f"{modality_name}_analyzed"
            data_manager.store_modality(
                name=analyzed_name,
                adata=final_adata,
                parent_name=modality_name,
                step_summary=f"Pattern analysis: {analysis_type}",
            )

            # Save if requested
            if save_result:
                save_path = f"{modality_name}_analyzed.h5ad"
                data_manager.save_modality(analyzed_name, save_path)

            # Log operation
            data_manager.log_tool_usage(
                tool_name="analyze_proteomics_patterns",
                parameters={
                    "modality_name": modality_name,
                    "platform_type": platform_config.platform_type,
                    "analysis_type": analysis_type,
                    "n_components": n_pcs,
                    "n_clusters": n_clusters,
                },
                description=f"Pattern analysis for {platform_config.display_name} data",
                ir=ir,
            )

            # Generate response
            response = (
                f"Successfully analyzed proteomics patterns in '{modality_name}'!\n\n"
            )
            response += f"**Platform:** {platform_config.display_name}\n\n"
            response += "**PCA Results:**\n"
            response += f"- Components computed: {n_pcs}\n"

            if "explained_variance_ratio" in combined_stats:
                ev_ratio = combined_stats["explained_variance_ratio"][:3]
                response += f"- Explained variance (PC1-PC3): {[f'{x * 100:.1f}%' for x in ev_ratio]}\n"

            if "components_for_90_variance" in combined_stats:
                response += f"- Components for 90% variance: {combined_stats['components_for_90_variance']}\n"

            if analysis_type == "pca_clustering":
                response += "\n**Clustering Results:**\n"
                if "n_clusters_found" in combined_stats:
                    response += f"- Clusters: {combined_stats['n_clusters_found']}\n"
                if "silhouette_score" in combined_stats:
                    response += f"- Silhouette score: {combined_stats['silhouette_score']:.3f}\n"

            response += f"\n**New modality created**: '{analyzed_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            analysis_results["details"]["pattern_analysis"] = response
            return response

        except Exception as e:
            logger.error(f"Error analyzing proteomics patterns: {e}")
            return f"Error in pattern analysis: {str(e)}"

    # -------------------------
    # IMPUTE MISSING VALUES TOOL
    # -------------------------
    @tool
    def impute_missing_values(
        modality_name: str,
        method: str = "knn",
        platform_type: str = "auto",
        save_result: bool = True,
    ) -> str:
        """
        Impute missing values in proteomics data using platform-appropriate methods.

        For mass spectrometry data (MNAR pattern), use 'min_prob' or 'min' methods.
        For affinity data (MAR pattern), use 'knn' method.

        Args:
            modality_name: Name of the proteomics modality
            method: Imputation method ('knn', 'min_prob', 'min', 'median')
            platform_type: Platform type ("mass_spec", "affinity", or "auto")
            save_result: Whether to save the imputed modality

        Returns:
            str: Imputation report with statistics
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            if platform_type == "auto":
                platform_config = _get_platform_for_modality(modality_name)
            else:
                platform_config = get_platform_config(platform_type)

            imputed_adata, impute_stats, impute_ir = (
                preprocessing_service.impute_missing_values(adata, method=method)
            )

            imputed_name = f"{modality_name}_imputed"
            data_manager.store_modality(
                name=imputed_name,
                adata=imputed_adata,
                parent_name=modality_name,
                step_summary=f"Imputed missing values using {method}",
            )

            if save_result:
                save_path = f"{modality_name}_imputed.h5ad"
                data_manager.save_modality(imputed_name, save_path)

            data_manager.log_tool_usage(
                tool_name="impute_missing_values",
                parameters={
                    "modality_name": modality_name,
                    "method": method,
                    "platform_type": platform_config.platform_type,
                },
                description=f"Imputed missing values using {method}",
                ir=impute_ir,
            )

            response = f"Successfully imputed missing values in '{modality_name}'!\n\n"
            response += f"**Platform:** {platform_config.display_name}\n"
            response += f"**Method:** {method}\n\n"

            if "n_imputed_values" in impute_stats:
                response += f"- Values imputed: {impute_stats['n_imputed_values']}\n"
            if "imputation_percentage" in impute_stats:
                response += f"- Percentage imputed: {impute_stats['imputation_percentage']:.1f}%\n"

            if platform_config.platform_type == "mass_spec" and method == "knn":
                response += "\n**Warning:** KNN imputation may not be ideal for MS data "
                response += "(MNAR pattern). Consider 'min_prob' for MNAR-appropriate imputation.\n"

            response += f"\n**New modality created**: '{imputed_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            analysis_results["details"]["imputation"] = response
            return response

        except Exception as e:
            logger.error(f"Error imputing missing values: {e}")
            return f"Error in imputation: {str(e)}"

    # -------------------------
    # SELECT VARIABLE PROTEINS TOOL
    # -------------------------
    @tool
    def select_variable_proteins(
        modality_name: str,
        n_top_proteins: int = 500,
        method: str = "cv",
        min_detection_rate: float = 0.5,
        save_result: bool = True,
    ) -> str:
        """
        Select highly variable proteins for downstream analysis.

        Analogous to highly variable gene selection in transcriptomics.
        Identifies proteins with the most biological variation across samples,
        filtering out low-detection proteins first.

        Args:
            modality_name: Name of the proteomics modality
            n_top_proteins: Number of top variable proteins to select (default: 500)
            method: Variability method ('cv' for coefficient of variation,
                    'variance' for raw variance, 'mad' for median absolute deviation)
            min_detection_rate: Minimum fraction of non-missing samples (default: 0.5)
            save_result: Whether to save the result

        Returns:
            str: Selection report with selected proteins
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            result_adata, stats, ir = quality_service.select_variable_proteins(
                adata,
                n_top_proteins=n_top_proteins,
                method=method,
                min_detection_rate=min_detection_rate,
            )

            result_name = f"{modality_name}_hvp_selected"
            data_manager.store_modality(
                name=result_name,
                adata=result_adata,
                parent_name=modality_name,
                step_summary=f"Variable protein selection ({method}): {stats['n_selected']} proteins",
            )

            if save_result:
                save_path = f"{modality_name}_hvp_selected.h5ad"
                data_manager.save_modality(result_name, save_path)

            data_manager.log_tool_usage(
                tool_name="select_variable_proteins",
                parameters={
                    "modality_name": modality_name,
                    "n_top_proteins": n_top_proteins,
                    "method": method,
                    "min_detection_rate": min_detection_rate,
                },
                description=f"Selected {stats['n_selected']} variable proteins using {method}",
                ir=ir,
            )

            response = f"Variable protein selection complete for '{modality_name}'!\n\n"
            response += f"**Method**: {method}\n"
            response += f"**Proteins selected**: {stats['n_selected']} / {stats['n_total']} "
            response += f"({stats['n_selected'] / stats['n_total'] * 100:.1f}%)\n"
            response += f"**Proteins passing detection filter**: {stats['n_passing_detection']}\n"
            response += f"**Min detection rate**: {min_detection_rate * 100:.0f}%\n"

            if stats.get("top_proteins"):
                response += f"\n**Top 10 variable proteins**: {', '.join(stats['top_proteins'][:10])}\n"

            response += f"\n**New modality created**: '{result_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n\n**Next steps**: analyze_proteomics_patterns() for PCA/clustering"

            analysis_results["details"]["variable_protein_selection"] = response
            return response

        except Exception as e:
            logger.error(f"Error selecting variable proteins: {e}")
            return f"Error in variable protein selection: {str(e)}"

    # -------------------------
    # SUMMARY TOOL
    # -------------------------
    @tool
    def create_proteomics_summary() -> str:
        """
        Create comprehensive summary of all proteomics analysis steps performed.

        Returns:
            str: Complete analysis summary
        """
        if not analysis_results["details"]:
            return "No proteomics analyses have been performed yet. Run some analysis tools first."

        summary = "# Proteomics Analysis Summary\n\n"

        for step, details in analysis_results["details"].items():
            summary += f"## {step.replace('_', ' ').title()}\n"
            summary += f"{details}\n\n"

        # Add current modality status
        modalities = data_manager.list_modalities()
        proteomics_terms = ["proteomics", "protein", "ms", "olink", "soma", "affinity"]
        proteomics_modalities = [
            m for m in modalities if any(term in m.lower() for term in proteomics_terms)
        ]
        summary += "## Current Proteomics Modalities\n"
        summary += f"Proteomics modalities: {', '.join(proteomics_modalities)}\n\n"

        analysis_results["summary"] = summary
        return summary

    # Return all shared tools
    return [
        check_proteomics_status,
        assess_proteomics_quality,
        filter_proteomics_data,
        normalize_proteomics_data,
        analyze_proteomics_patterns,
        impute_missing_values,
        select_variable_proteins,
        create_proteomics_summary,
    ]
