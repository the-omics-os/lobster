"""
Shared tools for proteomics analysis (mass spectrometry and affinity platforms).

This module provides tools shared by the proteomics parent agent and sub-agents.
Tools auto-detect platform type and use appropriate defaults.

Following the same factory pattern as transcriptomics shared_tools.py.
"""

from pathlib import Path
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
                logger.warning(
                    f"Platform type ambiguous for '{modality_name}'. Defaulting to mass_spec. Use platform_type parameter to override."
                )
                return get_platform_config("mass_spec")
            return get_platform_config(detected_type)
        except ValueError:
            # Default to mass_spec if modality not found
            return get_platform_config("mass_spec")

    # -------------------------
    # GENERIC MATRIX IMPORT HELPER (used by import_proteomics_data and import_affinity_data)
    # -------------------------
    def _import_generic_matrix(
        file_path: str,
        classification,
        sample_metadata_path: str = None,
        modality_name: str = "",
        save_result: bool = True,
        tool_name: str = "import_proteomics_data",
        modality_prefix: str = "proteomics",
    ) -> str:
        """Import a generic CSV/TSV expression matrix via ProteomicsAdapter.

        Uses the FileClassification's orientation info to auto-transpose if needed.
        """
        try:
            from lobster.core.adapters.proteomics_adapter import ProteomicsAdapter

            adapter = ProteomicsAdapter(data_type="mass_spectrometry")
            adata = adapter._load_csv_proteomics_data(
                path=file_path,
                orientation=classification.orientation,
                sample_metadata_path=sample_metadata_path,
            )

            # Generate modality name
            name = modality_name or f"{modality_prefix}_generic_{Path(file_path).stem}"

            # Tag as generic import in uns
            adata.uns["import_method"] = "generic_matrix"
            adata.uns["file_classification"] = {
                "format": classification.format,
                "orientation": classification.orientation,
                "n_rows": classification.n_rows,
                "n_cols": classification.n_cols,
                "confidence": classification.confidence,
            }

            data_manager.store_modality(
                name=name,
                adata=adata,
                step_summary=f"Imported generic expression matrix: {adata.n_obs} samples x {adata.n_vars} proteins",
            )

            if save_result:
                data_manager.save_modality(name, f"{name}.h5ad")

            ir = AnalysisStep(
                operation="proteomics.import.generic_matrix",
                tool_name=tool_name,
                description=f"Imported generic expression matrix from {Path(file_path).name}",
                library="lobster.core.adapters.proteomics_adapter",
                code_template="""# Import generic proteomics expression matrix
import pandas as pd
import anndata as ad

df = pd.read_csv({{ file_path | tojson }}, index_col=0)
{% if transpose %}df = df.T{% endif %}
adata = ad.AnnData(df.select_dtypes(include='number').values.astype('float32'),
                    obs=pd.DataFrame(index=df.index),
                    var=pd.DataFrame(index=df.columns))""",
                imports=["import pandas as pd", "import anndata as ad"],
                parameters={
                    "file_path": file_path,
                    "orientation": classification.orientation,
                    "transpose": classification.orientation == "features_as_rows",
                },
                parameter_schema={
                    "file_path": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value="",
                        required=True,
                        description="Path to proteomics expression matrix file",
                    ),
                    "orientation": ParameterSpec(
                        param_type="str",
                        papermill_injectable=False,
                        default_value="features_as_rows",
                        required=False,
                        description="Matrix orientation: features_as_rows or samples_as_rows",
                    ),
                    "transpose": ParameterSpec(
                        param_type="bool",
                        papermill_injectable=False,
                        default_value=True,
                        required=False,
                        description="Whether to transpose the matrix",
                    ),
                },
            )

            data_manager.log_tool_usage(
                tool_name=tool_name,
                parameters={
                    "file_path": file_path,
                    "format": "generic_matrix",
                    "orientation": classification.orientation,
                },
                description=f"Imported generic expression matrix from {Path(file_path).name}",
                ir=ir,
            )

            X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            missing_pct = np.isnan(X).sum() / X.size * 100 if X.size > 0 else 0

            response = f"Successfully imported generic expression matrix from '{Path(file_path).name}'!\n\n"
            response += f"**Format:** Generic CSV/TSV matrix\n"
            response += f"**Orientation:** {classification.orientation} (auto-detected)\n"
            response += f"**Samples:** {adata.n_obs}\n"
            response += f"**Proteins/Features:** {adata.n_vars}\n"
            response += f"**Missing values:** {missing_pct:.1f}%\n"

            if sample_metadata_path:
                response += f"**Metadata:** merged from {Path(sample_metadata_path).name}\n"

            response += (
                f"\n**Modality created:** '{name}'\n"
                "**Note:** Imported as generic expression matrix. Platform-specific QC features "
                "(contaminant filtering, peptide mapping) are not available for this format."
            )

            if save_result:
                response += f"\n**Saved to:** {name}.h5ad"

            return response

        except Exception as e:
            logger.error(f"Error importing generic matrix: {e}")
            return f"Error importing generic expression matrix: {str(e)}"

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

                        # AFP-06: LOD summary for affinity data
                        lod_col = None
                        for candidate in ["lod", "LOD", "Lod"]:
                            if candidate in adata.var.columns:
                                lod_col = candidate
                                break
                        if lod_col is not None:
                            n_with_lod = adata.var[lod_col].notna().sum()
                            response += f"- LOD info: available ({n_with_lod} proteins with LOD values)\n"
                        elif "lod_values" in adata.uns:
                            response += "- LOD info: available (stored in uns)\n"

                        # AFP-06: Bridge sample detection
                        bridge_indicators = ["bridge", "ref", "control_type"]
                        bridge_col_found = None
                        n_bridge = 0
                        for obs_col in adata.obs.columns:
                            if any(ind in obs_col.lower() for ind in bridge_indicators):
                                bridge_col_found = obs_col
                                try:
                                    n_bridge = int(
                                        adata.obs[obs_col].astype(bool).sum()
                                    )
                                except (ValueError, TypeError):
                                    n_bridge = 0
                                break
                        if bridge_col_found and n_bridge > 0:
                            response += f"- Bridge samples detected: {n_bridge} samples (column: {bridge_col_found})\n"

                        # AFP-06: Panel info
                        panel_info = adata.uns.get("assay_version") or adata.uns.get(
                            "panel_info"
                        )
                        platform_name = adata.uns.get("platform", "")
                        if panel_info:
                            response += f"- Panel info: {panel_info}\n"
                        elif platform_name:
                            response += f"- Platform: {platform_name}\n"

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
                # AFP-05: LOD-based metrics for affinity data
                lod_col = None
                for candidate in ["lod", "LOD", "Lod"]:
                    if candidate in final_adata.var.columns:
                        lod_col = candidate
                        break
                has_lod_uns = "lod_values" in final_adata.uns

                if lod_col is not None or has_lod_uns:
                    response += "\n**LOD Quality:**\n"
                    if lod_col is not None:
                        X_vals = (
                            final_adata.X.toarray()
                            if hasattr(final_adata.X, "toarray")
                            else final_adata.X
                        )
                        lod_values = final_adata.var[lod_col].values
                        n_below_lod = 0
                        n_proteins_with_lod = 0
                        total_comparisons = 0
                        for j in range(final_adata.n_vars):
                            lod_val = lod_values[j]
                            if not np.isnan(lod_val):
                                n_proteins_with_lod += 1
                                col_vals = X_vals[:, j]
                                valid_mask = ~np.isnan(col_vals)
                                below = (col_vals[valid_mask] < lod_val).sum()
                                n_below_lod += int(below)
                                total_comparisons += int(valid_mask.sum())

                        overall_below_pct = (
                            (n_below_lod / total_comparisons * 100)
                            if total_comparisons > 0
                            else 0
                        )
                        response += (
                            f"- Proteins with LOD values: {n_proteins_with_lod}\n"
                        )
                        response += (
                            f"- Overall below-LOD rate: {overall_below_pct:.1f}%\n"
                        )
                        response += f"- Measurements below LOD: {n_below_lod} / {total_comparisons}\n"
                    else:
                        response += "- LOD values available in uns (run assess_lod_quality for details)\n"
                    response += "- Run `assess_lod_quality` for detailed per-protein LOD analysis\n"

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
                response += (
                    "\n**Warning:** KNN imputation may not be ideal for MS data "
                )
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
            response += (
                f"**Proteins selected**: {stats['n_selected']} / {stats['n_total']} "
            )
            response += f"({stats['n_selected'] / stats['n_total'] * 100:.1f}%)\n"
            response += f"**Proteins passing detection filter**: {stats['n_passing_detection']}\n"
            response += f"**Min detection rate**: {min_detection_rate * 100:.0f}%\n"

            if stats.get("top_proteins"):
                response += f"\n**Top 10 variable proteins**: {', '.join(stats['top_proteins'][:10])}\n"

            response += f"\n**New modality created**: '{result_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += (
                "\n\n**Next steps**: analyze_proteomics_patterns() for PCA/clustering"
            )

            analysis_results["details"]["variable_protein_selection"] = response
            return response

        except Exception as e:
            logger.error(f"Error selecting variable proteins: {e}")
            return f"Error in variable protein selection: {str(e)}"

    # -------------------------
    # IMPORT PROTEOMICS DATA TOOL
    # -------------------------
    @tool
    def import_proteomics_data(
        file_path: str,
        software: str = "auto",
        intensity_type: str = "auto",
        filter_contaminants: bool = True,
        filter_reverse: bool = True,
        sample_metadata_path: str = "",
        modality_name: str = "",
        save_result: bool = True,
    ) -> str:
        """Import MS proteomics data from search engine outputs or generic expression matrices.

        Auto-detects the file format (MaxQuant, DIA-NN, Spectronaut) and uses the
        appropriate parser. Also handles generic CSV/TSV expression matrices (e.g.,
        preprocessed data from GEO) with automatic orientation detection.

        Args:
            file_path: Path to the proteomics output file (e.g., proteinGroups.txt, report.tsv, expression.csv)
            software: Parser to use ("auto", "maxquant", "diann", "spectronaut")
            intensity_type: Intensity column type ("auto", "lfq", "intensity", "maxlfq")
            filter_contaminants: Remove known contaminant proteins (default True)
            filter_reverse: Remove reverse database hits (default True)
            sample_metadata_path: Optional path to CSV with sample metadata to merge (for generic matrix import)
            modality_name: Name for the imported modality (auto-generated if empty)
            save_result: Whether to save the modality to disk

        Returns:
            str: Import summary with sample count, protein count, and data characteristics
        """
        try:
            from lobster.services.data_access.proteomics_parsers import (
                DIANNParser,
                MaxQuantParser,
                SpectronautParser,
                get_parser_for_file,
            )

            file_p = Path(file_path)
            if not file_p.exists():
                return f"File not found: {file_path}"

            parser = None
            classification = None

            if software == "auto":
                parser, classification = get_parser_for_file(file_path)

                # If no vendor parser but classified as generic matrix, use adapter path
                if parser is None and classification.format == "generic_matrix":
                    return _import_generic_matrix(
                        file_path=file_path,
                        classification=classification,
                        sample_metadata_path=sample_metadata_path or None,
                        modality_name=modality_name,
                        save_result=save_result,
                        tool_name="import_proteomics_data",
                        modality_prefix="proteomics",
                    )

                if parser is None:
                    diag = classification.diagnostics if classification else ""
                    return (
                        f"Could not identify format for '{file_p.name}'. {diag}\n\n"
                        "Supported vendor formats: MaxQuant proteinGroups.txt, DIA-NN report.tsv, "
                        "Spectronaut report. Generic CSV/TSV expression matrices are also supported.\n"
                        "Specify software='maxquant'/'diann'/'spectronaut' to force a specific parser."
                    )
            else:
                parser_map = {
                    "maxquant": MaxQuantParser,
                    "diann": DIANNParser,
                    "spectronaut": SpectronautParser,
                }
                parser_cls = parser_map.get(software)
                if parser_cls is None:
                    return (
                        f"Parser for '{software}' not available. "
                        f"Supported: {list(parser_map.keys())}. "
                        "Ensure lobster-proteomics is installed."
                    )
                parser = parser_cls()

            # Build parse kwargs based on what the parser accepts
            parse_kwargs = {}
            if intensity_type != "auto":
                parse_kwargs["intensity_type"] = intensity_type
            if hasattr(parser, "parse"):
                # Try passing filter flags
                parse_kwargs["filter_contaminants"] = filter_contaminants
                parse_kwargs["filter_reverse"] = filter_reverse

            try:
                result = parser.parse(file_path, **parse_kwargs)
            except TypeError:
                # Parser may not accept all kwargs; fall back to minimal call
                result = parser.parse(file_path)

            # Handle both 2-tuple and 3-tuple returns
            if isinstance(result, tuple) and len(result) == 3:
                adata, stats, ir = result
            elif isinstance(result, tuple) and len(result) == 2:
                adata, stats = result
                ir = AnalysisStep(
                    operation="proteomics.import.parse_file",
                    tool_name="import_proteomics_data",
                    description=f"Imported proteomics data using {parser.__class__.__name__}",
                    library="lobster.services.data_access.proteomics_parsers",
                    code_template="""# Import proteomics data
from lobster.services.data_access.proteomics_parsers import get_parser_for_file
parser, classification = get_parser_for_file({{ file_path | tojson }})
adata, stats = parser.parse({{ file_path | tojson }})""",
                    imports=[
                        "from lobster.services.data_access.proteomics_parsers import get_parser_for_file"
                    ],
                    parameters={"file_path": file_path, "software": software},
                )
            else:
                # Single AnnData return
                adata = result
                stats = {}
                ir = AnalysisStep(
                    operation="proteomics.import.parse_file",
                    tool_name="import_proteomics_data",
                    description=f"Imported proteomics data using {parser.__class__.__name__}",
                    library="lobster.services.data_access.proteomics_parsers",
                    code_template="""# Import proteomics data
from lobster.services.data_access.proteomics_parsers import get_parser_for_file
parser, classification = get_parser_for_file({{ file_path | tojson }})
adata = parser.parse({{ file_path | tojson }})""",
                    imports=[
                        "from lobster.services.data_access.proteomics_parsers import get_parser_for_file"
                    ],
                    parameters={"file_path": file_path, "software": software},
                )

            # Generate modality name if not provided
            parser_name = parser.__class__.__name__.replace("Parser", "").lower()
            name = modality_name or f"ms_{parser_name}_{Path(file_path).stem}"

            data_manager.store_modality(
                name=name,
                adata=adata,
                step_summary=f"Imported {parser.__class__.__name__} data: {adata.n_obs} samples x {adata.n_vars} proteins",
            )

            if save_result:
                save_path = f"{name}.h5ad"
                data_manager.save_modality(name, save_path)

            data_manager.log_tool_usage(
                tool_name="import_proteomics_data",
                parameters={
                    "file_path": file_path,
                    "software": software,
                    "intensity_type": intensity_type,
                    "filter_contaminants": filter_contaminants,
                    "filter_reverse": filter_reverse,
                },
                description=f"Imported proteomics data from {Path(file_path).name}",
                ir=ir,
            )

            # Build response
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            missing_pct = np.isnan(X).sum() / X.size * 100 if X.size > 0 else 0

            response = f"Successfully imported proteomics data from '{Path(file_path).name}'!\n\n"
            response += f"**Parser:** {parser.__class__.__name__}\n"
            response += f"**Samples:** {adata.n_obs}\n"
            response += f"**Proteins:** {adata.n_vars}\n"
            response += f"**Missing values:** {missing_pct:.1f}%\n"

            # Peptide mapping info
            peptide_cols = ["n_peptides", "unique_peptides", "sequence_coverage"]
            present_peptide_cols = [c for c in peptide_cols if c in adata.var.columns]
            if present_peptide_cols:
                response += f"\n**Peptide mapping columns:** {present_peptide_cols}\n"
                if "n_peptides" in adata.var.columns:
                    response += f"- Median peptides/protein: {adata.var['n_peptides'].median():.0f}\n"

            response += f"\n**Modality created:** '{name}'"
            if save_result:
                response += f"\n**Saved to:** {name}.h5ad"

            return response

        except Exception as e:
            logger.error(f"Error importing proteomics data: {e}")
            return f"Error importing proteomics data: {str(e)}"

    # -------------------------
    # IMPORT PTM SITES TOOL
    # -------------------------
    @tool
    def import_ptm_sites(
        file_path: str,
        ptm_type: str = "phospho",
        localization_threshold: float = 0.75,
        filter_contaminants: bool = True,
        modality_name: str = "",
        save_result: bool = True,
    ) -> str:
        """Import PTM site-level quantification data (phospho/acetyl/ubiquitin).

        Reads MaxQuant-style site-level output and creates a site-level AnnData.
        Sites are identified as gene_residuePosition (e.g., EGFR_Y1068).
        Only class I sites (localization probability >= threshold) are kept.

        Args:
            file_path: Path to PTM site file (e.g., Phospho(STY)Sites.txt)
            ptm_type: Type of PTM ("phospho", "acetyl", "ubiquitin")
            localization_threshold: Minimum localization probability (default 0.75)
            filter_contaminants: Remove contaminant proteins
            modality_name: Name for modality (auto-generated if empty)
            save_result: Whether to save to disk

        Returns:
            str: Import summary with site count, sample count, PTM type details
        """
        try:
            adata, stats, ir = preprocessing_service.import_ptm_site_data(
                file_path=file_path,
                ptm_type=ptm_type,
                localization_threshold=localization_threshold,
                filter_contaminants=filter_contaminants,
            )

            name = modality_name or f"ptm_{ptm_type}_{Path(file_path).stem}"

            data_manager.store_modality(
                name=name,
                adata=adata,
                step_summary=f"Imported {ptm_type} PTM sites: {adata.n_obs} samples x {adata.n_vars} sites",
            )

            if save_result:
                save_path = f"{name}.h5ad"
                data_manager.save_modality(name, save_path)

            data_manager.log_tool_usage(
                tool_name="import_ptm_sites",
                parameters={
                    "file_path": file_path,
                    "ptm_type": ptm_type,
                    "localization_threshold": localization_threshold,
                    "filter_contaminants": filter_contaminants,
                },
                description=f"Imported {ptm_type} PTM site data",
                ir=ir,
            )

            response = f"Successfully imported {ptm_type} PTM sites from '{Path(file_path).name}'!\n\n"
            response += f"**PTM type:** {ptm_type}\n"
            response += f"**Samples:** {adata.n_obs}\n"
            response += f"**Sites:** {adata.n_vars}\n"
            response += f"**Localization threshold:** {localization_threshold}\n"

            if "n_class_i_sites" in stats:
                response += f"**Class I sites (prob >= {localization_threshold}):** {stats['n_class_i_sites']}\n"
            if "n_total_sites" in stats:
                response += (
                    f"**Total sites before filtering:** {stats['n_total_sites']}\n"
                )

            response += f"\n**Modality created:** '{name}'"
            if save_result:
                response += f"\n**Saved to:** {name}.h5ad"

            return response

        except Exception as e:
            logger.error(f"Error importing PTM site data: {e}")
            return f"Error importing PTM sites: {str(e)}"

    # -------------------------
    # CORRECT BATCH EFFECTS TOOL
    # -------------------------
    @tool
    def correct_batch_effects(
        modality_name: str,
        batch_column: str = "batch",
        method: str = "combat",
        reference_batch: str = "",
        save_result: bool = True,
    ) -> str:
        """Correct batch effects in MS proteomics data using ComBat or median centering.

        For multi-batch MS experiments (different runs, instruments, or processing dates).
        This is distinct from correct_plate_effects which is specific to affinity platforms.

        Args:
            modality_name: Name of the proteomics modality
            batch_column: Column in obs containing batch identifiers
            method: Correction method ("combat", "median_centering", "reference_based")
            reference_batch: Reference batch for reference_based method (empty for auto)
            save_result: Whether to save corrected modality

        Returns:
            str: Batch correction report with before/after metrics
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            if batch_column not in adata.obs.columns:
                return (
                    f"Batch column '{batch_column}' not found in obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )

            corrected_adata, batch_stats, batch_ir = (
                preprocessing_service.correct_batch_effects(
                    adata,
                    batch_key=batch_column,
                    method=method,
                    reference_batch=reference_batch or None,
                )
            )

            corrected_name = f"{modality_name}_batch_corrected"
            data_manager.store_modality(
                name=corrected_name,
                adata=corrected_adata,
                parent_name=modality_name,
                step_summary=f"Batch corrected using {method}",
            )

            if save_result:
                save_path = f"{modality_name}_batch_corrected.h5ad"
                data_manager.save_modality(corrected_name, save_path)

            data_manager.log_tool_usage(
                tool_name="correct_batch_effects",
                parameters={
                    "modality_name": modality_name,
                    "batch_column": batch_column,
                    "method": method,
                    "reference_batch": reference_batch,
                },
                description=f"Corrected batch effects using {method}",
                ir=batch_ir,
            )

            response = f"Successfully corrected batch effects in '{modality_name}'!\n\n"
            response += f"**Method:** {method}\n"
            response += f"**Batch column:** {batch_column}\n"

            if "n_batches_corrected" in batch_stats:
                response += (
                    f"**Batches corrected:** {batch_stats['n_batches_corrected']}\n"
                )
            if "n_samples" in batch_stats:
                response += f"**Samples:** {batch_stats['n_samples']}\n"

            response += f"\n**New modality created:** '{corrected_name}'"
            if save_result:
                response += f"\n**Saved to:** {modality_name}_batch_corrected.h5ad"

            return response

        except Exception as e:
            logger.error(f"Error correcting batch effects: {e}")
            return f"Error in batch correction: {str(e)}"

    # -------------------------
    # SUMMARIZE PEPTIDE TO PROTEIN TOOL
    # -------------------------
    @tool
    def summarize_peptide_to_protein(
        modality_name: str,
        method: str = "median",
        protein_column: str = "protein_id",
        save_result: bool = True,
    ) -> str:
        """Roll up peptide/PSM-level quantification to protein-level.

        Required for TMT workflows where reporter ion intensities are at peptide/PSM level.
        Aggregates peptides to proteins using median (robust) or sum methods.

        Args:
            modality_name: Name of the peptide-level modality
            method: Aggregation method ("median", "sum")
            protein_column: Column in var mapping peptides to proteins
            save_result: Whether to save protein-level modality

        Returns:
            str: Summarization report with peptide-to-protein statistics
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            protein_adata, rollup_stats, rollup_ir = (
                preprocessing_service.summarize_peptide_to_protein(
                    adata,
                    method=method,
                    protein_column=protein_column,
                )
            )

            rollup_name = f"{modality_name}_protein_rollup"
            data_manager.store_modality(
                name=rollup_name,
                adata=protein_adata,
                parent_name=modality_name,
                step_summary=f"Peptide-to-protein rollup ({method}): {protein_adata.n_vars} proteins",
            )

            if save_result:
                save_path = f"{modality_name}_protein_rollup.h5ad"
                data_manager.save_modality(rollup_name, save_path)

            data_manager.log_tool_usage(
                tool_name="summarize_peptide_to_protein",
                parameters={
                    "modality_name": modality_name,
                    "method": method,
                    "protein_column": protein_column,
                },
                description=f"Rolled up peptides to proteins using {method}",
                ir=rollup_ir,
            )

            response = f"Successfully rolled up peptide data to protein level for '{modality_name}'!\n\n"
            response += f"**Aggregation method:** {method}\n"
            response += f"**Protein column:** {protein_column}\n"

            if "n_peptides" in rollup_stats:
                response += f"**Input peptides:** {rollup_stats['n_peptides']}\n"
            if "n_proteins" in rollup_stats:
                response += f"**Output proteins:** {rollup_stats['n_proteins']}\n"
            if "median_peptides_per_protein" in rollup_stats:
                response += f"**Median peptides/protein:** {rollup_stats['median_peptides_per_protein']:.1f}\n"

            response += f"\n**New modality created:** '{rollup_name}'"
            if save_result:
                response += f"\n**Saved to:** {modality_name}_protein_rollup.h5ad"

            return response

        except Exception as e:
            logger.error(f"Error in peptide-to-protein summarization: {e}")
            return f"Error in peptide-to-protein rollup: {str(e)}"

    # -------------------------
    # NORMALIZE PTM TO PROTEIN TOOL
    # -------------------------
    @tool
    def normalize_ptm_to_protein(
        ptm_modality_name: str,
        protein_modality_name: str,
        method: str = "ratio",
        save_result: bool = True,
    ) -> str:
        """Normalize PTM site abundances against total protein levels.

        Separates true PTM regulation from protein abundance changes.
        Essential for phosphoproteomics where increased phosphorylation
        may just reflect increased total protein.

        Args:
            ptm_modality_name: Name of the PTM site-level modality
            protein_modality_name: Name of the protein-level modality
            method: "ratio" (log subtraction) or "regression" (residuals)
            save_result: Whether to save normalized modality

        Returns:
            str: Normalization report with matching statistics
        """
        try:
            ptm_adata = data_manager.get_modality(ptm_modality_name)
        except ValueError:
            return f"PTM modality '{ptm_modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            protein_adata = data_manager.get_modality(protein_modality_name)
        except ValueError:
            return f"Protein modality '{protein_modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            normalized_adata, norm_stats, norm_ir = (
                preprocessing_service.normalize_ptm_to_protein(
                    ptm_adata=ptm_adata,
                    protein_adata=protein_adata,
                    method=method,
                )
            )

            normalized_name = f"{ptm_modality_name}_ptm_normalized"
            data_manager.store_modality(
                name=normalized_name,
                adata=normalized_adata,
                parent_name=ptm_modality_name,
                step_summary=f"PTM normalized against {protein_modality_name} ({method})",
            )

            if save_result:
                save_path = f"{ptm_modality_name}_ptm_normalized.h5ad"
                data_manager.save_modality(normalized_name, save_path)

            data_manager.log_tool_usage(
                tool_name="normalize_ptm_to_protein",
                parameters={
                    "ptm_modality_name": ptm_modality_name,
                    "protein_modality_name": protein_modality_name,
                    "method": method,
                },
                description=f"Normalized PTM sites against protein levels ({method})",
                ir=norm_ir,
            )

            response = f"Successfully normalized PTM sites against protein levels!\n\n"
            response += f"**PTM modality:** {ptm_modality_name}\n"
            response += f"**Protein modality:** {protein_modality_name}\n"
            response += f"**Method:** {method}\n"

            if "n_matched_sites" in norm_stats:
                response += f"**Matched sites:** {norm_stats['n_matched_sites']}\n"
            if "n_unmatched_sites" in norm_stats:
                response += f"**Unmatched sites (kept with raw values):** {norm_stats['n_unmatched_sites']}\n"
            if "n_total_sites" in norm_stats:
                response += f"**Total sites:** {norm_stats['n_total_sites']}\n"

            response += f"\n**New modality created:** '{normalized_name}'"
            if save_result:
                response += f"\n**Saved to:** {ptm_modality_name}_ptm_normalized.h5ad"

            return response

        except Exception as e:
            logger.error(f"Error normalizing PTM to protein: {e}")
            return f"Error in PTM-to-protein normalization: {str(e)}"

    # =========================================================================
    # AFFINITY PROTEOMICS TOOLS (AFP-01 through AFP-04)
    # =========================================================================

    # -------------------------
    # IMPORT AFFINITY DATA TOOL (AFP-01)
    # -------------------------
    @tool
    def import_affinity_data(
        file_path: str,
        platform: str = "auto",
        sample_metadata_path: str = None,
        modality_name: str = "",
        save_result: bool = True,
    ) -> str:
        """Import affinity proteomics data (Olink NPX, SomaScan ADAT, Luminex MFI).

        Auto-detects platform from file extension and content, or uses specified platform.
        Also handles generic CSV/TSV expression matrices with automatic orientation detection.
        Optionally merges external sample metadata from a CSV file.

        Args:
            file_path: Path to the affinity proteomics data file
            platform: Platform parser to use ("auto", "olink", "somascan", "luminex")
            sample_metadata_path: Optional path to CSV with additional sample metadata to merge
            modality_name: Name for the imported modality (auto-generated if empty)
            save_result: Whether to save the modality to disk

        Returns:
            str: Import summary with platform, sample count, analyte count, and data characteristics
        """
        try:
            import pandas as pd

            from lobster.services.data_access.proteomics_parsers import (
                FileClassifier,
                get_parser_for_file,
            )

            file_p = Path(file_path)
            if not file_p.exists():
                return f"File not found: {file_path}"

            parser = None
            detected_platform = platform

            if platform == "auto":
                # Use FileClassifier for initial classification
                classification = FileClassifier.classify(file_path)

                # Extension-based shortcuts for native formats
                if classification.format == "somascan_adat":
                    from lobster.services.data_access.somascan_parser import SomaScanParser
                    parser = SomaScanParser()
                    detected_platform = "somascan"
                elif classification.format == "olink_npx":
                    from lobster.services.data_access.olink_parser import OlinkParser
                    parser = OlinkParser()
                    detected_platform = "olink"
                elif classification.format == "luminex":
                    from lobster.services.data_access.luminex_parser import LuminexParser
                    parser = LuminexParser()
                    detected_platform = "luminex"
                elif classification.format == "generic_matrix":
                    # Generic expression matrix — use adapter path
                    return _import_generic_matrix(
                        file_path=file_path,
                        classification=classification,
                        sample_metadata_path=sample_metadata_path or None,
                        modality_name=modality_name,
                        save_result=save_result,
                        tool_name="import_affinity_data",
                        modality_prefix="affinity",
                    )
                else:
                    # Try vendor parser validation as fallback
                    from lobster.services.data_access.luminex_parser import LuminexParser
                    from lobster.services.data_access.olink_parser import OlinkParser
                    from lobster.services.data_access.somascan_parser import SomaScanParser

                    for parser_cls, pname in [
                        (OlinkParser, "olink"),
                        (SomaScanParser, "somascan"),
                        (LuminexParser, "luminex"),
                    ]:
                        try:
                            p = parser_cls()
                            if p.validate_file(file_path):
                                parser = p
                                detected_platform = pname
                                break
                        except Exception:
                            continue

                if parser is None:
                    diag = classification.diagnostics if classification else ""
                    return (
                        f"Could not auto-detect affinity platform for '{file_p.name}'. {diag}\n\n"
                        "Supported: Olink (.npx, .csv with NPX columns), "
                        "SomaScan (.adat), Luminex (.csv/.xlsx with MFI columns), "
                        "or any generic CSV/TSV expression matrix.\n"
                        "Specify platform='olink'/'somascan'/'luminex' to force."
                    )
            else:
                from lobster.services.data_access.luminex_parser import LuminexParser
                from lobster.services.data_access.olink_parser import OlinkParser
                from lobster.services.data_access.somascan_parser import SomaScanParser

                parser_map = {
                    "olink": OlinkParser,
                    "somascan": SomaScanParser,
                    "luminex": LuminexParser,
                }
                parser_cls = parser_map.get(platform.lower())
                if parser_cls is None:
                    return (
                        f"Unknown platform '{platform}'. "
                        f"Supported: {list(parser_map.keys())}"
                    )
                parser = parser_cls()
                detected_platform = platform.lower()

            # Parse the file
            result = parser.parse(file_path)
            if isinstance(result, tuple) and len(result) == 2:
                adata, stats = result
            elif isinstance(result, tuple) and len(result) == 3:
                adata, stats, _ = result
            else:
                adata = result
                stats = {}

            # Merge external sample metadata if provided
            metadata_merged = False
            n_metadata_cols = 0
            if sample_metadata_path:
                meta_path = Path(sample_metadata_path)
                if meta_path.exists():
                    try:
                        meta_df = pd.read_csv(meta_path, index_col=0)
                        # Match on index (sample IDs)
                        common_samples = adata.obs.index.intersection(meta_df.index)
                        if len(common_samples) > 0:
                            for col in meta_df.columns:
                                adata.obs[col] = meta_df.loc[
                                    adata.obs.index, col
                                ].values
                                n_metadata_cols += 1
                            metadata_merged = True
                        else:
                            logger.warning(
                                "No matching sample IDs between data and metadata file"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to merge sample metadata: {e}")

            # Generate modality name
            name = modality_name
            if not name:
                platform_prefix = detected_platform
                panel_info = adata.uns.get("assay_version", "")
                if panel_info and panel_info != "unknown":
                    name = f"{platform_prefix}_{panel_info}".replace(" ", "_").lower()
                else:
                    name = f"{platform_prefix}_{file_p.stem}".lower()

            data_manager.store_modality(
                name=name,
                adata=adata,
                step_summary=f"Imported {detected_platform} affinity data: {adata.n_obs} samples x {adata.n_vars} analytes",
            )

            if save_result:
                save_path = f"{name}.h5ad"
                data_manager.save_modality(name, save_path)

            # Create IR
            ir = AnalysisStep(
                operation=f"proteomics.affinity.import_{detected_platform}",
                tool_name="import_affinity_data",
                description=f"Imported {detected_platform} affinity proteomics data",
                library=f"lobster.services.data_access.{detected_platform}_parser",
                code_template=f"""# Import affinity proteomics data
from lobster.services.data_access.{detected_platform}_parser import {parser.__class__.__name__}
parser = {parser.__class__.__name__}()
adata, stats = parser.parse({{{{ file_path | tojson }}}})""",
                imports=[
                    f"from lobster.services.data_access.{detected_platform}_parser import {parser.__class__.__name__}"
                ],
                parameters={
                    "file_path": file_path,
                    "platform": detected_platform,
                },
            )

            data_manager.log_tool_usage(
                tool_name="import_affinity_data",
                parameters={
                    "file_path": file_path,
                    "platform": detected_platform,
                    "sample_metadata_path": sample_metadata_path,
                },
                description=f"Imported {detected_platform} affinity data from {file_p.name}",
                ir=ir,
            )

            # Build response
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            missing_pct = np.isnan(X).sum() / X.size * 100 if X.size > 0 else 0

            response = f"Successfully imported affinity proteomics data from '{file_p.name}'!\n\n"
            response += f"**Platform:** {detected_platform.capitalize()}\n"
            response += f"**Samples:** {adata.n_obs}\n"
            response += f"**Analytes:** {adata.n_vars}\n"
            response += f"**Missing values:** {missing_pct:.1f}%\n"

            if metadata_merged:
                response += f"**Metadata merged:** {n_metadata_cols} columns from {Path(sample_metadata_path).name}\n"

            response += f"\n**Modality created:** '{name}'"
            if save_result:
                response += f"\n**Saved to:** {name}.h5ad"

            return response

        except Exception as e:
            logger.error(f"Error importing affinity data: {e}")
            return f"Error importing affinity data: {str(e)}"

    # -------------------------
    # ASSESS LOD QUALITY TOOL (AFP-02)
    # -------------------------
    @tool
    def assess_lod_quality(
        modality_name: str,
        lod_column: str = "LOD",
        max_below_lod_pct: float = 50.0,
    ) -> str:
        """Assess limit of detection (LOD) quality for affinity proteomics data.

        Computes per-protein below-LOD percentages and flags proteins exceeding
        the threshold. Works with Olink (NPX LOD), SomaScan (RFU near-zero),
        and Luminex (MFI vs blank) data.

        Args:
            modality_name: Name of the affinity proteomics modality
            lod_column: Column in var containing LOD values (default "LOD")
            max_below_lod_pct: Maximum acceptable percentage of samples below LOD (default 50%)

        Returns:
            str: LOD quality report with per-protein statistics and recommendations
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            import pandas as pd

            adata_result = adata.copy()
            X = (
                adata_result.X.toarray()
                if hasattr(adata_result.X, "toarray")
                else adata_result.X
            )

            platform = adata.uns.get("platform", "unknown")

            # Find LOD values
            lod_values = None
            lod_source = None

            # Check var columns
            for candidate in [lod_column, lod_column.lower(), "lod", "LOD", "Lod"]:
                if candidate in adata_result.var.columns:
                    lod_values = pd.to_numeric(
                        adata_result.var[candidate], errors="coerce"
                    ).values
                    lod_source = f"var['{candidate}']"
                    break

            # Check uns
            if lod_values is None and "lod_values" in adata.uns:
                lod_values = np.array(adata.uns["lod_values"], dtype=float)
                lod_source = "uns['lod_values']"

            # Platform-specific LOD handling
            if lod_values is None:
                if platform == "somascan":
                    # SomaScan: flag near-zero RFU as below LOD
                    # Use 5th percentile of non-zero values as LOD proxy
                    lod_values = np.zeros(adata_result.n_vars)
                    for j in range(adata_result.n_vars):
                        col = X[:, j]
                        nonzero = col[~np.isnan(col) & (col > 0)]
                        if len(nonzero) > 0:
                            lod_values[j] = np.percentile(nonzero, 5)
                        else:
                            lod_values[j] = np.nan
                    lod_source = "estimated (5th percentile of non-zero RFU)"
                elif platform == "luminex":
                    # Luminex: use global 1st percentile as LOD proxy
                    valid_vals = X[~np.isnan(X)]
                    if len(valid_vals) > 0:
                        global_lod = (
                            np.percentile(valid_vals[valid_vals > 0], 1)
                            if np.any(valid_vals > 0)
                            else 0
                        )
                        lod_values = np.full(adata_result.n_vars, global_lod)
                    else:
                        lod_values = np.full(adata_result.n_vars, np.nan)
                    lod_source = "estimated (1st percentile of positive MFI)"
                else:
                    return (
                        f"No LOD information found for modality '{modality_name}'. "
                        f"Checked var columns and uns. "
                        f"For Olink data, LOD should be in var['{lod_column}']. "
                        "Import data with LOD information or specify the correct lod_column."
                    )

            # Compute per-protein below-LOD percentage
            below_lod_pct = np.zeros(adata_result.n_vars)
            for j in range(adata_result.n_vars):
                lod_val = lod_values[j]
                if np.isnan(lod_val):
                    below_lod_pct[j] = 0.0
                    continue
                col = X[:, j]
                valid_mask = ~np.isnan(col)
                if valid_mask.sum() == 0:
                    below_lod_pct[j] = 100.0
                else:
                    below_lod_pct[j] = (
                        (col[valid_mask] < lod_val).sum() / valid_mask.sum() * 100
                    )

            # Store results in adata
            adata_result.var["below_lod_pct"] = below_lod_pct
            adata_result.var["lod_pass"] = below_lod_pct <= max_below_lod_pct

            n_passing = int((below_lod_pct <= max_below_lod_pct).sum())
            n_flagged = adata_result.n_vars - n_passing
            median_below = float(np.median(below_lod_pct))

            # Find worst proteins
            worst_indices = np.argsort(below_lod_pct)[::-1][:10]
            worst_proteins = [
                (str(adata_result.var_names[i]), float(below_lod_pct[i]))
                for i in worst_indices
                if below_lod_pct[i] > 0
            ]

            # Store result modality
            result_name = f"{modality_name}_lod_assessed"
            data_manager.store_modality(
                name=result_name,
                adata=adata_result,
                parent_name=modality_name,
                step_summary=f"LOD assessed: {n_passing} passing, {n_flagged} flagged",
            )

            # Create IR
            ir = AnalysisStep(
                operation="proteomics.affinity.assess_lod_quality",
                tool_name="assess_lod_quality",
                description=f"LOD quality assessment for {platform} data",
                library="lobster.agents.proteomics.shared_tools",
                code_template="""# LOD quality assessment
import numpy as np
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
lod_values = adata.var['{{ lod_column }}'].values
below_lod_pct = np.zeros(adata.n_vars)
for j in range(adata.n_vars):
    col = X[:, j]
    valid = ~np.isnan(col)
    if valid.sum() > 0 and not np.isnan(lod_values[j]):
        below_lod_pct[j] = (col[valid] < lod_values[j]).sum() / valid.sum() * 100
adata.var['below_lod_pct'] = below_lod_pct
adata.var['lod_pass'] = below_lod_pct <= {{ max_below_lod_pct }}""",
                imports=["import numpy as np"],
                parameters={
                    "lod_column": lod_column,
                    "max_below_lod_pct": max_below_lod_pct,
                },
                parameter_schema={
                    "max_below_lod_pct": ParameterSpec(
                        param_type="float",
                        papermill_injectable=True,
                        default_value=50.0,
                        required=False,
                        validation_rule="0 < max_below_lod_pct <= 100",
                        description="Maximum acceptable below-LOD percentage per protein",
                    ),
                },
                input_entities=["adata"],
                output_entities=["adata_lod_assessed"],
            )

            data_manager.log_tool_usage(
                tool_name="assess_lod_quality",
                parameters={
                    "modality_name": modality_name,
                    "lod_column": lod_column,
                    "max_below_lod_pct": max_below_lod_pct,
                },
                description=f"LOD quality assessment: {n_passing} passing, {n_flagged} flagged",
                ir=ir,
            )

            # Build response
            response = f"LOD Quality Assessment for '{modality_name}':\n\n"
            response += f"**Platform:** {platform}\n"
            response += f"**LOD source:** {lod_source}\n"
            response += f"**LOD threshold:** {max_below_lod_pct}% below-LOD maximum\n\n"

            response += "**Summary:**\n"
            response += (
                f"- Proteins passing LOD filter: {n_passing} / {adata_result.n_vars}\n"
            )
            response += (
                f"- Proteins flagged (>{max_below_lod_pct}% below LOD): {n_flagged}\n"
            )
            response += f"- Median below-LOD percentage: {median_below:.1f}%\n\n"

            if worst_proteins:
                response += "**Top flagged proteins (highest below-LOD %):**\n"
                for prot_name, prot_pct in worst_proteins[:10]:
                    response += f"- {prot_name}: {prot_pct:.1f}% below LOD\n"

            response += f"\n**New modality created:** '{result_name}'"

            if n_flagged > 0:
                response += f"\n\n**Recommendation:** Consider filtering {n_flagged} flagged proteins "
                response += "before downstream analysis, or use filter_proteomics_data with LOD-aware settings."

            analysis_results["details"]["lod_assessment"] = response
            return response

        except Exception as e:
            logger.error(f"Error in LOD quality assessment: {e}")
            return f"Error in LOD assessment: {str(e)}"

    # -------------------------
    # NORMALIZE BRIDGE SAMPLES TOOL (AFP-03)
    # -------------------------
    @tool
    def normalize_bridge_samples(
        modality_name: str,
        bridge_column: str = "is_bridge",
        plate_column: str = "plate_id",
        remove_bridges: bool = True,
        save_result: bool = True,
    ) -> str:
        """Normalize affinity proteomics data across plates using bridge/reference samples.

        Bridge samples are run on every plate and used to compute plate-specific
        normalization factors. For each plate, the correction factor per protein
        is: global_bridge_median - plate_bridge_median (in log or linear space).

        Args:
            modality_name: Name of the affinity proteomics modality
            bridge_column: Column in obs indicating bridge samples (boolean or 0/1)
            plate_column: Column in obs containing plate identifiers
            remove_bridges: Remove bridge samples from final output (default True)
            save_result: Whether to save the normalized modality

        Returns:
            str: Normalization report with plate factors and correction statistics
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            import pandas as pd

            # Validate required columns
            if bridge_column not in adata.obs.columns:
                return (
                    f"Bridge column '{bridge_column}' not found in obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )
            if plate_column not in adata.obs.columns:
                return (
                    f"Plate column '{plate_column}' not found in obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )

            adata_result = adata.copy()
            X = (
                adata_result.X.toarray()
                if hasattr(adata_result.X, "toarray")
                else adata_result.X.copy()
            )

            # Identify bridge samples
            bridge_mask = adata_result.obs[bridge_column].astype(bool).values
            plates = adata_result.obs[plate_column].unique()
            n_bridges = int(bridge_mask.sum())

            if n_bridges == 0:
                return (
                    f"No bridge samples found in column '{bridge_column}'. "
                    "Check that bridge samples are correctly labeled."
                )

            if len(plates) < 2:
                return (
                    f"Only {len(plates)} plate(s) found. "
                    "Bridge normalization requires at least 2 plates."
                )

            # Detect if data is log-transformed (Olink NPX is log2, others may be linear)
            platform = adata.uns.get("platform", "")
            is_log_space = platform == "olink"  # Olink NPX is already log2
            if not is_log_space:
                # Check if data looks log-transformed (values in typical log range)
                valid_vals = X[~np.isnan(X)]
                if len(valid_vals) > 0 and np.median(valid_vals) < 30:
                    is_log_space = True

            # Compute per-plate bridge medians for each protein
            plate_factors = {}
            global_bridge_medians = np.nanmedian(X[bridge_mask, :], axis=0)

            for plate in plates:
                plate_mask = (adata_result.obs[plate_column] == plate).values
                plate_bridge_mask = plate_mask & bridge_mask

                if plate_bridge_mask.sum() == 0:
                    logger.warning(f"No bridge samples on plate '{plate}', skipping")
                    plate_factors[str(plate)] = np.zeros(X.shape[1])
                    continue

                plate_bridge_medians = np.nanmedian(X[plate_bridge_mask, :], axis=0)

                if is_log_space:
                    # In log space: additive correction
                    factors = global_bridge_medians - plate_bridge_medians
                else:
                    # In linear space: multiplicative correction (as ratio)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        factors = global_bridge_medians - plate_bridge_medians
                    factors = np.nan_to_num(factors, nan=0.0)

                plate_factors[str(plate)] = factors

                # Apply correction to all samples on this plate
                sample_mask = plate_mask & (
                    ~bridge_mask
                    if remove_bridges
                    else np.ones_like(plate_mask, dtype=bool)
                )
                X[plate_mask, :] = X[plate_mask, :] + factors

            # Remove bridge samples if requested
            if remove_bridges:
                non_bridge_mask = ~bridge_mask
                adata_result = adata_result[non_bridge_mask, :].copy()
                X = X[non_bridge_mask, :]

            adata_result.X = X.astype(np.float32)

            # Store plate factors in uns
            adata_result.uns["bridge_normalization"] = {
                "bridge_column": bridge_column,
                "plate_column": plate_column,
                "n_plates": len(plates),
                "n_bridge_samples": n_bridges,
                "is_log_space": is_log_space,
                "bridges_removed": remove_bridges,
            }

            # Store result
            result_name = f"{modality_name}_bridge_normalized"
            data_manager.store_modality(
                name=result_name,
                adata=adata_result,
                parent_name=modality_name,
                step_summary=f"Bridge normalized: {len(plates)} plates, {n_bridges} bridge samples",
            )

            if save_result:
                save_path = f"{modality_name}_bridge_normalized.h5ad"
                data_manager.save_modality(result_name, save_path)

            # Create IR
            ir = AnalysisStep(
                operation="proteomics.affinity.normalize_bridge_samples",
                tool_name="normalize_bridge_samples",
                description="Normalize across plates using bridge sample medians",
                library="lobster.agents.proteomics.shared_tools",
                code_template="""# Bridge sample normalization
import numpy as np

bridge_mask = adata.obs['{{ bridge_column }}'].astype(bool).values
plates = adata.obs['{{ plate_column }}'].unique()
X = adata.X.copy()

global_bridge_medians = np.nanmedian(X[bridge_mask, :], axis=0)
for plate in plates:
    plate_mask = (adata.obs['{{ plate_column }}'] == plate).values
    plate_bridge = plate_mask & bridge_mask
    if plate_bridge.sum() > 0:
        plate_medians = np.nanmedian(X[plate_bridge, :], axis=0)
        factors = global_bridge_medians - plate_medians
        X[plate_mask, :] += factors

adata.X = X""",
                imports=["import numpy as np"],
                parameters={
                    "bridge_column": bridge_column,
                    "plate_column": plate_column,
                    "remove_bridges": remove_bridges,
                },
            )

            data_manager.log_tool_usage(
                tool_name="normalize_bridge_samples",
                parameters={
                    "modality_name": modality_name,
                    "bridge_column": bridge_column,
                    "plate_column": plate_column,
                    "remove_bridges": remove_bridges,
                },
                description=f"Bridge normalization: {len(plates)} plates, {n_bridges} bridges",
                ir=ir,
            )

            # Compute correction magnitude summary
            all_factors = np.concatenate(list(plate_factors.values()))
            median_correction = (
                float(np.median(np.abs(all_factors[~np.isnan(all_factors)])))
                if len(all_factors) > 0
                else 0
            )

            response = (
                f"Successfully normalized '{modality_name}' using bridge samples!\n\n"
            )
            response += "**Bridge Normalization Results:**\n"
            response += f"- Plates: {len(plates)}\n"
            response += f"- Bridge samples: {n_bridges}\n"
            response += f"- Correction space: {'log' if is_log_space else 'linear'}\n"
            response += f"- Median correction magnitude: {median_correction:.3f}\n"
            response += f"- Bridges removed from output: {remove_bridges}\n"

            if remove_bridges:
                response += f"- Final samples: {adata_result.n_obs} (after removing {n_bridges} bridges)\n"

            response += "\n**Per-Plate Factors (median absolute correction):**\n"
            for plate, factors in plate_factors.items():
                plate_median = (
                    float(np.median(np.abs(factors[~np.isnan(factors)])))
                    if len(factors) > 0
                    else 0
                )
                response += f"- Plate {plate}: {plate_median:.3f}\n"

            response += f"\n**New modality created:** '{result_name}'"
            if save_result:
                response += f"\n**Saved to:** {modality_name}_bridge_normalized.h5ad"

            analysis_results["details"]["bridge_normalization"] = response
            return response

        except Exception as e:
            logger.error(f"Error in bridge sample normalization: {e}")
            return f"Error in bridge normalization: {str(e)}"

    # -------------------------
    # ASSESS CROSS-PLATFORM CONCORDANCE TOOL (AFP-04)
    # -------------------------
    @tool
    def assess_cross_platform_concordance(
        modality_name_1: str,
        modality_name_2: str,
        method: str = "spearman",
    ) -> str:
        """Assess concordance between two proteomics modalities from different platforms.

        Computes protein-level correlations for proteins measured on both platforms.
        Useful for comparing Olink vs SomaScan, MS vs affinity, or any cross-platform validation.

        Args:
            modality_name_1: Name of the first proteomics modality
            modality_name_2: Name of the second proteomics modality
            method: Correlation method ("spearman" or "pearson")

        Returns:
            str: Concordance report with per-protein correlations and summary statistics
        """
        try:
            adata1 = data_manager.get_modality(modality_name_1)
        except ValueError:
            return f"Modality '{modality_name_1}' not found. Available: {data_manager.list_modalities()}"

        try:
            adata2 = data_manager.get_modality(modality_name_2)
        except ValueError:
            return f"Modality '{modality_name_2}' not found. Available: {data_manager.list_modalities()}"

        try:
            import pandas as pd
            from scipy import stats as scipy_stats

            # Find overlapping proteins by var_names
            proteins_1 = set(adata1.var_names)
            proteins_2 = set(adata2.var_names)
            overlapping = sorted(proteins_1 & proteins_2)

            # If no direct overlap, try gene symbol matching (strip suffixes)
            if len(overlapping) == 0:
                # Try matching by stripping common platform suffixes
                def clean_name(name):
                    return name.split("_")[0].split(".")[0].upper()

                name_map_1 = {clean_name(n): n for n in proteins_1}
                name_map_2 = {clean_name(n): n for n in proteins_2}
                common_clean = sorted(set(name_map_1.keys()) & set(name_map_2.keys()))

                if len(common_clean) == 0:
                    return (
                        f"No overlapping proteins found between '{modality_name_1}' "
                        f"({len(proteins_1)} proteins) and '{modality_name_2}' "
                        f"({len(proteins_2)} proteins). "
                        "Check that protein/gene names are compatible across platforms."
                    )

                # Build matched pairs
                overlapping_pairs = [
                    (name_map_1[c], name_map_2[c]) for c in common_clean
                ]
            else:
                overlapping_pairs = [(p, p) for p in overlapping]

            # Find matching samples
            samples_1 = set(adata1.obs.index)
            samples_2 = set(adata2.obs.index)
            common_samples = sorted(samples_1 & samples_2)

            if len(common_samples) < 3:
                return (
                    f"Insufficient matching samples between modalities. "
                    f"Found {len(common_samples)} matching samples (need >= 3). "
                    f"Modality 1 has {len(samples_1)} samples, modality 2 has {len(samples_2)} samples. "
                    "Ensure sample IDs match between modalities."
                )

            # Compute per-protein correlations
            X1 = adata1[common_samples, :].X
            X2 = adata2[common_samples, :].X
            X1 = X1.toarray() if hasattr(X1, "toarray") else X1
            X2 = X2.toarray() if hasattr(X2, "toarray") else X2

            correlations = []
            for p1_name, p2_name in overlapping_pairs:
                idx1 = list(adata1.var_names).index(p1_name)
                idx2 = list(adata2.var_names).index(p2_name)

                vals1 = X1[:, idx1]
                vals2 = X2[:, idx2]

                # Remove NaN pairs
                valid = ~np.isnan(vals1) & ~np.isnan(vals2)
                if valid.sum() < 3:
                    continue

                if method == "spearman":
                    r, p = scipy_stats.spearmanr(vals1[valid], vals2[valid])
                else:
                    r, p = scipy_stats.pearsonr(vals1[valid], vals2[valid])

                correlations.append(
                    {
                        "protein": p1_name,
                        "protein_2": p2_name,
                        "correlation": float(r) if not np.isnan(r) else 0.0,
                        "p_value": float(p) if not np.isnan(p) else 1.0,
                        "n_samples": int(valid.sum()),
                    }
                )

            if not correlations:
                return (
                    f"Could not compute correlations. "
                    f"Matching proteins: {len(overlapping_pairs)}, matching samples: {len(common_samples)}. "
                    "Insufficient valid (non-NaN) paired data."
                )

            corr_df = pd.DataFrame(correlations)
            median_corr = float(corr_df["correlation"].median())
            n_concordant = int((corr_df["correlation"] > 0.5).sum())
            n_discordant = int((corr_df["correlation"] < 0.3).sum())
            n_total = len(corr_df)

            # Store concordance results in first modality
            adata1_updated = adata1.copy()
            adata1_updated.uns["cross_platform_concordance"] = {
                "compared_with": modality_name_2,
                "method": method,
                "n_overlapping_proteins": n_total,
                "median_correlation": median_corr,
                "n_concordant": n_concordant,
                "n_discordant": n_discordant,
                "n_matching_samples": len(common_samples),
            }
            data_manager.store_modality(
                name=modality_name_1,
                adata=adata1_updated,
                step_summary=f"Cross-platform concordance: median r={median_corr:.3f}",
            )

            # Create IR
            ir = AnalysisStep(
                operation="proteomics.affinity.cross_platform_concordance",
                tool_name="assess_cross_platform_concordance",
                description=f"Cross-platform concordance: {modality_name_1} vs {modality_name_2}",
                library="lobster.agents.proteomics.shared_tools",
                code_template="""# Cross-platform concordance
from scipy import stats
import numpy as np

common_samples = sorted(set(adata1.obs.index) & set(adata2.obs.index))
common_proteins = sorted(set(adata1.var_names) & set(adata2.var_names))

correlations = []
for protein in common_proteins:
    v1 = adata1[common_samples, protein].X.flatten()
    v2 = adata2[common_samples, protein].X.flatten()
    valid = ~np.isnan(v1) & ~np.isnan(v2)
    if valid.sum() >= 3:
        r, p = stats.{{ method }}r(v1[valid], v2[valid])
        correlations.append({'protein': protein, 'r': r, 'p': p})""",
                imports=["from scipy import stats", "import numpy as np"],
                parameters={
                    "modality_name_1": modality_name_1,
                    "modality_name_2": modality_name_2,
                    "method": method,
                },
            )

            data_manager.log_tool_usage(
                tool_name="assess_cross_platform_concordance",
                parameters={
                    "modality_name_1": modality_name_1,
                    "modality_name_2": modality_name_2,
                    "method": method,
                },
                description=f"Cross-platform concordance: median r={median_corr:.3f}",
                ir=ir,
            )

            # Sort by correlation for reporting
            corr_df_sorted = corr_df.sort_values("correlation", ascending=False)
            top_concordant = corr_df_sorted.head(5)
            top_discordant = corr_df_sorted.tail(5)

            # Build response
            platform1 = adata1.uns.get("platform", "unknown")
            platform2 = adata2.uns.get("platform", "unknown")

            response = f"Cross-Platform Concordance: '{modality_name_1}' vs '{modality_name_2}'\n\n"
            response += f"**Platforms:** {platform1} vs {platform2}\n"
            response += f"**Method:** {method.capitalize()} correlation\n"
            response += f"**Matching samples:** {len(common_samples)}\n"
            response += f"**Overlapping proteins:** {n_total}\n\n"

            response += "**Summary:**\n"
            response += f"- Median correlation: {median_corr:.3f}\n"
            response += f"- Concordant proteins (r > 0.5): {n_concordant} ({n_concordant / n_total * 100:.0f}%)\n"
            response += f"- Discordant proteins (r < 0.3): {n_discordant} ({n_discordant / n_total * 100:.0f}%)\n\n"

            response += "**Top concordant proteins:**\n"
            for _, row in top_concordant.iterrows():
                response += f"- {row['protein']}: r={row['correlation']:.3f}\n"

            response += "\n**Top discordant proteins:**\n"
            for _, row in top_discordant.iterrows():
                response += f"- {row['protein']}: r={row['correlation']:.3f}\n"

            if median_corr > 0.7:
                response += "\n**Assessment:** Good cross-platform concordance."
            elif median_corr > 0.5:
                response += "\n**Assessment:** Moderate concordance. Some platform-specific effects may be present."
            else:
                response += "\n**Assessment:** Low concordance. Investigate platform-specific biases before combining data."

            analysis_results["details"]["cross_platform_concordance"] = response
            return response

        except Exception as e:
            logger.error(f"Error in cross-platform concordance: {e}")
            return f"Error in concordance assessment: {str(e)}"

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
        # Status & QC
        check_proteomics_status,
        assess_proteomics_quality,
        # Import (MS + affinity)
        import_proteomics_data,
        import_ptm_sites,
        import_affinity_data,
        # Filtering & preprocessing
        filter_proteomics_data,
        normalize_proteomics_data,
        correct_batch_effects,
        # Rollup & PTM normalization
        summarize_peptide_to_protein,
        normalize_ptm_to_protein,
        # Affinity-specific tools
        assess_lod_quality,
        normalize_bridge_samples,
        assess_cross_platform_concordance,
        # Analysis
        analyze_proteomics_patterns,
        impute_missing_values,
        select_variable_proteins,
        # Summary
        create_proteomics_summary,
    ]
