"""
Shared tools for transcriptomics analysis (single-cell and bulk RNA-seq).

This module provides tools that are shared by both single-cell and bulk RNA-seq
analysis agents. Tools auto-detect data type and use appropriate defaults.

Scientific note: These tools follow Scanpy best practices and standard bioinformatics
conventions. The auto-detection heuristics are designed to be robust across common
dataset characteristics.
"""

from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import tool

from lobster.agents.transcriptomics.config import detect_data_type, get_qc_defaults
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.quality.preprocessing_service import PreprocessingService
from lobster.services.quality.quality_service import QualityError, QualityService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SHARED TOOL FACTORY
# =============================================================================


def create_shared_tools(
    data_manager: DataManagerV2,
    quality_service: QualityService,
    preprocessing_service: PreprocessingService,
    clustering_service=None,
) -> List[Callable]:
    """
    Create shared transcriptomics tools with auto-detection.

    These tools are shared between single-cell and bulk RNA-seq analysis agents.
    Each tool auto-detects the data type and applies appropriate defaults when
    parameters are not explicitly specified.

    Args:
        data_manager: DataManagerV2 instance for modality management
        quality_service: QualityService instance for QC operations
        preprocessing_service: PreprocessingService for filtering/normalization
        clustering_service: Optional ClusteringService for PCA/embedding tools.
            If None, PCA and embedding tools are not included.

    Returns:
        List of tool functions to be added to agent tools
    """
    # Analysis results storage (shared across tools)
    analysis_results: Dict[str, Any] = {"summary": "", "details": {}}

    # -------------------------
    # DATA STATUS TOOL
    # -------------------------
    @tool
    def check_data_status(modality_name: str = "") -> str:
        """
        Check if transcriptomics data is loaded and ready for analysis.

        Auto-detects whether data is single-cell or bulk RNA-seq and provides
        appropriate status information including:
        - Data dimensions (observations x features)
        - Available metadata columns
        - Quality metrics if calculated
        - Data type classification

        Args:
            modality_name: Name of specific modality to check. If empty,
                          lists all available transcriptomics modalities.

        Returns:
            Formatted status report string
        """
        try:
            if modality_name == "":
                modalities = data_manager.list_modalities()
                if not modalities:
                    return (
                        "No modalities loaded. Please ask the data expert to load "
                        "a transcriptomics dataset first."
                    )

                response = f"Available modalities ({len(modalities)}):\n"
                for mod_name in modalities:
                    adata = data_manager.get_modality(mod_name)
                    data_type = detect_data_type(adata)
                    data_type_label = (
                        "cells" if data_type == "single_cell" else "samples"
                    )
                    response += (
                        f"- **{mod_name}**: {adata.n_obs} {data_type_label} x "
                        f"{adata.n_vars} genes [{data_type}]\n"
                    )

                return response

            else:
                # Check specific modality
                if modality_name not in data_manager.list_modalities():
                    return (
                        f"Modality '{modality_name}' not found. "
                        f"Available: {data_manager.list_modalities()}"
                    )

                adata = data_manager.get_modality(modality_name)
                metrics = data_manager.get_quality_metrics(modality_name)
                data_type = detect_data_type(adata)
                obs_label = "cells" if data_type == "single_cell" else "samples"

                response = (
                    f"Modality '{modality_name}' ready for analysis:\n"
                    f"- **Data type**: {data_type.replace('_', ' ').title()}\n"
                    f"- **Shape**: {adata.n_obs} {obs_label} x {adata.n_vars} genes\n"
                    f"- **Obs columns**: {list(adata.obs.columns)}\n"
                    f"- **Var columns**: {list(adata.var.columns)[:5]}...\n"
                )

                if "total_counts" in metrics:
                    response += f"- **Total counts**: {metrics['total_counts']:,.0f}\n"
                if "mean_counts_per_obs" in metrics:
                    response += (
                        f"- **Mean counts/{obs_label[:-1]}**: "
                        f"{metrics['mean_counts_per_obs']:.1f}\n"
                    )

                # Add data-type specific checks
                if data_type == "single_cell":
                    if adata.n_obs > 1000:
                        response += (
                            f"- **Dataset size**: Large ({adata.n_obs:,} cells) - "
                            "suitable for clustering\n"
                        )
                    elif adata.n_obs > 100:
                        response += (
                            f"- **Dataset size**: Medium ({adata.n_obs:,} cells) - "
                            "good for analysis\n"
                        )
                    else:
                        response += (
                            f"- **Dataset size**: Small ({adata.n_obs:,} cells) - "
                            "may limit analysis\n"
                        )
                else:  # bulk
                    if adata.n_obs < 6:
                        response += (
                            f"- **Sample size**: Small ({adata.n_obs} samples) - "
                            "may limit statistical power\n"
                        )
                    elif adata.n_obs < 20:
                        response += (
                            f"- **Sample size**: Moderate ({adata.n_obs} samples) - "
                            "good for analysis\n"
                        )
                    else:
                        response += (
                            f"- **Sample size**: Large ({adata.n_obs} samples) - "
                            "excellent statistical power\n"
                        )

                analysis_results["details"]["data_status"] = response
                return response

        except Exception as e:
            logger.error(f"Error checking data status: {e}")
            return f"Error checking data status: {str(e)}"

    # -------------------------
    # QUALITY ASSESSMENT TOOL
    # -------------------------
    @tool
    def assess_data_quality(
        modality_name: str,
        min_genes: Optional[int] = None,
        max_genes: Optional[int] = None,
        max_mt_pct: Optional[float] = None,
        max_ribo_pct: Optional[float] = None,
    ) -> str:
        """
        Run comprehensive quality control assessment on transcriptomics data.

        Auto-detects data type (single-cell vs bulk) and applies appropriate
        defaults when parameters are not specified.

        Args:
            modality_name: Name of the modality to assess
            min_genes: Minimum genes per cell/sample (lower bound).
                      Default: 200 (SC), 1000 (bulk)
            max_genes: Maximum genes per cell (upper bound, for doublet filtering).
                      Default: 5000 (SC), None (bulk).
                      NOTE: For cardiac/muscle tissue or metabolically active cells,
                      consider 8000-10000 to avoid filtering legitimate cells.
            max_mt_pct: Maximum mitochondrial percentage.
                       Default: 20.0% (SC), 30.0% (bulk).
                       TISSUE WARNING: For cardiac/muscle tissue, neurons, or hepatocytes,
                       mitochondrial content is naturally higher. Consider 30-50% for
                       these cell types to avoid removing healthy cells.
            max_ribo_pct: Maximum ribosomal percentage.
                         Default: 50.0% (SC), 100.0% (bulk)

        Returns:
            Formatted QC assessment report with statistics and recommendations
        """
        try:
            if modality_name == "":
                return (
                    "Please specify modality_name for quality assessment. "
                    "Use check_data_status() to see available modalities."
                )

            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                return (
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the modality and auto-detect type
            adata = data_manager.get_modality(modality_name)
            data_type = detect_data_type(adata)
            defaults = get_qc_defaults(data_type)

            # Apply defaults for unspecified parameters
            min_genes = min_genes if min_genes is not None else defaults["min_genes"]
            max_genes = max_genes if max_genes is not None else defaults["max_genes"]
            max_mt_pct = (
                max_mt_pct if max_mt_pct is not None else defaults["max_mt_pct"]
            )
            max_ribo_pct = (
                max_ribo_pct if max_ribo_pct is not None else defaults["max_ribo_pct"]
            )

            # Handle None max_genes for bulk (use very high value)
            effective_max_genes = max_genes if max_genes is not None else 100000

            # Run quality assessment using service
            # Note: min_housekeeping_score removed - not scientifically validated
            adata_qc, assessment_stats, ir = quality_service.assess_quality(
                adata=adata,
                min_genes=min_genes,
                max_genes=effective_max_genes,
                max_mt_pct=max_mt_pct,
                max_ribo_pct=max_ribo_pct,
                min_housekeeping_score=0.0,  # Disabled - not validated
            )

            # Create new modality with QC annotations
            qc_modality_name = f"{modality_name}_quality_assessed"
            data_manager.store_modality(
                name=qc_modality_name,
                adata=adata_qc,
                parent_name=modality_name,
                step_summary=f"QC assessed: {assessment_stats['cells_after_qc']:,} cells passing ({assessment_stats['cells_retained_pct']:.1f}%)",
            )

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="assess_data_quality",
                parameters={
                    "modality_name": modality_name,
                    "data_type": data_type,
                    "min_genes": min_genes,
                    "max_genes": max_genes,
                    "max_mt_pct": max_mt_pct,
                    "max_ribo_pct": max_ribo_pct,
                },
                description=f"{data_type} quality assessment for {modality_name}",
                ir=ir,
            )

            # Format professional response
            obs_label = "Cells" if data_type == "single_cell" else "Samples"
            obs_label_lower = "cells" if data_type == "single_cell" else "samples"
            type_label = data_type.replace("_", " ").title()

            response = f"""Quality Assessment Complete for '{modality_name}'!

**Data Type**: {type_label}

**Assessment Results:**
- {obs_label} analyzed: {assessment_stats["cells_before_qc"]:,}
- {obs_label} passing QC: {assessment_stats["cells_after_qc"]:,} ({assessment_stats["cells_retained_pct"]:.1f}%)
- Quality status: {assessment_stats["quality_status"]}

**Quality Metrics:**
- Mean genes per {obs_label_lower[:-1]}: {assessment_stats["mean_genes_per_cell"]:.0f}
- Mean mitochondrial %: {assessment_stats["mean_mt_pct"]:.1f}%
- Mean ribosomal %: {assessment_stats["mean_ribo_pct"]:.1f}%
- Mean total counts: {assessment_stats["mean_total_counts"]:.0f}

**QC Parameters Applied:**
- Min genes: {min_genes}
- Max genes: {max_genes if max_genes is not None else "None (no upper limit)"}
- Max MT%: {max_mt_pct}%
- Max Ribo%: {max_ribo_pct}%

**QC Summary:**
{assessment_stats["qc_summary"]}

**New modality created**: '{qc_modality_name}' (with QC annotations)

Proceed with filtering and normalization for downstream analysis."""

            analysis_results["details"]["quality_assessment"] = response
            return response

        except QualityError as e:
            logger.error(f"Quality assessment error: {e}")
            return f"Quality assessment failed: {str(e)}"
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return f"Error in quality assessment: {str(e)}"

    # -------------------------
    # FILTER AND NORMALIZE TOOL
    # -------------------------
    @tool
    def filter_and_normalize(
        modality_name: str,
        min_genes_per_cell: Optional[int] = None,
        max_genes_per_cell: Optional[int] = None,
        min_cells_per_gene: Optional[int] = None,
        max_mito_percent: Optional[float] = None,
        normalization_method: Optional[str] = None,
        target_sum: Optional[int] = None,
        save_result: bool = True,
    ) -> str:
        """
        Filter and normalize transcriptomics data using professional QC standards.

        Auto-detects data type (single-cell vs bulk) and applies appropriate
        defaults when parameters are not specified.

        Args:
            modality_name: Name of the modality to process
            min_genes_per_cell: Minimum genes expressed per cell/sample.
                               Default: 200 (SC), 1000 (bulk)
            max_genes_per_cell: Maximum genes per cell (doublet filtering).
                               Default: 5000 (SC), None (bulk).
                               TISSUE NOTE: For metabolically active cells (neurons,
                               hepatocytes) or proliferative populations, consider
                               8000-10000 to avoid filtering legitimate high-complexity cells.
            min_cells_per_gene: Minimum cells/samples expressing each gene.
                               Default: 3 (SC), 2 (bulk)
            max_mito_percent: Maximum mitochondrial gene percentage.
                             Default: 20.0% (SC), 30.0% (bulk).
                             TISSUE WARNING: For cardiac/muscle tissue, neurons, or
                             hepatocytes, mitochondrial content is naturally higher.
                             Consider 30-50% for these cell types.
            normalization_method: Normalization method ('log1p', 'cpm', 'tpm').
                                 Default: 'log1p' for both types
            target_sum: Target sum for normalization.
                       Default: 10000 (SC), 1000000 (bulk)
            save_result: Whether to save the filtered modality

        Returns:
            Formatted processing report with statistics
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                return (
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the modality and auto-detect type
            adata = data_manager.get_modality(modality_name)
            data_type = detect_data_type(adata)
            defaults = get_qc_defaults(data_type)

            # Apply defaults for unspecified parameters
            min_genes_per_cell = (
                min_genes_per_cell
                if min_genes_per_cell is not None
                else defaults["min_genes"]
            )
            max_genes_per_cell = (
                max_genes_per_cell
                if max_genes_per_cell is not None
                else defaults["max_genes"]
            )
            min_cells_per_gene = (
                min_cells_per_gene
                if min_cells_per_gene is not None
                else defaults["min_cells_per_gene"]
            )
            max_mito_percent = (
                max_mito_percent
                if max_mito_percent is not None
                else defaults["max_mt_pct"]
            )
            normalization_method = (
                normalization_method
                if normalization_method is not None
                else defaults["normalization_method"]
            )
            target_sum = (
                target_sum if target_sum is not None else defaults["target_sum"]
            )

            # Handle None max_genes for bulk (use very high value)
            effective_max_genes = (
                max_genes_per_cell if max_genes_per_cell is not None else 100000
            )

            logger.info(
                f"Processing {data_type} modality '{modality_name}': "
                f"{adata.shape[0]} obs x {adata.shape[1]} vars"
            )

            # Use preprocessing service
            adata_processed, processing_stats, ir = (
                preprocessing_service.filter_and_normalize_cells(
                    adata=adata,
                    min_genes_per_cell=min_genes_per_cell,
                    max_genes_per_cell=effective_max_genes,
                    min_cells_per_gene=min_cells_per_gene,
                    max_mito_percent=max_mito_percent,
                    normalization_method=normalization_method,
                    target_sum=target_sum,
                )
            )

            # Save as new modality
            filtered_modality_name = f"{modality_name}_filtered_normalized"
            data_manager.store_modality(
                name=filtered_modality_name,
                adata=adata_processed,
                parent_name=modality_name,
                step_summary=f"Filtered and normalized: {processing_stats['cells_retained_pct']:.1f}% cells retained",
            )

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_filtered_normalized.h5ad"
                data_manager.save_modality(filtered_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="filter_and_normalize",
                parameters={
                    "modality_name": modality_name,
                    "data_type": data_type,
                    "min_genes_per_cell": min_genes_per_cell,
                    "max_genes_per_cell": max_genes_per_cell,
                    "min_cells_per_gene": min_cells_per_gene,
                    "max_mito_percent": max_mito_percent,
                    "normalization_method": normalization_method,
                    "target_sum": target_sum,
                },
                description=f"{data_type} filtered and normalized {modality_name}",
                ir=ir,
            )

            # Format professional response
            original_shape = processing_stats["original_shape"]
            final_shape = processing_stats["final_shape"]
            obs_retained_pct = processing_stats["cells_retained_pct"]
            genes_retained_pct = processing_stats["genes_retained_pct"]

            obs_label = "cells" if data_type == "single_cell" else "samples"
            type_label = data_type.replace("_", " ").title()

            response = f"""Successfully filtered and normalized modality '{modality_name}'!

**Data Type**: {type_label}

**Filtering Results:**
- Original: {original_shape[0]:,} {obs_label} x {original_shape[1]:,} genes
- Filtered: {final_shape[0]:,} {obs_label} x {final_shape[1]:,} genes
- {obs_label.title()} retained: {obs_retained_pct:.1f}%
- Genes retained: {genes_retained_pct:.1f}%

**Processing Parameters:**
- Min genes/{obs_label[:-1]}: {min_genes_per_cell}
- Max genes/{obs_label[:-1]}: {max_genes_per_cell if max_genes_per_cell is not None else "None"}
- Min {obs_label}/gene: {min_cells_per_gene}
- Max mitochondrial %: {max_mito_percent}%
- Normalization: {normalization_method} (target_sum={target_sum:,})

**New modality created**: '{filtered_modality_name}'"""

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n\nNext recommended steps: " + (
                "doublet detection, then clustering and cell type annotation."
                if data_type == "single_cell"
                else "differential expression analysis between experimental groups."
            )

            analysis_results["details"]["filter_normalize"] = response
            return response

        except Exception as e:
            logger.error(f"Error in filtering/normalization: {e}")
            return f"Error filtering and normalizing modality: {str(e)}"

    # -------------------------
    # ANALYSIS SUMMARY TOOL
    # -------------------------
    @tool
    def create_analysis_summary(modality_name: str = "") -> str:
        """
        Create a comprehensive summary of all transcriptomics analysis steps performed.

        Generates a report including:
        - All analysis steps completed
        - Current modality status
        - Data type classification for each modality
        - Key metrics and parameters used

        Args:
            modality_name: Optional specific modality to summarize.
                          If empty, summarizes all modalities.

        Returns:
            Formatted analysis summary report
        """
        try:
            summary = "# Transcriptomics Analysis Summary\n\n"

            # Add analysis steps if any
            if analysis_results["details"]:
                summary += "## Analysis Steps Performed\n\n"
                for step, details in analysis_results["details"].items():
                    summary += f"### {step.replace('_', ' ').title()}\n"
                    summary += f"{details}\n\n"
            else:
                summary += (
                    "No analyses have been performed yet. "
                    "Run some analysis tools first.\n\n"
                )

            # Add current modality status
            modalities = data_manager.list_modalities()
            if modalities:
                if modality_name and modality_name in modalities:
                    # Specific modality summary
                    summary += f"## Modality Details: {modality_name}\n\n"
                    adata = data_manager.get_modality(modality_name)
                    data_type = detect_data_type(adata)
                    obs_label = "cells" if data_type == "single_cell" else "samples"

                    summary += (
                        f"- **Data type**: {data_type.replace('_', ' ').title()}\n"
                    )
                    summary += f"- **Shape**: {adata.n_obs} {obs_label} x {adata.n_vars} genes\n"
                    summary += f"- **Obs columns**: {list(adata.obs.columns)}\n"
                    summary += f"- **Var columns**: {list(adata.var.columns)}\n"

                    # Check for analysis indicators
                    _known_cluster_names = {"leiden", "louvain", "seurat_clusters", "cluster"}
                    cluster_cols = [
                        c for c in adata.obs.columns
                        if c in _known_cluster_names
                        or c.startswith(("leiden_", "louvain_", "RNA_snn_res"))
                    ]
                    if cluster_cols:
                        summary += f"- **Clustering**: {', '.join(cluster_cols)}\n"
                    if "cell_type" in adata.obs.columns:
                        summary += "- **Cell type annotation**: Performed\n"
                    if "qc_pass" in adata.obs.columns:
                        summary += "- **QC assessment**: Performed\n"

                else:
                    # All modalities summary
                    summary += f"## Current Modalities ({len(modalities)})\n\n"

                    # Categorize by data type
                    sc_modalities = []
                    bulk_modalities = []

                    for mod_name in modalities:
                        try:
                            adata = data_manager.get_modality(mod_name)
                            data_type = detect_data_type(adata)
                            if data_type == "single_cell":
                                sc_modalities.append((mod_name, adata))
                            else:
                                bulk_modalities.append((mod_name, adata))
                        except Exception:
                            pass

                    if sc_modalities:
                        summary += "### Single-Cell Modalities\n"
                        for mod_name, adata in sc_modalities:
                            summary += (
                                f"- **{mod_name}**: {adata.n_obs} cells x "
                                f"{adata.n_vars} genes\n"
                            )

                            # Add key single-cell columns if present
                            _sc_key_names = {
                                "leiden", "louvain", "seurat_clusters", "cluster",
                                "cell_type", "doublet_score", "qc_pass",
                            }
                            key_cols = [
                                col
                                for col in adata.obs.columns
                                if col in _sc_key_names
                                or col.startswith(("leiden_", "louvain_", "RNA_snn_res"))
                            ]
                            if key_cols:
                                summary += f"  - Annotations: {', '.join(key_cols)}\n"
                        summary += "\n"

                    if bulk_modalities:
                        summary += "### Bulk RNA-seq Modalities\n"
                        for mod_name, adata in bulk_modalities:
                            summary += (
                                f"- **{mod_name}**: {adata.n_obs} samples x "
                                f"{adata.n_vars} genes\n"
                            )

                            # Add key bulk columns if present
                            key_cols = [
                                col
                                for col in adata.obs.columns
                                if col.lower()
                                in [
                                    "condition",
                                    "treatment",
                                    "group",
                                    "batch",
                                    "time_point",
                                ]
                            ]
                            if key_cols:
                                summary += (
                                    f"  - Experimental design: {', '.join(key_cols)}\n"
                                )
                        summary += "\n"

            else:
                summary += "## No Modalities Loaded\n\n"
                summary += "Please load transcriptomics data to begin analysis.\n"

            analysis_results["summary"] = summary
            logger.info(
                f"Created analysis summary with {len(analysis_results['details'])} steps"
            )
            return summary

        except Exception as e:
            logger.error(f"Error creating analysis summary: {e}")
            return f"Error creating analysis summary: {str(e)}"

    # -------------------------
    # FEATURE SELECTION TOOL
    # -------------------------
    @tool
    def select_variable_features(
        modality_name: str,
        method: str = "deviance",
        n_top_genes: int = 2000,
        flavor: str = "seurat",
    ) -> str:
        """
        Select highly variable or highly deviant genes for downstream analysis.

        This is a STANDALONE feature selection tool. Use it when the user asks
        specifically for HVG/feature selection WITHOUT full clustering.
        For full clustering pipelines, use cluster_cells() instead.

        Supports two methods:
        - "deviance" (default, recommended): Binomial deviance from multinomial null.
          Works on RAW COUNTS. Run BEFORE normalization.
        - "hvg": Traditional highly variable genes (scanpy).
          Works on NORMALIZED data. Run AFTER filter_and_normalize().

        Args:
            modality_name: Name of the modality to process
            method: Feature selection method ('deviance' or 'hvg').
            n_top_genes: Number of top genes to select (default: 2000)
            flavor: HVG flavor, only used when method='hvg' ('seurat', 'cell_ranger', 'seurat_v3')

        Returns:
            Formatted report with selected genes and next-step guidance
        """
        try:
            if modality_name not in data_manager.list_modalities():
                return (
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Selecting features for '{modality_name}' using method='{method}' "
                f"(n_top_genes={n_top_genes})"
            )

            if method == "deviance":
                adata_result, stats, ir = preprocessing_service.select_features_deviance(
                    adata, n_top_genes=n_top_genes
                )
            elif method == "hvg":
                adata_result, stats, ir = preprocessing_service.select_features_hvg(
                    adata, n_top_genes=n_top_genes, flavor=flavor
                )
            else:
                return (
                    f"Unknown method '{method}'. Use 'deviance' (recommended, on raw counts) "
                    f"or 'hvg' (on normalized data)."
                )

            # Store as new modality
            result_name = f"{modality_name}_hvg_selected"
            data_manager.store_modality(
                name=result_name,
                adata=adata_result,
                parent_name=modality_name,
                step_summary=f"Feature selection ({method}): {stats['n_features_selected']} genes selected",
            )

            data_manager.log_tool_usage(
                tool_name="select_variable_features",
                parameters={
                    "modality_name": modality_name,
                    "method": method,
                    "n_top_genes": n_top_genes,
                    "flavor": flavor,
                },
                description=f"Selected {stats['n_features_selected']} features using {method}",
                ir=ir,
            )

            # Build response
            response = f"""Feature selection complete for '{modality_name}'!

**Method**: {method}{f" (flavor={flavor})" if method == "hvg" else " (binomial deviance)"}
**Genes selected**: {stats['n_features_selected']} / {stats['original_n_genes']} ({stats['selection_rate']:.1f}%)
**Top 10 genes**: {', '.join(stats['top_10_genes'])}"""

            if stats.get("warning"):
                response += f"\n\n**Warning**: {stats['warning']}"

            response += f"\n\n**New modality created**: '{result_name}'"

            # Next-step guidance based on method
            if method == "deviance":
                response += (
                    "\n\n**Next steps**: filter_and_normalize() → run_pca()"
                )
            else:
                response += "\n\n**Next steps**: run_pca() for dimensionality reduction"

            analysis_results["details"]["feature_selection"] = response
            return response

        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return f"Error selecting features: {str(e)}"

    # -------------------------
    # PCA TOOL
    # -------------------------
    @tool
    def run_pca(
        modality_name: str,
        n_comps: int = 30,
        scale_data: bool = True,
        use_highly_variable: bool = True,
    ) -> str:
        """
        Run PCA dimensionality reduction as a standalone step.

        Performs scaling (optional) and PCA. Stores adata.raw BEFORE scaling
        so downstream marker gene analysis can access unscaled expression values.

        Prerequisites: Feature selection should be applied first
        (select_variable_features or filter_and_normalize).

        Args:
            modality_name: Name of the modality to process
            n_comps: Number of principal components (default: 30)
            scale_data: Whether to scale data before PCA (default: True)
            use_highly_variable: Whether to subset to selected features (default: True)

        Returns:
            Formatted PCA report with variance explained and next-step guidance
        """
        if clustering_service is None:
            return (
                "Error: PCA tool requires clustering_service. "
                "This tool is not available in the current configuration."
            )

        try:
            if modality_name not in data_manager.list_modalities():
                return (
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(f"Running PCA for '{modality_name}' (n_comps={n_comps})")

            adata_result, stats, ir = clustering_service.run_pca(
                adata,
                n_comps=n_comps,
                scale_data=scale_data,
                use_highly_variable=use_highly_variable,
            )

            result_name = f"{modality_name}_pca"
            data_manager.store_modality(
                name=result_name,
                adata=adata_result,
                parent_name=modality_name,
                step_summary=f"PCA: {stats['n_comps_computed']} components, {stats['variance_explained']}% variance",
            )

            data_manager.log_tool_usage(
                tool_name="run_pca",
                parameters={
                    "modality_name": modality_name,
                    "n_comps": n_comps,
                    "scale_data": scale_data,
                    "use_highly_variable": use_highly_variable,
                },
                description=f"PCA with {stats['n_comps_computed']} components on {modality_name}",
                ir=ir,
            )

            response = f"""PCA complete for '{modality_name}'!

**Components computed**: {stats['n_comps_computed']}
**Variance explained**: {stats['variance_explained']}%
**Features used**: {stats['n_features_used']}
**Data scaled**: {stats['scaled']}

**New modality created**: '{result_name}'

**Next steps**: compute_neighbors_and_embed() for UMAP/tSNE visualization"""

            analysis_results["details"]["pca"] = response
            return response

        except Exception as e:
            logger.error(f"Error in PCA: {e}")
            return f"Error running PCA: {str(e)}"

    # -------------------------
    # NEIGHBORS + EMBEDDING TOOL
    # -------------------------
    @tool
    def compute_neighbors_and_embed(
        modality_name: str,
        n_neighbors: int = 15,
        n_pcs: int = 30,
        embedding_method: str = "umap",
    ) -> str:
        """
        Compute neighborhood graph and embedding (UMAP or tSNE) as a standalone step.

        Prerequisites: PCA must be computed first (run_pca or cluster_cells).

        Args:
            modality_name: Name of the modality to process
            n_neighbors: Number of neighbors for KNN graph (default: 15)
            n_pcs: Number of PCs for neighbor computation (default: 30)
            embedding_method: Embedding method ('umap' or 'tsne', default: 'umap')

        Returns:
            Formatted embedding report with next-step guidance
        """
        if clustering_service is None:
            return (
                "Error: Embedding tool requires clustering_service. "
                "This tool is not available in the current configuration."
            )

        try:
            if modality_name not in data_manager.list_modalities():
                return (
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Computing neighbors + {embedding_method} for '{modality_name}'"
            )

            adata_result, stats, ir = clustering_service.compute_neighbors_and_embed(
                adata,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                embedding_method=embedding_method,
            )

            result_name = f"{modality_name}_embedded"
            data_manager.store_modality(
                name=result_name,
                adata=adata_result,
                parent_name=modality_name,
                step_summary=f"{embedding_method.upper()} embedding computed",
            )

            data_manager.log_tool_usage(
                tool_name="compute_neighbors_and_embed",
                parameters={
                    "modality_name": modality_name,
                    "n_neighbors": n_neighbors,
                    "n_pcs": n_pcs,
                    "embedding_method": embedding_method,
                },
                description=f"Computed neighbors + {embedding_method} on {modality_name}",
                ir=ir,
            )

            response = f"""Neighbors + {embedding_method.upper()} embedding complete for '{modality_name}'!

**Neighbors**: {stats['n_neighbors_used']}
**PCs used**: {stats['n_pcs_used']}
**Embedding**: {embedding_method.upper()} ({stats['embedding_shape'][0]} x {stats['embedding_shape'][1]})

**New modality created**: '{result_name}'

**Next steps**: Use cluster_cells() for Leiden clustering, or visualize directly"""

            analysis_results["details"]["embedding"] = response
            return response

        except Exception as e:
            logger.error(f"Error in neighbors/embedding: {e}")
            return f"Error computing neighbors/embedding: {str(e)}"

    # Return list of tools
    tools = [
        check_data_status,
        assess_data_quality,
        filter_and_normalize,
        select_variable_features,
        create_analysis_summary,
    ]
    if clustering_service is not None:
        tools.extend([run_pca, compute_neighbors_and_embed])
    return tools
