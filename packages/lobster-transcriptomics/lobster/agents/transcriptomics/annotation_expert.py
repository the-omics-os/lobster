"""
Annotation Expert Sub-Agent for single-cell RNA-seq cell type annotation.

This sub-agent handles all cell type annotation tools for single-cell data.
It is called by the parent transcriptomics_expert via delegation tools.

Tools included:
1. annotate_cell_types_auto - Automated cell type annotation using marker databases
2. score_gene_set - Score cells for a user-provided gene list
3. manually_annotate_clusters - Direct cluster-to-celltype assignment
4. collapse_clusters_to_celltype - Merge multiple clusters into a single cell type
5. mark_clusters_as_debris - Flag clusters as debris for QC
6. suggest_debris_clusters - Smart suggestions for potential debris clusters
7. review_annotation_assignments - Review current annotation coverage
8. apply_annotation_template - Apply tissue-specific annotation templates
9. export_annotation_mapping - Export annotations for reuse
10. import_annotation_mapping - Import and apply saved annotations
11. annotate_cell_types_semantic - Semantic annotation via Cell Ontology (optional)

Deprecated:
- manually_annotate_clusters_interactive - DEPRECATED (cloud-incompatible)
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="annotation_expert",
    display_name="Annotation Expert",
    description="Cell type annotation sub-agent: automatic annotation, manual cluster labeling, debris detection, annotation templates",
    factory_function="lobster.agents.transcriptomics.annotation_expert.annotation_expert",
    handoff_tool_name=None,  # Not directly accessible
    handoff_tool_description=None,
    supervisor_accessible=False,  # Only via transcriptomics_expert
)

# === Heavy imports below ===
import datetime
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.state import SingleCellExpertState
from lobster.agents.transcriptomics.prompts import create_annotation_expert_prompt
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
)
from lobster.services.metadata.manual_annotation_service import ManualAnnotationService
from lobster.services.templates.annotation_templates import (
    AnnotationTemplateService,
    TissueType,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Optional vector search for semantic annotation
try:
    from lobster.services.vector.service import VectorSearchService  # noqa: F401

    HAS_VECTOR_SEARCH = True
except ImportError:
    HAS_VECTOR_SEARCH = False


# Type alias for AnnotationExpertState - uses SingleCellExpertState for now
# as annotations share the same state structure
AnnotationExpertState = SingleCellExpertState


class AnnotationAgentError(Exception):
    """Base exception for annotation agent operations."""

    pass


class ModalityNotFoundError(AnnotationAgentError):
    """Raised when requested modality doesn't exist."""

    pass


def annotation_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "annotation_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for annotation expert sub-agent.

    This sub-agent handles all cell type annotation tools for single-cell data.
    It is delegated to by the transcriptomics_expert for annotation tasks.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM responses
        agent_name: Name of the agent for routing
        delegation_tools: Optional list of delegation tools from parent agent

    Returns:
        A LangGraph ReAct agent configured for cell type annotation
    """

    settings = get_settings()
    model_params = settings.get_agent_llm_params("annotation_expert")
    llm = create_llm(
        "annotation_expert",
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Initialize services for annotation
    manual_annotation_service = ManualAnnotationService()
    template_service = AnnotationTemplateService()
    singlecell_service = EnhancedSingleCellService()

    # Track analysis results
    analysis_results = {"summary": "", "details": {}}

    # -------------------------
    # ANNOTATION TOOLS
    # -------------------------

    @tool
    def annotate_cell_types_auto(
        modality_name: str,
        cluster_key: str,
        reference_markers: dict = None,
        save_result: bool = True,
    ) -> str:
        """
        Annotate single-cell clusters with cell types based on marker gene expression patterns.

        IMPORTANT: You must specify cluster_key -- the obs column containing cluster assignments.
        Use check_data_status(modality_name) first to inspect available obs columns.
        Common cluster column names: 'leiden', 'louvain', 'seurat_clusters', 'RNA_snn_res.1'.

        Args:
            modality_name: Name of the single-cell modality with clustering results
            cluster_key: Column name in adata.obs containing cluster assignments
            reference_markers: Custom marker genes dict (None to use defaults)
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Annotating cell types in single-cell modality '{modality_name}': {adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Use singlecell service for cell type annotation
            adata_annotated, annotation_stats, ir = (
                singlecell_service.annotate_cell_types(
                    adata=adata,
                    cluster_key=cluster_key,
                    reference_markers=reference_markers,
                )
            )

            # Save as new modality
            annotated_modality_name = f"{modality_name}_annotated"
            data_manager.store_modality(
                name=annotated_modality_name,
                adata=adata_annotated,
                parent_name=modality_name,
                step_summary=f"Annotated {annotation_stats['n_cell_types_identified']} cell types",
            )

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_annotated.h5ad"
                data_manager.save_modality(annotated_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="annotate_cell_types_auto",
                parameters={
                    "modality_name": modality_name,
                    "cluster_key": cluster_key,
                    "reference_markers": "custom" if reference_markers else "default",
                },
                description=f"Annotated {annotation_stats['n_cell_types_identified']} cell types in single-cell data {modality_name}",
                ir=ir,
            )

            # Format professional response
            response = f"""Successfully annotated cell types in single-cell modality '{modality_name}'!

**Single-cell Annotation Results:**
- Cell types identified: {annotation_stats["n_cell_types_identified"]}
- Clusters annotated: {annotation_stats["n_clusters"]}
- Marker sets used: {annotation_stats["n_marker_sets"]}

**Single-cell Type Distribution:**"""

            for cell_type, count in list(annotation_stats["cell_type_counts"].items())[
                :8
            ]:
                response += f"\n- {cell_type}: {count} cells"

            if len(annotation_stats["cell_type_counts"]) > 8:
                remaining = len(annotation_stats["cell_type_counts"]) - 8
                response += f"\n... and {remaining} more types"

            # Add confidence distribution if available
            if "confidence_mean" in annotation_stats:
                response += "\n\n**Confidence Scoring:**"
                response += (
                    f"\n- Mean confidence: {annotation_stats['confidence_mean']:.3f}"
                )
                response += f"\n- Median confidence: {annotation_stats['confidence_median']:.3f}"
                response += (
                    f"\n- Std deviation: {annotation_stats['confidence_std']:.3f}"
                )

                response += "\n\n**Annotation Quality Distribution:**"
                quality_dist = annotation_stats["quality_distribution"]
                response += f"\n- High confidence: {quality_dist['high']} cells"
                response += f"\n- Medium confidence: {quality_dist['medium']} cells"
                response += f"\n- Low confidence: {quality_dist['low']} cells"

                response += "\n\n**Note**: Per-cell confidence scores available in:"
                response += (
                    "\n  - adata.obs['cell_type_confidence']: Correlation score (0-1)"
                )
                response += "\n  - adata.obs['cell_type_top3']: Top 3 predictions"
                response += "\n  - adata.obs['annotation_entropy']: Shannon entropy"
                response += "\n  - adata.obs['annotation_quality']: Quality flag (high/medium/low)"

            response += f"\n\n**New modality created**: '{annotated_modality_name}'"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n**Cell type annotations added to**: adata.obs['cell_type']"
            response += "\n\nProceed with cell type-specific downstream analysis or comparative studies."

            analysis_results["details"]["cell_type_annotation"] = response
            return response

        except (ModalityNotFoundError,) as e:
            logger.error(f"Error in single-cell cell type annotation: {e}")
            return f"Error annotating cell types in single-cell data: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in single-cell cell type annotation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def manually_annotate_clusters_interactive(
        modality_name: str, cluster_key: str, save_result: bool = True
    ) -> str:
        """
        DEPRECATED: Launch Rich terminal interface for manual cluster annotation.

        This tool requires terminal access and is incompatible with cloud deployment.
        Use manually_annotate_clusters for direct cluster annotation instead.

        Args:
            modality_name: Name of clustered single-cell modality
            cluster_key: Column containing cluster assignments (REQUIRED).
            save_result: Whether to save annotated modality
        """
        logger.warning(
            "manually_annotate_clusters_interactive is deprecated (cloud-incompatible). "
            "Use manually_annotate_clusters instead."
        )
        return (
            "DEPRECATED: This tool requires terminal access and is incompatible with "
            "cloud deployment. Use manually_annotate_clusters for direct cluster annotation."
        )

    @tool
    def manually_annotate_clusters(
        modality_name: str,
        annotations: dict,
        cluster_key: str,
        save_result: bool = True,
    ) -> str:
        """
        Directly assign cell types to clusters without interactive interface.

        IMPORTANT: Call check_data_status() first to identify the actual cluster column name.

        Args:
            modality_name: Name of clustered single-cell modality
            annotations: Dictionary mapping cluster IDs to cell type names
            cluster_key: Column containing cluster assignments (REQUIRED).
                        Common values: 'leiden', 'louvain', 'seurat_clusters'.
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Validate cluster column exists
            if cluster_key not in adata.obs.columns:
                available_cols = list(adata.obs.columns)
                return (
                    f"Cluster column '{cluster_key}' not found.\n\n"
                    f"Available columns: {available_cols}\n\n"
                    f"Use check_data_status() to identify the correct cluster column."
                )

            # Apply annotations directly
            adata_copy = adata.copy()
            cell_type_mapping = {}

            for cluster_id, cell_type in annotations.items():
                cell_type_mapping[str(cluster_id)] = cell_type

            # Create cell type column
            adata_copy.obs["cell_type_manual"] = (
                adata_copy.obs[cluster_key]
                .astype(str)
                .map(cell_type_mapping)
                .fillna("Unassigned")
            )

            # Save as new modality
            annotated_modality_name = f"{modality_name}_manually_annotated"
            data_manager.store_modality(
                name=annotated_modality_name,
                adata=adata_copy,
                parent_name=modality_name,
                step_summary=f"Manually annotated {len(annotations)} clusters",
            )

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_manually_annotated.h5ad"
                data_manager.save_modality(annotated_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="manually_annotate_clusters",
                parameters={
                    "modality_name": modality_name,
                    "cluster_key": cluster_key,
                    "annotations": annotations,
                },
                description=f"Direct manual annotation of {len(annotations)} clusters",
            )

            response = f"""Manual cluster annotation applied to '{modality_name}'!

**Annotation Results:**
- Clusters annotated: {len(annotations)}
- Cell types assigned: {len(set(annotations.values()))}

**Annotations Applied:**"""

            for cluster_id, cell_type in list(annotations.items())[:10]:
                response += f"\n- Cluster {cluster_id}: {cell_type}"

            if len(annotations) > 10:
                response += f"\n... and {len(annotations) - 10} more clusters"

            response += f"\n\n**New modality created**: '{annotated_modality_name}'"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            return response

        except Exception as e:
            logger.error(f"Error in manual annotation: {e}")
            return f"Error applying manual annotations: {str(e)}"

    @tool
    def collapse_clusters_to_celltype(
        modality_name: str,
        cluster_list: List[str],
        cell_type_name: str,
        cluster_key: str,
        save_result: bool = True,
    ) -> str:
        """
        Collapse multiple clusters into a single cell type.

        IMPORTANT: Call check_data_status() first to identify the actual cluster column name.

        Args:
            modality_name: Name of single-cell modality
            cluster_list: List of cluster IDs to collapse
            cell_type_name: New cell type name for collapsed clusters
            cluster_key: Column containing cluster assignments (REQUIRED).
                        Common values: 'leiden', 'louvain', 'seurat_clusters'.
            save_result: Whether to save result
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Validate clusters exist
            unique_clusters = set(adata.obs[cluster_key].astype(str).unique())
            invalid_clusters = [
                c for c in cluster_list if str(c) not in unique_clusters
            ]
            if invalid_clusters:
                return f"Invalid cluster IDs: {invalid_clusters}. Available: {sorted(unique_clusters)}"

            # Create collapsed annotation
            adata_copy = adata.copy()

            # Create or update manual cell type column
            if "cell_type_manual" not in adata_copy.obs:
                adata_copy.obs["cell_type_manual"] = "Unassigned"

            # Apply collapse
            for cluster_id in cluster_list:
                mask = adata_copy.obs[cluster_key].astype(str) == str(cluster_id)
                adata_copy.obs.loc[mask, "cell_type_manual"] = cell_type_name

            # Calculate statistics
            total_cells_collapsed = sum(
                (adata_copy.obs[cluster_key].astype(str) == str(c)).sum()
                for c in cluster_list
            )

            # Save as new modality
            collapsed_modality_name = f"{modality_name}_collapsed"
            data_manager.store_modality(
                name=collapsed_modality_name,
                adata=adata_copy,
                parent_name=modality_name,
                step_summary=f"Collapsed {len(cluster_list)} clusters to '{cell_type_name}'",
            )

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_collapsed.h5ad"
                data_manager.save_modality(collapsed_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="collapse_clusters_to_celltype",
                parameters={
                    "modality_name": modality_name,
                    "cluster_list": cluster_list,
                    "cell_type_name": cell_type_name,
                    "cluster_key": cluster_key,
                },
                description=f"Collapsed {len(cluster_list)} clusters into '{cell_type_name}'",
            )

            response = f"""Successfully collapsed clusters in '{modality_name}'!

**Collapse Results:**
- Clusters collapsed: {", ".join(str(c) for c in cluster_list)}
- New cell type: {cell_type_name}
- Total cells affected: {total_cells_collapsed:,}

**New modality created**: '{collapsed_modality_name}'"""

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += f"\n\nClusters {', '.join(str(c) for c in cluster_list)} are now annotated as '{cell_type_name}'."

            return response

        except Exception as e:
            logger.error(f"Error collapsing clusters: {e}")
            return f"Error collapsing clusters: {str(e)}"

    @tool
    def mark_clusters_as_debris(
        modality_name: str,
        debris_clusters: List[str],
        remove_debris: bool = False,
        cluster_key: str = "",
        save_result: bool = True,
    ) -> str:
        """
        Mark specified clusters as debris for quality control.

        IMPORTANT: Call check_data_status() first to identify the actual cluster column name.

        Args:
            modality_name: Name of single-cell modality
            debris_clusters: List of cluster IDs to mark as debris
            remove_debris: Whether to remove debris clusters from data
            cluster_key: Column containing cluster assignments (REQUIRED).
                        Common values: 'leiden', 'louvain', 'seurat_clusters'.
            save_result: Whether to save result
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Validate cluster_key provided
            if not cluster_key:
                available_cols = list(adata.obs.columns)
                return (
                    f"Error: cluster_key is required.\n\n"
                    f"Available columns: {available_cols}\n\n"
                    f"Use check_data_status() to identify the correct cluster column."
                )

            # Validate clusters exist
            unique_clusters = set(adata.obs[cluster_key].astype(str).unique())
            invalid_clusters = [
                c for c in debris_clusters if str(c) not in unique_clusters
            ]
            if invalid_clusters:
                return f"Invalid cluster IDs: {invalid_clusters}. Available: {sorted(unique_clusters)}"

            adata_copy = adata.copy()

            # Mark debris clusters
            if "cell_type_manual" not in adata_copy.obs:
                adata_copy.obs["cell_type_manual"] = "Unassigned"

            debris_mask = (
                adata_copy.obs[cluster_key]
                .astype(str)
                .isin([str(c) for c in debris_clusters])
            )
            adata_copy.obs.loc[debris_mask, "cell_type_manual"] = "Debris"

            # Add debris flag
            adata_copy.obs["is_debris"] = False
            adata_copy.obs.loc[debris_mask, "is_debris"] = True

            # Optionally remove debris
            if remove_debris:
                adata_copy = adata_copy[~debris_mask].copy()

            total_debris_cells = debris_mask.sum()

            # Save as new modality
            debris_modality_name = f"{modality_name}_debris_marked"
            if remove_debris:
                debris_modality_name = f"{modality_name}_debris_removed"

            data_manager.store_modality(
                name=debris_modality_name,
                adata=adata_copy,
                parent_name=modality_name,
                step_summary=f"{'Removed' if remove_debris else 'Marked'} {total_debris_cells} debris cells",
            )

            # Save to file if requested
            if save_result:
                save_path = f"{debris_modality_name}.h5ad"
                data_manager.save_modality(debris_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="mark_clusters_as_debris",
                parameters={
                    "modality_name": modality_name,
                    "debris_clusters": debris_clusters,
                    "remove_debris": remove_debris,
                    "cluster_key": cluster_key,
                },
                description=f"Marked {len(debris_clusters)} clusters as debris ({total_debris_cells} cells)",
            )

            response = f"""Successfully marked debris clusters in '{modality_name}'!

**Debris Marking Results:**
- Clusters marked: {", ".join(str(c) for c in debris_clusters)}
- Total debris cells: {total_debris_cells:,}
- Action: {"Removed" if remove_debris else "Marked only"}

**New modality created**: '{debris_modality_name}'"""

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            if remove_debris:
                remaining_cells = adata_copy.n_obs
                response += f"\n**Remaining cells**: {remaining_cells:,} ({remaining_cells / adata.n_obs * 100:.1f}%)"
            else:
                response += "\n**Debris flag added**: adata.obs['is_debris']"

            return response

        except Exception as e:
            logger.error(f"Error marking debris clusters: {e}")
            return f"Error marking clusters as debris: {str(e)}"

    @tool
    def suggest_debris_clusters(
        modality_name: str,
        min_genes: int = 200,
        max_mt_percent: float = 50,
        min_umi: int = 500,
        cluster_key: str = "",
    ) -> str:
        """
        Get smart suggestions for potential debris clusters based on QC metrics.

        IMPORTANT: Call check_data_status() first to identify the actual cluster column name.

        Args:
            modality_name: Name of single-cell modality
            min_genes: Minimum genes per cell threshold
            max_mt_percent: Maximum mitochondrial percentage
            min_umi: Minimum UMI count threshold
            cluster_key: Column containing cluster assignments (REQUIRED).
                        Common values: 'leiden', 'louvain', 'seurat_clusters'.
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Validate cluster_key provided
            if not cluster_key:
                available_cols = list(adata.obs.columns)
                return (
                    f"Error: cluster_key is required.\n\n"
                    f"Available columns: {available_cols}\n\n"
                    f"Use check_data_status() to identify the correct cluster column."
                )

            # Get suggestions using manual annotation service
            suggested_debris = manual_annotation_service.suggest_debris_clusters(
                adata=adata,
                min_genes=min_genes,
                max_mt_percent=max_mt_percent,
                min_umi=min_umi,
            )

            if not suggested_debris:
                return f"No debris clusters suggested based on QC thresholds (min_genes={min_genes}, max_mt%={max_mt_percent}, min_umi={min_umi})"

            # Get cluster statistics for suggestions
            response = f"""Smart debris cluster suggestions for '{modality_name}':

**QC-Based Suggestions:**
- Clusters flagged: {len(suggested_debris)}
- Thresholds used: min_genes={min_genes}, max_mt%={max_mt_percent}, min_umi={min_umi}

**Suggested Debris Clusters:**"""

            for cluster_id in suggested_debris[:10]:
                cluster_mask = adata.obs[cluster_key].astype(str) == cluster_id
                n_cells = cluster_mask.sum()

                # Get QC stats for cluster
                if cluster_mask.sum() > 0:
                    mean_genes = (
                        adata.obs.loc[cluster_mask, "n_genes"].mean()
                        if "n_genes" in adata.obs
                        else 0
                    )
                    mean_mt = (
                        adata.obs.loc[cluster_mask, "percent_mito"].mean()
                        if "percent_mito" in adata.obs
                        else 0
                    )
                    mean_umi = (
                        adata.obs.loc[cluster_mask, "n_counts"].mean()
                        if "n_counts" in adata.obs
                        else 0
                    )

                    response += f"\n- Cluster {cluster_id}: {n_cells} cells (genes: {mean_genes:.0f}, MT: {mean_mt:.1f}%, UMI: {mean_umi:.0f})"

            if len(suggested_debris) > 10:
                response += f"\n... and {len(suggested_debris) - 10} more clusters"

            response += "\n\n**Recommendation:**"
            response += "\nUse 'mark_clusters_as_debris' to apply these suggestions."
            response += f"\nExample: mark_clusters_as_debris('{modality_name}', {suggested_debris[:5]})"

            return response

        except Exception as e:
            logger.error(f"Error suggesting debris clusters: {e}")
            return f"Error suggesting debris clusters: {str(e)}"

    @tool
    def review_annotation_assignments(
        modality_name: str,
        annotation_col: str = "cell_type_manual",
        show_unassigned: bool = True,
        show_debris: bool = True,
    ) -> str:
        """
        Review current manual annotation assignments.

        Args:
            modality_name: Name of modality with annotations
            annotation_col: Column containing cell type annotations
            show_unassigned: Whether to show unassigned clusters
            show_debris: Whether to show debris clusters
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            if annotation_col not in adata.obs.columns:
                return f"Annotation column '{annotation_col}' not found. Available columns: {list(adata.obs.columns)[:10]}"

            # Validate annotation coverage
            validation = manual_annotation_service.validate_annotation_coverage(
                adata, annotation_col
            )

            response = f"""Annotation review for '{modality_name}':

**Coverage Summary:**
- Total cells: {validation["total_cells"]:,}
- Annotated cells: {validation["annotated_cells"]:,} ({validation["coverage_percentage"]:.1f}%)
- Unassigned cells: {validation["unassigned_cells"]:,}
- Debris cells: {validation["debris_cells"]:,}
- Unique cell types: {validation["unique_cell_types"]}

**Cell Type Distribution:**"""

            # Show all cell types
            for cell_type, count in validation["cell_type_counts"].items():
                if cell_type == "Unassigned" and not show_unassigned:
                    continue
                if cell_type == "Debris" and not show_debris:
                    continue

                percentage = (count / validation["total_cells"]) * 100
                response += f"\n- {cell_type}: {count:,} cells ({percentage:.1f}%)"

            # Add quality assessment
            if validation["coverage_percentage"] >= 90:
                response += "\n\n**Quality**: Excellent annotation coverage"
            elif validation["coverage_percentage"] >= 70:
                response += "\n\n**Quality**: Good annotation coverage, consider annotating remaining clusters"
            else:
                response += (
                    "\n\n**Quality**: Low annotation coverage, more annotation needed"
                )

            return response

        except Exception as e:
            logger.error(f"Error reviewing annotations: {e}")
            return f"Error reviewing annotation assignments: {str(e)}"

    @tool
    def apply_annotation_template(
        modality_name: str,
        tissue_type: str,
        cluster_key: str,
        expression_threshold: float = 0.5,
        save_result: bool = True,
    ) -> str:
        """
        Apply predefined tissue-specific annotation template to suggest cell types.

        IMPORTANT: Call check_data_status() first to identify the actual cluster column name.

        Args:
            modality_name: Name of single-cell modality
            tissue_type: Type of tissue (pbmc, brain, lung, heart, kidney, liver, intestine, skin, tumor)
            cluster_key: Column containing cluster assignments (REQUIRED).
                        Common values: 'leiden', 'louvain', 'seurat_clusters'.
            expression_threshold: Minimum expression for marker detection
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Validate tissue type
            try:
                tissue_enum = TissueType(tissue_type.lower())
            except ValueError:
                available_tissues = [t.value for t in TissueType]
                return f"Invalid tissue type '{tissue_type}'. Available: {available_tissues}"

            # Apply template
            cluster_suggestions = template_service.apply_template_to_clusters(
                adata=adata,
                tissue_type=tissue_enum,
                cluster_col=cluster_key,
                expression_threshold=expression_threshold,
            )

            if not cluster_suggestions:
                return (
                    f"No template suggestions generated for tissue type '{tissue_type}'"
                )

            # Apply suggestions to data
            adata_copy = adata.copy()
            adata_copy.obs["cell_type_template"] = (
                adata_copy.obs[cluster_key]
                .astype(str)
                .map(cluster_suggestions)
                .fillna("Unknown")
            )

            # Save as new modality
            template_modality_name = f"{modality_name}_template_{tissue_type}"
            data_manager.store_modality(
                name=template_modality_name,
                adata=adata_copy,
                parent_name=modality_name,
                step_summary=f"Applied {tissue_type} template: {len(cluster_suggestions)} clusters annotated",
            )

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_template_{tissue_type}.h5ad"
                data_manager.save_modality(template_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="apply_annotation_template",
                parameters={
                    "modality_name": modality_name,
                    "tissue_type": tissue_type,
                    "cluster_key": cluster_key,
                    "expression_threshold": expression_threshold,
                },
                description=f"Applied {tissue_type} template: {len(cluster_suggestions)} clusters annotated",
            )

            # Get template cell types
            template = template_service.get_template(tissue_enum)
            available_types = list(template.keys()) if template else []

            response = f"""Applied {tissue_type.upper()} annotation template to '{modality_name}'!

**Template Application Results:**
- Tissue type: {tissue_type.upper()}
- Clusters analyzed: {len(cluster_suggestions)}
- Expression threshold: {expression_threshold}

**Suggested Annotations:**"""

            # Show suggestions
            suggestion_counts = {}
            for cluster_id, cell_type in cluster_suggestions.items():
                if cell_type not in suggestion_counts:
                    suggestion_counts[cell_type] = []
                suggestion_counts[cell_type].append(cluster_id)

            for cell_type, clusters in suggestion_counts.items():
                response += f"\n- {cell_type}: clusters {', '.join(str(c) for c in sorted(clusters))}"

            response += f"\n\n**Available cell types in {tissue_type} template:**"
            response += f"\n{', '.join(available_types[:8])}"
            if len(available_types) > 8:
                response += f"... and {len(available_types) - 8} more"

            response += f"\n\n**New modality created**: '{template_modality_name}'"
            response += "\n**Template suggestions in**: adata.obs['cell_type_template']"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n\n**Next steps:** Review suggestions and refine with manual annotation if needed."

            return response

        except Exception as e:
            logger.error(f"Error applying annotation template: {e}")
            return f"Error applying annotation template: {str(e)}"

    @tool
    def export_annotation_mapping(
        modality_name: str,
        annotation_col: str = "cell_type_manual",
        output_filename: str = "annotation_mapping.json",
        format: str = "json",
    ) -> str:
        """
        Export annotation mapping for reuse in other analyses.

        Args:
            modality_name: Name of annotated modality
            annotation_col: Column containing cell type annotations
            output_filename: Output filename
            format: Export format ('json' or 'csv')
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            if annotation_col not in adata.obs.columns:
                return f"Annotation column '{annotation_col}' not found."

            # Create export data
            export_data = {
                "modality_name": modality_name,
                "annotation_column": annotation_col,
                "export_timestamp": datetime.datetime.now().isoformat(),
                "total_cells": adata.n_obs,
                "cell_type_mapping": {},
                "cell_type_counts": adata.obs[annotation_col].value_counts().to_dict(),
            }

            # Create cluster-to-celltype mapping if cluster info available
            _known_cluster_names = {
                "leiden", "louvain", "seurat_clusters", "cluster", "clusters",
            }
            cluster_cols = [
                col for col in adata.obs.columns
                if col in _known_cluster_names
                or col.startswith(("leiden_", "louvain_", "RNA_snn_res"))
            ]
            if cluster_cols:
                cluster_col = cluster_cols[0]
                cluster_mapping = {}
                for cluster_id in adata.obs[cluster_col].unique():
                    cluster_mask = adata.obs[cluster_col] == cluster_id
                    most_common_type = (
                        adata.obs.loc[cluster_mask, annotation_col].mode().iloc[0]
                    )
                    cluster_mapping[str(cluster_id)] = most_common_type

                export_data["cluster_to_celltype"] = cluster_mapping
                export_data["cluster_column"] = cluster_col

            # Export based on format
            if format.lower() == "json":
                with open(output_filename, "w") as f:
                    json.dump(export_data, f, indent=2)
            elif format.lower() == "csv":
                # Export as CSV
                df_data = []
                for cell_type, count in export_data["cell_type_counts"].items():
                    df_data.append(
                        {
                            "cell_type": cell_type,
                            "cell_count": count,
                            "percentage": (count / export_data["total_cells"]) * 100,
                        }
                    )

                df = pd.DataFrame(df_data)
                df.to_csv(output_filename, index=False)
            else:
                return f"Unsupported export format: {format}. Use 'json' or 'csv'."

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="export_annotation_mapping",
                parameters={
                    "modality_name": modality_name,
                    "annotation_col": annotation_col,
                    "output_filename": output_filename,
                    "format": format,
                },
                description=f"Exported annotation mapping with {len(export_data['cell_type_counts'])} cell types",
            )

            response = f"""Successfully exported annotation mapping for '{modality_name}'!

**Export Details:**
- Annotation column: {annotation_col}
- Output file: {output_filename}
- Format: {format.upper()}
- Cell types: {len(export_data["cell_type_counts"])}

**Exported Data:**
- Total cells: {export_data["total_cells"]:,}
- Cell type counts included
- Cluster mapping included (if available)
- Export timestamp: {export_data["export_timestamp"]}

**File created**: {output_filename}

Use this mapping to apply consistent annotations to similar datasets."""

            return response

        except Exception as e:
            logger.error(f"Error exporting annotation mapping: {e}")
            return f"Error exporting annotation mapping: {str(e)}"

    @tool
    def import_annotation_mapping(
        modality_name: str,
        mapping_file: str,
        preview_only: bool = False,
        save_result: bool = True,
    ) -> str:
        """
        Import and apply annotation mapping from previous analysis.

        Args:
            modality_name: Name of modality to annotate
            mapping_file: Path to mapping file (JSON format)
            preview_only: If True, only show what would be applied
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Load mapping file
            with open(mapping_file, "r") as f:
                mapping_data = json.load(f)

            if preview_only:
                response = f"""Preview of annotation mapping from '{mapping_file}':

**Mapping File Details:**
- Source modality: {mapping_data.get("modality_name", "N/A")}
- Annotation column: {mapping_data.get("annotation_column", "N/A")}
- Export timestamp: {mapping_data.get("export_timestamp", "N/A")}

**Cell Types in Mapping:**"""

                for cell_type, count in mapping_data.get(
                    "cell_type_counts", {}
                ).items():
                    response += f"\n- {cell_type}: {count} cells"

                if "cluster_to_celltype" in mapping_data:
                    response += "\n\n**Cluster Mappings:**"
                    cluster_mapping = mapping_data["cluster_to_celltype"]
                    for cluster_id, cell_type in list(cluster_mapping.items())[:10]:
                        response += f"\n- Cluster {cluster_id}: {cell_type}"

                    if len(cluster_mapping) > 10:
                        response += (
                            f"\n... and {len(cluster_mapping) - 10} more clusters"
                        )

                response += f"\n\nUse preview_only=False to apply this mapping to '{modality_name}'."
                return response

            # Apply mapping
            adata_copy = adata.copy()

            if (
                "cluster_to_celltype" in mapping_data
                and "cluster_column" in mapping_data
            ):
                cluster_col = mapping_data["cluster_column"]
                cluster_mapping = mapping_data["cluster_to_celltype"]

                if cluster_col in adata_copy.obs.columns:
                    adata_copy.obs["cell_type_imported"] = (
                        adata_copy.obs[cluster_col]
                        .astype(str)
                        .map(cluster_mapping)
                        .fillna("Unassigned")
                    )
                else:
                    return f"Cluster column '{cluster_col}' from mapping not found in modality."
            else:
                return "Mapping file does not contain cluster-to-celltype information."

            # Save as new modality
            imported_modality_name = f"{modality_name}_imported_annotations"
            data_manager.store_modality(
                name=imported_modality_name,
                adata=adata_copy,
                parent_name=modality_name,
                step_summary=f"Imported annotation mapping from {mapping_file}",
            )

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_imported_annotations.h5ad"
                data_manager.save_modality(imported_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="import_annotation_mapping",
                parameters={
                    "modality_name": modality_name,
                    "mapping_file": mapping_file,
                    "preview_only": preview_only,
                },
                description=f"Imported annotation mapping from {mapping_file}",
            )

            # Validate imported annotations
            validation = manual_annotation_service.validate_annotation_coverage(
                adata_copy, "cell_type_imported"
            )

            response = f"""Successfully imported annotation mapping to '{modality_name}'!

**Import Results:**
- Mapping file: {mapping_file}
- Clusters mapped: {len(cluster_mapping)}
- Coverage: {validation["coverage_percentage"]:.1f}%

**Imported Cell Types:**"""

            for cell_type, count in list(validation["cell_type_counts"].items())[:8]:
                response += f"\n- {cell_type}: {count:,} cells"

            response += f"\n\n**New modality created**: '{imported_modality_name}'"
            response += "\n**Imported annotations in**: adata.obs['cell_type_imported']"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            return response

        except FileNotFoundError:
            return f"Mapping file not found: {mapping_file}"
        except Exception as e:
            logger.error(f"Error importing annotation mapping: {e}")
            return f"Error importing annotation mapping: {str(e)}"

    # -------------------------
    # GENE SET SCORING
    # -------------------------

    @tool
    def score_gene_set(
        modality_name: str,
        gene_list: list,
        score_name: str = "gene_set_score",
        ctrl_size: int = 50,
        use_raw: bool = False,
    ) -> str:
        """
        Score cells for expression of a user-provided gene set.

        Uses scanpy's score_genes to compute per-cell gene set activity scores,
        stored in adata.obs[score_name]. Useful for validating cell type annotations,
        identifying pathway activity, or scoring custom gene signatures.

        Args:
            modality_name: Name of the single-cell modality
            gene_list: List of gene names to score (e.g., ["CD3D", "CD3E", "CD4"])
            score_name: Name for the score column in adata.obs (default: "gene_set_score")
            ctrl_size: Number of reference genes for background (default: 50)
            use_raw: Whether to use raw counts for scoring (default: False)
        """
        import scanpy as sc

        from lobster.core.analysis_ir import AnalysisStep, ParameterSpec

        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            # Filter gene_list to genes present in the data
            all_genes = set(adata.var_names)
            valid_genes = [g for g in gene_list if g in all_genes]
            missing_genes = [g for g in gene_list if g not in all_genes]

            if not valid_genes:
                return (
                    f"Error: None of the provided genes were found in the data.\n"
                    f"Missing genes: {missing_genes[:20]}\n"
                    f"Check gene naming convention (e.g., symbols vs Ensembl IDs)."
                )

            # Score the gene set
            sc.tl.score_genes(
                adata,
                gene_list=valid_genes,
                score_name=score_name,
                ctrl_size=ctrl_size,
                use_raw=use_raw,
            )

            # Compute score statistics
            scores = adata.obs[score_name]
            stats = {
                "n_genes_scored": len(valid_genes),
                "n_genes_missing": len(missing_genes),
                "score_mean": float(scores.mean()),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "score_std": float(scores.std()),
            }

            # Store result via store_modality
            scored_modality_name = f"{modality_name}_scored"
            data_manager.store_modality(
                name=scored_modality_name,
                adata=adata,
                parent_name=modality_name,
                step_summary=f"Scored {len(valid_genes)} genes as '{score_name}'",
            )

            # Build parameters dict
            params = {
                "modality_name": modality_name,
                "gene_list": valid_genes,
                "score_name": score_name,
                "ctrl_size": ctrl_size,
                "use_raw": use_raw,
            }

            # Create AnalysisStep IR
            ir = AnalysisStep(
                operation="scanpy.tl.score_genes",
                tool_name="score_gene_set",
                description=(
                    f"Scored {len(valid_genes)} genes as '{score_name}' "
                    f"(mean={stats['score_mean']:.4f})"
                ),
                library="scanpy",
                code_template=(
                    "import scanpy as sc\n"
                    "sc.tl.score_genes(\n"
                    "    adata,\n"
                    '    gene_list={{ gene_list }},\n'
                    '    score_name="{{ score_name }}",\n'
                    "    ctrl_size={{ ctrl_size }},\n"
                    "    use_raw={{ use_raw }},\n"
                    ")"
                ),
                imports=["import scanpy as sc"],
                parameters=params,
                parameter_schema={
                    "modality_name": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value="",
                        required=True,
                        description="Name of single-cell modality",
                    ),
                    "gene_list": ParameterSpec(
                        param_type="list",
                        papermill_injectable=True,
                        default_value=[],
                        required=True,
                        description="List of gene names to score",
                    ),
                    "score_name": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value="gene_set_score",
                        required=False,
                        description="Name for score column in adata.obs",
                    ),
                    "ctrl_size": ParameterSpec(
                        param_type="int",
                        papermill_injectable=True,
                        default_value=50,
                        required=False,
                        description="Number of reference genes for background",
                    ),
                    "use_raw": ParameterSpec(
                        param_type="bool",
                        papermill_injectable=True,
                        default_value=False,
                        required=False,
                        description="Whether to use raw counts",
                    ),
                },
                input_entities=["adata"],
                output_entities=["adata_scored"],
            )

            # Log with IR
            data_manager.log_tool_usage("score_gene_set", params, stats, ir=ir)

            # Format response
            response = (
                f"Successfully scored gene set in '{modality_name}'!\n\n"
                f"**Gene Set Scoring Results:**\n"
                f"- Genes scored: {len(valid_genes)} / {len(gene_list)}\n"
                f"- Score column: adata.obs['{score_name}']\n\n"
                f"**Score Statistics:**\n"
                f"- Mean: {stats['score_mean']:.4f}\n"
                f"- Min: {stats['score_min']:.4f}\n"
                f"- Max: {stats['score_max']:.4f}\n"
                f"- Std: {stats['score_std']:.4f}\n"
            )

            if missing_genes:
                response += (
                    f"\n**Missing genes** ({len(missing_genes)}): "
                    f"{', '.join(missing_genes[:10])}"
                )
                if len(missing_genes) > 10:
                    response += f" ... and {len(missing_genes) - 10} more"

            response += f"\n\n**New modality created**: '{scored_modality_name}'"

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in gene set scoring: {e}")
            return f"Error scoring gene set: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in gene set scoring: {e}")
            return f"Unexpected error in gene set scoring: {str(e)}"

    # -------------------------
    # SEMANTIC ANNOTATION (optional - requires vector-search deps)
    # -------------------------

    _vector_service = None

    def _get_vector_service():
        nonlocal _vector_service
        if _vector_service is None:
            from lobster.services.vector.service import VectorSearchService

            _vector_service = VectorSearchService()
        return _vector_service

    @tool
    def annotate_cell_types_semantic(
        modality_name: str,
        cluster_key: str = "leiden",
        k: int = 5,
        min_confidence: float = 0.5,
        validate_graph: bool = False,
        save_result: bool = True,
    ) -> str:
        """
        Annotate single-cell clusters with cell types using semantic search against Cell Ontology.

        Uses SapBERT embeddings to match marker gene signatures to Cell Ontology terms,
        providing confidence-scored cell type assignments. Complements existing keyword-based
        annotation with ontology-grounded semantic matching.

        IMPORTANT: Requires clustered data with marker gene information. Run clustering
        and marker gene detection first. Use check_data_status() to verify prerequisites.

        Args:
            modality_name: Name of the single-cell modality with clustering results
            cluster_key: Column name in adata.obs containing cluster assignments
            k: Number of candidate ontology matches per cluster (default 5)
            min_confidence: Minimum similarity score threshold (0-1) for accepting a match
            validate_graph: If True, validate top matches via ontology graph traversal
            save_result: Whether to save annotated modality
        """
        from lobster.core.analysis_ir import AnalysisStep, ParameterSpec

        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            # Validate cluster_key exists
            if cluster_key not in adata.obs.columns:
                available_cols = list(adata.obs.columns)
                return (
                    f"Cluster column '{cluster_key}' not found.\n\n"
                    f"Available columns: {available_cols}\n\n"
                    f"Use check_data_status() to identify the correct cluster column."
                )

            logger.info(
                f"Semantic annotation of '{modality_name}': {adata.shape[0]} cells, "
                f"cluster_key='{cluster_key}', k={k}, min_confidence={min_confidence}"
            )

            # Extract marker scores using the singlecell service
            marker_scores = singlecell_service._calculate_marker_scores_from_adata(
                adata, singlecell_service.cell_type_markers, cluster_key=cluster_key
            )

            # Build text queries from marker scores for each cluster
            vs = _get_vector_service()
            unique_clusters = sorted(adata.obs[cluster_key].astype(str).unique())
            cluster_annotations = {}

            for cluster_id in unique_clusters:
                # Get top 3 cell types by score for this cluster
                scores = marker_scores.get(cluster_id, {})
                sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

                # Collect top marker genes from those cell types (max 5 total)
                query_genes = []
                for cell_type_name, _ in sorted_types:
                    markers_for_type = singlecell_service.cell_type_markers.get(
                        cell_type_name, []
                    )
                    for gene in markers_for_type[:3]:
                        if gene not in query_genes:
                            query_genes.append(gene)
                        if len(query_genes) >= 5:
                            break
                    if len(query_genes) >= 5:
                        break

                # Format query text
                if query_genes:
                    query_text = f"Cluster {cluster_id}: high {', '.join(query_genes)}"
                else:
                    query_text = f"Cluster {cluster_id}: unknown markers"

                # Search Cell Ontology
                matches = vs.match_ontology(query_text, "cell_ontology", k=k)

                # Filter by min_confidence
                valid_matches = [m for m in matches if m.score >= min_confidence]

                if valid_matches:
                    top_match = valid_matches[0]

                    # Optional graph validation
                    if validate_graph and top_match.ontology_id:
                        try:
                            from lobster.services.vector.ontology_graph import (
                                get_neighbors,
                                load_ontology_graph,
                            )

                            graph = load_ontology_graph("cell_ontology")
                            neighbors = get_neighbors(graph, top_match.ontology_id)
                            has_neighbors = bool(
                                neighbors.get("parents") or neighbors.get("children")
                            )
                            if not has_neighbors:
                                logger.warning(
                                    f"Cluster {cluster_id}: top match '{top_match.term}' "
                                    f"({top_match.ontology_id}) has no graph neighbors, "
                                    f"skipping validation"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Graph validation failed for cluster {cluster_id}: {e}"
                            )

                    cluster_annotations[cluster_id] = {
                        "term": top_match.term,
                        "ontology_id": top_match.ontology_id,
                        "score": top_match.score,
                    }
                else:
                    cluster_annotations[cluster_id] = None

            # Apply annotations to adata copy
            adata_annotated = adata.copy()
            n_annotated = 0
            n_unknown = 0
            confidence_values = []

            for cluster_id in unique_clusters:
                mask = adata_annotated.obs[cluster_key].astype(str) == cluster_id
                annotation = cluster_annotations.get(cluster_id)

                if annotation is not None:
                    adata_annotated.obs.loc[mask, "cell_type"] = annotation["term"]
                    adata_annotated.obs.loc[mask, "cell_type_confidence"] = annotation[
                        "score"
                    ]
                    n_annotated += 1
                    confidence_values.append(annotation["score"])
                else:
                    adata_annotated.obs.loc[mask, "cell_type"] = "Unknown"
                    adata_annotated.obs.loc[mask, "cell_type_confidence"] = 0.0
                    n_unknown += 1

            mean_confidence = (
                round(sum(confidence_values) / len(confidence_values), 4)
                if confidence_values
                else 0.0
            )

            # Store annotated modality
            if save_result:
                annotated_modality_name = f"{modality_name}_semantic_annotated"
                data_manager.modalities[annotated_modality_name] = adata_annotated

            # Build stats dict
            stats = {
                "n_clusters_annotated": n_annotated,
                "n_unknown": n_unknown,
                "mean_confidence": mean_confidence,
                "cluster_annotations": cluster_annotations,
            }

            # Build parameters dict
            params = {
                "modality_name": modality_name,
                "cluster_key": cluster_key,
                "k": k,
                "min_confidence": min_confidence,
                "validate_graph": validate_graph,
                "save_result": save_result,
            }

            # Create AnalysisStep IR
            ir = AnalysisStep(
                operation="semantic_cell_type_annotation",
                tool_name="annotate_cell_types_semantic",
                description=(
                    f"Semantic cell type annotation of {n_annotated} clusters "
                    f"via Cell Ontology vector search (mean confidence: {mean_confidence})"
                ),
                library="lobster.services.vector",
                code_template=(
                    "from lobster.services.vector.service import VectorSearchService\n"
                    "vs = VectorSearchService()\n"
                    "# For each cluster, build query from marker genes and match to Cell Ontology\n"
                    "matches = vs.match_ontology(\n"
                    '    "{{ query_text }}", "cell_ontology", k={{ k }}\n'
                    ")\n"
                    "# Filter by min_confidence={{ min_confidence }}\n"
                    '# Apply top match to adata.obs["cell_type"]'
                ),
                imports=[
                    "from lobster.services.vector.service import VectorSearchService",
                ],
                parameters=params,
                parameter_schema={
                    "modality_name": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value="",
                        required=True,
                        description="Name of single-cell modality",
                    ),
                    "cluster_key": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value="leiden",
                        required=False,
                        description="Cluster assignment column in adata.obs",
                    ),
                    "k": ParameterSpec(
                        param_type="int",
                        papermill_injectable=True,
                        default_value=5,
                        required=False,
                        description="Number of ontology candidates per cluster",
                    ),
                    "min_confidence": ParameterSpec(
                        param_type="float",
                        papermill_injectable=True,
                        default_value=0.5,
                        required=False,
                        description="Minimum similarity score threshold",
                    ),
                },
                input_entities=["adata"],
                output_entities=["adata_annotated"],
                execution_context={
                    "ontology": "cell_ontology",
                    "n_clusters": len(unique_clusters),
                },
            )

            # Log with IR
            data_manager.log_tool_usage(
                "annotate_cell_types_semantic", params, stats, ir=ir
            )

            # Format response
            response = (
                f"Successfully annotated cell types in '{modality_name}' "
                f"using semantic Cell Ontology matching!\n\n"
                f"**Semantic Annotation Results:**\n"
                f"- Clusters annotated: {n_annotated}\n"
                f"- Unknown clusters: {n_unknown}\n"
                f"- Mean confidence: {mean_confidence}\n\n"
                f"**Cluster Annotations:**"
            )

            for cid in unique_clusters:
                ann = cluster_annotations.get(cid)
                if ann is not None:
                    response += (
                        f"\n- Cluster {cid}: {ann['term']} "
                        f"({ann['ontology_id']}, score={ann['score']})"
                    )
                else:
                    response += f"\n- Cluster {cid}: Unknown (below threshold)"

            if save_result:
                response += (
                    f"\n\n**New modality created**: "
                    f"'{modality_name}_semantic_annotated'"
                )

            response += "\n**Cell type annotations in**: adata.obs['cell_type']"
            response += (
                "\n**Confidence scores in**: adata.obs['cell_type_confidence']"
            )

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in semantic cell type annotation: {e}")
            return f"Error in semantic annotation: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in semantic cell type annotation: {e}")
            return f"Unexpected error in semantic annotation: {str(e)}"

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        # Automated annotation
        annotate_cell_types_auto,
        # Gene set scoring
        score_gene_set,
        # Manual annotation
        manually_annotate_clusters,
        collapse_clusters_to_celltype,
        mark_clusters_as_debris,
        suggest_debris_clusters,
        # Annotation management
        review_annotation_assignments,
        apply_annotation_template,
        export_annotation_mapping,
        import_annotation_mapping,
    ]

    # Conditionally add semantic annotation tool when vector-search deps available
    if HAS_VECTOR_SEARCH:
        base_tools.append(annotate_cell_types_semantic)

    tools = base_tools + (delegation_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = create_annotation_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=AnnotationExpertState,
    )
