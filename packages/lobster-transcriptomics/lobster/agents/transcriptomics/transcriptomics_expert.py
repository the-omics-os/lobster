"""
Transcriptomics Expert Parent Agent for orchestrating single-cell and bulk RNA-seq analysis.

This agent serves as the main orchestrator for transcriptomics analysis, with:
- Shared QC tools (from shared_tools.py) available directly
- Clustering tools (SC-specific) available directly
- Delegation to annotation_expert for cell type annotation
- Delegation to de_analysis_expert for differential expression analysis

The agent auto-detects single-cell vs bulk data and adapts its behavior accordingly.
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="transcriptomics_expert",
    display_name="Transcriptomics Expert",
    description="Unified expert for single-cell AND bulk RNA-seq analysis. Handles QC, clustering, and orchestrates annotation and DE analysis via specialized sub-agents.",
    factory_function="lobster.agents.transcriptomics.transcriptomics_expert.transcriptomics_expert",
    handoff_tool_name="handoff_to_transcriptomics_expert",
    handoff_tool_description="Assign ALL transcriptomics analysis tasks (single-cell OR bulk RNA-seq): QC, clustering, cell type annotation, differential expression, pseudobulk, pathway enrichment/functional analysis (GO/KEGG/Reactome gene set enrichment)",
    child_agents=["annotation_expert", "de_analysis_expert"],
)

# === Heavy imports below ===
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.transcriptomics.prompts import create_transcriptomics_expert_prompt
from lobster.agents.transcriptomics.shared_tools import create_shared_tools
from lobster.agents.transcriptomics.state import TranscriptomicsExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.analysis_ir import AnalysisStep
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.bulk_preprocessing_service import (
    BulkPreprocessingService,
)
from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService
from lobster.services.analysis.clustering_service import (
    ClusteringError,
    ClusteringService,
)
from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
)
from lobster.services.analysis.enhanced_singlecell_service import (
    SingleCellError as ServiceSingleCellError,
)
from lobster.services.quality.preprocessing_service import PreprocessingService
from lobster.services.quality.quality_service import QualityService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class TranscriptomicsAgentError(Exception):
    """Base exception for transcriptomics agent operations."""

    pass


class ModalityNotFoundError(TranscriptomicsAgentError):
    """Raised when requested modality doesn't exist."""

    pass


def transcriptomics_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "transcriptomics_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for transcriptomics expert parent agent.

    This agent orchestrates single-cell and bulk RNA-seq analysis.
    It has QC and clustering tools directly, and delegates annotation
    and DE analysis to specialized sub-agents.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: List of delegation tools for sub-agents (annotation_expert, de_analysis_expert)

    Returns:
        Configured ReAct agent with transcriptomics analysis capabilities
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("transcriptomics_expert")
    llm = create_llm(
        "transcriptomics_expert",
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

    # Initialize services
    quality_service = QualityService()
    preprocessing_service = PreprocessingService()
    clustering_service = ClusteringService()
    enhanced_service = EnhancedSingleCellService()
    bulk_service = BulkRNASeqService(data_manager=data_manager)
    bulk_preprocessing_service = BulkPreprocessingService()

    # Get shared tools (QC, preprocessing, feature selection, PCA, embedding, analysis summary)
    shared_tools = create_shared_tools(
        data_manager, quality_service, preprocessing_service,
        clustering_service=clustering_service,
    )

    # Analysis results storage (for clustering tools)
    analysis_results = {"summary": "", "details": {}}

    # =========================================================================
    # CLUSTERING TOOLS
    # =========================================================================

    @tool
    def cluster_cells(
        modality_name: str,
        resolution: float = None,
        resolutions: Optional[List[float]] = None,
        use_rep: Optional[str] = None,
        batch_correction: bool = True,
        batch_key: str = None,
        demo_mode: bool = False,
        save_result: bool = True,
        feature_selection_method: str = "deviance",
        n_features: int = 4000,
    ) -> str:
        """
        Perform single-cell clustering and UMAP visualization.

        Args:
            modality_name: Name of the single-cell modality to cluster
            resolution: Single Leiden clustering resolution (0.1-2.0, higher = more clusters).
                       Use this for single-resolution clustering. Default: 1.0 if neither resolution nor resolutions specified.
            resolutions: List of resolutions for multi-resolution testing (e.g., [0.25, 0.5, 1.0]).
                        Creates multiple clustering results with descriptive keys (leiden_res0_25, leiden_res0_5, leiden_res1_0).
                        Use this to explore clustering granularity. If specified, overrides 'resolution' parameter.
            use_rep: Representation to use for clustering (e.g., 'X_scvi' for deep learning embeddings, 'X_pca' for PCA).
                    If None, uses standard PCA workflow. Custom embeddings like scVI often provide better results.
            batch_correction: Whether to perform batch correction for multi-sample data
            batch_key: Column name for batch information (auto-detected if None)
            demo_mode: Use faster processing for large single-cell datasets (>50k cells)
            save_result: Whether to save the clustered modality
            feature_selection_method: Method for feature selection ('deviance' or 'hvg').
                                     'deviance' (default, recommended): Binomial deviance from multinomial null, works on raw counts, no normalization bias.
                                     'hvg': Traditional highly variable genes (Seurat method), works on normalized data.
            n_features: Number of features to select (default: 4000)

        Returns:
            str: Formatted report with clustering results and cluster distribution

        Examples:
            # Single resolution clustering (traditional)
            cluster_cells("geo_gse12345_filtered", resolution=0.5)

            # Multi-resolution testing (recommended for exploration)
            cluster_cells("geo_gse12345_filtered", resolutions=[0.25, 0.5, 1.0])

            # Using scVI embeddings
            cluster_cells("geo_gse12345_filtered", resolutions=[0.5, 1.0], use_rep="X_scvi")
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
                f"Clustering single-cell modality '{modality_name}': {adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Use clustering service
            adata_clustered, clustering_stats, ir = (
                clustering_service.cluster_and_visualize(
                    adata=adata,
                    resolution=resolution,
                    resolutions=resolutions,
                    use_rep=use_rep,
                    batch_correction=batch_correction,
                    batch_key=batch_key,
                    demo_mode=demo_mode,
                    feature_selection_method=feature_selection_method,
                    n_features=n_features,
                )
            )

            # Save as new modality
            clustered_modality_name = f"{modality_name}_clustered"
            data_manager.store_modality(
                name=clustered_modality_name,
                adata=adata_clustered,
                parent_name=modality_name,
                step_summary=f"Clustered into {clustering_stats['n_clusters']} clusters",
            )

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_clustered.h5ad"
                data_manager.save_modality(clustered_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="cluster_cells",
                parameters={
                    "modality_name": modality_name,
                    "resolution": resolution,
                    "resolutions": resolutions,
                    "batch_correction": batch_correction,
                    "demo_mode": demo_mode,
                    "feature_selection_method": feature_selection_method,
                    "n_features": n_features,
                },
                description=f"Single-cell clustered {modality_name} into {clustering_stats['n_clusters']} clusters using {feature_selection_method} feature selection",
                ir=ir,
            )

            # Format professional response
            response = f"""Successfully clustered single-cell modality '{modality_name}'!

**Single-cell Clustering Results:**"""

            # Check if multi-resolution testing was performed
            if clustering_stats.get("n_resolutions", 1) > 1:
                response += (
                    f"\n- Resolutions tested: {clustering_stats['resolutions_tested']}"
                )
                response += (
                    "\n- Cluster columns (use these exact names for visualization):"
                )
                for res, n_clusters in clustering_stats.get(
                    "multi_resolution_summary", {}
                ).items():
                    key_name = f"leiden_res{res}".replace(".", "_")
                    response += (
                        f"\n  - `{key_name}` (resolution={res}): {n_clusters} clusters"
                    )
            else:
                # Single resolution mode (existing behavior)
                response += f"\n- Number of clusters: {clustering_stats['n_clusters']}"
                response += f"\n- Leiden resolution: {clustering_stats.get('resolution', 'N/A')}"
                response += "\n- Cluster column name: `leiden` (use this exact name for visualization)"

            # Continue with common details
            response += f"\n- UMAP coordinates: {'Yes' if clustering_stats['has_umap'] else 'No'}"
            response += f"\n- Marker genes: {'Yes' if clustering_stats['has_marker_genes'] else 'No'}"

            response += f"""

**Processing Details:**
- Original shape: {clustering_stats["original_shape"][0]} x {clustering_stats["original_shape"][1]}
- Final shape: {clustering_stats["final_shape"][0]} x {clustering_stats["final_shape"][1]}
- Feature selection: {feature_selection_method} ({n_features} features)
- Batch correction: {"Yes" if clustering_stats["batch_correction"] else "No"}
- Demo mode: {"Yes" if clustering_stats["demo_mode"] else "No"}

**Cluster Distribution:**"""

            # Add cluster size information
            for cluster_id, size in list(clustering_stats["cluster_sizes"].items())[:8]:
                percentage = (size / clustering_stats["final_shape"][0]) * 100
                response += (
                    f"\n- Cluster {cluster_id}: {size} cells ({percentage:.1f}%)"
                )

            if len(clustering_stats["cluster_sizes"]) > 8:
                response += f"\n... and {len(clustering_stats['cluster_sizes']) - 8} more clusters"

            response += f"\n\n**New modality created**: '{clustered_modality_name}'"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            # Add multi-resolution guidance if applicable
            if clustering_stats.get("n_resolutions", 1) > 1:
                response += "\n\n**Multi-Resolution Analysis:**"
                response += "\n- Compare clustering results across resolutions using visualization"
                response += (
                    "\n- Lower resolutions (0.25-0.5) identify major cell populations"
                )
                response += "\n- Higher resolutions (1.0-2.0) reveal finer cell states"
                response += "\n- Choose resolution based on biological expectations and marker gene validation"

            response += "\n\n**Next steps**: find_marker_genes(), then INVOKE handoff_to_annotation_expert if annotation requested."

            analysis_results["details"]["clustering"] = response
            return response

        except (ClusteringError, ModalityNotFoundError) as e:
            logger.error(f"Error in single-cell clustering: {e}")
            return f"Error clustering single-cell modality: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in single-cell clustering: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def subcluster_cells(
        modality_name: str,
        cluster_key: str,
        clusters_to_refine: Optional[List[str]] = None,
        resolution: float = 0.5,
        resolutions: Optional[List[float]] = None,
        n_pcs: int = 20,
        n_neighbors: int = 15,
        demo_mode: bool = False,
    ) -> str:
        """
        Re-cluster specific cell subsets for finer-grained population identification.

        Useful when initial clustering groups heterogeneous populations and you want
        to refine specific clusters without affecting others. Common scenarios:
        - "Split cluster 0 into subtypes"
        - "Refine the T cell clusters"
        - "Sub-cluster the heterogeneous populations"

        IMPORTANT: Call check_data_status() first to identify the actual cluster column name.
        NEVER assume 'leiden' — data may use 'seurat_clusters', 'louvain', 'RNA_snn_res.1', etc.

        Args:
            modality_name: Name of the modality to sub-cluster
            cluster_key: Key in adata.obs containing cluster assignments (REQUIRED).
                        Common values: 'leiden', 'louvain', 'seurat_clusters', 'RNA_snn_res.1'.
                        Use check_data_status() to find the correct column name.
            clusters_to_refine: List of cluster IDs to re-cluster (e.g., ["0", "3", "5"])
                               If None, re-clusters ALL cells (full re-clustering)
            resolution: Single resolution for sub-clustering (default: 0.5)
                       Typical range: 0.1-2.0 (higher = more clusters)
            resolutions: List of resolutions for multi-resolution testing (e.g., [0.25, 0.5, 1.0])
                        Creates multiple sub-clustering results with descriptive keys
                        Use for exploring different granularities
            n_pcs: Number of PCs for sub-clustering (default: 20, fewer than full clustering)
                   Typical range: 15-30
            n_neighbors: Number of neighbors for KNN graph (default: 15)
                        Typical range: 10-30
            demo_mode: Use faster parameters for testing (default: False)

        Returns:
            str: Summary of sub-clustering results including cluster sizes and new column names

        Examples:
            # Sub-cluster a single cluster (after checking data status)
            subcluster_cells("geo_gse12345_clustered", cluster_key="leiden", clusters_to_refine=["0"])

            # Sub-cluster with Seurat-style clusters
            subcluster_cells("my_data", cluster_key="seurat_clusters", clusters_to_refine=["0", "3"])

            # Multi-resolution sub-clustering
            subcluster_cells("geo_gse12345_clustered", cluster_key="leiden",
                            clusters_to_refine=["0"], resolutions=[0.25, 0.5, 1.0])

            # Full re-clustering (all cells)
            subcluster_cells("geo_gse12345_clustered", cluster_key="leiden", clusters_to_refine=None)
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
                f"Sub-clustering modality '{modality_name}': {adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Perform sub-clustering using service
            result, stats, ir = clustering_service.subcluster_cells(
                adata,
                cluster_key=cluster_key,
                clusters_to_refine=clusters_to_refine,
                resolution=resolution,
                resolutions=resolutions,
                n_pcs=n_pcs,
                n_neighbors=n_neighbors,
                demo_mode=demo_mode,
            )

            # Compute cluster count safely (BUG-01 fix: clusters_to_refine may be None)
            n_refined = len(clusters_to_refine) if clusters_to_refine else "all"

            # Store result with descriptive suffix
            new_name = f"{modality_name}_subclustered"
            data_manager.store_modality(
                name=new_name,
                adata=result,
                parent_name=modality_name,
                step_summary=f"Subclustered {n_refined} clusters from {cluster_key}",
            )

            # Log with IR (mandatory for reproducibility)
            data_manager.log_tool_usage(
                "subcluster_cells",
                {
                    "cluster_key": cluster_key,
                    "clusters_to_refine": clusters_to_refine,
                    "resolution": resolution,
                    "resolutions": resolutions,
                    "n_pcs": n_pcs,
                    "n_neighbors": n_neighbors,
                    "demo_mode": demo_mode,
                },
                description=f"Subclustered {n_refined} clusters from {cluster_key}",
                ir=ir,
            )

            # Format response based on single vs multi-resolution
            n_resolutions_tested = len(stats.get("resolutions_tested", [resolution]))

            if n_resolutions_tested > 1:
                # Multi-resolution formatting
                response = f"""Sub-clustering complete! Created '{new_name}' modality.

**Results:**
- Processed {stats["n_cells_subclustered"]:,} cells from {len(stats["parent_clusters"])} parent cluster(s): {stats["parent_clusters"]}
- Tested {n_resolutions_tested} resolutions: {stats["resolutions_tested"]}
- New columns in adata.obs:"""

                # Show all resolution results
                for res_key, n_clusters in stats["multi_resolution_summary"].items():
                    primary_marker = (
                        " (primary)" if res_key == stats["primary_column"] else ""
                    )
                    response += (
                        f"\n  * {res_key}: {n_clusters} sub-clusters{primary_marker}"
                    )

                response += (
                    f"\n- Execution time: {stats['execution_time']:.2f} seconds\n"
                )

                # Show primary sub-clustering results
                response += (
                    f"\n**Primary sub-clustering ({stats['primary_column']}):**\n"
                )
                for cluster_id, size in list(stats["subcluster_sizes"].items())[:10]:
                    response += f"  - {cluster_id}: {size} cells\n"

                if len(stats["subcluster_sizes"]) > 10:
                    remaining = len(stats["subcluster_sizes"]) - 10
                    response += f"  ... and {remaining} more sub-clusters\n"

                response += """
**Interpretation:**
- Lower resolutions (0.25) = broader populations
- Higher resolutions (1.0) = finer-grained clusters
- Compare results across resolutions to determine optimal granularity"""

            else:
                # Single-resolution formatting
                response = f"""Sub-clustering complete! Created '{new_name}' modality.

**Results:**
- Processed {stats["n_cells_subclustered"]:,} cells from {len(stats["parent_clusters"])} parent cluster(s): {stats["parent_clusters"]}
- Generated {stats["n_subclusters"]} sub-clusters at resolution {stats.get("resolution", resolution)}
- New column: '{stats["primary_column"]}' in adata.obs
- Execution time: {stats["execution_time"]:.2f} seconds

**Sub-cluster sizes:**"""

                for cluster_id, size in list(stats["subcluster_sizes"].items())[:10]:
                    response += f"\n  - {cluster_id}: {size} cells"

                if len(stats["subcluster_sizes"]) > 10:
                    remaining = len(stats["subcluster_sizes"]) - 10
                    response += f"\n  ... and {remaining} more sub-clusters"

                response += """

**Next steps:**
- Use visualization to display sub-clusters on UMAP
- Use find_marker_genes() to characterize each sub-cluster
- INVOKE handoff_to_annotation_expert immediately if annotation requested (do NOT just suggest it)"""

            analysis_results["details"]["sub_clustering"] = response
            return response

        except ValueError as e:
            # User-friendly error messages for validation failures
            return f"""Error: {str(e)}

Please check:
- Cluster key '{cluster_key}' exists in adata.obs
- Cluster IDs in clusters_to_refine are valid
- Initial clustering has been performed"""
        except (ClusteringError, ModalityNotFoundError) as e:
            logger.error(f"Error in sub-clustering: {e}")
            return f"Error sub-clustering modality: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in sub-clustering: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def evaluate_clustering_quality(
        modality_name: str,
        cluster_key: str,
        use_rep: str = "X_pca",
        n_pcs: Optional[int] = None,
        metrics: Optional[List[str]] = None,
    ) -> str:
        """
        Evaluate clustering quality using multiple quantitative metrics.

        Computes 3 scientifically-validated metrics to assess clustering results:
        - Silhouette score: How well-separated clusters are (-1 to 1, higher better)
        - Davies-Bouldin index: Ratio of intra/inter-cluster distances (lower better)
        - Calinski-Harabasz score: Ratio of between/within variance (higher better)

        These metrics help answer:
        - "Is my clustering good?"
        - "Which resolution gives the best separation?"
        - "Am I over-clustering or under-clustering?"

        IMPORTANT: Call check_data_status() first to identify the actual cluster column name.
        NEVER assume 'leiden' — data may use 'seurat_clusters', 'louvain', 'RNA_snn_res.1', etc.

        Args:
            modality_name: Name of the modality to evaluate
            cluster_key: Key in adata.obs containing cluster labels (REQUIRED).
                        Common values: 'leiden', 'louvain', 'seurat_clusters', 'leiden_res0_5'.
                        Use check_data_status() to find the correct column name.
            use_rep: Representation to use for distance calculations (default: "X_pca")
                    Options: "X_pca", "X_umap", or any key in adata.obsm
            n_pcs: Number of PCs to use (default: None = use all available)
                   Recommended: 30 for full clustering, 20 for sub-clustering
            metrics: List of specific metrics to compute (default: None = all 3)
                    Options: ["silhouette", "davies_bouldin", "calinski_harabasz"]

        Returns:
            str: Detailed quality report with scores, interpretations, and recommendations

        Examples:
            # Evaluate single clustering result
            evaluate_clustering_quality("geo_gse12345_clustered", cluster_key="leiden")

            # Compare multiple resolutions
            evaluate_clustering_quality("geo_gse12345_clustered", cluster_key="leiden_res0_25")
            evaluate_clustering_quality("geo_gse12345_clustered", cluster_key="leiden_res0_5")

            # Evaluate with Seurat-style clusters
            evaluate_clustering_quality("my_data", cluster_key="seurat_clusters")

            # Compute only silhouette score
            evaluate_clustering_quality("geo_gse12345_clustered", cluster_key="leiden", metrics=["silhouette"])
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                available = ", ".join(data_manager.list_modalities()[:5])
                return (
                    f"Error: Modality '{modality_name}' not found.\n\n"
                    f"Available modalities: {available}...\n\n"
                    f"Use check_data_status() to see all available modalities."
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)

            # Validate cluster_key exists
            if cluster_key not in adata.obs.columns:
                available_cols = list(adata.obs.columns)
                return (
                    f"Error: Cluster key '{cluster_key}' not found in adata.obs.\n\n"
                    f"Available columns: {available_cols}\n\n"
                    f"Use check_data_status() to identify the correct cluster column."
                )

            # Call service
            try:
                result, stats, ir = clustering_service.compute_clustering_quality(
                    adata,
                    cluster_key=cluster_key,
                    use_rep=use_rep,
                    n_pcs=n_pcs,
                    metrics=metrics,
                )
            except ValueError as e:
                return (
                    f"Error: {str(e)}\n\n"
                    f"Please check:\n"
                    f"- Cluster key '{cluster_key}' exists in adata.obs\n"
                    f"- Representation '{use_rep}' exists in adata.obsm\n"
                    f"- At least 2 clusters are present\n"
                    f"- PCA has been computed (if use_rep='X_pca')"
                )
            except Exception as e:
                raise ClusteringError(f"Clustering quality evaluation failed: {str(e)}")

            # Store result with quality evaluation suffix
            new_name = f"{modality_name}_quality_evaluated"
            data_manager.store_modality(
                name=new_name,
                adata=result,
                parent_name=modality_name,
                step_summary=f"Evaluated clustering quality with {len(stats['metrics'])} metrics",
            )

            # Log with IR (mandatory for reproducibility)
            data_manager.log_tool_usage(
                "evaluate_clustering_quality",
                {
                    "cluster_key": cluster_key,
                    "use_rep": use_rep,
                    "n_pcs": n_pcs,
                    "metrics": (
                        metrics
                        if metrics
                        else ["silhouette", "davies_bouldin", "calinski_harabasz"]
                    ),
                },
                description=f"Evaluated clustering quality with {len(stats['metrics'])} metrics",
                ir=ir,
            )

            # Build response
            response_lines = []

            # Header
            response_lines.append("=" * 70)
            response_lines.append(f"CLUSTERING QUALITY EVALUATION: {cluster_key}")
            response_lines.append("=" * 70)
            response_lines.append("")

            # Basic info
            response_lines.append(f"**Modality**: {modality_name} -> {new_name}")
            response_lines.append(f"**Cells**: {stats['n_cells']:,}")
            response_lines.append(f"**Clusters**: {stats['n_clusters']}")
            response_lines.append(
                f"**Representation**: {stats['use_rep']} ({stats['n_pcs_used']} components)"
            )
            response_lines.append("")

            # Quality metrics section
            response_lines.append("**Quality Metrics:**")
            response_lines.append("-" * 70)

            # Silhouette score
            if "silhouette_score" in stats:
                score = stats["silhouette_score"]
                emoji = "[GOOD]" if score > 0.5 else "[OK]" if score > 0.25 else "[LOW]"
                response_lines.append(
                    f"{emoji} **Silhouette Score**: {score:.4f} "
                    f"(range: -1 to 1, higher = better separation)"
                )

            # Davies-Bouldin index
            if "davies_bouldin_index" in stats:
                score = stats["davies_bouldin_index"]
                emoji = "[GOOD]" if score < 1.0 else "[OK]" if score < 2.0 else "[HIGH]"
                response_lines.append(
                    f"{emoji} **Davies-Bouldin Index**: {score:.4f} "
                    f"(range: 0 to inf, lower = better compactness)"
                )

            # Calinski-Harabasz score
            if "calinski_harabasz_score" in stats:
                score = stats["calinski_harabasz_score"]
                emoji = "[GOOD]" if score > 1000 else "[OK]" if score > 100 else "[LOW]"
                response_lines.append(
                    f"{emoji} **Calinski-Harabasz Score**: {score:.1f} "
                    f"(range: 0 to inf, higher = better variance ratio)"
                )

            response_lines.append("")

            # Per-cluster silhouette scores (if available)
            if "per_cluster_silhouette" in stats:
                response_lines.append("**Per-Cluster Silhouette Scores:**")
                per_cluster = stats["per_cluster_silhouette"]

                # Sort by score (lowest first to highlight problems)
                sorted_clusters = sorted(per_cluster.items(), key=lambda x: x[1])

                for cluster_id, score in sorted_clusters[:10]:
                    size = stats["cluster_sizes"].get(cluster_id, 0)
                    emoji = (
                        "[GOOD]" if score > 0.5 else "[OK]" if score > 0.25 else "[LOW]"
                    )
                    response_lines.append(
                        f"  {emoji} Cluster {cluster_id}: {score:.4f} ({size} cells)"
                    )

                if len(sorted_clusters) > 10:
                    response_lines.append(
                        f"  ... and {len(sorted_clusters) - 10} more clusters"
                    )
                response_lines.append("")

            # Interpretation section
            response_lines.append("**Interpretation:**")
            response_lines.append("-" * 70)
            interpretation = stats.get("interpretation", "")
            for line in interpretation.split("\n"):
                if line.strip():
                    response_lines.append(f"* {line}")
            response_lines.append("")

            # Recommendations section
            recommendations = stats.get("recommendations", [])
            if recommendations:
                response_lines.append("**Recommendations:**")
                response_lines.append("-" * 70)
                for rec in recommendations:
                    response_lines.append(f"* {rec}")
                response_lines.append("")

            # Next steps
            response_lines.append("**Next Steps:**")
            response_lines.append("-" * 70)
            response_lines.append(
                "* If comparing resolutions: Run this on each resolution key (leiden_res0_25, leiden_res0_5, etc.)"
            )
            response_lines.append(
                "* If silhouette < 0.25: Try lower resolution or different preprocessing"
            )
            response_lines.append(
                "* If clusters look good: Proceed with find_marker_genes()"
            )
            response_lines.append(
                "* To visualize: Request visualization through supervisor"
            )
            response_lines.append("")

            # Footer
            response_lines.append("=" * 70)
            response_lines.append(
                f"Evaluation completed in {stats['execution_time_seconds']:.2f}s"
            )
            response_lines.append("=" * 70)

            return "\n".join(response_lines)

        except ModalityNotFoundError as e:
            logger.error(f"Error in clustering quality evaluation: {e}")
            return f"Error evaluating clustering quality: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in clustering quality evaluation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def find_marker_genes(
        modality_name: str,
        groupby: str,
        groups: List[str] = None,
        method: str = "wilcoxon",
        n_genes: int = 25,
        min_fold_change: float = 1.5,
        min_pct: float = 0.25,
        max_out_pct: float = 0.5,
        save_result: bool = True,
    ) -> str:
        """
        Find marker genes for single-cell clusters using differential expression analysis.

        IMPORTANT: Call check_data_status() first to identify the actual cluster column name.
        NEVER assume 'leiden' — data may use 'seurat_clusters', 'louvain', 'RNA_snn_res.1', etc.

        Args:
            modality_name: Name of the single-cell modality to analyze
            groupby: Column name to group by (REQUIRED).
                    Common values: 'leiden', 'louvain', 'seurat_clusters', 'cell_type'.
                    Use check_data_status() to find the correct column name.
            groups: Specific clusters to analyze (None for all)
            method: Statistical method ('wilcoxon', 't-test', 'logreg')
            n_genes: Number of top marker genes per cluster
            min_fold_change: Minimum fold-change threshold (default: 1.5).
                Filters genes with fold-change below this value.
            min_pct: Minimum in-group expression fraction (default: 0.25).
                Filters genes expressed in <25% of in-group cells.
            max_out_pct: Maximum out-group expression fraction (default: 0.5).
                Filters genes expressed in >50% of out-group cells (less specific).
            save_result: Whether to save the results
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
                f"Finding marker genes in single-cell modality '{modality_name}': {adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Validate groupby column exists
            if groupby not in adata.obs.columns:
                available_columns = list(adata.obs.columns)
                return (
                    f"Column '{groupby}' not found in adata.obs.\n\n"
                    f"Available columns: {available_columns}\n\n"
                    f"Use check_data_status() to identify the correct column name."
                )

            # Use singlecell service for marker gene detection
            adata_markers, marker_stats, ir = enhanced_service.find_marker_genes(
                adata=adata,
                groupby=groupby,
                groups=groups,
                method=method,
                n_genes=n_genes,
                min_fold_change=min_fold_change,
                min_pct=min_pct,
                max_out_pct=max_out_pct,
            )

            # Save as new modality
            marker_modality_name = f"{modality_name}_markers"
            data_manager.store_modality(
                name=marker_modality_name,
                adata=adata_markers,
                parent_name=modality_name,
                step_summary=f"Found marker genes for {marker_stats.get('n_groups', 'all')} groups",
            )

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_markers.h5ad"
                data_manager.save_modality(marker_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="find_marker_genes",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "method": method,
                    "n_genes": n_genes,
                    "min_fold_change": min_fold_change,
                    "min_pct": min_pct,
                    "max_out_pct": max_out_pct,
                },
                description=f"Found marker genes for {marker_stats['n_groups']} clusters (method: {marker_stats['method']}, pre-filter: {sum(marker_stats['pre_filter_counts'].values())}, post-filter: {sum(marker_stats['post_filter_counts'].values())}, filtered: {marker_stats['total_genes_filtered']})",
                ir=ir,
            )

            # Format professional response with filtering statistics
            response_parts = [
                f"Successfully found marker genes for single-cell clusters in '{modality_name}'!",
                "\n**Single-cell Marker Gene Analysis:**",
                f"- Grouping by: {marker_stats['groupby']}",
                f"- Number of clusters: {marker_stats['n_groups']}",
                f"- Method: {marker_stats['method']}",
                f"- Top genes per cluster: {marker_stats['n_genes']}",
                "\n**Filtering Parameters:**",
                f"  - Min fold-change: {marker_stats['filtering_params']['min_fold_change']}",
                f"  - Min in-group %: {marker_stats['filtering_params']['min_pct'] * 100:.1f}%",
                f"  - Max out-group %: {marker_stats['filtering_params']['max_out_pct'] * 100:.1f}%",
            ]

            # Add filtering summary
            if "filtered_counts" in marker_stats:
                response_parts.append(
                    f"\n**Filtering Summary:** {marker_stats['total_genes_filtered']} genes removed"
                )
                response_parts.append("\n**Genes per Cluster (after filtering):**")
                for group in marker_stats["groups_analyzed"][:10]:
                    if group in marker_stats["post_filter_counts"]:
                        post = marker_stats["post_filter_counts"][group]
                        filtered = marker_stats["filtered_counts"][group]
                        pre = marker_stats["pre_filter_counts"][group]
                        response_parts.append(
                            f"  - Cluster {group}: {post} genes (filtered {filtered}/{pre})"
                        )

                if len(marker_stats["groups_analyzed"]) > 10:
                    remaining = len(marker_stats["groups_analyzed"]) - 10
                    response_parts.append(f"  ... and {remaining} more clusters")

            response_parts.append("\n**Top Marker Genes by Cluster:**")

            # Show top marker genes for each cluster (first 5 clusters)
            if "top_markers_per_group" in marker_stats:
                for cluster_id in list(marker_stats["top_markers_per_group"].keys())[
                    :5
                ]:
                    top_genes = marker_stats["top_markers_per_group"][cluster_id][:5]
                    gene_names = [gene["gene"] for gene in top_genes]
                    response_parts.append(
                        f"  - **Cluster {cluster_id}**: {', '.join(gene_names)}"
                    )

                if len(marker_stats["top_markers_per_group"]) > 5:
                    remaining = len(marker_stats["top_markers_per_group"]) - 5
                    response_parts.append(f"  ... and {remaining} more clusters")

            response_parts.append(
                f"\n**New modality created**: '{marker_modality_name}'"
            )

            if save_result:
                response_parts.append(f"**Saved to**: {save_path}")

            response_parts.append(
                "**Access detailed results**: adata.uns['rank_genes_groups']"
            )
            response_parts.append(
                "\n**CRITICAL**: If annotation requested, INVOKE handoff_to_annotation_expert immediately (do NOT just suggest it)."
            )

            response = "\n".join(response_parts)

            analysis_results["details"]["marker_genes"] = response
            return response

        except (ServiceSingleCellError, ModalityNotFoundError) as e:
            logger.error(f"Error finding single-cell marker genes: {e}")
            return f"Error finding marker genes for single-cell clusters: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error finding single-cell marker genes: {e}")
            return f"Unexpected error: {str(e)}"

    # =========================================================================
    # SC ANALYSIS TOOLS
    # =========================================================================

    @tool
    def detect_doublets(
        modality_name: str,
        expected_doublet_rate: float = 0.025,
        threshold: Optional[float] = None,
    ) -> str:
        """
        Detect doublets using Scrublet on raw counts. Run BEFORE filtering/normalization.

        Doublets are artifactual cell barcodes from two cells captured together.
        Removing them before downstream analysis prevents spurious clusters and
        incorrect cell type assignments.

        Args:
            modality_name: Name of the modality to process (must contain raw counts)
            expected_doublet_rate: Expected fraction of doublets (default: 0.025 = 2.5%).
                                  Typical range: 0.01 (low loading) to 0.08 (high loading).
                                  Higher loading densities produce more doublets.
            threshold: Custom threshold for doublet calling. If None, Scrublet
                      auto-selects using bimodal distribution detection.

        Returns:
            Report with doublet count, rate, and recommendation to filter
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Detecting doublets in '{modality_name}': "
                f"{adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Call service
            result, stats, ir = enhanced_service.detect_doublets(
                adata,
                expected_doublet_rate=expected_doublet_rate,
                threshold=threshold,
            )

            # Store result
            new_name = f"{modality_name}_doublets_detected"
            data_manager.store_modality(
                name=new_name,
                adata=result,
                parent_name=modality_name,
                step_summary=f"Doublet detection: {stats.get('n_doublets', 0)} doublets found ({stats.get('doublet_rate', 0)*100:.1f}%)",
            )

            # Log with IR
            data_manager.log_tool_usage(
                tool_name="detect_doublets",
                parameters={
                    "modality_name": modality_name,
                    "expected_doublet_rate": expected_doublet_rate,
                    "threshold": threshold,
                },
                description=f"Detected {stats.get('n_doublets', 0)} doublets in {modality_name}",
                ir=ir,
            )

            # Format response
            n_doublets = stats.get("n_doublets", 0)
            doublet_rate = stats.get("doublet_rate", 0)
            method = stats.get("method", "scrublet")
            threshold_used = stats.get("threshold", "auto")

            response = f"""Doublet detection complete for '{modality_name}'!

**Method**: {method}
**Doublets detected**: {n_doublets} / {adata.shape[0]} cells ({doublet_rate*100:.1f}%)
**Threshold**: {threshold_used}
**Expected rate**: {expected_doublet_rate*100:.1f}%

**New modality created**: '{new_name}'"""

            if doublet_rate > 0.10:
                response += "\n\n**WARNING**: Doublet rate >10% is unusually high. Check loading density or consider adjusting threshold."
            elif doublet_rate > 0.05:
                response += "\n\n**Note**: Doublet rate is moderate. This is within normal range for high-density loading."

            response += "\n\n**Recommendation**: Filter doublets before downstream analysis using filter_and_normalize()."

            analysis_results["details"]["doublet_detection"] = response
            return response

        except (ServiceSingleCellError, ModalityNotFoundError) as e:
            logger.error(f"Error in doublet detection: {e}")
            return f"Error detecting doublets: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in doublet detection: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def integrate_batches(
        modality_name: str,
        batch_key: str,
        method: str = "harmony",
        n_pcs: int = 30,
    ) -> str:
        """
        Integrate multi-sample batches using Harmony or ComBat. Returns quality
        metrics (LISI, silhouette) for iterative refinement -- re-invoke with
        different parameters if metrics are poor.

        Args:
            modality_name: Name of the modality to integrate
            batch_key: Column in adata.obs containing batch labels (e.g., 'batch', 'sample')
            method: Integration method ('harmony' or 'combat', default: 'harmony').
                   Harmony: Fast, works in PCA space, good for most cases.
                   ComBat: Statistical, works on expression matrix, better for small batches.
            n_pcs: Number of principal components for PCA (default: 30)

        Returns:
            Report with method, batch count, silhouette score, LISI, and quality guidance
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Integrating batches in '{modality_name}' using {method}: "
                f"{adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Call service
            result, stats, ir = enhanced_service.integrate_batches(
                adata,
                batch_key=batch_key,
                method=method,
                n_pcs=n_pcs,
            )

            # Store result
            new_name = f"{modality_name}_integrated"
            data_manager.store_modality(
                name=new_name,
                adata=result,
                parent_name=modality_name,
                step_summary=f"Batch integration ({method}): {stats.get('n_batches', 0)} batches, silhouette={stats.get('batch_silhouette', 0):.3f}",
            )

            # Log with IR
            data_manager.log_tool_usage(
                tool_name="integrate_batches",
                parameters={
                    "modality_name": modality_name,
                    "batch_key": batch_key,
                    "method": method,
                    "n_pcs": n_pcs,
                },
                description=f"Integrated {stats.get('n_batches', 0)} batches in {modality_name} using {method}",
                ir=ir,
            )

            # Format response with quality metrics
            n_batches = stats.get("n_batches", 0)
            batch_silhouette = stats.get("batch_silhouette", 0)
            median_lisi = stats.get("median_lisi", 0)
            integrated_key = stats.get("integrated_key", "X_pca_harmony")

            response = f"""Batch integration complete for '{modality_name}'!

**Method**: {method}
**Batches integrated**: {n_batches}
**Integrated representation**: '{integrated_key}' (use for downstream clustering)

**Quality Metrics:**
- **Batch silhouette score**: {batch_silhouette:.3f} (closer to 0 = better batch mixing; >0.3 suggests residual batch effects)
- **Median LISI**: {median_lisi:.2f} (closer to {n_batches} = better mixing; <1.5 suggests poor integration)

**New modality created**: '{new_name}'"""

            # Add quality guidance for iterative refinement
            if batch_silhouette > 0.3 or median_lisi < 1.5:
                response += f"""

**Quality concerns detected:**
- {"Batch silhouette > 0.3: significant batch effects remain." if batch_silhouette > 0.3 else ""}
- {"Median LISI < 1.5: batches are not well mixed." if median_lisi < 1.5 else ""}

**Suggestions**: Re-run with different parameters:
- Try method='{"combat" if method == "harmony" else "harmony"}'
- Increase n_pcs (e.g., 50) for more information
- Check if batch_key is the correct batch variable"""
            else:
                response += "\n\n**Integration quality**: Good. Proceed with clustering using the integrated representation."

            analysis_results["details"]["batch_integration"] = response
            return response

        except (ServiceSingleCellError, ModalityNotFoundError) as e:
            logger.error(f"Error in batch integration: {e}")
            return f"Error integrating batches: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in batch integration: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def compute_trajectory(
        modality_name: str,
        root_cell: Optional[int] = None,
        root_group: Optional[str] = None,
        cluster_key: str = "leiden",
        n_dcs: int = 15,
    ) -> str:
        """
        Compute DPT pseudotime and PAGA trajectory. Requires clustered data
        with neighbors computed.

        Trajectory inference reveals temporal ordering of cells along
        differentiation or activation paths. DPT (Diffusion Pseudotime) computes
        a pseudotime value for each cell, while PAGA (Partition-based Graph
        Abstraction) reveals connectivity between clusters.

        Args:
            modality_name: Name of the modality to process (must have neighbors computed)
            root_cell: Explicit root cell index (highest priority). If None,
                      auto-selects using diffusion component minimum.
            root_group: Cluster name to use as root (e.g., 'stem_cells', '0').
                       Used only when root_cell is None.
            cluster_key: Key in adata.obs for cluster assignments (default: 'leiden')
            n_dcs: Number of diffusion components (default: 15)

        Returns:
            Report with root cell, pseudotime range, PAGA status, and interpretation guidance
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Computing trajectory for '{modality_name}': "
                f"{adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Call service
            result, stats, ir = enhanced_service.compute_trajectory(
                adata,
                root_cell=root_cell,
                root_group=root_group,
                cluster_key=cluster_key,
                n_dcs=n_dcs,
            )

            # Store result
            new_name = f"{modality_name}_trajectory"
            data_manager.store_modality(
                name=new_name,
                adata=result,
                parent_name=modality_name,
                step_summary=f"Trajectory: pseudotime [{stats.get('pseudotime_min', 0):.2f}, {stats.get('pseudotime_max', 0):.2f}]",
            )

            # Log with IR
            data_manager.log_tool_usage(
                tool_name="compute_trajectory",
                parameters={
                    "modality_name": modality_name,
                    "root_cell": root_cell,
                    "root_group": root_group,
                    "cluster_key": cluster_key,
                    "n_dcs": n_dcs,
                },
                description=f"Computed DPT trajectory for {modality_name}",
                ir=ir,
            )

            # Format response
            root_idx = stats.get("root_cell", "auto")
            pt_min = stats.get("pseudotime_min", 0)
            pt_max = stats.get("pseudotime_max", 0)
            has_paga = stats.get("has_paga", False)
            root_method = stats.get("root_selection_method", "auto")

            response = f"""Trajectory inference complete for '{modality_name}'!

**Root cell**: index {root_idx} (selected via: {root_method})
**Pseudotime range**: [{pt_min:.3f}, {pt_max:.3f}]
**PAGA computed**: {'Yes' if has_paga else 'No'}
**Diffusion components**: {n_dcs}

**New modality created**: '{new_name}'

**Interpreting pseudotime:**
- Values closer to 0 = earlier in trajectory (near root)
- Values closer to {pt_max:.3f} = later in trajectory (terminal states)
- Pseudotime is stored in adata.obs['dpt_pseudotime']
- PAGA connectivity is stored in adata.uns['paga']

**Next steps**: Visualize trajectory on UMAP colored by dpt_pseudotime, or identify genes that change along the trajectory."""

            analysis_results["details"]["trajectory"] = response
            return response

        except (ServiceSingleCellError, ModalityNotFoundError) as e:
            logger.error(f"Error in trajectory computation: {e}")
            return f"Error computing trajectory: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in trajectory computation: {e}")
            return f"Unexpected error: {str(e)}"

    # =========================================================================
    # BULK RNA-SEQ TOOLS
    # =========================================================================

    @tool
    def import_bulk_counts(
        modality_name: str,
        file_path: str,
        source: str = "auto",
        gene_id_column: str = "auto",
    ) -> str:
        """
        Import bulk RNA-seq count data from Salmon, kallisto, featureCounts, or CSV/TSV files.

        Supports multiple input formats:
        - Salmon/kallisto: Directory of per-sample quantification files
        - featureCounts: Tab-delimited output (auto-detects # comment header)
        - CSV/TSV: Generic count matrix (auto-transposes if genes > samples)

        Args:
            modality_name: Name for the imported modality
            file_path: Path to file or directory containing count data
            source: Data source format ('salmon', 'kallisto', 'featurecounts', 'csv', 'auto')
            gene_id_column: Column containing gene IDs ('auto' for auto-detection)

        Returns:
            Summary of imported data (samples, genes, source detected)
        """
        try:
            import anndata

            path = Path(file_path)
            if not path.exists():
                return f"Error: Path '{file_path}' does not exist."

            adata = None
            detected_source = source

            # Salmon/kallisto: directory-based import
            if path.is_dir() and source in ("salmon", "kallisto", "auto"):
                try:
                    df, metadata = bulk_service.load_from_quantification_files(
                        path, tool=source
                    )
                    detected_source = metadata.get("tool", source)
                    # df has genes as rows, samples as columns -> transpose for AnnData
                    adata = anndata.AnnData(
                        X=df.T.values.astype(np.float32),
                        obs=pd.DataFrame(index=df.columns),
                        var=pd.DataFrame(index=df.index),
                    )
                    # Store raw counts in layer
                    adata.layers["counts"] = adata.X.copy()
                except Exception as e:
                    if source != "auto":
                        return f"Error loading {source} data: {str(e)}"
                    # Fall through to CSV/TSV handling for auto

            # featureCounts format
            if adata is None and (
                source == "featurecounts"
                or (
                    source == "auto"
                    and path.is_file()
                    and path.suffix in (".txt", ".tsv")
                )
            ):
                try:
                    # Read and check for featureCounts header
                    with open(path) as f:
                        first_lines = [f.readline() for _ in range(3)]

                    is_fc = any(
                        line.startswith("#") or "Geneid" in line for line in first_lines
                    )

                    if is_fc or source == "featurecounts":
                        df = pd.read_csv(path, sep="\t", comment="#")
                        if "Geneid" in df.columns:
                            df = df.set_index("Geneid")
                        # Drop featureCounts annotation columns
                        fc_cols = {"Chr", "Start", "End", "Strand", "Length"}
                        df = df.drop(
                            columns=[c for c in fc_cols if c in df.columns]
                        )
                        # Genes as rows, samples as columns -> transpose
                        adata = anndata.AnnData(
                            X=df.T.values.astype(np.float32),
                            obs=pd.DataFrame(index=df.columns),
                            var=pd.DataFrame(index=df.index),
                        )
                        adata.layers["counts"] = adata.X.copy()
                        detected_source = "featurecounts"
                except Exception:
                    pass  # Fall through to CSV/TSV

            # Generic CSV/TSV
            if adata is None and path.is_file():
                sep = "\t" if path.suffix in (".tsv", ".txt") else ","
                df = pd.read_csv(path, sep=sep, index_col=0)

                # Auto-detect gene ID column if needed
                if gene_id_column != "auto" and gene_id_column in df.columns:
                    df = df.set_index(gene_id_column)

                # Auto-transpose: if columns > rows, genes are likely rows
                if df.shape[1] > df.shape[0]:
                    df = df.T

                adata = anndata.AnnData(
                    X=df.values.astype(np.float32),
                    obs=pd.DataFrame(index=df.index),
                    var=pd.DataFrame(index=df.columns),
                )
                adata.layers["counts"] = adata.X.copy()
                detected_source = "csv" if sep == "," else "tsv"

            if adata is None:
                return (
                    f"Error: Could not import data from '{file_path}'. "
                    "Check the file format and try specifying the 'source' parameter."
                )

            # Store modality
            data_manager.store_modality(
                name=modality_name,
                adata=adata,
                parent_name=None,
                step_summary=f"Imported bulk counts from {detected_source}: {adata.shape[0]} samples x {adata.shape[1]} genes",
            )

            # Build IR
            ir = AnalysisStep(
                operation="import_bulk_counts",
                tool_name="import_bulk_counts",
                description=f"Imported bulk RNA-seq counts from {detected_source}",
                library="pandas",
                code_template=f'adata = sc.read_csv("{file_path}")  # source={detected_source}',
                imports=["import scanpy as sc"],
                parameters={
                    "file_path": file_path,
                    "source": detected_source,
                    "gene_id_column": gene_id_column,
                },
            )

            data_manager.log_tool_usage(
                tool_name="import_bulk_counts",
                parameters={
                    "modality_name": modality_name,
                    "file_path": file_path,
                    "source": source,
                    "detected_source": detected_source,
                },
                description=f"Imported bulk counts ({detected_source}): {adata.shape[0]} samples x {adata.shape[1]} genes",
                ir=ir,
            )

            return (
                f"Successfully imported bulk RNA-seq counts as '{modality_name}'!\n\n"
                f"**Import Details:**\n"
                f"- Source format: {detected_source}\n"
                f"- Samples (obs): {adata.shape[0]}\n"
                f"- Genes (var): {adata.shape[1]}\n"
                f"- Raw counts stored in: adata.layers['counts']\n\n"
                f"**Next steps**: merge_sample_metadata() if metadata available, "
                f"then assess_bulk_sample_quality()."
            )

        except Exception as e:
            logger.error(f"Error importing bulk counts: {e}")
            return f"Error importing bulk counts: {str(e)}"

    @tool
    def merge_sample_metadata(
        modality_name: str,
        metadata_file: str,
        sample_id_column: str = "auto",
    ) -> str:
        """
        Join external sample metadata (CSV/TSV/Excel) with count matrix obs.

        Auto-detects the sample ID column by finding overlap between metadata
        column values and adata.obs_names.

        Args:
            modality_name: Name of the bulk modality to update
            metadata_file: Path to metadata file (CSV, TSV, or Excel)
            sample_id_column: Column containing sample IDs ('auto' for auto-detection)

        Returns:
            Summary of merged metadata (columns added, samples matched/unmatched)
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            meta_path = Path(metadata_file)

            if not meta_path.exists():
                return f"Error: Metadata file '{metadata_file}' does not exist."

            # Read metadata (detect format from extension)
            if meta_path.suffix in (".xlsx", ".xls"):
                meta_df = pd.read_excel(meta_path)
            elif meta_path.suffix == ".tsv":
                meta_df = pd.read_csv(meta_path, sep="\t")
            else:
                meta_df = pd.read_csv(meta_path)

            # Auto-detect sample ID column
            if sample_id_column == "auto":
                obs_names = set(adata.obs_names)
                best_col = None
                best_overlap = 0
                for col in meta_df.columns:
                    col_values = set(meta_df[col].astype(str))
                    overlap = len(obs_names & col_values)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_col = col
                if best_col is None or best_overlap == 0:
                    return (
                        "Error: Could not auto-detect sample ID column. "
                        "None of the metadata columns match adata.obs_names. "
                        "Please specify sample_id_column explicitly."
                    )
                sample_id_column = best_col
                logger.info(
                    f"Auto-detected sample ID column: '{sample_id_column}' "
                    f"({best_overlap} matches)"
                )

            # Set index for merge
            meta_df = meta_df.set_index(sample_id_column)

            # Track columns before merge
            cols_before = set(adata.obs.columns)

            # Left-join on sample IDs
            matched = set(adata.obs_names) & set(meta_df.index)
            unmatched = set(adata.obs_names) - set(meta_df.index)

            # Merge metadata into obs
            for col in meta_df.columns:
                if col not in adata.obs.columns:
                    adata.obs[col] = meta_df[col].reindex(adata.obs_names)

            cols_added = set(adata.obs.columns) - cols_before

            # Store updated modality
            data_manager.store_modality(
                name=modality_name,
                adata=adata,
                parent_name=None,
                step_summary=f"Merged {len(cols_added)} metadata columns, {len(matched)}/{len(matched)+len(unmatched)} samples matched",
            )

            # Build IR
            ir = AnalysisStep(
                operation="merge_sample_metadata",
                tool_name="merge_sample_metadata",
                description=f"Merged external metadata from {metadata_file}",
                library="pandas",
                code_template=(
                    f'meta = pd.read_csv("{metadata_file}")\n'
                    f'meta = meta.set_index("{sample_id_column}")\n'
                    "adata.obs = adata.obs.join(meta)"
                ),
                imports=["import pandas as pd"],
                parameters={
                    "metadata_file": metadata_file,
                    "sample_id_column": sample_id_column,
                },
            )

            data_manager.log_tool_usage(
                tool_name="merge_sample_metadata",
                parameters={
                    "modality_name": modality_name,
                    "metadata_file": metadata_file,
                    "sample_id_column": sample_id_column,
                },
                description=f"Merged {len(cols_added)} metadata columns into {modality_name}",
                ir=ir,
            )

            response = (
                f"Successfully merged sample metadata into '{modality_name}'!\n\n"
                f"**Merge Details:**\n"
                f"- Sample ID column: {sample_id_column}\n"
                f"- Metadata columns added: {len(cols_added)}\n"
                f"- Samples matched: {len(matched)}\n"
                f"- Samples unmatched: {len(unmatched)}\n"
            )

            if cols_added:
                response += f"- New columns: {', '.join(sorted(cols_added))}\n"

            if unmatched:
                unmatched_list = sorted(unmatched)[:5]
                response += (
                    f"\n**Unmatched samples** (first 5): {', '.join(unmatched_list)}\n"
                )

            response += (
                "\n**Next steps**: assess_bulk_sample_quality() to check for outliers."
            )
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error merging metadata: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error merging sample metadata: {e}")
            return f"Error merging sample metadata: {str(e)}"

    @tool
    def assess_bulk_sample_quality(
        modality_name: str,
        batch_key: Optional[str] = None,
    ) -> str:
        """
        Assess bulk sample quality via PCA outlier detection, sample correlation,
        and optional batch effect estimation.

        Args:
            modality_name: Name of the bulk modality to assess
            batch_key: Optional column in adata.obs for batch grouping (e.g., 'batch', 'plate')

        Returns:
            Quality report with outliers, correlation, batch R-squared, and recommendations
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            # Call service
            result, stats, ir = bulk_preprocessing_service.assess_sample_quality(
                adata, batch_key=batch_key
            )

            # Store as quality-assessed modality
            new_name = f"{modality_name}_quality_assessed"
            data_manager.store_modality(
                name=new_name,
                adata=result,
                parent_name=modality_name,
                step_summary=f"Quality assessed: {stats.get('n_outliers', 0)} outliers found",
            )

            # Log with IR
            data_manager.log_tool_usage(
                tool_name="assess_bulk_sample_quality",
                parameters={
                    "modality_name": modality_name,
                    "batch_key": batch_key,
                },
                description=f"Assessed bulk sample quality for {modality_name}",
                ir=ir,
            )

            # Format response
            n_outliers = stats.get("n_outliers", 0)
            outlier_names = stats.get("outlier_samples", [])
            median_corr = stats.get("median_correlation", 0)

            response = (
                f"Bulk sample quality assessment complete for '{modality_name}'!\n\n"
                f"**Quality Metrics:**\n"
                f"- Samples assessed: {stats.get('n_samples', adata.shape[0])}\n"
                f"- Outlier samples: {n_outliers}\n"
                f"- Median pairwise correlation: {median_corr:.3f}\n"
            )

            if outlier_names:
                response += (
                    f"- Outlier names: {', '.join(str(o) for o in outlier_names)}\n"
                )

            if batch_key and "batch_r_squared" in stats:
                response += (
                    f"- Batch R-squared (PC1-3): {stats['batch_r_squared']:.3f}\n"
                )

            response += f"\n**New modality created**: '{new_name}'\n"

            # Recommendations
            if n_outliers > 0:
                response += (
                    f"\n**Recommendation**: Consider removing {n_outliers} outlier sample(s) "
                    "before downstream analysis."
                )
            if median_corr < 0.8:
                response += (
                    "\n**Warning**: Low median correlation (<0.8) may indicate "
                    "sample quality issues or strong biological variation."
                )

            response += "\n\n**Next steps**: filter_bulk_genes() to remove lowly-expressed genes."
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error assessing bulk quality: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error assessing bulk sample quality: {e}")
            return f"Error assessing bulk sample quality: {str(e)}"

    @tool
    def filter_bulk_genes(
        modality_name: str,
        min_counts: int = 10,
        min_samples: int = 3,
    ) -> str:
        """
        Filter lowly-expressed genes from bulk RNA-seq data.

        Removes genes that do not meet minimum expression thresholds across samples.
        Standard practice before normalization and differential expression.

        Args:
            modality_name: Name of the bulk modality to filter
            min_counts: Minimum total counts across all samples (default: 10)
            min_samples: Minimum number of samples with non-zero expression (default: 3)

        Returns:
            Summary of genes before/after filtering
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            # Call service
            result, stats, ir = bulk_preprocessing_service.filter_genes(
                adata, min_counts=min_counts, min_samples=min_samples
            )

            # Store as filtered modality
            new_name = f"{modality_name}_filtered"
            data_manager.store_modality(
                name=new_name,
                adata=result,
                parent_name=modality_name,
                step_summary=f"Filtered genes: {stats.get('n_genes_before', 0)} -> {stats.get('n_genes_after', 0)}",
            )

            # Log with IR
            data_manager.log_tool_usage(
                tool_name="filter_bulk_genes",
                parameters={
                    "modality_name": modality_name,
                    "min_counts": min_counts,
                    "min_samples": min_samples,
                },
                description=f"Filtered bulk genes: {stats.get('n_genes_removed', 0)} removed",
                ir=ir,
            )

            genes_before = stats.get("n_genes_before", 0)
            genes_after = stats.get("n_genes_after", 0)
            genes_removed = stats.get("n_genes_removed", 0)

            return (
                f"Gene filtering complete for '{modality_name}'!\n\n"
                f"**Filtering Results:**\n"
                f"- Genes before: {genes_before:,}\n"
                f"- Genes after: {genes_after:,}\n"
                f"- Genes removed: {genes_removed:,} ({genes_removed/max(genes_before,1)*100:.1f}%)\n"
                f"- Filter criteria: min_counts={min_counts}, min_samples={min_samples}\n\n"
                f"**New modality created**: '{new_name}'\n\n"
                f"**Next steps**: normalize_bulk_counts() to normalize the filtered data."
            )

        except ModalityNotFoundError as e:
            logger.error(f"Error filtering bulk genes: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error filtering bulk genes: {e}")
            return f"Error filtering bulk genes: {str(e)}"

    @tool
    def normalize_bulk_counts(
        modality_name: str,
        method: str = "deseq2",
    ) -> str:
        """
        Normalize bulk RNA-seq counts using DESeq2 size factors, VST, or CPM.

        Args:
            modality_name: Name of the bulk modality to normalize
            method: Normalization method ('deseq2', 'vst', or 'cpm').
                   - 'deseq2' (default): DESeq2 median-of-ratios size factors. Best for DE analysis.
                   - 'vst': Variance-stabilizing transformation. Best for visualization/clustering.
                   - 'cpm': Counts per million. Simple, good for comparisons.

        Returns:
            Summary of normalization (method used, key statistics)
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            # Call service
            result, stats, ir = bulk_preprocessing_service.normalize_counts(
                adata, method=method
            )

            # Store as normalized modality
            new_name = f"{modality_name}_normalized"
            data_manager.store_modality(
                name=new_name,
                adata=result,
                parent_name=modality_name,
                step_summary=f"Normalized with {method}",
            )

            # Log with IR
            data_manager.log_tool_usage(
                tool_name="normalize_bulk_counts",
                parameters={
                    "modality_name": modality_name,
                    "method": method,
                },
                description=f"Normalized {modality_name} using {method}",
                ir=ir,
            )

            response = (
                f"Normalization complete for '{modality_name}'!\n\n"
                f"**Normalization Details:**\n"
                f"- Method: {method}\n"
            )

            if method == "deseq2" and "mean_size_factor" in stats:
                response += f"- Mean size factor: {stats['mean_size_factor']:.3f}\n"
                response += f"- Size factor range: [{stats.get('min_size_factor', 0):.3f}, {stats.get('max_size_factor', 0):.3f}]\n"
            elif method == "vst":
                response += "- Variance-stabilized values stored in adata.X\n"
            elif method == "cpm":
                response += "- CPM values stored in adata.X\n"

            response += (
                f"\n**New modality created**: '{new_name}'\n\n"
                f"**Next steps**: detect_batch_effects() if multiple batches, "
                f"or prepare_bulk_for_de() to validate DE readiness."
            )
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error normalizing bulk counts: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error normalizing bulk counts: {e}")
            return f"Error normalizing bulk counts: {str(e)}"

    @tool
    def detect_batch_effects(
        modality_name: str,
        batch_key: str,
        condition_key: Optional[str] = None,
    ) -> str:
        """
        Detect batch effects by decomposing variance between batch and condition.

        Uses PCA + R-squared to quantify how much variance is explained by batch
        vs biological condition. Helps decide whether batch correction is needed.

        Args:
            modality_name: Name of the bulk modality to assess
            batch_key: Column in adata.obs containing batch labels (e.g., 'batch', 'plate')
            condition_key: Optional column for biological condition (e.g., 'treatment', 'group')

        Returns:
            Variance decomposition report and batch correction recommendation
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            # Call service
            result, stats, ir = bulk_preprocessing_service.detect_batch_effects(
                adata, batch_key=batch_key, condition_key=condition_key
            )

            # Store as batch-assessed modality
            new_name = f"{modality_name}_batch_assessed"
            data_manager.store_modality(
                name=new_name,
                adata=result,
                parent_name=modality_name,
                step_summary=f"Batch effects assessed: R2_batch={stats.get('batch_r_squared', 0):.3f}",
            )

            # Log with IR
            data_manager.log_tool_usage(
                tool_name="detect_batch_effects",
                parameters={
                    "modality_name": modality_name,
                    "batch_key": batch_key,
                    "condition_key": condition_key,
                },
                description=f"Detected batch effects in {modality_name}",
                ir=ir,
            )

            batch_r2 = stats.get("batch_r_squared", 0)
            response = (
                f"Batch effect assessment complete for '{modality_name}'!\n\n"
                f"**Variance Decomposition:**\n"
                f"- Batch variable: {batch_key}\n"
                f"- Batch R-squared (PC1-3): {batch_r2:.3f}\n"
            )

            if condition_key and "condition_r_squared" in stats:
                cond_r2 = stats["condition_r_squared"]
                response += f"- Condition variable: {condition_key}\n"
                response += f"- Condition R-squared (PC1-3): {cond_r2:.3f}\n"
                response += f"- Batch/Condition ratio: {batch_r2/max(cond_r2, 0.001):.2f}\n"

            response += f"\n**New modality created**: '{new_name}'\n"

            # Recommendation
            if batch_r2 > 0.3:
                response += (
                    "\n**Recommendation**: Strong batch effects detected (R2 > 0.3). "
                    "Include batch as covariate in DE model or apply batch correction."
                )
            elif batch_r2 > 0.1:
                response += (
                    "\n**Recommendation**: Moderate batch effects (0.1 < R2 < 0.3). "
                    "Consider including batch as covariate in DE model."
                )
            else:
                response += (
                    "\n**Recommendation**: Minimal batch effects (R2 < 0.1). "
                    "Batch correction likely not needed."
                )

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error detecting batch effects: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error detecting batch effects: {e}")
            return f"Error detecting batch effects: {str(e)}"

    @tool
    def convert_gene_identifiers(
        modality_name: str,
        source_type: str = "auto",
        target_type: str = "symbol",
    ) -> str:
        """
        Convert gene identifiers between Ensembl, Symbol, and Entrez formats using mygene.

        Auto-detects the source ID type from patterns:
        - ENSG*: Ensembl gene IDs
        - Numeric: Entrez gene IDs
        - Other: Gene symbols

        Args:
            modality_name: Name of the modality to convert
            source_type: Source ID type ('ensembl', 'entrez', 'symbol', 'auto')
            target_type: Target ID type ('symbol', 'ensembl', 'entrez')

        Returns:
            Conversion summary (converted count, unmapped genes)
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            gene_ids = adata.var_names.tolist()

            # Auto-detect source type
            if source_type == "auto":
                sample = gene_ids[:20]
                ensembl_count = sum(1 for g in sample if str(g).startswith("ENSG"))
                numeric_count = sum(1 for g in sample if str(g).isdigit())
                if ensembl_count > len(sample) * 0.5:
                    source_type = "ensembl"
                elif numeric_count > len(sample) * 0.5:
                    source_type = "entrez"
                else:
                    source_type = "symbol"
                logger.info(f"Auto-detected gene ID type: {source_type}")

            # Map types to mygene fields
            type_to_scope = {
                "ensembl": "ensembl.gene",
                "symbol": "symbol",
                "entrez": "entrezgene",
            }
            type_to_field = {
                "ensembl": "ensembl.gene",
                "symbol": "symbol",
                "entrez": "entrezgene",
            }

            scope = type_to_scope.get(source_type, source_type)
            field = type_to_field.get(target_type, target_type)

            # Lazy import mygene
            try:
                import mygene
            except ImportError:
                return (
                    "Error: mygene package not installed. "
                    "Install with: pip install mygene\n\n"
                    "mygene is required for gene ID conversion between "
                    "Ensembl, Symbol, and Entrez formats."
                )

            # Strip Ensembl version suffixes
            query_ids = gene_ids
            if source_type == "ensembl":
                query_ids = [g.split(".")[0] for g in gene_ids]

            # Query mygene
            mg = mygene.MyGeneInfo()
            results = mg.querymany(
                query_ids,
                scopes=scope,
                fields=field,
                species="human",
                returnall=True,
            )

            # Build mapping
            mapping = {}
            for hit in results.get("out", []):
                query = hit.get("query", "")
                if "notfound" in hit and hit["notfound"]:
                    continue
                # Extract target field (handle nested fields like ensembl.gene)
                value = hit
                for part in field.split("."):
                    if isinstance(value, dict):
                        value = value.get(part, None)
                    elif isinstance(value, list) and value:
                        value = value[0]
                        if isinstance(value, dict):
                            value = value.get(part, None)
                    else:
                        value = None
                        break
                if value:
                    mapping[query] = str(value)

            # Store original IDs and apply new ones
            adata.var["original_id"] = gene_ids
            new_names = []
            for i, gene in enumerate(gene_ids):
                query = query_ids[i] if source_type == "ensembl" else gene
                new_names.append(mapping.get(query, gene))

            adata.var_names = pd.Index(new_names)
            adata.var_names_make_unique()

            n_converted = sum(1 for q, g in zip(query_ids, gene_ids) if mapping.get(q if source_type == "ensembl" else g))
            n_unmapped = len(gene_ids) - n_converted
            unmapped_genes = [g for i, g in enumerate(gene_ids) if (query_ids[i] if source_type == "ensembl" else g) not in mapping][:10]

            # Store updated modality
            data_manager.store_modality(
                name=modality_name,
                adata=adata,
                parent_name=None,
                step_summary=f"Converted gene IDs: {source_type} -> {target_type}, {n_converted}/{len(gene_ids)} mapped",
            )

            # Build IR
            ir = AnalysisStep(
                operation="convert_gene_identifiers",
                tool_name="convert_gene_identifiers",
                description=f"Converted gene IDs from {source_type} to {target_type}",
                library="mygene",
                code_template=(
                    "import mygene\n"
                    "mg = mygene.MyGeneInfo()\n"
                    f'results = mg.querymany(adata.var_names, scopes="{scope}", '
                    f'fields="{field}", species="human")'
                ),
                imports=["import mygene"],
                parameters={
                    "source_type": source_type,
                    "target_type": target_type,
                },
            )

            data_manager.log_tool_usage(
                tool_name="convert_gene_identifiers",
                parameters={
                    "modality_name": modality_name,
                    "source_type": source_type,
                    "target_type": target_type,
                },
                description=f"Converted gene IDs: {source_type} -> {target_type}",
                ir=ir,
            )

            response = (
                f"Gene ID conversion complete for '{modality_name}'!\n\n"
                f"**Conversion Details:**\n"
                f"- Source type: {source_type}\n"
                f"- Target type: {target_type}\n"
                f"- Genes converted: {n_converted:,}\n"
                f"- Genes unmapped: {n_unmapped:,}\n"
                f"- Original IDs saved in: adata.var['original_id']\n"
            )

            if unmapped_genes:
                response += f"\n**Unmapped genes** (first 10): {', '.join(unmapped_genes)}\n"

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error converting gene IDs: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error converting gene identifiers: {e}")
            return f"Error converting gene identifiers: {str(e)}"

    @tool
    def prepare_bulk_for_de(
        modality_name: str,
        group_key: str,
        design_factors: Optional[List[str]] = None,
    ) -> str:
        """
        Validate that bulk data is ready for differential expression analysis before handoff.

        Performs readiness checks without modifying data:
        1. Raw counts exist (adata.layers['counts'] or integer-like X)
        2. Group key exists with at least 2 groups
        3. Minimum 2 samples per group (warns if <3)
        4. Design factors exist in obs (if specified)

        Args:
            modality_name: Name of the bulk modality to validate
            group_key: Column in adata.obs for group comparison (e.g., 'condition', 'treatment')
            design_factors: Optional list of additional design factor columns (e.g., ['batch', 'sex'])

        Returns:
            Validation report with pass/fail for each check and overall readiness
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            checks = []
            all_pass = True
            warnings = []

            # Check 1: Raw counts exist
            has_counts_layer = "counts" in adata.layers
            if has_counts_layer:
                checks.append(("Raw counts (layers['counts'])", "PASS"))
            else:
                # Check if X has integer-like values
                X = adata.X
                if hasattr(X, "toarray"):
                    X = X.toarray()
                sample = X.flatten()[:1000]
                is_integer_like = np.allclose(sample, np.round(sample), atol=0.01)
                if is_integer_like:
                    checks.append(("Raw counts (integer-like X)", "PASS"))
                else:
                    checks.append(("Raw counts", "FAIL - no raw counts found"))
                    all_pass = False

            # Check 2: Group key exists with >= 2 groups
            if group_key in adata.obs.columns:
                groups = adata.obs[group_key].dropna().unique()
                n_groups = len(groups)
                if n_groups >= 2:
                    checks.append((f"Group key '{group_key}' ({n_groups} groups)", "PASS"))
                else:
                    checks.append((f"Group key '{group_key}'", f"FAIL - only {n_groups} group(s)"))
                    all_pass = False
            else:
                checks.append((f"Group key '{group_key}'", "FAIL - column not found in obs"))
                all_pass = False
                groups = []

            # Check 3: Minimum samples per group
            if group_key in adata.obs.columns and len(groups) >= 2:
                group_sizes = adata.obs[group_key].value_counts()
                min_size = group_sizes.min()
                if min_size >= 3:
                    checks.append((f"Sample count per group (min={min_size})", "PASS"))
                elif min_size >= 2:
                    checks.append((f"Sample count per group (min={min_size})", "WARN"))
                    warnings.append(
                        f"Group '{group_sizes.idxmin()}' has only {min_size} samples. "
                        "DESeq2 recommends >= 3 for stable variance estimation."
                    )
                else:
                    checks.append((f"Sample count per group (min={min_size})", "FAIL - need >= 2"))
                    all_pass = False

            # Check 4: Design factors exist
            if design_factors:
                missing_factors = [f for f in design_factors if f not in adata.obs.columns]
                if not missing_factors:
                    checks.append((f"Design factors ({', '.join(design_factors)})", "PASS"))
                else:
                    checks.append((f"Design factors", f"FAIL - missing: {', '.join(missing_factors)}"))
                    all_pass = False

            # Build IR (validation only, no data modification)
            ir = AnalysisStep(
                operation="prepare_bulk_for_de",
                tool_name="prepare_bulk_for_de",
                description=f"Validated DE readiness: {'READY' if all_pass else 'NOT READY'}",
                library="lobster",
                code_template="# Validation step - no code generated",
                imports=[],
                parameters={
                    "group_key": group_key,
                    "design_factors": design_factors,
                },
            )

            data_manager.log_tool_usage(
                tool_name="prepare_bulk_for_de",
                parameters={
                    "modality_name": modality_name,
                    "group_key": group_key,
                    "design_factors": design_factors,
                },
                description=f"DE readiness validation: {'READY' if all_pass else 'NOT READY'}",
                ir=ir,
            )

            # Format response
            status = "READY" if all_pass else "NOT READY"
            response = (
                f"DE Readiness Validation for '{modality_name}': **{status}**\n\n"
                f"**Validation Checks:**\n"
            )
            for check_name, result in checks:
                marker = "[PASS]" if result == "PASS" else "[WARN]" if result == "WARN" else "[FAIL]"
                response += f"  {marker} {check_name}: {result}\n"

            if warnings:
                response += "\n**Warnings:**\n"
                for w in warnings:
                    response += f"  - {w}\n"

            if all_pass:
                response += (
                    "\n**Data is ready for DE analysis.** "
                    "INVOKE handoff_to_de_analysis_expert to proceed."
                )
            else:
                response += (
                    "\n**Data is NOT ready for DE analysis.** "
                    "Please fix the failing checks above before proceeding."
                )

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error validating DE readiness: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error preparing bulk for DE: {e}")
            return f"Error preparing bulk for DE: {str(e)}"

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    # SC analysis tools
    sc_analysis_tools = [
        detect_doublets,
        integrate_batches,
        compute_trajectory,
    ]

    # Clustering tools (single-cell specific)
    clustering_tools = [
        cluster_cells,
        subcluster_cells,
        evaluate_clustering_quality,
        find_marker_genes,
    ]

    # Bulk RNA-seq tools
    bulk_tools = [
        import_bulk_counts,
        merge_sample_metadata,
        assess_bulk_sample_quality,
        filter_bulk_genes,
        normalize_bulk_counts,
        detect_batch_effects,
        convert_gene_identifiers,
        prepare_bulk_for_de,
    ]

    # Combine all direct tools
    direct_tools = shared_tools + clustering_tools + sc_analysis_tools + bulk_tools

    # Add delegation tools if provided (annotation_expert, de_analysis_expert)
    tools = direct_tools
    if delegation_tools:
        tools = tools + delegation_tools

    # Create system prompt
    system_prompt = create_transcriptomics_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=TranscriptomicsExpertState,
    )
