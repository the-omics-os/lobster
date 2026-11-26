"""
Visualization Expert Agent for creating publication-quality plots.

This agent specializes in interactive visualizations through supervisor-mediated
workflows. The retained figure producers publish only after their success fence.
"""

# Agent configuration for entry point discovery (before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="visualization_expert_agent",
    display_name="Visualization Expert",
    description=(
        "Creates publication-quality plots from loaded AnnData modalities: "
        "UMAP/t-SNE/PCA embeddings, QC plots, violin, feature, dot plots, "
        "heatmaps, elbow, and cluster-composition figures."
    ),
    factory_function="lobster.agents.visualization_expert.visualization_expert",
    handoff_tool_name="handoff_to_visualization_expert_agent",
    handoff_tool_description=(
        "Assign figure/plot generation from loaded single-cell or expression "
        "modalities: UMAP, t-SNE, PCA embeddings, QC plots (violin of counts/"
        "genes/mito), violin plots, feature/gene expression plots, dot plots, "
        "heatmaps, elbow plots, and cluster-composition bar charts. Use when the "
        "user asks to plot, visualize, or draw a figure of AnnData results."
    ),
    code_tool_default=True,
)

# === Heavy imports below ===
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.visualization_state import VisualizationExpertState  # noqa: F401
from lobster.config.inference_context import (
    BoundModelFactory,
    InferenceContext,
    validate_inference_binding,
)
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.figures.errors import FigureError
from lobster.core.figures.records import (
    STATUS_AVAILABLE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUSES,
    CommitResult,
)
from lobster.core.figures.saver import (
    HEADER_NONE,
    HEADER_PENDING,
    failure_summary,
    outcome_state,
    summarize,
)
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.visualization.visualization_service import (
    SingleCellVisualizationService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def _plot_ir(
    tool_name: str,
    service_method: str,
    plot_kwargs: dict,
    description: str,
    replayable: bool = False,
) -> AnalysisStep:
    """Notebook drawing recipe; QC computations belong to their analysis step."""
    kwargs_template = "".join(
        f"    {name}={{{{ plot_kwargs[{name!r}] | repr }}}},\n" for name in plot_kwargs
    )
    return AnalysisStep(
        operation=f"lobster.visualization.{service_method}",
        tool_name=tool_name,
        description=description,
        library="plotly",
        code_template=(
            "from lobster.services.visualization.visualization_service import (\n"
            "    SingleCellVisualizationService,\n"
            ")\n\n"
            f"fig = SingleCellVisualizationService().{service_method}(\n"
            "    adata=adata,\n"
            f"{kwargs_template}"
            ")\n"
            "fig.show()"
        ),
        imports=[],
        parameters={"plot_kwargs": dict(plot_kwargs)},
        parameter_schema={},
        input_entities=["adata"],
        output_entities=["adata", "fig"] if replayable else ["fig"],
        replayable=replayable,
    )


def visualization_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "visualization_expert_agent",
    delegation_tools: List = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    inference_context: Optional[InferenceContext] = None,
    model_factory: Optional[BoundModelFactory] = None,
):
    """
    Create visualization expert agent with supervisor-mediated flow.

    Args:
        data_manager: DataManagerV2 instance for modality access
        callback_handler: Optional callback handler
        agent_name: Name for the agent
        delegation_tools: List of delegation tools for child agent handoffs
        workspace_path: Optional workspace path for LLM configuration
    """

    strict_binding = validate_inference_binding(inference_context, model_factory)
    model_params = {}
    if not strict_binding:
        settings = get_settings()
        model_params = settings.get_agent_llm_params("visualization_expert_agent")
    llm = create_llm(
        "visualization_expert_agent",
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
        **(
            {"inference_context": inference_context, "model_factory": model_factory}
            if strict_binding
            else {}
        ),
    )

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Initialize visualization service
    visualization_service = SingleCellVisualizationService()

    def _require_staged(staged_id, stats):
        if not staged_id or stats.get("status") != "staged":
            raise ValueError("Figure staging failed; no revision was created")

    def _discard_failed_stage(staged_id):
        if staged_id is None:
            return
        # Best-effort cleanup must not mask the primary failure.
        try:
            data_manager.plot_manager.discard_staged([staged_id])
        except Exception:
            logger.warning("Unable to discard an unavailable staged figure")

    def _failure_reply(staged_id, exc, tool_name):
        _discard_failed_stage(staged_id)
        summary = failure_summary(exc)
        normalized = []
        if isinstance(exc, FigureError) and isinstance(exc.details, dict):
            results = exc.details.get("results")
            if (
                isinstance(results, list)
                and results
                and all(isinstance(result, dict) for result in results)
            ):
                for index, result in enumerate(results):
                    status = result.get("status")
                    if not isinstance(status, str) or status not in STATUSES:
                        status = STATUS_FAILED
                    normalized.append(CommitResult(index, status))
        # Raised details never authenticate an available ref. Preserve pending
        # status, but surface all reason text through the central summary only.
        if any(result.status == STATUS_PENDING for result in normalized):
            logger.warning(
                "Figure producer %s publication pending: %s", tool_name, summary
            )
            overall = (
                "PENDING"
                if all(result.status == STATUS_PENDING for result in normalized)
                else "PARTIAL"
            )
            return f"STATUS: {overall}\ntool={tool_name}\n{summary}\n{HEADER_PENDING}"
        logger.error("Figure producer %s failed: %s", tool_name, summary)
        return f"STATUS: FAILED\ntool={tool_name}\n{summary}\nNo available figure revision was returned."

    def _finish_figure(
        staged_id, activity, tool_call_id, tool_name, save_plot, details
    ):
        # All service work, metadata/IR construction and logging precede this fence.
        if not save_plot:
            data_manager.plot_manager.discard_staged([staged_id])
            return (
                f"STATUS: DRY_RUN\n{details}\n"
                "Drawing discarded; not saved; no figure revision created; no view delivered."
            )
        results = data_manager.plot_manager.commit_staged(
            activity,
            tool_call_id=tool_call_id,
            staged_ids=[staged_id],
            tool_name=tool_name,
            agent=agent_name,
        )
        # Only render trusted results after commit: no writes or data lookups.
        state = outcome_state(results)
        summary = summarize(results)
        if state == "pending":
            return f"STATUS: PENDING\n{details}\n{HEADER_PENDING}\n{summary}"
        if state == "none":
            return f"STATUS: FAILED\n{details}\n{HEADER_NONE}\n{summary}"
        references = "\n".join(
            f"figure_ref={result.ref} path={result.path} status={result.status}"
            for result in results
            if result.status == STATUS_AVAILABLE
        )
        overall = (
            "PARTIAL"
            if any(result.status == STATUS_PENDING for result in results)
            else "SUCCESS"
        )
        return f"STATUS: {overall}\n{details}\n{references}\n{summary}\nFigure revision available; display not confirmed."

    @tool
    def check_visualization_readiness(modality_name: str) -> str:
        """Check if modality has required data for visualization."""
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"
            adata = data_manager.get_modality(modality_name)
            available_plots = []
            if "X_umap" in adata.obsm:
                available_plots.append("UMAP")
            if "X_pca" in adata.obsm:
                available_plots.append("PCA")
            if "X_tsne" in adata.obsm:
                available_plots.append("t-SNE")
            _known_cluster_names = {"leiden", "louvain", "seurat_clusters", "cluster"}
            has_clusters = any(
                c in _known_cluster_names
                or c.startswith(("leiden_", "louvain_", "RNA_snn_res"))
                for c in adata.obs.columns
            )
            if has_clusters:
                available_plots.append("Cluster-based plots")
            if "cell_type" in adata.obs.columns:
                available_plots.append("Cell type plots")
            # Check for batch information
            batch_cols = [
                col
                for col in adata.obs.columns
                if any(b in col.lower() for b in ["batch", "sample", "donor"])
            ]
            if batch_cols:
                available_plots.append("Batch composition plots")
            return f"""Visualization readiness for '{modality_name}':
- Available plots: {", ".join(available_plots) if available_plots else "None"}
- Data shape: {adata.n_obs} × {adata.n_vars}
- Has raw data: {"✓" if adata.raw else "✗"}
- Embeddings: {list(adata.obsm.keys()) if adata.obsm else "None"}"""
        except Exception as e:
            return f"Error checking visualization readiness: {str(e)}"

    check_visualization_readiness.metadata = {
        "categories": ["UTILITY"],
        "provenance": False,
    }
    check_visualization_readiness.tags = ["UTILITY"]

    @tool
    def create_umap_plot(
        modality_name: str,
        color_by: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        point_size: Optional[int] = None,
        title: Optional[str] = None,
        save_plot: bool = True,
    ) -> str:
        """Draw a stored UMAP, then save a revision only on success.

        Call check_visualization_readiness first; color_by is an existing obs
        column (never assume leiden). save_plot=False discards the drawing; no view is delivered.
        """
        staged_id = None
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"
            adata = data_manager.get_modality(modality_name)
            if point_size is None:
                n_cells = adata.n_obs
                point_size = (
                    8
                    if n_cells < 1000
                    else 5 if n_cells < 10000 else 3 if n_cells < 50000 else 2
                )
            plot_kwargs = {
                "color_by": color_by,
                "point_size": point_size,
                "title": title or f"UMAP - {color_by}",
            }
            fig = visualization_service.create_umap_plot(adata=adata, **plot_kwargs)
            details = f"plot_type=umap modality_name={modality_name} color_by={color_by} cells={adata.n_obs} point_size={point_size}"
            staged_id, stats, _ = data_manager.plot_manager.add_plot(
                plot=fig,
                title=f"UMAP - {color_by}",
                source="visualization_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "plot_type": "umap",
                    "color_by": color_by,
                    "n_cells": adata.n_obs,
                    "parameters": {"point_size": point_size, "title": title},
                },
                modalities=data_manager.modalities,
                current_dataset=data_manager.current_dataset,
            )
            _require_staged(staged_id, stats)
            activity = data_manager.log_tool_usage(
                tool_name="create_umap_plot",
                parameters={
                    "modality_name": modality_name,
                    "color_by": color_by,
                    "save_plot": save_plot,
                    "tool_call_id": tool_call_id,
                },
                description="Prepared UMAP figure",
                ir=_plot_ir(
                    "create_umap_plot",
                    "create_umap_plot",
                    plot_kwargs,
                    f"UMAP of '{modality_name}' coloured by {color_by}",
                ),
                agent=agent_name,
            )
            return _finish_figure(
                staged_id,
                activity,
                tool_call_id,
                "create_umap_plot",
                save_plot,
                details,
            )
        except Exception as exc:
            return _failure_reply(staged_id, exc, "create_umap_plot")

    create_umap_plot.metadata = {"categories": ["ANALYZE"], "provenance": True}
    create_umap_plot.tags = ["ANALYZE"]

    @tool
    def create_qc_plots(
        modality_name: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        metrics: List[str] = None,
        save_plot: bool = True,
    ) -> str:
        """Draw stored QC metrics; run assess_data_quality first. Dry runs deliver no view."""
        staged_id = None
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"
            adata = data_manager.get_modality(modality_name)
            is_de_modality = (
                "_de_" in modality_name.lower()
                or modality_name.lower().startswith("de_")
                or "differential" in modality_name.lower()
            )
            is_small_subset = adata.n_vars < 1000
            if is_de_modality or is_small_subset:
                return f"""⚠️ Cannot create QC plots for '{modality_name}'
QC metrics are intended for full datasets, not filtered DE/gene subsets.
Current modality: {adata.n_obs} cells × {adata.n_vars} genes.
Detection: {"DE result modality" if is_de_modality else "Small gene subset (<1000 genes)"}.
Use the full QC-assessed dataset; for DE results request volcano/MA/heatmap drawings."""
            required_qc_metrics = {
                "n_genes",
                "n_counts",
                "gene_detection_rate",
                "log10_counts",
                "log10_genes",
                "percent_mito",
                "percent_ribo",
                "percent_hb",
            }
            if not required_qc_metrics.issubset(adata.obs.columns):
                return (
                    f"STATUS: FAILED\nmodality_name={modality_name}\n"
                    "QC metrics are not ready. Run assess_data_quality "
                    "first, then draw the full QC-assessed modality. "
                    "Drawing does not compute QC metrics."
                )
            if metrics is None:
                metrics = [
                    column
                    for column in ("n_genes_by_counts", "total_counts", "pct_counts_mt")
                    if column in adata.obs.columns
                ]
                if not metrics:
                    metrics = adata.obs.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()[:3]
            if not metrics:
                return "No suitable QC metrics found; run assess_data_quality first"
            plot_kwargs = {"title": f"QC Plots - {modality_name}"}
            fig = visualization_service.create_qc_plots(adata=adata, **plot_kwargs)
            details = f"plot_type=qc modality_name={modality_name} metrics={','.join(metrics)} cells={adata.n_obs}"
            staged_id, stats, _ = data_manager.plot_manager.add_plot(
                plot=fig,
                title=f"QC Plots - {modality_name}",
                source="visualization_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "plot_type": "qc",
                    "metrics": metrics,
                    "n_cells": adata.n_obs,
                },
                modalities=data_manager.modalities,
                current_dataset=data_manager.current_dataset,
            )
            _require_staged(staged_id, stats)
            activity = data_manager.log_tool_usage(
                tool_name="create_qc_plots",
                parameters={
                    "modality_name": modality_name,
                    "metrics": metrics,
                    "save_plot": save_plot,
                    "tool_call_id": tool_call_id,
                },
                description="Prepared QC figure from stored analysis metrics",
                ir=_plot_ir(
                    "create_qc_plots",
                    "create_qc_plots",
                    plot_kwargs,
                    f"QC drawing of '{modality_name}' from existing metrics",
                ),
                agent=agent_name,
            )
            return _finish_figure(
                staged_id, activity, tool_call_id, "create_qc_plots", save_plot, details
            )
        except Exception as exc:
            return _failure_reply(staged_id, exc, "create_qc_plots")

    create_qc_plots.metadata = {"categories": ["ANALYZE"], "provenance": True}
    create_qc_plots.tags = ["ANALYZE"]

    @tool
    def create_violin_plot(
        modality_name: str,
        genes: List[str],
        groupby: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        save_plot: bool = True,
    ) -> str:
        """Draw supplied genes by an existing group column; dry runs discard the drawing."""
        staged_id = None
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"
            adata = data_manager.get_modality(modality_name)
            available_genes = [gene for gene in genes if gene in adata.var_names]
            if not available_genes:
                return f"None of the specified genes {genes} found in {modality_name}"
            plot_kwargs = {"genes": available_genes, "groupby": groupby}
            fig = visualization_service.create_violin_plot(adata=adata, **plot_kwargs)
            details = f"plot_type=violin modality_name={modality_name} genes={','.join(available_genes)} groupby={groupby} cells={adata.n_obs}"
            staged_id, stats, _ = data_manager.plot_manager.add_plot(
                plot=fig,
                title=f"Violin Plot - {', '.join(available_genes[:3])}{'...' if len(available_genes) > 3 else ''}",
                source="visualization_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "plot_type": "violin",
                    "genes": available_genes,
                    "groupby": groupby,
                    "n_cells": adata.n_obs,
                },
                modalities=data_manager.modalities,
                current_dataset=data_manager.current_dataset,
            )
            _require_staged(staged_id, stats)
            activity = data_manager.log_tool_usage(
                tool_name="create_violin_plot",
                parameters={
                    "modality_name": modality_name,
                    "genes": available_genes,
                    "groupby": groupby,
                    "save_plot": save_plot,
                    "tool_call_id": tool_call_id,
                },
                description="Prepared violin figure",
                ir=_plot_ir(
                    "create_violin_plot",
                    "create_violin_plot",
                    plot_kwargs,
                    f"Violin plot of {len(available_genes)} genes in '{modality_name}'",
                ),
                agent=agent_name,
            )
            return _finish_figure(
                staged_id,
                activity,
                tool_call_id,
                "create_violin_plot",
                save_plot,
                details,
            )
        except Exception as exc:
            return _failure_reply(staged_id, exc, "create_violin_plot")

    create_violin_plot.metadata = {"categories": ["ANALYZE"], "provenance": True}
    create_violin_plot.tags = ["ANALYZE"]

    @tool
    def create_feature_plot(
        modality_name: str,
        genes: List[str],
        tool_call_id: Annotated[str, InjectedToolCallId],
        use_raw: bool = True,
        ncols: int = 2,
        point_size: Optional[int] = None,
        save_plot: bool = True,
    ) -> str:
        """Draw gene expression on a stored UMAP; no statistical fitting."""
        staged_id = None
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"
            adata = data_manager.get_modality(modality_name)
            available_genes = [gene for gene in genes if gene in adata.var_names]
            if not available_genes:
                return f"None of the specified genes {genes} found in {modality_name}"
            plot_kwargs = {
                "genes": available_genes,
                "use_raw": use_raw,
                "ncols": ncols,
                "point_size": point_size,
            }
            fig = visualization_service.create_feature_plot(adata=adata, **plot_kwargs)
            details = f"plot_type=feature modality_name={modality_name} genes={','.join(available_genes[:5])} n_genes={len(available_genes)} ncols={ncols} use_raw={use_raw}"
            staged_id, stats, _ = data_manager.plot_manager.add_plot(
                plot=fig,
                title=f"Feature Plot - {', '.join(available_genes[:3])}{'...' if len(available_genes) > 3 else ''}",
                source="visualization_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "plot_type": "feature",
                    "genes": available_genes,
                    "n_genes": len(available_genes),
                    "parameters": {
                        "use_raw": use_raw,
                        "ncols": ncols,
                        "point_size": point_size,
                    },
                },
                modalities=data_manager.modalities,
                current_dataset=data_manager.current_dataset,
            )
            _require_staged(staged_id, stats)
            activity = data_manager.log_tool_usage(
                tool_name="create_feature_plot",
                parameters={
                    "modality_name": modality_name,
                    "genes": available_genes,
                    "save_plot": save_plot,
                    "tool_call_id": tool_call_id,
                },
                description="Prepared feature figure",
                ir=_plot_ir(
                    "create_feature_plot",
                    "create_feature_plot",
                    plot_kwargs,
                    f"Feature plot of {len(available_genes)} genes in '{modality_name}'",
                ),
                agent=agent_name,
            )
            return _finish_figure(
                staged_id,
                activity,
                tool_call_id,
                "create_feature_plot",
                save_plot,
                details,
            )
        except Exception as exc:
            return _failure_reply(staged_id, exc, "create_feature_plot")

    create_feature_plot.metadata = {"categories": ["ANALYZE"], "provenance": True}
    create_feature_plot.tags = ["ANALYZE"]

    @tool
    def create_dot_plot(
        modality_name: str,
        genes: List[str],
        groupby: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        use_raw: bool = True,
        standard_scale: str = "var",
        save_plot: bool = True,
    ) -> str:
        """Draw marker gene means/percent-expressing; groupby is an existing column."""
        staged_id = None
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"
            adata = data_manager.get_modality(modality_name)
            available_genes = [gene for gene in genes if gene in adata.var_names]
            if not available_genes:
                return f"None of the specified genes {genes} found in {modality_name}"
            plot_kwargs = {
                "genes": available_genes,
                "groupby": groupby,
                "use_raw": use_raw,
                "standard_scale": standard_scale,
            }
            fig = visualization_service.create_dot_plot(adata=adata, **plot_kwargs)
            details = f"plot_type=dot modality_name={modality_name} n_genes={len(available_genes)} groupby={groupby} use_raw={use_raw} scale={standard_scale} size=percent_expressing color=mean_expression"
            staged_id, stats, _ = data_manager.plot_manager.add_plot(
                plot=fig,
                title=f"Dot Plot - {groupby}",
                source="visualization_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "plot_type": "dot",
                    "genes": available_genes,
                    "groupby": groupby,
                    "n_genes": len(available_genes),
                    "parameters": {
                        "use_raw": use_raw,
                        "standard_scale": standard_scale,
                    },
                },
                modalities=data_manager.modalities,
                current_dataset=data_manager.current_dataset,
            )
            _require_staged(staged_id, stats)
            activity = data_manager.log_tool_usage(
                tool_name="create_dot_plot",
                parameters={
                    "modality_name": modality_name,
                    "genes": available_genes,
                    "groupby": groupby,
                    "save_plot": save_plot,
                    "tool_call_id": tool_call_id,
                },
                description="Prepared dot figure",
                ir=_plot_ir(
                    "create_dot_plot",
                    "create_dot_plot",
                    plot_kwargs,
                    f"Dot plot of {len(available_genes)} genes by {groupby} in '{modality_name}'",
                ),
                agent=agent_name,
            )
            return _finish_figure(
                staged_id, activity, tool_call_id, "create_dot_plot", save_plot, details
            )
        except Exception as exc:
            return _failure_reply(staged_id, exc, "create_dot_plot")

    create_dot_plot.metadata = {"categories": ["ANALYZE"], "provenance": True}
    create_dot_plot.tags = ["ANALYZE"]

    @tool
    def create_heatmap(
        modality_name: str,
        groupby: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        genes: Optional[List[str]] = None,
        use_raw: bool = True,
        n_top_genes: int = 5,
        standard_scale: bool = True,
        save_plot: bool = True,
    ) -> str:
        """Draw expression heatmap from explicit genes or stored marker results."""
        staged_id = None
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"
            adata = data_manager.get_modality(modality_name)
            plot_kwargs = {
                "genes": genes,
                "groupby": groupby,
                "use_raw": use_raw,
                "n_top_genes": n_top_genes,
                "standard_scale": standard_scale,
            }
            fig = visualization_service.create_heatmap(adata=adata, **plot_kwargs)
            details = f"plot_type=heatmap modality_name={modality_name} groupby={groupby} genes={len(genes) if genes else 'stored_markers'} n_top_genes={n_top_genes} use_raw={use_raw} standard_scale={standard_scale}"
            staged_id, stats, _ = data_manager.plot_manager.add_plot(
                plot=fig,
                title=f"Heatmap - {groupby}",
                source="visualization_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "plot_type": "heatmap",
                    "groupby": groupby,
                    "n_genes": len(genes) if genes else "auto",
                    "parameters": {
                        "use_raw": use_raw,
                        "n_top_genes": n_top_genes,
                        "standard_scale": standard_scale,
                    },
                },
                modalities=data_manager.modalities,
                current_dataset=data_manager.current_dataset,
            )
            _require_staged(staged_id, stats)
            activity = data_manager.log_tool_usage(
                tool_name="create_heatmap",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "save_plot": save_plot,
                    "tool_call_id": tool_call_id,
                },
                description="Prepared heatmap figure",
                ir=_plot_ir(
                    "create_heatmap",
                    "create_heatmap",
                    plot_kwargs,
                    f"Expression heatmap by {groupby} in '{modality_name}'",
                ),
                agent=agent_name,
            )
            return _finish_figure(
                staged_id, activity, tool_call_id, "create_heatmap", save_plot, details
            )
        except Exception as exc:
            return _failure_reply(staged_id, exc, "create_heatmap")

    create_heatmap.metadata = {"categories": ["ANALYZE"], "provenance": True}
    create_heatmap.tags = ["ANALYZE"]

    @tool
    def create_elbow_plot(
        modality_name: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        n_pcs: int = 50,
        save_plot: bool = True,
    ) -> str:
        """Draw stored PCA variance explained; does not compute PCA."""
        staged_id = None
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"
            adata = data_manager.get_modality(modality_name)
            plot_kwargs = {"n_pcs": n_pcs}
            fig = visualization_service.create_elbow_plot(adata=adata, **plot_kwargs)
            details = (
                f"plot_type=elbow modality_name={modality_name} requested_n_pcs={n_pcs}"
            )
            staged_id, stats, _ = data_manager.plot_manager.add_plot(
                plot=fig,
                title="PCA Elbow Plot",
                source="visualization_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "plot_type": "elbow",
                    "n_pcs": n_pcs,
                    "parameters": {"n_pcs": n_pcs},
                },
                modalities=data_manager.modalities,
                current_dataset=data_manager.current_dataset,
            )
            _require_staged(staged_id, stats)
            activity = data_manager.log_tool_usage(
                tool_name="create_elbow_plot",
                parameters={
                    "modality_name": modality_name,
                    "n_pcs": n_pcs,
                    "save_plot": save_plot,
                    "tool_call_id": tool_call_id,
                },
                description="Prepared elbow figure",
                ir=_plot_ir(
                    "create_elbow_plot",
                    "create_elbow_plot",
                    plot_kwargs,
                    f"PCA elbow plot of '{modality_name}'",
                ),
                agent=agent_name,
            )
            return _finish_figure(
                staged_id,
                activity,
                tool_call_id,
                "create_elbow_plot",
                save_plot,
                details,
            )
        except Exception as exc:
            return _failure_reply(staged_id, exc, "create_elbow_plot")

    create_elbow_plot.metadata = {"categories": ["ANALYZE"], "provenance": True}
    create_elbow_plot.tags = ["ANALYZE"]

    @tool
    def create_cluster_composition_plot(
        modality_name: str,
        cluster_col: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        sample_col: Optional[str] = None,
        normalize: bool = True,
        save_plot: bool = True,
    ) -> str:
        """Draw cluster counts/composition using an existing cluster column."""
        staged_id = None
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"
            adata = data_manager.get_modality(modality_name)
            plot_kwargs = {
                "cluster_col": cluster_col,
                "sample_col": sample_col,
                "normalize": normalize,
            }
            fig = visualization_service.create_cluster_composition_plot(
                adata=adata, **plot_kwargs
            )
            details = f"plot_type=composition modality_name={modality_name} cluster_col={cluster_col} sample_col={sample_col or 'auto'} normalize={normalize}"
            staged_id, stats, _ = data_manager.plot_manager.add_plot(
                plot=fig,
                title=f"Cluster Composition - {cluster_col}",
                source="visualization_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "plot_type": "composition",
                    "cluster_col": cluster_col,
                    "sample_col": sample_col,
                    "parameters": {
                        "cluster_col": cluster_col,
                        "sample_col": sample_col,
                        "normalize": normalize,
                    },
                },
                modalities=data_manager.modalities,
                current_dataset=data_manager.current_dataset,
            )
            _require_staged(staged_id, stats)
            activity = data_manager.log_tool_usage(
                tool_name="create_cluster_composition_plot",
                parameters={
                    "modality_name": modality_name,
                    "cluster_col": cluster_col,
                    "sample_col": sample_col,
                    "save_plot": save_plot,
                    "tool_call_id": tool_call_id,
                },
                description="Prepared cluster composition figure",
                ir=_plot_ir(
                    "create_cluster_composition_plot",
                    "create_cluster_composition_plot",
                    plot_kwargs,
                    f"Cluster composition of '{modality_name}' by {sample_col}",
                ),
                agent=agent_name,
            )
            return _finish_figure(
                staged_id,
                activity,
                tool_call_id,
                "create_cluster_composition_plot",
                save_plot,
                details,
            )
        except Exception as exc:
            return _failure_reply(staged_id, exc, "create_cluster_composition_plot")

    create_cluster_composition_plot.metadata = {
        "categories": ["ANALYZE"],
        "provenance": True,
    }
    create_cluster_composition_plot.tags = ["ANALYZE"]

    @tool
    def get_visualization_history() -> str:
        """Read LEGACY visualization history, not the managed figure catalog."""
        try:
            history, _, _ = data_manager.plot_manager.get_visualization_history(
                limit=10
            )
            if not history:
                return "LEGACY history is empty. This does not mean no managed figures exist; use the canonical workspace figure listing."
            response = "LEGACY Visualization History (not managed revisions):\n\n"
            for index, record in enumerate(history, 1):
                metadata = record.get("metadata", {})
                response += f"{index}. {metadata.get('type', 'unknown')} for {metadata.get('modality', 'unknown')}\n"
                response += f"   Legacy plot ID: {record.get('plot_id', 'N/A')}\n"
                response += f"   Created: {record.get('timestamp', 'N/A')}\n\n"
            return (
                response
                + "For managed figure revisions, use the canonical workspace listing."
            )
        except Exception as exc:
            return f"Error retrieving legacy history: {exc}"

    get_visualization_history.metadata = {
        "categories": ["UTILITY"],
        "provenance": False,
    }
    get_visualization_history.tags = ["UTILITY"]

    @tool
    def report_visualization_complete(
        requesting_agent: str, plot_id: str, status: str = "success", message: str = ""
    ) -> str:
        """Report visualization completion back to supervisor."""
        try:
            completion_report = f"""
📊 **Visualization Task Complete**

**Status**: {status.upper()}
**Plot ID**: {plot_id}
**Requesting Agent**: {requesting_agent}
**Timestamp**: {pd.Timestamp.now()}

{f"**Message**: {message}" if message else ""}

**Action Required**: Please inform {requesting_agent} that visualization {plot_id} is ready."""
            return completion_report
        except Exception as exc:
            return f"Error reporting completion: {exc}"

    report_visualization_complete.metadata = {
        "categories": ["UTILITY"],
        "provenance": False,
    }
    report_visualization_complete.tags = ["UTILITY"]

    base_tools = [
        check_visualization_readiness,
        create_umap_plot,
        create_qc_plots,
        create_violin_plot,
        create_feature_plot,
        create_dot_plot,
        create_heatmap,
        create_elbow_plot,
        create_cluster_composition_plot,
        get_visualization_history,
        report_visualization_complete,
    ]
    tools = base_tools + (delegation_tools or [])
    system_prompt = """
You are a visualization expert for interactive bioinformatics figures.
All requests and results go through the supervisor; never contact another agent directly.

<Task>
1. Validate modality and required data with check_visualization_readiness.
2. Draw the requested figure from existing analysis results.
3. The retained plotting tools stage a candidate, log the activity, then commit
   through the trusted saver only on success. save_plot=False is a dry run:
   the drawing is discarded, not saved, and no view is delivered.
4. Report ONLY committed figure_ref/path from the tool reply. Refer to canonical
   workspace figure listings for managed revisions; legacy history is not the catalog.
5. Report failures or dry runs explicitly; never claim a figure was displayed.
</Task>

<Guidelines>
- Do not create UUIDs or write visualization history: revision identity is assigned by the authority.
- QC metrics must already have been computed by assess_data_quality; drawings never mutate datasets.
- Prefer templates/custom code for flexible figures, following the shared tool guidance.
- Preserve the existing data and report the actual tool outcome, not an invented success.
- If using report_visualization_complete, its plot_id argument must be the actual committed FigureRef;
  it is a report formatter, not evidence that a revision exists or was displayed.
</Guidelines>

<Response_Format>
Responses are read by the supervisor AI. Lead with STATUS: SUCCESS | DRY_RUN | PENDING | PARTIAL | FAILED.
Use compact key=value fields: figure_ref, plot_type, modality_name, path, warnings, next steps.
Use only committed refs/paths for SUCCESS; dry runs create no revision and deliver no view.
PENDING means publication is unresolved, not available yet; do not claim failure or that nothing was saved.
</Response_Format>
""".strip()
    from lobster.tools.code_tool_switch import attach_code_tool

    tools = attach_code_tool(
        tools,
        data_manager,
        AGENT_CONFIG.name,
        default=AGENT_CONFIG.code_tool_default,
        tool_agent_name=agent_name,
    )
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=VisualizationExpertState,
    )
