"""
Shared tool factories for ML agents.

Provides common tool creation functions used across
feature_selection_expert and survival_analysis_expert.
"""

from typing import List, Optional

from langchain_core.tools import tool

from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.analysis.pathway_enrichment_bridge_service import (
    PathwayEnrichmentBridgeService,
    PathwayEnrichmentError,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "create_feature_selection_tools",
    "create_survival_analysis_tools",
]


def create_feature_selection_tools(data_manager: DataManagerV2) -> List:
    """
    Create tools for the feature selection expert agent.

    Args:
        data_manager: DataManagerV2 instance for data access

    Returns:
        List of tools for feature selection
    """
    from lobster.services.ml.feature_selection_service import FeatureSelectionService

    fs_service = FeatureSelectionService()
    pathway_bridge_service = PathwayEnrichmentBridgeService()

    @tool
    def run_stability_selection(
        modality_name: str,
        target_column: str,
        n_features: int = 100,
        n_rounds: int = 10,
        method: str = "xgboost",
    ) -> str:
        """
        Run stability-based feature selection using bootstrap resampling.

        Trains models on multiple data subsamples and ranks features by
        selection frequency/importance stability across rounds.

        Args:
            modality_name: Name of the modality to analyze
            target_column: Column in obs containing target variable
            n_features: Number of top features to select
            n_rounds: Number of bootstrap rounds
            method: Selection method ("xgboost", "random_forest", "lasso")

        Returns:
            Summary of feature selection results
        """
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

            adata = data_manager.get_modality(modality_name)

            adata_result, stats, ir = fs_service.stability_selection(
                adata=adata,
                target_column=target_column,
                n_features=n_features,
                n_rounds=n_rounds,
                method=method,
            )

            # Store result
            result_name = f"{modality_name}_feature_selected"
            data_manager.store_modality(
                name=result_name,
                adata=adata_result,
                parent_name=modality_name,
                step_summary=f"Stability selection: {n_features} features",
            )

            data_manager.log_tool_usage(
                tool_name="run_stability_selection",
                parameters={"modality_name": modality_name, "n_features": n_features},
                description=stats,
                ir=ir,
            )

            # Format response
            response = f"""Stability-based feature selection complete!

**Method**: {method} with {n_rounds} bootstrap rounds

**Results**:
- Total features: {stats["n_total_features"]}
- Selected features: {stats["n_selected_features"]}
- Average selection probability: {stats["avg_selection_probability"]:.2f}

**Top 10 Features**:
"""
            for f in stats["top_features"][:10]:
                response += f"  {f['rank']}. {f['feature']} (importance={f['mean_importance']:.4f}, prob={f['selection_probability']:.2f})\n"

            response += f"\n**Result stored as**: '{result_name}'"

            return response

        except Exception as e:
            logger.error(f"Error in stability selection: {e}")
            return f"Error: {str(e)}"

    run_stability_selection.metadata = {"categories": ["ANALYZE"], "provenance": True}
    run_stability_selection.tags = ["ANALYZE"]

    @tool
    def run_lasso_selection(
        modality_name: str,
        target_column: str,
        alpha: float = 0.1,
    ) -> str:
        """
        Run LASSO (L1) feature selection.

        Uses L1 regularization to automatically select features
        by driving coefficients to zero.

        Args:
            modality_name: Name of the modality to analyze
            target_column: Column in obs containing target variable
            alpha: Regularization strength (higher = more sparsity)

        Returns:
            Summary of LASSO selection results
        """
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found."

            adata = data_manager.get_modality(modality_name)

            adata_result, stats, ir = fs_service.lasso_selection(
                adata=adata,
                target_column=target_column,
                alpha=alpha,
            )

            result_name = f"{modality_name}_lasso_selected"
            data_manager.store_modality(
                name=result_name,
                adata=adata_result,
                parent_name=modality_name,
                step_summary=f"LASSO selection: {stats['n_selected_features']} features",
            )

            data_manager.log_tool_usage(
                tool_name="run_lasso_selection",
                parameters={"modality_name": modality_name, "alpha": alpha},
                description=stats,
                ir=ir,
            )

            response = f"""LASSO feature selection complete!

**Parameters**: alpha={alpha}

**Results**:
- Total features: {stats["n_total_features"]}
- Selected features: {stats["n_selected_features"]} ({stats["selection_rate"]:.1%})

**Top Features by Coefficient**:
"""
            for f in stats["top_features"][:10]:
                response += f"  - {f['feature']}: {f['coefficient']:.4f}\n"

            response += f"\n**Result stored as**: '{result_name}'"

            return response

        except Exception as e:
            logger.error(f"Error in LASSO selection: {e}")
            return f"Error: {str(e)}"

    run_lasso_selection.metadata = {"categories": ["ANALYZE"], "provenance": True}
    run_lasso_selection.tags = ["ANALYZE"]

    @tool
    def run_variance_filter(
        modality_name: str,
        percentile: float = 10.0,
    ) -> str:
        """
        Filter low-variance features.

        Removes features below the specified variance percentile.

        Args:
            modality_name: Name of the modality to filter
            percentile: Remove features below this variance percentile

        Returns:
            Summary of variance filtering
        """
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found."

            adata = data_manager.get_modality(modality_name)

            adata_result, stats, ir = fs_service.variance_filter(
                adata=adata,
                percentile=percentile,
            )

            result_name = f"{modality_name}_variance_filtered"
            data_manager.store_modality(
                name=result_name,
                adata=adata_result,
                parent_name=modality_name,
                step_summary=f"Variance filter: kept {stats['n_selected_features']} features",
            )

            data_manager.log_tool_usage(
                tool_name="run_variance_filter",
                parameters={"modality_name": modality_name, "percentile": percentile},
                description=stats,
                ir=ir,
            )

            return f"""Variance filtering complete!

**Threshold**: {stats["variance_threshold"]:.6f} (percentile {percentile})

**Results**:
- Original features: {stats["n_total_features"]}
- Kept features: {stats["n_selected_features"]}
- Removed features: {stats["n_removed_features"]} ({stats["removal_rate"]:.1%})

**Result stored as**: '{result_name}'"""

        except Exception as e:
            return f"Error: {str(e)}"

    run_variance_filter.metadata = {"categories": ["FILTER"], "provenance": True}
    run_variance_filter.tags = ["FILTER"]

    @tool
    def enrich_pathways_for_selected_features(
        modality_name: str,
        selection_method: Optional[str] = None,
        sources: Optional[List[str]] = None,
        fdr_threshold: float = 0.05,
    ) -> str:
        """
        Perform pathway enrichment on features selected by feature selection.

        IMPORTANT: Call this AFTER stability_selection, lasso_selection, or variance_filter.
        The service reads *_selected columns from adata.var to identify selected features.

        Uses INDRA Discovery API (hosted REST) for pathway enrichment.
        NO local setup required - just HTTP requests to discovery.indra.bio.

        Args:
            modality_name: AnnData modality containing selected features
            selection_method: Which selection to use if multiple exist
                ("stability", "lasso", "variance"). Auto-detects if only one exists.
            sources: Pathway databases (default: ["go", "reactome"])
                Options: "go", "reactome", "wikipathways", "indra-upstream", "indra-downstream"
            fdr_threshold: FDR significance threshold (default: 0.05)

        Returns:
            Summary with pathway counts, top pathways, and storage locations.
            Full results stored in adata.uns['pathway_enrichment'] and workspace CSV.

        Example:
            # After running stability_selection:
            enrich_pathways_for_selected_features(
                modality_name="geo_gse12345_filtered",
            )
        """
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Error: Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

            adata = data_manager.get_modality(modality_name)

            result, stats, ir = pathway_bridge_service.enrich_selected_features(
                adata=adata,
                modality_name=modality_name,
                selection_method=selection_method,
                sources=sources,
                fdr_threshold=fdr_threshold,
            )

            # Store updated AnnData (same modality, enrichment added to uns)
            data_manager.modalities[modality_name] = result

            # Log with IR (CRITICAL: ir=ir parameter for provenance)
            data_manager.log_tool_usage(
                tool_name="enrich_pathways_for_selected_features",
                parameters={
                    "modality_name": modality_name,
                    "selection_method": selection_method,
                    "sources": sources,
                    "fdr_threshold": fdr_threshold,
                },
                description=stats,
                ir=ir,
            )

            # Format summary response
            response = f"""Pathway enrichment complete!

**Selection**: {stats["selection_method"]}
**Genes selected**: {stats["n_genes_selected"]}

**Enrichment Results** (via INDRA Discovery API):
- Sources queried: {", ".join(stats["sources"])}
- Pathways found: {stats["n_pathways_tested"]}
- Significant (FDR < {stats["fdr_threshold"]}): {stats["n_pathways_significant"]}

**Top Pathways** (by FDR):
"""
            for i, pathway in enumerate(stats.get("top_pathways", [])[:5], 1):
                genes = (
                    pathway["gene_overlap"][:50] + "..."
                    if len(pathway["gene_overlap"]) > 50
                    else pathway["gene_overlap"]
                )
                response += f"  {i}. {pathway['pathway_name']}\n"
                response += f"     FDR: {pathway['fdr']:.2e}, Overlap: {pathway['overlap_count']} genes\n"
                response += f"     Genes: {genes}\n"

            response += f"""
**Results stored in**:
- adata.uns['pathway_enrichment'] (summary)
- Workspace CSV: {modality_name}_pathway_enrichment.csv (full details)
"""
            return response

        except ValueError as e:
            # User-facing errors (no selection, multiple selections without specifying)
            return f"Error: {str(e)}"
        except PathwayEnrichmentError as e:
            # API errors (timeout, unavailable)
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Pathway enrichment failed: {e}")
            return f"Error: Pathway enrichment failed. {str(e)}"

    enrich_pathways_for_selected_features.metadata = {"categories": ["ANALYZE"], "provenance": True}
    enrich_pathways_for_selected_features.tags = ["ANALYZE"]

    return [
        run_stability_selection,
        run_lasso_selection,
        run_variance_filter,
        enrich_pathways_for_selected_features,
    ]


def create_survival_analysis_tools(data_manager: DataManagerV2) -> List:
    """
    Create tools for the survival analysis expert agent.

    Args:
        data_manager: DataManagerV2 instance for data access

    Returns:
        List of tools for survival analysis
    """
    from lobster.services.analysis.survival_analysis_service import (
        SurvivalAnalysisService,
    )

    survival_service = SurvivalAnalysisService()

    @tool
    def train_cox_model(
        modality_name: str,
        time_column: str,
        event_column: str,
        l1_ratio: float = 0.5,
    ) -> str:
        """
        Train Cox proportional hazards model with elastic net regularization.

        Args:
            modality_name: Name of the modality to analyze
            time_column: Column in obs containing time-to-event
            event_column: Column in obs containing event indicator (0/1)
            l1_ratio: Elastic net mixing parameter (0=L2, 1=L1)

        Returns:
            Summary of Cox model training
        """
        try:
            avail = survival_service.check_availability()
            if not avail["ready"]:
                return f"Survival analysis not available. Install with: {avail['install_command']}"

            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found."

            adata = data_manager.get_modality(modality_name)

            adata_result, stats, ir = survival_service.train_cox_model(
                adata=adata,
                time_column=time_column,
                event_column=event_column,
                l1_ratio=l1_ratio,
            )

            result_name = f"{modality_name}_cox_model"
            data_manager.store_modality(
                name=result_name,
                adata=adata_result,
                parent_name=modality_name,
                step_summary=f"Cox model: C-index={stats['c_index']:.3f}",
            )

            data_manager.log_tool_usage(
                tool_name="train_cox_model",
                parameters={"modality_name": modality_name, "time_column": time_column},
                description=stats,
                ir=ir,
            )

            return f"""Cox proportional hazards model trained!

**Model Performance**:
- C-index: {stats["c_index"]:.3f}
- Concordant pairs: {stats["concordant_pairs"]}
- Discordant pairs: {stats["discordant_pairs"]}

**Data Summary**:
- Samples: {stats["n_samples"]}
- Events: {stats["n_events"]} ({stats["event_rate"]:.1%})
- Features selected: {stats["n_features_selected"]}

**Risk scores stored in**: '{result_name}' (obs['cox_risk_score'])"""

        except Exception as e:
            return f"Error: {str(e)}"

    train_cox_model.metadata = {"categories": ["ANALYZE"], "provenance": True}
    train_cox_model.tags = ["ANALYZE"]

    @tool
    def optimize_risk_threshold(
        modality_name: str,
        time_column: str,
        event_column: str,
        time_horizon: Optional[float] = None,
        n_bootstrap: int = 100,
    ) -> str:
        """
        Find optimal threshold for risk classification using MCC optimization.

        Args:
            modality_name: Name of modality with Cox risk scores
            time_column: Column with time-to-event
            event_column: Column with event indicator
            time_horizon: Time cutoff for binary outcome (default: median event time)
            n_bootstrap: Number of bootstrap iterations

        Returns:
            Summary of threshold optimization
        """
        try:
            avail = survival_service.check_availability()
            if not avail["ready"]:
                return "Survival analysis not available."

            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found."

            adata = data_manager.get_modality(modality_name)

            adata_result, stats, ir = survival_service.optimize_threshold(
                adata=adata,
                time_column=time_column,
                event_column=event_column,
                time_horizon=time_horizon,
                n_bootstrap=n_bootstrap,
            )

            data_manager.modalities[modality_name] = adata_result

            data_manager.log_tool_usage(
                tool_name="optimize_risk_threshold",
                parameters={"modality_name": modality_name},
                description=stats,
                ir=ir,
            )

            cm = stats["confusion_matrix"]

            return f"""Risk threshold optimization complete!

**Optimal Threshold**: {stats["best_threshold"]:.4f}
**Time Horizon**: {stats["time_horizon"]:.1f}

**Performance Metrics**:
- MCC: {stats["mcc"]:.3f} (+/- {stats["mcc_std"]:.3f})
- Sensitivity: {stats["sensitivity"]:.3f}
- Specificity: {stats["specificity"]:.3f}
- PPV: {stats["ppv"]:.3f}
- NPV: {stats["npv"]:.3f}
- Accuracy: {stats["accuracy"]:.3f}

**Risk Groups**:
- High risk: {stats["n_high_risk"]}
- Low risk: {stats["n_low_risk"]}

**Confusion Matrix**:
                Predicted
              Neg    Pos
Actual Neg   {cm["tn"]:4d}   {cm["fp"]:4d}
       Pos   {cm["fn"]:4d}   {cm["tp"]:4d}

Risk categories stored in obs['risk_category']"""

        except Exception as e:
            return f"Error: {str(e)}"

    optimize_risk_threshold.metadata = {"categories": ["ANALYZE"], "provenance": True}
    optimize_risk_threshold.tags = ["ANALYZE"]

    @tool
    def run_kaplan_meier(
        modality_name: str,
        time_column: str,
        event_column: str,
        group_column: Optional[str] = None,
    ) -> str:
        """
        Perform Kaplan-Meier survival analysis.

        Args:
            modality_name: Name of the modality to analyze
            time_column: Column with time-to-event
            event_column: Column with event indicator
            group_column: Optional column for stratification

        Returns:
            Summary of Kaplan-Meier analysis
        """
        try:
            avail = survival_service.check_availability()
            if not avail["ready"]:
                return "Survival analysis not available."

            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found."

            adata = data_manager.get_modality(modality_name)

            adata_result, stats, ir = survival_service.kaplan_meier_analysis(
                adata=adata,
                time_column=time_column,
                event_column=event_column,
                group_column=group_column,
            )

            data_manager.modalities[modality_name] = adata_result

            data_manager.log_tool_usage(
                tool_name="run_kaplan_meier",
                parameters={
                    "modality_name": modality_name,
                    "group_column": group_column,
                },
                description=stats,
                ir=ir,
            )

            if group_column:
                response = f"""Kaplan-Meier analysis complete (stratified by {group_column})!

**Groups**: {stats["n_groups"]}
"""
                for group, group_stats in stats["groups"].items():
                    if isinstance(group_stats, dict):
                        response += f"\n**{group}**:\n"
                        response += f"  - Samples: {group_stats['n_samples']}\n"
                        response += f"  - Events: {group_stats['n_events']}\n"
                        if group_stats["median_survival"]:
                            response += f"  - Median survival: {group_stats['median_survival']:.1f}\n"

                if "log_rank_p_value" in stats["groups"]:
                    response += f"\n**Log-rank test**: chi2={stats['groups']['log_rank_chi2']:.2f}, p={stats['groups']['log_rank_p_value']:.4f}"
            else:
                response = f"""Kaplan-Meier analysis complete!

**Summary**:
- Samples: {stats["n_samples"]}
- Events: {stats["n_events"]}
- Median survival: {stats["median_survival"]:.1f if stats['median_survival'] else 'Not reached'}
- Max follow-up: {stats["max_time"]:.1f}"""

            response += "\n\nSurvival curves stored in adata.uns['kaplan_meier']"

            return response

        except Exception as e:
            return f"Error: {str(e)}"

    run_kaplan_meier.metadata = {"categories": ["ANALYZE"], "provenance": True}
    run_kaplan_meier.tags = ["ANALYZE"]

    return [
        train_cox_model,
        optimize_risk_threshold,
        run_kaplan_meier,
    ]
