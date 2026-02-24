"""
Proteomics Differential Expression Analysis Sub-Agent.

This sub-agent handles all proteomics differential expression analysis tools including:
- Statistical testing between groups with FDR control and platform-aware defaults
- Time course analysis for longitudinal proteomics studies
- Correlation analysis between proteins and continuous clinical variables

Platform-aware behavior:
- Mass spectrometry: limma-like moderated t-test, fold change threshold 1.5x
- Affinity proteomics: standard t-test, fold change threshold 1.2x
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="proteomics_de_analysis_expert",
    display_name="Proteomics DE Analysis Expert",
    description="Proteomics differential expression: statistical testing, time course, correlation analysis",
    factory_function="lobster.agents.proteomics.de_analysis_expert.de_analysis_expert",
    handoff_tool_name=None,
    handoff_tool_description=None,
    supervisor_accessible=False,
    tier_requirement="free",
)

# === Heavy imports below ===
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.proteomics.config import detect_platform_type, get_platform_config
from lobster.agents.proteomics.prompts import create_de_analysis_expert_prompt
from lobster.agents.proteomics.state import DEAnalysisExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.proteomics_differential_service import (
    ProteomicsDifferentialService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class DEAnalysisError(Exception):
    """Base exception for proteomics DE analysis operations."""

    pass


class ModalityNotFoundError(DEAnalysisError):
    """Raised when requested modality doesn't exist."""

    pass


def de_analysis_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "proteomics_de_analysis_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for proteomics differential expression analysis sub-agent.

    This agent handles DE analysis for proteomics data with platform-aware defaults,
    time course analysis for longitudinal studies, and correlation analysis with
    continuous clinical variables.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: List of delegation tools from parent agent
        workspace_path: Optional workspace path for LLM operations
        provider_override: Optional LLM provider override
        model_override: Optional model override

    Returns:
        Configured ReAct agent with proteomics DE analysis capabilities
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("proteomics_de_analysis_expert")
    llm = create_llm(
        "proteomics_de_analysis_expert",
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

    # Initialize stateless services
    differential_service = ProteomicsDifferentialService()

    # Lazy imports for downstream analysis services (inside factory, not module level)
    from lobster.services.analysis.proteomics_kinase_service import (
        ProteomicsKinaseService,
    )
    from lobster.services.analysis.proteomics_pathway_service import (
        ProteomicsPathwayService,
    )
    from lobster.services.analysis.proteomics_string_service import (
        ProteomicsStringService,
    )

    pathway_service = ProteomicsPathwayService()
    kinase_service = ProteomicsKinaseService()
    string_service = ProteomicsStringService()

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def _get_platform_for_modality(modality_name: str) -> str:
        """
        Detect platform type for a given modality from the data manager.

        Args:
            modality_name: Name of the modality to check

        Returns:
            Platform type string: "mass_spec" or "affinity"
        """
        try:
            adata = data_manager.get_modality(modality_name)
            return detect_platform_type(adata)
        except Exception:
            return "mass_spec"  # Default to mass spec if detection fails

    # =========================================================================
    # TOOL 1: find_differential_proteins
    # =========================================================================

    @tool
    def find_differential_proteins(
        modality_name: str,
        group_column: str,
        platform_type: str = "auto",
        method: str = None,
        fdr_threshold: float = 0.05,
        fold_change_threshold: float = None,
    ) -> str:
        """
        Find differentially expressed proteins between groups with platform-aware defaults.

        Auto-detects platform type (mass spectrometry vs affinity) and applies
        appropriate statistical method and fold change thresholds. Performs all
        pairwise group comparisons with FDR correction.

        Args:
            modality_name: Name of the proteomics modality to analyze
            group_column: Column in obs containing group labels for comparison
            platform_type: Platform type override ("mass_spec", "affinity", or "auto" for detection)
            method: Statistical test method (default: platform-dependent). Options: t_test, welch_t_test, mann_whitney, limma_like
            fdr_threshold: FDR threshold for significance (default: 0.05)
            fold_change_threshold: Minimum fold change threshold (default: platform-dependent, 1.5 for MS, 1.2 for affinity)

        Returns:
            str: Formatted results with platform info, parameters, significant proteins
        """
        try:
            # Validate modality existence
            if modality_name not in data_manager.list_modalities():
                return (
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            adata_copy = adata.copy()

            # BUG-02 FIX: Initialize min_group before conditional block
            min_group = None

            # M5 FIX: Sample size warning
            if group_column in adata_copy.obs.columns:
                group_counts = adata_copy.obs[group_column].value_counts()
                min_group = group_counts.min()
                if min_group < 3:
                    return (
                        f"Insufficient samples for DE analysis: smallest group has {min_group} samples. "
                        f"Minimum 3 per group required for meaningful statistical testing. "
                        f"Group sizes: {group_counts.to_dict()}"
                    )

            # Detect or use forced platform type
            if platform_type == "auto":
                detected_platform = _get_platform_for_modality(modality_name)
            else:
                detected_platform = platform_type

            platform_config = get_platform_config(detected_platform)

            # Apply platform-aware defaults
            if method is None:
                method = "limma_like" if detected_platform == "mass_spec" else "t_test"

            if fold_change_threshold is None:
                fold_change_threshold = platform_config.default_fold_change_threshold

            logger.info(
                f"Running differential expression on '{modality_name}' "
                f"(platform: {detected_platform}, method: {method}, "
                f"FC threshold: {fold_change_threshold})"
            )

            # Call the stateless differential service
            adata_de, de_stats, de_ir = (
                differential_service.perform_differential_expression(
                    adata_copy,
                    group_column=group_column,
                    comparison_pairs=None,  # All pairwise comparisons
                    test_method=method,
                    fdr_method="benjamini_hochberg",
                    fdr_threshold=fdr_threshold,
                    fold_change_threshold=fold_change_threshold,
                )
            )

            # Store result as new modality
            de_modality_name = f"{modality_name}_de_analysis"
            data_manager.store_modality(
                name=de_modality_name,
                adata=adata_de,
                parent_name=modality_name,
                step_summary=(
                    f"DE analysis ({detected_platform}): "
                    f"{de_stats.get('n_significant_proteins', 0)} significant proteins"
                ),
            )

            # Log tool usage with IR for provenance
            data_manager.log_tool_usage(
                tool_name="find_differential_proteins",
                parameters={
                    "modality_name": modality_name,
                    "group_column": group_column,
                    "platform_type": detected_platform,
                    "method": method,
                    "fdr_threshold": fdr_threshold,
                    "fold_change_threshold": fold_change_threshold,
                },
                description=(
                    f"Proteomics DE analysis: {de_stats.get('n_significant_proteins', 0)} "
                    f"significant proteins ({detected_platform})"
                ),
                ir=de_ir,
            )

            # Format response
            response = "## Differential Protein Expression Analysis Complete\n\n"
            response += f"**Modality**: '{modality_name}'\n"
            response += f"**Platform**: {platform_config.display_name} ({detected_platform})\n\n"

            response += "**Parameters:**\n"
            response += f"- Group column: {group_column}\n"
            response += f"- Statistical method: {method}\n"
            response += f"- FDR threshold: {fdr_threshold}\n"
            response += f"- Fold change threshold: {fold_change_threshold}x\n"
            response += f"- Comparisons: {de_stats.get('n_comparisons', 0)}\n\n"

            response += "**Results:**\n"
            response += f"- Samples processed: {de_stats.get('samples_processed', 0)}\n"
            response += f"- Proteins tested: {de_stats.get('proteins_processed', 0)}\n"
            response += (
                f"- Total tests performed: {de_stats.get('total_tests_performed', 0)}\n"
            )
            response += (
                f"- Significant proteins: {de_stats.get('n_significant_proteins', 0)}\n"
            )
            response += f"- Significance rate: {de_stats.get('overall_significance_rate', 0):.1%}\n\n"

            # Top significant proteins
            top_up = de_stats.get("top_upregulated", [])
            top_down = de_stats.get("top_downregulated", [])

            if top_up:
                response += "**Top Upregulated Proteins:**\n"
                for protein_info in top_up[:5]:
                    name = protein_info.get("protein", "Unknown")
                    log2fc = protein_info.get("log2_fold_change", 0)
                    padj = protein_info.get("p_adjusted", 1.0)
                    comparison = protein_info.get("comparison", "")
                    response += f"- {name}: log2FC={log2fc:.2f}, FDR={padj:.2e} ({comparison})\n"

            if top_down:
                response += "\n**Top Downregulated Proteins:**\n"
                for protein_info in top_down[:5]:
                    name = protein_info.get("protein", "Unknown")
                    log2fc = protein_info.get("log2_fold_change", 0)
                    padj = protein_info.get("p_adjusted", 1.0)
                    comparison = protein_info.get("comparison", "")
                    response += f"- {name}: log2FC={log2fc:.2f}, FDR={padj:.2e} ({comparison})\n"

            response += f"\n**New modality created**: '{de_modality_name}'"
            response += (
                "\n**Detailed results stored in**: adata.uns['differential_expression']"
            )
            response += "\n**Volcano plot data**: adata.uns['volcano_plot_data']"

            # Sample size warning
            if min_group is not None and min_group < 6:
                response += (
                    f"\n\n**Statistical power warning:** Smallest group has {min_group} samples. "
                    f"With < 6 samples per group, statistical power is very limited "
                    f"and only large effect sizes can be detected. "
                    f"Consider increasing sample size for robust DE analysis."
                )

            return response

        except Exception as e:
            logger.error(f"Error in differential protein expression analysis: {e}")
            return f"Error in differential expression analysis: {str(e)}"

    # =========================================================================
    # TOOL 2: run_time_course_analysis
    # =========================================================================

    @tool
    def run_time_course_analysis(
        modality_name: str,
        time_column: str,
        group_column: str = None,
        method: str = "linear_trend",
        fdr_threshold: float = 0.05,
    ) -> str:
        """
        Run time course differential expression analysis on proteomics data.

        Identifies proteins with significant temporal expression changes using
        linear or polynomial trend analysis. Optionally stratifies by group
        for separate time course analysis per condition.

        Args:
            modality_name: Name of the proteomics modality to analyze
            time_column: Column in obs containing time points (numeric)
            group_column: Optional grouping column for separate time course analysis per condition
            method: Time course analysis method ("linear_trend" or "polynomial")
            fdr_threshold: FDR threshold for significance (default: 0.05)

        Returns:
            str: Formatted results with time points, significant time-dependent proteins
        """
        try:
            # Validate modality existence
            if modality_name not in data_manager.list_modalities():
                return (
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            adata_copy = adata.copy()

            # Validate time column exists
            if time_column not in adata_copy.obs.columns:
                return (
                    f"Time column '{time_column}' not found in obs. "
                    f"Available columns: {list(adata_copy.obs.columns)}"
                )

            # Validate group column if provided
            if group_column is not None and group_column not in adata_copy.obs.columns:
                return (
                    f"Group column '{group_column}' not found in obs. "
                    f"Available columns: {list(adata_copy.obs.columns)}"
                )

            logger.info(
                f"Running time course analysis on '{modality_name}' "
                f"(time_column: {time_column}, method: {method})"
            )

            # Call the stateless differential service
            adata_tc, tc_stats, tc_ir = (
                differential_service.perform_time_course_analysis(
                    adata_copy,
                    time_column=time_column,
                    group_column=group_column,
                    test_method=method,
                    fdr_threshold=fdr_threshold,
                )
            )

            # Store result as new modality
            tc_modality_name = f"{modality_name}_time_course"
            data_manager.store_modality(
                name=tc_modality_name,
                adata=adata_tc,
                parent_name=modality_name,
                step_summary=(
                    f"Time course analysis: "
                    f"{tc_stats.get('n_significant_results', 0)} significant proteins"
                ),
            )

            # Log tool usage with IR for provenance
            data_manager.log_tool_usage(
                tool_name="run_time_course_analysis",
                parameters={
                    "modality_name": modality_name,
                    "time_column": time_column,
                    "group_column": group_column,
                    "method": method,
                    "fdr_threshold": fdr_threshold,
                },
                description=(
                    f"Time course analysis: {tc_stats.get('n_significant_results', 0)} "
                    f"significant time-dependent proteins"
                ),
                ir=tc_ir,
            )

            # Format response
            response = "## Time Course Analysis Complete\n\n"
            response += f"**Modality**: '{modality_name}'\n"
            response += f"**Method**: {method}\n\n"

            response += "**Parameters:**\n"
            response += f"- Time column: {time_column}\n"
            if group_column:
                response += f"- Group column: {group_column}\n"
            response += f"- FDR threshold: {fdr_threshold}\n\n"

            response += "**Data Summary:**\n"
            response += f"- Samples processed: {tc_stats.get('samples_processed', 0)}\n"
            response += (
                f"- Proteins analyzed: {tc_stats.get('proteins_processed', 0)}\n"
            )

            time_range = tc_stats.get("time_range", (0, 0))
            response += f"- Time points: {tc_stats.get('n_time_points', 0)}\n"
            response += f"- Time range: {time_range[0]} to {time_range[1]}\n\n"

            response += "**Results:**\n"
            response += f"- Tests performed: {tc_stats.get('n_tests_performed', 0)}\n"
            response += f"- Significant time-dependent proteins: {tc_stats.get('n_significant_results', 0)}\n"
            response += (
                f"- Significance rate: {tc_stats.get('significance_rate', 0):.1%}\n"
            )

            # Show significant results from uns
            tc_results = adata_tc.uns.get("time_course_analysis", {})
            significant_results = tc_results.get("significant_results", [])
            if significant_results:
                response += "\n**Top Time-Dependent Proteins:**\n"
                # Sort by p_adjusted
                sorted_results = sorted(
                    significant_results, key=lambda x: x.get("p_adjusted", 1.0)
                )
                for result in sorted_results[:10]:
                    name = result.get("protein", "Unknown")
                    padj = result.get("p_adjusted", 1.0)
                    r2 = result.get("r_squared", 0.0)
                    group = result.get("group", "all")
                    group_str = f" [{group}]" if group != "all" else ""
                    response += f"- {name}: FDR={padj:.2e}, R2={r2:.3f}{group_str}\n"

            response += f"\n**New modality created**: '{tc_modality_name}'"
            response += (
                "\n**Detailed results stored in**: adata.uns['time_course_analysis']"
            )

            return response

        except Exception as e:
            logger.error(f"Error in time course analysis: {e}")
            return f"Error in time course analysis: {str(e)}"

    # =========================================================================
    # TOOL 3: run_correlation_analysis
    # =========================================================================

    @tool
    def run_correlation_analysis(
        modality_name: str,
        target_column: str,
        method: str = "pearson",
        correlation_threshold: float = 0.3,
        fdr_threshold: float = 0.05,
    ) -> str:
        """
        Run correlation analysis between protein expression and a continuous variable.

        Calculates per-protein correlation with a clinical or phenotypic variable
        (e.g., age, BMI, biomarker level) with FDR correction. Identifies proteins
        whose expression significantly correlates with the target variable.

        Args:
            modality_name: Name of the proteomics modality to analyze
            target_column: Column in obs containing continuous target variable
            method: Correlation method ("pearson", "spearman", or "kendall")
            correlation_threshold: Minimum absolute correlation for significance (default: 0.3)
            fdr_threshold: FDR threshold for significance (default: 0.05)

        Returns:
            str: Formatted results with correlation statistics and significant proteins
        """
        try:
            # Validate modality existence
            if modality_name not in data_manager.list_modalities():
                return (
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            adata_copy = adata.copy()

            # Validate target column exists
            if target_column not in adata_copy.obs.columns:
                return (
                    f"Target column '{target_column}' not found in obs. "
                    f"Available columns: {list(adata_copy.obs.columns)}"
                )

            logger.info(
                f"Running correlation analysis on '{modality_name}' "
                f"(target: {target_column}, method: {method})"
            )

            # Call the stateless differential service
            adata_corr, corr_stats, corr_ir = (
                differential_service.perform_correlation_analysis(
                    adata_copy,
                    target_column=target_column,
                    correlation_method=method,
                    fdr_threshold=fdr_threshold,
                    min_correlation=correlation_threshold,
                )
            )

            # Store result as new modality
            corr_modality_name = f"{modality_name}_correlation"
            data_manager.store_modality(
                name=corr_modality_name,
                adata=adata_corr,
                parent_name=modality_name,
                step_summary=(
                    f"Correlation analysis: "
                    f"{corr_stats.get('n_significant_results', 0)} significant correlations"
                ),
            )

            # Log tool usage with IR for provenance
            data_manager.log_tool_usage(
                tool_name="run_correlation_analysis",
                parameters={
                    "modality_name": modality_name,
                    "target_column": target_column,
                    "method": method,
                    "correlation_threshold": correlation_threshold,
                    "fdr_threshold": fdr_threshold,
                },
                description=(
                    f"Correlation analysis: {corr_stats.get('n_significant_results', 0)} "
                    f"significant correlations with {target_column}"
                ),
                ir=corr_ir,
            )

            # Format response
            response = "## Correlation Analysis Complete\n\n"
            response += f"**Modality**: '{modality_name}'\n"
            response += f"**Target variable**: {target_column}\n\n"

            response += "**Parameters:**\n"
            response += f"- Correlation method: {method}\n"
            response += f"- Minimum |correlation|: {correlation_threshold}\n"
            response += f"- FDR threshold: {fdr_threshold}\n\n"

            response += "**Data Summary:**\n"
            response += (
                f"- Samples processed: {corr_stats.get('samples_processed', 0)}\n"
            )
            response += (
                f"- Proteins analyzed: {corr_stats.get('proteins_processed', 0)}\n"
            )

            target_range = corr_stats.get("target_range", (0, 0))
            response += (
                f"- Target range: {target_range[0]:.2f} to {target_range[1]:.2f}\n\n"
            )

            response += "**Results:**\n"
            response += f"- Tests performed: {corr_stats.get('n_tests_performed', 0)}\n"
            response += f"- Significant correlations: {corr_stats.get('n_significant_results', 0)}\n"
            response += (
                f"- Significance rate: {corr_stats.get('significance_rate', 0):.1%}\n"
            )
            response += f"- Median |correlation|: {corr_stats.get('median_abs_correlation', 0):.3f}\n"
            response += (
                f"- Max |correlation|: {corr_stats.get('max_abs_correlation', 0):.3f}\n"
            )

            # Show significant results from uns
            corr_results = adata_corr.uns.get("correlation_analysis", {})
            significant_results = corr_results.get("significant_results", [])
            if significant_results:
                # Split into positive and negative correlations
                positive = [
                    r for r in significant_results if r.get("correlation", 0) > 0
                ]
                negative = [
                    r for r in significant_results if r.get("correlation", 0) < 0
                ]

                if positive:
                    response += "\n**Top Positive Correlations:**\n"
                    sorted_pos = sorted(
                        positive,
                        key=lambda x: abs(x.get("correlation", 0)),
                        reverse=True,
                    )
                    for result in sorted_pos[:5]:
                        name = result.get("protein", "Unknown")
                        corr = result.get("correlation", 0)
                        padj = result.get("p_adjusted", 1.0)
                        response += f"- {name}: r={corr:.3f}, FDR={padj:.2e}\n"

                if negative:
                    response += "\n**Top Negative Correlations:**\n"
                    sorted_neg = sorted(
                        negative,
                        key=lambda x: abs(x.get("correlation", 0)),
                        reverse=True,
                    )
                    for result in sorted_neg[:5]:
                        name = result.get("protein", "Unknown")
                        corr = result.get("correlation", 0)
                        padj = result.get("p_adjusted", 1.0)
                        response += f"- {name}: r={corr:.3f}, FDR={padj:.2e}\n"

            response += f"\n**New modality created**: '{corr_modality_name}'"
            response += (
                "\n**Detailed results stored in**: adata.uns['correlation_analysis']"
            )
            response += (
                "\n**Per-protein correlations**: adata.var['correlation_with_target']"
            )

            return response

        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return f"Error in correlation analysis: {str(e)}"

    # =========================================================================
    # TOOL 4: run_pathway_enrichment
    # =========================================================================

    @tool
    def run_pathway_enrichment(
        modality_name: str,
        databases: str = "go_reactome",
        fdr_threshold: float = 0.05,
        max_genes: int = 500,
    ) -> str:
        """
        Run pathway enrichment on proteomics differential expression results.

        Extracts significant protein gene symbols from DE results and performs
        Over-Representation Analysis (ORA) via Enrichr API. Supports GO, KEGG,
        and Reactome databases.

        Args:
            modality_name: Name of the proteomics modality with DE results
            databases: Database shorthand: "go", "reactome", "kegg", "go_reactome", or "go_reactome_kegg"
            fdr_threshold: FDR threshold for enrichment significance (default: 0.05)
            max_genes: Maximum number of genes to include in enrichment (default: 500)

        Returns:
            str: Formatted results with enriched pathways, term counts, databases queried
        """
        try:
            # Validate modality existence
            if modality_name not in data_manager.list_modalities():
                return (
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            # Validate DE results exist
            if "differential_expression" not in adata.uns:
                return (
                    f"No differential expression results found in '{modality_name}'. "
                    f"Run find_differential_proteins first."
                )

            logger.info(
                f"Running pathway enrichment on '{modality_name}' "
                f"(databases: {databases}, FDR: {fdr_threshold})"
            )

            # Call the pathway service
            adata_enriched, enrich_stats, enrich_ir = pathway_service.run_enrichment(
                adata,
                databases=databases,
                fdr_threshold=fdr_threshold,
                max_genes=max_genes,
            )

            # Store result as new modality
            enriched_name = f"{modality_name}_enriched"
            data_manager.store_modality(
                name=enriched_name,
                adata=adata_enriched,
                parent_name=modality_name,
                step_summary=(
                    f"Pathway enrichment: "
                    f"{enrich_stats.get('n_significant_terms', 0)} significant terms"
                ),
            )

            # Log tool usage with IR
            data_manager.log_tool_usage(
                tool_name="run_pathway_enrichment",
                parameters={
                    "modality_name": modality_name,
                    "databases": databases,
                    "fdr_threshold": fdr_threshold,
                    "max_genes": max_genes,
                },
                description=(
                    f"Pathway enrichment: {enrich_stats.get('n_significant_terms', 0)} "
                    f"significant terms from {enrich_stats.get('n_genes_input', 0)} proteins"
                ),
                ir=enrich_ir,
            )

            # Format response
            response = "## Pathway Enrichment Analysis Complete\n\n"
            response += f"**Modality**: '{modality_name}'\n"
            response += f"**Databases**: {', '.join(enrich_stats.get('databases_queried', []))}\n\n"

            response += "**Summary:**\n"
            response += f"- Proteins analyzed: {enrich_stats.get('n_genes_input', 0)}\n"
            response += f"- Total terms found: {enrich_stats.get('n_total_terms', 0)}\n"
            response += f"- Significant terms (FDR < {fdr_threshold}): {enrich_stats.get('n_significant_terms', 0)}\n\n"

            # Top enriched terms
            top_terms = enrich_stats.get("top_terms", [])
            if top_terms:
                response += "**Top Enriched Terms:**\n"
                for term in top_terms:
                    response += (
                        f"- {term['term']}: "
                        f"FDR={term['p_value']:.2e}, "
                        f"overlap={term['overlap']} "
                        f"[{term['database']}]\n"
                    )

            response += f"\n**New modality created**: '{enriched_name}'"
            response += (
                "\n**Detailed results stored in**: adata.uns['pathway_enrichment']"
            )

            return response

        except Exception as e:
            logger.error(f"Error in pathway enrichment: {e}")
            return f"Error in pathway enrichment: {str(e)}"

    # =========================================================================
    # TOOL 5: run_differential_ptm_analysis
    # =========================================================================

    @tool
    def run_differential_ptm_analysis(
        modality_name: str,
        protein_modality_name: str,
        group_column: str,
        fdr_threshold: float = 0.05,
    ) -> str:
        """
        Run differential PTM analysis comparing site-level vs protein-level changes.

        Performs DE on both PTM site modality and protein modality, then adjusts
        site-level fold changes by subtracting the corresponding protein-level
        fold change. Identifies PTM sites with changes beyond what protein abundance
        explains.

        Args:
            modality_name: Name of the PTM site-level modality (e.g., phosphoproteomics)
            protein_modality_name: Name of the protein-level modality for normalization
            group_column: Column in obs containing group labels for comparison
            fdr_threshold: FDR threshold for significance (default: 0.05)

        Returns:
            str: Formatted results with raw vs adjusted fold changes, discordant sites
        """
        try:
            # Validate both modalities exist
            available = data_manager.list_modalities()
            if modality_name not in available:
                return (
                    f"PTM modality '{modality_name}' not found. "
                    f"Available: {available}"
                )
            if protein_modality_name not in available:
                return (
                    f"Protein modality '{protein_modality_name}' not found. "
                    f"Available: {available}"
                )

            ptm_adata = data_manager.get_modality(modality_name)
            protein_adata = data_manager.get_modality(protein_modality_name)

            logger.info(
                f"Running differential PTM analysis: PTM='{modality_name}', "
                f"Protein='{protein_modality_name}', group='{group_column}'"
            )

            # Run DE on PTM site modality
            ptm_de, ptm_stats, ptm_ir = (
                differential_service.perform_differential_expression(
                    ptm_adata.copy(),
                    group_column=group_column,
                    comparison_pairs=None,
                    test_method="limma_like",
                    fdr_method="benjamini_hochberg",
                    fdr_threshold=fdr_threshold,
                )
            )

            # Run DE on protein modality
            prot_de, prot_stats, prot_ir = (
                differential_service.perform_differential_expression(
                    protein_adata.copy(),
                    group_column=group_column,
                    comparison_pairs=None,
                    test_method="limma_like",
                    fdr_method="benjamini_hochberg",
                    fdr_threshold=fdr_threshold,
                )
            )

            # Build protein-level FC lookup
            prot_fc_lookup = {}
            prot_de_data = prot_de.uns.get("differential_expression", {})
            for result in prot_de_data.get("all_results", []):
                protein = result.get("protein", "")
                log2fc = result.get("log2_fold_change", 0.0)
                comparison = result.get("comparison", "")
                key = f"{protein}_{comparison}"
                prot_fc_lookup[key] = log2fc

            # Compute adjusted site fold changes
            ptm_de_data = ptm_de.uns.get("differential_expression", {})
            significant_sites = []
            all_adjusted = []

            for result in ptm_de_data.get("significant_results", []):
                site_id = result.get("protein", "")
                site_log2fc = result.get("log2_fold_change", 0.0)
                comparison = result.get("comparison", "")

                # Extract gene name prefix from site ID (e.g., EGFR_Y1068 -> EGFR)
                gene_name = site_id.split("_")[0] if "_" in site_id else site_id

                # Look up protein-level FC
                prot_key = f"{gene_name}_{comparison}"
                prot_log2fc = prot_fc_lookup.get(prot_key, 0.0)

                # Adjusted FC = site FC - protein FC
                adjusted_log2fc = site_log2fc - prot_log2fc

                site_entry = {
                    "site": site_id,
                    "gene": gene_name,
                    "comparison": comparison,
                    "log2_fold_change": site_log2fc,
                    "protein_log2fc": prot_log2fc,
                    "adjusted_log2fc": adjusted_log2fc,
                    "p_adjusted": result.get("p_adjusted", 1.0),
                    "discordant": (site_log2fc > 0) != (prot_log2fc > 0),
                }
                significant_sites.append(site_entry)
                all_adjusted.append(adjusted_log2fc)

            # Store results in AnnData
            ptm_de.uns["differential_ptm"] = {
                "significant_sites": significant_sites,
                "n_sites_analyzed": len(ptm_de_data.get("all_results", [])),
                "n_significant_sites": len(significant_sites),
                "parameters": {
                    "ptm_modality": modality_name,
                    "protein_modality": protein_modality_name,
                    "group_column": group_column,
                    "fdr_threshold": fdr_threshold,
                },
            }

            # Store as new modality
            ptm_de_name = f"{modality_name}_differential_ptm"
            data_manager.store_modality(
                name=ptm_de_name,
                adata=ptm_de,
                parent_name=modality_name,
                step_summary=(
                    f"Differential PTM: {len(significant_sites)} significant sites"
                ),
            )

            # Log tool usage
            data_manager.log_tool_usage(
                tool_name="run_differential_ptm_analysis",
                parameters={
                    "modality_name": modality_name,
                    "protein_modality_name": protein_modality_name,
                    "group_column": group_column,
                    "fdr_threshold": fdr_threshold,
                },
                description=(
                    f"Differential PTM analysis: {len(significant_sites)} significant sites"
                ),
                ir=ptm_ir,
            )

            # Format response
            response = "## Differential PTM Analysis Complete\n\n"
            response += f"**PTM Modality**: '{modality_name}'\n"
            response += f"**Protein Modality**: '{protein_modality_name}'\n\n"

            response += "**Summary:**\n"
            response += (
                f"- PTM sites tested: {ptm_stats.get('proteins_processed', 0)}\n"
            )
            response += f"- Significant PTM sites: {len(significant_sites)}\n"
            response += (
                f"- Proteins tested: {prot_stats.get('proteins_processed', 0)}\n"
            )
            n_discordant = sum(1 for s in significant_sites if s["discordant"])
            response += (
                f"- Discordant sites (PTM vs protein direction): {n_discordant}\n\n"
            )

            # Top sites by adjusted FC
            if significant_sites:
                sorted_sites = sorted(
                    significant_sites,
                    key=lambda x: abs(x["adjusted_log2fc"]),
                    reverse=True,
                )
                response += "**Top Sites by Adjusted Fold Change:**\n"
                for site in sorted_sites[:10]:
                    direction = "UP" if site["adjusted_log2fc"] > 0 else "DOWN"
                    disc_flag = " [DISCORDANT]" if site["discordant"] else ""
                    response += (
                        f"- {site['site']}: "
                        f"raw={site['log2_fold_change']:.2f}, "
                        f"protein={site['protein_log2fc']:.2f}, "
                        f"adjusted={site['adjusted_log2fc']:.2f} ({direction})"
                        f"{disc_flag}\n"
                    )

            response += f"\n**New modality created**: '{ptm_de_name}'"
            response += (
                "\n**Detailed results stored in**: adata.uns['differential_ptm']"
            )

            return response

        except Exception as e:
            logger.error(f"Error in differential PTM analysis: {e}")
            return f"Error in differential PTM analysis: {str(e)}"

    # =========================================================================
    # TOOL 6: run_kinase_enrichment
    # =========================================================================

    @tool
    def run_kinase_enrichment(
        modality_name: str,
        custom_mapping_path: str = None,
        min_substrates: int = 3,
        fdr_threshold: float = 0.05,
    ) -> str:
        """
        Run Kinase-Substrate Enrichment Analysis (KSEA) to infer kinase activity.

        Computes KSEA z-scores from phosphosite fold changes using a built-in
        kinase-substrate mapping (~20 kinases) or a custom CSV mapping file.
        Identifies activated and inhibited kinases from phosphoproteomics data.

        Args:
            modality_name: Name of the modality with DE or PTM DE results
            custom_mapping_path: Optional path to custom CSV with columns: kinase, substrate_site
            min_substrates: Minimum matched substrates required per kinase (default: 3)
            fdr_threshold: FDR threshold for kinase significance (default: 0.05)

        Returns:
            str: Formatted results with significant kinases, z-scores, activity direction
        """
        try:
            # Validate modality existence
            if modality_name not in data_manager.list_modalities():
                return (
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            # Validate DE or PTM DE results exist
            has_de = "differential_expression" in adata.uns
            has_ptm = "differential_ptm" in adata.uns
            has_var_fc = (
                "log2_fold_change" in adata.var.columns
                if hasattr(adata.var, "columns")
                else False
            )

            if not (has_de or has_ptm or has_var_fc):
                return (
                    f"No fold change data found in '{modality_name}'. "
                    f"Run find_differential_proteins or run_differential_ptm_analysis first."
                )

            logger.info(
                f"Running KSEA on '{modality_name}' "
                f"(min_substrates: {min_substrates}, FDR: {fdr_threshold})"
            )

            # Call the kinase service
            adata_ksea, ksea_stats, ksea_ir = kinase_service.compute_ksea(
                adata,
                custom_mapping_path=custom_mapping_path,
                min_substrates=min_substrates,
                fdr_threshold=fdr_threshold,
            )

            # Store result as new modality
            ksea_name = f"{modality_name}_ksea"
            data_manager.store_modality(
                name=ksea_name,
                adata=adata_ksea,
                parent_name=modality_name,
                step_summary=(
                    f"KSEA: {ksea_stats.get('n_significant', 0)} significant kinases"
                ),
            )

            # Log tool usage
            data_manager.log_tool_usage(
                tool_name="run_kinase_enrichment",
                parameters={
                    "modality_name": modality_name,
                    "custom_mapping_path": custom_mapping_path,
                    "min_substrates": min_substrates,
                    "fdr_threshold": fdr_threshold,
                },
                description=(
                    f"KSEA: {ksea_stats.get('n_significant', 0)} significant kinases "
                    f"from {ksea_stats.get('n_kinases_tested', 0)} tested"
                ),
                ir=ksea_ir,
            )

            # Format response
            response = "## Kinase-Substrate Enrichment Analysis (KSEA) Complete\n\n"
            response += f"**Modality**: '{modality_name}'\n\n"

            response += "**Summary:**\n"
            response += (
                f"- Sites with fold changes: {ksea_stats.get('n_sites_available', 0)}\n"
            )
            response += (
                f"- Kinases in mapping: {ksea_stats.get('n_kinases_in_map', 0)}\n"
            )
            response += f"- Kinases tested (>= {min_substrates} substrates): {ksea_stats.get('n_kinases_tested', 0)}\n"
            response += f"- Significant kinases (FDR < {fdr_threshold}): {ksea_stats.get('n_significant', 0)}\n\n"

            # Top kinases
            top_kinases = ksea_stats.get("top_kinases", [])
            if top_kinases:
                response += "**Top Kinases by Activity (|z-score|):**\n"
                for k in top_kinases:
                    direction = "ACTIVATED" if k["z_score"] > 0 else "INHIBITED"
                    sig_marker = " *" if k.get("fdr", 1.0) < fdr_threshold else ""
                    response += (
                        f"- {k['kinase']}: z={k['z_score']:.2f}, "
                        f"substrates={k['n_substrates']}, "
                        f"FDR={k.get('fdr', 1.0):.2e} ({direction}){sig_marker}\n"
                    )
                response += "\n*= significant at FDR threshold\n"

            response += f"\n**New modality created**: '{ksea_name}'"
            response += "\n**Detailed results stored in**: adata.uns['ksea_results']"

            return response

        except Exception as e:
            logger.error(f"Error in kinase enrichment: {e}")
            return f"Error in kinase enrichment: {str(e)}"

    # =========================================================================
    # TOOL 7: run_string_network_analysis
    # =========================================================================

    @tool
    def run_string_network_analysis(
        modality_name: str,
        species: int = 9606,
        score_threshold: int = 400,
        network_type: str = "functional",
    ) -> str:
        """
        Query STRING database for protein-protein interaction network from DE results.

        Extracts significant proteins from DE results and queries the STRING
        REST API for known and predicted protein interactions. Computes
        network topology metrics (density, hub proteins) when networkx is available.

        Args:
            modality_name: Name of the modality with DE results
            species: NCBI taxonomy ID (default: 9606 for human, 10090 for mouse)
            score_threshold: Minimum combined score 0-1000 (400=medium, 700=high, 900=highest)
            network_type: Evidence type: "functional" (all evidence) or "physical" (binding only)

        Returns:
            str: Formatted results with network statistics, hub proteins, interaction count
        """
        try:
            # Validate modality existence
            if modality_name not in data_manager.list_modalities():
                return (
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            # Validate DE results exist
            if "differential_expression" not in adata.uns:
                return (
                    f"No differential expression results found in '{modality_name}'. "
                    f"Run find_differential_proteins first."
                )

            logger.info(
                f"Running STRING network analysis on '{modality_name}' "
                f"(species: {species}, score >= {score_threshold})"
            )

            # Call the STRING service
            adata_net, net_stats, net_ir = string_service.query_network(
                adata,
                species=species,
                score_threshold=score_threshold,
                network_type=network_type,
            )

            # Store result as new modality
            network_name = f"{modality_name}_network"
            data_manager.store_modality(
                name=network_name,
                adata=adata_net,
                parent_name=modality_name,
                step_summary=(
                    f"STRING network: {net_stats.get('n_interactions_found', 0)} interactions"
                ),
            )

            # Log tool usage
            data_manager.log_tool_usage(
                tool_name="run_string_network_analysis",
                parameters={
                    "modality_name": modality_name,
                    "species": species,
                    "score_threshold": score_threshold,
                    "network_type": network_type,
                },
                description=(
                    f"STRING PPI network: {net_stats.get('n_interactions_found', 0)} "
                    f"interactions, {net_stats.get('n_hub_proteins', 0)} hub proteins"
                ),
                ir=net_ir,
            )

            # Format response
            response = "## STRING PPI Network Analysis Complete\n\n"
            response += f"**Modality**: '{modality_name}'\n\n"

            response += "**Query Parameters:**\n"
            response += (
                f"- Species: {species} ({'human' if species == 9606 else 'other'})\n"
            )
            response += f"- Score threshold: {score_threshold}\n"
            response += f"- Network type: {network_type}\n\n"

            response += "**Network Statistics:**\n"
            response += (
                f"- Proteins queried: {net_stats.get('n_proteins_queried', 0)}\n"
            )
            response += (
                f"- Nodes in network: {net_stats.get('n_nodes_in_network', 0)}\n"
            )
            response += (
                f"- Interactions found: {net_stats.get('n_interactions_found', 0)}\n"
            )
            response += (
                f"- Network density: {net_stats.get('network_density', 0):.4f}\n\n"
            )

            # Hub proteins
            network_data = adata_net.uns.get("string_network", {})
            hub_proteins = network_data.get("hub_proteins", [])
            if hub_proteins:
                response += "**Hub Proteins (top by degree):**\n"
                for hub in hub_proteins[:10]:
                    response += f"- {hub['protein']}: degree={hub['degree']}\n"

            response += f"\n**New modality created**: '{network_name}'"
            response += "\n**Detailed results stored in**: adata.uns['string_network']"

            return response

        except Exception as e:
            logger.error(f"Error in STRING network analysis: {e}")
            return f"Error in STRING network analysis: {str(e)}"

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    tools = [
        find_differential_proteins,
        run_time_course_analysis,
        run_correlation_analysis,
        run_pathway_enrichment,
        run_differential_ptm_analysis,
        run_kinase_enrichment,
        run_string_network_analysis,
    ]

    if delegation_tools:
        tools += delegation_tools

    # =========================================================================
    # CREATE AGENT
    # =========================================================================

    system_prompt = create_de_analysis_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=DEAnalysisExpertState,
    )
