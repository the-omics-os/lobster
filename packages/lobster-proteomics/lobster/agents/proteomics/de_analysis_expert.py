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
    name="de_analysis_expert",
    display_name="DE Analysis Expert",
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
    agent_name: str = "de_analysis_expert",
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
    model_params = settings.get_agent_llm_params("de_analysis_expert")
    llm = create_llm(
        "de_analysis_expert",
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

    # Initialize stateless service
    differential_service = ProteomicsDifferentialService()

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
                method = (
                    "limma_like" if detected_platform == "mass_spec" else "t_test"
                )

            if fold_change_threshold is None:
                fold_change_threshold = platform_config.default_fold_change_threshold

            logger.info(
                f"Running differential expression on '{modality_name}' "
                f"(platform: {detected_platform}, method: {method}, "
                f"FC threshold: {fold_change_threshold})"
            )

            # Call the stateless differential service
            adata_de, de_stats, de_ir = differential_service.perform_differential_expression(
                adata_copy,
                group_column=group_column,
                comparison_pairs=None,  # All pairwise comparisons
                test_method=method,
                fdr_method="benjamini_hochberg",
                fdr_threshold=fdr_threshold,
                fold_change_threshold=fold_change_threshold,
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
            response = f"## Differential Protein Expression Analysis Complete\n\n"
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
            response += f"- Total tests performed: {de_stats.get('total_tests_performed', 0)}\n"
            response += f"- Significant proteins: {de_stats.get('n_significant_proteins', 0)}\n"
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
            response += f"\n**Detailed results stored in**: adata.uns['differential_expression']"
            response += f"\n**Volcano plot data**: adata.uns['volcano_plot_data']"

            # Sample size warning
            if min_group < 6:
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
            adata_tc, tc_stats, tc_ir = differential_service.perform_time_course_analysis(
                adata_copy,
                time_column=time_column,
                group_column=group_column,
                test_method=method,
                fdr_threshold=fdr_threshold,
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
            response = f"## Time Course Analysis Complete\n\n"
            response += f"**Modality**: '{modality_name}'\n"
            response += f"**Method**: {method}\n\n"

            response += "**Parameters:**\n"
            response += f"- Time column: {time_column}\n"
            if group_column:
                response += f"- Group column: {group_column}\n"
            response += f"- FDR threshold: {fdr_threshold}\n\n"

            response += "**Data Summary:**\n"
            response += f"- Samples processed: {tc_stats.get('samples_processed', 0)}\n"
            response += f"- Proteins analyzed: {tc_stats.get('proteins_processed', 0)}\n"

            time_range = tc_stats.get("time_range", (0, 0))
            response += f"- Time points: {tc_stats.get('n_time_points', 0)}\n"
            response += f"- Time range: {time_range[0]} to {time_range[1]}\n\n"

            response += "**Results:**\n"
            response += f"- Tests performed: {tc_stats.get('n_tests_performed', 0)}\n"
            response += f"- Significant time-dependent proteins: {tc_stats.get('n_significant_results', 0)}\n"
            response += f"- Significance rate: {tc_stats.get('significance_rate', 0):.1%}\n"

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
            response += f"\n**Detailed results stored in**: adata.uns['time_course_analysis']"

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
            adata_corr, corr_stats, corr_ir = differential_service.perform_correlation_analysis(
                adata_copy,
                target_column=target_column,
                correlation_method=method,
                fdr_threshold=fdr_threshold,
                min_correlation=correlation_threshold,
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
            response = f"## Correlation Analysis Complete\n\n"
            response += f"**Modality**: '{modality_name}'\n"
            response += f"**Target variable**: {target_column}\n\n"

            response += "**Parameters:**\n"
            response += f"- Correlation method: {method}\n"
            response += f"- Minimum |correlation|: {correlation_threshold}\n"
            response += f"- FDR threshold: {fdr_threshold}\n\n"

            response += "**Data Summary:**\n"
            response += f"- Samples processed: {corr_stats.get('samples_processed', 0)}\n"
            response += f"- Proteins analyzed: {corr_stats.get('proteins_processed', 0)}\n"

            target_range = corr_stats.get("target_range", (0, 0))
            response += f"- Target range: {target_range[0]:.2f} to {target_range[1]:.2f}\n\n"

            response += "**Results:**\n"
            response += f"- Tests performed: {corr_stats.get('n_tests_performed', 0)}\n"
            response += f"- Significant correlations: {corr_stats.get('n_significant_results', 0)}\n"
            response += f"- Significance rate: {corr_stats.get('significance_rate', 0):.1%}\n"
            response += f"- Median |correlation|: {corr_stats.get('median_abs_correlation', 0):.3f}\n"
            response += f"- Max |correlation|: {corr_stats.get('max_abs_correlation', 0):.3f}\n"

            # Show significant results from uns
            corr_results = adata_corr.uns.get("correlation_analysis", {})
            significant_results = corr_results.get("significant_results", [])
            if significant_results:
                # Split into positive and negative correlations
                positive = [r for r in significant_results if r.get("correlation", 0) > 0]
                negative = [r for r in significant_results if r.get("correlation", 0) < 0]

                if positive:
                    response += "\n**Top Positive Correlations:**\n"
                    sorted_pos = sorted(
                        positive, key=lambda x: abs(x.get("correlation", 0)), reverse=True
                    )
                    for result in sorted_pos[:5]:
                        name = result.get("protein", "Unknown")
                        corr = result.get("correlation", 0)
                        padj = result.get("p_adjusted", 1.0)
                        response += f"- {name}: r={corr:.3f}, FDR={padj:.2e}\n"

                if negative:
                    response += "\n**Top Negative Correlations:**\n"
                    sorted_neg = sorted(
                        negative, key=lambda x: abs(x.get("correlation", 0)), reverse=True
                    )
                    for result in sorted_neg[:5]:
                        name = result.get("protein", "Unknown")
                        corr = result.get("correlation", 0)
                        padj = result.get("p_adjusted", 1.0)
                        response += f"- {name}: r={corr:.3f}, FDR={padj:.2e}\n"

            response += f"\n**New modality created**: '{corr_modality_name}'"
            response += f"\n**Detailed results stored in**: adata.uns['correlation_analysis']"
            response += f"\n**Per-protein correlations**: adata.var['correlation_with_target']"

            return response

        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return f"Error in correlation analysis: {str(e)}"

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    tools = [
        find_differential_proteins,
        run_time_course_analysis,
        run_correlation_analysis,
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
