"""
Biomarker Discovery Expert sub-agent for proteomics network and survival analysis.

This sub-agent wraps WGCNALiteService and ProteomicsSurvivalService with 4 tools:
- identify_coexpression_modules: WGCNA-lite module identification
- correlate_modules_with_traits: Module-trait correlation analysis
- perform_survival_analysis: Cox proportional hazards regression
- find_survival_biomarkers: Batch Kaplan-Meier biomarker screening

The agent is delegated to by the proteomics_expert parent agent for
network analysis and survival-based biomarker discovery tasks.
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="biomarker_discovery_expert",
    display_name="Biomarker Discovery Expert",
    description="Network analysis (WGCNA), survival analysis (Cox, Kaplan-Meier), biomarker identification",
    factory_function="lobster.agents.proteomics.biomarker_discovery_expert.biomarker_discovery_expert",
    handoff_tool_name=None,
    handoff_tool_description=None,
    supervisor_accessible=False,
    tier_requirement="free",
)

# === Heavy imports below ===
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.proteomics.prompts import create_biomarker_discovery_expert_prompt
from lobster.agents.proteomics.state import BiomarkerDiscoveryExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.proteomics_network_service import WGCNALiteService
from lobster.services.analysis.proteomics_survival_service import (
    ProteomicsSurvivalService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class BiomarkerDiscoveryError(Exception):
    """Base exception for biomarker discovery agent operations."""

    pass


def biomarker_discovery_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "biomarker_discovery_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for the biomarker discovery expert sub-agent.

    This agent specializes in network-based and survival-based biomarker
    discovery for proteomics data. It wraps WGCNALiteService for
    co-expression module analysis and ProteomicsSurvivalService for
    Cox regression and Kaplan-Meier survival analysis.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: Optional list of delegation tools from parent agent
        workspace_path: Optional workspace path for LLM operations
        provider_override: Optional LLM provider override
        model_override: Optional model override

    Returns:
        Configured ReAct agent with biomarker discovery capabilities
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("biomarker_discovery_expert")
    llm = create_llm(
        "biomarker_discovery_expert",
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
    network_service = WGCNALiteService()
    survival_service = ProteomicsSurvivalService()

    # =========================================================================
    # TOOL 1: Identify co-expression modules (WGCNA-lite)
    # =========================================================================

    @tool
    def identify_coexpression_modules(
        modality_name: str,
        n_top_variable: int = 5000,
        soft_power: Optional[int] = None,
        min_module_size: int = 20,
        merge_cut_height: float = 0.25,
    ) -> str:
        """
        Identify protein co-expression modules using WGCNA-lite algorithm.

        Constructs a correlation network from the most variable proteins,
        applies hierarchical clustering, and assigns WGCNA-style color labels.

        Args:
            modality_name: Name of the proteomics modality to analyze
            n_top_variable: Number of most variable proteins to use (default: 5000)
            soft_power: Soft thresholding power (None for auto/signed correlation)
            min_module_size: Minimum proteins per module (default: 20)
            merge_cut_height: Eigengene correlation threshold for merging similar modules (default: 0.25)

        Returns:
            str: Module identification results with counts, sizes, and colors
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return (
                f"Modality '{modality_name}' not found. "
                f"Available: {data_manager.list_modalities()}"
            )

        try:
            adata_copy = adata.copy()

            result_adata, stats, ir = network_service.identify_modules(
                adata_copy,
                n_top_variable=n_top_variable,
                soft_power=soft_power,
                min_module_size=min_module_size,
                merge_cut_height=merge_cut_height,
            )

            # Store result as new modality
            result_name = f"{modality_name}_modules"
            data_manager.store_modality(
                name=result_name,
                adata=result_adata,
                parent_name=modality_name,
                step_summary=f"WGCNA module identification: {stats['n_modules']} modules",
            )

            # Log tool usage with provenance
            data_manager.log_tool_usage(
                tool_name="identify_coexpression_modules",
                parameters={
                    "modality_name": modality_name,
                    "n_top_variable": n_top_variable,
                    "soft_power": soft_power,
                    "min_module_size": min_module_size,
                    "merge_cut_height": merge_cut_height,
                },
                description=f"Identified {stats['n_modules']} co-expression modules",
                ir=ir,
            )

            # Format response
            response = f"Successfully identified co-expression modules in '{modality_name}'!\n\n"
            response += "**WGCNA-lite Module Identification Results:**\n"
            response += f"- Modules found: {stats['n_modules']}\n"
            response += f"- Proteins in modules: {stats['n_proteins_in_modules']}\n"
            response += f"- Proteins unassigned (grey): {stats['n_proteins_unassigned']}\n"
            response += f"- Proteins analyzed: {stats['n_proteins_analyzed']}\n"
            response += f"- Correlation method: {stats['correlation_method']}\n"

            if soft_power is not None:
                response += f"- Soft power: {soft_power}\n"

            response += "\n**Module Sizes:**\n"
            module_sizes = stats.get("module_sizes", {})
            for color, size in sorted(module_sizes.items(), key=lambda x: -x[1]):
                response += f"- {color}: {size} proteins\n"

            response += "\n**Module Colors:**\n"
            module_colors = stats.get("module_colors", [])
            response += f"- {', '.join(module_colors)}\n"

            response += f"\n**New modality created**: '{result_name}'"
            response += "\n\n**Next steps**: correlate_modules_with_traits() to find trait associations"

            return response

        except Exception as e:
            logger.error(f"Error identifying co-expression modules: {e}")
            return f"Error in module identification: {str(e)}"

    # =========================================================================
    # TOOL 2: Correlate modules with clinical traits
    # =========================================================================

    @tool
    def correlate_modules_with_traits(
        modality_name: str,
        trait_columns: List[str],
        correlation_method: str = "pearson",
    ) -> str:
        """
        Correlate module eigengenes with clinical traits to find biologically relevant modules.

        Requires identify_coexpression_modules to have been run first on the modality
        (the modality must contain module eigengenes in obs columns).

        Args:
            modality_name: Name of the modality with module assignments (from identify_coexpression_modules)
            trait_columns: List of clinical trait column names in obs to correlate with
            correlation_method: Correlation method ('pearson' or 'spearman', default: 'pearson')

        Returns:
            str: Module-trait correlation results with significant associations
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return (
                f"Modality '{modality_name}' not found. "
                f"Available: {data_manager.list_modalities()}"
            )

        try:
            adata_copy = adata.copy()

            result_adata, stats, ir = network_service.correlate_modules_with_traits(
                adata_copy,
                traits=trait_columns,
                correlation_method=correlation_method,
            )

            # Store result as new modality
            result_name = f"{modality_name}_module_traits"
            data_manager.store_modality(
                name=result_name,
                adata=result_adata,
                parent_name=modality_name,
                step_summary=f"Module-trait correlation: {stats['n_significant_correlations']} significant",
            )

            # Log tool usage with provenance
            data_manager.log_tool_usage(
                tool_name="correlate_modules_with_traits",
                parameters={
                    "modality_name": modality_name,
                    "trait_columns": trait_columns,
                    "correlation_method": correlation_method,
                },
                description=f"Correlated {stats['n_modules']} modules with {stats['n_traits']} traits",
                ir=ir,
            )

            # Format response
            response = f"Successfully correlated modules with traits in '{modality_name}'!\n\n"
            response += "**Module-Trait Correlation Results:**\n"
            response += f"- Modules analyzed: {stats['n_modules']}\n"
            response += f"- Traits tested: {stats['n_traits']}\n"
            response += f"- Total tests: {stats['n_tests']}\n"
            response += f"- Significant correlations (FDR < 0.05): {stats['n_significant_correlations']}\n"
            response += f"- Correlation method: {stats['correlation_method']}\n"

            significant_pairs = stats.get("significant_pairs", [])
            if significant_pairs:
                response += "\n**Significant Module-Trait Associations:**\n"
                for pair in significant_pairs:
                    direction = "positive" if pair["correlation"] > 0 else "negative"
                    response += (
                        f"- Module **{pair['module']}** <-> **{pair['trait']}**: "
                        f"r={pair['correlation']:.3f} ({direction})\n"
                    )
            else:
                response += "\nNo significant module-trait correlations found at FDR < 0.05.\n"
                response += "Consider using more samples or relaxing the threshold.\n"

            response += f"\n**New modality created**: '{result_name}'"
            response += "\n\n**Next steps**: Examine hub proteins in significant modules"

            return response

        except Exception as e:
            logger.error(f"Error correlating modules with traits: {e}")
            return f"Error in module-trait correlation: {str(e)}"

    # =========================================================================
    # TOOL 3: Cox proportional hazards survival analysis
    # =========================================================================

    @tool
    def perform_survival_analysis(
        modality_name: str,
        time_column: str = "PFS_days",
        event_column: str = "PFS_event",
        covariates: Optional[List[str]] = None,
        fdr_threshold: float = 0.05,
        penalizer: float = 0.1,
    ) -> str:
        """
        Perform Cox proportional hazards regression across all proteins to identify
        proteins significantly associated with survival outcomes.

        Each protein is tested individually (univariate Cox models), with optional
        covariate adjustment. Results are FDR-corrected for multiple testing.

        Args:
            modality_name: Name of the proteomics modality to analyze
            time_column: Column in obs with survival duration in days (default: 'PFS_days')
            event_column: Column in obs with event indicator, 1=event 0=censored (default: 'PFS_event')
            covariates: Optional list of covariate columns to adjust for (e.g., ['age', 'stage'])
            fdr_threshold: FDR significance threshold (default: 0.05)
            penalizer: L2 regularization strength for model stability (default: 0.1)

        Returns:
            str: Cox regression results with significant hazard ratios
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return (
                f"Modality '{modality_name}' not found. "
                f"Available: {data_manager.list_modalities()}"
            )

        try:
            adata_copy = adata.copy()

            result_adata, stats, ir = survival_service.perform_cox_regression(
                adata_copy,
                duration_col=time_column,
                event_col=event_column,
                covariates=covariates,
                fdr_threshold=fdr_threshold,
                penalizer=penalizer,
            )

            # Store result as new modality
            result_name = f"{modality_name}_cox_survival"
            data_manager.store_modality(
                name=result_name,
                adata=result_adata,
                parent_name=modality_name,
                step_summary=f"Cox regression: {stats['n_significant_proteins']} significant proteins",
            )

            # Log tool usage with provenance
            data_manager.log_tool_usage(
                tool_name="perform_survival_analysis",
                parameters={
                    "modality_name": modality_name,
                    "time_column": time_column,
                    "event_column": event_column,
                    "covariates": covariates,
                    "fdr_threshold": fdr_threshold,
                    "penalizer": penalizer,
                },
                description=f"Cox regression: {stats['n_significant_proteins']} significant at FDR < {fdr_threshold}",
                ir=ir,
            )

            # Format response
            response = f"Successfully performed Cox survival analysis on '{modality_name}'!\n\n"
            response += "**Cox Proportional Hazards Regression Results:**\n"
            response += f"- Proteins tested: {stats['n_proteins_tested']}\n"
            response += f"- Proteins converged: {stats['n_proteins_converged']}\n"
            response += f"- Significant proteins (FDR < {fdr_threshold}): {stats['n_significant_proteins']}\n"
            response += f"- Failed convergence: {stats['n_failed_convergence']}\n"
            response += f"- Low variance skipped: {stats['n_low_variance_skipped']}\n"
            response += f"- Valid samples: {stats['n_valid_samples']}\n"
            response += f"- Total events: {stats['n_events']}\n"
            response += f"- Median survival: {stats['median_survival_days']:.1f} days\n"

            top_hr = stats.get("top_hazard_ratios", [])
            if top_hr:
                response += "\n**Top Hazard Ratios (highest risk):**\n"
                for entry in top_hr:
                    response += (
                        f"- **{entry['protein']}**: HR={entry['hr']:.3f}, "
                        f"FDR={entry['fdr']:.4f}\n"
                    )

            significant_proteins = stats.get("significant_proteins", [])
            if significant_proteins:
                display_count = min(len(significant_proteins), 20)
                response += f"\n**Significant Proteins** ({len(significant_proteins)} total):\n"
                response += f"- {', '.join(significant_proteins[:display_count])}"
                if len(significant_proteins) > display_count:
                    response += f"\n- ... and {len(significant_proteins) - display_count} more"
                response += "\n"
            else:
                response += "\nNo proteins reached significance at the specified FDR threshold.\n"
                response += "Consider relaxing the threshold or checking data quality.\n"

            response += f"\n**New modality created**: '{result_name}'"
            response += "\n\n**Next steps**: find_survival_biomarkers() for Kaplan-Meier curves of top candidates"

            return response

        except Exception as e:
            logger.error(f"Error in Cox survival analysis: {e}")
            return f"Error in survival analysis: {str(e)}"

    # =========================================================================
    # TOOL 4: Batch Kaplan-Meier biomarker screening
    # =========================================================================

    @tool
    def find_survival_biomarkers(
        modality_name: str,
        time_column: str = "PFS_days",
        event_column: str = "PFS_event",
        proteins: Optional[List[str]] = None,
        stratify_method: str = "median",
        fdr_threshold: float = 0.05,
    ) -> str:
        """
        Screen proteins as survival biomarkers using batch Kaplan-Meier analysis.

        Performs log-rank tests for each protein by stratifying patients into
        high/low expression groups and comparing survival curves. Applies
        FDR correction across all tested proteins.

        Args:
            modality_name: Name of the proteomics modality to analyze
            time_column: Column in obs with survival duration in days (default: 'PFS_days')
            event_column: Column in obs with event indicator (default: 'PFS_event')
            proteins: Specific proteins to test (default: all proteins in the modality)
            stratify_method: Patient stratification method ('median', 'tertile', 'quartile', 'optimal')
            fdr_threshold: FDR significance threshold (default: 0.05)

        Returns:
            str: Biomarker screening results with significant proteins
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return (
                f"Modality '{modality_name}' not found. "
                f"Available: {data_manager.list_modalities()}"
            )

        try:
            adata_copy = adata.copy()

            result_adata, stats, ir = survival_service.batch_kaplan_meier(
                adata_copy,
                duration_col=time_column,
                event_col=event_column,
                proteins=proteins,
                stratify_method=stratify_method,
                fdr_threshold=fdr_threshold,
            )

            # Store result as new modality
            result_name = f"{modality_name}_km_biomarkers"
            data_manager.store_modality(
                name=result_name,
                adata=result_adata,
                parent_name=modality_name,
                step_summary=f"KM biomarker screening: {stats['n_significant']} significant",
            )

            # Log tool usage with provenance
            data_manager.log_tool_usage(
                tool_name="find_survival_biomarkers",
                parameters={
                    "modality_name": modality_name,
                    "time_column": time_column,
                    "event_column": event_column,
                    "proteins": proteins,
                    "stratify_method": stratify_method,
                    "fdr_threshold": fdr_threshold,
                },
                description=f"KM biomarker screening: {stats['n_significant']} significant biomarkers",
                ir=ir,
            )

            # Format response
            response = f"Successfully screened survival biomarkers in '{modality_name}'!\n\n"
            response += "**Batch Kaplan-Meier Biomarker Screening Results:**\n"
            response += f"- Proteins tested: {stats['n_proteins_tested']}\n"
            response += f"- Significant biomarkers (FDR < {fdr_threshold}): {stats['n_significant']}\n"
            response += f"- Stratification method: {stratify_method}\n"

            significant_proteins = stats.get("significant_proteins", [])
            if significant_proteins:
                display_count = min(len(significant_proteins), 20)
                response += f"\n**Significant Survival Biomarkers** ({len(significant_proteins)} total):\n"
                response += f"- {', '.join(significant_proteins[:display_count])}"
                if len(significant_proteins) > display_count:
                    response += f"\n- ... and {len(significant_proteins) - display_count} more"
                response += "\n"

                response += "\n**Interpretation:**\n"
                response += "- These proteins show significantly different survival curves between high/low expression groups\n"
                response += "- Higher log-rank statistic indicates stronger separation\n"
                response += "- Validate top candidates with independent cohorts\n"
            else:
                response += "\nNo proteins reached significance as survival biomarkers.\n"
                response += "Consider:\n"
                response += "- Increasing sample size\n"
                response += "- Using 'optimal' stratification for better cutpoint selection\n"
                response += "- Relaxing the FDR threshold\n"

            response += f"\n**New modality created**: '{result_name}'"
            response += "\n\n**Next steps**: Examine individual KM curves for top biomarkers"

            return response

        except Exception as e:
            logger.error(f"Error in survival biomarker screening: {e}")
            return f"Error in biomarker screening: {str(e)}"

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    tools = [
        identify_coexpression_modules,
        correlate_modules_with_traits,
        perform_survival_analysis,
        find_survival_biomarkers,
    ]

    # Add delegation tools if provided by parent agent
    if delegation_tools:
        tools = tools + delegation_tools

    # Create system prompt
    system_prompt = create_biomarker_discovery_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=BiomarkerDiscoveryExpertState,
    )
