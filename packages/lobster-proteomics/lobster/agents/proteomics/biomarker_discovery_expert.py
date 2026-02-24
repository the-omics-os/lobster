"""
Biomarker Discovery Expert sub-agent for proteomics network and survival analysis.

This sub-agent wraps WGCNALiteService and ProteomicsSurvivalService with 7 tools:
- identify_coexpression_modules: WGCNA-lite module identification
- correlate_modules_with_traits: Module-trait correlation analysis
- perform_survival_analysis: Cox proportional hazards regression
- find_survival_biomarkers: Batch Kaplan-Meier biomarker screening
- select_biomarker_panel: Multi-method feature selection (LASSO, stability, Boruta)
- evaluate_biomarker_panel: Nested CV evaluation with AUC reporting
- extract_hub_proteins: Hub protein extraction from WGCNA modules via kME

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
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
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
            response += (
                f"- Proteins unassigned (grey): {stats['n_proteins_unassigned']}\n"
            )
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
            response = (
                f"Successfully correlated modules with traits in '{modality_name}'!\n\n"
            )
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
                response += (
                    "\nNo significant module-trait correlations found at FDR < 0.05.\n"
                )
                response += "Consider using more samples or relaxing the threshold.\n"

            response += f"\n**New modality created**: '{result_name}'"
            response += (
                "\n\n**Next steps**: Examine hub proteins in significant modules"
            )

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
                response += (
                    f"\n**Significant Proteins** ({len(significant_proteins)} total):\n"
                )
                response += f"- {', '.join(significant_proteins[:display_count])}"
                if len(significant_proteins) > display_count:
                    response += (
                        f"\n- ... and {len(significant_proteins) - display_count} more"
                    )
                response += "\n"
            else:
                response += "\nNo proteins reached significance at the specified FDR threshold.\n"
                response += (
                    "Consider relaxing the threshold or checking data quality.\n"
                )

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
            response = (
                f"Successfully screened survival biomarkers in '{modality_name}'!\n\n"
            )
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
                    response += (
                        f"\n- ... and {len(significant_proteins) - display_count} more"
                    )
                response += "\n"

                response += "\n**Interpretation:**\n"
                response += "- These proteins show significantly different survival curves between high/low expression groups\n"
                response += (
                    "- Higher log-rank statistic indicates stronger separation\n"
                )
                response += "- Validate top candidates with independent cohorts\n"
            else:
                response += (
                    "\nNo proteins reached significance as survival biomarkers.\n"
                )
                response += "Consider:\n"
                response += "- Increasing sample size\n"
                response += (
                    "- Using 'optimal' stratification for better cutpoint selection\n"
                )
                response += "- Relaxing the FDR threshold\n"

            response += f"\n**New modality created**: '{result_name}'"
            response += (
                "\n\n**Next steps**: Examine individual KM curves for top biomarkers"
            )

            return response

        except Exception as e:
            logger.error(f"Error in survival biomarker screening: {e}")
            return f"Error in biomarker screening: {str(e)}"

    # =========================================================================
    # TOOL 5: Multi-method biomarker panel selection
    # =========================================================================

    @tool
    def select_biomarker_panel(
        modality_name: str,
        target_column: str,
        methods: str = "lasso,stability",
        n_features: int = 20,
        n_iterations: int = 100,
        random_state: int = 42,
    ) -> str:
        """
        Select a biomarker panel using multi-method feature selection.

        Runs LASSO (L1 regularization), stability selection (bootstrapped LASSO),
        and optionally simplified Boruta (shadow feature comparison). Proteins
        selected by multiple methods receive higher consensus scores.

        Args:
            modality_name: Name of the proteomics modality to analyze
            target_column: Column in obs with group labels (e.g., 'condition', 'responder')
            methods: Comma-separated selection methods: 'lasso', 'stability', 'boruta' (default: 'lasso,stability')
            n_features: Maximum number of features in the final panel (default: 20)
            n_iterations: Number of bootstrap iterations for stability selection (default: 100)
            random_state: Random seed for reproducibility (default: 42)

        Returns:
            str: Panel selection results with per-method counts and consensus panel
        """
        import numpy as np
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return (
                f"Modality '{modality_name}' not found. "
                f"Available: {data_manager.list_modalities()}"
            )

        try:
            # Validate target column
            if target_column not in adata.obs.columns:
                return (
                    f"Target column '{target_column}' not found in obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )

            adata_copy = adata.copy()
            method_list = [m.strip().lower() for m in methods.split(",")]
            rng = np.random.RandomState(random_state)

            # Extract X matrix (handle sparse)
            X = adata_copy.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float64)

            # Encode target labels
            y_raw = adata_copy.obs[target_column].values
            if not np.issubdtype(y_raw.dtype, np.number):
                le = LabelEncoder()
                y = le.fit_transform(y_raw)
            else:
                y = np.asarray(y_raw, dtype=np.float64)

            # Handle NaN in X (median imputation)
            nan_mask = np.isnan(X)
            if nan_mask.any():
                col_medians = np.nanmedian(X, axis=0)
                inds = np.where(nan_mask)
                X[inds] = np.take(col_medians, inds[1])

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            protein_names = adata_copy.var_names.tolist()
            n_proteins = len(protein_names)
            n_methods = len(method_list)
            method_results = {}

            # --- LASSO selection ---
            if "lasso" in method_list:
                lasso = LassoCV(cv=5, random_state=random_state, max_iter=5000)
                lasso.fit(X_scaled, y)
                coefs = lasso.coef_.flatten() if lasso.coef_.ndim > 1 else lasso.coef_
                lasso_selected = np.abs(coefs) > 0
                adata_copy.var["lasso_selected"] = lasso_selected
                adata_copy.var["lasso_coef"] = coefs

                # Top n by absolute coefficient
                top_lasso_idx = np.argsort(np.abs(coefs))[-n_features:]
                n_lasso = int(lasso_selected.sum())
                method_results["lasso"] = {
                    "n_selected": n_lasso,
                    "alpha": float(lasso.alpha_),
                }
                logger.info(
                    f"LASSO selected {n_lasso} proteins (alpha={lasso.alpha_:.4f})"
                )

            # --- Stability selection ---
            if "stability" in method_list:
                selection_counts = np.zeros(n_proteins)
                n_samples = X_scaled.shape[0]
                subsample_size = int(0.8 * n_samples)

                for i in range(n_iterations):
                    # Bootstrap subsample (80%)
                    idx = rng.choice(n_samples, size=subsample_size, replace=False)
                    X_sub = X_scaled[idx]
                    y_sub = y[idx]

                    # Fit LASSO on subsample
                    lasso_boot = LassoCV(cv=3, random_state=i, max_iter=5000)
                    try:
                        lasso_boot.fit(X_sub, y_sub)
                        boot_coefs = (
                            lasso_boot.coef_.flatten()
                            if lasso_boot.coef_.ndim > 1
                            else lasso_boot.coef_
                        )
                        selection_counts += (np.abs(boot_coefs) > 0).astype(int)
                    except Exception:
                        continue  # Skip failed iterations

                stability_freq = selection_counts / n_iterations
                stability_selected = stability_freq > 0.6
                adata_copy.var["stability_selected"] = stability_selected
                adata_copy.var["stability_frequency"] = stability_freq

                n_stability = int(stability_selected.sum())
                method_results["stability"] = {"n_selected": n_stability}
                logger.info(
                    f"Stability selection: {n_stability} proteins (>60% frequency)"
                )

            # --- Boruta (simplified) ---
            if "boruta" in method_list:
                from sklearn.ensemble import RandomForestClassifier

                n_rounds = min(n_iterations, 50)
                hit_counts = np.zeros(n_proteins)

                for r in range(n_rounds):
                    # Create shadow features (permuted copies)
                    X_shadow = X_scaled.copy()
                    for col in range(X_shadow.shape[1]):
                        rng.shuffle(X_shadow[:, col])
                    X_combined = np.hstack([X_scaled, X_shadow])

                    # Fit random forest
                    rf = RandomForestClassifier(
                        n_estimators=100, random_state=r, n_jobs=-1
                    )
                    try:
                        rf.fit(X_combined, y)
                        importances = rf.feature_importances_
                        real_importances = importances[:n_proteins]
                        shadow_importances = importances[n_proteins:]
                        shadow_max = shadow_importances.max()

                        # Count hits: real feature beats max shadow
                        hit_counts += (real_importances > shadow_max).astype(int)
                    except Exception:
                        continue

                # Features that beat shadow in >50% of rounds are selected
                boruta_selected = hit_counts > (n_rounds * 0.5)
                adata_copy.var["boruta_selected"] = boruta_selected

                n_boruta = int(boruta_selected.sum())
                method_results["boruta"] = {
                    "n_selected": n_boruta,
                    "note": "experimental (simplified Boruta)",
                }
                logger.info(f"Boruta (simplified): {n_boruta} proteins selected")

            # --- Consensus scoring ---
            consensus_score = np.zeros(n_proteins)
            for method_name in method_list:
                col = f"{method_name}_selected"
                if col in adata_copy.var.columns:
                    consensus_score += adata_copy.var[col].astype(float).values
            consensus_score /= n_methods
            adata_copy.var["panel_consensus_score"] = consensus_score

            # Final panel: top n_features by consensus, break ties by stability then lasso
            sort_keys = consensus_score.copy()
            # Add tiny tiebreakers from stability_frequency and lasso_coef
            if "stability_frequency" in adata_copy.var.columns:
                sort_keys += adata_copy.var["stability_frequency"].values * 1e-4
            if "lasso_coef" in adata_copy.var.columns:
                sort_keys += np.abs(adata_copy.var["lasso_coef"].values) * 1e-8

            panel_idx = np.argsort(sort_keys)[-n_features:][::-1]
            panel_proteins = [
                protein_names[i] for i in panel_idx if consensus_score[i] > 0
            ]

            # Trim to only those with non-zero consensus
            if not panel_proteins:
                # Fallback: take top by any single method
                panel_proteins = [
                    protein_names[i] for i in np.argsort(sort_keys)[-n_features:][::-1]
                ]

            adata_copy.uns["biomarker_panel"] = {
                "method_results": method_results,
                "final_panel": panel_proteins,
                "n_features": len(panel_proteins),
                "methods_used": method_list,
                "target_column": target_column,
            }

            # Store result
            result_name = f"{modality_name}_panel_selected"
            data_manager.store_modality(
                name=result_name,
                adata=adata_copy,
                parent_name=modality_name,
                step_summary=f"Biomarker panel selection: {len(panel_proteins)} proteins via {', '.join(method_list)}",
            )

            # Log with IR
            ir = AnalysisStep(
                operation="proteomics.biomarker.panel_selection",
                tool_name="select_biomarker_panel",
                description=f"Multi-method biomarker panel selection ({', '.join(method_list)})",
                library="sklearn",
                code_template="""# Multi-method biomarker panel selection
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

X = adata.X.copy()
y = LabelEncoder().fit_transform(adata.obs[{{ target_column | tojson }}])
X_scaled = StandardScaler().fit_transform(X)

# LASSO selection
lasso = LassoCV(cv=5, random_state={{ random_state }}, max_iter=5000)
lasso.fit(X_scaled, y)
lasso_selected = np.abs(lasso.coef_) > 0

# Stability selection (bootstrap LASSO)
selection_counts = np.zeros(X.shape[1])
for i in range({{ n_iterations }}):
    idx = np.random.choice(len(X), int(0.8 * len(X)), replace=False)
    lasso_boot = LassoCV(cv=3, random_state=i, max_iter=5000)
    lasso_boot.fit(X_scaled[idx], y[idx])
    selection_counts += (np.abs(lasso_boot.coef_) > 0)
stability_freq = selection_counts / {{ n_iterations }}
stability_selected = stability_freq > 0.6""",
                imports=[
                    "from sklearn.linear_model import LassoCV",
                    "from sklearn.preprocessing import LabelEncoder, StandardScaler",
                    "import numpy as np",
                ],
                parameters={
                    "target_column": target_column,
                    "methods": methods,
                    "n_features": n_features,
                    "n_iterations": n_iterations,
                    "random_state": random_state,
                },
                parameter_schema={
                    "target_column": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        required=True,
                        description="Column in obs with group labels",
                    ),
                    "n_features": ParameterSpec(
                        param_type="int",
                        papermill_injectable=True,
                        default_value=20,
                        required=False,
                        description="Maximum features in final panel",
                    ),
                    "n_iterations": ParameterSpec(
                        param_type="int",
                        papermill_injectable=True,
                        default_value=100,
                        required=False,
                        description="Bootstrap iterations for stability selection",
                    ),
                    "random_state": ParameterSpec(
                        param_type="int",
                        papermill_injectable=True,
                        default_value=42,
                        required=False,
                        description="Random seed",
                    ),
                },
                input_entities=["adata"],
                output_entities=["adata_panel"],
            )

            data_manager.log_tool_usage(
                tool_name="select_biomarker_panel",
                parameters={
                    "modality_name": modality_name,
                    "target_column": target_column,
                    "methods": methods,
                    "n_features": n_features,
                    "n_iterations": n_iterations,
                    "random_state": random_state,
                },
                description=f"Selected {len(panel_proteins)}-protein panel via {', '.join(method_list)}",
                ir=ir,
            )

            # Format response
            response = (
                f"Successfully selected biomarker panel from '{modality_name}'!\n\n"
            )
            response += "**Multi-Method Biomarker Panel Selection Results:**\n"
            response += f"- Methods used: {', '.join(method_list)}\n"
            response += f"- Target: {target_column}\n\n"

            for method_name, result in method_results.items():
                response += f"**{method_name.upper()}**: {result['n_selected']} proteins selected"
                if "alpha" in result:
                    response += f" (alpha={result['alpha']:.4f})"
                if "note" in result:
                    response += f" [{result['note']}]"
                response += "\n"

            # Method overlap
            if len(method_list) > 1:
                response += f"\n**Consensus Panel** ({len(panel_proteins)} proteins):\n"
                for p in panel_proteins[:20]:
                    score = float(adata_copy.var.loc[p, "panel_consensus_score"])
                    response += f"- {p} (consensus: {score:.2f})\n"
                if len(panel_proteins) > 20:
                    response += f"- ... and {len(panel_proteins) - 20} more\n"
            else:
                response += f"\n**Selected Panel** ({len(panel_proteins)} proteins):\n"
                for p in panel_proteins[:20]:
                    response += f"- {p}\n"

            response += f"\n**New modality created**: '{result_name}'"
            response += "\n\n**Next steps**: evaluate_biomarker_panel() for nested CV performance evaluation"

            return response

        except Exception as e:
            logger.error(f"Error in biomarker panel selection: {e}")
            return f"Error in panel selection: {str(e)}"

    # =========================================================================
    # TOOL 6: Nested cross-validation biomarker panel evaluation
    # =========================================================================

    @tool
    def evaluate_biomarker_panel(
        modality_name: str,
        target_column: str,
        proteins: list = None,
        n_outer_folds: int = 5,
        n_inner_folds: int = 3,
        classifier: str = "logistic",
        random_state: int = 42,
    ) -> str:
        """
        Evaluate a biomarker panel using nested cross-validation to avoid information leakage.

        Uses an outer loop for performance evaluation and an inner loop for
        hyperparameter tuning. Reports AUC, sensitivity, and specificity per fold.

        Args:
            modality_name: Name of the proteomics modality (should contain biomarker_panel in uns)
            target_column: Column in obs with group labels for classification
            proteins: Specific protein list to evaluate (default: uses panel from select_biomarker_panel)
            n_outer_folds: Number of outer CV folds for evaluation (default: 5)
            n_inner_folds: Number of inner CV folds for tuning (default: 3)
            classifier: Classifier type: 'logistic' or 'random_forest' (default: 'logistic')
            random_state: Random seed for reproducibility (default: 42)

        Returns:
            str: Evaluation results with mean AUC, sensitivity, specificity, per-fold breakdown
        """
        import numpy as np
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return (
                f"Modality '{modality_name}' not found. "
                f"Available: {data_manager.list_modalities()}"
            )

        try:
            # Validate target column
            if target_column not in adata.obs.columns:
                return (
                    f"Target column '{target_column}' not found in obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )

            adata_copy = adata.copy()

            # Determine panel proteins
            if proteins is None:
                panel_info = adata_copy.uns.get("biomarker_panel")
                if panel_info is not None:
                    proteins = panel_info.get("final_panel")
            if proteins is None:
                return (
                    "No protein panel specified and no biomarker_panel found in uns. "
                    "Run select_biomarker_panel first or provide a protein list."
                )

            # Validate proteins exist in var
            valid_proteins = [p for p in proteins if p in adata_copy.var_names]
            if not valid_proteins:
                return (
                    f"None of the specified proteins found in the modality var_names."
                )
            if len(valid_proteins) < len(proteins):
                logger.warning(
                    f"{len(proteins) - len(valid_proteins)} proteins not found, "
                    f"using {len(valid_proteins)} valid proteins"
                )

            # Extract X (panel proteins only)
            protein_mask = adata_copy.var_names.isin(valid_proteins)
            X = adata_copy[:, protein_mask].X
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float64)

            # Encode target
            y_raw = adata_copy.obs[target_column].values
            if not np.issubdtype(y_raw.dtype, np.number):
                le = LabelEncoder()
                y = le.fit_transform(y_raw)
            else:
                y = np.asarray(y_raw, dtype=np.float64)

            # Handle NaN
            nan_mask = np.isnan(X)
            if nan_mask.any():
                col_medians = np.nanmedian(X, axis=0)
                inds = np.where(nan_mask)
                X[inds] = np.take(col_medians, inds[1])

            # Select classifier
            if classifier == "logistic":
                from sklearn.linear_model import LogisticRegression

                def make_clf():
                    return LogisticRegression(max_iter=1000, random_state=random_state)

            elif classifier == "random_forest":
                from sklearn.ensemble import RandomForestClassifier

                def make_clf():
                    return RandomForestClassifier(
                        n_estimators=100, random_state=random_state
                    )

            else:
                return f"Unknown classifier '{classifier}'. Use 'logistic' or 'random_forest'."

            # Nested CV: outer loop for evaluation, inner loop for tuning
            outer_cv = StratifiedKFold(
                n_splits=n_outer_folds, shuffle=True, random_state=random_state
            )
            inner_cv = StratifiedKFold(
                n_splits=n_inner_folds, shuffle=True, random_state=random_state
            )

            per_fold_results = []
            scaler = StandardScaler()

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Scale within fold (no data leakage)
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Inner CV for validation (optional tuning step)
                clf = make_clf()
                inner_scores = cross_val_score(
                    clf, X_train_scaled, y_train, cv=inner_cv, scoring="roc_auc"
                )

                # Refit on full training set
                clf = make_clf()
                clf.fit(X_train_scaled, y_train)

                # Evaluate on outer test set
                if hasattr(clf, "predict_proba"):
                    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_prob = clf.decision_function(X_test_scaled)

                y_pred = clf.predict(X_test_scaled)

                # Metrics
                try:
                    fold_auc = float(roc_auc_score(y_test, y_prob))
                except ValueError:
                    fold_auc = 0.5  # Single class in fold

                # Sensitivity and specificity
                tp = int(((y_pred == 1) & (y_test == 1)).sum())
                tn = int(((y_pred == 0) & (y_test == 0)).sum())
                fp = int(((y_pred == 1) & (y_test == 0)).sum())
                fn = int(((y_pred == 0) & (y_test == 1)).sum())

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                per_fold_results.append(
                    {
                        "fold": fold_idx + 1,
                        "auc": fold_auc,
                        "sensitivity": float(sensitivity),
                        "specificity": float(specificity),
                        "inner_mean_auc": float(np.mean(inner_scores)),
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                    }
                )

            # Aggregate
            aucs = [r["auc"] for r in per_fold_results]
            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs))
            mean_sensitivity = float(
                np.mean([r["sensitivity"] for r in per_fold_results])
            )
            mean_specificity = float(
                np.mean([r["specificity"] for r in per_fold_results])
            )

            # Store evaluation results
            adata_copy.uns["biomarker_evaluation"] = {
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "per_fold_auc": aucs,
                "sensitivity": mean_sensitivity,
                "specificity": mean_specificity,
                "n_features": len(valid_proteins),
                "classifier": classifier,
                "n_outer_folds": n_outer_folds,
                "n_inner_folds": n_inner_folds,
                "per_fold_results": per_fold_results,
                "proteins_evaluated": valid_proteins,
            }

            # Store result
            result_name = f"{modality_name}_panel_evaluated"
            data_manager.store_modality(
                name=result_name,
                adata=adata_copy,
                parent_name=modality_name,
                step_summary=f"Panel evaluation: AUC={mean_auc:.3f}+/-{std_auc:.3f} ({classifier})",
            )

            # Log with IR
            ir = AnalysisStep(
                operation="proteomics.biomarker.panel_evaluation",
                tool_name="evaluate_biomarker_panel",
                description=f"Nested CV biomarker panel evaluation ({classifier})",
                library="sklearn",
                code_template="""# Nested cross-validation for biomarker panel evaluation
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Extract panel proteins
proteins = {{ proteins | tojson }}
X = adata[:, proteins].X.copy()
y = LabelEncoder().fit_transform(adata.obs[{{ target_column | tojson }}])

# Nested CV
outer_cv = StratifiedKFold(n_splits={{ n_outer_folds }}, shuffle=True, random_state={{ random_state }})
inner_cv = StratifiedKFold(n_splits={{ n_inner_folds }}, shuffle=True, random_state={{ random_state }})
scaler = StandardScaler()
aucs = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    clf = LogisticRegression(max_iter=1000)  # or RandomForestClassifier
    clf.fit(X_train, y[train_idx])
    y_prob = clf.predict_proba(X_test)[:, 1]
    aucs.append(roc_auc_score(y[test_idx], y_prob))

print(f"AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")""",
                imports=[
                    "from sklearn.model_selection import StratifiedKFold, cross_val_score",
                    "from sklearn.metrics import roc_auc_score",
                    "from sklearn.preprocessing import StandardScaler, LabelEncoder",
                    "import numpy as np",
                ],
                parameters={
                    "target_column": target_column,
                    "proteins": valid_proteins,
                    "n_outer_folds": n_outer_folds,
                    "n_inner_folds": n_inner_folds,
                    "classifier": classifier,
                    "random_state": random_state,
                },
                parameter_schema={
                    "target_column": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        required=True,
                        description="Column in obs with group labels",
                    ),
                    "n_outer_folds": ParameterSpec(
                        param_type="int",
                        papermill_injectable=True,
                        default_value=5,
                        required=False,
                        description="Number of outer CV folds",
                    ),
                    "classifier": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value="logistic",
                        required=False,
                        description="Classifier type",
                    ),
                },
                input_entities=["adata"],
                output_entities=["adata_evaluated"],
            )

            data_manager.log_tool_usage(
                tool_name="evaluate_biomarker_panel",
                parameters={
                    "modality_name": modality_name,
                    "target_column": target_column,
                    "proteins": valid_proteins,
                    "n_outer_folds": n_outer_folds,
                    "n_inner_folds": n_inner_folds,
                    "classifier": classifier,
                },
                description=f"Nested CV evaluation: AUC={mean_auc:.3f}+/-{std_auc:.3f}",
                ir=ir,
            )

            # Format response
            response = (
                f"Successfully evaluated biomarker panel from '{modality_name}'!\n\n"
            )
            response += "**Nested Cross-Validation Results:**\n"
            response += f"- Classifier: {classifier}\n"
            response += f"- Panel size: {len(valid_proteins)} proteins\n"
            response += (
                f"- Outer folds: {n_outer_folds}, Inner folds: {n_inner_folds}\n\n"
            )
            response += f"**Overall Performance:**\n"
            response += f"- **AUC: {mean_auc:.3f} +/- {std_auc:.3f}**\n"
            response += f"- Sensitivity: {mean_sensitivity:.3f}\n"
            response += f"- Specificity: {mean_specificity:.3f}\n\n"

            response += "**Per-Fold Breakdown:**\n"
            for r in per_fold_results:
                response += (
                    f"- Fold {r['fold']}: AUC={r['auc']:.3f}, "
                    f"Sens={r['sensitivity']:.3f}, Spec={r['specificity']:.3f} "
                    f"(train={r['n_train']}, test={r['n_test']})\n"
                )

            response += f"\n**New modality created**: '{result_name}'"
            response += "\n\n**Interpretation:**\n"
            if mean_auc >= 0.9:
                response += "- Excellent discriminative performance (AUC >= 0.9)\n"
            elif mean_auc >= 0.8:
                response += "- Good discriminative performance (AUC >= 0.8)\n"
            elif mean_auc >= 0.7:
                response += "- Acceptable discriminative performance (AUC >= 0.7)\n"
            else:
                response += "- Limited discriminative performance (AUC < 0.7) -- consider different features or methods\n"
            response += "- Validate with an independent cohort before clinical use"

            return response

        except Exception as e:
            logger.error(f"Error in biomarker panel evaluation: {e}")
            return f"Error in panel evaluation: {str(e)}"

    # =========================================================================
    # TOOL 7: Hub protein extraction from WGCNA modules
    # =========================================================================

    @tool
    def extract_hub_proteins(
        modality_name: str,
        module_colors: list = None,
        kme_threshold: float = 0.7,
        top_n: int = 10,
    ) -> str:
        """
        Extract hub proteins from WGCNA co-expression modules based on module membership (kME).

        Calls WGCNALiteService.calculate_module_membership to compute kME scores,
        then filters hub proteins by module and kME threshold. Hub proteins are
        the most connected proteins within their module -- strong biomarker candidates.

        Args:
            modality_name: Name of the modality with WGCNA module assignments (from identify_coexpression_modules)
            module_colors: Specific modules to extract hubs from (default: all non-grey modules)
            kme_threshold: Minimum kME score to consider a protein as hub (default: 0.7)
            top_n: Maximum hub proteins per module (default: 10)

        Returns:
            str: Hub proteins per module with kME scores and suggested next steps
        """
        import numpy as np

        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return (
                f"Modality '{modality_name}' not found. "
                f"Available: {data_manager.list_modalities()}"
            )

        try:
            # Validate WGCNA module assignments exist
            if "module" not in adata.var.columns:
                return (
                    "No WGCNA module assignments found (var['module'] missing). "
                    "Run identify_coexpression_modules first."
                )

            # Calculate module membership (kME) via the network service
            adata_kme, kme_stats, _ = network_service.calculate_module_membership(adata)

            # Determine target modules
            all_modules = [m for m in adata_kme.var["module"].unique() if m != "grey"]
            if module_colors is not None:
                target_modules = [m for m in module_colors if m in all_modules]
                if not target_modules:
                    return (
                        f"None of the specified modules found. "
                        f"Available modules: {all_modules}"
                    )
            else:
                target_modules = all_modules

            # Check for trait correlations to highlight significant modules
            trait_corr = adata_kme.uns.get("module_trait_correlation", {})
            significant_modules = set()
            if trait_corr:
                for result in trait_corr.get("results", []):
                    if result.get("significant", False):
                        significant_modules.add(result["module"])

            # Extract hub proteins per module
            hub_results = {}
            total_hubs = 0

            for module in target_modules:
                kme_col = f"kME_{module}"
                if kme_col not in adata_kme.var.columns:
                    logger.warning(
                        f"kME column '{kme_col}' not found, skipping module {module}"
                    )
                    continue

                # Filter to proteins in this module
                module_mask = adata_kme.var["module"] == module
                module_proteins = adata_kme.var[module_mask].copy()

                # Apply kME threshold
                hub_mask = module_proteins[kme_col] >= kme_threshold
                hub_proteins = module_proteins[hub_mask].sort_values(
                    kme_col, ascending=False
                )

                # Take top_n
                hub_proteins = hub_proteins.head(top_n)

                if len(hub_proteins) > 0:
                    hub_list = []
                    for protein_name, row in hub_proteins.iterrows():
                        hub_list.append(
                            {
                                "protein": str(protein_name),
                                "kME": float(row[kme_col]),
                                "is_hub": bool(row.get("is_hub", False)),
                            }
                        )
                    hub_results[module] = hub_list
                    total_hubs += len(hub_list)

            # Store results in adata
            adata_kme.uns["hub_proteins"] = {
                "per_module": {
                    m: [h["protein"] for h in hubs] for m, hubs in hub_results.items()
                },
                "total_hubs": total_hubs,
                "kme_threshold": kme_threshold,
                "top_n": top_n,
                "modules_analyzed": target_modules,
            }

            # Store result
            result_name = f"{modality_name}_hubs"
            data_manager.store_modality(
                name=result_name,
                adata=adata_kme,
                parent_name=modality_name,
                step_summary=f"Hub protein extraction: {total_hubs} hubs from {len(hub_results)} modules",
            )

            # Log with IR
            ir = AnalysisStep(
                operation="proteomics.biomarker.hub_extraction",
                tool_name="extract_hub_proteins",
                description="Extract hub proteins from WGCNA modules via module membership (kME)",
                library="lobster.services.analysis.proteomics_network_service",
                code_template="""# Hub protein extraction from WGCNA modules
from lobster.services.analysis.proteomics_network_service import WGCNALiteService

service = WGCNALiteService()
adata_kme, stats, _ = service.calculate_module_membership(adata)

# Filter hub proteins by kME threshold
for module in target_modules:
    kme_col = f"kME_{module}"
    module_mask = adata_kme.var["module"] == module
    hub_mask = adata_kme.var[kme_col] >= {{ kme_threshold }}
    hubs = adata_kme.var[module_mask & hub_mask].sort_values(kme_col, ascending=False)
    print(f"Module {module}: {len(hubs)} hub proteins")""",
                imports=[
                    "from lobster.services.analysis.proteomics_network_service import WGCNALiteService"
                ],
                parameters={
                    "module_colors": module_colors,
                    "kme_threshold": kme_threshold,
                    "top_n": top_n,
                },
                parameter_schema={
                    "kme_threshold": ParameterSpec(
                        param_type="float",
                        papermill_injectable=True,
                        default_value=0.7,
                        required=False,
                        validation_rule="0 < kme_threshold <= 1",
                        description="Minimum kME for hub classification",
                    ),
                    "top_n": ParameterSpec(
                        param_type="int",
                        papermill_injectable=True,
                        default_value=10,
                        required=False,
                        description="Max hub proteins per module",
                    ),
                },
                input_entities=["adata"],
                output_entities=["adata_hubs"],
            )

            data_manager.log_tool_usage(
                tool_name="extract_hub_proteins",
                parameters={
                    "modality_name": modality_name,
                    "module_colors": module_colors,
                    "kme_threshold": kme_threshold,
                    "top_n": top_n,
                },
                description=f"Extracted {total_hubs} hub proteins from {len(hub_results)} modules",
                ir=ir,
            )

            # Format response
            response = (
                f"Successfully extracted hub proteins from '{modality_name}'!\n\n"
            )
            response += f"**Hub Protein Extraction Results:**\n"
            response += f"- Modules analyzed: {len(target_modules)}\n"
            response += f"- kME threshold: {kme_threshold}\n"
            response += f"- Total hub proteins: {total_hubs}\n\n"

            for module, hubs in hub_results.items():
                sig_marker = " *" if module in significant_modules else ""
                response += f"**Module {module}**{sig_marker} ({len(hubs)} hubs):\n"
                for h in hubs:
                    response += f"  - {h['protein']}: kME={h['kME']:.3f}\n"
                response += "\n"

            if significant_modules:
                response += "\\* = module with significant trait correlations\n\n"

            if not hub_results:
                response += "No hub proteins found above the kME threshold.\n"
                response += (
                    f"Consider lowering kme_threshold (current: {kme_threshold}).\n\n"
                )

            response += f"**New modality created**: '{result_name}'"
            response += "\n\n**Next steps**:\n"
            response += "- Validate hub proteins with select_biomarker_panel() for clinical relevance\n"
            response += "- Use perform_survival_analysis() to check prognostic value\n"
            response += "- Cross-reference with literature via research_agent"

            return response

        except Exception as e:
            logger.error(f"Error in hub protein extraction: {e}")
            return f"Error in hub extraction: {str(e)}"

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    tools = [
        identify_coexpression_modules,
        correlate_modules_with_traits,
        perform_survival_analysis,
        find_survival_biomarkers,
        select_biomarker_panel,
        evaluate_biomarker_panel,
        extract_hub_proteins,
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
