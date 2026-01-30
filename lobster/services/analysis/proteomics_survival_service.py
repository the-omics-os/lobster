"""
Proteomics survival analysis service for clinical outcome correlation.

This service implements Cox proportional hazards regression and Kaplan-Meier survival analysis
for proteomics data, enabling identification of proteins associated with clinical outcomes
such as progression-free survival (PFS) or overall survival (OS).

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from scipy import stats

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger
from lobster.utils.statistics import benjamini_hochberg

logger = get_logger(__name__)


# ============================================================================
# Class Constants - Configurable thresholds and defaults
# ============================================================================

# Variance thresholds
LOW_VARIANCE_THRESHOLD = 1e-6  # Skip proteins with variance below this

# Statistical defaults
DEFAULT_FDR_THRESHOLD = 0.05  # False discovery rate significance cutoff
DEFAULT_PENALIZER = 0.1  # L2 regularization for Cox model stability
DEFAULT_MIN_SAMPLES = 20  # Minimum samples for reliable Cox fitting

# Stratification
DEFAULT_STRATIFY_METHOD = "median"
MIN_GROUP_SIZE = 5  # Minimum samples per group for KM analysis
MIN_CUTPOINT_GROUP_SIZE = 10  # Minimum per group for optimal cutpoint search

# Numerical stability
LOG_EPSILON = 1e-300  # Small value to avoid log(0) in p-value calculations

# Progress reporting
PROGRESS_LOG_INTERVAL = 500  # Log progress every N proteins


class ProteomicsSurvivalError(Exception):
    """Base exception for proteomics survival analysis operations."""

    pass


class ProteomicsSurvivalService:
    """
    Survival analysis service for proteomics data.

    This stateless service provides comprehensive survival analysis methods including
    Cox proportional hazards regression, Kaplan-Meier analysis, and log-rank tests
    for identifying proteins associated with clinical outcomes.

    Designed for clinical proteomics studies with survival endpoints such as:
    - Progression-free survival (PFS)
    - Overall survival (OS)
    - Time to recurrence
    - Treatment response duration

    Example usage:
        service = ProteomicsSurvivalService()
        adata_cox, stats, ir = service.perform_cox_regression(
            adata,
            duration_col='PFS_days',
            event_col='PFS_event',
            fdr_threshold=0.05
        )
    """

    def __init__(self):
        """
        Initialize the proteomics survival service.

        This service is stateless and doesn't require a data manager instance.
        """
        logger.debug("Initializing stateless ProteomicsSurvivalService")
        self._lifelines_available = self._check_lifelines()
        logger.debug(f"lifelines available: {self._lifelines_available}")

    def _check_lifelines(self) -> bool:
        """Check if lifelines package is available."""
        try:
            import lifelines  # noqa: F401

            return True
        except ImportError:
            return False

    def _fit_single_protein_cox(
        self,
        protein_index: int,
        protein_name: str,
        protein_values: np.ndarray,
        duration: np.ndarray,
        event: np.ndarray,
        valid_mask: np.ndarray,
        covariate_data: Dict[str, np.ndarray],
        min_samples: int,
        penalizer: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Fit Cox model for a single protein.

        This helper method encapsulates the per-protein fitting logic for parallelization.

        Args:
            protein_index: Index of the protein in the expression matrix
            protein_name: Name of the protein
            protein_values: Expression values for this protein across all samples
            duration: Survival duration array
            event: Event indicator array
            valid_mask: Boolean mask for valid samples
            covariate_data: Dictionary of covariate arrays
            min_samples: Minimum samples required
            penalizer: L2 penalizer for Cox model

        Returns:
            Dictionary with Cox results, or None if protein should be skipped (low variance)
        """
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index

        # Check for low variance
        protein_valid = protein_values[valid_mask]
        if np.std(protein_valid) < LOW_VARIANCE_THRESHOLD:
            return None  # Signal to skip (low variance)

        # Check for missing values in protein
        protein_mask = valid_mask & ~np.isnan(protein_values)
        if protein_mask.sum() < min_samples:
            return None  # Signal to skip (insufficient samples)

        # Prepare data for Cox model
        df_cox = pd.DataFrame(
            {
                "T": duration[protein_mask],
                "E": event[protein_mask].astype(int),
                "protein": protein_values[protein_mask],
            }
        )

        # Add covariates
        for cov_name, cov_values in covariate_data.items():
            df_cox[cov_name] = cov_values[protein_mask]

        # Standardize protein expression for stable fitting
        df_cox["protein"] = (
            df_cox["protein"] - df_cox["protein"].mean()
        ) / df_cox["protein"].std()

        # Fit Cox model
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(
                df_cox,
                duration_col="T",
                event_col="E",
                show_progress=False,
            )

            # Extract results
            hr = float(np.exp(cph.params_["protein"]))
            hr_ci_lower = float(np.exp(cph.confidence_intervals_.loc["protein", "95% lower-bound"]))
            hr_ci_upper = float(np.exp(cph.confidence_intervals_.loc["protein", "95% upper-bound"]))
            p_value = float(cph.summary.loc["protein", "p"])
            z_score = float(cph.summary.loc["protein", "z"])
            coef = float(cph.params_["protein"])
            se = float(cph.summary.loc["protein", "se(coef)"])

            # Calculate concordance index
            try:
                c_index = concordance_index(
                    df_cox["T"],
                    -df_cox["protein"],  # Negative for risk prediction
                    df_cox["E"],
                )
            except Exception:
                c_index = np.nan

            return {
                "protein": protein_name,
                "protein_index": protein_index,
                "hazard_ratio": hr,
                "hr_ci_lower": hr_ci_lower,
                "hr_ci_upper": hr_ci_upper,
                "coefficient": coef,
                "std_error": se,
                "z_score": z_score,
                "p_value": p_value,
                "concordance_index": float(c_index) if not np.isnan(c_index) else None,
                "n_samples": int(df_cox.shape[0]),
                "n_events": int(df_cox["E"].sum()),
                "converged": True,
            }

        except Exception as e:
            logger.debug(f"Cox model failed for {protein_name}: {e}")
            return {
                "protein": protein_name,
                "protein_index": protein_index,
                "hazard_ratio": np.nan,
                "hr_ci_lower": np.nan,
                "hr_ci_upper": np.nan,
                "coefficient": np.nan,
                "std_error": np.nan,
                "z_score": np.nan,
                "p_value": np.nan,
                "concordance_index": None,
                "n_samples": int(valid_mask.sum()),
                "n_events": int(event[valid_mask].sum()),
                "converged": False,
            }

    def _create_ir_cox_regression(
        self,
        duration_col: str,
        event_col: str,
        covariates: Optional[List[str]],
        fdr_threshold: float,
        min_samples: int,
        penalizer: float,
    ) -> AnalysisStep:
        """Create IR for Cox regression analysis."""
        return AnalysisStep(
            operation="proteomics.survival.perform_cox_regression",
            tool_name="perform_cox_regression",
            description="Perform Cox proportional hazards regression to identify proteins associated with survival",
            library="lobster.services.analysis.proteomics_survival_service",
            code_template="""# Cox proportional hazards regression
from lobster.services.analysis.proteomics_survival_service import ProteomicsSurvivalService

service = ProteomicsSurvivalService()
adata_cox, stats, _ = service.perform_cox_regression(
    adata,
    duration_col={{ duration_col | tojson }},
    event_col={{ event_col | tojson }},
    covariates={{ covariates | tojson }},
    fdr_threshold={{ fdr_threshold }},
    min_samples={{ min_samples }},
    penalizer={{ penalizer }}
)
print(f"Significant proteins: {stats['n_significant_proteins']}")
print(f"Top hazard ratios: {stats['top_hazard_ratios'][:5]}")""",
            imports=[
                "from lobster.services.analysis.proteomics_survival_service import ProteomicsSurvivalService"
            ],
            parameters={
                "duration_col": duration_col,
                "event_col": event_col,
                "covariates": covariates,
                "fdr_threshold": fdr_threshold,
                "min_samples": min_samples,
                "penalizer": penalizer,
            },
            parameter_schema={
                "duration_col": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="PFS_days",
                    required=True,
                    description="Column in obs containing survival duration (days)",
                ),
                "event_col": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="PFS_event",
                    required=True,
                    description="Column in obs containing event indicator (1=event, 0=censored)",
                ),
                "covariates": ParameterSpec(
                    param_type="Optional[List[str]]",
                    papermill_injectable=True,
                    default_value=None,
                    required=False,
                    description="Additional covariates to adjust for (e.g., ['age', 'stage'])",
                ),
                "fdr_threshold": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.05,
                    required=False,
                    validation_rule="0 < fdr_threshold <= 1",
                    description="FDR threshold for significance",
                ),
                "min_samples": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=20,
                    required=False,
                    validation_rule="min_samples >= 10",
                    description="Minimum samples with valid survival data",
                ),
                "penalizer": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.1,
                    required=False,
                    validation_rule="penalizer >= 0",
                    description="L2 penalizer for Cox model regularization",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_cox"],
        )

    def _create_ir_kaplan_meier(
        self,
        duration_col: str,
        event_col: str,
        protein: str,
        stratify_method: str,
        n_groups: int,
    ) -> AnalysisStep:
        """Create IR for Kaplan-Meier analysis."""
        return AnalysisStep(
            operation="proteomics.survival.kaplan_meier_analysis",
            tool_name="kaplan_meier_analysis",
            description="Perform Kaplan-Meier survival analysis stratified by protein expression",
            library="lobster.services.analysis.proteomics_survival_service",
            code_template="""# Kaplan-Meier survival analysis
from lobster.services.analysis.proteomics_survival_service import ProteomicsSurvivalService

service = ProteomicsSurvivalService()
adata_km, stats, _ = service.kaplan_meier_analysis(
    adata,
    duration_col={{ duration_col | tojson }},
    event_col={{ event_col | tojson }},
    protein={{ protein | tojson }},
    stratify_method={{ stratify_method | tojson }},
    n_groups={{ n_groups }}
)
print(f"Log-rank p-value: {stats['log_rank_p_value']:.4f}")
print(f"Median survival by group: {stats['median_survival_by_group']}")""",
            imports=[
                "from lobster.services.analysis.proteomics_survival_service import ProteomicsSurvivalService"
            ],
            parameters={
                "duration_col": duration_col,
                "event_col": event_col,
                "protein": protein,
                "stratify_method": stratify_method,
                "n_groups": n_groups,
            },
            parameter_schema={
                "duration_col": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="PFS_days",
                    required=True,
                    description="Column in obs containing survival duration",
                ),
                "event_col": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="PFS_event",
                    required=True,
                    description="Column in obs containing event indicator",
                ),
                "protein": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="Protein to stratify by",
                ),
                "stratify_method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="median",
                    required=False,
                    validation_rule="stratify_method in ['median', 'tertile', 'quartile', 'optimal']",
                    description="Method to stratify patients by protein expression",
                ),
                "n_groups": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=2,
                    required=False,
                    validation_rule="n_groups >= 2",
                    description="Number of groups for stratification",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_km"],
        )

    def perform_cox_regression(
        self,
        adata: anndata.AnnData,
        duration_col: str = "PFS_days",
        event_col: str = "PFS_event",
        covariates: Optional[List[str]] = None,
        fdr_threshold: float = 0.05,
        min_samples: int = 20,
        penalizer: float = 0.1,
        n_jobs: int = 1,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform Cox proportional hazards regression across all proteins.

        Fits a Cox model for each protein individually, testing the association
        between protein expression and survival outcome. Adjusts for multiple
        testing using Benjamini-Hochberg FDR correction.

        Args:
            adata: AnnData object with proteomics data (samples × proteins)
            duration_col: Column in obs containing survival duration (days)
            event_col: Column in obs containing event indicator (1=event, 0=censored)
            covariates: Additional covariates to adjust for (e.g., ['age', 'stage'])
            fdr_threshold: FDR threshold for significance
            min_samples: Minimum samples with valid survival data
            penalizer: L2 penalizer for Cox model regularization (helps convergence)
            n_jobs: Number of parallel jobs for Cox fitting (1=sequential, >1=parallel)

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - AnnData with Cox results in var and uns
                - Analysis statistics dict
                - IR for notebook export

        Raises:
            ProteomicsSurvivalError: If analysis fails
        """
        try:
            logger.info("Starting Cox proportional hazards regression")

            # Check lifelines availability
            if not self._lifelines_available:
                raise ProteomicsSurvivalError(
                    "lifelines package not available. Install with: pip install lifelines"
                )

            from lifelines import CoxPHFitter
            from lifelines.utils import concordance_index

            # Create working copy
            adata_cox = adata.copy()
            original_shape = adata_cox.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            # Validate survival columns
            if duration_col not in adata_cox.obs.columns:
                raise ProteomicsSurvivalError(
                    f"Duration column '{duration_col}' not found in obs. "
                    f"Available columns: {list(adata_cox.obs.columns)}"
                )
            if event_col not in adata_cox.obs.columns:
                raise ProteomicsSurvivalError(
                    f"Event column '{event_col}' not found in obs. "
                    f"Available columns: {list(adata_cox.obs.columns)}"
                )

            # Get survival data
            duration = adata_cox.obs[duration_col].values
            event = adata_cox.obs[event_col].values

            # Identify valid samples (non-missing survival data)
            valid_mask = ~(np.isnan(duration) | np.isnan(event))
            valid_mask &= (duration > 0) & (event >= 0)
            n_valid = valid_mask.sum()

            logger.info(f"Valid samples with survival data: {n_valid}")

            if n_valid < min_samples:
                raise ProteomicsSurvivalError(
                    f"Insufficient valid samples: {n_valid} < {min_samples}"
                )

            # Get expression matrix
            X = adata_cox.X.copy()
            # Ensure protein names are strings (handles integer indices from AnnData without var)
            protein_names = [str(name) for name in adata_cox.var_names.tolist()]

            # Prepare covariate data as dictionary for helper method
            covariate_cols = []
            covariate_data: Dict[str, np.ndarray] = {}
            if covariates:
                for cov in covariates:
                    if cov in adata_cox.obs.columns:
                        covariate_cols.append(cov)
                        covariate_data[cov] = adata_cox.obs[cov].values
                    else:
                        logger.warning(f"Covariate '{cov}' not found in obs, skipping")

            # Run Cox regression for each protein
            cox_results = []
            n_proteins = len(protein_names)
            n_failed = 0
            n_low_variance = 0

            if n_jobs > 1:
                # Parallel execution using ThreadPoolExecutor
                logger.info(f"Running Cox regression on {n_proteins} proteins with {n_jobs} parallel jobs...")

                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    # Submit all protein fitting tasks
                    futures = {}
                    for i, protein_name in enumerate(protein_names):
                        protein_values = X[:, i]
                        future = executor.submit(
                            self._fit_single_protein_cox,
                            protein_index=i,
                            protein_name=protein_name,
                            protein_values=protein_values,
                            duration=duration,
                            event=event,
                            valid_mask=valid_mask,
                            covariate_data=covariate_data,
                            min_samples=min_samples,
                            penalizer=penalizer,
                        )
                        futures[future] = i

                    # Collect results as they complete
                    completed = 0
                    for future in as_completed(futures):
                        completed += 1
                        if completed % PROGRESS_LOG_INTERVAL == 0:
                            logger.info(f"Progress: {completed}/{n_proteins} proteins processed")

                        result = future.result()
                        if result is None:
                            n_low_variance += 1
                        elif not result.get("converged", True):
                            n_failed += 1
                            cox_results.append(result)
                        else:
                            cox_results.append(result)

                # Sort by protein index to maintain consistent order
                cox_results.sort(key=lambda x: x["protein_index"])

            else:
                # Sequential execution (original behavior)
                logger.info(f"Running Cox regression on {n_proteins} proteins...")

                for i, protein_name in enumerate(protein_names):
                    if i % PROGRESS_LOG_INTERVAL == 0 and i > 0:
                        logger.info(f"Progress: {i}/{n_proteins} proteins processed")

                    protein_values = X[:, i]
                    result = self._fit_single_protein_cox(
                        protein_index=i,
                        protein_name=protein_name,
                        protein_values=protein_values,
                        duration=duration,
                        event=event,
                        valid_mask=valid_mask,
                        covariate_data=covariate_data,
                        min_samples=min_samples,
                        penalizer=penalizer,
                    )

                    if result is None:
                        n_low_variance += 1
                    elif not result.get("converged", True):
                        n_failed += 1
                        cox_results.append(result)
                    else:
                        cox_results.append(result)

            logger.info(
                f"Cox regression completed: {len(cox_results)} tested, "
                f"{n_failed} failed, {n_low_variance} low variance"
            )

            # Apply FDR correction
            valid_results = [r for r in cox_results if not np.isnan(r["p_value"])]
            if valid_results:
                p_values = [r["p_value"] for r in valid_results]
                fdr_values = self._benjamini_hochberg(p_values)
                for i, result in enumerate(valid_results):
                    result["fdr"] = fdr_values[i]
                    result["significant"] = fdr_values[i] < fdr_threshold

            # Add FDR to failed results
            for result in cox_results:
                if "fdr" not in result:
                    result["fdr"] = np.nan
                    result["significant"] = False

            # Store results in AnnData
            results_df = pd.DataFrame(cox_results)
            results_df = results_df.set_index("protein")

            # Add to var (use string protein_names to match results_df index)
            # Reindex to include all proteins (missing ones get NaN)
            results_df = results_df.reindex(protein_names)
            for col in ["hazard_ratio", "p_value", "fdr", "significant", "z_score", "concordance_index"]:
                if col in results_df.columns:
                    adata_cox.var[f"cox_{col}"] = results_df[col].values

            # Store full results in uns
            adata_cox.uns["cox_regression"] = {
                "results": cox_results,
                "parameters": {
                    "duration_col": duration_col,
                    "event_col": event_col,
                    "covariates": covariate_cols,
                    "fdr_threshold": fdr_threshold,
                    "min_samples": min_samples,
                    "penalizer": penalizer,
                },
            }

            # Calculate summary statistics
            significant_results = [r for r in cox_results if r.get("significant", False)]
            top_hr = sorted(
                [r for r in valid_results],
                key=lambda x: x["hazard_ratio"],
                reverse=True,
            )[:10]

            analysis_stats = {
                "n_proteins_tested": len(cox_results),
                "n_proteins_converged": len(valid_results),
                "n_significant_proteins": len(significant_results),
                "n_failed_convergence": n_failed,
                "n_low_variance_skipped": n_low_variance,
                "fdr_threshold": fdr_threshold,
                "n_valid_samples": n_valid,
                "n_events": int(event[valid_mask].sum()),
                "median_survival_days": float(np.median(duration[valid_mask])),
                "top_hazard_ratios": [
                    {"protein": r["protein"], "hr": r["hazard_ratio"], "fdr": r["fdr"]}
                    for r in top_hr[:5]
                ],
                "significant_proteins": [r["protein"] for r in significant_results],
                "analysis_type": "cox_regression",
            }

            logger.info(
                f"Cox analysis complete: {len(significant_results)} significant proteins at FDR < {fdr_threshold}"
            )

            # Create IR for provenance
            ir = self._create_ir_cox_regression(
                duration_col,
                event_col,
                covariates,
                fdr_threshold,
                min_samples,
                penalizer,
            )

            return adata_cox, analysis_stats, ir

        except Exception as e:
            logger.exception(f"Error in Cox regression: {e}")
            raise ProteomicsSurvivalError(f"Cox regression failed: {str(e)}")

    def kaplan_meier_analysis(
        self,
        adata: anndata.AnnData,
        duration_col: str = "PFS_days",
        event_col: str = "PFS_event",
        protein: str = "",
        stratify_method: str = "median",
        n_groups: int = 2,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform Kaplan-Meier survival analysis stratified by protein expression.

        Stratifies patients by protein expression level and compares survival curves
        using the log-rank test.

        Args:
            adata: AnnData object with proteomics data
            duration_col: Column in obs containing survival duration
            event_col: Column in obs containing event indicator
            protein: Protein to stratify by (must be in var_names)
            stratify_method: Method to stratify ('median', 'tertile', 'quartile', 'optimal')
            n_groups: Number of groups for stratification (2 for median, 3 for tertile, etc.)

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - AnnData with KM results in uns
                - Analysis statistics with plot data
                - IR for notebook export

        Raises:
            ProteomicsSurvivalError: If analysis fails
        """
        try:
            logger.info(f"Starting Kaplan-Meier analysis for protein: {protein}")

            # Check lifelines availability
            if not self._lifelines_available:
                raise ProteomicsSurvivalError(
                    "lifelines package not available. Install with: pip install lifelines"
                )

            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test

            # Create working copy
            adata_km = adata.copy()

            # Validate protein
            if protein not in adata_km.var_names:
                raise ProteomicsSurvivalError(
                    f"Protein '{protein}' not found in var_names. "
                    f"Available: {list(adata_km.var_names[:10])}..."
                )

            # Validate survival columns
            if duration_col not in adata_km.obs.columns:
                raise ProteomicsSurvivalError(
                    f"Duration column '{duration_col}' not found in obs"
                )
            if event_col not in adata_km.obs.columns:
                raise ProteomicsSurvivalError(f"Event column '{event_col}' not found in obs")

            # Get data
            protein_idx = list(adata_km.var_names).index(protein)
            protein_values = adata_km.X[:, protein_idx]
            duration = adata_km.obs[duration_col].values
            event = adata_km.obs[event_col].values

            # Identify valid samples
            valid_mask = ~(
                np.isnan(duration) | np.isnan(event) | np.isnan(protein_values)
            )
            valid_mask &= (duration > 0) & (event >= 0)

            protein_valid = protein_values[valid_mask]
            duration_valid = duration[valid_mask]
            event_valid = event[valid_mask]

            logger.info(f"Valid samples: {len(protein_valid)}")

            # Stratify patients
            if stratify_method == "median":
                cutpoint = np.median(protein_valid)
                groups = (protein_valid > cutpoint).astype(int)
                group_labels = ["Low", "High"]
            elif stratify_method == "tertile":
                cutpoints = np.percentile(protein_valid, [33.3, 66.7])
                groups = np.digitize(protein_valid, cutpoints)
                group_labels = ["Low", "Medium", "High"]
                n_groups = 3
            elif stratify_method == "quartile":
                cutpoints = np.percentile(protein_valid, [25, 50, 75])
                groups = np.digitize(protein_valid, cutpoints)
                group_labels = ["Q1", "Q2", "Q3", "Q4"]
                n_groups = 4
            elif stratify_method == "optimal":
                # Find optimal cutpoint using maximally selected rank statistics
                cutpoint, best_p = self._find_optimal_cutpoint(
                    protein_valid, duration_valid, event_valid
                )
                groups = (protein_valid > cutpoint).astype(int)
                group_labels = ["Low", "High"]
                logger.info(
                    f"Optimal cutpoint: {cutpoint:.3f} (p={best_p:.4f})"
                )
            else:
                raise ProteomicsSurvivalError(
                    f"Unknown stratify_method: {stratify_method}"
                )

            # Fit Kaplan-Meier curves for each group
            kmf_results = {}
            survival_curves = {}

            for g in range(n_groups):
                group_mask = groups == g
                if group_mask.sum() < MIN_GROUP_SIZE:
                    logger.warning(
                        f"Group {group_labels[g]} has only {group_mask.sum()} samples"
                    )
                    continue

                kmf = KaplanMeierFitter()
                kmf.fit(
                    duration_valid[group_mask],
                    event_valid[group_mask],
                    label=group_labels[g],
                )

                kmf_results[group_labels[g]] = kmf

                # Store survival curve data for plotting
                survival_curves[group_labels[g]] = {
                    "timeline": kmf.timeline.tolist(),
                    "survival_function": kmf.survival_function_[group_labels[g]].tolist(),
                    "confidence_lower": kmf.confidence_interval_[
                        f"{group_labels[g]}_lower_0.95"
                    ].tolist(),
                    "confidence_upper": kmf.confidence_interval_[
                        f"{group_labels[g]}_upper_0.95"
                    ].tolist(),
                    "n_at_risk": kmf.event_table["at_risk"].tolist(),
                    "n_events": int(event_valid[group_mask].sum()),
                    "n_samples": int(group_mask.sum()),
                    "median_survival": float(kmf.median_survival_time_)
                    if not np.isinf(kmf.median_survival_time_)
                    else None,
                }

            # Perform log-rank test
            if len(kmf_results) >= 2:
                group_keys = list(kmf_results.keys())
                if n_groups == 2:
                    # Two-group log-rank test
                    lr_result = logrank_test(
                        duration_valid[groups == 0],
                        duration_valid[groups == 1],
                        event_valid[groups == 0],
                        event_valid[groups == 1],
                    )
                    log_rank_statistic = float(lr_result.test_statistic)
                    log_rank_p_value = float(lr_result.p_value)
                else:
                    # Multi-group log-rank test (pairwise)
                    from lifelines.statistics import multivariate_logrank_test

                    lr_result = multivariate_logrank_test(
                        duration_valid, groups, event_valid
                    )
                    log_rank_statistic = float(lr_result.test_statistic)
                    log_rank_p_value = float(lr_result.p_value)
            else:
                log_rank_statistic = np.nan
                log_rank_p_value = np.nan

            # Store results in AnnData
            adata_km.uns["kaplan_meier"] = {
                "protein": protein,
                "stratify_method": stratify_method,
                "n_groups": n_groups,
                "group_labels": group_labels[:n_groups],
                "survival_curves": survival_curves,
                "log_rank_statistic": log_rank_statistic,
                "log_rank_p_value": log_rank_p_value,
                "parameters": {
                    "duration_col": duration_col,
                    "event_col": event_col,
                    "protein": protein,
                    "stratify_method": stratify_method,
                    "n_groups": n_groups,
                },
            }

            # Calculate statistics
            median_survival_by_group = {
                label: data.get("median_survival")
                for label, data in survival_curves.items()
            }

            analysis_stats = {
                "protein": protein,
                "stratify_method": stratify_method,
                "n_groups": len(survival_curves),
                "log_rank_statistic": log_rank_statistic,
                "log_rank_p_value": log_rank_p_value,
                "significant": log_rank_p_value < DEFAULT_FDR_THRESHOLD if not np.isnan(log_rank_p_value) else False,
                "median_survival_by_group": median_survival_by_group,
                "n_samples_per_group": {
                    label: data["n_samples"] for label, data in survival_curves.items()
                },
                "n_events_per_group": {
                    label: data["n_events"] for label, data in survival_curves.items()
                },
                "survival_curves": survival_curves,
                "analysis_type": "kaplan_meier",
            }

            logger.info(
                f"Kaplan-Meier analysis complete: log-rank p={log_rank_p_value:.4f}"
            )

            # Create IR
            ir = self._create_ir_kaplan_meier(
                duration_col, event_col, protein, stratify_method, n_groups
            )

            return adata_km, analysis_stats, ir

        except Exception as e:
            logger.exception(f"Error in Kaplan-Meier analysis: {e}")
            raise ProteomicsSurvivalError(f"Kaplan-Meier analysis failed: {str(e)}")

    def batch_kaplan_meier(
        self,
        adata: anndata.AnnData,
        duration_col: str = "PFS_days",
        event_col: str = "PFS_event",
        proteins: Optional[List[str]] = None,
        stratify_method: str = "median",
        fdr_threshold: float = 0.05,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform Kaplan-Meier analysis across multiple proteins.

        Runs log-rank tests for all specified proteins and applies FDR correction.

        Args:
            adata: AnnData object
            duration_col: Column with survival duration
            event_col: Column with event indicator
            proteins: List of proteins to analyze (default: all)
            stratify_method: Stratification method
            fdr_threshold: FDR threshold for significance

        Returns:
            Tuple with AnnData, statistics, and IR
        """
        try:
            logger.info("Starting batch Kaplan-Meier analysis")

            if not self._lifelines_available:
                raise ProteomicsSurvivalError("lifelines package not available")

            from lifelines.statistics import logrank_test

            adata_batch = adata.copy()

            # Get proteins to analyze
            if proteins is None:
                proteins = adata_batch.var_names.tolist()

            # Get survival data
            duration = adata_batch.obs[duration_col].values
            event = adata_batch.obs[event_col].values
            valid_mask = ~(np.isnan(duration) | np.isnan(event))
            valid_mask &= (duration > 0) & (event >= 0)

            results = []
            for protein in proteins:
                try:
                    protein_idx = list(adata_batch.var_names).index(protein)
                    protein_values = adata_batch.X[:, protein_idx]

                    # Combine masks
                    prot_valid = valid_mask & ~np.isnan(protein_values)
                    if prot_valid.sum() < DEFAULT_MIN_SAMPLES:
                        continue

                    # Stratify by median
                    prot_vals = protein_values[prot_valid]
                    cutpoint = np.median(prot_vals)
                    high_mask = prot_vals > cutpoint

                    # Log-rank test
                    lr_result = logrank_test(
                        duration[prot_valid][~high_mask],
                        duration[prot_valid][high_mask],
                        event[prot_valid][~high_mask],
                        event[prot_valid][high_mask],
                    )

                    results.append(
                        {
                            "protein": protein,
                            "log_rank_statistic": float(lr_result.test_statistic),
                            "p_value": float(lr_result.p_value),
                            "n_samples": int(prot_valid.sum()),
                            "n_high": int(high_mask.sum()),
                            "n_low": int((~high_mask).sum()),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Failed for {protein}: {e}")

            # Apply FDR correction
            if results:
                p_values = [r["p_value"] for r in results]
                fdr_values = self._benjamini_hochberg(p_values)
                for i, result in enumerate(results):
                    result["fdr"] = fdr_values[i]
                    result["significant"] = fdr_values[i] < fdr_threshold

            # Store in AnnData
            adata_batch.uns["batch_kaplan_meier"] = {"results": results}

            significant = [r for r in results if r.get("significant", False)]

            stats = {
                "n_proteins_tested": len(results),
                "n_significant": len(significant),
                "fdr_threshold": fdr_threshold,
                "significant_proteins": [r["protein"] for r in significant],
                "analysis_type": "batch_kaplan_meier",
            }

            # Simple IR
            ir = AnalysisStep(
                operation="proteomics.survival.batch_kaplan_meier",
                tool_name="batch_kaplan_meier",
                description="Batch Kaplan-Meier analysis across multiple proteins",
                library="lobster.services.analysis.proteomics_survival_service",
                code_template="# See perform_cox_regression for full analysis",
                imports=[],
                parameters={},
                parameter_schema={},
                input_entities=["adata"],
                output_entities=["adata_batch"],
            )

            return adata_batch, stats, ir

        except Exception as e:
            raise ProteomicsSurvivalError(f"Batch KM analysis failed: {str(e)}")

    def _benjamini_hochberg(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction.

        Delegates to shared implementation in lobster.utils.statistics.
        """
        return benjamini_hochberg(p_values)

    def _find_optimal_cutpoint(
        self,
        protein_values: np.ndarray,
        duration: np.ndarray,
        event: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Find optimal cutpoint using maximally selected rank statistics.

        Tests multiple cutpoints and returns the one with the most significant
        log-rank test (with proper multiple testing adjustment).
        """
        from lifelines.statistics import logrank_test

        # Test cutpoints from 25th to 75th percentile
        percentiles = np.arange(25, 76, 5)
        cutpoints = np.percentile(protein_values, percentiles)

        best_p = 1.0
        best_cutpoint = np.median(protein_values)

        for cutpoint in cutpoints:
            high_mask = protein_values > cutpoint

            # Skip if groups too small
            if high_mask.sum() < MIN_CUTPOINT_GROUP_SIZE or (~high_mask).sum() < MIN_CUTPOINT_GROUP_SIZE:
                continue

            try:
                lr_result = logrank_test(
                    duration[~high_mask],
                    duration[high_mask],
                    event[~high_mask],
                    event[high_mask],
                )

                if lr_result.p_value < best_p:
                    best_p = lr_result.p_value
                    best_cutpoint = cutpoint

            except Exception:
                continue

        # Adjust for multiple testing (conservative Bonferroni)
        n_tests = len(cutpoints)
        adjusted_p = min(best_p * n_tests, 1.0)

        return best_cutpoint, adjusted_p
