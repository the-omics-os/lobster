"""
Metabolomics analysis service for univariate statistics, multivariate analysis,
and fold change computation.

This service implements statistical analysis methods for metabolomics data including
t-tests, ANOVA, Kruskal-Wallis with FDR correction, PCA, PLS-DA with VIP scores,
OPLS-DA (custom NIPALS implementation), and fold change calculations.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

from typing import Any, Dict, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class MetabolomicsAnalysisError(Exception):
    """Base exception for metabolomics analysis operations."""

    pass


class MetabolomicsAnalysisService:
    """
    Stateless analysis service for metabolomics data.

    Provides univariate statistics, multivariate analysis (PCA, PLS-DA, OPLS-DA),
    and fold change computation for metabolomics datasets.
    """

    def __init__(self):
        """Initialize the metabolomics analysis service (stateless)."""
        logger.debug("Initializing stateless MetabolomicsAnalysisService")

    # =========================================================================
    # IR creation helpers
    # =========================================================================

    def _create_ir_univariate(
        self,
        group_column: str,
        method: str,
        fdr_method: str,
    ) -> AnalysisStep:
        """Create IR for univariate statistics."""
        return AnalysisStep(
            operation="metabolomics.analysis.run_univariate_statistics",
            tool_name="run_metabolomics_statistics",
            description="Run univariate statistics with FDR correction",
            library="scipy/statsmodels",
            code_template="""# Metabolomics univariate statistics
from lobster.services.analysis.metabolomics_analysis_service import MetabolomicsAnalysisService

service = MetabolomicsAnalysisService()
adata_stats, stats, _ = service.run_univariate_statistics(
    adata,
    group_column={{ group_column | tojson }},
    method={{ method | tojson }},
    fdr_method={{ fdr_method | tojson }}
)
print(f"Tested {stats['n_tested']} features: {stats['n_significant_fdr']} significant (FDR < 0.05)")""",
            imports=[
                "from lobster.services.analysis.metabolomics_analysis_service import MetabolomicsAnalysisService"
            ],
            parameters={
                "group_column": group_column,
                "method": method,
                "fdr_method": fdr_method,
            },
            parameter_schema={
                "group_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="condition",
                    required=True,
                    description="Column in obs containing group labels",
                ),
                "method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="auto",
                    required=False,
                    validation_rule="method in ['auto', 'ttest', 'wilcoxon', 'anova', 'kruskal']",
                    description="Statistical test method",
                ),
                "fdr_method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="fdr_bh",
                    required=False,
                    description="FDR correction method",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_stats"],
        )

    def _create_ir_pca(self, n_components: int) -> AnalysisStep:
        """Create IR for PCA."""
        return AnalysisStep(
            operation="metabolomics.analysis.run_pca",
            tool_name="run_multivariate_analysis",
            description="Run PCA on metabolomics data",
            library="sklearn",
            code_template="""# PCA analysis
from lobster.services.analysis.metabolomics_analysis_service import MetabolomicsAnalysisService

service = MetabolomicsAnalysisService()
adata_pca, stats, _ = service.run_pca(adata, n_components={{ n_components }})
print(f"PCA: {stats['total_variance_explained']:.1f}% variance explained by {stats['n_components']} components")""",
            imports=[
                "from lobster.services.analysis.metabolomics_analysis_service import MetabolomicsAnalysisService"
            ],
            parameters={"n_components": n_components},
            parameter_schema={
                "n_components": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=10,
                    required=False,
                    validation_rule="n_components > 0",
                    description="Number of PCA components",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_pca"],
        )

    def _create_ir_pls_da(
        self,
        group_column: str,
        n_components: int,
        permutation_test: bool,
        n_permutations: int,
    ) -> AnalysisStep:
        """Create IR for PLS-DA."""
        return AnalysisStep(
            operation="metabolomics.analysis.run_pls_da",
            tool_name="run_multivariate_analysis",
            description="Run PLS-DA with VIP scores and optional permutation test",
            library="sklearn",
            code_template="""# PLS-DA analysis
from lobster.services.analysis.metabolomics_analysis_service import MetabolomicsAnalysisService

service = MetabolomicsAnalysisService()
adata_plsda, stats, _ = service.run_pls_da(
    adata,
    group_column={{ group_column | tojson }},
    n_components={{ n_components }},
    permutation_test={{ permutation_test | tojson }},
    n_permutations={{ n_permutations }}
)
print(f"PLS-DA: R2={stats['r2']:.3f}, Q2={stats['q2']:.3f}, VIP>1: {stats['vip_gt_1_count']}")""",
            imports=[
                "from lobster.services.analysis.metabolomics_analysis_service import MetabolomicsAnalysisService"
            ],
            parameters={
                "group_column": group_column,
                "n_components": n_components,
                "permutation_test": permutation_test,
                "n_permutations": n_permutations,
            },
            parameter_schema={
                "group_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="condition",
                    required=True,
                    description="Column in obs containing group labels",
                ),
                "n_components": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=2,
                    required=False,
                    description="Number of PLS components",
                ),
                "permutation_test": ParameterSpec(
                    param_type="bool",
                    papermill_injectable=True,
                    default_value=True,
                    required=False,
                    description="Run permutation test for model validation",
                ),
                "n_permutations": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=100,
                    required=False,
                    description="Number of permutations",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_plsda"],
        )

    def _create_ir_opls_da(
        self,
        group_column: str,
        n_orthogonal: int,
        n_predictive: int,
        permutation_test: bool,
        n_permutations: int,
    ) -> AnalysisStep:
        """Create IR for OPLS-DA."""
        return AnalysisStep(
            operation="metabolomics.analysis.run_opls_da",
            tool_name="run_multivariate_analysis",
            description="Run OPLS-DA separating predictive from orthogonal variation",
            library="sklearn/numpy",
            code_template="""# OPLS-DA analysis
from lobster.services.analysis.metabolomics_analysis_service import MetabolomicsAnalysisService

service = MetabolomicsAnalysisService()
adata_oplsda, stats, _ = service.run_opls_da(
    adata,
    group_column={{ group_column | tojson }},
    n_orthogonal={{ n_orthogonal }},
    n_predictive={{ n_predictive }},
    permutation_test={{ permutation_test | tojson }},
    n_permutations={{ n_permutations }}
)
print(f"OPLS-DA: R2={stats['r2']:.3f}, Q2={stats['q2']:.3f}")""",
            imports=[
                "from lobster.services.analysis.metabolomics_analysis_service import MetabolomicsAnalysisService"
            ],
            parameters={
                "group_column": group_column,
                "n_orthogonal": n_orthogonal,
                "n_predictive": n_predictive,
                "permutation_test": permutation_test,
                "n_permutations": n_permutations,
            },
            parameter_schema={
                "group_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="condition",
                    required=True,
                    description="Column in obs containing group labels",
                ),
                "n_orthogonal": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=1,
                    required=False,
                    description="Number of orthogonal components",
                ),
                "n_predictive": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=1,
                    required=False,
                    description="Number of predictive components",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_oplsda"],
        )

    def _create_ir_fold_changes(
        self,
        group_column: str,
        reference_group: Optional[str],
        log_space: bool,
    ) -> AnalysisStep:
        """Create IR for fold change calculation."""
        return AnalysisStep(
            operation="metabolomics.analysis.calculate_fold_changes",
            tool_name="run_metabolomics_statistics",
            description="Calculate log2 fold changes between groups",
            library="numpy",
            code_template="""# Fold change calculation
from lobster.services.analysis.metabolomics_analysis_service import MetabolomicsAnalysisService

service = MetabolomicsAnalysisService()
adata_fc, stats, _ = service.calculate_fold_changes(
    adata,
    group_column={{ group_column | tojson }},
    reference_group={{ reference_group | tojson }},
    log_space={{ log_space | tojson }}
)
print(f"FC: {stats['n_upregulated']} up, {stats['n_downregulated']} down (vs {stats['reference']})")""",
            imports=[
                "from lobster.services.analysis.metabolomics_analysis_service import MetabolomicsAnalysisService"
            ],
            parameters={
                "group_column": group_column,
                "reference_group": reference_group,
                "log_space": log_space,
            },
            parameter_schema={
                "group_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="condition",
                    required=True,
                    description="Column in obs with group labels",
                ),
                "reference_group": ParameterSpec(
                    param_type="Optional[str]",
                    papermill_injectable=True,
                    default_value=None,
                    required=False,
                    description="Reference group (default: first alphabetically)",
                ),
                "log_space": ParameterSpec(
                    param_type="bool",
                    papermill_injectable=True,
                    default_value=True,
                    required=False,
                    description="Whether data is already log-transformed",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_fc"],
        )

    # =========================================================================
    # Public methods
    # =========================================================================

    def run_univariate_statistics(
        self,
        adata: anndata.AnnData,
        group_column: str,
        method: str = "auto",
        fdr_method: str = "fdr_bh",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Run univariate statistical tests with FDR correction.

        Args:
            adata: AnnData object with metabolomics data
            group_column: Column in obs containing group labels
            method: Statistical test:
                - "auto": Wilcoxon/Kruskal-Wallis (non-parametric, default for metabolomics)
                - "ttest": Independent t-test (2 groups)
                - "wilcoxon": Mann-Whitney U test (2 groups)
                - "anova": One-way ANOVA (3+ groups)
                - "kruskal": Kruskal-Wallis test (3+ groups)
            fdr_method: Method for multipletests (default "fdr_bh")

        Returns:
            Tuple of (AnnData with stats in var, stats dict, AnalysisStep)
        """
        try:
            logger.info(f"Starting univariate statistics: method={method}")

            if group_column not in adata.obs.columns:
                raise MetabolomicsAnalysisError(
                    f"Group column '{group_column}' not found in obs"
                )

            adata_stats = adata.copy()

            if hasattr(adata_stats.X, "toarray"):
                X = adata_stats.X.toarray().astype(np.float64)
            else:
                X = np.array(adata_stats.X, dtype=np.float64)

            groups = adata_stats.obs[group_column].values
            unique_groups = sorted(set(groups))
            n_groups = len(unique_groups)

            if n_groups < 2:
                raise MetabolomicsAnalysisError(
                    f"Need at least 2 groups, found {n_groups}"
                )

            # Auto-select method
            if method == "auto":
                if n_groups == 2:
                    method = "wilcoxon"
                else:
                    method = "kruskal"
                logger.info(f"Auto-selected method: {method} for {n_groups} groups")

            # Validate method vs number of groups
            if method in ("ttest", "wilcoxon") and n_groups != 2:
                raise MetabolomicsAnalysisError(
                    f"{method} requires exactly 2 groups, found {n_groups}"
                )
            if method in ("anova", "kruskal") and n_groups < 2:
                raise MetabolomicsAnalysisError(
                    f"{method} requires at least 2 groups, found {n_groups}"
                )

            # Run tests per feature
            p_values = np.full(adata_stats.n_vars, np.nan)
            test_statistics = np.full(adata_stats.n_vars, np.nan)

            for j in range(adata_stats.n_vars):
                feature_data = X[:, j]
                group_data = []
                for g in unique_groups:
                    g_vals = feature_data[groups == g]
                    g_vals = g_vals[~np.isnan(g_vals)]
                    group_data.append(g_vals)

                # Need at least 2 observations per group
                if any(len(gd) < 2 for gd in group_data):
                    continue

                try:
                    if method == "ttest":
                        stat, pval = scipy_stats.ttest_ind(
                            group_data[0], group_data[1], equal_var=False
                        )
                    elif method == "wilcoxon":
                        stat, pval = scipy_stats.mannwhitneyu(
                            group_data[0], group_data[1], alternative="two-sided"
                        )
                    elif method == "anova":
                        stat, pval = scipy_stats.f_oneway(*group_data)
                    elif method == "kruskal":
                        stat, pval = scipy_stats.kruskal(*group_data)
                    else:
                        raise MetabolomicsAnalysisError(f"Unknown method: {method}")

                    p_values[j] = pval
                    test_statistics[j] = stat
                except Exception:
                    continue

            # FDR correction
            from statsmodels.stats.multitest import multipletests

            valid_mask = ~np.isnan(p_values)
            fdr_values = np.full(adata_stats.n_vars, np.nan)
            if valid_mask.sum() > 0:
                _, fdr_corrected, _, _ = multipletests(
                    p_values[valid_mask], method=fdr_method
                )
                fdr_values[valid_mask] = fdr_corrected

            # Store results in var
            adata_stats.var["p_value"] = p_values
            adata_stats.var["fdr"] = fdr_values
            adata_stats.var["test_statistic"] = test_statistics
            adata_stats.var["significant"] = fdr_values < 0.05

            n_tested = int(valid_mask.sum())
            n_sig_raw = int((p_values[valid_mask] < 0.05).sum()) if n_tested > 0 else 0
            n_sig_fdr = (
                int((fdr_values[valid_mask] < 0.05).sum()) if n_tested > 0 else 0
            )

            stats = {
                "n_tested": n_tested,
                "n_significant_raw": n_sig_raw,
                "n_significant_fdr": n_sig_fdr,
                "method_used": method,
                "fdr_method": fdr_method,
                "groups": unique_groups,
                "n_groups": n_groups,
                "analysis_type": "metabolomics_univariate_statistics",
            }

            logger.info(
                f"Univariate statistics complete: {n_tested} tested, "
                f"{n_sig_fdr} significant (FDR < 0.05)"
            )

            ir = self._create_ir_univariate(group_column, method, fdr_method)
            return adata_stats, stats, ir

        except Exception as e:
            logger.exception(f"Error in univariate statistics: {e}")
            raise MetabolomicsAnalysisError(f"Univariate statistics failed: {str(e)}")

    def run_pca(
        self,
        adata: anndata.AnnData,
        n_components: int = 10,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Run PCA on metabolomics data.

        Args:
            adata: AnnData object with metabolomics data (imputed recommended)
            n_components: Number of PCA components to compute

        Returns:
            Tuple of (AnnData with PCA results, stats dict, AnalysisStep)
        """
        try:
            logger.info(f"Starting PCA with {n_components} components")
            adata_pca = adata.copy()

            if hasattr(adata_pca.X, "toarray"):
                X = adata_pca.X.toarray().astype(np.float64)
            else:
                X = np.array(adata_pca.X, dtype=np.float64)

            # Impute NaN for PCA (mean imputation as fallback)
            from sklearn.impute import SimpleImputer

            if np.isnan(X).any():
                imputer = SimpleImputer(strategy="mean")
                X = imputer.fit_transform(X)

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # PCA
            n_comp = min(n_components, min(X_scaled.shape) - 1)
            pca = PCA(n_components=n_comp)
            scores = pca.fit_transform(X_scaled)

            # Store results
            adata_pca.obsm["X_pca"] = scores
            adata_pca.varm["PCs"] = pca.components_.T
            adata_pca.uns["pca_variance_ratio"] = pca.explained_variance_ratio_.tolist()

            variance_per_component = [
                float(v * 100) for v in pca.explained_variance_ratio_
            ]
            total_var = float(pca.explained_variance_ratio_.sum() * 100)

            stats = {
                "n_components": n_comp,
                "variance_explained": variance_per_component,
                "total_variance_explained": total_var,
                "analysis_type": "metabolomics_pca",
            }

            logger.info(
                f"PCA complete: {total_var:.1f}% variance explained by {n_comp} components"
            )

            ir = self._create_ir_pca(n_components)
            return adata_pca, stats, ir

        except Exception as e:
            logger.exception(f"Error in PCA: {e}")
            raise MetabolomicsAnalysisError(f"PCA failed: {str(e)}")

    def run_pls_da(
        self,
        adata: anndata.AnnData,
        group_column: str,
        n_components: int = 2,
        permutation_test: bool = True,
        n_permutations: int = 100,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Run PLS-DA using sklearn PLSRegression with VIP scores.

        Args:
            adata: AnnData object with metabolomics data
            group_column: Column in obs containing group labels
            n_components: Number of PLS components
            permutation_test: Whether to run permutation test for model validation
            n_permutations: Number of permutations for validation

        Returns:
            Tuple of (AnnData with PLS-DA results, stats dict, AnalysisStep)
        """
        try:
            logger.info(f"Starting PLS-DA: {n_components} components")

            if group_column not in adata.obs.columns:
                raise MetabolomicsAnalysisError(
                    f"Group column '{group_column}' not found in obs"
                )

            adata_plsda = adata.copy()

            if hasattr(adata_plsda.X, "toarray"):
                X = adata_plsda.X.toarray().astype(np.float64)
            else:
                X = np.array(adata_plsda.X, dtype=np.float64)

            # Impute NaN for PLS-DA
            from sklearn.impute import SimpleImputer

            if np.isnan(X).any():
                imputer = SimpleImputer(strategy="mean")
                X = imputer.fit_transform(X)

            y_labels = adata_plsda.obs[group_column].values

            # Encode class labels
            lb = LabelBinarizer()
            Y = lb.fit_transform(y_labels).astype(float)
            if Y.shape[1] == 1:  # Binary case: expand to 2 columns
                Y = np.hstack([1 - Y, Y])

            # Fit PLS
            n_comp = min(n_components, min(X.shape) - 1, Y.shape[1])
            pls = PLSRegression(n_components=n_comp, scale=True)
            pls.fit(X, Y)

            # Scores
            T = pls.x_scores_

            # VIP scores
            W = pls.x_weights_
            Q = pls.y_loadings_
            p = X.shape[1]

            # SS = diag(T'T * Q'Q) for each component
            SS = np.diag(T.T @ T @ Q.T @ Q)
            with np.errstate(divide="ignore", invalid="ignore"):
                vip = np.sqrt(p * np.sum(SS * W**2, axis=1) / np.sum(SS))
            vip = np.nan_to_num(vip, nan=0.0)

            # R2 and Q2 via cross-validation
            from sklearn.model_selection import cross_val_predict

            Y_pred_cv = cross_val_predict(
                PLSRegression(n_components=n_comp, scale=True), X, Y, cv=7
            )
            ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
            ss_res_cv = np.sum((Y - Y_pred_cv) ** 2)
            Y_pred = pls.predict(X)
            ss_res = np.sum((Y - Y_pred) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            q2 = float(1 - ss_res_cv / ss_tot) if ss_tot > 0 else 0.0

            # Store results
            adata_plsda.obsm["X_plsda"] = T
            adata_plsda.var["vip_score"] = vip
            adata_plsda.uns["plsda_r2"] = r2
            adata_plsda.uns["plsda_q2"] = q2

            # Permutation test
            permutation_p_value = None
            if permutation_test:
                n_better = 0
                rng = np.random.RandomState(42)
                for _ in range(n_permutations):
                    y_perm = rng.permutation(y_labels)
                    Y_perm = lb.transform(y_perm).astype(float)
                    if Y_perm.shape[1] == 1:
                        Y_perm = np.hstack([1 - Y_perm, Y_perm])
                    pls_perm = PLSRegression(n_components=n_comp, scale=True)
                    pls_perm.fit(X, Y_perm)
                    Y_pred_perm = pls_perm.predict(X)
                    ss_res_perm = np.sum((Y_perm - Y_pred_perm) ** 2)
                    ss_tot_perm = np.sum((Y_perm - Y_perm.mean(axis=0)) ** 2)
                    r2_perm = 1 - ss_res_perm / ss_tot_perm if ss_tot_perm > 0 else 0
                    if r2_perm >= r2:
                        n_better += 1
                permutation_p_value = float((n_better + 1) / (n_permutations + 1))
                adata_plsda.uns["plsda_permutation_p"] = permutation_p_value

            vip_gt_1 = int((vip > 1.0).sum())

            stats = {
                "n_components": n_comp,
                "r2": r2,
                "q2": q2,
                "vip_gt_1_count": vip_gt_1,
                "permutation_p_value": permutation_p_value,
                "analysis_type": "metabolomics_pls_da",
            }

            logger.info(f"PLS-DA complete: R2={r2:.3f}, Q2={q2:.3f}, VIP>1: {vip_gt_1}")

            ir = self._create_ir_pls_da(
                group_column, n_components, permutation_test, n_permutations
            )
            return adata_plsda, stats, ir

        except Exception as e:
            logger.exception(f"Error in PLS-DA: {e}")
            raise MetabolomicsAnalysisError(f"PLS-DA failed: {str(e)}")

    def run_opls_da(
        self,
        adata: anndata.AnnData,
        group_column: str,
        n_orthogonal: int = 1,
        n_predictive: int = 1,
        permutation_test: bool = True,
        n_permutations: int = 100,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Run OPLS-DA separating predictive from orthogonal variation.

        Custom NIPALS-based implementation (~100 lines). Separates biological
        variation (predictive) from systematic noise (orthogonal).

        Args:
            adata: AnnData object with metabolomics data
            group_column: Column in obs containing group labels
            n_orthogonal: Number of orthogonal components (default 1)
            n_predictive: Number of predictive components (default 1)
            permutation_test: Run permutation test for validation
            n_permutations: Number of permutations

        Returns:
            Tuple of (AnnData with OPLS-DA results, stats dict, AnalysisStep)
        """
        try:
            logger.info(
                f"Starting OPLS-DA: {n_predictive} predictive, {n_orthogonal} orthogonal"
            )

            if group_column not in adata.obs.columns:
                raise MetabolomicsAnalysisError(
                    f"Group column '{group_column}' not found in obs"
                )

            adata_opls = adata.copy()

            if hasattr(adata_opls.X, "toarray"):
                X_raw = adata_opls.X.toarray().astype(np.float64)
            else:
                X_raw = np.array(adata_opls.X, dtype=np.float64)

            # Impute NaN
            from sklearn.impute import SimpleImputer

            if np.isnan(X_raw).any():
                imputer = SimpleImputer(strategy="mean")
                X_raw = imputer.fit_transform(X_raw)

            y_labels = adata_opls.obs[group_column].values

            # Run OPLS-DA
            result = self._opls_da_nipals(X_raw, y_labels, n_orthogonal, n_predictive)

            # Store results
            adata_opls.obsm["X_oplsda_pred"] = result["predictive_scores"]
            if result["orthogonal_scores"] is not None:
                adata_opls.obsm["X_oplsda_orth"] = result["orthogonal_scores"]

            r2 = result["r2"]
            q2 = result["q2"]

            # Warn about overfitting
            if q2 < 0.5:
                logger.warning(
                    f"Q2 = {q2:.3f} < 0.5: model may have poor predictive ability"
                )
            if r2 - q2 > 0.3:
                logger.warning(
                    f"R2-Q2 gap = {r2 - q2:.3f} > 0.3: potential overfitting"
                )

            adata_opls.uns["oplsda_r2"] = r2
            adata_opls.uns["oplsda_q2"] = q2

            # Permutation test
            permutation_p_value = None
            if permutation_test:
                n_better = 0
                rng = np.random.RandomState(42)
                for _ in range(n_permutations):
                    y_perm = rng.permutation(y_labels)
                    try:
                        result_perm = self._opls_da_nipals(
                            X_raw, y_perm, n_orthogonal, n_predictive
                        )
                        if result_perm["r2"] >= r2:
                            n_better += 1
                    except Exception:
                        continue
                permutation_p_value = float((n_better + 1) / (n_permutations + 1))
                adata_opls.uns["oplsda_permutation_p"] = permutation_p_value

            stats = {
                "r2": r2,
                "q2": q2,
                "n_orthogonal": n_orthogonal,
                "n_predictive": n_predictive,
                "permutation_p_value": permutation_p_value,
                "analysis_type": "metabolomics_opls_da",
            }

            logger.info(f"OPLS-DA complete: R2={r2:.3f}, Q2={q2:.3f}")

            ir = self._create_ir_opls_da(
                group_column,
                n_orthogonal,
                n_predictive,
                permutation_test,
                n_permutations,
            )
            return adata_opls, stats, ir

        except Exception as e:
            logger.exception(f"Error in OPLS-DA: {e}")
            raise MetabolomicsAnalysisError(f"OPLS-DA failed: {str(e)}")

    def calculate_fold_changes(
        self,
        adata: anndata.AnnData,
        group_column: str,
        reference_group: Optional[str] = None,
        log_space: bool = True,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Calculate log2 fold changes between groups.

        Args:
            adata: AnnData object with metabolomics data
            group_column: Column in obs containing group labels
            reference_group: Reference group (default: first alphabetically)
            log_space: Whether data is already log-transformed

        Returns:
            Tuple of (AnnData with fold changes in var, stats dict, AnalysisStep)
        """
        try:
            logger.info("Starting fold change calculation")

            if group_column not in adata.obs.columns:
                raise MetabolomicsAnalysisError(
                    f"Group column '{group_column}' not found in obs"
                )

            adata_fc = adata.copy()

            if hasattr(adata_fc.X, "toarray"):
                X = adata_fc.X.toarray().astype(np.float64)
            else:
                X = np.array(adata_fc.X, dtype=np.float64)

            groups = adata_fc.obs[group_column].values
            unique_groups = sorted(set(groups))

            if len(unique_groups) < 2:
                raise MetabolomicsAnalysisError(
                    "Need at least 2 groups for fold change"
                )

            # Select reference group
            if reference_group is None:
                reference_group = unique_groups[0]
            elif reference_group not in unique_groups:
                raise MetabolomicsAnalysisError(
                    f"Reference group '{reference_group}' not found. "
                    f"Available: {unique_groups}"
                )

            # Get comparison group(s)
            comparison_groups = [g for g in unique_groups if g != reference_group]
            if len(comparison_groups) == 1:
                comparison = comparison_groups[0]
            else:
                # Multi-group: compare each against reference
                comparison = ",".join(comparison_groups)

            # Calculate means
            ref_mask = groups == reference_group
            ref_means = np.nanmean(X[ref_mask], axis=0)

            # For simplicity, compare first non-reference group (most common use case)
            comp_group = comparison_groups[0]
            comp_mask = groups == comp_group
            comp_means = np.nanmean(X[comp_mask], axis=0)

            # Calculate fold changes
            if log_space:
                # Data is log-transformed: difference = log ratio
                log2_fc = comp_means - ref_means
            else:
                # Raw intensity: compute ratio then log2
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(ref_means > 0, comp_means / ref_means, np.nan)
                    log2_fc = np.log2(np.where(ratio > 0, ratio, np.nan))

            adata_fc.var["log2_fold_change"] = log2_fc
            adata_fc.var["mean_group"] = comp_means
            adata_fc.var["mean_reference"] = ref_means

            n_up = int(np.nansum(log2_fc > 1))
            n_down = int(np.nansum(log2_fc < -1))

            stats = {
                "reference": reference_group,
                "comparison": comp_group,
                "n_upregulated": n_up,
                "n_downregulated": n_down,
                "analysis_type": "metabolomics_fold_change",
            }

            logger.info(
                f"Fold change complete: {n_up} up, {n_down} down "
                f"({comp_group} vs {reference_group})"
            )

            ir = self._create_ir_fold_changes(group_column, reference_group, log_space)
            return adata_fc, stats, ir

        except Exception as e:
            logger.exception(f"Error in fold change calculation: {e}")
            raise MetabolomicsAnalysisError(f"Fold change failed: {str(e)}")

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _opls_da_nipals(
        self,
        X_raw: np.ndarray,
        y_labels: np.ndarray,
        n_orthogonal: int,
        n_predictive: int,
    ) -> Dict[str, Any]:
        """
        OPLS-DA via NIPALS algorithm.

        Separates predictive from orthogonal variation in X with respect to Y.
        """
        # Encode Y
        lb = LabelBinarizer()
        Y = lb.fit_transform(y_labels).astype(float)
        if Y.shape[1] == 1:
            Y = np.hstack([1 - Y, Y])

        # Center and scale
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_x.fit_transform(X_raw)
        Y_scaled = scaler_y.fit_transform(Y)

        # Extract orthogonal components via NIPALS
        X_filtered = X_scaled.copy()
        orthogonal_scores_list = []
        orthogonal_loadings_list = []

        for _ in range(n_orthogonal):
            # PLS weight vector
            w = X_filtered.T @ Y_scaled
            w = w[:, 0]
            w_norm = np.linalg.norm(w)
            if w_norm < 1e-10:
                break
            w = w / w_norm

            # Predictive score
            t = X_filtered @ w

            # Predictive loading
            t_dot = t.T @ t
            if abs(t_dot) < 1e-10:
                break
            p = X_filtered.T @ t / t_dot

            # Orthogonal weight
            w_dot = w.T @ w
            w_orth = p - (w.T @ p / w_dot) * w if abs(w_dot) > 1e-10 else p
            w_orth_norm = np.linalg.norm(w_orth)
            if w_orth_norm < 1e-10:
                break
            w_orth = w_orth / w_orth_norm

            # Orthogonal score and loading
            t_orth = X_filtered @ w_orth
            t_orth_dot = t_orth.T @ t_orth
            if abs(t_orth_dot) < 1e-10:
                break
            p_orth = X_filtered.T @ t_orth / t_orth_dot

            # Remove orthogonal variation
            X_filtered = X_filtered - np.outer(t_orth, p_orth)
            orthogonal_scores_list.append(t_orth)
            orthogonal_loadings_list.append(p_orth)

        # Final predictive PLS on filtered X
        n_pred = min(n_predictive, min(X_filtered.shape) - 1, Y_scaled.shape[1])
        pls = PLSRegression(n_components=n_pred, scale=False)
        pls.fit(X_filtered, Y_scaled)

        predictive_scores = pls.x_scores_
        orthogonal_scores = (
            np.column_stack(orthogonal_scores_list) if orthogonal_scores_list else None
        )

        # R2
        Y_pred = pls.predict(X_filtered)
        ss_tot = np.sum((Y_scaled - Y_scaled.mean(axis=0)) ** 2)
        ss_res = np.sum((Y_scaled - Y_pred) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Q2 via 7-fold CV
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=min(7, X_raw.shape[0]), shuffle=True, random_state=42)
        press = 0.0
        for train_idx, test_idx in kf.split(X_raw):
            X_train = scaler_x.transform(X_raw[train_idx])
            X_test = scaler_x.transform(X_raw[test_idx])
            y_train = y_labels[train_idx]
            y_test_labels = y_labels[test_idx]

            Y_train = lb.transform(y_train).astype(float)
            if Y_train.shape[1] == 1:
                Y_train = np.hstack([1 - Y_train, Y_train])
            Y_test = lb.transform(y_test_labels).astype(float)
            if Y_test.shape[1] == 1:
                Y_test = np.hstack([1 - Y_test, Y_test])

            Y_train_scaled = scaler_y.transform(Y_train)
            Y_test_scaled = scaler_y.transform(Y_test)

            # Remove orthogonal from train and test
            X_train_f = X_train.copy()
            X_test_f = X_test.copy()
            for p_orth in orthogonal_loadings_list:
                t_orth_train = X_train_f @ (p_orth / (p_orth.T @ p_orth + 1e-10))
                X_train_f = X_train_f - np.outer(t_orth_train, p_orth)
                t_orth_test = X_test_f @ (p_orth / (p_orth.T @ p_orth + 1e-10))
                X_test_f = X_test_f - np.outer(t_orth_test, p_orth)

            try:
                pls_cv = PLSRegression(n_components=n_pred, scale=False)
                pls_cv.fit(X_train_f, Y_train_scaled)
                Y_pred_test = pls_cv.predict(X_test_f)
                press += np.sum((Y_test_scaled - Y_pred_test) ** 2)
            except Exception:
                press += np.sum(Y_test_scaled**2)

        q2 = float(1 - press / ss_tot) if ss_tot > 0 else 0.0

        return {
            "predictive_scores": predictive_scores,
            "orthogonal_scores": orthogonal_scores,
            "r2": r2,
            "q2": q2,
            "model": pls,
            "X_filtered": X_filtered,
        }
