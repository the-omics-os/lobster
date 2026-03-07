"""
Survival Analysis Service for time-to-event modeling.

Provides Cox proportional hazards, Kaplan-Meier estimation,
threshold optimization, and risk stratification for biomedical data.

Requires: pip install lobster-ml[survival]
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from anndata import AnnData
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.services.ml.sparse_utils import (
    SparseConversionError,
    check_sparse_conversion_safe,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["SurvivalAnalysisService"]


class SurvivalAnalysisService:
    """
    Stateless service for survival analysis on AnnData objects.

    All methods return the standard lobster 3-tuple:
    (AnnData, stats_dict, AnalysisStep)
    """

    def __init__(self):
        """Initialize the service and check for optional dependencies."""
        self._sksurv_available = self._check_sksurv()

    def _check_sksurv(self) -> bool:
        """Check if scikit-survival is available."""
        try:
            import sksurv

            return True
        except ImportError:
            return False

    def _validate_and_extract_features(
        self,
        adata: AnnData,
        feature_space_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract feature matrix from AnnData, validating feature_space_key.

        Args:
            adata: AnnData object
            feature_space_key: If provided, use adata.obsm[feature_space_key]
                              instead of adata.X as feature matrix.
                              Example: 'X_mofa' for integrated factors.

        Returns:
            X: Feature matrix (n_obs, n_features)
            feature_names: List of feature names

        Raises:
            KeyError: If feature_space_key not found in adata.obsm
            ValueError: If feature matrix shape mismatches adata.obs
        """
        if feature_space_key is not None:
            if feature_space_key not in adata.obsm:
                available_keys = list(adata.obsm.keys())
                raise KeyError(
                    f"Feature space key '{feature_space_key}' not found in adata.obsm. "
                    f"Available keys: {available_keys}. "
                    f"If using integrated factors, ensure integration was run first."
                )
            X = adata.obsm[feature_space_key]
            feature_names = [f"Factor_{i + 1}" for i in range(X.shape[1])]

            if X.shape[0] != len(adata.obs):
                raise ValueError(
                    f"Feature space '{feature_space_key}' has {X.shape[0]} samples "
                    f"but adata has {len(adata.obs)}. Shape mismatch detected."
                )
        else:
            X = adata.X
            if hasattr(X, "toarray"):
                check_sparse_conversion_safe(X, overhead_multiplier=1.5)
                X = X.toarray()
            feature_names = adata.var_names.tolist()

        return X, feature_names

    def _create_cox_model_ir(
        self,
        time_column: str,
        event_column: str,
        regularized: bool,
        l1_ratio: float,
        alpha_min_ratio: float,
        cv_folds: int,
        fit_baseline_model: bool,
        has_test_data: bool = False,
        feature_space_key: Optional[str] = None,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for Cox proportional hazards modeling.

        Supports both regularized and unregularized Cox models with conditional
        Jinja2 templates.

        Args:
            time_column: Column containing time-to-event
            event_column: Column containing event indicator
            regularized: Whether to use regularized model with CV-based alpha selection
            l1_ratio: Elastic net mixing parameter (0=L2, 1=L1)
            alpha_min_ratio: Minimum alpha ratio for regularization path
            cv_folds: Number of cross-validation folds for alpha selection
            fit_baseline_model: Whether to fit baseline hazard
            has_test_data: Flag for template conditional (enables test data section)
            feature_space_key: If provided, use adata.obsm[key] instead of adata.X

        Returns:
            AnalysisStep with Cox modeling pipeline code template
        """
        # Parameter schema with ParameterSpec objects
        parameter_schema = {
            "time_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=time_column,
                required=True,
                description="Column in obs containing time-to-event data",
            ),
            "event_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=event_column,
                required=True,
                description="Column in obs containing event indicator (0/1 or bool)",
            ),
            "regularized": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=regularized,
                required=False,
                description="Use regularized CoxnetSurvivalAnalysis with CV-based alpha selection (for high-dimensional data)",
            ),
            "l1_ratio": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=l1_ratio,
                required=False,
                validation_rule="0 <= l1_ratio <= 1",
                description="Elastic net mixing parameter (0=Ridge/L2, 1=Lasso/L1, only if regularized=True)",
            ),
            "alpha_min_ratio": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=alpha_min_ratio,
                required=False,
                validation_rule="alpha_min_ratio > 0",
                description="Minimum alpha ratio for regularization path (only if regularized=True)",
            ),
            "cv_folds": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=cv_folds,
                required=False,
                validation_rule="cv_folds >= 2",
                description="Number of cross-validation folds for alpha selection (only used if regularized=True)",
            ),
            "fit_baseline_model": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=fit_baseline_model,
                required=False,
                description="Whether to fit baseline cumulative hazard function",
            ),
            "test_adata": ParameterSpec(
                param_type="Optional[AnnData]",
                papermill_injectable=False,  # Cannot inject AnnData via papermill
                default_value=None,
                required=False,
                description="Held-out test set for unbiased C-index evaluation",
            ),
            "feature_space_key": ParameterSpec(
                param_type="Optional[str]",
                papermill_injectable=True,
                default_value=feature_space_key,
                required=False,
                description="Key in adata.obsm for feature matrix. None uses adata.X. Example: 'X_mofa'",
            ),
        }

        # 50+ line Jinja2 template for Cox model training with conditional logic
        code_template = """# Cox Proportional Hazards Model
# {% if regularized %}Regularized with CV-based alpha selection{% else %}Unregularized (standard Cox PH){% endif %}

import numpy as np
{% if regularized %}
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
{% else %}
from sksurv.linear_model import CoxPHSurvivalAnalysis
{% endif %}
from sksurv.metrics import concordance_index_censored

# 1. Extract feature matrix
{% if feature_space_key %}
# Using integrated feature space: {{ feature_space_key }}
X = adata.obsm["{{ feature_space_key }}"]
print(f"Feature space: {X.shape[1]} factors from '{{ feature_space_key }}'")
{% else %}
X = adata.X
if hasattr(X, "toarray"):
    from lobster.services.ml.sparse_utils import check_sparse_conversion_safe
    check_sparse_conversion_safe(X, overhead_multiplier=1.5)
    X = X.toarray()
{% endif %}

print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

# 2. Prepare survival data (structured array format)
events = adata.obs["{{ event_column }}"].astype(bool).values
times = adata.obs["{{ time_column }}"].values
y = np.array(
    [(event, time) for event, time in zip(events, times)],
    dtype=[("event", bool), ("time", float)]
)

n_events = events.sum()
print(f"Survival: {n_events} events / {len(events)} samples ({n_events/len(events):.1%} event rate)")

{% if regularized %}
# 3. Train regularized Cox model with CV-based alpha selection
# Step 3a: Get candidate alphas from initial fit
coxnet_init = CoxnetSurvivalAnalysis(
    l1_ratio={{ l1_ratio }},
    alpha_min_ratio={{ alpha_min_ratio }},
    fit_baseline_model={{ fit_baseline_model }},
)
coxnet_init.fit(X, y)
candidate_alphas = coxnet_init.alphas_
print(f"Testing {len(candidate_alphas)} candidate alphas via {{ cv_folds }}-fold CV")

# Step 3b: Cross-validate to select best alpha
gcv = GridSearchCV(
    make_pipeline(
        StandardScaler(),
        CoxnetSurvivalAnalysis(l1_ratio={{ l1_ratio }}, fit_baseline_model={{ fit_baseline_model }})
    ),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in candidate_alphas]},
    cv={{ cv_folds }},
    n_jobs=-1,
)
gcv.fit(X, y)

model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
best_alpha = gcv.best_params_["coxnetsurvivalanalysis__alphas"][0]
cv_c_index = gcv.best_score_
coef = model.coef_.ravel()

print(f"Best alpha: {best_alpha:.6f}")
print(f"CV C-index: {cv_c_index:.3f}")

{% else %}
# 3. Train unregularized Cox proportional hazards model
model = CoxPHSurvivalAnalysis()
model.fit(X, y)
coef = model.coef_
print("Trained unregularized CoxPHSurvivalAnalysis")

{% endif %}
# 4. Predict risk scores
risk_scores = model.predict(X)
adata.obs["cox_risk_score"] = risk_scores
print(f"Risk scores: min={risk_scores.min():.3f}, max={risk_scores.max():.3f}")

# 5. Calculate concordance index (C-index)
# ⚠️ WARNING: Training C-index is OPTIMISTICALLY BIASED!
# The model was trained on this data, so performance appears better than it will on new data.
#
# For UNBIASED evaluation, you should:
#   1. Provide held-out test_adata parameter (best option), OR
#   2. Use regularized=True for cross-validated C-index
#
# Training C-index should NEVER be reported in publications without this disclaimer.

c_result = concordance_index_censored(events, times, risk_scores)
c_index = c_result[0]
print(f"Training C-index: {c_index:.3f}")
print("⚠️  WARNING: This is training C-index (optimistically biased)")
print("   For publication, report test set or CV C-index instead")

{% if has_test_data %}
# If test data provided, evaluate on test set for unbiased C-index
# Uncomment and provide test_adata to get unbiased performance estimate:
#
# X_test = test_adata.X
# if hasattr(X_test, "toarray"):
#     X_test = X_test.toarray()
# events_test = test_adata.obs["{{ event_column }}"].astype(bool).values
# times_test = test_adata.obs["{{ time_column }}"].values
# risk_scores_test = model.predict(X_test)
# c_result_test = concordance_index_censored(events_test, times_test, risk_scores_test)
# print(f"\\nTest C-index: {c_result_test[0]:.3f} (UNBIASED)")
# print(f"Test samples: {X_test.shape[0]}, events: {events_test.sum()}")
{% endif %}

# 6. Store coefficients
adata.var["cox_coefficient"] = coef
hazard_ratios = np.exp(coef)
adata.var["hazard_ratio"] = hazard_ratios

nonzero_features = np.sum(np.abs(coef) > 1e-8)
print(f"Selected {nonzero_features} / {len(coef)} features with non-zero coefficients")

print(f"\\nCox model complete: C-index={c_index:.3f}, {nonzero_features} features")
"""

        # Get sksurv version for execution context
        try:
            import sksurv

            sksurv_version = sksurv.__version__
        except (ImportError, AttributeError):
            sksurv_version = "unknown"

        # Conditional imports list based on regularized parameter
        imports = [
            "import numpy as np",
            "from sksurv.metrics import concordance_index_censored",
            "from lobster.services.ml.sparse_utils import check_sparse_conversion_safe",
        ]
        if regularized:
            imports.extend(
                [
                    "from sksurv.linear_model import CoxnetSurvivalAnalysis",
                    "from sklearn.model_selection import GridSearchCV",
                    "from sklearn.pipeline import make_pipeline",
                    "from sklearn.preprocessing import StandardScaler",
                ]
            )
        else:
            imports.append("from sksurv.linear_model import CoxPHSurvivalAnalysis")

        # Description and operation based on regularized parameter
        if regularized:
            operation = "sksurv.linear_model.CoxnetSurvivalAnalysis.fit_with_cv"
            description = f"Cox PH model with CV-based alpha selection (l1_ratio={l1_ratio}, cv_folds={cv_folds})"
        else:
            operation = "sksurv.linear_model.CoxPHSurvivalAnalysis.fit"
            description = "Cox PH model (unregularized)"

        return AnalysisStep(
            operation=operation,
            tool_name="train_cox_model",
            description=description,
            library="scikit-survival",
            code_template=code_template,
            imports=imports,
            parameters={
                "time_column": time_column,
                "event_column": event_column,
                "regularized": regularized,
                "l1_ratio": l1_ratio,
                "alpha_min_ratio": alpha_min_ratio,
                "cv_folds": cv_folds,
                "fit_baseline_model": fit_baseline_model,
                "feature_space_key": feature_space_key,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "sksurv_version": sksurv_version,
                "regularized": regularized,
                "l1_ratio": l1_ratio if regularized else None,
                "cv_folds": cv_folds if regularized else None,
                "feature_space_key": feature_space_key,
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_threshold_ir(
        self,
        time_column: str,
        event_column: str,
        risk_score_column: str,
        time_horizon: float,
        n_iterations: int,
        subsample_fraction: float,
        quantile_clip: float,
        exclude_early_censored: bool,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for threshold optimization.

        Args:
            time_column: Column containing time-to-event
            event_column: Column containing event indicator
            risk_score_column: Column containing continuous risk scores
            time_horizon: Time cutoff for binary outcome
            n_iterations: Number of subsampling iterations
            subsample_fraction: Fraction of data per subsample
            quantile_clip: Clip thresholds at this quantile
            exclude_early_censored: Exclude samples censored before time horizon

        Returns:
            AnalysisStep with threshold optimization pipeline code template
        """
        # Parameter schema
        parameter_schema = {
            "time_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=time_column,
                required=True,
                description="Column containing time-to-event data",
            ),
            "event_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=event_column,
                required=True,
                description="Column containing event indicator",
            ),
            "risk_score_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=risk_score_column,
                required=True,
                description="Column containing continuous risk scores from Cox model",
            ),
            "time_horizon": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=time_horizon,
                required=True,
                validation_rule="time_horizon > 0",
                description="Time cutoff for binary outcome (events before = positive)",
            ),
            "n_iterations": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_iterations,
                required=False,
                validation_rule="n_iterations > 0",
                description="Number of subsampling iterations for robust threshold selection",
            ),
            "subsample_fraction": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=subsample_fraction,
                required=False,
                validation_rule="0 < subsample_fraction <= 1",
                description="Fraction of data to use per subsampling iteration",
            ),
            "quantile_clip": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=quantile_clip,
                required=False,
                validation_rule="0 < quantile_clip <= 1",
                description="Clip thresholds at this quantile to avoid outliers",
            ),
            "exclude_early_censored": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=exclude_early_censored,
                required=False,
                description="Exclude samples censored before time horizon (unknown outcome status)",
            ),
        }

        # 50+ line Jinja2 template
        code_template = """# Threshold Optimization for Binary Risk Classification
# Uses subsampling resampling (NOT bootstrap) + MCC maximization

import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix

# 1. Extract survival data and risk scores
risk_scores = adata.obs["{{ risk_score_column }}"].values.copy()
times = adata.obs["{{ time_column }}"].values.copy()
events = adata.obs["{{ event_column }}"].astype(bool).values.copy()

print(f"Optimizing threshold for {len(risk_scores)} samples")
print(f"Time horizon: {{ time_horizon }}")

# 2. Handle censored-before-horizon samples
{% if exclude_early_censored %}
# Exclude samples with unknown outcome (censored before horizon)
# These samples: (times < horizon) AND (not event)
censored_early = (times < {{ time_horizon }}) & ~events
n_excluded = censored_early.sum()
if n_excluded > 0:
    print(f"Excluding {n_excluded} samples censored before horizon (unknown outcome)")
    valid_mask = ~censored_early
    risk_scores = risk_scores[valid_mask]
    times = times[valid_mask]
    events = events[valid_mask]
{% else %}
# Including all samples (censored-before-horizon treated as negative)
censored_early = (times < {{ time_horizon }}) & ~events
n_censored = censored_early.sum()
if n_censored > 0:
    print(f"WARNING: {n_censored} samples censored before horizon")
    print("  These have unknown outcome and are treated as 'no event'")
    print("  Consider exclude_early_censored=True for conservative analysis")
{% endif %}

# 3. Create binary outcome: event before time horizon
binary_outcome = (times <= {{ time_horizon }}) & events
n_positive = binary_outcome.sum()
n_negative = len(binary_outcome) - n_positive
print(f"Binary outcome: {n_positive} positive / {n_negative} negative")

# 4. Get candidate thresholds (unique risk scores, clipped to avoid outliers)
thresholds = np.unique(risk_scores)
threshold_clip = np.quantile(risk_scores, {{ quantile_clip }})
thresholds = thresholds[thresholds <= threshold_clip]
print(f"Testing {len(thresholds)} candidate thresholds")

# 5. Subsampling threshold optimization
# NOTE: This is subsampling (replace=False), NOT bootstrap (replace=True)
print(f"Running {{ n_iterations }} subsampling iterations...")
all_mccs = []

for i in range({{ n_iterations }}):
    # Subsample WITHOUT replacement (not bootstrap!)
    n_samples = len(risk_scores)
    n_subsample = int(n_samples * {{ subsample_fraction }})
    indices = np.random.choice(n_samples, size=n_subsample, replace=False)

    subsample_scores = risk_scores[indices]
    subsample_outcome = binary_outcome[indices]

    # Calculate MCC for each threshold
    mccs = []
    for threshold in thresholds:
        predicted = (subsample_scores >= threshold).astype(int)
        mcc = matthews_corrcoef(subsample_outcome, predicted)
        mccs.append(mcc)

    all_mccs.append(mccs)

    if (i + 1) % 20 == 0:
        print(f"  Subsampling iteration {i + 1}/{{ n_iterations }}")

# 6. Find best threshold (maximum mean MCC across subsamples)
all_mccs = np.array(all_mccs)
mean_mccs = np.mean(all_mccs, axis=0)
std_mccs = np.std(all_mccs, axis=0)

best_idx = np.argmax(mean_mccs)
best_threshold = thresholds[best_idx]
best_mcc = mean_mccs[best_idx]
best_mcc_std = std_mccs[best_idx]

print(f"\\nOptimal threshold: {best_threshold:.3f}")
print(f"MCC: {best_mcc:.3f} ± {best_mcc_std:.3f}")

# 7. Apply threshold to all data
predictions = (risk_scores >= best_threshold).astype(int)
adata.obs["risk_prediction"] = predictions
adata.obs["risk_category"] = np.where(predictions == 1, "high_risk", "low_risk")

n_high_risk = predictions.sum()
n_low_risk = len(predictions) - n_high_risk
print(f"Risk stratification: {n_high_risk} high risk / {n_low_risk} low risk")

# 8. Calculate final performance metrics
cm = confusion_matrix(binary_outcome, predictions)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"\\nPerformance metrics:")
print(f"  Sensitivity (TPR): {sensitivity:.3f}")
print(f"  Specificity (TNR): {specificity:.3f}")
print(f"  PPV (Precision): {ppv:.3f}")
print(f"  NPV: {npv:.3f}")
print(f"  Accuracy: {accuracy:.3f}")
print(f"\\nConfusion matrix:")
print(f"  TN={tn}, FP={fp}")
print(f"  FN={fn}, TP={tp}")

# 9. Store optimization results
adata.uns["threshold_optimization"] = {
    "best_threshold": float(best_threshold),
    "time_horizon": {{ time_horizon }},
    "mcc": float(best_mcc),
    "mcc_std": float(best_mcc_std),
    "n_iterations": {{ n_iterations }},
    "subsample_fraction": {{ subsample_fraction }},
    "resampling_method": "subsampling",
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "ppv": float(ppv),
    "npv": float(npv),
    "accuracy": float(accuracy),
}

print(f"\\nThreshold optimization complete")
"""

        return AnalysisStep(
            operation="threshold_optimization",
            tool_name="optimize_threshold",
            description=f"Subsampling MCC-based threshold optimization (n_iterations={n_iterations})",
            library="scikit-learn",
            code_template=code_template,
            imports=[
                "import numpy as np",
                "from sklearn.metrics import matthews_corrcoef, confusion_matrix",
            ],
            parameters={
                "time_column": time_column,
                "event_column": event_column,
                "risk_score_column": risk_score_column,
                "time_horizon": time_horizon,
                "n_iterations": n_iterations,
                "subsample_fraction": subsample_fraction,
                "quantile_clip": quantile_clip,
                "exclude_early_censored": exclude_early_censored,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "time_horizon": time_horizon,
                "n_iterations": n_iterations,
                "subsample_fraction": subsample_fraction,
                "exclude_early_censored": exclude_early_censored,
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_kaplan_meier_ir(
        self,
        time_column: str,
        event_column: str,
        group_column: Optional[str],
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for Kaplan-Meier survival analysis.

        Args:
            time_column: Column containing time-to-event
            event_column: Column containing event indicator
            group_column: Optional column for group stratification

        Returns:
            AnalysisStep with Kaplan-Meier analysis code template
        """
        # Parameter schema
        parameter_schema = {
            "time_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=time_column,
                required=True,
                description="Column containing time-to-event data",
            ),
            "event_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=event_column,
                required=True,
                description="Column containing event indicator (0/1 or bool)",
            ),
            "group_column": ParameterSpec(
                param_type="Optional[str]",
                papermill_injectable=True,
                default_value=group_column,
                required=False,
                description="Optional column for group stratification (e.g., treatment groups)",
            ),
        }

        # 50+ line Jinja2 template with conditional logic
        code_template = """# Kaplan-Meier Survival Analysis
# Estimates survival curves with optional group comparison

import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator
{% if group_column %}
from sksurv.compare import compare_survival
{% endif %}

# 1. Extract survival data
times = adata.obs["{{ time_column }}"].values
events = adata.obs["{{ event_column }}"].astype(bool).values

n_events = events.sum()
n_censored = len(events) - n_events
print(f"Survival data: {len(events)} samples, {n_events} events, {n_censored} censored")

{% if group_column %}
# 2. Group-stratified Kaplan-Meier analysis
group_column = "{{ group_column }}"
groups = adata.obs[group_column].unique()
print(f"Stratifying by {group_column}: {len(groups)} groups")

km_results = {}
group_stats = {}

for group in groups:
    # Filter data for this group
    mask = adata.obs[group_column] == group
    g_times = times[mask]
    g_events = events[mask]

    print(f"\\nGroup: {group}")
    print(f"  Samples: {len(g_times)}")
    print(f"  Events: {g_events.sum()}")

    # Estimate Kaplan-Meier curve
    km_time, km_survival = kaplan_meier_estimator(g_events, g_times)

    # Store results
    km_results[str(group)] = {
        "time": km_time.tolist(),
        "survival": km_survival.tolist(),
    }

    # Calculate median survival (FIRST time survival drops to 0.5)
    median_idx = np.where(km_survival <= 0.5)[0]
    if len(median_idx) > 0:
        median_survival = km_time[median_idx[0]]  # First crossing
        print(f"  Median survival: {median_survival:.1f}")
    else:
        median_survival = None
        print(f"  Median survival: Not reached")

    # Calculate restricted mean survival time (RMST)
    tau = g_times.max()
    rmst = np.trapz(km_survival, km_time)
    print(f"  RMST (to {tau:.1f}): {rmst:.1f}")

    group_stats[str(group)] = {
        "n_samples": int(mask.sum()),
        "n_events": int(g_events.sum()),
        "median_survival": float(median_survival) if median_survival is not None else None,
        "rmst": float(rmst),
    }

adata.uns["kaplan_meier"] = km_results

# 3. Log-rank test for group comparison (if 2 groups)
if len(groups) == 2:
    print(f"\\nLog-rank test (comparing {groups[0]} vs {groups[1]}):")

    # Prepare data for log-rank test
    group_indicator = adata.obs[group_column].values
    y = np.array(
        [(event, time) for event, time in zip(events, times)],
        dtype=[("event", bool), ("time", float)]
    )

    try:
        chi2, p_value = compare_survival(y, group_indicator)
        print(f"  Chi-square: {chi2:.3f}")
        print(f"  P-value: {p_value:.4f}")

        group_stats["log_rank_chi2"] = float(chi2)
        group_stats["log_rank_p_value"] = float(p_value)
    except Exception as e:
        print(f"  Log-rank test failed: {e}")

adata.uns["kaplan_meier_stats"] = group_stats
print(f"\\nGroup-stratified KM analysis complete")

{% else %}
# 2. Single Kaplan-Meier curve (no stratification)
km_time, km_survival = kaplan_meier_estimator(events, times)

print(f"Kaplan-Meier curve estimated with {len(km_time)} time points")

# Store results
adata.uns["kaplan_meier"] = {
    "time": km_time.tolist(),
    "survival": km_survival.tolist(),
}

# Calculate median survival
median_idx = np.where(km_survival <= 0.5)[0]
if len(median_idx) > 0:
    median_survival = km_time[median_idx[0]]
    print(f"Median survival: {median_survival:.1f}")
else:
    median_survival = None
    print(f"Median survival: Not reached (censored)")

# Calculate restricted mean survival time (RMST)
# RMST = area under KM curve up to a specified time horizon
tau = times.max()  # Use max follow-up as restriction time
# Approximate integral using trapezoidal rule
rmst = np.trapz(km_survival, km_time)
print(f"Restricted mean survival time (to {tau:.1f}): {rmst:.1f}")

# Generate at-risk table at key time points
quartiles = np.quantile(times, [0.25, 0.5, 0.75, 1.0])
print(f"\\nAt-risk table:")
print(f"{'Time':<10} {'At Risk':<10} {'Events':<10} {'Censored':<10}")
for t in quartiles:
    at_risk = (times >= t).sum()
    events_by_t = ((times <= t) & events).sum()
    censored_by_t = ((times <= t) & ~events).sum()
    print(f"{t:<10.1f} {at_risk:<10} {events_by_t:<10} {censored_by_t:<10}")

adata.uns["kaplan_meier_stats"] = {
    "n_samples": len(times),
    "n_events": int(n_events),
    "n_censored": int(n_censored),
    "median_survival": float(median_survival) if median_survival is not None else None,
    "rmst": float(rmst),
    "max_time": float(times.max()),
}

print(f"\\nSingle KM analysis complete")
{% endif %}
"""

        return AnalysisStep(
            operation="sksurv.nonparametric.kaplan_meier_estimator",
            tool_name="kaplan_meier_analysis",
            description="Kaplan-Meier survival analysis"
            + (f" stratified by {group_column}" if group_column else ""),
            library="scikit-survival",
            code_template=code_template,
            imports=[
                "import numpy as np",
                "from sksurv.nonparametric import kaplan_meier_estimator",
            ]
            + (["from sksurv.compare import compare_survival"] if group_column else []),
            parameters={
                "time_column": time_column,
                "event_column": event_column,
                "group_column": group_column,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "has_groups": group_column is not None,
                "group_column": group_column,
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_hazard_ratios_ir(
        self,
        top_n: int,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for hazard ratio extraction.

        Args:
            top_n: Number of top features to return

        Returns:
            AnalysisStep with hazard ratio extraction code template
        """
        # Parameter schema
        parameter_schema = {
            "top_n": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=top_n,
                required=False,
                validation_rule="top_n > 0",
                description="Number of top features by absolute coefficient to return",
            ),
        }

        # Code template
        code_template = """# Hazard Ratio Extraction and Ranking
# Extracts hazard ratios from trained Cox model and ranks by importance
#
# Hazard Ratio Interpretation:
#   HR > 1: Risk factor (increases hazard)
#   HR < 1: Protective factor (decreases hazard)
#   HR = 1: No effect on hazard

import numpy as np

# 1. Extract Cox coefficients from trained model
if "cox_coefficient" not in adata.var.columns:
    raise ValueError("No Cox coefficients found. Train Cox model first.")

coef = adata.var["cox_coefficient"].values
feature_names = adata.var_names.tolist()

print(f"Extracting hazard ratios for {len(coef)} features")

# 2. Calculate hazard ratios (HR = exp(coefficient))
# Hazard ratio represents the multiplicative change in hazard per unit increase in feature
hazard_ratios = np.exp(coef)
adata.var["hazard_ratio"] = hazard_ratios

# 3. Sort features by absolute coefficient (feature importance)
# Features with large |coef| have strongest association with outcome
abs_coef = np.abs(coef)
sorted_idx = np.argsort(abs_coef)[::-1]

# 4. Get top N features
top_n = min({{ top_n }}, len(coef))
top_idx = sorted_idx[:top_n]

print(f"\\nTop {top_n} features by Cox coefficient magnitude:")
print(f"{'Feature':<30} {'Coef':>8} {'HR':>8} {'Effect':<12}")
print("-" * 62)

top_features = []
for idx in top_idx:
    feature = feature_names[idx]
    coefficient = coef[idx]
    hazard_ratio = hazard_ratios[idx]
    effect = "protective" if coefficient < 0 else "risk"

    top_features.append({
        "feature": feature,
        "coefficient": float(coefficient),
        "hazard_ratio": float(hazard_ratio),
        "effect": effect,
    })

    print(f"{feature:<30} {coefficient:>8.3f} {hazard_ratio:>8.3f} {effect:<12}")

# 5. Summary statistics
n_nonzero = np.sum(np.abs(coef) > 1e-8)
n_protective = np.sum(coef < 0)
n_risk = np.sum(coef > 0)

print(f"\\nSummary:")
print(f"  Total features: {len(coef)}")
print(f"  Non-zero coefficients: {n_nonzero}")
print(f"  Protective features (HR < 1): {n_protective}")
print(f"  Risk features (HR > 1): {n_risk}")
print(f"  Max hazard ratio: {hazard_ratios.max():.3f}")
print(f"  Min hazard ratio: {hazard_ratios[hazard_ratios > 0].min():.3f}")

# 6. Store top features in uns
adata.uns["top_hazard_ratios"] = top_features

print(f"\\nHazard ratio extraction complete")
"""

        return AnalysisStep(
            operation="hazard_ratio_extraction",
            tool_name="get_hazard_ratios",
            description=f"Extract and rank top {top_n} hazard ratios from Cox model",
            library="numpy",
            code_template=code_template,
            imports=["import numpy as np"],
            parameters={"top_n": top_n},
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "top_n": top_n,
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _validate_survival_data(
        self,
        events: np.ndarray,
        times: np.ndarray,
        min_events: int = 20,
    ) -> List[str]:
        """
        Validate survival data and return list of warnings.

        Raises errors for fatal issues (per user constraints):
        - All censored (no events)
        - Invalid times (zero/negative)
        - Too few events (<min_events)

        Returns list of warnings for non-fatal issues.
        """
        warnings = []

        # Fatal: No events (all censored)
        n_events = events.sum()
        if n_events == 0:
            raise ValueError(
                "No events observed (all samples censored). "
                "Survival analysis requires at least some observed events."
            )

        # Fatal: Invalid survival times
        if np.any(times <= 0):
            n_invalid = (times <= 0).sum()
            raise ValueError(
                f"{n_invalid} samples have invalid survival times (zero or negative). "
                f"Ensure preprocessing removed invalid records."
            )

        # Fatal: Too few events
        if n_events < min_events:
            raise ValueError(
                f"Only {n_events} events observed (minimum: {min_events}). "
                f"Survival analysis with <{min_events} events produces unreliable estimates. "
                f"Consider: (1) longer follow-up, (2) combining cohorts, or (3) simpler models."
            )

        # Warning: Very low event rate
        event_rate = n_events / len(events)
        if event_rate < 0.1:
            warnings.append(
                f"Low event rate ({event_rate:.1%}). Results may have wide confidence intervals."
            )

        return warnings

    def train_cox_model(
        self,
        adata: AnnData,
        time_column: str,
        event_column: str,
        feature_space_key: Optional[str] = None,
        test_adata: Optional[AnnData] = None,
        regularized: bool = False,
        l1_ratio: float = 0.5,
        alpha_min_ratio: float = 0.01,
        cv_folds: int = 5,
        fit_baseline_model: bool = True,
        workspace_path: Optional[Path] = None,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Train Cox proportional hazards model.

        By default uses unregularized CoxPHSurvivalAnalysis (appropriate for
        low-dimensional data). For high-dimensional data, use regularized=True
        to enable CoxnetSurvivalAnalysis with CV-based alpha selection.

        Args:
            adata: AnnData with features in X and survival info in obs
            time_column: Column in obs containing time-to-event
            event_column: Column in obs containing event indicator (0/1 or bool)
            feature_space_key: If provided, use adata.obsm[feature_space_key]
                              instead of adata.X as feature matrix.
                              Example: 'X_mofa' for integrated factors.
            test_adata: Optional held-out test set for unbiased C-index evaluation
            regularized: Use regularized model with CV-based alpha selection
            l1_ratio: Elastic net mixing parameter (0=L2, 1=L1, only if regularized=True)
            alpha_min_ratio: Minimum alpha ratio for regularization path (only if regularized=True)
            cv_folds: Number of cross-validation folds for alpha selection (only if regularized=True)
            fit_baseline_model: Whether to fit baseline hazard
            workspace_path: Optional workspace path for external model storage

        Returns:
            Tuple of (adata, stats, ir)
        """
        if not self._sksurv_available:
            raise ImportError(
                "scikit-survival not installed. Run: pip install lobster-ml[survival]"
            )

        from sksurv.metrics import concordance_index_censored

        # Validate columns exist
        if time_column not in adata.obs.columns:
            raise ValueError(f"Time column '{time_column}' not found in adata.obs")
        if event_column not in adata.obs.columns:
            raise ValueError(f"Event column '{event_column}' not found in adata.obs")

        # Validate and extract features
        X, feature_names = self._validate_and_extract_features(adata, feature_space_key)

        # Create structured array for survival
        events = adata.obs[event_column].astype(bool).values
        times = adata.obs[time_column].values
        y = np.array(
            [(e, t) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        )

        logger.info(f"Training Cox model: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Events: {events.sum()} / {len(events)} ({events.mean():.1%})")

        # Track warnings
        warnings = []

        # Validate survival data (raises on fatal issues)
        warnings = self._validate_survival_data(events, times)

        # Train model (regularized or unregularized)
        if not regularized:
            # Default: Unregularized Cox PH (no alpha selection needed)
            from sksurv.linear_model import CoxPHSurvivalAnalysis

            model = CoxPHSurvivalAnalysis()
            model.fit(X, y)
            coef = model.coef_
            best_alpha = None
            cv_score = None
            logger.info("Trained unregularized CoxPHSurvivalAnalysis")

        else:
            # Regularized: CoxnetSurvivalAnalysis with CV-based alpha selection
            from sksurv.linear_model import CoxnetSurvivalAnalysis

            # Step 1: Fit initial model to get candidate alphas
            coxnet_init = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alpha_min_ratio=alpha_min_ratio,
                fit_baseline_model=fit_baseline_model,
            )
            coxnet_init.fit(X, y)
            candidate_alphas = coxnet_init.alphas_

            # Step 2: Cross-validate to select best alpha
            gcv = GridSearchCV(
                make_pipeline(
                    StandardScaler(),
                    CoxnetSurvivalAnalysis(
                        l1_ratio=l1_ratio, fit_baseline_model=fit_baseline_model
                    ),
                ),
                param_grid={
                    "coxnetsurvivalanalysis__alphas": [[v] for v in candidate_alphas]
                },
                cv=cv_folds,
                n_jobs=-1,
            )
            gcv.fit(X, y)

            # Step 3: Get best model and alpha
            model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
            coef = model.coef_.ravel()  # Already best alpha, single column
            best_alpha = gcv.best_params_["coxnetsurvivalanalysis__alphas"][0]
            cv_score = gcv.best_score_

            logger.info(
                f"Trained CoxnetSurvivalAnalysis: best_alpha={best_alpha:.4f}, CV C-index={cv_score:.3f}"
            )

        # Get predictions for training data
        risk_scores = model.predict(X)

        # Store results in adata
        adata = adata.copy()
        adata.obs["cox_risk_score"] = risk_scores

        # Store coefficients in var
        if "cox_coefficient" not in adata.var.columns:
            adata.var["cox_coefficient"] = 0.0
        adata.var["cox_coefficient"] = coef

        # Identify non-zero features
        nonzero_features = np.sum(np.abs(coef) > 1e-8)

        # Evaluate model performance (unbiased C-index)
        c_index_info = {}

        if test_adata is not None:
            # Option 1: Evaluate on held-out test set (unbiased)
            X_test = test_adata.X
            if hasattr(X_test, "toarray"):
                check_sparse_conversion_safe(
                    X_test, overhead_multiplier=1.0
                )  # Lower overhead for simple evaluation
                X_test = X_test.toarray()

            events_test = test_adata.obs[event_column].astype(bool).values
            times_test = test_adata.obs[time_column].values

            risk_scores_test = model.predict(X_test)
            c_result_test = concordance_index_censored(
                events_test, times_test, risk_scores_test
            )

            c_index_info["c_index_test"] = float(c_result_test[0])
            c_index_info["c_index_source"] = "test_set"
            c_index_info["test_n_samples"] = X_test.shape[0]
            c_index_info["test_n_events"] = int(events_test.sum())
            c_index_info["concordant_pairs"] = int(c_result_test[1])
            c_index_info["discordant_pairs"] = int(c_result_test[2])
            c_index_info["tied_risk"] = int(c_result_test[3])
            c_index_info["tied_time"] = int(c_result_test[4])

            logger.info(
                f"Test C-index: {c_result_test[0]:.3f} (n={X_test.shape[0]}, events={events_test.sum()})"
            )

            # Store test predictions
            test_adata = test_adata.copy()
            test_adata.obs["cox_risk_score"] = risk_scores_test

        elif regularized and cv_score is not None:
            # Option 2: Use CV C-index from GridSearchCV (already computed for regularized)
            c_index_info["c_index_cv"] = float(cv_score)
            c_index_info["c_index_source"] = "cross_validation"
            c_index_info["cv_folds"] = cv_folds

            logger.info(f"CV C-index: {cv_score:.3f} ({cv_folds}-fold)")

        else:
            # Option 3: Training C-index only (warn about bias)
            c_result_train = concordance_index_censored(events, times, risk_scores)
            c_index_info["c_index_train"] = float(c_result_train[0])
            c_index_info["c_index_source"] = "training_set"
            c_index_info["concordant_pairs"] = int(c_result_train[1])
            c_index_info["discordant_pairs"] = int(c_result_train[2])
            c_index_info["tied_risk"] = int(c_result_train[3])
            c_index_info["tied_time"] = int(c_result_train[4])

            # Add explicit warning per user constraint
            warnings.append(
                "C-index computed on training data. This is optimistically biased and "
                "may overestimate model performance. For unbiased evaluation, provide "
                "test_adata parameter or use regularized=True for cross-validated C-index."
            )

            logger.warning(
                "C-index on training data only (biased). Provide test_adata for unbiased estimate."
            )

        # Store model externally (per user constraint: models NEVER in adata.uns)
        if workspace_path is not None:
            models_dir = workspace_path / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"cox_model_{timestamp}.pkl"
            model_path = models_dir / model_filename

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Store reference in adata.uns (NOT the model itself)
            adata.uns["model_reference"] = {
                "model_type": "CoxPHSurvivalAnalysis"
                if not regularized
                else "CoxnetSurvivalAnalysis",
                "model_path": str(model_path),
                "timestamp": timestamp,
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
                "time_column": time_column,
                "event_column": event_column,
                "regularized": regularized,
                "l1_ratio": l1_ratio if regularized else None,
                "alpha_min_ratio": alpha_min_ratio if regularized else None,
                "best_alpha": float(best_alpha) if best_alpha is not None else None,
                "cv_folds": cv_folds if regularized else None,
                "cv_score": float(cv_score) if cv_score is not None else None,
            }
            logger.info(f"Model saved to {model_path}")
        else:
            # Store minimal metadata without model
            adata.uns["cox_model_metadata"] = {
                "time_column": time_column,
                "event_column": event_column,
                "regularized": regularized,
                "l1_ratio": l1_ratio if regularized else None,
                "n_features_selected": nonzero_features,
            }

        # Build stats dict
        stats = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_events": int(events.sum()),
            "event_rate": float(events.mean()),
            "n_features_selected": int(nonzero_features),
            "regularized": regularized,
            "l1_ratio": l1_ratio if regularized else None,
            "best_alpha": float(best_alpha) if best_alpha is not None else None,
            "feature_space_key": feature_space_key,  # For provenance
            **c_index_info,  # Merge C-index evaluation results
            "warnings": warnings,
        }

        # Add model_path to stats if saved
        if workspace_path is not None:
            stats["model_path"] = str(model_path)

        # Create IR
        ir = self._create_cox_model_ir(
            time_column=time_column,
            event_column=event_column,
            regularized=regularized,
            l1_ratio=l1_ratio,
            alpha_min_ratio=alpha_min_ratio,
            cv_folds=cv_folds,
            fit_baseline_model=fit_baseline_model,
            has_test_data=(test_adata is not None),
            feature_space_key=feature_space_key,
        )

        # Log final result based on C-index source
        if "c_index_test" in c_index_info:
            logger.info(
                f"Cox model trained: Test C-index={c_index_info['c_index_test']:.3f}, {nonzero_features} features selected"
            )
        elif "c_index_cv" in c_index_info:
            logger.info(
                f"Cox model trained: CV C-index={c_index_info['c_index_cv']:.3f}, {nonzero_features} features selected"
            )
        else:
            logger.info(
                f"Cox model trained: Training C-index={c_index_info['c_index_train']:.3f}, {nonzero_features} features selected (biased)"
            )

        return adata, stats, ir

    def _create_binary_outcome(
        self,
        times: np.ndarray,
        events: np.ndarray,
        time_horizon: float,
        exclude_early_censored: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create binary outcome for threshold optimization.

        Handles censored-before-horizon samples appropriately:
        - These samples were censored BEFORE the time horizon
        - Their true outcome is unknown (could have had event after censoring)
        - Including them as "negative" may introduce bias

        Args:
            times: Array of survival times
            events: Array of event indicators (True = event, False = censored)
            time_horizon: Time cutoff for binary outcome
            exclude_early_censored: If True, exclude ambiguous samples

        Returns:
            Tuple of (binary_outcome, valid_mask, warnings)
        """
        warnings = []

        # Identify censored-before-horizon samples (ambiguous outcome)
        censored_before_horizon = (times < time_horizon) & ~events
        n_censored_early = censored_before_horizon.sum()

        if exclude_early_censored:
            # Exclude samples with unknown outcome status
            # Keep only: (1) events before horizon, (2) survived past horizon
            valid_mask = (times > time_horizon) | ((times <= time_horizon) & events)
            binary_outcome = (times[valid_mask] <= time_horizon) & events[valid_mask]

            if n_censored_early > 0:
                warnings.append(
                    f"Excluded {n_censored_early} samples censored before time horizon "
                    f"({time_horizon}). These had unknown outcome status."
                )
                logger.info(
                    f"Excluded {n_censored_early} censored-before-horizon samples"
                )

        else:
            # Include all samples (censored-before-horizon treated as negative)
            valid_mask = np.ones(len(times), dtype=bool)
            binary_outcome = (times <= time_horizon) & events

            if n_censored_early > 0:
                pct_early = n_censored_early / len(times) * 100
                warnings.append(
                    f"Warning: {n_censored_early} samples ({pct_early:.1f}%) were censored "
                    f"before time horizon ({time_horizon}). These have unknown outcome status "
                    f"and are treated as 'no event'. This may introduce bias. "
                    f"Consider using exclude_early_censored=True for more conservative analysis."
                )
                logger.warning(
                    f"{n_censored_early} samples censored before horizon (potential bias). "
                    f"Use exclude_early_censored=True to remove them."
                )

        return binary_outcome, valid_mask, warnings

    def optimize_threshold(
        self,
        adata: AnnData,
        time_column: str,
        event_column: str,
        risk_score_column: str = "cox_risk_score",
        time_horizon: Optional[float] = None,
        n_iterations: int = 100,
        subsample_fraction: float = 0.5,
        quantile_clip: float = 0.95,
        exclude_early_censored: bool = False,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Find optimal threshold for binary classification using MCC optimization.

        Uses subsampling resampling (NOT bootstrap) for robust threshold selection.
        Subsampling samples WITHOUT replacement from a fraction of the data,
        unlike bootstrap which samples WITH replacement at full size.

        Args:
            adata: AnnData with risk scores and survival info
            time_column: Column with time-to-event
            event_column: Column with event indicator
            risk_score_column: Column with continuous risk scores
            time_horizon: Time cutoff for binary outcome (events before = positive)
            n_iterations: Number of subsampling iterations (renamed from n_bootstrap)
            subsample_fraction: Fraction of data per subsample (renamed from bootstrap_fraction)
            quantile_clip: Clip thresholds at this quantile to avoid outliers
            exclude_early_censored: Exclude samples censored before time horizon (see SAV-05)

        Returns:
            Tuple of (adata, stats, ir)
        """
        from sklearn.metrics import confusion_matrix, matthews_corrcoef

        if risk_score_column not in adata.obs.columns:
            raise ValueError(
                f"Risk score column '{risk_score_column}' not found. Train Cox model first."
            )

        risk_scores = adata.obs[risk_score_column].values
        times = adata.obs[time_column].values
        events = adata.obs[event_column].astype(bool).values

        # Validate survival data (raises on fatal issues)
        warnings = self._validate_survival_data(events, times)

        # Determine time horizon (default: median time)
        if time_horizon is None:
            time_horizon = np.median(times[events])
            logger.info(f"Using median event time as horizon: {time_horizon:.1f}")

        # Store original times/events for censoring stats
        original_times = times.copy()
        original_events = events.copy()

        # Create binary outcome with censoring awareness
        binary_outcome, valid_mask, censoring_warnings = self._create_binary_outcome(
            times, events, time_horizon, exclude_early_censored
        )
        warnings.extend(censoring_warnings)

        # Apply valid_mask if excluding early censored samples
        if exclude_early_censored:
            risk_scores = risk_scores[valid_mask]
            times = times[valid_mask]
            events = events[valid_mask]

        # Get unique thresholds, clipped to avoid outliers
        thresholds = np.unique(risk_scores)
        threshold_clip = np.quantile(risk_scores, quantile_clip)
        thresholds = thresholds[thresholds <= threshold_clip]

        logger.info(f"Optimizing threshold with {n_iterations} subsampling iterations")

        # Subsampling for robust threshold selection
        # NOTE: Uses replace=False (subsampling), NOT replace=True (bootstrap)
        all_mccs = []
        for _ in range(n_iterations):
            # Subsample: sample WITHOUT replacement from fraction of data
            n_samples = len(risk_scores)
            n_subsample = int(n_samples * subsample_fraction)
            indices = np.random.choice(n_samples, size=n_subsample, replace=False)

            subsample_scores = risk_scores[indices]
            subsample_outcome = binary_outcome[indices]

            # Calculate MCC for each threshold
            mccs = []
            for threshold in thresholds:
                predicted = (subsample_scores >= threshold).astype(int)
                mcc = matthews_corrcoef(subsample_outcome, predicted)
                mccs.append(mcc)

            all_mccs.append(mccs)

        # Find best threshold (max mean MCC)
        all_mccs = np.array(all_mccs)
        mean_mccs = np.mean(all_mccs, axis=0)
        std_mccs = np.std(all_mccs, axis=0)

        best_idx = np.argmax(mean_mccs)
        best_threshold = thresholds[best_idx]
        best_mcc = mean_mccs[best_idx]
        best_mcc_std = std_mccs[best_idx]

        # Apply threshold to all data
        predictions = (risk_scores >= best_threshold).astype(int)

        # Calculate final metrics
        cm = confusion_matrix(binary_outcome, predictions)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Store in adata
        adata = adata.copy()
        adata.obs["risk_prediction"] = predictions
        adata.obs["risk_category"] = np.where(predictions == 1, "high_risk", "low_risk")

        # Calculate censoring stats
        n_censored_before_horizon = int(
            ((original_times < time_horizon) & ~original_events).sum()
        )

        adata.uns["threshold_optimization"] = {
            "best_threshold": float(best_threshold),
            "time_horizon": float(time_horizon),
            "mcc": float(best_mcc),
            "mcc_std": float(best_mcc_std),
            "n_iterations": n_iterations,
            "subsample_fraction": subsample_fraction,
            "resampling_method": "subsampling",
            "warnings": warnings,
        }

        stats = {
            "best_threshold": float(best_threshold),
            "time_horizon": float(time_horizon),
            "mcc": float(best_mcc),
            "mcc_std": float(best_mcc_std),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "npv": float(npv),
            "accuracy": float(accuracy),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "n_high_risk": int(predictions.sum()),
            "n_low_risk": int(len(predictions) - predictions.sum()),
            "n_iterations": n_iterations,
            "subsample_fraction": subsample_fraction,
            "resampling_method": "subsampling",
            "exclude_early_censored": exclude_early_censored,
            "n_censored_before_horizon": n_censored_before_horizon,
            "n_samples_used": len(risk_scores),
            "warnings": warnings,
        }

        ir = self._create_threshold_ir(
            time_column=time_column,
            event_column=event_column,
            risk_score_column=risk_score_column,
            time_horizon=time_horizon,
            n_iterations=n_iterations,
            subsample_fraction=subsample_fraction,
            quantile_clip=quantile_clip,
            exclude_early_censored=exclude_early_censored,
        )

        logger.info(
            f"Optimal threshold: {best_threshold:.3f}, MCC={best_mcc:.3f} (sens={sensitivity:.2f}, spec={specificity:.2f})"
        )

        return adata, stats, ir

    def _calculate_median_survival(
        self,
        km_time: np.ndarray,
        km_survival: np.ndarray,
    ) -> Optional[float]:
        """
        Calculate median survival time from Kaplan-Meier curve.

        Median survival is the FIRST time survival probability drops to or below 0.5.
        Returns None if survival never drops below 0.5 (median not reached).

        Args:
            km_time: Array of time points from KM estimator
            km_survival: Array of survival probabilities at each time point

        Returns:
            Median survival time, or None if not reached
        """
        # Find first time where survival drops to or below 0.5
        median_idx = np.where(km_survival <= 0.5)[0]
        if len(median_idx) > 0:
            return float(km_time[median_idx[0]])  # First crossing
        return None  # Median not reached (>50% survived entire follow-up)

    def kaplan_meier_analysis(
        self,
        adata: AnnData,
        time_column: str,
        event_column: str,
        group_column: Optional[str] = None,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform Kaplan-Meier survival analysis with optional group comparison.

        Args:
            adata: AnnData with survival info in obs
            time_column: Column with time-to-event
            event_column: Column with event indicator
            group_column: Optional column for group stratification

        Returns:
            Tuple of (adata, stats, ir)
        """
        if not self._sksurv_available:
            raise ImportError(
                "scikit-survival not installed. Run: pip install lobster-ml[survival]"
            )

        from sksurv.nonparametric import kaplan_meier_estimator

        times = adata.obs[time_column].values
        events = adata.obs[event_column].astype(bool).values

        # Validate survival data (raises on fatal issues)
        warnings = self._validate_survival_data(events, times)

        adata = adata.copy()

        if group_column is None:
            # Single KM curve
            km_time, km_survival = kaplan_meier_estimator(events, times)

            adata.uns["kaplan_meier"] = {
                "time": km_time.tolist(),
                "survival": km_survival.tolist(),
            }

            # Median survival time (first crossing of 0.5)
            median_survival = self._calculate_median_survival(km_time, km_survival)

            stats = {
                "n_samples": len(times),
                "n_events": int(events.sum()),
                "n_censored": int(len(events) - events.sum()),
                "event_rate": float(events.mean()),
                "median_survival": float(median_survival) if median_survival else None,
                "max_time": float(times.max()),
                "min_time": float(times.min()),
                # Full survival table data
                "survival_curve": {
                    "time_points": km_time.tolist(),
                    "survival_probability": km_survival.tolist(),
                    "n_time_points": len(km_time),
                },
                "warnings": [],
            }
        else:
            # Group-stratified KM curves
            if group_column not in adata.obs.columns:
                raise ValueError(f"Group column '{group_column}' not found")

            groups = adata.obs[group_column].unique()
            km_results = {}
            group_stats = {}

            for group in groups:
                mask = adata.obs[group_column] == group
                g_times = times[mask]
                g_events = events[mask]

                km_time, km_survival = kaplan_meier_estimator(g_events, g_times)
                km_results[str(group)] = {
                    "time": km_time.tolist(),
                    "survival": km_survival.tolist(),
                }

                # Median survival time (first crossing of 0.5)
                median_surv = self._calculate_median_survival(km_time, km_survival)

                group_stats[str(group)] = {
                    "n_samples": int(mask.sum()),
                    "n_events": int(g_events.sum()),
                    "n_censored": int(mask.sum() - g_events.sum()),
                    "event_rate": float(g_events.mean()),
                    "median_survival": float(median_surv) if median_surv else None,
                    "max_time": float(g_times.max()),
                }

            adata.uns["kaplan_meier"] = km_results

            # Log-rank test between groups (if 2 groups)
            if len(groups) == 2:
                from sksurv.compare import compare_survival

                group_indicator = adata.obs[group_column].values
                y = np.array(
                    [(e, t) for e, t in zip(events, times)],
                    dtype=[("event", bool), ("time", float)],
                )

                try:
                    chi2, p_value = compare_survival(y, group_indicator)
                    group_stats["log_rank_chi2"] = float(chi2)
                    group_stats["log_rank_p_value"] = float(p_value)
                except Exception as e:
                    logger.warning(f"Log-rank test failed: {e}")

            stats = {
                "n_groups": len(groups),
                "groups": group_stats,
                "warnings": warnings,
            }

        ir = self._create_kaplan_meier_ir(
            time_column=time_column,
            event_column=event_column,
            group_column=group_column,
        )

        return adata, stats, ir

    def get_hazard_ratios(
        self,
        adata: AnnData,
        top_n: int = 20,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Extract and rank hazard ratios from a trained Cox model.

        Args:
            adata: AnnData with trained Cox model (cox_coefficient in var)
            top_n: Number of top features to return

        Returns:
            Tuple of (adata, stats, ir)
        """
        if "cox_coefficient" not in adata.var.columns:
            raise ValueError("No Cox coefficients found. Train Cox model first.")

        coef = adata.var["cox_coefficient"].values
        feature_names = adata.var_names.tolist()

        # Calculate hazard ratios
        hazard_ratios = np.exp(coef)

        # Sort by absolute coefficient (importance)
        abs_coef = np.abs(coef)
        sorted_idx = np.argsort(abs_coef)[::-1]

        # Get top features
        top_idx = sorted_idx[:top_n]

        top_features = []
        for idx in top_idx:
            top_features.append(
                {
                    "feature": feature_names[idx],
                    "coefficient": float(coef[idx]),
                    "hazard_ratio": float(hazard_ratios[idx]),
                    "effect": "protective" if coef[idx] < 0 else "risk",
                }
            )

        # Store hazard ratios in var
        adata = adata.copy()
        adata.var["hazard_ratio"] = hazard_ratios

        stats = {
            "n_nonzero_features": int(np.sum(np.abs(coef) > 1e-8)),
            "top_features": top_features,
            "max_hazard_ratio": float(hazard_ratios.max()),
            "min_hazard_ratio": float(hazard_ratios[hazard_ratios > 0].min())
            if np.any(hazard_ratios > 0)
            else None,
        }

        ir = self._create_hazard_ratios_ir(top_n=top_n)

        return adata, stats, ir

    def check_availability(self) -> Dict[str, Any]:
        """Check if survival analysis dependencies are available."""
        return {
            "sksurv_available": self._sksurv_available,
            "ready": self._sksurv_available,
            "install_command": "pip install lobster-ml[survival]"
            if not self._sksurv_available
            else None,
        }
