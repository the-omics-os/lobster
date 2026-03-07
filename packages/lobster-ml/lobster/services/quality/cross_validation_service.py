"""
Cross-Validation Service for robust ML evaluation.

Provides stratified K-fold cross-validation with per-fold preprocessing,
metric aggregation, feature importance stability tracking, and comprehensive output.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

from lobster.core.provenance.analysis_ir import AnalysisStep, ParameterSpec
from lobster.services.ml.sparse_utils import (
    SparseConversionError,
    check_sparse_conversion_safe,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["CrossValidationService"]


class CrossValidationService:
    """
    Stateless service for cross-validation on AnnData objects.

    Key features:
    - Stratified K-fold for balanced splits
    - Per-fold preprocessing to prevent data leakage
    - Metric aggregation (mean, std, CI)
    - Feature importance stability tracking
    - Prediction tracking with fold IDs
    - Support for alternate feature spaces (e.g., integrated factors from MOFA)

    All methods return the standard lobster 3-tuple:
    (AnnData, stats_dict, AnalysisStep)
    """

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
            Tuple of (feature_matrix, feature_names)

        Raises:
            KeyError: If feature_space_key not found in adata.obsm
            ValueError: If feature space shape doesn't match samples
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
            # Check sparse conversion safety using centralized utility (Phase 10)
            if issparse(X):
                check_sparse_conversion_safe(X, overhead_multiplier=1.5)
                logger.info(
                    f"Densifying sparse matrix: {X.shape[0]:,} x {X.shape[1]:,}"
                )
                X = X.toarray()
            elif hasattr(X, "toarray"):
                X = X.toarray()
            feature_names = adata.var_names.tolist()

        return X, feature_names

    def _create_stratified_kfold_ir(
        self,
        target_column: str,
        model_class: str,
        model_params: Dict[str, Any],
        n_splits: int,
        shuffle: bool,
        random_state: int,
        scale_features: bool,
        return_predictions: bool,
        return_importances: bool,
        store_predictions: bool = False,
        feature_space_key: Optional[str] = None,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for stratified K-fold cross-validation.

        Args:
            target_column: Column in obs containing target variable
            model_class: Model class name for provenance
            model_params: Model hyperparameters for provenance
            n_splits: Number of CV folds
            shuffle: Shuffle before splitting
            random_state: Random seed
            scale_features: Apply per-fold standardization
            return_predictions: Track predictions with fold IDs
            return_importances: Track feature importances per fold
            store_predictions: Store sampled predictions or summary-only
            feature_space_key: If provided, use adata.obsm[feature_space_key]
                              instead of adata.X as feature matrix

        Returns:
            AnalysisStep with full CV pipeline code template
        """
        # Parameter schema with ParameterSpec objects
        parameter_schema = {
            "target_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=target_column,
                required=True,
                validation_rule="target_column in adata.obs.columns",
                description="Column in obs containing target variable",
            ),
            "model_class": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=model_class,
                required=True,
                validation_rule="model_class in ['RandomForestClassifier', 'XGBClassifier', 'LogisticRegression', 'SVC', 'GradientBoostingClassifier']",
                description="Scikit-learn estimator class name for model instantiation",
            ),
            "model_params": ParameterSpec(
                param_type="Dict[str, Any]",
                papermill_injectable=True,
                default_value=model_params if model_params else {},
                required=False,
                description="Model hyperparameters (e.g., {'n_estimators': 100, 'max_depth': 10})",
            ),
            "n_splits": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_splits,
                required=False,
                validation_rule="n_splits > 1",
                description="Number of CV folds",
            ),
            "shuffle": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=shuffle,
                required=False,
                description="Shuffle before splitting",
            ),
            "random_state": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=random_state,
                required=False,
                description="Random seed for reproducibility",
            ),
            "scale_features": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=scale_features,
                required=False,
                description="Apply per-fold standardization to prevent data leakage",
            ),
            "return_predictions": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=return_predictions,
                required=False,
                description="Track predictions with fold IDs",
            ),
            "return_importances": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=return_importances,
                required=False,
                description="Track feature importances per fold",
            ),
            "store_predictions": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=store_predictions,
                required=False,
                description="Store sampled predictions (max 1000/fold) or summary-only",
            ),
            "feature_space_key": ParameterSpec(
                param_type="Optional[str]",
                papermill_injectable=True,
                default_value=feature_space_key,
                required=False,
                validation_rule="feature_space_key is None or feature_space_key in adata.obsm.keys()",
                description="If provided, use adata.obsm[feature_space_key] instead of adata.X (e.g., 'X_mofa' for integrated factors)",
            ),
        }

        # Comprehensive Jinja2 template for stratified K-fold CV
        code_template = """# Stratified K-fold cross-validation with per-fold preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)
from scipy.sparse import issparse
from lobster.services.ml.sparse_utils import (
    check_sparse_conversion_safe,
    SparseConversionError,
)
import numpy as np
import pandas as pd

# Validate target column
target_column = "{{ target_column }}"
if target_column not in adata.obs.columns:
    raise ValueError(f"Target column '{target_column}' not found in adata.obs")

# Extract features and target
{% if feature_space_key %}
# Use alternate feature space (e.g., integrated factors from MOFA)
feature_space_key = "{{ feature_space_key }}"
if feature_space_key not in adata.obsm:
    available_keys = list(adata.obsm.keys())
    raise KeyError(
        f"Feature space key '{feature_space_key}' not found in adata.obsm. "
        f"Available keys: {available_keys}"
    )
X = adata.obsm[feature_space_key]
feature_names = [f"Factor_{i+1}" for i in range(X.shape[1])]
print(f"Using feature space: {feature_space_key} ({X.shape[1]} factors)")
{% else %}
# Use default feature matrix (adata.X)
X = adata.X
feature_names = adata.var_names.tolist()
{% endif %}

# Check sparse matrix memory before densification (Phase 10)
if issparse(X):
    check_sparse_conversion_safe(X, overhead_multiplier=1.5)
    print(f"Densifying sparse matrix: {X.shape[0]:,} x {X.shape[1]:,}")
    X = X.toarray()
elif hasattr(X, "toarray"):
    X = X.toarray()

y = adata.obs[target_column].values
sample_names = adata.obs_names.tolist()

# Encode target if categorical
label_encoder = None
if y.dtype == object or str(y.dtype) == "category":
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_.tolist()
else:
    class_names = list(map(str, np.unique(y)))

n_classes = len(np.unique(y))
is_binary = n_classes == 2

print(f"Starting {{ n_splits }}-fold CV: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")

# Initialize stratified K-fold
skf = StratifiedKFold(
    n_splits={{ n_splits }},
    shuffle={{ shuffle }},
    random_state={{ random_state }}
)

# Create model with captured parameters (inline, not callable)
{% if model_class == "RandomForestClassifier" %}
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators={{ model_params.get('n_estimators', 100) }},
    max_depth={{ model_params.get('max_depth', None) }},
    min_samples_split={{ model_params.get('min_samples_split', 2) }},
    random_state={{ random_state }}
)
{% elif model_class == "XGBClassifier" %}
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators={{ model_params.get('n_estimators', 100) }},
    max_depth={{ model_params.get('max_depth', 6) }},
    learning_rate={{ model_params.get('learning_rate', 0.3) }},
    random_state={{ random_state }}
)
{% elif model_class == "LogisticRegression" %}
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
    C={{ model_params.get('C', 1.0) }},
    max_iter={{ model_params.get('max_iter', 100) }},
    random_state={{ random_state }}
)
{% elif model_class == "SVC" %}
from sklearn.svm import SVC
model = SVC(
    C={{ model_params.get('C', 1.0) }},
    kernel='{{ model_params.get("kernel", "rbf") }}',
    probability=True,
    random_state={{ random_state }}
)
{% else %}
# Generic sklearn estimator fallback
from sklearn.utils import all_estimators
model = {{ model_class }}(**{{ model_params }})
{% endif %}

# Storage for results
fold_metrics = []
all_predictions = []
fold_predictions_by_fold = []
all_importances = []

# Cross-validation loop
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"Processing fold {fold_idx + 1}/{{ n_splits }}")

    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Per-fold preprocessing (prevents data leakage)
    {% if scale_features %}
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    {% endif %}

    # Train model (create fresh instance per fold)
    {% if model_class == "RandomForestClassifier" %}
    fold_model = RandomForestClassifier(
        n_estimators={{ model_params.get('n_estimators', 100) }},
        max_depth={{ model_params.get('max_depth', None) }},
        min_samples_split={{ model_params.get('min_samples_split', 2) }},
        random_state={{ random_state }} + fold_idx
    )
    {% elif model_class == "XGBClassifier" %}
    fold_model = XGBClassifier(
        n_estimators={{ model_params.get('n_estimators', 100) }},
        max_depth={{ model_params.get('max_depth', 6) }},
        learning_rate={{ model_params.get('learning_rate', 0.3) }},
        random_state={{ random_state }} + fold_idx
    )
    {% elif model_class == "LogisticRegression" %}
    fold_model = LogisticRegression(
        C={{ model_params.get('C', 1.0) }},
        max_iter={{ model_params.get('max_iter', 100) }},
        random_state={{ random_state }} + fold_idx
    )
    {% elif model_class == "SVC" %}
    fold_model = SVC(
        C={{ model_params.get('C', 1.0) }},
        kernel='{{ model_params.get("kernel", "rbf") }}',
        probability=True,
        random_state={{ random_state }} + fold_idx
    )
    {% else %}
    fold_model = {{ model_class }}(**{{ model_params }})
    {% endif %}
    fold_model.fit(X_train, y_train)

    # Get predictions
    y_pred = fold_model.predict(X_test)

    # Get probabilities if available
    y_prob = None
    if hasattr(fold_model, "predict_proba"):
        y_prob = fold_model.predict_proba(X_test)
        if is_binary:
            y_prob = y_prob[:, 1]  # Probability of positive class

    # Calculate metrics
    metrics = {
        "fold": fold_idx,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }

    # Add ROC-AUC for binary classification
    if is_binary and y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except ValueError:
            metrics["roc_auc"] = np.nan

    fold_metrics.append(metrics)

    # Track predictions
    {% if return_predictions %}
    fold_preds = []
    for i, idx in enumerate(test_idx):
        pred_info = {
            "sample": sample_names[idx],
            "fold": fold_idx,
            "true_label": int(y_test[i]),
            "predicted_label": int(y_pred[i]),
        }
        if y_prob is not None:
            if is_binary:
                pred_info["probability"] = float(y_prob[i])
            else:
                pred_info["probabilities"] = y_prob[i].tolist()
        fold_preds.append(pred_info)
        all_predictions.append(pred_info)
    fold_predictions_by_fold.append(fold_preds)
    {% endif %}

    # Track feature importances
    {% if return_importances %}
    if hasattr(fold_model, "feature_importances_"):
        importances = fold_model.feature_importances_
    elif hasattr(fold_model, "coef_"):
        importances = np.abs(fold_model.coef_).flatten()
        if len(importances) != len(feature_names):
            importances = np.mean(np.abs(fold_model.coef_), axis=0)
    else:
        importances = None

    if importances is not None:
        all_importances.append(importances)
    {% endif %}

# Aggregate metrics across folds
metrics_df = pd.DataFrame(fold_metrics)
aggregated_metrics = {}

for col in metrics_df.columns:
    if col != "fold":
        values = metrics_df[col].dropna().values
        aggregated_metrics[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values.tolist(),
        }

# Aggregate feature importances efficiently
{% if return_importances %}
if all_importances:
    all_importances = np.array(all_importances)
    mean_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)

    # Stability score: importance / std (higher = more stable)
    stability = mean_importance / (std_importance + 1e-8)

    # Store full arrays in var (efficient AnnData storage)
    adata.var["cv_mean_importance"] = mean_importance
    adata.var["cv_importance_std"] = std_importance
    adata.var["cv_stability"] = stability

    # Store only top 100 in uns
    top_indices = np.argsort(mean_importance)[-100:][::-1]
    top_features = {
        "features": adata.var_names[top_indices].tolist(),
        "mean_importance": mean_importance[top_indices].tolist(),
        "stability": stability[top_indices].tolist(),
    }
{% endif %}

# Store results in adata.uns
adata.uns["cross_validation"] = {
    "n_splits": {{ n_splits }},
    "target_column": target_column,
    "class_names": class_names,
    "aggregated_metrics": aggregated_metrics,
    "fold_metrics": fold_metrics,
}

# Store predictions with sampling to prevent metadata bloat
{% if return_predictions %}
{% if store_predictions %}
# Store sampled predictions (max 1000/fold to prevent bloat)
MAX_PREDICTIONS_PER_FOLD = 1000
sampled_predictions = []
for fold_idx in range({{ n_splits }}):
    fold_preds = [p for p in all_predictions if p['fold'] == fold_idx]
    if len(fold_preds) > MAX_PREDICTIONS_PER_FOLD:
        import random
        random.seed({{ random_state }} + fold_idx)
        fold_preds = random.sample(fold_preds, MAX_PREDICTIONS_PER_FOLD)
    sampled_predictions.extend(fold_preds)
adata.uns['cross_validation']['predictions'] = sampled_predictions
adata.uns['cross_validation']['predictions_sampled'] = len(all_predictions) > ({{ n_splits }} * MAX_PREDICTIONS_PER_FOLD)
adata.uns['cross_validation']['predictions_stored'] = True
{% else %}
# Summary-only storage (prevent metadata bloat)
adata.uns['cross_validation']['predictions_stored'] = False
adata.uns['cross_validation']['n_predictions'] = len(all_predictions)
{% endif %}
{% endif %}

# Store top features if available
{% if return_importances %}
if all_importances:
    adata.uns['cross_validation']['top_features'] = top_features
{% endif %}

print(f"CV complete: accuracy={aggregated_metrics['accuracy']['mean']:.3f}+/-{aggregated_metrics['accuracy']['std']:.3f}")
"""

        return AnalysisStep(
            operation="sklearn.model_selection.StratifiedKFold",
            tool_name="stratified_kfold_cv",
            description=f"{n_splits}-fold stratified cross-validation",
            library="sklearn",
            code_template=code_template,
            imports=[
                "from sklearn.model_selection import StratifiedKFold",
                "from sklearn.preprocessing import StandardScaler, LabelEncoder",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef",
                "import numpy as np",
                "import pandas as pd",
            ],
            parameters={
                "target_column": target_column,
                "model_class": model_class,
                "model_params": model_params if model_params else {},
                "n_splits": n_splits,
                "shuffle": shuffle,
                "random_state": random_state,
                "scale_features": scale_features,
                "return_predictions": return_predictions,
                "return_importances": return_importances,
                "store_predictions": store_predictions,
                "feature_space_key": feature_space_key,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "sklearn_version": self._get_sklearn_version(),
                "operation_type": "cross_validation",
                "model_class": model_class,
                "n_splits": n_splits,
                "random_state": random_state,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _get_sklearn_version(self) -> str:
        """Get exact sklearn version (not requirement range)."""
        try:
            import sklearn

            return sklearn.__version__
        except (ImportError, AttributeError):
            return "unknown"

    def stratified_kfold_cv(
        self,
        adata: AnnData,
        target_column: str,
        model_factory: Callable,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        scale_features: bool = True,
        return_predictions: bool = True,
        return_importances: bool = True,
        store_predictions: bool = False,
        model_class: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        feature_space_key: Optional[str] = None,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform stratified K-fold cross-validation with comprehensive tracking.

        Args:
            adata: AnnData with features in X and target in obs
            target_column: Column in obs containing target variable
            model_factory: Callable that returns a fresh model instance
                          (e.g., lambda: XGBClassifier(n_estimators=100))
            n_splits: Number of CV folds
            shuffle: Shuffle before splitting
            random_state: Random seed
            scale_features: Apply per-fold standardization
            return_predictions: Track predictions with fold IDs
            return_importances: Track feature importances per fold
            store_predictions: If True, store sampled predictions (max 1000/fold).
                If False (default), store aggregated metrics only to prevent metadata bloat.
            model_class: Model class name for provenance (auto-extracted if not provided)
            model_params: Model hyperparameters for provenance (auto-extracted if not provided)
            feature_space_key: If provided, use adata.obsm[feature_space_key]
                              instead of adata.X as feature matrix.
                              Example: 'X_mofa' for integrated factors.

        Returns:
            Tuple of (adata, stats, ir) with CV results stored in adata.uns
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        # Extract model info from factory if not provided (for provenance)
        if model_factory and not model_class:
            try:
                model_instance = model_factory()
                model_class = model_instance.__class__.__name__
                # Extract params if available (sklearn models have get_params())
                if hasattr(model_instance, "get_params"):
                    model_params = model_instance.get_params()
                else:
                    model_params = {}
                logger.debug(
                    f"Extracted model info: {model_class} with {len(model_params)} params"
                )
            except Exception as e:
                logger.warning(f"Could not extract model info from factory: {e}")
                model_class = "UnknownModel"
                model_params = {}

        # Validate target column
        if target_column not in adata.obs.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        # Extract features using helper (handles feature_space_key and sparse conversion)
        X, feature_names = self._validate_and_extract_features(adata, feature_space_key)

        y = adata.obs[target_column].values
        sample_names = adata.obs_names.tolist()

        # Encode target if categorical
        label_encoder = None
        if y.dtype == object or str(y.dtype) == "category":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            class_names = label_encoder.classes_.tolist()
        else:
            class_names = list(map(str, np.unique(y)))

        n_classes = len(np.unique(y))
        is_binary = n_classes == 2

        feature_space_info = f" from {feature_space_key}" if feature_space_key else ""
        logger.info(
            f"Starting {n_splits}-fold CV: {X.shape[0]} samples, {X.shape[1]} features{feature_space_info}, {n_classes} classes"
        )

        # Initialize CV
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

        # Storage for results
        fold_metrics = []
        all_predictions = []
        fold_predictions_by_fold = []  # For per-fold sampling
        all_importances = []
        all_probabilities = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Processing fold {fold_idx + 1}/{n_splits}")

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Per-fold preprocessing (prevents data leakage)
            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Train model
            model = model_factory()
            model.fit(X_train, y_train)

            # Get predictions
            y_pred = model.predict(X_test)

            # Get probabilities if available
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                if is_binary:
                    y_prob = y_prob[:, 1]  # Probability of positive class

            # Calculate metrics
            metrics = {
                "fold": fold_idx,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "mcc": matthews_corrcoef(y_test, y_pred),
            }

            # Add ROC-AUC for binary classification
            if is_binary and y_prob is not None:
                try:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
                except ValueError:
                    metrics["roc_auc"] = np.nan

            fold_metrics.append(metrics)

            # Track predictions
            if return_predictions:
                fold_preds = []
                for i, idx in enumerate(test_idx):
                    pred_info = {
                        "sample": sample_names[idx],
                        "fold": fold_idx,
                        "true_label": int(y_test[i]),
                        "predicted_label": int(y_pred[i]),
                    }
                    if y_prob is not None:
                        if is_binary:
                            pred_info["probability"] = float(y_prob[i])
                        else:
                            pred_info["probabilities"] = y_prob[i].tolist()
                    fold_preds.append(pred_info)
                    all_predictions.append(pred_info)
                fold_predictions_by_fold.append(fold_preds)

            # Track feature importances
            if return_importances:
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_).flatten()
                    if len(importances) != len(feature_names):
                        importances = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importances = None

                if importances is not None:
                    all_importances.append(importances)

        # Aggregate metrics
        metrics_df = pd.DataFrame(fold_metrics)
        aggregated_metrics = {}

        for col in metrics_df.columns:
            if col != "fold":
                values = metrics_df[col].dropna().values
                aggregated_metrics[col] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values.tolist(),
                }

        # Aggregate feature importances efficiently
        importance_stats = None
        if all_importances:
            all_importances = np.array(all_importances)
            mean_importance = np.mean(all_importances, axis=0)
            std_importance = np.std(all_importances, axis=0)

            # Stability score: importance / std (higher = more stable)
            stability = mean_importance / (std_importance + 1e-8)

            # Store full arrays in var (efficient AnnData storage)
            adata.var["cv_mean_importance"] = mean_importance
            adata.var["cv_importance_std"] = std_importance
            adata.var["cv_stability"] = stability

            # Store only top 100 in uns for quick access
            top_indices = np.argsort(mean_importance)[-100:][::-1]
            feature_names_array = np.array(feature_names)
            importance_stats = {
                "features": feature_names_array[top_indices].tolist(),
                "mean_importance": mean_importance[top_indices].tolist(),
                "stability": stability[top_indices].tolist(),
            }
            logger.info(
                f"Stored feature importance: full arrays in var, top 100 in uns"
            )

        # Store results in adata
        adata = adata.copy()

        adata.uns["cross_validation"] = {
            "n_splits": n_splits,
            "target_column": target_column,
            "class_names": class_names,
            "aggregated_metrics": aggregated_metrics,
            "fold_metrics": fold_metrics,
        }

        # Store predictions with sampling to prevent metadata bloat
        if return_predictions and store_predictions:
            MAX_PREDICTIONS_PER_FOLD = 1000
            sampled_predictions = []

            for fold_idx, fold_preds in enumerate(fold_predictions_by_fold):
                # Sample if too many predictions in this fold
                if len(fold_preds) > MAX_PREDICTIONS_PER_FOLD:
                    import random

                    random.seed(random_state + fold_idx)
                    fold_preds = random.sample(fold_preds, MAX_PREDICTIONS_PER_FOLD)
                sampled_predictions.extend(fold_preds)

            adata.uns["cross_validation"]["predictions"] = sampled_predictions
            adata.uns["cross_validation"]["predictions_sampled"] = (
                len(all_predictions) > n_splits * MAX_PREDICTIONS_PER_FOLD
            )
            adata.uns["cross_validation"]["predictions_stored"] = True
            logger.info(
                f"Stored {len(sampled_predictions)} sampled predictions (max 1000/fold)"
            )
        elif return_predictions:
            # return_predictions=True but store_predictions=False: just count
            adata.uns["cross_validation"]["predictions_stored"] = False
            adata.uns["cross_validation"]["n_predictions"] = len(all_predictions)
            logger.info(
                f"Prediction tracking enabled but storage disabled ({len(all_predictions)} predictions computed)"
            )

        if importance_stats:
            adata.uns["cross_validation"]["top_features"] = importance_stats

        # Build stats summary
        stats = {
            "n_splits": n_splits,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": n_classes,
            "class_names": class_names,
            "feature_space_key": feature_space_key,
            "metrics": {
                metric: {
                    "mean": agg["mean"],
                    "std": agg["std"],
                }
                for metric, agg in aggregated_metrics.items()
            },
        }

        # Add top features if available (from top 100 stored in uns)
        if importance_stats:
            stats["top_features"] = [
                {
                    "feature": importance_stats["features"][i],
                    "mean_importance": importance_stats["mean_importance"][i],
                    "stability": importance_stats["stability"][i],
                }
                for i in range(min(10, len(importance_stats["features"])))
            ]

        ir = self._create_stratified_kfold_ir(
            target_column=target_column,
            model_class=model_class,
            model_params=model_params,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            scale_features=scale_features,
            return_predictions=return_predictions,
            return_importances=return_importances,
            store_predictions=store_predictions,
            feature_space_key=feature_space_key,
        )

        logger.info(
            f"CV complete: accuracy={aggregated_metrics['accuracy']['mean']:.3f}+/-{aggregated_metrics['accuracy']['std']:.3f}"
        )

        return adata, stats, ir

    def get_cv_predictions_df(
        self,
        adata: AnnData,
    ) -> pd.DataFrame:
        """
        Get cross-validation predictions as a DataFrame.

        Args:
            adata: AnnData with CV results in uns

        Returns:
            DataFrame with columns: sample, fold, true_label, predicted_label, probability
        """
        if "cv_predictions" not in adata.uns:
            raise ValueError("No CV predictions found. Run stratified_kfold_cv first.")

        return pd.DataFrame(adata.uns["cv_predictions"])

    def get_cv_metrics_df(
        self,
        adata: AnnData,
    ) -> pd.DataFrame:
        """
        Get per-fold metrics as a DataFrame.

        Args:
            adata: AnnData with CV results in uns

        Returns:
            DataFrame with metrics per fold
        """
        if "cross_validation" not in adata.uns:
            raise ValueError("No CV results found. Run stratified_kfold_cv first.")

        return pd.DataFrame(adata.uns["cross_validation"]["fold_metrics"])

    def get_cv_feature_importance_df(
        self,
        adata: AnnData,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get feature importance statistics from CV as a DataFrame.

        Args:
            adata: AnnData with CV results in uns
            top_n: Return only top N features (None for all)

        Returns:
            DataFrame sorted by mean importance
        """
        if "cv_feature_importances" not in adata.uns:
            raise ValueError(
                "No CV feature importances found. Run CV with return_importances=True."
            )

        df = pd.DataFrame(adata.uns["cv_feature_importances"]).T
        df.index.name = "feature"
        df = df.sort_values("mean_importance", ascending=False)

        if top_n is not None:
            df = df.head(top_n)

        return df.reset_index()

    def export_cv_results(
        self,
        adata: AnnData,
        output_dir: str,
        prefix: str = "cv_results",
    ) -> Dict[str, str]:
        """
        Export CV results to CSV files.

        Args:
            adata: AnnData with CV results
            output_dir: Directory for output files
            prefix: Filename prefix

        Returns:
            Dict mapping result type to file path
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # Export metrics
        if "cross_validation" in adata.uns:
            metrics_df = self.get_cv_metrics_df(adata)
            metrics_path = output_path / f"{prefix}_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            exported_files["metrics"] = str(metrics_path)

        # Export predictions
        if "cv_predictions" in adata.uns:
            preds_df = self.get_cv_predictions_df(adata)
            preds_path = output_path / f"{prefix}_predictions.csv"
            preds_df.to_csv(preds_path, index=False)
            exported_files["predictions"] = str(preds_path)

        # Export feature importances
        if "cv_feature_importances" in adata.uns:
            imp_df = self.get_cv_feature_importance_df(adata)
            imp_path = output_path / f"{prefix}_feature_importances.csv"
            imp_df.to_csv(imp_path, index=False)
            exported_files["feature_importances"] = str(imp_path)

        logger.info(
            f"Exported CV results to {output_path}: {list(exported_files.keys())}"
        )

        return exported_files
