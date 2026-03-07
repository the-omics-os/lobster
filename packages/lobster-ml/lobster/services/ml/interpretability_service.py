"""
Interpretability Service for ML model explanations.

Provides SHAP-based feature importance extraction for various model types
including tree-based, linear, and neural network models.

Requires: pip install lobster-ml[interpretability]
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.services.ml.sparse_utils import (
    SparseConversionError,
    check_sparse_conversion_safe,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["InterpretabilityService"]


class InterpretabilityService:
    """
    Stateless service for ML model interpretability on AnnData objects.

    Supports SHAP explanations for multiple model types:
    - TreeExplainer: XGBoost, LightGBM, Random Forest
    - LinearExplainer: Logistic Regression, Lasso, Ridge
    - DeepExplainer: PyTorch neural networks
    - KernelExplainer: Any black-box model (fallback)

    Supports alternate feature spaces (e.g., integrated factors from MOFA).

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

        Note:
            When using integrated factors, feature_names will be Factor_1, Factor_2, etc.
            SHAP values then explain factor contributions rather than original features.
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

    def __init__(self):
        """Initialize service and check dependencies."""
        self._shap_available = self._check_shap()

    def _check_shap(self) -> bool:
        """Check if SHAP is available."""
        try:
            import shap

            return True
        except ImportError:
            return False

    def _create_shap_extraction_ir(
        self,
        model_type: str,
        background_samples: int,
        random_state: int,
        feature_space_key: Optional[str] = None,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for SHAP value extraction.

        Args:
            model_type: Type of SHAP explainer to use
            background_samples: Number of background samples for KernelExplainer
            random_state: Random seed for background sampling
            feature_space_key: If provided, use adata.obsm[feature_space_key]

        Returns:
            AnalysisStep with full SHAP extraction code template
        """
        # Parameter schema with ParameterSpec objects
        parameter_schema = {
            "model_type": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=model_type,
                required=False,
                description="Type of SHAP explainer: 'tree', 'linear', 'deep', or 'kernel'",
            ),
            "background_samples": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=background_samples,
                required=False,
                validation_rule="background_samples > 0",
                description="Number of background samples for KernelExplainer",
            ),
            "random_state": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=random_state,
                required=False,
                description="Random seed for reproducible background sampling",
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

        # Comprehensive Jinja2 template with model type conditionals
        code_template = """# SHAP feature importance extraction
import shap
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from lobster.services.ml.sparse_utils import (
    check_sparse_conversion_safe,
    SparseConversionError,
)

# Prepare data
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
print("Note: SHAP values explain factor contributions, not original features")
{% else %}
# Use default feature matrix (adata.X)
X = adata.X
# Check sparse conversion safety (Phase 10)
if issparse(X):
    check_sparse_conversion_safe(X, overhead_multiplier=1.5)
    print(f"Densifying sparse matrix: {X.shape[0]:,} x {X.shape[1]:,}")
    X = X.toarray()
elif hasattr(X, 'toarray'):
    X = X.toarray()
feature_names = adata.var_names.tolist()
{% endif %}

X_df = pd.DataFrame(X, index=adata.obs_names, columns=feature_names)

# Create appropriate SHAP explainer based on model type
model_type = "{{ model_type }}"
print(f"Extracting SHAP values using {model_type} explainer")

{% if model_type == "tree" %}
# TreeExplainer for tree-based models (XGBoost, LightGBM, Random Forest)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

{% elif model_type == "linear" %}
# LinearExplainer for linear models (Logistic Regression, Lasso, Ridge)
explainer = shap.LinearExplainer(model, X)
shap_values = explainer.shap_values(X)

{% elif model_type == "deep" %}
# DeepExplainer for PyTorch neural networks
import torch
model.eval()

# Sample background data
np.random.seed({{ random_state }})
bg_indices = np.random.choice(len(X), min({{ background_samples }}, len(X)), replace=False)
background = torch.tensor(X[bg_indices], dtype=torch.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_tensor)

{% else %}
# KernelExplainer as fallback for any black-box model
np.random.seed({{ random_state }})
background = shap.kmeans(X_df, min({{ background_samples }}, len(X_df)))

# Create predict function
if hasattr(model, 'predict_proba'):
    predict_fn = model.predict_proba
else:
    predict_fn = model.predict

explainer = shap.KernelExplainer(predict_fn, background)
shap_values = explainer.shap_values(X_df)
{% endif %}

# Handle multi-class output - SHAP returns different formats:
# - List of 2D arrays for some explainers
# - 3D array for TreeExplainer: (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    # List format: multi-class as list of 2D arrays
    n_classes = len(shap_values)
    print(f"Multi-class model detected: {n_classes} classes (list format)")

    # Store each class separately (AnnData layers must be 2D)
    for class_idx, class_shap in enumerate(shap_values):
        layer_name = f"shap_class_{class_idx}"
        adata.layers[layer_name] = np.abs(class_shap)
        print(f"  Stored {layer_name}: shape {class_shap.shape}")

    # Create aggregate (mean absolute across classes)
    shap_values_combined = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    adata.layers['shap_values'] = shap_values_combined

    # Store class mapping if available
    if hasattr(model, 'classes_'):
        class_mapping = {i: str(cls) for i, cls in enumerate(model.classes_)}
        adata.uns['shap_class_mapping'] = class_mapping
        print(f"  Class mapping: {class_mapping}")

elif shap_values.ndim == 3:
    # 3D format: TreeExplainer returns (n_samples, n_features, n_classes)
    n_classes = shap_values.shape[2]
    print(f"Multi-class model detected: {n_classes} classes (3D array format)")

    # Store each class separately (split along class axis)
    for class_idx in range(n_classes):
        layer_name = f"shap_class_{class_idx}"
        adata.layers[layer_name] = np.abs(shap_values[:, :, class_idx])
        print(f"  Stored {layer_name}: shape {shap_values[:, :, class_idx].shape}")

    # Create aggregate (mean absolute across classes)
    shap_values_combined = np.mean(np.abs(shap_values), axis=2)
    adata.layers['shap_values'] = shap_values_combined

    # Store class mapping if available
    if hasattr(model, 'classes_'):
        class_mapping = {i: str(cls) for i, cls in enumerate(model.classes_)}
        adata.uns['shap_class_mapping'] = class_mapping
        print(f"  Class mapping: {class_mapping}")

else:
    # Binary or single output: 2D array (n_samples, n_features)
    n_classes = 2
    print(f"Binary model detected - stored both classes")
    adata.layers['shap_class_0'] = np.abs(shap_values)
    adata.layers['shap_class_1'] = np.abs(shap_values)
    adata.layers['shap_values'] = np.abs(shap_values)

    if hasattr(model, 'classes_'):
        class_mapping = {i: str(cls) for i, cls in enumerate(model.classes_)}
        adata.uns['shap_class_mapping'] = class_mapping
        print(f"  Binary class mapping: {class_mapping}")

    adata.uns['shap_analysis']['positive_class_idx'] = 1

# Calculate global feature importance (mean absolute SHAP)
global_importance = np.mean(shap_values_combined, axis=0)
adata.var['shap_importance'] = global_importance

# Calculate local importance std
importance_std = np.std(shap_values_combined, axis=0)
adata.var['shap_importance_std'] = importance_std

# Rank features
adata.var['shap_rank'] = np.argsort(np.argsort(global_importance)[::-1]) + 1

# Store metadata
adata.uns['shap_analysis'] = {
    'model_type': model_type,
    'n_classes': n_classes,
    'n_samples_explained': len(X),
    'aggregation_method': 'mean_absolute',
}

# Report top features
top_indices = np.argsort(global_importance)[::-1][:10]
print(f"Top 10 features by SHAP importance:")
for i, idx in enumerate(top_indices, 1):
    print(f"  {i}. {feature_names[idx]}: {global_importance[idx]:.4f}")

print(f"SHAP analysis complete: {len(X)} samples, {len(feature_names)} features")
"""

        return AnalysisStep(
            operation=f"shap.{model_type.capitalize()}Explainer",
            tool_name="extract_shap_values",
            description=f"Extract SHAP values using {model_type} explainer for model interpretability",
            library="shap",
            code_template=code_template,
            imports=["import shap", "import numpy as np", "import pandas as pd"],
            parameters={
                "model_type": model_type,
                "background_samples": background_samples,
                "random_state": random_state,
                "feature_space_key": feature_space_key,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata", "model"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "interpretability",
                "model_type": model_type,
                "shap_version": "0.41.0",
                "timestamp": "runtime",
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_shap_aggregation_ir(
        self,
        normalize: bool,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for SHAP aggregation.

        Args:
            normalize: Whether to normalize importances to sum to 1.0

        Returns:
            AnalysisStep with SHAP aggregation code template
        """
        parameter_schema = {
            "normalize": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=normalize,
                required=False,
                description="Normalize global importances to sum to 1.0",
            ),
        }

        code_template = """# Aggregate SHAP values to global feature importance
import numpy as np

# ============================================================
# SHAP GLOBAL FEATURE IMPORTANCE AGGREGATION
# ============================================================
print("=" * 60)
print("SHAP GLOBAL FEATURE IMPORTANCE AGGREGATION")
print("=" * 60)

# Retrieve SHAP values from layers
if 'shap_values' not in adata.layers:
    raise ValueError("No SHAP values found. Run extract_shap_values first.")

shap_values = adata.layers['shap_values']
feature_names = adata.var_names.tolist()

# Discover per-class SHAP layers
per_class_layers = {}
for layer_name in adata.layers.keys():
    if layer_name.startswith('shap_class_'):
        try:
            class_idx = int(layer_name.split('_')[-1])
            per_class_layers[class_idx] = layer_name
        except ValueError:
            continue

if per_class_layers:
    print(f"Found {len(per_class_layers)} per-class SHAP layers")
    # Calculate per-class global importance
    for class_idx in sorted(per_class_layers.keys()):
        layer_name = per_class_layers[class_idx]
        class_shap = adata.layers[layer_name]
        class_importance = np.mean(class_shap, axis=0)
        adata.var[f"{layer_name}_importance"] = class_importance

print(f"\\nInput shape: {shap_values.shape[0]} samples x {shap_values.shape[1]} features")

# ============================================================
# SHAP Statistics
# ============================================================
print(f"\\nSHAP value statistics across all samples:")
print(f"  Min:  {shap_values.min():.6f}")
print(f"  Max:  {shap_values.max():.6f}")
print(f"  Mean: {shap_values.mean():.6f}")
print(f"  Std:  {shap_values.std():.6f}")

# Non-zero contributions
n_total = shap_values.size
n_nonzero = np.count_nonzero(shap_values)
sparsity = 1.0 - (n_nonzero / n_total)

print(f"\\nSHAP matrix sparsity:")
print(f"  Total values:    {n_total:,}")
print(f"  Non-zero values: {n_nonzero:,}")
print(f"  Sparsity:        {sparsity:.2%}")

# ============================================================
# Calculate Global Importance
# ============================================================
global_importance = np.mean(np.abs(shap_values), axis=0)

{% if normalize %}
# Normalize importances to sum to 1.0
total = global_importance.sum()
if total > 0:
    global_importance = global_importance / total
print(f"\\nNormalized importances to sum to 1.0")
{% else %}
print(f"\\nUsing raw (non-normalized) importances")
{% endif %}

# ============================================================
# Importance Distribution Analysis
# ============================================================
print(f"\\nGlobal importance distribution:")
print(f"  Min:    {global_importance.min():.6f}")
print(f"  Max:    {global_importance.max():.6f}")
print(f"  Mean:   {global_importance.mean():.6f}")
print(f"  Median: {np.median(global_importance):.6f}")
print(f"  Std:    {global_importance.std():.6f}")

# Percentile breakdown
print(f"\\nImportance percentiles:")
percentiles = [25, 50, 75, 90, 95, 99]
for p in percentiles:
    threshold = np.percentile(global_importance, p)
    n_above = (global_importance > threshold).sum()
    print(f"  {p}th: {threshold:.6f} ({n_above} features above)")

# Identify dominant features (>1% importance each)
{% if normalize %}
dominant_mask = global_importance > 0.01
n_dominant = dominant_mask.sum()
if n_dominant > 0:
    dominant_importance_sum = global_importance[dominant_mask].sum()
    print(f"\\nDominant features (>1% importance each): {n_dominant}")
    print(f"  Total contribution: {dominant_importance_sum:.2%}")
{% endif %}

# ============================================================
# Extended Top Features
# ============================================================
sorted_idx = np.argsort(global_importance)[::-1]

# Store results in adata.var
adata.var['global_importance'] = global_importance
adata.var['global_importance_pct'] = global_importance * 100
adata.var['importance_rank'] = np.argsort(sorted_idx) + 1

# Top 20 features with cumulative importance
print(f"\\nTop 20 features by global importance:")
print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12} {'Pct':<8} {'Cumulative':<12}")
print("-" * 78)

cumulative = 0.0
for i, idx in enumerate(sorted_idx[:20], 1):
    imp = global_importance[idx]
    cumulative += imp
    feat_name = feature_names[idx][:28]  # Truncate long names
    print(f"{i:<6} {feat_name:<30} {imp:.6f}    {imp*100:>5.2f}%   {cumulative:.6f}")

# Percentage contribution at different cutoffs
print(f"\\nCumulative importance by top N features:")
for n in [5, 10, 15, 20]:
    if n <= len(sorted_idx):
        cumulative_n = sum(global_importance[sorted_idx[:n]])
        print(f"  Top {n:2d}: {cumulative_n:.4f} ({cumulative_n*100:>5.2f}%)")

# ============================================================
# Feature Importance Tiers
# ============================================================
# Classify features into tiers based on importance percentiles
high_threshold = np.percentile(global_importance, 90)
medium_threshold = np.percentile(global_importance, 50)

n_high = (global_importance >= high_threshold).sum()
n_medium = ((global_importance >= medium_threshold) & (global_importance < high_threshold)).sum()
n_low = (global_importance < medium_threshold).sum()

print(f"\\nFeature importance tiers:")
print(f"  High (top 10%):     {n_high} features")
print(f"  Medium (10-50%):    {n_medium} features")
print(f"  Low (bottom 50%):   {n_low} features")

# ============================================================
# Summary
# ============================================================
print(f"\\n{'=' * 60}")
print(f"SUMMARY")
print(f"{'=' * 60}")
total_cumulative = global_importance.sum()
print(f"Total cumulative importance: {total_cumulative:.6f}")
{% if normalize %}
print(f"  (should be 1.0 after normalization)")
{% endif %}

print(f"\\nResults stored in adata.var:")
print(f"  - 'global_importance': raw importance values")
print(f"  - 'global_importance_pct': importance as percentage")
print(f"  - 'importance_rank': feature rank (1 = most important)")
print(f"{'=' * 60}")
"""

        return AnalysisStep(
            operation="shap_global_aggregation",
            tool_name="aggregate_shap_to_global",
            description=f"Aggregate local SHAP values to global feature importance (normalize={normalize})",
            library="shap",
            code_template=code_template,
            imports=["import numpy as np"],
            parameters={"normalize": normalize},
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "aggregation",
                "normalized": normalize,
                "timestamp": "runtime",
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def extract_shap_values(
        self,
        adata: AnnData,
        model: Any,
        model_type: Optional[str] = None,
        background_samples: int = 100,
        random_state: int = 42,
        feature_space_key: Optional[str] = None,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Extract SHAP values for model interpretability.

        Automatically detects model type and uses appropriate SHAP explainer.

        Args:
            adata: AnnData with features in X (same features model was trained on)
            model: Trained model object (XGBoost, sklearn, PyTorch, etc.)
            model_type: Override auto-detection ("tree", "linear", "deep", "kernel")
            background_samples: Number of background samples for KernelExplainer
            random_state: Random seed for background sampling
            feature_space_key: If provided, use adata.obsm[feature_space_key]
                              instead of adata.X as feature matrix.
                              Example: 'X_mofa' for integrated factors.
                              Note: Factor names will be Factor_1, Factor_2, etc.

        Returns:
            Tuple of (adata, stats, ir) with SHAP values stored in adata.layers['shap_values']
        """
        if not self._shap_available:
            raise ImportError(
                "SHAP not installed. Run: pip install lobster-ml[interpretability]"
            )

        import shap

        # Prepare data using helper (handles feature_space_key)
        X, feature_names = self._validate_and_extract_features(adata, feature_space_key)

        X_df = pd.DataFrame(X, index=adata.obs_names, columns=feature_names)

        # Auto-detect model type if not specified
        if model_type is None:
            model_type = self._detect_model_type(model)

        feature_space_info = f" from {feature_space_key}" if feature_space_key else ""
        logger.info(
            f"Extracting SHAP values using {model_type} explainer{feature_space_info}"
        )

        # Create appropriate explainer
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)

        elif model_type == "deep":
            try:
                import torch

                model.eval()
                # Sample background data
                np.random.seed(random_state)
                bg_indices = np.random.choice(
                    len(X), min(background_samples, len(X)), replace=False
                )
                background = torch.tensor(X[bg_indices], dtype=torch.float32)
                X_tensor = torch.tensor(X, dtype=torch.float32)

                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(X_tensor)
            except ImportError:
                raise ImportError("PyTorch required for deep model explanations")

        elif model_type == "kernel":
            # KernelExplainer as fallback
            np.random.seed(random_state)
            background = shap.kmeans(X_df, min(background_samples, len(X_df)))

            # Create predict function
            if hasattr(model, "predict_proba"):
                predict_fn = model.predict_proba
            else:
                predict_fn = model.predict

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_df)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Copy adata first
        adata = adata.copy()

        # Handle multi-class output - SHAP returns different formats:
        # - List of 2D arrays: [array(n_samples, n_features), ...] for some explainers
        # - 3D array: (n_samples, n_features, n_classes) for TreeExplainer
        if isinstance(shap_values, list):
            # List format: multi-class as list of 2D arrays
            n_classes = len(shap_values)

            # Store each class separately (AnnData layers must be 2D)
            for class_idx, class_shap in enumerate(shap_values):
                layer_name = f"shap_class_{class_idx}"
                adata.layers[layer_name] = np.abs(class_shap)

            # Create aggregate (mean absolute across classes) for backward compatibility
            shap_values_combined = np.mean([np.abs(sv) for sv in shap_values], axis=0)

            # Store class mapping if available
            if hasattr(model, "classes_"):
                class_mapping = {i: str(cls) for i, cls in enumerate(model.classes_)}
                adata.uns["shap_class_mapping"] = class_mapping

        elif shap_values.ndim == 3:
            # 3D format: TreeExplainer returns (n_samples, n_features, n_classes)
            n_classes = shap_values.shape[2]

            # Store each class separately (split along class axis)
            for class_idx in range(n_classes):
                layer_name = f"shap_class_{class_idx}"
                adata.layers[layer_name] = np.abs(shap_values[:, :, class_idx])

            # Create aggregate (mean absolute across classes)
            shap_values_combined = np.mean(np.abs(shap_values), axis=2)

            # Store class mapping if available
            if hasattr(model, "classes_"):
                class_mapping = {i: str(cls) for i, cls in enumerate(model.classes_)}
                adata.uns["shap_class_mapping"] = class_mapping

        else:
            # Binary or single output: 2D array (n_samples, n_features)
            n_classes = 2
            adata.layers["shap_class_0"] = np.abs(shap_values)
            adata.layers["shap_class_1"] = np.abs(shap_values)
            shap_values_combined = np.abs(shap_values)

            # Store class mapping and positive class indicator
            if hasattr(model, "classes_"):
                class_mapping = {i: str(cls) for i, cls in enumerate(model.classes_)}
                adata.uns["shap_class_mapping"] = class_mapping

        # Store aggregate layer
        adata.layers["shap_values"] = shap_values_combined

        # Calculate global feature importance (mean absolute SHAP)
        global_importance = np.mean(shap_values_combined, axis=0)
        adata.var["shap_importance"] = global_importance

        # Calculate local importance std
        importance_std = np.std(shap_values_combined, axis=0)
        adata.var["shap_importance_std"] = importance_std

        # Rank features
        ranked_indices = np.argsort(global_importance)[::-1]
        adata.var["shap_rank"] = np.argsort(np.argsort(global_importance)[::-1]) + 1

        # Store metadata
        adata.uns["shap_analysis"] = {
            "model_type": model_type,
            "n_classes": n_classes,
            "n_samples_explained": len(X),
            "aggregation_method": "mean_absolute",
        }

        # Mark positive class for binary models
        if n_classes == 2:
            adata.uns["shap_analysis"]["positive_class_idx"] = 1

        # Build stats
        top_features = [
            {
                "rank": i + 1,
                "feature": feature_names[idx],
                "mean_shap": float(global_importance[idx]),
                "std_shap": float(importance_std[idx]),
            }
            for i, idx in enumerate(ranked_indices[:20])
        ]

        # Build per-class top features
        per_class_top_features = {}
        layers_created = ["shap_values"]

        for class_idx in range(n_classes):
            layer_name = f"shap_class_{class_idx}"
            if layer_name in adata.layers:
                layers_created.append(layer_name)

                # Calculate class-specific importance
                class_shap = adata.layers[layer_name]
                class_importance = np.mean(class_shap, axis=0)
                top_indices = np.argsort(class_importance)[::-1][:10]

                per_class_top_features[layer_name] = [
                    {
                        "rank": i + 1,
                        "feature": feature_names[idx],
                        "mean_shap": float(class_importance[idx]),
                    }
                    for i, idx in enumerate(top_indices)
                ]

        stats = {
            "model_type": model_type,
            "n_samples": len(X),
            "n_features": len(feature_names),
            "n_classes": n_classes,
            "aggregation_method": "mean_absolute",
            "class_mapping": adata.uns.get("shap_class_mapping", {}),
            "layers_created": layers_created,
            "per_class_top_features": per_class_top_features,
            "top_features": top_features,
            "total_mean_shap": float(np.mean(global_importance)),
            "max_mean_shap": float(np.max(global_importance)),
            "feature_space_key": feature_space_key,
        }

        ir = self._create_shap_extraction_ir(
            model_type=model_type,
            background_samples=background_samples,
            random_state=random_state,
            feature_space_key=feature_space_key,
        )

        logger.info(
            f"SHAP analysis complete: top feature = {top_features[0]['feature']} ({top_features[0]['mean_shap']:.4f})"
        )

        return adata, stats, ir

    def _detect_model_type(self, model: Any) -> str:
        """Auto-detect appropriate SHAP explainer type for a model."""
        model_class_name = type(model).__name__

        # Tree-based models
        tree_models = [
            "XGBClassifier",
            "XGBRegressor",
            "LGBMClassifier",
            "LGBMRegressor",
            "RandomForestClassifier",
            "RandomForestRegressor",
            "GradientBoostingClassifier",
            "GradientBoostingRegressor",
            "DecisionTreeClassifier",
            "DecisionTreeRegressor",
            "ExtraTreesClassifier",
            "ExtraTreesRegressor",
        ]
        if model_class_name in tree_models:
            return "tree"

        # Linear models
        if hasattr(model, "coef_"):
            return "linear"

        # PyTorch models
        try:
            import torch.nn as nn

            if isinstance(model, nn.Module):
                return "deep"
        except ImportError:
            pass

        # Default to kernel explainer
        return "kernel"

    def aggregate_shap_to_global(
        self,
        adata: AnnData,
        normalize: bool = True,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Aggregate local SHAP values to global feature importance.

        Args:
            adata: AnnData with SHAP values in layers['shap_values']
            normalize: Normalize importances to sum to 1.0

        Returns:
            Tuple of (adata, stats, ir)
        """
        if "shap_values" not in adata.layers:
            raise ValueError("No SHAP values found. Run extract_shap_values first.")

        shap_values = adata.layers["shap_values"]

        # Discover per-class SHAP layers
        per_class_layers = {}
        for layer_name in adata.layers.keys():
            if layer_name.startswith("shap_class_"):
                try:
                    class_idx = int(layer_name.split("_")[-1])
                    per_class_layers[class_idx] = layer_name
                except ValueError:
                    continue

        if per_class_layers:
            logger.info(f"Found {len(per_class_layers)} per-class SHAP layers")
            # Calculate per-class importance (stored in var for optional analysis)
            for class_idx in sorted(per_class_layers.keys()):
                layer_name = per_class_layers[class_idx]
                class_shap = adata.layers[layer_name]
                class_importance = np.mean(class_shap, axis=0)
                adata.var[f"{layer_name}_importance"] = class_importance

        # Calculate mean absolute SHAP per feature
        global_importance = np.mean(np.abs(shap_values), axis=0)

        if normalize:
            total = global_importance.sum()
            if total > 0:
                global_importance = global_importance / total

        adata = adata.copy()
        adata.var["global_importance"] = global_importance
        adata.var["global_importance_pct"] = global_importance * 100

        # Sort features by importance
        sorted_idx = np.argsort(global_importance)[::-1]
        feature_names = adata.var_names.tolist()

        importance_dict = {
            feature_names[idx]: float(global_importance[idx]) for idx in sorted_idx
        }

        stats = {
            "n_features": len(feature_names),
            "normalized": normalize,
            "top_10_features": {
                feature_names[idx]: float(global_importance[idx])
                for idx in sorted_idx[:10]
            },
            "cumulative_top_10": float(sum(global_importance[sorted_idx[:10]])),
        }

        ir = self._create_shap_aggregation_ir(normalize=normalize)

        return adata, stats, ir

    def get_feature_importance_df(
        self,
        adata: AnnData,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get feature importance as a sorted DataFrame.

        Args:
            adata: AnnData with SHAP analysis results
            top_n: Return only top N features (None for all)

        Returns:
            DataFrame with feature importance columns
        """
        importance_cols = [
            col
            for col in adata.var.columns
            if "shap" in col.lower() or "importance" in col.lower()
        ]

        if not importance_cols:
            raise ValueError("No importance columns found. Run SHAP analysis first.")

        df = adata.var[importance_cols].copy()
        df.index.name = "feature"

        # Sort by primary importance column
        sort_col = (
            "shap_importance" if "shap_importance" in df.columns else importance_cols[0]
        )
        df = df.sort_values(sort_col, ascending=False)

        if top_n is not None:
            df = df.head(top_n)

        return df.reset_index()

    def check_availability(self) -> Dict[str, Any]:
        """Check if interpretability dependencies are available."""
        result = {
            "shap_available": self._shap_available,
            "ready": self._shap_available,
        }

        if not self._shap_available:
            result["install_command"] = "pip install lobster-ml[interpretability]"

        # Check optional dependencies
        try:
            import xgboost

            result["xgboost_available"] = True
        except ImportError:
            result["xgboost_available"] = False

        try:
            import torch

            result["pytorch_available"] = True
        except ImportError:
            result["pytorch_available"] = False

        return result
