"""
ML Preprocessing Service for data preparation.

Provides missing value handling, feature scaling, class imbalance handling,
and sklearn Pipeline integration for reproducible ML preprocessing.

Requires: pip install lobster-ml[imbalanced] for class balancing
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.services.ml.sparse_utils import (
    SparseConversionError,
    check_sparse_conversion_safe,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["MLPreprocessingService", "infer_task_type"]


def infer_task_type(y: np.ndarray) -> Tuple[str, float]:
    """
    Infer likely task type from target array.

    Returns (suggestion, confidence) where confidence is 0.0-1.0.

    NOTE: This provides guidance only. Users MUST still pass task_type explicitly.

    Examples:
        >>> y = np.array(["A", "B", "A", "B"])
        >>> infer_task_type(y)
        ('classification', 1.0)

        >>> y = np.array([1.5, 2.3, 0.8, 4.1, 3.2])
        >>> infer_task_type(y)
        ('regression', 0.9)
    """
    # Categorical dtype is definitively classification
    if y.dtype == object or str(y.dtype) == "category":
        return ("classification", 1.0)

    n_unique = len(np.unique(y))
    n_samples = len(y)
    unique_ratio = n_unique / n_samples if n_samples > 0 else 0

    # Float dtype with high uniqueness ratio -> likely regression
    if np.issubdtype(y.dtype, np.floating):
        if unique_ratio > 0.5:
            return ("regression", 0.9)
        elif n_unique > 20:
            return ("regression", 0.7)
        else:
            # Ambiguous: could be binned continuous or real categories
            return ("classification", 0.5)

    # Integer dtype
    if n_unique == 2:
        return ("classification", 0.95)  # Binary is almost always classification
    elif n_unique <= 10:
        return ("classification", 0.7)
    elif n_unique <= 20:
        return ("classification", 0.5)  # Ambiguous
    else:
        return ("regression", 0.6)  # Many unique integers could be either


class MLPreprocessingService:
    """
    Stateless service for ML preprocessing on AnnData objects.

    Provides:
    - Missing value handling (imputation, filtering)
    - Feature scaling (standard, robust, minmax)
    - Class imbalance handling (SMOTE, undersampling)
    - Categorical encoding

    All methods return the standard lobster 3-tuple:
    (AnnData, stats_dict, AnalysisStep)
    """

    def _create_missing_values_ir(
        self,
        feature_threshold: float,
        sample_threshold: float,
        imputation_strategy: str,
        fill_value: Optional[float],
    ) -> AnalysisStep:
        """
        Create IR for missing value handling operation.

        Returns AnalysisStep with ParameterSpec objects and executable Jinja2 template.
        """
        parameter_schema = {
            "feature_threshold": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=feature_threshold,
                required=False,
                validation_rule="0 <= feature_threshold <= 1",
                description="Remove features with >threshold missing values",
            ),
            "sample_threshold": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=sample_threshold,
                required=False,
                validation_rule="0 <= sample_threshold <= 1",
                description="Remove samples with >threshold missing values",
            ),
            "imputation_strategy": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=imputation_strategy,
                required=False,
                validation_rule="imputation_strategy in ['mean', 'median', 'most_frequent', 'constant']",
                description="Strategy for imputing remaining NaNs",
            ),
            "fill_value": ParameterSpec(
                param_type="Optional[float]",
                papermill_injectable=True,
                default_value=fill_value,
                required=False,
                description="Value for 'constant' imputation strategy",
            ),
        }

        code_template = """# Missing value handling: filter and impute
import numpy as np
from sklearn.impute import SimpleImputer
from lobster.services.ml.sparse_utils import check_sparse_conversion_safe

# Extract data matrix
X = adata.X
if hasattr(X, "toarray"):
    check_sparse_conversion_safe(X, overhead_multiplier=1.5)
    X = X.toarray()

# Ensure float type for NaN detection
X = X.astype(float)
original_shape = X.shape
print(f"Original shape: {original_shape}")

# Detect missing values
missing_mask = np.isnan(X)
n_missing_original = missing_mask.sum()
missing_rate_original = n_missing_original / (original_shape[0] * original_shape[1])
print(f"Missing values: {n_missing_original} ({missing_rate_original:.2%})")

# Calculate missing rates per feature and sample
feature_missing_rate = np.mean(missing_mask, axis=0)
sample_missing_rate = np.mean(missing_mask, axis=1)

# Filter features with too many missing values
keep_features = feature_missing_rate <= {{ feature_threshold }}
n_features_removed = np.sum(~keep_features)
print(f"Removing {n_features_removed} features with >{{ feature_threshold }} missing")

# Filter samples with too many missing values
keep_samples = sample_missing_rate <= {{ sample_threshold }}
n_samples_removed = np.sum(~keep_samples)
print(f"Removing {n_samples_removed} samples with >{{ sample_threshold }} missing")

# Apply filters
X_filtered = X[keep_samples][:, keep_features]
missing_after_filter = np.isnan(X_filtered).sum()
print(f"After filtering: {X_filtered.shape}, {missing_after_filter} missing values remain")

# Impute remaining missing values
if missing_after_filter > 0 and "{{ imputation_strategy }}" != "none":
    {% if imputation_strategy == "constant" %}
    {% if fill_value is not none %}
    fill_value = {{ fill_value }}
    {% else %}
    fill_value = 0
    {% endif %}
    imputer = SimpleImputer(strategy="{{ imputation_strategy }}", fill_value=fill_value)
    {% else %}
    imputer = SimpleImputer(strategy="{{ imputation_strategy }}")
    {% endif %}
    X_imputed = imputer.fit_transform(X_filtered)
    print(f"Imputed {missing_after_filter} values using {{ imputation_strategy }} strategy")
else:
    X_imputed = X_filtered
    print("No imputation needed")

# Create filtered AnnData
adata = adata[keep_samples, keep_features].copy()
adata.X = X_imputed

# Store metadata
adata.uns["missing_value_handling"] = {
    "feature_threshold": {{ feature_threshold }},
    "sample_threshold": {{ sample_threshold }},
    "imputation_strategy": "{{ imputation_strategy }}",
    "features_removed": int(n_features_removed),
    "samples_removed": int(n_samples_removed),
}

print(f"Final shape: {adata.shape}")
"""

        return AnalysisStep(
            operation="sklearn.impute.SimpleImputer",
            tool_name="handle_missing_values",
            description=f"Handle missing values: filter features/samples by threshold, impute with {imputation_strategy}",
            library="sklearn",
            code_template=code_template,
            imports=[
                "import numpy as np",
                "from sklearn.impute import SimpleImputer",
                "from lobster.services.ml.sparse_utils import check_sparse_conversion_safe",
            ],
            parameters={
                "feature_threshold": feature_threshold,
                "sample_threshold": sample_threshold,
                "imputation_strategy": imputation_strategy,
                "fill_value": fill_value,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "missing_value_handling",
                "sklearn_version": "1.3+",
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_scale_features_ir(
        self,
        method: str,
        with_mean: bool,
        with_std: bool,
        feature_range: Tuple[float, float],
    ) -> AnalysisStep:
        """
        Create IR for feature scaling operation.

        Returns AnalysisStep with ParameterSpec objects and executable Jinja2 template.
        """
        parameter_schema = {
            "method": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=method,
                required=True,
                validation_rule="method in ['standard', 'robust', 'minmax', 'maxabs']",
                description="Scaling method to apply",
            ),
            "with_mean": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=with_mean,
                required=False,
                description="Center data (StandardScaler only)",
            ),
            "with_std": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=with_std,
                required=False,
                description="Scale to unit variance (StandardScaler only)",
            ),
            "feature_range": ParameterSpec(
                param_type="Tuple[float, float]",
                papermill_injectable=True,
                default_value=list(feature_range),
                required=False,
                description="Target range for MinMaxScaler",
            ),
        }

        code_template = """# Feature scaling with {{ method }} scaler
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from lobster.services.ml.sparse_utils import check_sparse_conversion_safe

# Extract data matrix
X = adata.X
if hasattr(X, "toarray"):
    check_sparse_conversion_safe(X, overhead_multiplier=1.5)
    X = X.toarray()

print(f"Data shape: {X.shape}")
print(f"Data type: {X.dtype}")

# Calculate original statistics
original_mean = float(np.mean(X))
original_std = float(np.std(X))
original_min = float(np.min(X))
original_max = float(np.max(X))
print(f"Original data statistics:")
print(f"  Mean: {original_mean:.4f}")
print(f"  Std: {original_std:.4f}")
print(f"  Range: [{original_min:.4f}, {original_max:.4f}]")

# Select and configure scaler based on method
print(f"\\nScaling method: {{ method }}")
{% if method == "standard" %}
# StandardScaler: (X - mean) / std
# Good for: normally distributed features, algorithms sensitive to scale (SVM, neural nets)
scaler = StandardScaler(with_mean={{ with_mean }}, with_std={{ with_std }})
print("Using StandardScaler (z-score normalization)")
print(f"  with_mean={{ with_mean }}, with_std={{ with_std }}")
{% elif method == "robust" %}
# RobustScaler: (X - median) / IQR
# Good for: data with outliers, uses median and interquartile range
scaler = RobustScaler()
print("Using RobustScaler (median and IQR)")
print("  Robust to outliers")
{% elif method == "minmax" %}
# MinMaxScaler: (X - min) / (max - min) * (max_range - min_range) + min_range
# Good for: neural networks, bounded optimization
feature_range = tuple({{ feature_range }})
scaler = MinMaxScaler(feature_range=feature_range)
print(f"Using MinMaxScaler (range={feature_range})")
print("  Scales to specified range")
{% elif method == "maxabs" %}
# MaxAbsScaler: X / max(abs(X))
# Good for: sparse data, scales to [-1, 1] without shifting
scaler = MaxAbsScaler()
print("Using MaxAbsScaler ([-1, 1] for sparse data)")
print("  Preserves zero entries (good for sparse matrices)")
{% endif %}

# Fit and transform
print("\\nFitting scaler...")
X_scaled = scaler.fit_transform(X)
print("Scaling complete")

# Store scaler parameters for reproducibility
scaler_params = {"method": "{{ method }}"}
{% if method == "standard" %}
scaler_params["mean"] = scaler.mean_.tolist()
scaler_params["scale"] = scaler.scale_.tolist()
scaler_params["with_mean"] = {{ with_mean }}
scaler_params["with_std"] = {{ with_std }}
print(f"Scaler mean: {np.mean(scaler.mean_):.4f}")
print(f"Scaler scale: {np.mean(scaler.scale_):.4f}")
{% elif method == "robust" %}
scaler_params["center"] = scaler.center_.tolist()
scaler_params["scale"] = scaler.scale_.tolist()
print(f"Scaler center (median): {np.mean(scaler.center_):.4f}")
print(f"Scaler scale (IQR): {np.mean(scaler.scale_):.4f}")
{% elif method == "minmax" %}
scaler_params["min"] = scaler.data_min_.tolist()
scaler_params["max"] = scaler.data_max_.tolist()
scaler_params["feature_range"] = feature_range
print(f"Data min: {np.mean(scaler.data_min_):.4f}")
print(f"Data max: {np.mean(scaler.data_max_):.4f}")
{% elif method == "maxabs" %}
scaler_params["max_abs"] = scaler.max_abs_.tolist()
print(f"Max absolute value: {np.mean(scaler.max_abs_):.4f}")
{% endif %}

# Update AnnData
adata.X = X_scaled
adata.uns["scaler"] = scaler_params

# Calculate and report statistics
scaled_mean = float(np.mean(X_scaled))
scaled_std = float(np.std(X_scaled))
scaled_min = float(np.min(X_scaled))
scaled_max = float(np.max(X_scaled))

print(f"\\nScaled data statistics:")
print(f"  Mean: {scaled_mean:.4f}")
print(f"  Std: {scaled_std:.4f}")
print(f"  Range: [{scaled_min:.4f}, {scaled_max:.4f}]")
print(f"\\nScaling complete for {X.shape[1]} features")
"""

        return AnalysisStep(
            operation=f"sklearn.preprocessing.{method}Scaler",
            tool_name="scale_features",
            description=f"Scale features using {method} scaler",
            library="sklearn",
            code_template=code_template,
            imports=[
                "import numpy as np",
                "from sklearn.preprocessing import StandardScaler",
                "from sklearn.preprocessing import RobustScaler",
                "from sklearn.preprocessing import MinMaxScaler",
                "from sklearn.preprocessing import MaxAbsScaler",
                "from lobster.services.ml.sparse_utils import check_sparse_conversion_safe",
            ],
            parameters={
                "method": method,
                "with_mean": with_mean,
                "with_std": with_std,
                "feature_range": list(feature_range),
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "feature_scaling",
                "sklearn_version": "1.3+",
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_class_imbalance_ir(
        self,
        target_column: str,
        method: str,
        sampling_strategy: Union[str, float, Dict],
        random_state: int,
        task_type: str,
        k_neighbors: int,
        balance_threshold: float,
    ) -> AnalysisStep:
        """
        Create IR for class imbalance handling operation.

        Returns AnalysisStep with ParameterSpec objects and executable Jinja2 template.
        """
        parameter_schema = {
            "target_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=target_column,
                required=True,
                description="Column in obs containing class labels",
            ),
            "method": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=method,
                required=True,
                validation_rule="method in ['smote', 'random_under', 'smote_tomek']",
                description="Resampling method for class balancing",
            ),
            "sampling_strategy": ParameterSpec(
                param_type="Union[str, float, Dict]",
                papermill_injectable=True,
                default_value=str(sampling_strategy),
                required=False,
                description="Sampling strategy (see imbalanced-learn docs)",
            ),
            "random_state": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=random_state,
                required=False,
                description="Random seed for reproducibility",
            ),
            "task_type": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=task_type,
                required=True,
                validation_rule="task_type == 'classification'",
                description="Task type (must be 'classification' for SMOTE)",
            ),
            "k_neighbors": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=k_neighbors,
                required=False,
                validation_rule="k_neighbors >= 1",
                description="Number of nearest neighbors for SMOTE",
            ),
            "balance_threshold": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=balance_threshold,
                required=False,
                validation_rule="0 < balance_threshold <= 1",
                description="Skip SMOTE if min/max class ratio exceeds threshold",
            ),
        }

        code_template = """# Class imbalance handling with {{ method }} - metadata preserving
import uuid
import numpy as np
import pandas as pd
import anndata
from sklearn.preprocessing import LabelEncoder
from lobster.services.ml.sparse_utils import check_sparse_conversion_safe
{% if method == "smote" %}
from imblearn.over_sampling import SMOTE
{% elif method == "random_under" %}
from imblearn.under_sampling import RandomUnderSampler
{% elif method == "smote_tomek" %}
from imblearn.combine import SMOTETomek
{% endif %}

# Configuration
target_column = "{{ target_column }}"
task_type = "{{ task_type }}"
k_neighbors = {{ k_neighbors }}
balance_threshold = {{ balance_threshold }}

print(f"Task type: {task_type}")
print(f"Target column: {target_column}")

# Validate task type
if task_type != "classification":
    raise ValueError(f"SMOTE is only for classification tasks, got {task_type}")

# Extract features and target
X = adata.X
if hasattr(X, "toarray"):
    check_sparse_conversion_safe(X, overhead_multiplier=2.0)  # SMOTE creates synthetic samples
    X = X.toarray()

y = adata.obs[target_column].values
n_original = X.shape[0]

# Encode target if categorical
label_encoder_classes = None
if y.dtype == object or str(y.dtype) == "category":
    le = LabelEncoder()
    y = le.fit_transform(y)
    label_encoder_classes = le.classes_.tolist()
    print(f"Encoded categorical target: {label_encoder_classes}")

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(unique, counts))
min_count, max_count = min(counts), max(counts)
balance_ratio = min_count / max_count
print(f"Original distribution: {class_distribution}")
print(f"Balance ratio: {balance_ratio:.2f}")

# Check if already balanced
if balance_ratio >= balance_threshold:
    print(f"Classes already balanced (ratio >= {balance_threshold}), skipping SMOTE")
    adata.obs["is_synthetic"] = False
else:
    # Validate minimum samples
    for class_label, count in class_distribution.items():
        if count < k_neighbors + 1:
            raise ValueError(f"Class {class_label} has {count} samples, need {k_neighbors + 1}")

    # Configure resampler
    {% if method == "smote" %}
    resampler = SMOTE(
        k_neighbors=k_neighbors,
        sampling_strategy="{{ sampling_strategy }}",
        random_state={{ random_state }}
    )
    print(f"Using SMOTE with k_neighbors={k_neighbors}")
    {% elif method == "random_under" %}
    resampler = RandomUnderSampler(
        sampling_strategy="{{ sampling_strategy }}",
        random_state={{ random_state }}
    )
    print("Using RandomUnderSampler")
    {% elif method == "smote_tomek" %}
    resampler = SMOTETomek(
        sampling_strategy="{{ sampling_strategy }}",
        random_state={{ random_state }}
    )
    print("Using SMOTETomek")
    {% endif %}

    # Resample data
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    n_resampled = X_resampled.shape[0]
    n_synthetic = n_resampled - n_original
    print(f"Resampled: {n_original} -> {n_resampled} samples ({n_synthetic} synthetic)")

    # CRITICAL: Preserve original obs for real samples
    original_obs = adata.obs.copy()
    original_obs["is_synthetic"] = False

    # Create synthetic sample obs with UUID IDs
    synthetic_ids = [f"synthetic_{uuid.uuid4().hex[:8]}" for _ in range(n_synthetic)]
    synthetic_obs = pd.DataFrame(index=synthetic_ids)
    synthetic_obs[target_column] = y_resampled[n_original:]
    for col in original_obs.columns:
        if col not in [target_column, "is_synthetic"]:
            synthetic_obs[col] = pd.NA
    synthetic_obs["is_synthetic"] = True

    # Concatenate obs (original first, then synthetic)
    new_obs = pd.concat([original_obs, synthetic_obs], axis=0)

    # Create new AnnData
    adata = anndata.AnnData(
        X=X_resampled,
        obs=new_obs,
        var=adata.var.copy(),
    )

    # Store metadata
    adata.uns["smote_config"] = {
        "method": "{{ method }}",
        "k_neighbors": k_neighbors,
        "sampling_strategy": "{{ sampling_strategy }}",
        "random_state": {{ random_state }},
        "balance_threshold": balance_threshold,
        "target_column": target_column,
    }
    if label_encoder_classes is not None:
        adata.uns["label_encoder_classes"] = label_encoder_classes

    new_distribution = dict(zip(*np.unique(y_resampled, return_counts=True)))
    adata.uns["class_imbalance_handling"] = {
        "method": "{{ method }}",
        "original_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "resampled_distribution": {str(k): int(v) for k, v in new_distribution.items()},
    }
    print(f"Resampled distribution: {new_distribution}")

print("Class imbalance handling complete")
"""

        return AnalysisStep(
            operation=f"imblearn.{method}",
            tool_name="handle_class_imbalance",
            description=f"Handle class imbalance using {method} resampling with metadata preservation",
            library="imblearn",
            code_template=code_template,
            imports=[
                "import uuid",
                "import numpy as np",
                "import pandas as pd",
                "import anndata",
                "from sklearn.preprocessing import LabelEncoder",
                "from lobster.services.ml.sparse_utils import check_sparse_conversion_safe",
                "from imblearn.over_sampling import SMOTE",
                "from imblearn.under_sampling import RandomUnderSampler",
                "from imblearn.combine import SMOTETomek",
            ],
            parameters={
                "target_column": target_column,
                "method": method,
                "sampling_strategy": str(sampling_strategy),
                "random_state": random_state,
                "task_type": task_type,
                "k_neighbors": k_neighbors,
                "balance_threshold": balance_threshold,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "class_imbalance_handling",
                "imblearn_version": "0.11+",
                "method": method,
                "sampling_strategy": str(sampling_strategy),
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_categorical_encoding_ir(
        self,
        columns: List[str],
        method: str,
        drop_first: bool,
    ) -> AnalysisStep:
        """
        Create IR for categorical encoding operation.

        Returns AnalysisStep with ParameterSpec objects and executable Jinja2 template.
        """
        parameter_schema = {
            "columns": ParameterSpec(
                param_type="List[str]",
                papermill_injectable=True,
                default_value=columns,
                required=True,
                description="Columns to encode (categorical)",
            ),
            "method": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=method,
                required=True,
                validation_rule="method in ['onehot', 'label', 'ordinal']",
                description="Encoding method",
            ),
            "drop_first": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=drop_first,
                required=False,
                description="Drop first category (one-hot encoding only)",
            ),
        }

        code_template = """# Categorical encoding with {{ method }} encoder
import numpy as np
{% if method == "label" %}
from sklearn.preprocessing import LabelEncoder
{% elif method == "onehot" %}
from sklearn.preprocessing import OneHotEncoder
{% elif method == "ordinal" %}
from sklearn.preprocessing import OrdinalEncoder
{% endif %}

# Configuration
columns = {{ columns }}
encoding_info = {}
n_columns = len(columns)

print(f"Starting categorical encoding:")
print(f"  Method: {{ method }}")
print(f"  Columns: {columns}")
print(f"  Total columns: {n_columns}")

# Encoding method details
{% if method == "label" %}
print("\\nLabel Encoding:")
print("  - Converts each category to integer (0, 1, 2, ...)")
print("  - Simple and memory-efficient")
print("  - Implies ordinal relationship (not always desired)")
{% elif method == "onehot" %}
print("\\nOne-Hot Encoding:")
print("  - Creates binary column for each category")
print("  - No ordinal assumption")
print(f"  - Drop first: {{ drop_first }}")
{% elif method == "ordinal" %}
print("\\nOrdinal Encoding:")
print("  - Converts categories to ordered integers")
print("  - Use when categories have natural order")
{% endif %}

# Process each column
for col_idx, col in enumerate(columns, 1):
    print(f"\\n[{col_idx}/{n_columns}] Processing column: {col}")

    values = adata.obs[col].values
    n_samples = len(values)
    n_unique = len(np.unique(values))
    print(f"  Samples: {n_samples}")
    print(f"  Unique values: {n_unique}")

    {% if method == "label" %}
    # Label encoding: categorical -> integer
    # Best for: tree-based models, ordinal data
    le = LabelEncoder()
    encoded = le.fit_transform(values)
    new_col_name = f"{col}_encoded"
    adata.obs[new_col_name] = encoded

    encoding_info[col] = {
        "method": "label",
        "classes": le.classes_.tolist(),
        "n_classes": len(le.classes_),
        "new_column": new_col_name,
    }

    print(f"  Classes: {le.classes_.tolist()}")
    print(f"  Encoded range: [0, {len(le.classes_)-1}]")
    print(f"  New column: {new_col_name}")

    {% elif method == "onehot" %}
    # One-hot encoding: categorical -> binary columns
    # Best for: linear models, neural networks
    ohe = OneHotEncoder(sparse_output=False, drop="{{ "first" if drop_first else "None" }}")
    encoded = ohe.fit_transform(values.reshape(-1, 1))
    categories = ohe.categories_[0]

    {% if drop_first %}
    # Drop first category to avoid multicollinearity
    categories = categories[1:]
    dropped_category = ohe.categories_[0][0]
    print(f"  Dropped first category: {dropped_category}")
    {% endif %}

    new_col_names = []
    for i, cat in enumerate(categories):
        new_col_name = f"{col}_{cat}"
        adata.obs[new_col_name] = encoded[:, i]
        new_col_names.append(new_col_name)

    encoding_info[col] = {
        "method": "onehot",
        "categories": list(categories),
        "n_categories": len(categories),
        "new_columns": new_col_names,
        {% if drop_first %}
        "dropped_first": True,
        "dropped_category": dropped_category,
        {% else %}
        "dropped_first": False,
        {% endif %}
    }

    print(f"  Categories: {list(categories)}")
    print(f"  Created {len(categories)} binary columns")
    print(f"  New columns: {new_col_names}")

    {% elif method == "ordinal" %}
    # Ordinal encoding: categorical -> ordered integer
    # Best for: ordered categories (low/medium/high)
    oe = OrdinalEncoder()
    encoded = oe.fit_transform(values.reshape(-1, 1))
    new_col_name = f"{col}_ordinal"
    adata.obs[new_col_name] = encoded.flatten()

    encoding_info[col] = {
        "method": "ordinal",
        "categories": oe.categories_[0].tolist(),
        "n_categories": len(oe.categories_[0]),
        "new_column": new_col_name,
    }

    print(f"  Categories (ordered): {oe.categories_[0].tolist()}")
    print(f"  Encoded range: [0, {len(oe.categories_[0])-1}]")
    print(f"  New column: {new_col_name}")
    {% endif %}

# Store metadata
adata.uns["categorical_encoding"] = encoding_info

# Summary
print(f"\\nEncoding complete:")
print(f"  Method: {{ method }}")
print(f"  Columns processed: {len(columns)}")
{% if method == "onehot" %}
total_new_cols = sum(len(info.get("new_columns", [])) for info in encoding_info.values())
print(f"  Total new columns: {total_new_cols}")
{% endif %}
"""

        return AnalysisStep(
            operation=f"sklearn.preprocessing.{method}Encoder",
            tool_name="encode_categorical",
            description=f"Encode categorical columns using {method} encoder",
            library="sklearn",
            code_template=code_template,
            imports=[
                "import numpy as np",
                "from sklearn.preprocessing import LabelEncoder",
                "from sklearn.preprocessing import OneHotEncoder",
                "from sklearn.preprocessing import OrdinalEncoder",
            ],
            parameters={
                "columns": columns,
                "method": method,
                "drop_first": drop_first,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "categorical_encoding",
                "sklearn_version": "1.3+",
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def handle_missing_values(
        self,
        adata: AnnData,
        feature_threshold: float = 0.5,
        sample_threshold: float = 0.5,
        imputation_strategy: str = "median",
        fill_value: Optional[float] = None,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Handle missing values by filtering and/or imputation.

        Args:
            adata: AnnData with potential missing values in X
            feature_threshold: Remove features with >threshold missing values
            sample_threshold: Remove samples with >threshold missing values
            imputation_strategy: Strategy for remaining NaNs ("mean", "median", "constant", "most_frequent")
            fill_value: Value for "constant" strategy

        Returns:
            Tuple of (adata, stats, ir)
        """
        X = adata.X
        if hasattr(X, "toarray"):
            check_sparse_conversion_safe(X, overhead_multiplier=1.5)
            X = X.toarray()

        X = X.astype(float)  # Ensure float for NaN detection
        original_shape = X.shape

        # Detect missing values (NaN and zero depending on data type)
        missing_mask = np.isnan(X)
        n_missing_original = missing_mask.sum()

        # Calculate missing rates
        feature_missing_rate = np.mean(missing_mask, axis=0)
        sample_missing_rate = np.mean(missing_mask, axis=1)

        # Filter features with too many missing values
        keep_features = feature_missing_rate <= feature_threshold
        n_features_removed = np.sum(~keep_features)

        # Filter samples with too many missing values
        keep_samples = sample_missing_rate <= sample_threshold
        n_samples_removed = np.sum(~keep_samples)

        # Apply filters
        X_filtered = X[keep_samples][:, keep_features]
        missing_after_filter = np.isnan(X_filtered).sum()

        # Impute remaining missing values
        if missing_after_filter > 0 and imputation_strategy != "none":
            from sklearn.impute import SimpleImputer

            if imputation_strategy == "constant" and fill_value is None:
                fill_value = 0

            imputer = SimpleImputer(
                strategy=imputation_strategy,
                fill_value=fill_value,
            )
            X_imputed = imputer.fit_transform(X_filtered)
        else:
            X_imputed = X_filtered

        # Create new AnnData
        adata = adata.copy()
        adata = adata[keep_samples, keep_features].copy()
        adata.X = X_imputed

        # Store preprocessing info
        adata.uns["missing_value_handling"] = {
            "feature_threshold": feature_threshold,
            "sample_threshold": sample_threshold,
            "imputation_strategy": imputation_strategy,
            "features_removed": int(n_features_removed),
            "samples_removed": int(n_samples_removed),
        }

        stats = {
            "original_shape": original_shape,
            "final_shape": X_imputed.shape,
            "n_missing_original": int(n_missing_original),
            "missing_rate_original": float(
                n_missing_original / (original_shape[0] * original_shape[1])
            ),
            "n_features_removed": int(n_features_removed),
            "n_samples_removed": int(n_samples_removed),
            "n_values_imputed": int(missing_after_filter),
            "imputation_strategy": imputation_strategy,
        }

        ir = self._create_missing_values_ir(
            feature_threshold=feature_threshold,
            sample_threshold=sample_threshold,
            imputation_strategy=imputation_strategy,
            fill_value=fill_value,
        )

        logger.info(
            f"Missing value handling: {original_shape} -> {X_imputed.shape}, imputed {missing_after_filter} values"
        )

        return adata, stats, ir

    def scale_features(
        self,
        adata: AnnData,
        method: str = "standard",
        with_mean: bool = True,
        with_std: bool = True,
        feature_range: Tuple[float, float] = (0, 1),
        store_scaler: bool = True,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Scale features using various strategies.

        Args:
            adata: AnnData with features in X
            method: Scaling method ("standard", "robust", "minmax", "maxabs")
            with_mean: Center data (for standard scaler)
            with_std: Scale to unit variance (for standard scaler)
            feature_range: Range for MinMaxScaler
            store_scaler: Store fitted scaler in adata.uns for transform

        Returns:
            Tuple of (adata, stats, ir)
        """
        from sklearn.preprocessing import (
            MaxAbsScaler,
            MinMaxScaler,
            RobustScaler,
            StandardScaler,
        )

        X = adata.X
        if hasattr(X, "toarray"):
            check_sparse_conversion_safe(X, overhead_multiplier=1.5)
            X = X.toarray()

        # Select scaler
        if method == "standard":
            scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "minmax":
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == "maxabs":
            scaler = MaxAbsScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        # Fit and transform
        X_scaled = scaler.fit_transform(X)

        # Store results
        adata = adata.copy()
        adata.X = X_scaled

        if store_scaler:
            # Store scaler parameters (not the object itself for serialization)
            scaler_params = {
                "method": method,
            }
            if method == "standard":
                scaler_params["mean"] = scaler.mean_.tolist()
                scaler_params["scale"] = scaler.scale_.tolist()
            elif method == "robust":
                scaler_params["center"] = scaler.center_.tolist()
                scaler_params["scale"] = scaler.scale_.tolist()
            elif method == "minmax":
                scaler_params["min"] = scaler.data_min_.tolist()
                scaler_params["max"] = scaler.data_max_.tolist()

            adata.uns["scaler"] = scaler_params

        # Calculate statistics
        stats = {
            "method": method,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "original_mean": float(np.mean(X)),
            "original_std": float(np.std(X)),
            "scaled_mean": float(np.mean(X_scaled)),
            "scaled_std": float(np.std(X_scaled)),
            "scaled_min": float(np.min(X_scaled)),
            "scaled_max": float(np.max(X_scaled)),
        }

        ir = self._create_scale_features_ir(
            method=method,
            with_mean=with_mean,
            with_std=with_std,
            feature_range=feature_range,
        )

        logger.info(
            f"Scaled features using {method}: mean={stats['scaled_mean']:.4f}, std={stats['scaled_std']:.4f}"
        )

        return adata, stats, ir

    def handle_class_imbalance(
        self,
        adata: AnnData,
        target_column: str,
        task_type: Optional[str] = None,
        method: str = "smote",
        sampling_strategy: Union[str, float, Dict] = "auto",
        k_neighbors: int = 5,
        random_state: int = 42,
        balance_threshold: float = 0.8,
        force: bool = False,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Handle class imbalance using various resampling strategies.

        CRITICAL: task_type parameter is REQUIRED (no silent fallback).
        Use infer_task_type() to get guidance if needed.

        Args:
            adata: AnnData with features in X and target in obs
            target_column: Column in obs containing class labels
            task_type: REQUIRED - "classification" or "regression" (use infer_task_type for guidance)
            method: Resampling method ("smote", "random_under", "smote_tomek", "cluster_centroids")
            sampling_strategy: Sampling strategy (see imbalanced-learn docs)
            k_neighbors: Number of neighbors for SMOTE (default: 5)
            random_state: Random seed
            balance_threshold: Skip SMOTE if min/max class ratio >= threshold (default: 0.8)
            force: Force processing even if estimated output >2GB (default: False)

        Returns:
            Tuple of (adata, stats, ir)

        Raises:
            ValueError: If task_type not provided or target_column not found
            MemoryError: If estimated output >2GB without force=True
        """
        # REQUIRED PARAMETER VALIDATION
        if task_type is None:
            # Get suggestion for error message
            y_temp = (
                adata.obs[target_column].values
                if target_column in adata.obs.columns
                else None
            )
            if y_temp is not None:
                suggestion, confidence = infer_task_type(y_temp)
                error_msg = (
                    f"task_type parameter required for handle_class_imbalance. "
                    f"Use infer_task_type(y) for guidance. "
                    f"Suggestion: task_type='{suggestion}' (confidence: {confidence:.2f})"
                )
            else:
                error_msg = (
                    f"task_type parameter required for handle_class_imbalance. "
                    f"Use infer_task_type(y) for guidance."
                )
            raise ValueError(error_msg)

        if task_type not in ["classification", "regression"]:
            raise ValueError(
                f"task_type must be 'classification' or 'regression', got: {task_type}"
            )

        try:
            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
        except ImportError:
            raise ImportError(
                "imbalanced-learn not installed. Run: pip install lobster-ml[imbalanced]"
            )

        if target_column not in adata.obs.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        X = adata.X
        if hasattr(X, "toarray"):
            check_sparse_conversion_safe(
                X, overhead_multiplier=2.0
            )  # SMOTE creates synthetic samples
            X = X.toarray()

        y = adata.obs[target_column].values
        n_original = X.shape[0]

        # Encode if categorical
        from sklearn.preprocessing import LabelEncoder

        label_encoder_classes = None
        if y.dtype == object or str(y.dtype) == "category":
            le = LabelEncoder()
            y = le.fit_transform(y)
            label_encoder_classes = le.classes_.tolist()
            class_names = label_encoder_classes
        else:
            class_names = list(map(str, np.unique(y)))

        # Calculate class distribution
        unique, counts = np.unique(y, return_counts=True)
        original_distribution = dict(zip(unique, counts))
        min_count = min(counts)
        max_count = max(counts)

        # Check balance threshold - skip if already balanced
        balance_ratio = min_count / max_count
        if balance_ratio >= balance_threshold:
            # Already balanced - return unchanged with is_synthetic=False for all
            adata_out = adata.copy()
            adata_out.obs["is_synthetic"] = False
            stats = {
                "skipped": True,
                "reason": f"Classes already balanced (ratio={balance_ratio:.2f} >= threshold={balance_threshold})",
                "original_distribution": {
                    str(k): int(v) for k, v in original_distribution.items()
                },
                "balance_ratio": balance_ratio,
            }
            ir = self._create_class_imbalance_ir(
                target_column=target_column,
                method=method,
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                task_type=task_type,
                k_neighbors=k_neighbors,
                balance_threshold=balance_threshold,
            )
            logger.info(
                f"Skipping SMOTE: classes already balanced (ratio={balance_ratio:.2f})"
            )
            return adata_out, stats, ir

        # Validate minimum samples per class for SMOTE
        if method in ("smote", "smote_tomek"):
            for class_label, count in original_distribution.items():
                if count < k_neighbors + 1:
                    class_name = class_label
                    if label_encoder_classes is not None:
                        class_name = label_encoder_classes[int(class_label)]
                    raise ValueError(
                        f"Class '{class_name}' has {count} samples, "
                        f"requires at least {k_neighbors + 1} for k_neighbors={k_neighbors}"
                    )

        # Memory estimation - estimate output size before processing
        n_features = adata.shape[1]
        # Conservative estimate: majority class count * number of classes
        n_output_estimate = max_count * len(unique)
        bytes_per_element = 8  # float64
        estimated_bytes = n_output_estimate * n_features * bytes_per_element
        estimated_gb = estimated_bytes / (1024**3)

        if estimated_gb > 2.0 and not force:
            raise MemoryError(
                f"Estimated output size {estimated_gb:.1f}GB exceeds 2GB threshold. "
                f"Dataset: {n_output_estimate} samples x {n_features} features. "
                f"Use force=True to proceed."
            )
        elif estimated_gb > 2.0:
            logger.warning(
                f"Processing large dataset: estimated {estimated_gb:.1f}GB output"
            )

        # Select resampler
        if method == "smote":
            resampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=random_state,
            )
        elif method == "random_under":
            resampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy, random_state=random_state
            )
        elif method == "smote_tomek":
            resampler = SMOTETomek(
                sampling_strategy=sampling_strategy, random_state=random_state
            )
        elif method == "cluster_centroids":
            resampler = ClusterCentroids(
                sampling_strategy=sampling_strategy, random_state=random_state
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Resample - returns indices for real samples + synthetic samples
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        n_resampled = X_resampled.shape[0]
        n_synthetic = n_resampled - n_original

        new_distribution = dict(zip(*np.unique(y_resampled, return_counts=True)))

        # CRITICAL FIX: Preserve original obs for real samples, create minimal obs for synthetic
        # Determine which samples are real (from original) vs synthetic (newly generated)
        if method in ["smote", "smote_tomek"]:
            # Oversampling methods: first n_original are real, rest are synthetic
            # Create is_synthetic flag
            is_synthetic = np.zeros(n_resampled, dtype=bool)
            is_synthetic[n_original:] = True

            # Preserve ALL original obs columns for real samples
            real_obs = adata.obs.copy()

            # Create synthetic sample metadata with UUID IDs
            synthetic_obs_list = []
            for i in range(n_synthetic):
                synthetic_id = f"synthetic_{uuid.uuid4().hex[:8]}"
                synthetic_row = {target_column: y_resampled[n_original + i]}
                synthetic_obs_list.append(pd.Series(synthetic_row, name=synthetic_id))

            if synthetic_obs_list:
                synthetic_obs = pd.DataFrame(synthetic_obs_list)
                # Align columns: synthetic_obs gets all columns from real_obs, filled with NaN
                for col in real_obs.columns:
                    if col not in synthetic_obs.columns:
                        synthetic_obs[col] = np.nan
                # Reorder to match real_obs
                synthetic_obs = synthetic_obs[real_obs.columns]

                # Concatenate real + synthetic
                new_obs = pd.concat(
                    [real_obs, synthetic_obs], axis=0, ignore_index=False
                )
            else:
                new_obs = real_obs

            # Add is_synthetic column
            new_obs["is_synthetic"] = is_synthetic

        else:
            # Undersampling methods: all samples are real (subset of original)
            # This is harder - we need to figure out which original samples were kept
            # For now, create basic obs with is_synthetic=False
            new_obs_names = [f"sample_{i}" for i in range(n_resampled)]
            new_obs = pd.DataFrame(
                {target_column: y_resampled, "is_synthetic": False},
                index=new_obs_names,
            )

        import anndata

        adata_resampled = anndata.AnnData(
            X=X_resampled,
            obs=new_obs,
            var=adata.var.copy(),
        )

        # Store comprehensive metadata in uns
        adata_resampled.uns["smote_config"] = {
            "method": method,
            "sampling_strategy": str(sampling_strategy),
            "k_neighbors": k_neighbors,
            "random_state": random_state,
            "balance_threshold": balance_threshold,
            "task_type": task_type,
        }

        adata_resampled.uns["class_imbalance_handling"] = {
            "method": method,
            "original_distribution": {
                str(k): int(v) for k, v in original_distribution.items()
            },
            "resampled_distribution": {
                str(k): int(v) for k, v in new_distribution.items()
            },
        }

        # Store label encoder mapping if categorical
        if label_encoder_classes is not None:
            adata_resampled.uns["label_encoder_classes"] = label_encoder_classes

        stats = {
            "method": method,
            "task_type": task_type,
            "original_n_samples": n_original,
            "resampled_n_samples": n_resampled,
            "n_synthetic": n_synthetic,
            "original_distribution": {
                str(k): int(v) for k, v in original_distribution.items()
            },
            "resampled_distribution": {
                str(k): int(v) for k, v in new_distribution.items()
            },
            "samples_added": max(0, n_resampled - n_original),
            "samples_removed": max(0, n_original - n_resampled),
        }

        ir = self._create_class_imbalance_ir(
            target_column=target_column,
            method=method,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            task_type=task_type,
            k_neighbors=k_neighbors,
            balance_threshold=balance_threshold,
        )

        logger.info(
            f"Class imbalance handled: {n_original} -> {n_resampled} samples ({n_synthetic} synthetic) using {method}"
        )

        return adata_resampled, stats, ir

    def encode_categorical(
        self,
        adata: AnnData,
        columns: Optional[List[str]] = None,
        method: str = "onehot",
        drop_first: bool = False,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Encode categorical columns in adata.obs.

        Args:
            adata: AnnData with categorical columns in obs
            columns: Columns to encode (None for auto-detect)
            method: Encoding method ("onehot", "label", "ordinal")
            drop_first: Drop first category (for one-hot)

        Returns:
            Tuple of (adata, stats, ir)
        """
        adata = adata.copy()

        # Auto-detect categorical columns if not specified
        if columns is None:
            columns = [
                col
                for col in adata.obs.columns
                if adata.obs[col].dtype == object
                or str(adata.obs[col].dtype) == "category"
            ]

        if not columns:
            logger.info("No categorical columns found to encode")
            stats = {"n_columns_encoded": 0}
            # Return minimal IR for no-op case
            ir = AnalysisStep(
                operation="no_encoding_needed",
                tool_name="encode_categorical",
                description="No categorical columns found",
                library="sklearn",
                code_template="# No categorical columns to encode\nprint('No categorical columns found')",
                imports=[],
                parameters={},
                parameter_schema={},
                input_entities=["adata"],
                output_entities=["adata"],
            )
            return adata, stats, ir

        encoding_info = {}

        for col in columns:
            values = adata.obs[col].values

            if method == "label":
                from sklearn.preprocessing import LabelEncoder

                le = LabelEncoder()
                encoded = le.fit_transform(values)
                adata.obs[f"{col}_encoded"] = encoded
                encoding_info[col] = {
                    "method": "label",
                    "classes": le.classes_.tolist(),
                }

            elif method == "onehot":
                from sklearn.preprocessing import OneHotEncoder

                ohe = OneHotEncoder(
                    sparse_output=False, drop="first" if drop_first else None
                )
                encoded = ohe.fit_transform(values.reshape(-1, 1))
                categories = ohe.categories_[0]

                if drop_first:
                    categories = categories[1:]

                for i, cat in enumerate(categories):
                    adata.obs[f"{col}_{cat}"] = encoded[:, i]

                encoding_info[col] = {
                    "method": "onehot",
                    "categories": list(categories),
                }

            elif method == "ordinal":
                from sklearn.preprocessing import OrdinalEncoder

                oe = OrdinalEncoder()
                encoded = oe.fit_transform(values.reshape(-1, 1))
                adata.obs[f"{col}_ordinal"] = encoded.flatten()
                encoding_info[col] = {
                    "method": "ordinal",
                    "categories": oe.categories_[0].tolist(),
                }

        adata.uns["categorical_encoding"] = encoding_info

        stats = {
            "n_columns_encoded": len(columns),
            "columns_encoded": columns,
            "method": method,
            "encoding_info": encoding_info,
        }

        ir = self._create_categorical_encoding_ir(
            columns=columns,
            method=method,
            drop_first=drop_first,
        )

        logger.info(f"Encoded {len(columns)} categorical columns using {method}")

        return adata, stats, ir

    def check_availability(self) -> Dict[str, Any]:
        """Check if preprocessing dependencies are available."""
        result = {
            "sklearn_available": True,  # Core dependency
            "ready": True,
        }

        try:
            from imblearn.over_sampling import SMOTE

            result["imblearn_available"] = True
        except ImportError:
            result["imblearn_available"] = False
            result["imbalanced_install"] = "pip install lobster-ml[imbalanced]"

        return result
