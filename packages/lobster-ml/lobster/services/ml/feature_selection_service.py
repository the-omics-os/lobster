"""
Feature Selection Service for biomarker discovery.

Provides stability-based feature selection, LASSO/Elastic Net,
and tree-based importance ranking for high-dimensional omics data.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from anndata import AnnData

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.services.ml.sparse_utils import (
    SparseConversionError,
    check_sparse_conversion_safe,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["FeatureSelectionService"]


class FeatureSelectionService:
    """
    Stateless service for feature selection on AnnData objects.

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
                # Check memory before densification (per Phase 10)
                check_sparse_conversion_safe(X, overhead_multiplier=1.5)
                X = X.toarray()
            feature_names = adata.var_names.tolist()

        return X, feature_names

    def _chunked_variance(
        self,
        X,
        chunk_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute variance per feature using chunked processing.

        Uses Welford's online algorithm for numerical stability:
        - Process one chunk of rows at a time
        - Accumulate mean and M2 (sum of squared deviations)
        - Final variance = M2 / n

        Memory usage: O(n_features) instead of O(n_samples * n_features)

        Args:
            X: Sparse matrix (n_samples, n_features)
            chunk_size: Number of rows per chunk. If None, auto-detect from available memory.

        Returns:
            variances: Per-feature variance array
            n_chunks: Number of chunks processed
        """
        import psutil
        from scipy.sparse import issparse

        n_samples, n_features = X.shape

        # Auto-detect chunk size from available memory
        if chunk_size is None:
            available_bytes = psutil.virtual_memory().available
            # Target 10% of available memory per chunk
            bytes_per_row = n_features * 8  # float64
            chunk_size = max(100, int(available_bytes * 0.1 / bytes_per_row))
            chunk_size = min(chunk_size, n_samples)  # Don't exceed data size

        logger.info(f"Chunked variance: {n_samples} samples in chunks of {chunk_size}")

        # Welford's online algorithm accumulators
        count = 0
        mean = np.zeros(n_features)
        M2 = np.zeros(n_features)  # Sum of squared differences from mean

        n_chunks = (n_samples + chunk_size - 1) // chunk_size

        for i in range(0, n_samples, chunk_size):
            chunk_end = min(i + chunk_size, n_samples)
            chunk_idx = i // chunk_size + 1

            # Progress feedback (Local CLI)
            logger.info(f"Processing chunk {chunk_idx}/{n_chunks}")

            # Get chunk and convert to dense
            chunk = X[i:chunk_end]
            if issparse(chunk):
                chunk = chunk.toarray()

            # Update Welford accumulators for this chunk
            for row in chunk:
                count += 1
                delta = row - mean
                mean += delta / count
                delta2 = row - mean
                M2 += delta * delta2

        # Final variance
        variance = M2 / count if count > 1 else np.zeros(n_features)

        return variance, n_chunks

    def _create_stability_selection_ir(
        self,
        target_column: str,
        n_features: int,
        n_rounds: int,
        subsample_fraction: float,
        method: str,
        random_state: int,
        feature_space_key: Optional[str] = None,
    ) -> AnalysisStep:
        """
        Create IR for stability-based feature selection.

        Implements Meinshausen & Buhlmann (2010) stability selection with
        standard probability calculation: selection_counts / n_rounds.

        Args:
            target_column: Target variable column name
            n_features: Number of top features to select (fallback)
            n_rounds: Number of subsampling rounds
            subsample_fraction: Fraction of data per round
            method: Selection method (xgboost, random_forest, lasso)
            random_state: Random seed
            feature_space_key: If provided, use adata.obsm[key] instead of adata.X

        Returns:
            AnalysisStep with executable stability selection code
        """
        parameter_schema = {
            "target_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=target_column,
                required=True,
                description="Column in obs containing target variable",
            ),
            "n_features": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_features,
                required=False,
                validation_rule="n_features > 0",
                description="Number of top features to select (fallback if threshold yields fewer)",
            ),
            "n_rounds": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_rounds,
                required=False,
                validation_rule="n_rounds > 0",
                description="Number of subsampling rounds",
            ),
            "subsample_fraction": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=subsample_fraction,
                required=False,
                validation_rule="0.0 < subsample_fraction <= 1.0",
                description="Fraction of data to use per round",
            ),
            "method": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=method,
                required=False,
                validation_rule="method in ['xgboost', 'random_forest', 'lasso']",
                description="Selection method",
            ),
            "random_state": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=random_state,
                required=False,
                description="Random seed for reproducibility",
            ),
            "feature_space_key": ParameterSpec(
                param_type="Optional[str]",
                papermill_injectable=True,
                default_value=feature_space_key,
                required=False,
                description="Key in adata.obsm for feature matrix. None uses adata.X. Example: 'X_mofa'",
            ),
        }

        code_template = """# ============================================================
# MEINSHAUSEN & BUHLMANN STABILITY SELECTION
# ============================================================
# Reference: Meinshausen & Buhlmann (2010) - Stability Selection
# JRSS-B, doi:10.1111/j.1467-9868.2010.00740.x
#
# Key principle: selection_probability = selection_counts / n_rounds
# Features with probability >= 0.6 are selected (M&B default threshold)
# ============================================================

import numpy as np
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("STABILITY SELECTION (Meinshausen & Buhlmann 2010)")
print("=" * 60)

# ============================================================
# Input Validation
# ============================================================
if "{{ target_column }}" not in adata.obs.columns:
    raise ValueError(f"Target column '{{ target_column }}' not found in adata.obs")

# Prepare data
{% if feature_space_key %}
# Using integrated feature space: {{ feature_space_key }}
X = adata.obsm["{{ feature_space_key }}"]
feature_names = [f"Factor_{i+1}" for i in range(X.shape[1])]
print(f"Feature space: {X.shape[1]} factors from '{{ feature_space_key }}'")
{% else %}
X = adata.X
if hasattr(X, "toarray"):
    X = X.toarray()
feature_names = adata.var_names.tolist()
{% endif %}

y = adata.obs["{{ target_column }}"].values
n_samples = X.shape[0]
n_total_features = X.shape[1]

print(f"\\nInput: {n_samples} samples x {n_total_features} features")
print(f"Target: '{{ target_column }}'")

# Encode categorical target if needed
if y.dtype == object or str(y.dtype) == "category":
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"Encoded {len(le.classes_)} target classes: {list(le.classes_)}")

# ============================================================
# Stability Selection Configuration
# ============================================================
n_rounds = {{ n_rounds }}
subsample_fraction = {{ subsample_fraction }}
subsample_size = int(n_samples * subsample_fraction)
probability_threshold = 0.6  # M&B recommended threshold
n_features_fallback = {{ n_features }}
random_state = {{ random_state }}

print(f"\\nConfiguration:")
print(f"  Rounds: {n_rounds}")
print(f"  Subsample: {subsample_fraction:.0%} ({subsample_size} samples/round)")
print(f"  Probability threshold: {probability_threshold}")
print(f"  Fallback n_features: {n_features_fallback}")

# ============================================================
# Subsampling Loop with Isolated RNG
# ============================================================
print(f"\\nRunning {n_rounds} subsampling rounds...")

selection_counts = np.zeros(n_total_features)
all_feature_importances = []

for round_idx in range(n_rounds):
    # Isolated RNG per round for reproducibility (no global state pollution)
    round_rng = np.random.default_rng(random_state + round_idx)
    indices = round_rng.choice(n_samples, size=subsample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]

    # Train model based on method
    {% if method == "xgboost" %}
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=random_state + round_idx,
        verbosity=0,
    )
    model.fit(X_sample, y_sample)
    importances = model.feature_importances_
    {% elif method == "random_forest" %}
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state + round_idx,
        n_jobs=-1,
    )
    model.fit(X_sample, y_sample)
    importances = model.feature_importances_
    {% else %}
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=1000,
        random_state=random_state + round_idx,
        C=0.1,
    )
    model.fit(X_sample, y_sample)
    # Use absolute coefficient as importance
    importances = np.abs(model.coef_).flatten()
    if len(importances) != n_total_features:
        # Multi-class case - average across classes
        importances = np.mean(np.abs(model.coef_), axis=0)
    {% endif %}

    # Track selections: feature selected if importance > 0
    selected_this_round = importances > 1e-8
    selection_counts += selected_this_round.astype(int)
    all_feature_importances.append(importances)

    if (round_idx + 1) % max(1, n_rounds // 5) == 0:
        print(f"  Round {round_idx + 1}/{n_rounds} complete")

# ============================================================
# Calculate M&B Selection Probability
# ============================================================
print(f"\\nCalculating selection probabilities...")

# Standard M&B probability: selection_counts / n_rounds
selection_probability = selection_counts / n_rounds

# Calculate mean importance across rounds
all_feature_importances = np.array(all_feature_importances)
mean_importances = np.mean(all_feature_importances, axis=0)

print(f"  Probability range: [{selection_probability.min():.3f}, {selection_probability.max():.3f}]")
print(f"  Features always selected (p=1.0): {(selection_probability == 1.0).sum()}")
print(f"  Features never selected (p=0.0): {(selection_probability == 0.0).sum()}")

# ============================================================
# Feature Selection with Fallback
# ============================================================
print(f"\\nSelecting features (threshold={probability_threshold})...")

selected_mask = selection_probability >= probability_threshold
n_selected_by_threshold = selected_mask.sum()

# Fallback to top K by probability if threshold yields empty selection
used_fallback = False
if n_selected_by_threshold == 0:
    print(f"  WARNING: No features met threshold {probability_threshold}")
    print(f"  Falling back to top {n_features_fallback} by selection probability")
    used_fallback = True

    top_indices = np.argsort(selection_probability)[::-1][:n_features_fallback]
    selected_mask = np.zeros(n_total_features, dtype=bool)
    selected_mask[top_indices] = True

selected_features = [feature_names[i] for i in np.where(selected_mask)[0]]
n_selected = len(selected_features)

# ============================================================
# Store Results in adata.var (method-prefixed columns)
# ============================================================
adata.var["stability_probability"] = selection_probability
adata.var["stability_mean_importance"] = mean_importances
adata.var["stability_selected"] = selected_mask

# ============================================================
# Report Results
# ============================================================
print(f"\\n{'=' * 60}")
print(f"RESULTS")
print(f"{'=' * 60}")
print(f"Selected {n_selected} features {'(fallback)' if used_fallback else ''}")

if n_selected > 0:
    avg_prob = selection_probability[selected_mask].mean()
    print(f"Average selection probability: {avg_prob:.3f}")

    # Show top 10 selected features
    selected_indices = np.where(selected_mask)[0]
    sorted_selected = selected_indices[np.argsort(selection_probability[selected_indices])[::-1]]

    print(f"\\nTop 10 selected features:")
    for i, idx in enumerate(sorted_selected[:10], 1):
        print(f"  {i}. {feature_names[idx]}: p={selection_probability[idx]:.3f}, imp={mean_importances[idx]:.6f}")

print(f"\\nResults stored in adata.var:")
print(f"  - stability_probability: Selection probability (M&B)")
print(f"  - stability_mean_importance: Mean importance across rounds")
print(f"  - stability_selected: Boolean selection mask")
print(f"{'=' * 60}")
"""

        imports = [
            "import numpy as np",
            "from sklearn.preprocessing import LabelEncoder",
        ]

        if method == "xgboost":
            imports.append("from xgboost import XGBClassifier")
        elif method == "random_forest":
            imports.append("from sklearn.ensemble import RandomForestClassifier")
        else:
            imports.append("from sklearn.linear_model import LogisticRegression")

        return AnalysisStep(
            operation=f"stability_selection_{method}",
            tool_name="stability_selection",
            description=f"M&B stability selection with {method} ({n_rounds} rounds, threshold=0.6)",
            library="xgboost" if method == "xgboost" else "sklearn",
            code_template=code_template,
            imports=imports,
            parameters={
                "target_column": target_column,
                "n_features": n_features,
                "n_rounds": n_rounds,
                "subsample_fraction": subsample_fraction,
                "method": method,
                "random_state": random_state,
                "feature_space_key": feature_space_key,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "method": method,
                "n_rounds": n_rounds,
                "random_state": random_state,
                "feature_space_key": feature_space_key,
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_lasso_selection_ir(
        self,
        target_column: str,
        alpha: float,
        max_iter: int,
        random_state: int,
        feature_space_key: Optional[str] = None,
    ) -> AnalysisStep:
        """
        Create IR for LASSO-based feature selection.

        Args:
            target_column: Target variable column name
            alpha: Regularization strength
            max_iter: Maximum iterations
            random_state: Random seed
            feature_space_key: If provided, use adata.obsm[key] instead of adata.X

        Returns:
            AnalysisStep with executable LASSO selection code
        """
        parameter_schema = {
            "target_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=target_column,
                required=True,
                description="Column in obs containing target variable",
            ),
            "alpha": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=alpha,
                required=False,
                validation_rule="alpha > 0",
                description="Regularization strength (higher = more sparsity)",
            ),
            "max_iter": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=max_iter,
                required=False,
                validation_rule="max_iter > 0",
                description="Maximum iterations for convergence",
            ),
            "random_state": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=random_state,
                required=False,
                description="Random seed for reproducibility",
            ),
            "feature_space_key": ParameterSpec(
                param_type="Optional[str]",
                papermill_injectable=True,
                default_value=feature_space_key,
                required=False,
                description="Key in adata.obsm for feature matrix. None uses adata.X. Example: 'X_mofa'",
            ),
        }

        code_template = """# LASSO (L1) feature selection
import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Validate target column
if "{{ target_column }}" not in adata.obs.columns:
    raise ValueError(f"Target column '{{ target_column }}' not found in adata.obs")

# Prepare data
{% if feature_space_key %}
# Using integrated feature space: {{ feature_space_key }}
X = adata.obsm["{{ feature_space_key }}"]
feature_names = [f"Factor_{i+1}" for i in range(X.shape[1])]
print(f"Feature space: {X.shape[1]} factors from '{{ feature_space_key }}'")
{% else %}
X = adata.X
if hasattr(X, "toarray"):
    X = X.toarray()
feature_names = adata.var_names.tolist()
{% endif %}

y = adata.obs["{{ target_column }}"].values

# Scale features for L1 regularization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine task type (classification vs regression)
is_classification = y.dtype == object or str(y.dtype) == "category" or len(np.unique(y)) < 10

if is_classification:
    # Encode target for classification
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train LogisticRegression with L1 penalty
    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=1.0 / {{ alpha }},  # sklearn uses inverse of alpha
        max_iter={{ max_iter }},
        random_state={{ random_state }},
    )
    model.fit(X_scaled, y)

    # Extract coefficients
    coef = np.abs(model.coef_).flatten()
    if model.coef_.ndim > 1 and model.coef_.shape[0] > 1:
        # Multi-class: average absolute coefficients
        coef = np.mean(np.abs(model.coef_), axis=0)

    print("LASSO classification complete")
else:
    # Train Lasso for regression
    model = Lasso(
        alpha={{ alpha }},
        max_iter={{ max_iter }},
        random_state={{ random_state }},
    )
    model.fit(X_scaled, y)
    coef = np.abs(model.coef_)

    print("LASSO regression complete")

# Identify selected features (non-zero coefficients)
selected_mask = coef > 1e-8
n_selected = selected_mask.sum()
selected_features = [feature_names[i] for i in np.where(selected_mask)[0]]

# Store results in adata.var
adata.var["lasso_coefficient"] = coef
adata.var["lasso_selected"] = selected_mask

print(f"LASSO selected {n_selected}/{len(feature_names)} features (alpha={{ alpha }})")
"""

        return AnalysisStep(
            operation="sklearn.linear_model.LogisticRegression",
            tool_name="lasso_selection",
            description=f"LASSO feature selection (alpha={alpha})",
            library="sklearn",
            code_template=code_template,
            imports=[
                "import numpy as np",
                "from sklearn.linear_model import LogisticRegression",
                "from sklearn.linear_model import Lasso",
                "from sklearn.preprocessing import StandardScaler",
                "from sklearn.preprocessing import LabelEncoder",
            ],
            parameters={
                "target_column": target_column,
                "alpha": alpha,
                "max_iter": max_iter,
                "random_state": random_state,
                "feature_space_key": feature_space_key,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "alpha": alpha,
                "random_state": random_state,
                "feature_space_key": feature_space_key,
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_variance_filter_ir(
        self,
        threshold: Optional[float],
        percentile: float,
        feature_space_key: Optional[str] = None,
        chunked: bool = False,
    ) -> AnalysisStep:
        """
        Create IR for variance-based feature filtering.

        Args:
            threshold: Absolute variance threshold
            percentile: Variance percentile threshold
            feature_space_key: If provided, use adata.obsm[key] instead of adata.X
            chunked: If True, use chunked processing for large sparse matrices

        Returns:
            AnalysisStep with executable variance filter code
        """
        parameter_schema = {
            "threshold": ParameterSpec(
                param_type="Optional[float]",
                papermill_injectable=True,
                default_value=threshold,
                required=False,
                validation_rule="threshold is None or threshold > 0",
                description="Absolute variance threshold (overrides percentile if set)",
            ),
            "percentile": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=percentile,
                required=False,
                validation_rule="0.0 <= percentile <= 100.0",
                description="Variance percentile threshold",
            ),
            "feature_space_key": ParameterSpec(
                param_type="Optional[str]",
                papermill_injectable=True,
                default_value=feature_space_key,
                required=False,
                description="Key in adata.obsm for feature matrix. None uses adata.X. Example: 'X_mofa'",
            ),
            "chunked": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=chunked,
                required=False,
                description="Enable chunked processing for large sparse matrices (memory efficient)",
            ),
        }

        code_template = """# Variance-based feature filtering
import numpy as np

# ============================================================
# VARIANCE-BASED FEATURE FILTERING
# ============================================================
print("=" * 60)
print("VARIANCE-BASED FEATURE FILTERING")
print("=" * 60)

# Extract data matrix
{% if feature_space_key %}
# Using integrated feature space: {{ feature_space_key }}
X = adata.obsm["{{ feature_space_key }}"]
print(f"Feature space: {X.shape[1]} factors from '{{ feature_space_key }}'")
{% else %}
X = adata.X
{% endif %}

print(f"Input shape: {adata.shape[0]} samples x {adata.shape[1]} features")

# Calculate variance for each feature
{% if chunked %}
# Chunked processing (memory efficient for large sparse matrices)
from scipy.sparse import issparse
import psutil

if issparse(X):
    print("\\nUsing chunked variance calculation...")
    n_samples, n_features = X.shape

    # Auto-detect chunk size
    available_bytes = psutil.virtual_memory().available
    bytes_per_row = n_features * 8  # float64
    chunk_size = max(100, int(available_bytes * 0.1 / bytes_per_row))
    chunk_size = min(chunk_size, n_samples)

    # Welford's online algorithm
    count = 0
    mean = np.zeros(n_features)
    M2 = np.zeros(n_features)

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    print(f"Processing {n_samples} samples in {n_chunks} chunks of {chunk_size}...")

    for i in range(0, n_samples, chunk_size):
        chunk_end = min(i + chunk_size, n_samples)
        chunk = X[i:chunk_end].toarray()

        for row in chunk:
            count += 1
            delta = row - mean
            mean += delta / count
            delta2 = row - mean
            M2 += delta * delta2

        if (i // chunk_size + 1) % 10 == 0:
            print(f"  Processed chunk {i // chunk_size + 1}/{n_chunks}")

    variances = M2 / count if count > 1 else np.zeros(n_features)
    print("Chunked variance calculation complete")
else:
    variances = np.var(X, axis=0)
{% else %}
# Standard variance calculation
if hasattr(X, "toarray"):
    X = X.toarray()
variances = np.var(X, axis=0)
{% endif %}

# ============================================================
# Variance Distribution Statistics
# ============================================================
print(f"\\nVariance distribution across {len(variances)} features:")
print(f"  Min variance:    {variances.min():.6f}")
print(f"  Max variance:    {variances.max():.6f}")
print(f"  Mean variance:   {variances.mean():.6f}")
print(f"  Median variance: {np.median(variances):.6f}")
print(f"  Std variance:    {variances.std():.6f}")

print(f"\\nVariance percentiles:")
print(f"  25th: {np.percentile(variances, 25):.6f}")
print(f"  50th: {np.percentile(variances, 50):.6f}")
print(f"  75th: {np.percentile(variances, 75):.6f}")
print(f"  90th: {np.percentile(variances, 90):.6f}")
print(f"  95th: {np.percentile(variances, 95):.6f}")

# ============================================================
# Threshold Selection
# ============================================================
{% if threshold is not none %}
threshold_value = {{ threshold }}
print(f"\\nUsing absolute variance threshold: {threshold_value:.6f}")
# Calculate which percentile this represents
percentile_equivalent = (variances < threshold_value).sum() / len(variances) * 100
print(f"  This threshold is at the {percentile_equivalent:.1f}th percentile")
{% else %}
threshold_value = np.percentile(variances, {{ percentile }})
print(f"\\nUsing {{ percentile }}th percentile threshold: {threshold_value:.6f}")
{% endif %}

# ============================================================
# Feature Selection
# ============================================================
selected_mask = variances > threshold_value
n_selected = selected_mask.sum()
n_removed = len(selected_mask) - n_selected

print(f"\\nSelection results:")
print(f"  Features kept:    {n_selected} ({n_selected/len(selected_mask)*100:.1f}%)")
print(f"  Features removed: {n_removed} ({n_removed/len(selected_mask)*100:.1f}%)")

if n_selected > 0:
    kept_variances = variances[selected_mask]
    print(f"\\nKept features variance range:")
    print(f"  Min: {kept_variances.min():.6f}")
    print(f"  Max: {kept_variances.max():.6f}")
    print(f"  Mean: {kept_variances.mean():.6f}")

    # Show top 5 highest-variance kept features
    top_kept_indices = np.where(selected_mask)[0][np.argsort(variances[selected_mask])[::-1][:5]]
    print(f"\\nTop 5 highest-variance features kept:")
    for i, idx in enumerate(top_kept_indices, 1):
        print(f"  {i}. Feature {idx}: {variances[idx]:.6f}")

if n_removed > 0:
    removed_variances = variances[~selected_mask]
    print(f"\\nRemoved features variance range:")
    print(f"  Min: {removed_variances.min():.6f}")
    print(f"  Max: {removed_variances.max():.6f}")
    print(f"  Mean: {removed_variances.mean():.6f}")

    # Show top 5 lowest-variance removed features
    top_removed_indices = np.where(~selected_mask)[0][np.argsort(variances[~selected_mask])[:5]]
    print(f"\\nTop 5 lowest-variance features removed:")
    for i, idx in enumerate(top_removed_indices, 1):
        print(f"  {i}. Feature {idx}: {variances[idx]:.6f}")

# Store results in adata.var
adata.var["variance"] = variances
adata.var["variance_selected"] = selected_mask

# ============================================================
# Summary
# ============================================================
print(f"\\n{'=' * 60}")
print(f"SUMMARY")
print(f"{'=' * 60}")
reduction_ratio = n_removed / len(selected_mask)
print(f"Dimensionality reduction: {len(selected_mask)} -> {n_selected} features")
print(f"Reduction ratio: {reduction_ratio:.1%}")

# Estimate memory savings for sparse matrices
if hasattr(adata.X, "toarray"):
    estimated_savings = reduction_ratio * 100
    print(f"Estimated memory savings: ~{estimated_savings:.1f}%")

print(f"{'=' * 60}")
"""

        return AnalysisStep(
            operation="variance_filter",
            tool_name="variance_filter",
            description=f"Variance filter (percentile={percentile})",
            library="numpy",
            code_template=code_template,
            imports=["import numpy as np"],
            parameters={
                "threshold": threshold,
                "percentile": percentile,
                "feature_space_key": feature_space_key,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "threshold": threshold,
                "percentile": percentile,
                "feature_space_key": feature_space_key,
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_apply_selection_ir(
        self,
        n_selected: int,
    ) -> AnalysisStep:
        """
        Create IR for applying feature selection.

        Args:
            n_selected: Number of selected features

        Returns:
            AnalysisStep with executable feature selection application
        """
        parameter_schema = {
            "n_selected": ParameterSpec(
                param_type="int",
                papermill_injectable=False,
                default_value=n_selected,
                required=True,
                validation_rule="n_selected > 0",
                description="Number of features to select (derived from selection)",
            ),
        }

        code_template = """# Apply feature selection to create filtered AnnData
import numpy as np

# ============================================================
# APPLYING FEATURE SELECTION
# ============================================================
print("=" * 60)
print("APPLYING FEATURE SELECTION")
print("=" * 60)

print(f"\\nInput AnnData shape: {adata.shape[0]} samples x {adata.shape[1]} features")

# ============================================================
# Feature Validation
# ============================================================
print(f"\\nValidating selected features...")

# Check that selected_features list exists and is non-empty
if not isinstance(selected_features, (list, np.ndarray)):
    raise ValueError(f"selected_features must be a list or array, got {type(selected_features)}")

if len(selected_features) == 0:
    raise ValueError("selected_features is empty - no features to select")

print(f"  Number of features to select: {len(selected_features)}")

# Check that all selected features exist in adata.var_names
missing_features = set(selected_features) - set(adata.var_names)
if missing_features:
    n_missing = len(missing_features)
    print(f"  WARNING: {n_missing} selected features not found in dataset:")
    for i, feat in enumerate(list(missing_features)[:5]):
        print(f"    - {feat}")
    if n_missing > 5:
        print(f"    ... and {n_missing - 5} more")
    raise ValueError(f"{n_missing} selected features not found in adata.var_names")

print(f"  ✓ All selected features exist in dataset")

# Print feature list summary
if len(selected_features) <= 10:
    print(f"\\nSelected features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")
else:
    print(f"\\nFirst 10 selected features:")
    for i, feat in enumerate(selected_features[:10], 1):
        print(f"  {i}. {feat}")
    print(f"  ... and {len(selected_features) - 10} more features")

# ============================================================
# Selection Statistics
# ============================================================
n_original = adata.shape[1]
n_selected = len(selected_features)
selection_ratio = n_selected / n_original

print(f"\\nSelection statistics:")
print(f"  Original features:  {n_original}")
print(f"  Selected features:  {n_selected}")
print(f"  Features removed:   {n_original - n_selected}")
print(f"  Selection ratio:    {selection_ratio:.1%}")

# Check if selection source column exists
source_columns = [col for col in adata.var.columns if 'selected' in col.lower()]
if source_columns:
    print(f"\\nSelection source(s) found in adata.var:")
    for col in source_columns:
        n_true = adata.var[col].sum()
        print(f"  - {col}: {n_true} features marked")

# ============================================================
# Before/After Shape Comparison
# ============================================================
print(f"\\nShape comparison:")
print(f"  Original: {adata.shape[0]} samples × {adata.shape[1]} features")

# Apply selection
adata_filtered = adata[:, selected_features].copy()

print(f"  Filtered: {adata_filtered.shape[0]} samples × {adata_filtered.shape[1]} features")

# Memory estimate comparison
original_size_mb = (adata.shape[0] * adata.shape[1] * 8) / (1024 * 1024)  # 8 bytes per float64
filtered_size_mb = (adata_filtered.shape[0] * adata_filtered.shape[1] * 8) / (1024 * 1024)
memory_reduction = (1 - filtered_size_mb / original_size_mb) * 100

print(f"\\nEstimated memory usage (dense float64):")
print(f"  Original: ~{original_size_mb:.1f} MB")
print(f"  Filtered: ~{filtered_size_mb:.1f} MB")
print(f"  Reduction: {memory_reduction:.1f}%")

# ASCII bar visualization
bar_length = 40
original_bar = "█" * bar_length
filtered_bar = "█" * int(bar_length * selection_ratio)
print(f"\\nDimensionality reduction visualization:")
print(f"  Original: {original_bar}")
print(f"  Filtered: {filtered_bar}")

# ============================================================
# Feature Characteristics
# ============================================================
print(f"\\nFeature characteristics:")

# Check for variance column
if 'variance' in adata.var.columns:
    selected_var = adata.var.loc[selected_features, 'variance']
    all_var = adata.var['variance']
    removed_mask = ~adata.var_names.isin(selected_features)
    removed_var = adata.var.loc[removed_mask, 'variance']

    print(f"  Variance statistics:")
    print(f"    Selected features - mean: {selected_var.mean():.6f}, range: [{selected_var.min():.6f}, {selected_var.max():.6f}]")
    if removed_mask.sum() > 0:
        print(f"    Removed features  - mean: {removed_var.mean():.6f}, range: [{removed_var.min():.6f}, {removed_var.max():.6f}]")

# Print top features
print(f"\\nTop 5 selected features (by name):")
for i, feat in enumerate(selected_features[:5], 1):
    print(f"  {i}. {feat}")

# ============================================================
# Summary
# ============================================================
print(f"\\n{'=' * 60}")
print(f"SUMMARY")
print(f"{'=' * 60}")
print(f"Selection applied successfully:")
print(f"  {adata.shape[1]} features → {adata_filtered.shape[1]} features")
print(f"  Reduction: {(1 - selection_ratio) * 100:.1f}%")
print(f"\\nFiltered AnnData is now available in 'adata_filtered' variable")
print(f"{'=' * 60}")
"""

        return AnalysisStep(
            operation="apply_feature_selection",
            tool_name="apply_selection",
            description=f"Applied selection: {n_selected} features",
            library="anndata",
            code_template=code_template,
            imports=[],
            parameters={"n_selected": n_selected},
            parameter_schema=parameter_schema,
            input_entities=["adata", "selected_features"],
            output_entities=["adata_filtered"],
            execution_context={
                "n_selected": n_selected,
                "timestamp": datetime.now().isoformat(),
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def stability_selection(
        self,
        adata: AnnData,
        target_column: str,
        feature_space_key: Optional[str] = None,
        n_features: int = 100,
        n_rounds: int = 100,
        subsample_fraction: float = 0.5,
        method: str = "xgboost",
        random_state: int = 42,
        probability_threshold: float = 0.6,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform stability-based feature selection using subsampling.

        Implements Meinshausen & Buhlmann (2010) stability selection:
        trains models on multiple data subsets and selects features
        by their selection probability (frequency selected across rounds).

        Args:
            adata: AnnData with features in X and target in obs
            target_column: Column in obs containing target variable
            feature_space_key: If provided, use adata.obsm[feature_space_key]
                              instead of adata.X as feature matrix.
                              Example: 'X_mofa' for integrated factors.
            n_features: Number of top features to select (fallback if threshold yields fewer)
            n_rounds: Number of subsampling rounds
            subsample_fraction: Fraction of data to use per round
            method: Selection method ("xgboost", "random_forest", "lasso")
            random_state: Random seed for reproducibility
            probability_threshold: Selection probability threshold (default 0.6 per M&B)

        Returns:
            Tuple of (adata, stats, ir)
        """
        if target_column not in adata.obs.columns:
            raise ValueError(f"Target column '{target_column}' not found in adata.obs")

        # Validate and extract features
        X, feature_names = self._validate_and_extract_features(adata, feature_space_key)

        y = adata.obs[target_column].values
        n_samples = X.shape[0]
        n_total_features = X.shape[1]

        # Encode categorical target if needed
        if y.dtype == object or str(y.dtype) == "category":
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y = le.fit_transform(y)

        logger.info(
            f"Stability selection: {n_rounds} rounds, {subsample_fraction:.0%} subsample"
        )

        # Track selection counts and importances across rounds
        selection_counts = np.zeros(n_total_features)
        all_feature_importances = []

        subsample_size = int(n_samples * subsample_fraction)

        for round_idx in range(n_rounds):
            # Isolated RNG per round for reproducibility
            round_rng = np.random.default_rng(random_state + round_idx)
            indices = round_rng.choice(n_samples, size=subsample_size, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]

            # Train model based on method
            if method == "xgboost":
                try:
                    from xgboost import XGBClassifier

                    model = XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        random_state=random_state + round_idx,
                        verbosity=0,
                    )
                    model.fit(X_sample, y_sample)
                    importances = model.feature_importances_
                except ImportError:
                    raise ImportError("XGBoost not installed. Run: pip install xgboost")

            elif method == "random_forest":
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=random_state + round_idx,
                    n_jobs=-1,
                )
                model.fit(X_sample, y_sample)
                importances = model.feature_importances_

            elif method == "lasso":
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    max_iter=1000,
                    random_state=random_state + round_idx,
                    C=0.1,
                )
                model.fit(X_sample, y_sample)
                # Use absolute coefficient as importance
                importances = np.abs(model.coef_).flatten()
                if len(importances) != n_total_features:
                    # Multi-class case - average across classes
                    importances = np.mean(np.abs(model.coef_), axis=0)

            else:
                raise ValueError(
                    f"Unknown method: {method}. Use 'xgboost', 'random_forest', or 'lasso'"
                )

            # Track selections: feature selected if importance > 0 (non-zero)
            # For tree models, this means feature was used in splits
            # For LASSO, this means non-zero coefficient
            selected_this_round = importances > 1e-8
            selection_counts += selected_this_round.astype(int)
            all_feature_importances.append(importances)

        # Calculate M&B selection probability: selection_counts / n_rounds
        selection_probability = selection_counts / n_rounds

        # Calculate mean importance across rounds
        all_feature_importances = np.array(all_feature_importances)
        mean_importances = np.mean(all_feature_importances, axis=0)

        # Select features by probability threshold (M&B recommendation: 0.6)
        selected_mask = selection_probability >= probability_threshold
        n_selected_by_threshold = selected_mask.sum()

        # Fallback to top K by probability if threshold yields empty or too few
        if n_selected_by_threshold == 0:
            logger.warning(
                f"No features met probability threshold {probability_threshold}. "
                f"Falling back to top {n_features} features by selection probability."
            )
            # Select top n_features by selection probability
            top_indices = np.argsort(selection_probability)[::-1][:n_features]
            selected_mask = np.zeros(n_total_features, dtype=bool)
            selected_mask[top_indices] = True

        selected_features = [feature_names[i] for i in np.where(selected_mask)[0]]
        n_selected = len(selected_features)

        # Store results in adata with method-prefixed columns
        adata = adata.copy()
        adata.var["stability_probability"] = selection_probability
        adata.var["stability_mean_importance"] = mean_importances
        adata.var["stability_selected"] = selected_mask

        adata.uns["stability_selection"] = {
            "method": method,
            "n_rounds": n_rounds,
            "subsample_fraction": subsample_fraction,
            "probability_threshold": probability_threshold,
            "n_selected": n_selected,
            "selected_features": selected_features,
            "used_fallback": n_selected_by_threshold == 0,
        }

        # Build stats with top features by selection probability
        sorted_indices = np.argsort(selection_probability)[::-1]
        top_features_info = []
        for i, idx in enumerate(sorted_indices[:20]):  # Top 20 for reporting
            top_features_info.append(
                {
                    "rank": i + 1,
                    "feature": feature_names[idx],
                    "selection_probability": float(selection_probability[idx]),
                    "mean_importance": float(mean_importances[idx]),
                }
            )

        stats = {
            "method": method,
            "n_rounds": n_rounds,
            "subsample_fraction": subsample_fraction,
            "probability_threshold": probability_threshold,
            "n_total_features": n_total_features,
            "n_selected_features": n_selected,
            "avg_selection_probability": float(
                np.mean(selection_probability[selected_mask])
            )
            if n_selected > 0
            else 0.0,
            "used_fallback": n_selected_by_threshold == 0,
            "top_features": top_features_info,
        }

        ir = self._create_stability_selection_ir(
            target_column=target_column,
            n_features=n_features,
            n_rounds=n_rounds,
            subsample_fraction=subsample_fraction,
            method=method,
            random_state=random_state,
            feature_space_key=feature_space_key,
        )

        logger.info(
            f"Selected {n_selected} features with avg selection probability: {stats['avg_selection_probability']:.3f}"
        )

        return adata, stats, ir

    def lasso_selection(
        self,
        adata: AnnData,
        target_column: str,
        feature_space_key: Optional[str] = None,
        alpha: float = 0.1,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform LASSO (L1) feature selection.

        Uses L1 regularization to automatically select features
        by driving coefficients to zero.

        Args:
            adata: AnnData with features in X and target in obs
            target_column: Column in obs containing target variable
            feature_space_key: If provided, use adata.obsm[feature_space_key]
                              instead of adata.X as feature matrix.
                              Example: 'X_mofa' for integrated factors.
            alpha: Regularization strength (higher = more sparsity)
            max_iter: Maximum iterations for convergence
            random_state: Random seed

        Returns:
            Tuple of (adata, stats, ir)
        """
        from sklearn.linear_model import Lasso, LogisticRegression
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        if target_column not in adata.obs.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        # Validate and extract features
        X, feature_names = self._validate_and_extract_features(adata, feature_space_key)

        y = adata.obs[target_column].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine task type
        is_classification = (
            y.dtype == object or str(y.dtype) == "category" or len(np.unique(y)) < 10
        )

        if is_classification:
            # Encode target
            if y.dtype == object or str(y.dtype) == "category":
                le = LabelEncoder()
                y = le.fit_transform(y)

            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=1.0 / alpha,  # sklearn uses inverse of alpha
                max_iter=max_iter,
                random_state=random_state,
            )
            model.fit(X_scaled, y)
            coef = np.abs(model.coef_).flatten()
            if model.coef_.ndim > 1 and model.coef_.shape[0] > 1:
                # Multi-class: average absolute coefficients
                coef = np.mean(np.abs(model.coef_), axis=0)
        else:
            model = Lasso(
                alpha=alpha,
                max_iter=max_iter,
                random_state=random_state,
            )
            model.fit(X_scaled, y)
            coef = np.abs(model.coef_)

        # Identify selected features (non-zero coefficients)
        selected_mask = coef > 1e-8
        n_selected = selected_mask.sum()
        selected_features = [feature_names[i] for i in np.where(selected_mask)[0]]

        # Store results
        adata = adata.copy()
        adata.var["lasso_coefficient"] = coef
        adata.var["lasso_selected"] = selected_mask

        adata.uns["lasso_selection"] = {
            "alpha": alpha,
            "n_selected": int(n_selected),
            "selected_features": selected_features,
        }

        # Build stats
        sorted_idx = np.argsort(coef)[::-1]
        top_features = [
            {
                "feature": feature_names[idx],
                "coefficient": float(coef[idx]),
            }
            for idx in sorted_idx[:20]
            if coef[idx] > 1e-8
        ]

        stats = {
            "alpha": alpha,
            "n_total_features": len(feature_names),
            "n_selected_features": int(n_selected),
            "selection_rate": float(n_selected / len(feature_names)),
            "max_coefficient": float(coef.max()),
            "top_features": top_features,
        }

        ir = self._create_lasso_selection_ir(
            target_column=target_column,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            feature_space_key=feature_space_key,
        )

        logger.info(f"LASSO selected {n_selected}/{len(feature_names)} features")

        return adata, stats, ir

    def variance_filter(
        self,
        adata: AnnData,
        feature_space_key: Optional[str] = None,
        threshold: Optional[float] = None,
        percentile: float = 10.0,
        chunked: bool = False,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Filter features by variance threshold.

        Removes low-variance features that are unlikely to be informative.

        Args:
            adata: AnnData with features in X
            feature_space_key: If provided, use adata.obsm[feature_space_key]
                              instead of adata.X as feature matrix.
                              Example: 'X_mofa' for integrated factors.
            threshold: Absolute variance threshold (overrides percentile)
            percentile: Remove features below this variance percentile
            chunked: Enable chunked processing for large sparse matrices (memory efficient)

        Returns:
            Tuple of (adata, stats, ir)
        """
        from scipy.sparse import issparse

        # Validate and extract features - but handle sparse specially for chunked mode
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
            feature_names = adata.var_names.tolist()

        # Calculate variances with chunked or standard approach
        if issparse(X):
            if chunked:
                # Chunked variance calculation (memory efficient)
                variances, n_chunks = self._chunked_variance(X)
            else:
                # Check memory before densification
                check_sparse_conversion_safe(X, overhead_multiplier=1.5)
                X = X.toarray()
                variances = np.var(X, axis=0)
                n_chunks = None
        else:
            variances = np.var(X, axis=0)
            n_chunks = None

        # Determine threshold
        if threshold is None:
            threshold = np.percentile(variances, percentile)

        # Select features
        selected_mask = variances > threshold
        n_selected = selected_mask.sum()
        n_removed = len(selected_mask) - n_selected

        # Store results
        adata = adata.copy()
        adata.var["variance"] = variances
        adata.var["variance_selected"] = selected_mask

        adata.uns["variance_filter"] = {
            "threshold": float(threshold),
            "n_selected": int(n_selected),
            "n_removed": int(n_removed),
        }

        stats = {
            "variance_threshold": float(threshold),
            "percentile_used": percentile if threshold is None else None,
            "n_total_features": len(selected_mask),
            "n_selected_features": int(n_selected),
            "n_removed_features": int(n_removed),
            "removal_rate": float(n_removed / len(selected_mask)),
            "min_variance_kept": float(variances[selected_mask].min())
            if n_selected > 0
            else None,
            "max_variance_removed": float(variances[~selected_mask].max())
            if n_removed > 0
            else None,
            "chunked": chunked,
        }

        # Add chunks_processed for Cloud API feedback
        if chunked and n_chunks is not None:
            stats["chunks_processed"] = n_chunks

        ir = self._create_variance_filter_ir(
            threshold=threshold,
            percentile=percentile,
            feature_space_key=feature_space_key,
            chunked=chunked,
        )

        logger.info(f"Variance filter: kept {n_selected}, removed {n_removed} features")

        return adata, stats, ir

    def get_selected_features(
        self,
        adata: AnnData,
        selection_column: Optional[str] = None,
    ) -> List[str]:
        """
        Get list of selected features from a previous selection operation.

        Supports method-prefixed column names from all selection methods:
        - stability_selected (from stability_selection)
        - lasso_selected (from lasso_selection)
        - variance_selected (from variance_filter)

        If selection_column is not specified, auto-detects the selection column.
        Raises an error if multiple selection columns exist (ambiguous) or none found.

        Args:
            adata: AnnData with selection results in var
            selection_column: Column name with boolean selection mask.
                If None, auto-detects from available *_selected columns.

        Returns:
            List of selected feature names

        Raises:
            ValueError: If no selection columns found, or multiple columns found
                without explicit selection_column parameter.
        """
        # Auto-detect selection column if not specified
        if selection_column is None:
            # Look for method-prefixed selection columns
            selection_columns = [
                col for col in adata.var.columns if col.endswith("_selected")
            ]
            if len(selection_columns) == 0:
                raise ValueError(
                    "No selection columns found. Run a selection method first "
                    "(stability_selection, lasso_selection, or variance_filter)."
                )
            elif len(selection_columns) == 1:
                selection_column = selection_columns[0]
                logger.info(f"Auto-detected selection column: {selection_column}")
            else:
                raise ValueError(
                    f"Multiple selection columns found: {selection_columns}. "
                    f"Specify which to use via selection_column parameter."
                )

        if selection_column not in adata.var.columns:
            raise ValueError(
                f"Selection column '{selection_column}' not found. Run feature selection first."
            )

        return adata.var_names[adata.var[selection_column]].tolist()

    def apply_selection(
        self,
        adata: AnnData,
        selected_features: List[str],
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Apply a feature selection to create a filtered AnnData.

        Args:
            adata: Source AnnData
            selected_features: List of feature names to keep

        Returns:
            Tuple of (filtered_adata, stats, ir)
        """
        # Validate features exist
        missing = set(selected_features) - set(adata.var_names)
        if missing:
            raise ValueError(f"Features not found in adata: {list(missing)[:5]}...")

        adata_filtered = adata[:, selected_features].copy()

        stats = {
            "n_original_features": adata.shape[1],
            "n_selected_features": len(selected_features),
            "reduction_ratio": float(len(selected_features) / adata.shape[1]),
        }

        ir = self._create_apply_selection_ir(
            n_selected=len(selected_features),
        )

        return adata_filtered, stats, ir
