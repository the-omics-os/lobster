"""
Platform configuration for metabolomics analysis.

This module defines platform-specific configurations for LC-MS, GC-MS, and NMR
metabolomics, including default parameters and platform detection logic.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import anndata
import numpy as np

__all__ = [
    "MetabPlatformConfig",
    "PLATFORM_CONFIGS",
    "detect_platform_type",
    "get_platform_config",
]


@dataclass
class MetabPlatformConfig:
    """Configuration for a metabolomics platform type."""

    platform_type: str  # "lc_ms", "gc_ms", or "nmr"
    display_name: str
    description: str

    # Missing value thresholds
    expected_missing_rate_range: tuple  # (min, max) expected missing %
    max_missing_per_sample: float  # filtering threshold
    max_missing_per_feature: float  # filtering threshold

    # QC thresholds
    max_rsd_qc_samples: float  # RSD threshold for QC samples (%)
    min_features_per_sample: int

    # Normalization defaults
    default_normalization: str
    log_transform: bool
    default_imputation: str

    # Analysis defaults
    default_fold_change_threshold: float
    default_n_pca_components: int

    def get_qc_defaults(self) -> Dict[str, Any]:
        """Get default QC parameters for this platform."""
        return {
            "max_missing_per_sample": self.max_missing_per_sample,
            "max_missing_per_feature": self.max_missing_per_feature,
            "max_rsd_qc_samples": self.max_rsd_qc_samples,
            "min_features_per_sample": self.min_features_per_sample,
        }

    def get_normalization_defaults(self) -> Dict[str, Any]:
        """Get default normalization parameters for this platform."""
        return {
            "method": self.default_normalization,
            "log_transform": self.log_transform,
            "imputation_method": self.default_imputation,
        }


# Platform configurations
PLATFORM_CONFIGS: Dict[str, MetabPlatformConfig] = {
    "lc_ms": MetabPlatformConfig(
        platform_type="lc_ms",
        display_name="LC-MS",
        description="Liquid chromatography-mass spectrometry",
        # LC-MS typically has 20-60% missing values
        expected_missing_rate_range=(0.20, 0.60),
        max_missing_per_sample=0.60,
        max_missing_per_feature=0.80,
        # LC-MS QC thresholds
        max_rsd_qc_samples=30.0,
        min_features_per_sample=50,
        # LC-MS normalization: PQN + log2
        default_normalization="pqn",
        log_transform=True,
        default_imputation="knn",
        # LC-MS analysis defaults
        default_fold_change_threshold=1.5,
        default_n_pca_components=10,
    ),
    "gc_ms": MetabPlatformConfig(
        platform_type="gc_ms",
        display_name="GC-MS",
        description="Gas chromatography-mass spectrometry",
        # GC-MS typically has 10-40% missing values
        expected_missing_rate_range=(0.10, 0.40),
        max_missing_per_sample=0.40,
        max_missing_per_feature=0.60,
        # GC-MS QC thresholds (tighter than LC-MS)
        max_rsd_qc_samples=25.0,
        min_features_per_sample=30,
        # GC-MS normalization: TIC + log2
        default_normalization="tic",
        log_transform=True,
        default_imputation="min",
        # GC-MS analysis defaults
        default_fold_change_threshold=2.0,
        default_n_pca_components=10,
    ),
    "nmr": MetabPlatformConfig(
        platform_type="nmr",
        display_name="NMR",
        description="Nuclear magnetic resonance spectroscopy",
        # NMR typically has 0-10% missing values
        expected_missing_rate_range=(0.0, 0.10),
        max_missing_per_sample=0.10,
        max_missing_per_feature=0.20,
        # NMR QC thresholds (tightest)
        max_rsd_qc_samples=20.0,
        min_features_per_sample=100,
        # NMR normalization: PQN, no log (data often already processed)
        default_normalization="pqn",
        log_transform=False,
        default_imputation="median",
        # NMR analysis defaults
        default_fold_change_threshold=1.5,
        default_n_pca_components=10,
    ),
}


def detect_platform_type(
    adata: anndata.AnnData,
    force_type: Optional[str] = None,
) -> str:
    """
    Auto-detect metabolomics platform type from data characteristics.

    Detection logic uses a scoring system:
    1. Check var columns for LC-MS hints (retention_time, mz, adduct)
    2. Check var columns for GC-MS hints (ri, retention_index)
    3. Check var columns for NMR hints (ppm, chemical_shift)
    4. Check uns for platform hints
    5. Analyze missing rate patterns

    Args:
        adata: AnnData object with metabolomics data
        force_type: Override detection with specific type ("lc_ms", "gc_ms", or "nmr")

    Returns:
        str: "lc_ms", "gc_ms", or "nmr"
    """
    if force_type:
        if force_type not in PLATFORM_CONFIGS:
            raise ValueError(
                f"Unknown platform type: {force_type}. "
                f"Choose from: {list(PLATFORM_CONFIGS.keys())}"
            )
        return force_type

    detection_scores: Dict[str, int] = {"lc_ms": 0, "gc_ms": 0, "nmr": 0}

    var_cols = (
        set(c.lower() for c in adata.var.columns)
        if len(adata.var.columns) > 0
        else set()
    )

    # 1. Check for LC-MS-specific columns
    lc_ms_hints = ["retention_time", "mz", "adduct", "rt", "precursor_mz"]
    for hint in lc_ms_hints:
        if hint in var_cols:
            detection_scores["lc_ms"] += 2

    # 2. Check for GC-MS-specific columns
    gc_ms_hints = ["ri", "retention_index", "kovats_index", "derivatization"]
    for hint in gc_ms_hints:
        if hint in var_cols:
            detection_scores["gc_ms"] += 3

    # 3. Check for NMR-specific columns
    nmr_hints = ["ppm", "chemical_shift", "bin_start", "bin_end", "multiplicity"]
    for hint in nmr_hints:
        if hint in var_cols:
            detection_scores["nmr"] += 3

    # 4. Check uns for platform hints
    if "platform" in adata.uns:
        platform_hint = str(adata.uns["platform"]).lower()
        if any(term in platform_hint for term in ["lc-ms", "lc_ms", "lcms", "liquid"]):
            detection_scores["lc_ms"] += 5
        elif any(term in platform_hint for term in ["gc-ms", "gc_ms", "gcms", "gas"]):
            detection_scores["gc_ms"] += 5
        elif any(term in platform_hint for term in ["nmr", "nuclear", "1h", "13c"]):
            detection_scores["nmr"] += 5

    # 5. Analyze missing rate as tiebreaker
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = adata.X

    if X.size > 0:
        missing_rate = np.isnan(X).sum() / X.size if X.size > 0 else 0

        # High missing rate -> LC-MS
        if missing_rate > 0.30:
            detection_scores["lc_ms"] += 2
        # Low missing rate -> NMR
        elif missing_rate < 0.05:
            detection_scores["nmr"] += 2
        # Medium missing rate -> GC-MS
        else:
            detection_scores["gc_ms"] += 1

    # Return platform with highest score; default to lc_ms on ties
    max_score = max(detection_scores.values())
    if max_score == 0:
        return "lc_ms"  # Default when no hints found

    # Prefer lc_ms > gc_ms > nmr on ties (most common platform first)
    for platform in ["lc_ms", "gc_ms", "nmr"]:
        if detection_scores[platform] == max_score:
            return platform

    return "lc_ms"


def get_platform_config(platform_type: str) -> MetabPlatformConfig:
    """
    Get the configuration for a specific platform type.

    Args:
        platform_type: "lc_ms", "gc_ms", or "nmr"

    Returns:
        MetabPlatformConfig for the specified platform
    """
    if platform_type not in PLATFORM_CONFIGS:
        raise ValueError(
            f"Unknown platform type: {platform_type}. "
            f"Choose from: {list(PLATFORM_CONFIGS.keys())}"
        )
    return PLATFORM_CONFIGS[platform_type]
