"""
Platform configuration for proteomics analysis.

This module defines platform-specific configurations for mass spectrometry
and affinity proteomics, including default parameters and platform detection logic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import anndata

__all__ = [
    "PlatformConfig",
    "PLATFORM_CONFIGS",
    "detect_platform_type",
    "get_platform_config",
]


@dataclass
class PlatformConfig:
    """Configuration for a proteomics platform type."""

    platform_type: str  # "mass_spec" or "affinity"
    display_name: str
    description: str

    # Missing value thresholds
    expected_missing_rate_range: tuple  # (min, max) expected missing %
    max_missing_per_sample: float  # filtering threshold
    max_missing_per_protein: float  # filtering threshold

    # Quality thresholds
    cv_threshold: float  # coefficient of variation threshold
    min_proteins_per_sample: int

    # Normalization defaults
    default_normalization: str
    log_transform: bool
    default_imputation: str

    # Analysis defaults
    default_fold_change_threshold: float
    default_n_pca_components: int

    # Platform-specific attributes
    platform_specific: Dict[str, Any] = field(default_factory=dict)

    def get_qc_defaults(self) -> Dict[str, Any]:
        """Get default QC parameters for this platform."""
        return {
            "max_missing_per_sample": self.max_missing_per_sample,
            "max_missing_per_protein": self.max_missing_per_protein,
            "cv_threshold": self.cv_threshold,
            "min_proteins_per_sample": self.min_proteins_per_sample,
        }

    def get_normalization_defaults(self) -> Dict[str, Any]:
        """Get default normalization parameters for this platform."""
        return {
            "method": self.default_normalization,
            "log_transform": self.log_transform,
            "imputation_method": self.default_imputation,
        }


# Platform configurations
PLATFORM_CONFIGS: Dict[str, PlatformConfig] = {
    "mass_spec": PlatformConfig(
        platform_type="mass_spec",
        display_name="Mass Spectrometry",
        description="DDA/DIA mass spectrometry proteomics",
        # MS data typically has 30-70% missing (MNAR pattern)
        expected_missing_rate_range=(0.30, 0.70),
        max_missing_per_sample=0.70,  # More tolerant for MS
        max_missing_per_protein=0.80,  # Allow high missing per protein
        # MS has higher variability
        cv_threshold=50.0,
        min_proteins_per_sample=100,
        # MS normalization: median + log2
        default_normalization="median",
        log_transform=True,
        default_imputation="keep",  # Preserve MNAR pattern
        # MS analysis defaults
        default_fold_change_threshold=1.5,
        default_n_pca_components=15,
        # MS-specific attributes
        platform_specific={
            "min_peptides_per_protein": 2,
            "remove_contaminants": True,
            "remove_reverse_hits": True,
            "common_contaminants": [
                "CON__",
                "KERATIN",
                "TRYP_PIG",
                "BSA",
                "ALBUMIN",
            ],
        },
    ),
    "affinity": PlatformConfig(
        platform_type="affinity",
        display_name="Affinity Proteomics",
        description="Targeted antibody-based proteomics (Olink, SomaScan, Luminex)",
        # Affinity data should have low missing (<30%)
        expected_missing_rate_range=(0.0, 0.30),
        max_missing_per_sample=0.30,  # Stricter for affinity
        max_missing_per_protein=0.50,
        # Affinity has better reproducibility
        cv_threshold=30.0,
        min_proteins_per_sample=20,  # Targeted panels are smaller
        # Affinity normalization: quantile, no log (NPX already log-scale)
        default_normalization="quantile",
        log_transform=False,  # NPX values already log-transformed
        default_imputation="impute_knn",  # OK to impute for MAR
        # Affinity analysis defaults
        default_fold_change_threshold=1.2,  # Lower threshold for controlled measurements
        default_n_pca_components=10,  # Fewer components for targeted panels
        # Affinity-specific attributes
        platform_specific={
            "correct_plate_effects": True,
            "check_antibody_specificity": True,
            "typical_panel_size_range": (50, 5000),
        },
    ),
}


def detect_platform_type(
    adata: anndata.AnnData,
    force_type: Optional[str] = None,
) -> str:
    """
    Auto-detect proteomics platform type from data characteristics.

    Detection logic:
    1. Check var columns for MS-specific metadata (n_peptides, sequence_coverage)
    2. Check var columns for affinity-specific metadata (antibody_id, panel_type, npx_value)
    3. Analyze missing value patterns (high = MS, low = affinity)
    4. Check protein count (large = MS discovery, small = affinity panel)

    Args:
        adata: AnnData object with proteomics data
        force_type: Override detection with specific type ("mass_spec" or "affinity")

    Returns:
        str: "mass_spec", "affinity", or "unknown" (if scores are tied or both zero)
    """
    if force_type:
        if force_type not in PLATFORM_CONFIGS:
            raise ValueError(
                f"Unknown platform type: {force_type}. Choose from: {list(PLATFORM_CONFIGS.keys())}"
            )
        return force_type

    detection_scores = {"mass_spec": 0, "affinity": 0}

    # 1. Check for MS-specific columns in var
    ms_columns = [
        "n_peptides",
        "n_unique_peptides",
        "sequence_coverage",
        "protein_group",
    ]
    ms_col_count = sum(1 for col in ms_columns if col in adata.var.columns)
    detection_scores["mass_spec"] += ms_col_count * 2

    # Check for contaminant/reverse hit columns (MS-specific)
    if "is_contaminant" in adata.var.columns or "is_reverse" in adata.var.columns:
        detection_scores["mass_spec"] += 3

    # 2. Check for affinity-specific columns in var
    affinity_columns = [
        "antibody_id",
        "antibody_clone",
        "panel_type",
        "lot_number",
        "npx_value",
    ]
    affinity_col_count = sum(1 for col in affinity_columns if col in adata.var.columns)
    detection_scores["affinity"] += affinity_col_count * 2

    # Check for plate information (affinity-specific)
    if "plate_id" in adata.obs.columns or "plate_number" in adata.obs.columns:
        detection_scores["affinity"] += 2

    # 3. Analyze missing value patterns
    import numpy as np

    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = adata.X

    missing_rate = np.isnan(X).sum() / X.size if X.size > 0 else 0

    # High missing rate (>30%) suggests MS data
    if missing_rate > 0.30:
        detection_scores["mass_spec"] += 3
    elif missing_rate < 0.15:
        detection_scores["affinity"] += 3
    elif missing_rate < 0.30:
        detection_scores["affinity"] += 1

    # 4. Check protein count (discovery vs targeted)
    n_proteins = adata.n_vars

    # Very large protein count suggests MS discovery
    if n_proteins > 3000:
        detection_scores["mass_spec"] += 2
    # Small panel suggests targeted affinity
    elif n_proteins < 200:
        detection_scores["affinity"] += 2
    elif n_proteins < 1000:
        detection_scores["affinity"] += 1

    # 5. Check for common platform indicators in uns
    if "platform" in adata.uns:
        platform_hint = str(adata.uns["platform"]).lower()
        if any(
            term in platform_hint
            for term in ["ms", "mass_spec", "dda", "dia", "maxquant", "spectronaut"]
        ):
            detection_scores["mass_spec"] += 5
        elif any(
            term in platform_hint
            for term in ["olink", "soma", "luminex", "affinity", "antibody"]
        ):
            detection_scores["affinity"] += 5

    # Return platform with higher score, or "unknown" if tied/zero
    if detection_scores["affinity"] > detection_scores["mass_spec"]:
        return "affinity"
    elif detection_scores["mass_spec"] > detection_scores["affinity"]:
        return "mass_spec"
    else:
        return "unknown"


def get_platform_config(platform_type: str) -> PlatformConfig:
    """
    Get the configuration for a specific platform type.

    Args:
        platform_type: "mass_spec", "affinity", or "unknown"

    Returns:
        PlatformConfig for the specified platform
    """
    if platform_type == "unknown":
        # Default to mass_spec for unknown platform types (safe defaults)
        platform_type = "mass_spec"
    if platform_type not in PLATFORM_CONFIGS:
        raise ValueError(
            f"Unknown platform type: {platform_type}. Choose from: {list(PLATFORM_CONFIGS.keys())}"
        )
    return PLATFORM_CONFIGS[platform_type]
