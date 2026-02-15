"""
Lineage tracking for AnnData modalities.

This module provides utilities for tracking the processing history of AnnData objects
via adata.uns['lineage'] metadata. The lineage system enables:

1. Grouping modalities by base_name (same dataset, different versions)
2. Sorting by version for timeline display
3. Showing processing_step as chips in the UI
4. Displaying step_summary on hover/tooltip
5. Tracking parent-child relationships for lineage graphs

Schema (adata.uns['lineage']):
    {
        "base_name": str,           # Original dataset name without suffixes
        "version": int,             # Sequential version (1=raw, 2=after first step, etc.)
        "processing_step": str,     # One of ProcessingStep values
        "parent_modality": str | None,  # Name of parent modality (None for raw)
        "step_summary": str | None,     # Human-readable summary of changes
        "created_at": str,          # ISO timestamp
    }
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from anndata import AnnData


LINEAGE_KEY = "lineage"


# Canonical processing steps — guidance, not enforcement.
# Any string is a valid step. Agent packages can define domain-specific steps
# (e.g., "imputed", "peak_called", "aligned") without modifying core.
CANONICAL_STEPS: set[str] = {
    "raw",
    "quality_assessed",
    "filtered",
    "normalized",
    "filtered_normalized",
    "batch_corrected",
    "reduced",
    "clustered",
    "subclustered",
    "annotated",
    "markers",
    "de_results",
    "pseudobulk",
    "custom",
}


# Known suffixes that indicate processing steps
# Sorted by length descending for correct matching (longest first)
SUFFIX_PATTERNS = [
    "_filtered_normalized",  # 20
    "_quality_evaluated",  # 18
    "_quality_assessed",  # 17
    "_batch_corrected",  # 16
    "_subclustered",  # 13
    "_concatenated",  # 13
    "_de_results",  # 11
    "_pseudobulk",  # 11
    "_normalized",  # 11
    "_annotated",  # 10
    "_clustered",  # 10
    "_autosave",  # 9
    "_filtered",  # 9
    "_reduced",  # 8
    "_markers",  # 8
]

# Mapping from suffix to processing step string
SUFFIX_TO_STEP: Dict[str, str] = {
    "_quality_assessed": "quality_assessed",
    "_quality_evaluated": "quality_assessed",
    "_filtered": "filtered",
    "_normalized": "normalized",
    "_filtered_normalized": "filtered_normalized",
    "_batch_corrected": "batch_corrected",
    "_reduced": "reduced",
    "_clustered": "clustered",
    "_subclustered": "subclustered",
    "_annotated": "annotated",
    "_markers": "markers",
    "_de_results": "de_results",
    "_pseudobulk": "pseudobulk",
    "_concatenated": "custom",
    "_autosave": "custom",
}


@dataclass
class LineageMetadata:
    """Lineage metadata schema for adata.uns['lineage']."""

    base_name: str
    version: int
    processing_step: str
    parent_modality: Optional[str]
    step_summary: Optional[str]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage in adata.uns."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageMetadata":
        """Create from dictionary stored in adata.uns."""
        # Handle any extra keys gracefully (e.g., _synthetic flag)
        valid_keys = {
            "base_name",
            "version",
            "processing_step",
            "parent_modality",
            "step_summary",
            "created_at",
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


def extract_base_name(modality_name: str) -> str:
    """
    Extract base dataset name by stripping known processing suffixes.

    Examples:
        geo_gse12345 -> geo_gse12345
        geo_gse12345_clustered -> geo_gse12345
        geo_gse12345_filtered_normalized_clustered -> geo_gse12345
        pride_pxd12345_proteomics_normalized -> pride_pxd12345_proteomics

    Args:
        modality_name: Full modality name with potential suffixes

    Returns:
        Base name with all known processing suffixes stripped
    """
    result = modality_name

    # Iteratively strip suffixes (handles chained suffixes like _filtered_normalized_clustered)
    changed = True
    while changed:
        changed = False
        for suffix in SUFFIX_PATTERNS:
            if result.endswith(suffix):
                result = result[: -len(suffix)]
                changed = True
                break  # Start over with sorted patterns

    return result


def infer_processing_step(modality_name: str) -> str:
    """
    Infer processing step from modality name suffix.

    Args:
        modality_name: Full modality name

    Returns:
        Processing step string ("raw" if no known suffix found)
    """
    # Check suffixes in order (longest first due to SUFFIX_PATTERNS sorting)
    for suffix in SUFFIX_PATTERNS:
        if modality_name.endswith(suffix):
            return SUFFIX_TO_STEP.get(suffix, "custom")

    return "raw"


def create_lineage_metadata(
    modality_name: str,
    parent_modality: Optional[str] = None,
    step: Optional[str] = None,
    step_summary: Optional[str] = None,
    base_name: Optional[str] = None,
    version: Optional[int] = None,
    parent_adata: Optional["AnnData"] = None,
) -> LineageMetadata:
    """
    Create lineage metadata for a modality.

    This is the main factory function for creating lineage metadata.
    It auto-computes fields when not explicitly provided:
    - base_name: extracted from modality_name or inherited from parent
    - version: incremented from parent or set to 1 for raw data
    - processing_step: inferred from modality name suffix

    Args:
        modality_name: Name of the new modality
        parent_modality: Name of parent modality this was derived from
        step: Processing step string (auto-inferred from name if not provided).
              Any string is valid — not limited to CANONICAL_STEPS.
        step_summary: Human-readable summary of what changed
        base_name: Base dataset name (auto-extracted if not provided)
        version: Version number (auto-computed if not provided)
        parent_adata: Parent AnnData for inheriting base_name/version

    Returns:
        LineageMetadata ready to attach to AnnData
    """
    # Auto-extract base_name
    if base_name is None:
        if parent_adata is not None and LINEAGE_KEY in parent_adata.uns:
            base_name = parent_adata.uns[LINEAGE_KEY].get("base_name")
        if base_name is None:
            base_name = extract_base_name(modality_name)

    # Auto-infer step
    if step is None:
        step = infer_processing_step(modality_name)

    # Auto-compute version from parent
    if version is None:
        if parent_adata is not None and LINEAGE_KEY in parent_adata.uns:
            parent_version = parent_adata.uns[LINEAGE_KEY].get("version", 0)
            version = parent_version + 1
        else:
            version = 1  # Raw data starts at 1

    return LineageMetadata(
        base_name=base_name,
        version=version,
        processing_step=step,
        parent_modality=parent_modality,
        step_summary=step_summary,
        created_at=datetime.now().isoformat(),
    )


def attach_lineage(adata: "AnnData", lineage: LineageMetadata) -> "AnnData":
    """
    Attach lineage metadata to AnnData object.

    Args:
        adata: AnnData object to attach lineage to
        lineage: LineageMetadata to attach

    Returns:
        Same AnnData object with lineage attached (modified in place)
    """
    adata.uns[LINEAGE_KEY] = lineage.to_dict()
    return adata


def get_lineage(adata: "AnnData") -> Optional[LineageMetadata]:
    """
    Extract lineage metadata from AnnData if present.

    Args:
        adata: AnnData object to extract lineage from

    Returns:
        LineageMetadata if present, None otherwise
    """
    if LINEAGE_KEY not in adata.uns:
        return None
    return LineageMetadata.from_dict(adata.uns[LINEAGE_KEY])


def has_lineage(adata: "AnnData") -> bool:
    """Check if AnnData has lineage metadata."""
    return LINEAGE_KEY in adata.uns


def ensure_lineage(
    adata: "AnnData",
    modality_name: str,
    step_summary: Optional[str] = None,
) -> "AnnData":
    """
    Ensure AnnData has lineage metadata, creating synthetic lineage if missing.

    This is used for backward compatibility with legacy h5ad files
    that don't have lineage metadata.

    Args:
        adata: AnnData object
        modality_name: Name of the modality (for inferring base_name and step)
        step_summary: Optional summary (defaults to "[Legacy] Loaded from file")

    Returns:
        Same AnnData object with lineage attached
    """
    if has_lineage(adata):
        return adata

    # Create synthetic lineage for legacy file
    lineage = create_lineage_metadata(
        modality_name=modality_name,
        parent_modality=None,
        step_summary=step_summary or f"[Legacy] Loaded from {modality_name}.h5ad",
    )

    # Add synthetic flag to indicate this was auto-generated
    lineage_dict = lineage.to_dict()
    lineage_dict["_synthetic"] = True

    adata.uns[LINEAGE_KEY] = lineage_dict
    return adata


def get_lineage_dict(adata: "AnnData") -> Optional[Dict[str, Any]]:
    """
    Get raw lineage dictionary from AnnData.

    This returns the raw dict rather than LineageMetadata dataclass,
    useful for direct access to extra fields like _synthetic.

    Args:
        adata: AnnData object

    Returns:
        Raw lineage dict if present, None otherwise
    """
    return adata.uns.get(LINEAGE_KEY)
