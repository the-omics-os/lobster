"""
Clinical trial metadata schema definitions for RECIST 1.1 and timepoint parsing.

This module provides schema definitions and utility functions for clinical trial
metadata, enabling natural language queries like "compare responders vs non-responders
at C2D1" to be properly interpreted.

Key features:
- RECIST 1.1 response category normalization
- Clinical trial timepoint parsing (C1D1, C2D1, etc.)
- Responder/Non-responder group classification
- Pydantic-based validation with graceful degradation

Used by:
- ClinicalMetadataService for metadata processing
- proteomics_expert for clinical proteomics analysis (Biognosys pilot)
"""

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# RECIST 1.1 Constants
# =============================================================================

# Canonical RECIST response codes
RECIST_RESPONSES: Dict[str, str] = {
    "CR": "Complete Response",
    "PR": "Partial Response",
    "SD": "Stable Disease",
    "PD": "Progressive Disease",
    "NE": "Not Evaluable",
}

# Response group classifications (sets for O(1) lookup)
RESPONDER_GROUP: Set[str] = {"CR", "PR"}
NON_RESPONDER_GROUP: Set[str] = {"SD", "PD"}

# Synonym mapping for response normalization (lowercase → canonical code)
# NOTE: "responder" and "non-responder" are GROUP classifications, not individual codes.
# They should be handled by classify_response_group(), not normalize_response().
RESPONSE_SYNONYMS: Dict[str, str] = {
    # Full names (lowercase)
    "complete response": "CR",
    "partial response": "PR",
    "stable disease": "SD",
    "progressive disease": "PD",
    "not evaluable": "NE",
    "not evaluated": "NE",
    # Abbreviations (for case-insensitive matching)
    "cr": "CR",
    "pr": "PR",
    "sd": "SD",
    "pd": "PD",
    "ne": "NE",
    # Common variations
    "complete": "CR",
    "partial": "PR",
    "stable": "SD",
    "progressive": "PD",
    "progression": "PD",
    # Clinical shorthand (unambiguous only)
    # NOTE: "resp" intentionally NOT mapped - ambiguous (could mean "responder" group or "response")
    "prog": "PD",
    "stab": "SD",
    # iRECIST categories (immune RECIST for immunotherapy trials)
    # These map to their RECIST equivalents for classification purposes
    "icr": "CR",  # immune Complete Response
    "ipr": "PR",  # immune Partial Response
    "isd": "SD",  # immune Stable Disease
    "ipd": "PD",  # immune Progressive Disease (confirmed)
    "icpd": "PD",  # immune Confirmed Progressive Disease
    "iupd": "PD",  # immune Unconfirmed Progressive Disease (treat as PD for grouping)
    "immune complete response": "CR",
    "immune partial response": "PR",
    "immune stable disease": "SD",
    "immune progressive disease": "PD",
}


# =============================================================================
# Timepoint Parsing
# =============================================================================

# Regex patterns for timepoint parsing (ordered by specificity)
# Each tuple: (compiled_pattern, extractor_function)
TIMEPOINT_PATTERNS: List[Tuple[re.Pattern, Any]] = [
    # C1D1 format (most common in oncology trials)
    (re.compile(r"^C(\d+)D(\d+)$", re.IGNORECASE), lambda m: (int(m.group(1)), int(m.group(2)))),
    # Cycle 1 Day 1 format (verbose)
    (
        re.compile(r"^Cycle\s*(\d+)\s*Day\s*(\d+)$", re.IGNORECASE),
        lambda m: (int(m.group(1)), int(m.group(2))),
    ),
    # W1D1 format (weekly cycles)
    (re.compile(r"^W(\d+)D(\d+)$", re.IGNORECASE), lambda m: (int(m.group(1)), int(m.group(2)))),
    # Week 1 Day 1 format (verbose weekly)
    (
        re.compile(r"^Week\s*(\d+)\s*Day\s*(\d+)$", re.IGNORECASE),
        lambda m: (int(m.group(1)), int(m.group(2))),
    ),
]

# Special timepoints that don't follow cycle/day pattern
SPECIAL_TIMEPOINTS: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "baseline": (0, 0),
    "screening": (0, 0),
    "pre-treatment": (0, 0),
    "pretreatment": (0, 0),
    "pre_treatment": (0, 0),
    "day0": (0, 0),
    "d0": (0, 0),
    "eot": (None, None),  # End of Treatment - no cycle/day representation
    "end of treatment": (None, None),
    "follow-up": (None, None),
    "followup": (None, None),
    "follow_up": (None, None),
}


def parse_timepoint(timepoint: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse clinical trial timepoint notation.

    Supported formats:
    - C1D1, C2D8 (Cycle Day) - most common in oncology
    - Cycle 1 Day 1 (verbose)
    - W1D1 (Week Day)
    - Baseline, Screening (special timepoints)

    Args:
        timepoint: Timepoint string to parse

    Returns:
        Tuple of (cycle, day) or (None, None) if unparseable.
        Special cases:
        - Baseline/Screening → (0, 0)
        - EOT/Follow-up → (None, None)
        - Invalid → (None, None) with warning logged
    """
    if not timepoint or not isinstance(timepoint, str):
        return (None, None)

    timepoint_clean = timepoint.strip()

    # Check special timepoints first
    timepoint_lower = timepoint_clean.lower()
    if timepoint_lower in SPECIAL_TIMEPOINTS:
        return SPECIAL_TIMEPOINTS[timepoint_lower]

    # Try regex patterns
    for pattern, extractor in TIMEPOINT_PATTERNS:
        match = pattern.match(timepoint_clean)
        if match:
            try:
                return extractor(match)
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to extract cycle/day from '{timepoint}': {e}")
                return (None, None)

    # Unknown format
    logger.debug(f"Unknown timepoint format: '{timepoint}'")
    return (None, None)


def timepoint_to_absolute_day(
    timepoint: str,
    cycle_length_days: int = 21,
) -> Optional[int]:
    """
    Convert timepoint notation to absolute day since treatment start.

    Args:
        timepoint: Timepoint string (e.g., "C2D1")
        cycle_length_days: Days per treatment cycle (default 21 for 3-week cycles)

    Returns:
        Absolute day number (1-indexed), or None if cannot be calculated.

    Examples:
        - C1D1 with 21-day cycles → 1
        - C2D1 with 21-day cycles → 22
        - C2D8 with 21-day cycles → 29
        - Baseline → 0
        - EOT → None (no absolute day)
    """
    cycle, day = parse_timepoint(timepoint)

    if cycle is None or day is None:
        return None

    if cycle == 0 and day == 0:
        return 0  # Baseline

    # Calculate absolute day: (cycle - 1) * cycle_length + day
    return (cycle - 1) * cycle_length_days + day


# =============================================================================
# Response Classification Functions
# =============================================================================


def normalize_response(value: Any) -> Optional[str]:
    """
    Normalize a response value to canonical RECIST code.

    This function handles individual response codes and their synonyms,
    NOT group classifications like "responder" or "non-responder".

    Args:
        value: Response value to normalize (string or None)

    Returns:
        Canonical RECIST code (CR, PR, SD, PD, NE) or None if invalid/empty.

    Examples:
        >>> normalize_response("complete response")
        'CR'
        >>> normalize_response("CR")
        'CR'
        >>> normalize_response("stable")
        'SD'
        >>> normalize_response("responder")  # NOT a valid single code
        None
    """
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    v_str = str(value).strip()
    if not v_str:
        return None

    v_lower = v_str.lower()

    # Direct match (already canonical)
    if v_str.upper() in RECIST_RESPONSES:
        return v_str.upper()

    # Synonym lookup
    if v_lower in RESPONSE_SYNONYMS:
        return RESPONSE_SYNONYMS[v_lower]

    # "responder" and "non-responder" are GROUP classifications
    # They should use classify_response_group() instead
    if v_lower in ("responder", "non-responder", "nonresponder", "non_responder"):
        logger.debug(
            f"'{value}' is a group classification, not a RECIST code. "
            "Use classify_response_group() or create_responder_groups() instead."
        )
        return None

    logger.debug(f"Unknown response value: '{value}'")
    return None


def classify_response_group(response_code: str) -> Optional[str]:
    """
    Classify a RECIST response code into responder/non-responder groups.

    Args:
        response_code: Canonical RECIST code (CR, PR, SD, PD, NE)

    Returns:
        "responder" if CR or PR
        "non_responder" if SD or PD
        None if NE or invalid

    Examples:
        >>> classify_response_group("CR")
        'responder'
        >>> classify_response_group("SD")
        'non_responder'
        >>> classify_response_group("NE")
        None
    """
    if not response_code:
        return None

    code_upper = response_code.upper().strip()

    if code_upper in RESPONDER_GROUP:
        return "responder"
    elif code_upper in NON_RESPONDER_GROUP:
        return "non_responder"
    else:
        return None


def is_responder(response_code: str) -> bool:
    """
    Check if a RECIST response code indicates a responder.

    Args:
        response_code: RECIST code (CR, PR, SD, PD, NE)

    Returns:
        True if CR or PR, False otherwise
    """
    return classify_response_group(response_code) == "responder"


def is_non_responder(response_code: str) -> bool:
    """
    Check if a RECIST response code indicates a non-responder.

    Args:
        response_code: RECIST code (CR, PR, SD, PD, NE)

    Returns:
        True if SD or PD, False otherwise
    """
    return classify_response_group(response_code) == "non_responder"


# =============================================================================
# Pydantic Schema: ClinicalSample
# =============================================================================


class ClinicalSample(BaseModel):
    """
    Clinical trial sample metadata following RECIST 1.1 standards.

    This Pydantic model validates and normalizes clinical trial metadata,
    enabling natural language queries like "compare responders vs non-responders".

    Used by ClinicalMetadataService for:
    - Per-sample metadata validation
    - RECIST response normalization
    - Timepoint parsing and absolute day calculation
    - Responder group classification

    Attributes:
        sample_id: Unique sample identifier (required)
        patient_id: Patient/subject identifier
        response_status: Normalized RECIST 1.1 code (CR, PR, SD, PD, NE)
        response_group: Derived group classification (responder/non_responder)
        pfs_days: Progression-Free Survival in days
        pfs_event: PFS event indicator (1=event, 0=censored)
        os_days: Overall Survival in days
        os_event: OS event indicator (1=event, 0=censored)
        timepoint: Original timepoint string (e.g., "C2D1")
        cycle: Parsed cycle number
        day: Parsed day within cycle
        absolute_day: Calculated days since treatment start
        age: Patient age in years
        sex: Patient sex (M/F)
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    # Identifiers
    sample_id: str = Field(..., description="Unique sample identifier")
    patient_id: Optional[str] = Field(None, description="Patient/subject identifier")

    # RECIST Response (normalized to canonical codes)
    response_status: Optional[str] = Field(
        None, description="RECIST 1.1 response code (CR, PR, SD, PD, NE)"
    )
    response_group: Optional[str] = Field(
        None, description="Derived response group (responder/non_responder)"
    )

    # Survival endpoints
    pfs_days: Optional[float] = Field(
        None, ge=0, description="Progression-Free Survival in days"
    )
    pfs_event: Optional[int] = Field(
        None, ge=0, le=1, description="PFS event indicator (1=event, 0=censored)"
    )
    os_days: Optional[float] = Field(None, ge=0, description="Overall Survival in days")
    os_event: Optional[int] = Field(
        None, ge=0, le=1, description="OS event indicator (1=event, 0=censored)"
    )

    # Timepoint information
    timepoint: Optional[str] = Field(None, description='Original timepoint string (e.g., "C2D1")')
    cycle: Optional[int] = Field(None, ge=0, description="Parsed cycle number")
    day: Optional[int] = Field(None, ge=0, description="Parsed day within cycle")
    absolute_day: Optional[int] = Field(
        None, ge=0, description="Calculated days since treatment start"
    )

    # Demographics
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age in years")
    sex: Optional[Literal["M", "F"]] = Field(None, description="Patient sex")

    # Additional metadata
    additional_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional custom metadata fields"
    )

    @field_validator("response_status", mode="before")
    @classmethod
    def normalize_response_validator(cls, v: Any) -> Optional[str]:
        """Normalize response status to canonical RECIST code."""
        return normalize_response(v)

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex(cls, v: Any) -> Optional[str]:
        """
        Normalize sex to M/F.

        NOTE: Numeric mappings (1, 0, 2) are intentionally NOT supported due to
        conflicting conventions across datasets:
        - ISO 5218: 1=Male, 2=Female, 0=Not known
        - Some datasets: 0=Female, 1=Male
        - CDISC: 1=Male, 2=Female

        Users should use explicit string labels or apply column_mapping.
        """
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None

        v_str = str(v).strip().upper()
        if not v_str:
            return None

        # Accept explicit string labels only (no numeric codes due to conflicting conventions)
        if v_str in ("M", "MALE", "MAN", "BOY"):
            return "M"
        if v_str in ("F", "FEMALE", "WOMAN", "GIRL"):
            return "F"

        # Warn about numeric values
        if v_str.isdigit():
            logger.warning(
                f"Numeric sex value '{v}' not supported due to conflicting conventions. "
                "Use explicit labels (M/F/Male/Female) or column_mapping to transform."
            )
            return None

        logger.debug(f"Unknown sex value: '{v}'")
        return None

    @model_validator(mode="after")
    def derive_computed_fields(self) -> "ClinicalSample":
        """Derive response_group and timepoint components."""
        # Derive response_group from response_status
        if self.response_status and not self.response_group:
            self.response_group = classify_response_group(self.response_status)

        # Parse timepoint if provided but cycle/day not set
        if self.timepoint and self.cycle is None and self.day is None:
            cycle, day = parse_timepoint(self.timepoint)
            if cycle is not None:
                self.cycle = cycle
            if day is not None:
                self.day = day

        return self

    def compute_absolute_day(self, cycle_length_days: int = 21) -> Optional[int]:
        """
        Compute absolute day from cycle/day with given cycle length.

        Args:
            cycle_length_days: Days per cycle (default 21)

        Returns:
            Absolute day number or None
        """
        if self.cycle is None or self.day is None:
            return None

        if self.cycle == 0 and self.day == 0:
            return 0

        return (self.cycle - 1) * cycle_length_days + self.day

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = self.model_dump(exclude={"additional_metadata"}, exclude_none=True)
        if self.additional_metadata:
            base_dict.update(self.additional_metadata)
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalSample":
        """
        Create schema from dictionary, automatically handling unknown fields.

        Args:
            data: Dictionary with metadata fields

        Returns:
            ClinicalSample: Validated schema instance
        """
        # Extract known fields
        known_fields = set(cls.model_fields.keys()) - {"additional_metadata"}
        schema_data = {k: v for k, v in data.items() if k in known_fields}

        # Put remaining fields in additional_metadata
        additional = {k: v for k, v in data.items() if k not in known_fields}
        if additional:
            schema_data["additional_metadata"] = additional

        return cls(**schema_data)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Constants
    "RECIST_RESPONSES",
    "RESPONDER_GROUP",
    "NON_RESPONDER_GROUP",
    "RESPONSE_SYNONYMS",
    "TIMEPOINT_PATTERNS",
    "SPECIAL_TIMEPOINTS",
    # Functions
    "parse_timepoint",
    "timepoint_to_absolute_day",
    "normalize_response",
    "classify_response_group",
    "is_responder",
    "is_non_responder",
    # Schema
    "ClinicalSample",
]
