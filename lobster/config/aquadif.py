"""
AQUADIF Tool Taxonomy Configuration.

This module defines the 10-category AQUADIF taxonomy for Lobster AI tools
and the provenance requirements for each category. AQUADIF makes the system
introspectable, enforceable, and teachable to coding agents by declaring
what each tool does (category) and whether it must produce provenance.

For full category definitions and usage examples, see:
    skills/lobster-dev/references/aquadif-contract.md

Categories:
    - IMPORT: Load external data formats into workspace
    - QUALITY: Assess data integrity and calculate QC metrics
    - FILTER: Subset data by removing samples/features
    - PREPROCESS: Transform data representation (normalize, batch correct, scale)
    - ANALYZE: Extract patterns, perform statistical tests, compute embeddings
    - ANNOTATE: Add biological meaning (cell types, gene names, pathway labels)
    - DELEGATE: Hand off work to specialist child agents
    - SYNTHESIZE: Combine or interpret results across analyses
    - UTILITY: Workspace management, status checks, exporting
    - CODE_EXEC: Custom code execution (escape hatch)

Provenance Requirements:
    Required (7 categories): IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, SYNTHESIZE
    Not required (3 categories): DELEGATE, UTILITY, CODE_EXEC (conditional)
"""

from enum import Enum
from typing import FrozenSet


class AquadifCategory(str, Enum):
    """
    AQUADIF tool taxonomy - 10 categories for classifying Lobster AI tools.

    Using (str, Enum) base allows string comparisons while maintaining type safety:
        AquadifCategory.IMPORT == "IMPORT"  # True
        AquadifCategory.IMPORT.value        # "IMPORT"

    Values match names for simplicity and JSON serialization compatibility.
    """

    IMPORT = "IMPORT"
    QUALITY = "QUALITY"
    FILTER = "FILTER"
    PREPROCESS = "PREPROCESS"
    ANALYZE = "ANALYZE"
    ANNOTATE = "ANNOTATE"
    DELEGATE = "DELEGATE"
    SYNTHESIZE = "SYNTHESIZE"
    UTILITY = "UTILITY"
    CODE_EXEC = "CODE_EXEC"


# Provenance-required categories (7 of 10)
# These categories represent operations that modify scientific data or produce
# analysis results that must be reproducible. Tools in these categories MUST
# call log_tool_usage(ir=ir) to track provenance.
PROVENANCE_REQUIRED: FrozenSet[AquadifCategory] = frozenset(
    {
        AquadifCategory.IMPORT,
        AquadifCategory.QUALITY,
        AquadifCategory.FILTER,
        AquadifCategory.PREPROCESS,
        AquadifCategory.ANALYZE,
        AquadifCategory.ANNOTATE,
        AquadifCategory.SYNTHESIZE,
    }
)


def requires_provenance(primary_category: str) -> bool:
    """
    Check if a category requires provenance tracking.

    Args:
        primary_category: Category string (e.g., "IMPORT", "UTILITY")

    Returns:
        True if the category requires provenance tracking, False otherwise

    Raises:
        ValueError: If the category string is not a valid AquadifCategory

    Example:
        >>> requires_provenance("IMPORT")
        True
        >>> requires_provenance("UTILITY")
        False
        >>> requires_provenance("INVALID")
        ValueError: 'INVALID' is not a valid AquadifCategory
    """
    try:
        category = AquadifCategory(primary_category)
    except ValueError as e:
        raise ValueError(
            f"'{primary_category}' is not a valid AquadifCategory. "
            f"Valid categories: {', '.join(cat.value for cat in AquadifCategory)}"
        ) from e

    return category in PROVENANCE_REQUIRED
