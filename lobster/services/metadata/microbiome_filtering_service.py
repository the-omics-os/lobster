"""
Microbiome Filtering Service

Stateless service for validating microbiome metadata using fuzzy matching and
multi-field detection. Returns 3-tuple: (filtered_data, stats, ir).

Capabilities:
- 16S amplicon detection via OR-based keyword matching (platform/library/assay)
- Host organism validation with fuzzy string matching
- Configurable strictness and host allowlists
"""

import difflib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from lobster.core.analysis_ir import AnalysisStep

logger = logging.getLogger(__name__)


# Host organism aliases for fuzzy matching
HOST_ALIASES = {
    "Human": [
        "Homo sapiens",
        "homo sapiens",
        "human",
        "Human",
        "HUMAN",
        "h. sapiens",
        "H. sapiens",
        "hsapiens",
    ],
    "Mouse": [
        "Mus musculus",
        "mus musculus",
        "mouse",
        "Mouse",
        "MOUSE",
        "m. musculus",
        "M. musculus",
        "mmusculus",
    ],
}


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    reason: str
    matched_field: Optional[str] = None
    matched_value: Optional[str] = None
    confidence: Optional[float] = None


class MicrobiomeFilteringService:
    """
    Service for microbiome metadata validation.

    Follows Lobster service pattern:
    - Stateless operations
    - Returns (data, stats, ir) tuples
    - W3C-PROV compliant IR generation
    """

    # 16S amplicon detection keywords (case-insensitive)
    AMPLICON_KEYWORDS = {
        "platform": ["illumina", "miseq", "hiseq", "nextseq", "pacbio", "ion torrent"],
        "library_strategy": [
            "amplicon",
            "16s",
            "16s rrna",
            "16s amplicon",
            "targeted locus",
        ],
        "assay_type": [
            "16s sequencing",
            "amplicon sequencing",
            "metagenomics 16s",
            "16s metagenomics",
        ],
    }

    def __init__(self):
        """Initialize the service."""
        pass

    def validate_16s_amplicon(
        self, metadata: Dict[str, Any], strict: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Validate if metadata represents a 16S amplicon study.

        Uses OR-based detection across multiple fields (platform, library_strategy,
        assay_type). In non-strict mode, matches any keyword. In strict mode,
        requires exact matches.

        Args:
            metadata: Metadata dictionary to validate
            strict: If True, require exact keyword matches (default: False)

        Returns:
            Tuple of:
            - filtered_metadata: Original metadata if valid, empty dict if invalid
            - stats: Validation summary statistics
            - ir: AnalysisStep for provenance tracking

        Example:
            >>> service = MicrobiomeFilteringService()
            >>> metadata = {"platform": "Illumina MiSeq", "library_strategy": "AMPLICON"}
            >>> result, stats, ir = service.validate_16s_amplicon(metadata)
            >>> stats["is_valid"]
            True
        """
        result = self._check_16s_amplicon(metadata, strict)

        # Filter metadata based on validation
        filtered_metadata = metadata if result.is_valid else {}

        # Build stats
        stats = {
            "is_valid": result.is_valid,
            "reason": result.reason,
            "matched_field": result.matched_field,
            "matched_value": result.matched_value,
            "strict_mode": strict,
            "fields_checked": list(self.AMPLICON_KEYWORDS.keys()),
        }

        # Generate IR
        ir = self._create_16s_ir(metadata, strict, result)

        return filtered_metadata, stats, ir

    def validate_host_organism(
        self,
        metadata: Dict[str, Any],
        allowed_hosts: Optional[List[str]] = None,
        fuzzy_threshold: float = 85.0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Validate host organism using fuzzy string matching.

        Args:
            metadata: Metadata dictionary containing host organism info
            allowed_hosts: List of allowed host names (default: ["Human", "Mouse"])
            fuzzy_threshold: Minimum fuzzy match score (0-100, default: 85.0)

        Returns:
            Tuple of:
            - filtered_metadata: Original metadata if valid, empty dict if invalid
            - stats: Validation summary with match scores
            - ir: AnalysisStep for provenance tracking

        Example:
            >>> service = MicrobiomeFilteringService()
            >>> metadata = {"organism": "Homo sapiens"}
            >>> result, stats, ir = service.validate_host_organism(metadata)
            >>> stats["is_valid"]
            True
            >>> stats["matched_host"]
            'Human'
        """
        if allowed_hosts is None:
            allowed_hosts = ["Human", "Mouse"]

        result = self._check_host_organism(metadata, allowed_hosts, fuzzy_threshold)

        # Filter metadata
        filtered_metadata = metadata if result.is_valid else {}

        # Build stats
        stats = {
            "is_valid": result.is_valid,
            "reason": result.reason,
            "matched_host": result.matched_value if result.is_valid else None,
            "confidence_score": result.confidence,
            "allowed_hosts": allowed_hosts,
            "fuzzy_threshold": fuzzy_threshold,
        }

        # Generate IR
        ir = self._create_host_ir(metadata, allowed_hosts, fuzzy_threshold, result)

        return filtered_metadata, stats, ir

    # -------------------------------------------------------------------------
    # Internal Validation Logic
    # -------------------------------------------------------------------------

    def _check_16s_amplicon(
        self, metadata: Dict[str, Any], strict: bool
    ) -> ValidationResult:
        """
        Check if metadata contains 16S amplicon indicators.

        Args:
            metadata: Metadata to check
            strict: Use strict matching

        Returns:
            ValidationResult with match details
        """
        # Check each field type
        for field_type, keywords in self.AMPLICON_KEYWORDS.items():
            # Try common field name variations
            field_names = [
                field_type,
                field_type.replace("_", " "),
                field_type.replace("_", ""),
            ]

            for field_name in field_names:
                if field_name in metadata:
                    value = self._normalize_field(metadata[field_name])
                    if self._contains_16s(value, keywords, strict):
                        return ValidationResult(
                            is_valid=True,
                            reason=f"16S amplicon detected in {field_type}",
                            matched_field=field_name,
                            matched_value=metadata[field_name],
                        )

        return ValidationResult(
            is_valid=False,
            reason="No 16S amplicon indicators found in metadata",
        )

    def _check_host_organism(
        self,
        metadata: Dict[str, Any],
        allowed_hosts: List[str],
        threshold: float,
    ) -> ValidationResult:
        """
        Check if metadata contains allowed host organism.

        Args:
            metadata: Metadata to check
            allowed_hosts: List of allowed host names
            threshold: Minimum fuzzy match score

        Returns:
            ValidationResult with match details
        """
        # Try common organism field names
        organism_fields = [
            "organism",
            "host",
            "host_organism",
            "source",
            "taxon",
            "species",
        ]

        for field_name in organism_fields:
            if field_name in metadata:
                organism_value = str(metadata[field_name])
                match_result = self._match_host(
                    organism_value, allowed_hosts, threshold
                )

                if match_result is not None:
                    matched_host, score = match_result
                    return ValidationResult(
                        is_valid=True,
                        reason=f"Host organism matched: {matched_host}",
                        matched_field=field_name,
                        matched_value=matched_host,
                        confidence=score,
                    )

        return ValidationResult(
            is_valid=False,
            reason=f"Host organism not in allowed list: {allowed_hosts}",
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _normalize_field(self, value: Any) -> str:
        """
        Normalize field value for comparison.

        Args:
            value: Field value to normalize

        Returns:
            Lowercase string representation
        """
        return str(value).lower().strip()

    def _contains_16s(self, value: str, keywords: List[str], strict: bool) -> bool:
        """
        Check if value contains 16S amplicon keywords.

        Args:
            value: Normalized field value
            keywords: List of keywords to check
            strict: If True, require exact match; if False, allow substring match

        Returns:
            True if match found
        """
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if strict:
                # Exact match
                if value == keyword_lower:
                    return True
            else:
                # Substring match
                if keyword_lower in value:
                    return True
        return False

    def _match_host(
        self, organism_value: str, allowed_hosts: List[str], threshold: float
    ) -> Optional[Tuple[str, float]]:
        """
        Match organism value against allowed hosts using fuzzy matching.

        Uses difflib.SequenceMatcher for fuzzy string matching (0-100 scale).

        Args:
            organism_value: Organism string from metadata
            allowed_hosts: List of allowed host names
            threshold: Minimum fuzzy match score (0-100)

        Returns:
            Tuple of (matched_host, score) if match found, None otherwise
        """
        best_match = None
        best_score = 0.0

        for host in allowed_hosts:
            # Get aliases for this host
            aliases = HOST_ALIASES.get(host, [host])

            for alias in aliases:
                # Compute fuzzy match score using difflib
                # SequenceMatcher.ratio() returns 0.0-1.0, scale to 0-100
                matcher = difflib.SequenceMatcher(
                    None, organism_value.lower(), alias.lower()
                )
                score = matcher.ratio() * 100.0

                if score > best_score:
                    best_score = score
                    best_match = host

        # Return match if score exceeds threshold
        if best_score >= threshold:
            return (best_match, best_score)

        return None

    # -------------------------------------------------------------------------
    # IR Generation (W3C-PROV Compliance)
    # -------------------------------------------------------------------------

    def _create_16s_ir(
        self,
        metadata: Dict[str, Any],
        strict: bool,
        result: ValidationResult,
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for 16S amplicon validation.

        Args:
            metadata: Input metadata
            strict: Strict mode flag
            result: Validation result

        Returns:
            AnalysisStep for provenance tracking
        """
        code_template = """
# 16S Amplicon Validation
metadata = {{ metadata }}
strict = {{ strict }}

# Keywords for detection
AMPLICON_KEYWORDS = {
    "platform": ["illumina", "miseq", "hiseq", "nextseq", "pacbio", "ion torrent"],
    "library_strategy": ["amplicon", "16s", "16s rrna", "16s amplicon", "targeted locus"],
    "assay_type": ["16s sequencing", "amplicon sequencing", "metagenomics 16s", "16s metagenomics"]
}

# Check each field type
is_valid = False
matched_field = None
for field_type, keywords in AMPLICON_KEYWORDS.items():
    if field_type in metadata:
        value = str(metadata[field_type]).lower().strip()
        for keyword in keywords:
            if strict and value == keyword.lower():
                is_valid = True
                matched_field = field_type
                break
            elif not strict and keyword.lower() in value:
                is_valid = True
                matched_field = field_type
                break
    if is_valid:
        break

print(f"Validation result: {is_valid}")
if matched_field:
    print(f"Matched field: {matched_field}")
"""

        return AnalysisStep(
            operation="microbiome.filtering.validate_16s_amplicon",
            tool_name="MicrobiomeFilteringService.validate_16s_amplicon",
            description=f"Validate 16S amplicon metadata (strict={strict})",
            library="lobster.tools.microbiome_filtering_service",
            code_template=code_template.strip(),
            imports=[],
            parameters={"metadata": metadata, "strict": strict},
            parameter_schema={
                "metadata": {"type": "dict", "description": "Metadata to validate"},
                "strict": {
                    "type": "bool",
                    "default": False,
                    "description": "Require exact keyword matches",
                },
            },
            input_entities=[{"type": "metadata", "name": "input_metadata"}],
            output_entities=[
                {
                    "type": "validation_result",
                    "name": "filtered_metadata",
                    "is_valid": result.is_valid,
                }
            ],
        )

    def _create_host_ir(
        self,
        metadata: Dict[str, Any],
        allowed_hosts: List[str],
        threshold: float,
        result: ValidationResult,
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for host organism validation.

        Args:
            metadata: Input metadata
            allowed_hosts: Allowed host list
            threshold: Fuzzy match threshold
            result: Validation result

        Returns:
            AnalysisStep for provenance tracking
        """
        code_template = """
# Host Organism Validation with Fuzzy Matching
import difflib

metadata = {{ metadata }}
allowed_hosts = {{ allowed_hosts }}
threshold = {{ threshold }}

# Host aliases
HOST_ALIASES = {
    "Human": ["Homo sapiens", "homo sapiens", "human", "Human", "HUMAN", "h. sapiens"],
    "Mouse": ["Mus musculus", "mus musculus", "mouse", "Mouse", "MOUSE", "m. musculus"]
}

# Find organism field
organism_fields = ["organism", "host", "host_organism", "source", "taxon", "species"]
organism_value = None
for field in organism_fields:
    if field in metadata:
        organism_value = str(metadata[field])
        break

# Fuzzy match against allowed hosts using difflib
is_valid = False
matched_host = None
best_score = 0.0

if organism_value:
    for host in allowed_hosts:
        aliases = HOST_ALIASES.get(host, [host])
        for alias in aliases:
            matcher = difflib.SequenceMatcher(None, organism_value.lower(), alias.lower())
            score = matcher.ratio() * 100.0
            if score > best_score:
                best_score = score
                matched_host = host

    if best_score >= threshold:
        is_valid = True

print(f"Validation result: {is_valid}")
if matched_host:
    print(f"Matched host: {matched_host} (score: {best_score:.1f})")
"""

        return AnalysisStep(
            operation="microbiome.filtering.validate_host_organism",
            tool_name="MicrobiomeFilteringService.validate_host_organism",
            description=f"Validate host organism with fuzzy matching (threshold={threshold})",
            library="lobster.tools.microbiome_filtering_service",
            code_template=code_template.strip(),
            imports=["import difflib"],
            parameters={
                "metadata": metadata,
                "allowed_hosts": allowed_hosts,
                "fuzzy_threshold": threshold,
            },
            parameter_schema={
                "metadata": {"type": "dict", "description": "Metadata to validate"},
                "allowed_hosts": {
                    "type": "list",
                    "default": ["Human", "Mouse"],
                    "description": "Allowed host organism names",
                },
                "fuzzy_threshold": {
                    "type": "float",
                    "default": 85.0,
                    "description": "Minimum fuzzy match score (0-100)",
                },
            },
            input_entities=[{"type": "metadata", "name": "input_metadata"}],
            output_entities=[
                {
                    "type": "validation_result",
                    "name": "filtered_metadata",
                    "is_valid": result.is_valid,
                    "confidence": result.confidence,
                }
            ],
        )
