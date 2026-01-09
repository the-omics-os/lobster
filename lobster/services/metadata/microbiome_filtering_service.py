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
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from lobster.core.analysis_ir import AnalysisStep

logger = logging.getLogger(__name__)


# Host organism aliases for fuzzy matching
HOST_ALIASES = {
    "Human": [
        # Individual organism (traditional)
        "Homo sapiens",
        "homo sapiens",
        "human",
        "Human",
        "HUMAN",
        "h. sapiens",
        "H. sapiens",
        "hsapiens",
        # Metagenome variants (Bug 4 fix - DataBioMix)
        # CRITICAL: Only add EXPLICIT human variants to avoid false positives
        # 16S amplicon studies use metagenome organism names instead of "Homo sapiens"
        "human gut metagenome",              # 800+ samples (DataBioMix)
        "human metagenome",                  # 200+ samples (generic)
        "human fecal metagenome",            # Alternative spelling
        "human feces metagenome",            # NCBI variant
        "human oral metagenome",             # Oral microbiome studies
        "human skin metagenome",             # Skin microbiome
        "human nasal metagenome",            # Nasal/respiratory
        "human vaginal metagenome",          # Vaginal microbiome
        "human respiratory tract metagenome",
        "human urogenital metagenome",
        # DO NOT ADD ambiguous terms (could match mouse/rat):
        # - "gut metagenome" (ambiguous)
        # - "metagenome" (too generic)
        # - "fecal metagenome" (could be any host)
    ],
    "Mouse": [
        # Individual organism
        "Mus musculus",
        "mus musculus",
        "mouse",
        "Mouse",
        "MOUSE",
        "m. musculus",
        "M. musculus",
        "mmusculus",
        # Metagenome variants
        "mouse gut metagenome",
        "mouse metagenome",
        "mouse fecal metagenome",
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
    # NOTE: Generic "amplicon" removed to prevent misclassification
    # (e.g., ITS amplicon, 18S amplicon, COI barcoding should NOT match)
    AMPLICON_KEYWORDS = {
        "platform": ["illumina", "miseq", "hiseq", "nextseq", "pacbio", "ion torrent"],
        "library_strategy": [
            "16s",
            "16s rrna",
            "16s amplicon",
            "16s rdna",
            "16s ribosomal",
            "v3v4",       # Common 16S hypervariable region
            "v4",         # Common 16S hypervariable region
            "v1v3",       # 16S hypervariable region
            "v3v5",       # 16S hypervariable region
            # REMOVED: "amplicon" - too generic (could be ITS, 18S, COI)
            # REMOVED: "targeted locus" - too generic (any targeted sequencing)
        ],
        "assay_type": [
            "16s sequencing",
            "16s metagenomics",
            "metagenomics 16s",
            "16s rrna sequencing",
            "16s profiling",
            "bacterial 16s",
            # REMOVED: "amplicon sequencing" - too generic
        ],
    }

    # Amplicon region detection patterns
    V_REGION_PATTERNS = [
        r'\b(V\d)(?:\s*-\s*V\d)?\b',           # V4, V3-V4, V1-V9
        r'\bvariable\s+region\s+(\d+)\b',      # "variable region 4"
        r'\b(V\d+)\s*-\s*(V\d+)\b',            # V3 - V4 (with spaces)
        r'\bfull[- ]?length\s+16S\b',          # "full length 16S"
        r'\b16S\s+(V\d+(?:-V\d+)?)\b',         # "16S V4", "16S V3-V4"
    ]

    # Primer pair to region mappings (most common in microbiome research)
    PRIMER_REGION_MAP = {
        # Earth Microbiome Project (most common)
        ("515F", "806R"): "V4",
        ("515f", "806r"): "V4",  # Case variants

        # Klindworth et al. 2013 (human microbiome)
        ("341F", "785R"): "V3-V4",
        ("341f", "785r"): "V3-V4",

        # Lane 1991 (full-length, rare)
        ("27F", "1492R"): "full-length",
        ("27f", "1492r"): "full-length",

        # Other common primers
        ("515F", "926R"): "V4-V5",
        ("515f", "926r"): "V4-V5",
        ("343F", "926R"): "V3-V5",
        ("343f", "926r"): "V3-V5",
    }

    # Standard region names (normalized)
    STANDARD_REGIONS = [
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V1-V2", "V1-V3", "V3-V4", "V4-V5", "V1-V9",
        "full-length"
    ]

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

    # Sample type detection keywords (case-insensitive)
    # Maps canonical sample type to keywords found in isolation_source, body_site, etc.
    # NOTE: Generic terms removed to prevent misclassification
    # Scientific principle: Luminal ≠ Mucosal, Stool ≠ Intestinal Content
    SAMPLE_TYPE_KEYWORDS = {
        # Fecal stool samples (distal colon, passed stool)
        "fecal_stool": [
            "fecal", "feces", "stool",
            "fecal sample", "stool sample",
            "faecal", "faeces",  # British spelling
            "rectal swab",  # Distal sampling
            "fecal matter", "stool specimen",
        ],

        # Gut luminal content (intestinal lumen, not passed)
        "gut_luminal_content": [
            "gut content", "intestinal content",
            "ileal content", "cecal content", "caecal content",
            "colonic content", "colon content",
            "luminal content", "lumen",
            "intestinal lumen", "gut lumen",
        ],

        # Gut mucosal biopsies (tissue-associated microbiome)
        "gut_mucosal_biopsy": [
            # Organ-specific biopsy terms
            "gut biopsy", "intestinal biopsy",
            "colon biopsy", "colonic biopsy",
            "rectal biopsy", "ileal biopsy",
            "cecal biopsy", "duodenal biopsy",
            "jejunal biopsy",
            # Organ-specific tissue terms
            "gut tissue", "intestinal tissue",
            "colon tissue", "rectal tissue",
            # Mucosal terms (with organ context)
            "colonic mucosa", "rectal mucosa",
            "intestinal mucosa", "gut mucosa",
            "mucosal biopsy",
            # Epithelial terms
            "intestinal epithelium", "gut epithelium",
        ],

        # Gut lavage (bowel preparation, potential artifacts)
        "gut_lavage": [
            "lavage", "colonic lavage",
            "bowel prep", "bowel preparation",
            "colon wash", "intestinal wash",
        ],

        # Oral samples
        "oral": [
            "oral", "saliva", "mouth", "tongue",
            "dental", "plaque", "gingiva", "gingival",
            "buccal", "subgingival", "supragingival",
        ],

        # Skin samples
        "skin": [
            "skin", "dermal", "cutaneous",
            "epidermis", "sebaceous",
        ],
    }

    # Backward compatibility aliases (DEPRECATED - will be removed in v2.0)
    SAMPLE_TYPE_ALIASES = {
        "fecal": "fecal_stool",  # Maps to specific category with warning
        "stool": "fecal_stool",  # Explicit mapping
        "luminal": "gut_luminal_content",  # Short alias
        "biopsy": "gut_mucosal_biopsy",  # Short alias
        # "gut" is NOT aliased - too ambiguous, will raise error
    }

    # Shotgun metagenomic detection keywords (case-insensitive)
    SHOTGUN_KEYWORDS = {
        "library_strategy": [
            "wgs",                      # Whole Genome Shotgun
            "wxs",                      # Whole Exome Shotgun
            "metagenomic",              # Generic metagenomic
            "whole genome shotgun",
            "shotgun metagenomics",
        ],
        "assay_type": [
            "metagenomic sequencing",
            "shotgun metagenomics",
            "whole genome sequencing",
            "metagenomic shotgun",
        ],
    }

    def validate_shotgun(
        self, metadata: Dict[str, Any], strict: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Validate if metadata represents shotgun metagenomic sequencing.

        Uses OR-based detection across library_strategy and assay_type fields.
        Explicitly EXCLUDES amplification-based methods (WGA, AMPLICON) to ensure
        unbiased metagenomic profiling.

        Args:
            metadata: Sample metadata dictionary with SRA fields
            strict: If True, require exact keyword match. If False, allow substring match (default: False)

        Returns:
            (filtered_metadata, stats, ir): Standard 3-tuple
            - filtered_metadata: Non-empty dict if valid shotgun, empty dict if invalid
            - stats: {"is_valid": bool, "matched_field": str, "matched_value": str}
            - ir: AnalysisStep for provenance tracking

        Examples:
            >>> service = MicrobiomeFilteringService()
            >>> metadata = {"library_strategy": "WGS", "assay_type": "metagenomic sequencing"}
            >>> filtered, stats, ir = service.validate_shotgun(metadata, strict=False)
            >>> stats["is_valid"]
            True

            >>> wga_metadata = {"library_strategy": "WGA"}
            >>> filtered, stats, ir = service.validate_shotgun(wga_metadata, strict=False)
            >>> stats["is_valid"]
            False  # WGA excluded (amplification-based)
        """
        library_strategy = str(metadata.get("library_strategy", "")).lower()
        assay_type = str(metadata.get("assay_type", "")).lower()

        # EXCLUSION RULES: Reject amplification-based methods
        # WGA and AMPLICON introduce bias incompatible with shotgun metagenomics
        exclusion_patterns = ["amplicon", "wga", "targeted"]
        for pattern in exclusion_patterns:
            if pattern in library_strategy or pattern in assay_type:
                stats = {
                    "is_valid": False,
                    "reason": f"excluded_{pattern}_based",
                    "library_strategy": library_strategy,
                    "assay_type": assay_type,
                    "strict": strict,
                }
                return {}, stats, None  # Rejected

        # INCLUSION RULES: Check for shotgun indicators
        is_valid = False
        matched_field = None
        matched_value = None

        if strict:
            # Exact match required
            if library_strategy in self.SHOTGUN_KEYWORDS["library_strategy"]:
                is_valid = True
                matched_field = "library_strategy"
                matched_value = library_strategy
            elif assay_type in self.SHOTGUN_KEYWORDS["assay_type"]:
                is_valid = True
                matched_field = "assay_type"
                matched_value = assay_type
        else:
            # Substring match (more permissive) - recommended for microbiome
            for keyword in self.SHOTGUN_KEYWORDS["library_strategy"]:
                if keyword in library_strategy:
                    is_valid = True
                    matched_field = "library_strategy"
                    matched_value = library_strategy
                    break

            if not is_valid:
                for keyword in self.SHOTGUN_KEYWORDS["assay_type"]:
                    if keyword in assay_type:
                        is_valid = True
                        matched_field = "assay_type"
                        matched_value = assay_type
                        break

        stats = {
            "is_valid": is_valid,
            "matched_field": matched_field,
            "matched_value": matched_value,
            "strict": strict,
            "library_strategy": library_strategy,
            "assay_type": assay_type,
        }

        # Create IR for provenance
        ir = AnalysisStep(
            operation="validate_shotgun_metagenomic",
            tool_name="MicrobiomeFilteringService.validate_shotgun",
            description=f"Validate shotgun metagenomic sequencing (strict={strict})",
            library="lobster.services.metadata.microbiome_filtering_service",
            code_template="""# Validate shotgun metagenomic sequencing
library_strategy = metadata.get('library_strategy', '').lower()
assay_type = metadata.get('assay_type', '').lower()

# Exclude amplification-based methods (WGA, AMPLICON)
if 'amplicon' in library_strategy or 'wga' in library_strategy:
    is_shotgun = False
else:
    # Check for shotgun indicators
    is_shotgun = any(
        kw in library_strategy or kw in assay_type
        for kw in ['wgs', 'metagenomic', 'shotgun', 'whole genome']
    )
""",
            imports=[],
            parameters={"strict": strict},
            parameter_schema={
                "strict": {
                    "type": "boolean",
                    "description": "Require exact match vs substring match",
                    "default": False,
                }
            },
            input_entities=[{"type": "sample_metadata", "name": "metadata"}],
            output_entities=[{"type": "validated_sample", "name": "filtered_metadata"}],
        )

        return (metadata if is_valid else {}), stats, ir

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

    def validate_sample_type(
        self,
        metadata: Dict[str, Any],
        allowed_sample_types: Optional[List[str]] = None,
        strict: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Validate sample type using keyword matching on isolation_source and related fields.

        Checks multiple metadata fields (isolation_source, body_site, sample_type, env_material)
        for keywords matching the allowed sample types. Uses OR logic across fields.

        **New in v1.0**: Sample types refined to 4 granular categories to prevent mixing
        biologically distinct samples:
        - "fecal_stool": Passed stool (distal colon)
        - "gut_luminal_content": Intestinal lumen content (not passed)
        - "gut_mucosal_biopsy": Tissue-associated microbiome
        - "gut_lavage": Bowel preparation (potential artifacts)

        **Backward Compatibility**: Legacy types "fecal", "stool", "luminal", "biopsy" are
        aliased to new categories with deprecation warnings. Type "gut" is NOT aliased
        (too ambiguous - raises ValueError).

        Args:
            metadata: Sample metadata dictionary with SRA/BioSample fields
            allowed_sample_types: List of allowed sample types (default: ["fecal_stool"])
                New types: "fecal_stool", "gut_luminal_content", "gut_mucosal_biopsy",
                           "gut_lavage", "oral", "skin"
                Deprecated: "fecal" → "fecal_stool", "stool" → "fecal_stool",
                           "luminal" → "gut_luminal_content", "biopsy" → "gut_mucosal_biopsy"
                Rejected: "gut" (ambiguous - specify explicit category)
            strict: If True, require exact keyword match. If False, allow substring match (default: False)

        Returns:
            Tuple of:
            - filtered_metadata: Original metadata if valid, empty dict if invalid
            - stats: Validation summary with match details
            - ir: AnalysisStep for provenance tracking

        Raises:
            ValueError: If "gut" is specified in allowed_sample_types (too ambiguous)

        Example:
            >>> service = MicrobiomeFilteringService()
            >>> # New explicit syntax (recommended)
            >>> metadata = {"isolation_source": "human feces"}
            >>> result, stats, ir = service.validate_sample_type(metadata, ["fecal_stool"])
            >>> stats["is_valid"]
            True
            >>> stats["matched_sample_type"]
            'fecal_stool'

            >>> # Legacy syntax with warning
            >>> result, stats, ir = service.validate_sample_type(metadata, ["fecal"])
            >>> # Logs: ⚠️ DEPRECATED: 'fecal' is deprecated. Use 'fecal_stool' instead.
            >>> stats["matched_sample_type"]
            'fecal_stool'

            >>> # Ambiguous syntax (error)
            >>> result, stats, ir = service.validate_sample_type(metadata, ["gut"])
            >>> # Raises ValueError: Sample type 'gut' is too ambiguous...
        """
        if allowed_sample_types is None:
            allowed_sample_types = ["fecal_stool"]

        # Handle backward compatibility aliases
        if allowed_sample_types:
            resolved_types = []
            for sample_type in allowed_sample_types:
                if sample_type in self.SAMPLE_TYPE_ALIASES:
                    resolved = self.SAMPLE_TYPE_ALIASES[sample_type]
                    logger.warning(
                        f"⚠️ DEPRECATED: '{sample_type}' is deprecated. "
                        f"Use '{resolved}' instead. "
                        f"This alias will be removed in v2.0."
                    )
                    resolved_types.append(resolved)
                elif sample_type == "gut":
                    # "gut" is too ambiguous - require explicit choice
                    raise ValueError(
                        f"Sample type 'gut' is too ambiguous. "
                        f"Please specify: 'gut_luminal_content' or 'gut_mucosal_biopsy'"
                    )
                else:
                    resolved_types.append(sample_type)

            allowed_sample_types = resolved_types

        result = self._check_sample_type(metadata, allowed_sample_types, strict)

        # Filter metadata based on validation
        filtered_metadata = metadata if result.is_valid else {}

        # Build stats
        stats = {
            "is_valid": result.is_valid,
            "reason": result.reason,
            "matched_sample_type": result.matched_value if result.is_valid else None,
            "matched_field": result.matched_field,
            "allowed_sample_types": allowed_sample_types,
            "strict_mode": strict,
        }

        # Generate IR
        ir = self._create_sample_type_ir(metadata, allowed_sample_types, strict, result)

        return filtered_metadata, stats, ir

    def validate_amplicon_region(
        self,
        metadata: Dict[str, Any],
        allowed_regions: Optional[List[str]] = None,
        strict: bool = True,
        confidence_threshold: float = 0.7,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Validate and filter samples by 16S amplicon region.

        Prevents mixing different variable regions (V4, V3-V4, full-length) which
        have different taxonomic resolution and create systematic bias in diversity
        estimates.

        Args:
            metadata: Sample metadata dict containing amplicon info
            allowed_regions: List of acceptable regions (e.g., ["V4"]).
                If None, detection is performed but no filtering applied.
                If specified, samples NOT in allowed_regions are rejected.
            strict: If True, require confident detection (>= confidence_threshold).
                If False, accept lower-confidence detections with warnings.
            confidence_threshold: Minimum confidence (0.0-1.0) for detection.
                Default 0.7 (medium-high confidence required).

        Returns:
            Tuple of (filtered_metadata, stats_dict, ir):
            - filtered_metadata: Original dict if valid, empty dict {} if invalid
            - stats_dict: {
                "is_valid": bool,
                "reason": str,
                "detected_region": Optional[str],
                "confidence": Optional[float],
                "allowed_regions": Optional[List[str]],
                "detection_source": str  # Which field was used
              }
            - ir: AnalysisStep for provenance tracking

        Examples:
            # Detect region (no filtering)
            result, stats, ir = service.validate_amplicon_region(
                metadata={"target_subfragment": "V4"},
                allowed_regions=None
            )
            # stats = {"is_valid": True, "detected_region": "V4", "confidence": 0.9}

            # Validate against required region
            result, stats, ir = service.validate_amplicon_region(
                metadata={"target_subfragment": "V4"},
                allowed_regions=["V4"]
            )
            # stats = {"is_valid": True, ...}

            # Reject wrong region
            result, stats, ir = service.validate_amplicon_region(
                metadata={"target_subfragment": "V3-V4"},
                allowed_regions=["V4"]
            )
            # stats = {"is_valid": False, "reason": "Region V3-V4 not in allowed list [V4]"}
        """
        # Step 1: Try detection
        detection_result = self._detect_region_from_metadata(metadata)

        if detection_result is None:
            # Could not detect region
            if allowed_regions is not None and strict:
                # Required but not found - fail
                return (
                    {},
                    {
                        "is_valid": False,
                        "reason": "Amplicon region not detected in metadata",
                        "detected_region": None,
                        "confidence": 0.0,
                        "allowed_regions": allowed_regions,
                        "detection_source": "none",
                    },
                    self._create_amplicon_validation_ir(
                        allowed_regions=allowed_regions,
                        strict=strict,
                        confidence_threshold=confidence_threshold,
                        detected_region=None,
                        confidence=0.0,
                        is_valid=False,
                    ),
                )
            else:
                # Not required or permissive mode - pass with warning
                return (
                    metadata,
                    {
                        "is_valid": True,
                        "reason": "Region not detected (permissive mode)",
                        "detected_region": None,
                        "confidence": 0.0,
                        "allowed_regions": allowed_regions,
                        "detection_source": "none",
                    },
                    self._create_amplicon_validation_ir(
                        allowed_regions=allowed_regions,
                        strict=strict,
                        confidence_threshold=confidence_threshold,
                        detected_region=None,
                        confidence=0.0,
                        is_valid=True,
                    ),
                )

        detected_region, confidence = detection_result

        # Step 2: Check confidence threshold
        if strict and confidence < confidence_threshold:
            return (
                {},
                {
                    "is_valid": False,
                    "reason": f"Low confidence detection: {confidence:.2f} < {confidence_threshold}",
                    "detected_region": detected_region,
                    "confidence": confidence,
                    "allowed_regions": allowed_regions,
                    "detection_source": "low_confidence",
                },
                self._create_amplicon_validation_ir(
                    allowed_regions=allowed_regions,
                    strict=strict,
                    confidence_threshold=confidence_threshold,
                    detected_region=detected_region,
                    confidence=confidence,
                    is_valid=False,
                ),
            )

        # Step 3: Validate against allowed regions (if specified)
        if allowed_regions is not None:
            if detected_region not in allowed_regions:
                return (
                    {},
                    {
                        "is_valid": False,
                        "reason": f"Region {detected_region} not in allowed list {allowed_regions}",
                        "detected_region": detected_region,
                        "confidence": confidence,
                        "allowed_regions": allowed_regions,
                        "detection_source": "region_mismatch",
                    },
                    self._create_amplicon_validation_ir(
                        allowed_regions=allowed_regions,
                        strict=strict,
                        confidence_threshold=confidence_threshold,
                        detected_region=detected_region,
                        confidence=confidence,
                        is_valid=False,
                    ),
                )

        # Step 4: Valid - pass sample through
        return (
            metadata,
            {
                "is_valid": True,
                "reason": f"Valid region: {detected_region} (confidence: {confidence:.2f})",
                "detected_region": detected_region,
                "confidence": confidence,
                "allowed_regions": allowed_regions,
                "detection_source": "detected",
            },
            self._create_amplicon_validation_ir(
                allowed_regions=allowed_regions,
                strict=strict,
                confidence_threshold=confidence_threshold,
                detected_region=detected_region,
                confidence=confidence,
                is_valid=True,
            ),
        )

    # -------------------------------------------------------------------------
    # Internal Validation Logic
    # -------------------------------------------------------------------------

    def _check_sample_type(
        self,
        metadata: Dict[str, Any],
        allowed_sample_types: List[str],
        strict: bool,
    ) -> ValidationResult:
        """
        Check if metadata contains allowed sample type indicators.

        Searches multiple fields for sample type keywords using OR logic.

        Args:
            metadata: Metadata to check
            allowed_sample_types: List of allowed sample types (e.g., ["fecal", "gut"])
            strict: Use strict (exact) matching

        Returns:
            ValidationResult with match details
        """
        # Fields to check for sample type information (priority order)
        sample_type_fields = [
            "isolation_source",
            "body_site",
            "sample_type",
            "env_material",
            "tissue",
            "body_product",
            "body_habitat",
            "env_biome",
        ]

        for field_name in sample_type_fields:
            if field_name not in metadata:
                continue

            value = self._normalize_field(metadata[field_name])
            if not value:
                continue

            # Check each allowed sample type
            for sample_type in allowed_sample_types:
                keywords = self.SAMPLE_TYPE_KEYWORDS.get(sample_type, [])
                if not keywords:
                    logger.warning(f"Unknown sample type '{sample_type}', skipping")
                    continue

                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if strict:
                        # Exact match
                        if value == keyword_lower:
                            return ValidationResult(
                                is_valid=True,
                                reason=f"Sample type '{sample_type}' matched in {field_name}",
                                matched_field=field_name,
                                matched_value=sample_type,
                                confidence=100.0,
                            )
                    else:
                        # Substring match
                        if keyword_lower in value:
                            return ValidationResult(
                                is_valid=True,
                                reason=f"Sample type '{sample_type}' matched in {field_name}",
                                matched_field=field_name,
                                matched_value=sample_type,
                                confidence=90.0,  # Substring match = slightly lower confidence
                            )

        return ValidationResult(
            is_valid=False,
            reason=f"Sample type not in allowed list: {allowed_sample_types}",
        )

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

    def _detect_region_from_metadata(
        self, metadata: Dict[str, Any]
    ) -> Optional[Tuple[str, float]]:
        """
        Detect amplicon region from SRA metadata fields.

        Checks: target_subfragment, pcr_primers, target_gene, amplicon_region

        Returns:
            Tuple of (region, confidence) or None if not detected
            Confidence: 1.0 = explicit field, 0.8 = regex match, 0.6 = inferred
        """
        # Priority 1: Explicit amplicon_region field (if exists)
        if "amplicon_region" in metadata and metadata["amplicon_region"]:
            region = self._normalize_region(str(metadata["amplicon_region"]))
            if region in self.STANDARD_REGIONS:
                return (region, 1.0)  # High confidence

        # Priority 2: target_subfragment field
        if "target_subfragment" in metadata and metadata["target_subfragment"]:
            value = str(metadata["target_subfragment"]).strip()
            for pattern in self.V_REGION_PATTERNS:
                match = re.search(pattern, value, re.IGNORECASE)
                if match:
                    region = self._normalize_region(match.group(0))
                    if region in self.STANDARD_REGIONS:
                        return (region, 0.9)  # High confidence from structured field

        # Priority 3: pcr_primers field
        if "pcr_primers" in metadata and metadata["pcr_primers"]:
            result = self._detect_region_from_primers(metadata["pcr_primers"])
            if result:
                return result  # Returns (region, confidence)

        # Priority 4: target_gene field (may contain region info)
        if "target_gene" in metadata and metadata["target_gene"]:
            value = str(metadata["target_gene"])
            for pattern in self.V_REGION_PATTERNS:
                match = re.search(pattern, value, re.IGNORECASE)
                if match:
                    region = self._normalize_region(match.group(0))
                    if region in self.STANDARD_REGIONS:
                        return (region, 0.7)  # Medium confidence

        return None  # Could not detect

    def _detect_region_from_primers(
        self, primer_text: str
    ) -> Optional[Tuple[str, float]]:
        """
        Detect amplicon region from primer names/sequences.

        Args:
            primer_text: String containing primer names (e.g., "515F/806R")

        Returns:
            Tuple of (region, confidence) or None
        """
        if not primer_text:
            return None

        primer_text = str(primer_text).upper()

        # Try exact primer pair matching
        for (fwd, rev), region in self.PRIMER_REGION_MAP.items():
            fwd_upper = fwd.upper()
            rev_upper = rev.upper()

            # Match patterns: "515F/806R", "515F-806R", "515F and 806R", "515F, 806R"
            if fwd_upper in primer_text and rev_upper in primer_text:
                return (region, 0.85)  # High confidence from primer mapping

        # Fallback: Extract region from text like "V4 region primers"
        for pattern in self.V_REGION_PATTERNS:
            match = re.search(pattern, primer_text, re.IGNORECASE)
            if match:
                region = self._normalize_region(match.group(0))
                if region in self.STANDARD_REGIONS:
                    return (region, 0.6)  # Medium-low confidence

        return None

    def _normalize_region(self, region_str: str) -> str:
        """
        Normalize region string to standard format.

        Examples:
            "V 4" → "V4"
            "v3-v4" → "V3-V4"
            "V3 - V4" → "V3-V4"
            "full length 16S" → "full-length"
        """
        region_str = str(region_str).strip()

        # Handle full-length
        if re.search(r'full[- ]?length', region_str, re.IGNORECASE):
            return "full-length"

        # Extract V regions with or without spans
        match = re.search(r'V\s*(\d+)(?:\s*-\s*V\s*(\d+))?', region_str, re.IGNORECASE)
        if match:
            v1 = match.group(1)
            v2 = match.group(2)

            if v2:
                return f"V{v1}-V{v2}"  # V3-V4 format
            else:
                return f"V{v1}"  # V4 format

        return region_str  # Return as-is if can't normalize

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

    def _create_sample_type_ir(
        self,
        metadata: Dict[str, Any],
        allowed_sample_types: List[str],
        strict: bool,
        result: ValidationResult,
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for sample type validation.

        Args:
            metadata: Input metadata
            allowed_sample_types: Allowed sample types list
            strict: Strict mode flag
            result: Validation result

        Returns:
            AnalysisStep for provenance tracking
        """
        code_template = """
# Sample Type Validation
metadata = {{ metadata }}
allowed_sample_types = {{ allowed_sample_types }}
strict = {{ strict }}

# Sample type keywords
SAMPLE_TYPE_KEYWORDS = {
    "fecal": ["fecal", "feces", "stool", "faecal", "faeces", "gut content"],
    "gut": ["gut", "intestine", "colon", "biopsy", "tissue", "mucosa"],
    "oral": ["oral", "saliva", "mouth", "dental", "plaque"],
    "skin": ["skin", "dermal", "cutaneous", "epidermis"],
}

# Fields to check (priority order)
sample_type_fields = ["isolation_source", "body_site", "sample_type", "env_material", "tissue"]

is_valid = False
matched_sample_type = None
matched_field = None

for field_name in sample_type_fields:
    if field_name not in metadata:
        continue
    value = str(metadata[field_name]).lower().strip()
    if not value:
        continue

    for sample_type in allowed_sample_types:
        keywords = SAMPLE_TYPE_KEYWORDS.get(sample_type, [])
        for keyword in keywords:
            if strict and value == keyword.lower():
                is_valid = True
                matched_sample_type = sample_type
                matched_field = field_name
                break
            elif not strict and keyword.lower() in value:
                is_valid = True
                matched_sample_type = sample_type
                matched_field = field_name
                break
        if is_valid:
            break
    if is_valid:
        break

print(f"Validation result: {is_valid}")
if matched_sample_type:
    print(f"Matched sample type: {matched_sample_type} (field: {matched_field})")
"""

        return AnalysisStep(
            operation="microbiome.filtering.validate_sample_type",
            tool_name="MicrobiomeFilteringService.validate_sample_type",
            description=f"Validate sample type (allowed: {allowed_sample_types}, strict={strict})",
            library="lobster.services.metadata.microbiome_filtering_service",
            code_template=code_template.strip(),
            imports=[],
            parameters={
                "metadata": metadata,
                "allowed_sample_types": allowed_sample_types,
                "strict": strict,
            },
            parameter_schema={
                "metadata": {"type": "dict", "description": "Metadata to validate"},
                "allowed_sample_types": {
                    "type": "list",
                    "default": ["fecal", "gut"],
                    "description": "Allowed sample types (fecal, gut, oral, skin)",
                },
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
                    "matched_sample_type": result.matched_value,
                    "confidence": result.confidence,
                }
            ],
        )

    def _create_amplicon_validation_ir(
        self,
        allowed_regions: Optional[List[str]],
        strict: bool,
        confidence_threshold: float,
        detected_region: Optional[str],
        confidence: Optional[float],
        is_valid: bool,
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for amplicon region validation.

        Args:
            allowed_regions: List of allowed regions (or None)
            strict: Strict mode flag
            confidence_threshold: Minimum confidence required
            detected_region: Detected region name
            confidence: Detection confidence score
            is_valid: Validation result

        Returns:
            AnalysisStep for provenance tracking
        """
        code_template = """
# Amplicon Region Validation
metadata = {{ metadata }}
allowed_regions = {{ allowed_regions }}
strict = {{ strict }}
confidence_threshold = {{ confidence_threshold }}

# Detection patterns for V regions
V_REGION_PATTERNS = [
    r'\\b(V\\d)(?:\\s*-\\s*V\\d)?\\b',           # V4, V3-V4, V1-V9
    r'\\bvariable\\s+region\\s+(\\d+)\\b',      # "variable region 4"
    r'\\b(V\\d+)\\s*-\\s*(V\\d+)\\b',            # V3 - V4 (with spaces)
    r'\\bfull[- ]?length\\s+16S\\b',            # "full length 16S"
    r'\\b16S\\s+(V\\d+(?:-V\\d+)?)\\b',         # "16S V4", "16S V3-V4"
]

# Primer pair to region mappings
PRIMER_REGION_MAP = {
    ("515F", "806R"): "V4",
    ("341F", "785R"): "V3-V4",
    ("27F", "1492R"): "full-length",
}

# Try detection from metadata fields
detected_region = None
confidence = 0.0

# Priority 1: target_subfragment field
if "target_subfragment" in metadata:
    value = str(metadata["target_subfragment"]).strip()
    for pattern in V_REGION_PATTERNS:
        import re
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            detected_region = match.group(0).upper()
            confidence = 0.9
            break

# Priority 2: pcr_primers field
if not detected_region and "pcr_primers" in metadata:
    primer_text = str(metadata["pcr_primers"]).upper()
    for (fwd, rev), region in PRIMER_REGION_MAP.items():
        if fwd in primer_text and rev in primer_text:
            detected_region = region
            confidence = 0.85
            break

# Validate against allowed regions
is_valid = False
if detected_region is None:
    is_valid = not strict or allowed_regions is None
elif confidence < confidence_threshold:
    is_valid = not strict
elif allowed_regions is not None:
    is_valid = detected_region in allowed_regions
else:
    is_valid = True

print(f"Detected region: {detected_region} (confidence: {confidence:.2f})")
print(f"Validation result: {is_valid}")
"""

        return AnalysisStep(
            operation="microbiome.filtering.validate_amplicon_region",
            tool_name="MicrobiomeFilteringService.validate_amplicon_region",
            description=f"Validate 16S amplicon region: {detected_region or 'unknown'} (strict={strict})",
            library="lobster.services.metadata.microbiome_filtering_service",
            code_template=code_template.strip(),
            imports=["import re"],
            parameters={
                "allowed_regions": allowed_regions,
                "strict": strict,
                "confidence_threshold": confidence_threshold,
                "detected_region": detected_region,
                "confidence": confidence,
            },
            parameter_schema={
                "allowed_regions": {
                    "type": "List[str]",
                    "description": "List of acceptable regions (e.g., ['V4'])",
                    "required": False,
                },
                "strict": {
                    "type": "bool",
                    "default": True,
                    "description": "Require confident detection",
                },
                "confidence_threshold": {
                    "type": "float",
                    "default": 0.7,
                    "description": "Minimum confidence (0.0-1.0) for detection",
                },
            },
            input_entities=[{"type": "metadata", "name": "input_metadata"}],
            output_entities=[
                {
                    "type": "validation_result",
                    "name": "filtered_metadata",
                    "is_valid": is_valid,
                    "detected_region": detected_region,
                    "confidence": confidence,
                }
            ],
            execution_context={
                "is_valid": is_valid,
                "detected_region": detected_region,
                "confidence": confidence,
            },
        )
