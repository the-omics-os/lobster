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

    # Sample type detection keywords (case-insensitive)
    # Maps canonical sample type to keywords found in isolation_source, body_site, etc.
    SAMPLE_TYPE_KEYWORDS = {
        "fecal": [
            "fecal",
            "feces",
            "stool",
            "faecal",
            "faeces",
            "fecal sample",
            "stool sample",
            "human feces",
            "mouse feces",
            "gut content",
            "intestinal content",
            "colon content",
            "cecal content",
            "caecal content",
        ],
        "gut": [
            "gut",
            "intestine",
            "intestinal",
            "colon",
            "colonic",
            "cecum",
            "caecum",
            "ileum",
            "jejunum",
            "duodenum",
            "small intestine",
            "large intestine",
            "gastrointestinal",
            "gi tract",
            "biopsy",
            "tissue",
            "mucosa",
            "mucosal",
            "epithelium",
            "epithelial",
        ],
        "oral": [
            "oral",
            "saliva",
            "mouth",
            "tongue",
            "dental",
            "plaque",
            "gingiva",
            "gingival",
            "buccal",
            "subgingival",
            "supragingival",
        ],
        "skin": [
            "skin",
            "dermal",
            "cutaneous",
            "epidermis",
            "sebaceous",
        ],
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

        Args:
            metadata: Sample metadata dictionary with SRA/BioSample fields
            allowed_sample_types: List of allowed sample types (default: ["fecal", "gut"])
                Supported types: "fecal", "gut", "oral", "skin"
            strict: If True, require exact keyword match. If False, allow substring match (default: False)

        Returns:
            Tuple of:
            - filtered_metadata: Original metadata if valid, empty dict if invalid
            - stats: Validation summary with match details
            - ir: AnalysisStep for provenance tracking

        Example:
            >>> service = MicrobiomeFilteringService()
            >>> metadata = {"isolation_source": "human feces"}
            >>> result, stats, ir = service.validate_sample_type(metadata, ["fecal"])
            >>> stats["is_valid"]
            True
            >>> stats["matched_sample_type"]
            'fecal'
        """
        if allowed_sample_types is None:
            allowed_sample_types = ["fecal", "gut"]

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
