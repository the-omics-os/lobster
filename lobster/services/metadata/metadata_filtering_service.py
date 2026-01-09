"""
Metadata filtering service with semantic understanding.

Parses natural language filter criteria and applies sequential filters:
- Disease extraction and standardization
- Sequencing method (16S, shotgun, ITS)
- Amplicon region (V4, V3-V4, full-length)
- Host organism (fuzzy matching)
- Sample type (fecal_stool, gut_luminal_content, gut_mucosal_biopsy, oral, skin)

Supported filter criteria syntax:
    "16S V4 human fecal_stool CRC"  (explicit, recommended)
    "16S human fecal CRC"            (legacy, uses aliases with warnings)
    "shotgun mouse gut_luminal_content"
    "16S V3-V4 human gut_mucosal_biopsy UC CD"
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from lobster.core.analysis_ir import AnalysisStep

logger = logging.getLogger(__name__)


# Disease source columns for fallback chain (Bug 3 fix - DataBioMix)
# Priority order: most specific → least specific
DISEASE_SOURCE_COLUMNS = [
    "disease",                    # Internal field (if already populated)
    "clinical condition",         # 25.0% fill (DataBioMix EDA)
    "ibd",                        # 24.3% fill
    "ibd_diagnosis_refined",      # 22.5% fill
    "diagnosis",                  # 15.2% fill
    "host_disease",               # 12.1% fill
    "host_disease_stat",          # MIMARKS standard
    "disease_stat",               # Variant spelling
    "condition",                  # Generic condition field
    "host_phenotype",             # Phenotype descriptions
]


def extract_disease_with_fallback(
    df: pd.DataFrame, study_context: Optional[Dict] = None
) -> Optional[str]:
    """
    Extract disease column with fallback chain across multiple source fields.

    Populates 'host_disease_stat' from first available source column.
    Designed for DataBioMix microbiome harmonization (Bug 3 fix).

    Args:
        df: DataFrame with sample metadata (modified in-place)
        study_context: Optional publication metadata (unused but required by interface)

    Returns:
        Column name to use for disease standardization ('host_disease_stat')
        or None if no disease data found

    Example:
        Sample 1: has "clinical condition" → copies to "host_disease_stat"
        Sample 2: missing "clinical condition", has "ibd" → copies "ibd"
        Sample 3: no disease fields → "host_disease_stat" remains None
    """
    # Initialize target column if missing
    if "host_disease_stat" not in df.columns:
        df["host_disease_stat"] = None

    # Apply fallback chain: fill from first available source
    for source_col in DISEASE_SOURCE_COLUMNS:
        if source_col in df.columns and source_col != "host_disease_stat":
            # Fill missing values in host_disease_stat from this source
            mask = df["host_disease_stat"].isna() & df[source_col].notna()
            if mask.any():
                df.loc[mask, "host_disease_stat"] = df.loc[mask, source_col]
                logger.debug(
                    f"Disease fallback: populated {mask.sum()} samples from '{source_col}'"
                )

    # Check if we populated any disease data
    disease_count = df["host_disease_stat"].notna().sum()
    if disease_count > 0:
        logger.info(
            f"Disease extraction: {disease_count}/{len(df)} samples "
            f"({disease_count/len(df)*100:.1f}%) have disease metadata"
        )
        return "host_disease_stat"

    logger.debug("Disease extraction: no disease fields found in metadata")
    return None


class MetadataFilteringService:
    """
    Service for parsing and applying metadata filters.

    Uses dependency injection for microbiome and disease services,
    enabling testing and flexible deployment (premium features optional).

    Example:
        >>> service = MetadataFilteringService(
        ...     microbiome_service=MicrobiomeFilteringService(),
        ...     disease_service=DiseaseStandardizationService()
        ... )
        >>> parsed = service.parse_criteria("16S human fecal CRC")
        >>> filtered, stats, ir = service.apply_filters(samples, parsed)
    """

    def __init__(
        self,
        microbiome_service=None,
        disease_service=None,
        disease_extractor: Optional[Callable[[pd.DataFrame, Optional[Dict]], Optional[str]]] = None,
    ):
        """
        Initialize filtering service with optional dependencies.

        Args:
            microbiome_service: MicrobiomeFilteringService instance (optional)
            disease_service: DiseaseStandardizationService instance (optional)
            disease_extractor: Function to extract disease column from DataFrame
                               Signature: (df, study_context) -> Optional[column_name]
        """
        self.microbiome_service = microbiome_service
        self.disease_service = disease_service
        self.disease_extractor = disease_extractor

    @property
    def is_available(self) -> bool:
        """Check if filtering services are available."""
        return self.microbiome_service is not None

    def parse_criteria(self, criteria: str) -> Dict[str, Any]:
        """
        Parse natural language filter criteria into structured format.

        Supports combined filters with OR logic for sequencing types
        (e.g., "16S shotgun" includes samples matching EITHER).

        Args:
            criteria: Natural language string (e.g., "16S shotgun human fecal CRC")

        Returns:
            Dictionary with parsed criteria:
            {
                "check_16s": bool,
                "check_shotgun": bool,
                "host_organisms": List[str],
                "sample_types": List[str],
                "standardize_disease": bool,
                "amplicon_region": Optional[str]
            }
        """
        criteria_lower = criteria.lower()

        # Check for 16S amplicon
        check_16s = "16s" in criteria_lower or "amplicon" in criteria_lower

        # Check for amplicon region specification
        amplicon_region = None
        region_match = re.search(
            r'\b(V\d+(?:-V\d+)?|full-length)\b',
            criteria_lower,
            re.IGNORECASE
        )
        if region_match:
            amplicon_region = region_match.group(1)
            # Normalize to standard format
            if amplicon_region.lower() == "full-length":
                amplicon_region = "full-length"
            else:
                # Ensure proper V capitalization (V4, V3-V4)
                amplicon_region = amplicon_region.upper()

        # Check for shotgun metagenomic
        check_shotgun = (
            "shotgun" in criteria_lower
            or "wgs" in criteria_lower
            or "metagenomic" in criteria_lower
        )

        # Check for host organisms
        host_organisms = []
        if "human" in criteria_lower or "homo sapiens" in criteria_lower:
            host_organisms.append("Human")
        if "mouse" in criteria_lower or "mus musculus" in criteria_lower:
            host_organisms.append("Mouse")

        # Check for sample types (supports new + legacy syntax)
        sample_types = []

        # New explicit categories (preferred)
        if "fecal_stool" in criteria_lower or "fecal stool" in criteria_lower:
            sample_types.append("fecal_stool")
        elif "fecal" in criteria_lower or "stool" in criteria_lower or "feces" in criteria_lower:
            # Legacy alias (with deprecation warning in service)
            sample_types.append("fecal")  # Will be resolved to fecal_stool

        if "gut_luminal_content" in criteria_lower or "gut luminal" in criteria_lower:
            sample_types.append("gut_luminal_content")
        elif "luminal" in criteria_lower:
            sample_types.append("luminal")  # Alias

        if "gut_mucosal_biopsy" in criteria_lower or "gut mucosal" in criteria_lower or "mucosal biopsy" in criteria_lower:
            sample_types.append("gut_mucosal_biopsy")
        elif "biopsy" in criteria_lower and "gut" not in sample_types:
            sample_types.append("biopsy")  # Alias

        if "gut_lavage" in criteria_lower or "gut lavage" in criteria_lower:
            sample_types.append("gut_lavage")
        elif "lavage" in criteria_lower:
            sample_types.append("gut_lavage")  # Direct mapping

        # "gut" alone is NOT parsed - too ambiguous
        # If user wants gut samples, they must specify luminal OR biopsy OR lavage

        # Keep existing (unchanged)
        if "oral" in criteria_lower or "saliva" in criteria_lower:
            sample_types.append("oral")
        if "skin" in criteria_lower:
            sample_types.append("skin")

        # Always enable disease extraction for microbiome studies (Bug 3 fix - DataBioMix)
        # Disease metadata is valuable even if not explicitly requested in filter criteria
        # Fallback chain will populate from multiple source columns (clinical condition, ibd, etc.)
        standardize_disease = True

        return {
            "check_16s": check_16s,
            "check_shotgun": check_shotgun,
            "host_organisms": host_organisms,
            "sample_types": sample_types,
            "standardize_disease": standardize_disease,
            "amplicon_region": amplicon_region,  # NEW: Amplicon region filter (V4, V3-V4, etc.)
            "min_disease_coverage": 0.5,  # Default: 50% threshold for disease-based meta-analysis
            "strict_disease_validation": True,  # Default: fail hard on insufficient coverage
        }

    def apply_filters(
        self,
        samples: List[Dict[str, Any]],
        parsed_criteria: Dict[str, Any],
        study_context: Optional[Dict] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], AnalysisStep]:
        """
        Apply parsed filter criteria to sample metadata.

        Includes disease extraction + standardization if disease terms in criteria.
        Uses OR logic for sequencing types (16S OR shotgun), AND for other filters.

        Args:
            samples: List of sample dictionaries
            parsed_criteria: Output from parse_criteria()
            study_context: Optional publication metadata for disease inference

        Returns:
            Tuple of (filtered_samples, stats_dict, analysis_step_ir)
        """
        if not self.is_available:
            logger.warning(
                "Microbiome services not available, returning unfiltered samples"
            )
            return samples, {"skipped": True, "reason": "services_unavailable"}, self._create_ir(
                parsed_criteria, len(samples), len(samples), skipped=True
            )

        filtered = samples.copy()
        original_count = len(samples)
        filter_steps = []

        # Apply disease extraction + standardization if requested
        if parsed_criteria.get("standardize_disease") and filtered and self.disease_service:
            filtered, disease_stats = self._apply_disease_filter(
                filtered,
                study_context,
                min_disease_coverage=parsed_criteria.get("min_disease_coverage", 0.5),
                strict_disease_validation=parsed_criteria.get("strict_disease_validation", True)
            )
            if disease_stats.get("applied"):
                filter_steps.append(f"disease_standardization({disease_stats.get('rate', 0):.1f}%)")

        # Apply sequencing method filters (16S OR shotgun - use OR logic)
        if parsed_criteria.get("check_16s") or parsed_criteria.get("check_shotgun"):
            filtered, seq_stats = self._apply_sequencing_filter(
                filtered,
                check_16s=parsed_criteria.get("check_16s", False),
                check_shotgun=parsed_criteria.get("check_shotgun", False),
            )
            filter_steps.append(f"sequencing({seq_stats['retained']}/{seq_stats['input']})")

        # Apply amplicon region validation (if 16S specified)
        if parsed_criteria.get("check_16s") and parsed_criteria.get("amplicon_region"):
            samples_before = len(filtered)
            amplicon_stats = {"applied": False, "retained": 0, "input": samples_before}

            allowed_regions = [parsed_criteria["amplicon_region"]]  # Single region required
            failed_samples = []
            passed_samples = []

            for sample in filtered:
                result_dict, validation_stats, _ = self.microbiome_service.validate_amplicon_region(
                    sample,
                    allowed_regions=allowed_regions,
                    strict=True,  # Always strict for region validation
                    confidence_threshold=0.7
                )

                if validation_stats["is_valid"]:
                    # Add detected region to sample metadata
                    sample["amplicon_region_detected"] = validation_stats.get("detected_region")
                    sample["amplicon_region_confidence"] = validation_stats.get("confidence", 0.0)
                    passed_samples.append(sample)
                else:
                    failed_samples.append(sample)

            filtered = passed_samples
            amplicon_stats["applied"] = True
            amplicon_stats["retained"] = len(passed_samples)
            amplicon_stats["rejected"] = len(failed_samples)

            filter_steps.append(
                f"amplicon({amplicon_stats['retained']}/{amplicon_stats['input']})"
            )

            logger.debug(
                f"After amplicon region filter ({parsed_criteria['amplicon_region']}): "
                f"{amplicon_stats['retained']}/{amplicon_stats['input']} samples retained"
            )

        # Apply host organism filter
        if parsed_criteria.get("host_organisms"):
            filtered, host_stats = self._apply_host_filter(
                filtered, parsed_criteria["host_organisms"]
            )
            filter_steps.append(f"host({host_stats['retained']}/{host_stats['input']})")

        # Apply sample type filter (fecal, gut, oral, skin)
        if parsed_criteria.get("sample_types"):
            filtered, sample_type_stats = self._apply_sample_type_filter(
                filtered, parsed_criteria["sample_types"]
            )
            filter_steps.append(
                f"sample_type({sample_type_stats['retained']}/{sample_type_stats['input']})"
            )

        # Build stats
        stats = {
            "samples_original": original_count,
            "samples_retained": len(filtered),
            "retention_rate": (len(filtered) / original_count * 100) if original_count else 0.0,
            "filters_applied": filter_steps,
            "criteria": parsed_criteria,
        }

        ir = self._create_ir(parsed_criteria, original_count, len(filtered))

        return filtered, stats, ir

    def _apply_disease_filter(
        self,
        samples: List[Dict[str, Any]],
        study_context: Optional[Dict] = None,
        min_disease_coverage: float = 0.5,
        strict_disease_validation: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Apply disease extraction and standardization with validation.

        Args:
            samples: List of sample dictionaries
            study_context: Optional publication metadata for disease inference
            min_disease_coverage: Minimum disease coverage rate (0.0-1.0) required for analysis
            strict_disease_validation: If True, raise ValueError on insufficient coverage

        Returns:
            Tuple of (filtered_samples, stats_dict)

        Raises:
            ValueError: If strict_disease_validation=True and coverage < min_disease_coverage
        """
        if not samples or not self.disease_service:
            return samples, {"applied": False}

        df = pd.DataFrame(samples)

        # Use injected disease extractor if available
        disease_col = None
        if self.disease_extractor:
            disease_col = self.disease_extractor(df, study_context)

        if disease_col:
            standardized_df, stats, _ = self.disease_service.standardize_disease_terms(
                df, disease_column=disease_col
            )

            # Validate disease coverage after standardization
            coverage_rate = stats.get("standardization_rate", 0) / 100.0
            total_samples = stats.get("total_samples", len(samples))
            mapping_stats = stats.get("mapping_stats", {})
            mapped_samples = total_samples - mapping_stats.get("unmapped", total_samples)

            # Check if coverage meets minimum threshold
            if min_disease_coverage > 0.0 and coverage_rate < min_disease_coverage:
                error_msg = (
                    f"❌ Insufficient disease data: {coverage_rate*100:.1f}% coverage "
                    f"({mapped_samples}/{total_samples} samples) is below the required "
                    f"{min_disease_coverage*100:.0f}% threshold for disease-based meta-analysis.\n\n"
                    f"Recommendations:\n"
                    f"  1. Use different datasets with better disease annotations\n"
                    f"  2. Manually enrich samples using publication metadata (execute_custom_code)\n"
                    f"  3. Lower threshold with min_disease_coverage={coverage_rate:.2f} (not recommended)\n"
                    f"  4. Skip disease filtering by omitting disease terms from filter_criteria"
                )

                if strict_disease_validation:
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    logger.warning(error_msg)

            filtered = standardized_df.to_dict("records")
            logger.debug(
                f"Disease extraction + standardization applied: "
                f"{stats.get('standardization_rate', 0):.1f}% mapped"
            )
            return filtered, {"applied": True, "rate": stats.get("standardization_rate", 0)}

        return samples, {"applied": False}

    def _apply_sequencing_filter(
        self,
        samples: List[Dict[str, Any]],
        check_16s: bool,
        check_shotgun: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply sequencing method filter with OR logic."""
        input_count = len(samples)
        sequencing_filtered = []

        for s in samples:
            is_16s = False
            is_shotgun = False

            if check_16s:
                is_16s = self.microbiome_service.validate_16s_amplicon(s, strict=False)[0]

            if check_shotgun:
                is_shotgun = self.microbiome_service.validate_shotgun(s, strict=False)[0]

            # OR logic: include if matches either
            if is_16s or is_shotgun:
                # Tag sample with matched method for transparency
                if is_16s and is_shotgun:
                    s["_matched_sequencing_method"] = "both_16S_and_shotgun"
                elif is_16s:
                    s["_matched_sequencing_method"] = "16S_amplicon"
                elif is_shotgun:
                    s["_matched_sequencing_method"] = "shotgun_metagenomic"
                sequencing_filtered.append(s)

        logger.debug(f"After sequencing filter: {len(sequencing_filtered)} samples retained")

        return sequencing_filtered, {
            "input": input_count,
            "retained": len(sequencing_filtered),
        }

    def _apply_host_filter(
        self,
        samples: List[Dict[str, Any]],
        allowed_hosts: List[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply host organism filter."""
        input_count = len(samples)

        filtered = [
            s
            for s in samples
            if self.microbiome_service.validate_host_organism(
                s, allowed_hosts=allowed_hosts
            )[0]
        ]

        return filtered, {
            "input": input_count,
            "retained": len(filtered),
        }

    def _apply_sample_type_filter(
        self,
        samples: List[Dict[str, Any]],
        allowed_sample_types: List[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Apply sample type filter (fecal, gut, oral, skin).

        Filters samples based on isolation_source and related fields,
        checking for keywords matching the allowed sample types.

        Args:
            samples: List of sample dictionaries
            allowed_sample_types: List of allowed sample types (e.g., ["fecal", "gut"])

        Returns:
            Tuple of (filtered_samples, stats_dict)
        """
        input_count = len(samples)
        sample_type_filtered = []

        for s in samples:
            # Use microbiome_service.validate_sample_type() for consistent validation
            is_valid, stats, _ = self.microbiome_service.validate_sample_type(
                s, allowed_sample_types=allowed_sample_types, strict=False
            )

            if stats.get("is_valid"):
                # Tag sample with matched type for transparency
                matched_type = stats.get("matched_sample_type")
                matched_field = stats.get("matched_field")
                if matched_type:
                    s["_matched_sample_type"] = matched_type
                    s["_matched_sample_type_field"] = matched_field
                sample_type_filtered.append(s)

        logger.debug(
            f"After sample_type filter ({allowed_sample_types}): "
            f"{len(sample_type_filtered)}/{input_count} samples retained"
        )

        return sample_type_filtered, {
            "input": input_count,
            "retained": len(sample_type_filtered),
            "allowed_types": allowed_sample_types,
        }

    def _create_ir(
        self,
        parsed_criteria: Dict[str, Any],
        original_count: int,
        retained_count: int,
        skipped: bool = False,
    ) -> AnalysisStep:
        """Create AnalysisStep IR for provenance tracking."""
        if skipped:
            description = "Metadata filtering skipped (services unavailable)"
            code_template = "# Filtering skipped - microbiome services not available"
        else:
            active_filters = []
            if parsed_criteria.get("check_16s"):
                active_filters.append("16S amplicon")
            if parsed_criteria.get("amplicon_region"):
                active_filters.append(f"region: {parsed_criteria['amplicon_region']}")
            if parsed_criteria.get("check_shotgun"):
                active_filters.append("shotgun/WGS")
            if parsed_criteria.get("host_organisms"):
                active_filters.append(f"host: {parsed_criteria['host_organisms']}")
            if parsed_criteria.get("sample_types"):
                active_filters.append(f"sample_type: {parsed_criteria['sample_types']}")
            if parsed_criteria.get("standardize_disease"):
                active_filters.append("disease standardization")

            description = (
                f"Applied metadata filters ({', '.join(active_filters) or 'none'}): "
                f"{original_count} → {retained_count} samples "
                f"({retained_count/original_count*100:.1f}% retained)"
                if original_count else "No samples to filter"
            )

            code_template = self._generate_code_template(parsed_criteria)

        return AnalysisStep(
            operation="metadata_filtering",
            tool_name="MetadataFilteringService.apply_filters",
            description=description,
            library="lobster.services.metadata.metadata_filtering_service",
            code_template=code_template,
            imports=[
                "from lobster.services.metadata.metadata_filtering_service import MetadataFilteringService",
                "from lobster.services.metadata.microbiome_filtering_service import MicrobiomeFilteringService",
            ],
            parameters={
                "check_16s": parsed_criteria.get("check_16s", False),
                "check_shotgun": parsed_criteria.get("check_shotgun", False),
                "amplicon_region": parsed_criteria.get("amplicon_region"),
                "host_organisms": parsed_criteria.get("host_organisms", []),
                "sample_types": parsed_criteria.get("sample_types", []),
                "standardize_disease": parsed_criteria.get("standardize_disease", False),
            },
            parameter_schema={
                "check_16s": {"type": "boolean", "description": "Filter for 16S amplicon data"},
                "check_shotgun": {"type": "boolean", "description": "Filter for shotgun/WGS data"},
                "amplicon_region": {"type": "string", "description": "Required amplicon region (V4, V3-V4, etc.)"},
                "host_organisms": {"type": "array", "items": {"type": "string"}, "description": "Allowed host organisms"},
                "sample_types": {"type": "array", "items": {"type": "string"}, "description": "Allowed sample types"},
                "standardize_disease": {"type": "boolean", "description": "Apply disease term standardization"},
            },
            input_entities=[{"type": "sample_metadata", "name": "input_samples"}],
            output_entities=[{"type": "filtered_metadata", "name": "filtered_samples"}],
        )

    def _generate_code_template(self, parsed_criteria: Dict[str, Any]) -> str:
        """Generate reproducible code template for notebook export."""
        lines = [
            "# Initialize filtering service",
            "from lobster.services.metadata.metadata_filtering_service import MetadataFilteringService",
            "from lobster.services.metadata.microbiome_filtering_service import MicrobiomeFilteringService",
            "from lobster.services.metadata.disease_standardization_service import DiseaseStandardizationService",
            "",
            "filtering_service = MetadataFilteringService(",
            "    microbiome_service=MicrobiomeFilteringService(),",
            "    disease_service=DiseaseStandardizationService(),",
            ")",
            "",
            "# Define filter criteria",
            f"criteria = {repr(parsed_criteria)}",
            "",
            "# Apply filters",
            "filtered_samples, stats, _ = filtering_service.apply_filters(samples, criteria)",
            "print(f\"Retained {len(filtered_samples)}/{len(samples)} samples ({stats['retention_rate']:.1f}%)\")",
        ]
        return "\n".join(lines)
