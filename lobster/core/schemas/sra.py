"""
SRA sample metadata schema for validation and harmonization.

This module provides Pydantic schemas for validating SRA sample metadata
extracted from NCBI's SRA Run Selector API. It follows the established
pattern from transcriptomics.py, proteomics.py, and metagenomics.py.

The schema is modality-agnostic and supports all library strategies:
- AMPLICON (16S, ITS, etc.)
- RNA-Seq (bulk, single-cell)
- WGS (whole genome shotgun)
- ChIP-Seq, ATAC-seq, etc.

Used by metadata_assistant agent for:
- Sample extraction validation (prevent malformed data)
- Pre-download validation (ensure URLs exist)
- Cross-database harmonization
"""

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from lobster.core.interfaces.validator import ValidationResult

logger = logging.getLogger(__name__)


class SRASampleSchema(BaseModel):
    """
    Pydantic schema for SRA sample metadata validation.

    This schema validates raw SRA sample records extracted from NCBI's
    SRA Run Selector API. It's modality-agnostic and supports all library
    strategies (AMPLICON, RNA-Seq, WGS, ChIP-Seq, etc.).

    SRA samples contain 71 fields from NCBI. This schema explicitly defines
    critical fields and stores the remaining fields in additional_metadata.

    Attributes:
        run_accession: SRA run accession (SRR*)
        experiment_accession: SRA experiment accession (SRX*)
        sample_accession: SRA sample accession (SRS*)
        study_accession: SRA study accession (SRP*)
        bioproject: BioProject accession (PRJ*)
        biosample: BioSample accession (SAM*)
        library_strategy: Sequencing strategy (AMPLICON, RNA-Seq, etc.)
        library_source: Library source (GENOMIC, TRANSCRIPTOMIC, etc.)
        library_selection: Library selection method (PCR, RANDOM, etc.)
        library_layout: SINGLE or PAIRED
        organism_name: Organism name (e.g., "Homo sapiens")
        organism_taxid: NCBI Taxonomy ID
        instrument: Sequencing instrument
        instrument_model: Instrument model
        public_url: NCBI public download URL
        ncbi_url: NCBI direct download URL
        aws_url: AWS S3 download URL
        gcp_url: GCP download URL
        study_title: Study title (optional)
        experiment_title: Experiment title (optional)
        sample_title: Sample title (optional)
        env_medium: Environmental medium (optional, critical for microbiome)
        env_broad_scale: Environmental broad scale (optional)
        env_local_scale: Environmental local scale (optional)
        collection_date: Sample collection date (optional)
        geo_loc_name: Geographic location (optional)
        total_spots: Total spots (optional)
        total_size: Total size in bytes (optional)
        run_total_spots: Run total spots (optional)
        run_total_bases: Run total bases (optional)
        additional_metadata: All other SRA fields (71 total)

    Examples:
        >>> sample_dict = {
        ...     "run_accession": "SRR21960766",
        ...     "experiment_accession": "SRX17944370",
        ...     "sample_accession": "SRS15461891",
        ...     "study_accession": "SRP403291",
        ...     "bioproject": "PRJNA891765",
        ...     "biosample": "SAMN31357800",
        ...     "library_strategy": "AMPLICON",
        ...     "library_source": "METAGENOMIC",
        ...     "library_selection": "PCR",
        ...     "library_layout": "PAIRED",
        ...     "organism_name": "human metagenome",
        ...     "organism_taxid": "646099",
        ...     "instrument": "Illumina MiSeq",
        ...     "instrument_model": "Illumina MiSeq",
        ...     "public_url": "https://sra-downloadb.be-md.ncbi.nlm.nih.gov/...",
        ...     # ... 50+ additional fields
        ... }
        >>> validated = SRASampleSchema.from_dict(sample_dict)
        >>> validated.has_download_url()
        True
    """

    # Core SRA identifiers (REQUIRED)
    run_accession: str = Field(..., description="SRA run accession (SRR*)")
    experiment_accession: str = Field(..., description="SRA experiment (SRX*)")
    sample_accession: str = Field(..., description="SRA sample (SRS*)")
    study_accession: str = Field(..., description="SRA study (SRP*)")

    # BioProject/BioSample linkage (REQUIRED)
    bioproject: str = Field(..., description="BioProject accession (PRJ*)")
    biosample: str = Field(..., description="BioSample accession (SAM*)")

    # Library metadata (REQUIRED)
    library_strategy: str = Field(..., description="Sequencing strategy")
    library_source: str = Field(..., description="Library source")
    library_selection: str = Field(..., description="Library selection method")
    library_layout: str = Field(..., description="SINGLE or PAIRED")

    # Organism (REQUIRED)
    organism_name: str = Field(..., description="Organism name")
    organism_taxid: str = Field(..., description="NCBI Taxonomy ID")

    # Instrument (REQUIRED)
    instrument: str = Field(..., description="Sequencing instrument")
    instrument_model: str = Field(..., description="Instrument model")

    # Download URLs (at least ONE required - validated separately)
    public_url: Optional[str] = Field(None, description="NCBI public URL")
    ncbi_url: Optional[str] = Field(None, description="NCBI direct URL")
    aws_url: Optional[str] = Field(None, description="AWS S3 URL")
    gcp_url: Optional[str] = Field(None, description="GCP URL")

    # Study metadata (OPTIONAL)
    study_title: Optional[str] = None
    experiment_title: Optional[str] = None
    sample_title: Optional[str] = None

    # Environmental context (OPTIONAL - critical for microbiome)
    env_medium: Optional[str] = Field(None, description="Environmental medium")
    env_broad_scale: Optional[str] = None
    env_local_scale: Optional[str] = None
    collection_date: Optional[str] = None
    geo_loc_name: Optional[str] = None

    # Sequencing metrics (OPTIONAL)
    total_spots: Optional[str] = None
    total_size: Optional[str] = None
    run_total_spots: Optional[str] = None
    run_total_bases: Optional[str] = None

    # Additional fields stored in additional_metadata
    additional_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="All other SRA fields (71 total fields)"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "run_accession": "SRR21960766",
                "experiment_accession": "SRX17944370",
                "sample_accession": "SRS15461891",
                "study_accession": "SRP403291",
                "bioproject": "PRJNA891765",
                "biosample": "SAMN31357800",
                "library_strategy": "AMPLICON",
                "library_source": "METAGENOMIC",
                "library_selection": "PCR",
                "library_layout": "PAIRED",
                "organism_name": "human metagenome",
                "organism_taxid": "646099",
                "instrument": "Illumina MiSeq",
                "instrument_model": "Illumina MiSeq",
                "public_url": "https://sra-downloadb.be-md.ncbi.nlm.nih.gov/...",
                "env_medium": "Stool",
                "collection_date": "2017",
            }
        }

    @field_validator("library_strategy")
    @classmethod
    def validate_library_strategy(cls, v: str) -> str:
        """
        Validate library strategy is recognized.

        Logs a warning for uncommon strategies but allows them.
        """
        # Common strategies - not exhaustive
        known_strategies = {
            "AMPLICON",
            "RNA-Seq",
            "WGS",
            "WXS",
            "ChIP-Seq",
            "ATAC-seq",
            "Bisulfite-Seq",
            "Hi-C",
            "FAIRE-seq",
            "MBD-Seq",
            "MRE-Seq",
            "MeDIP-Seq",
            "DNase-Hypersensitivity",
            "Tn-Seq",
            "VALIDATION",
            "OTHER",
        }
        if v not in known_strategies:
            logger.warning(f"Uncommon library_strategy: '{v}'")
        return v

    @field_validator("library_layout")
    @classmethod
    def validate_library_layout(cls, v: str) -> str:
        """
        Validate library layout is SINGLE or PAIRED.

        Raises:
            ValueError: If layout is not SINGLE or PAIRED
        """
        if v.upper() not in {"SINGLE", "PAIRED"}:
            raise ValueError(f"library_layout must be 'SINGLE' or 'PAIRED', got '{v}'")
        return v.upper()

    @field_validator("run_accession")
    @classmethod
    def validate_run_accession(cls, v: str) -> str:
        """Validate run accession format (SRR* or ERR* or DRR*)."""
        if not re.match(r"^[SED]RR\d+$", v):
            logger.warning(f"Unexpected run_accession format: '{v}'")
        return v

    @field_validator("bioproject")
    @classmethod
    def validate_bioproject(cls, v: str) -> str:
        """Validate BioProject accession format (PRJNA*, PRJEB*, PRJDB*)."""
        if not re.match(r"^PRJ[NED][A-Z]\d+$", v):
            logger.warning(f"Unexpected bioproject format: '{v}'")
        return v

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SRASampleSchema":
        """
        Create schema from dict, handling 71 fields flexibly.

        Extracts known fields into Pydantic fields, stores rest in
        additional_metadata. This allows us to validate critical fields
        while preserving all NCBI metadata.

        Args:
            data: SRA sample dictionary (71 fields from NCBI API)

        Returns:
            SRASampleSchema: Validated schema instance

        Examples:
            >>> sample = {"run_accession": "SRR001", ...}  # 71 fields
            >>> validated = SRASampleSchema.from_dict(sample)
            >>> validated.run_accession
            'SRR001'
            >>> len(validated.additional_metadata)
            50  # Remaining fields
        """
        known_fields = set(cls.model_fields.keys()) - {"additional_metadata"}
        schema_data = {k: v for k, v in data.items() if k in known_fields}

        # Store remaining fields in additional_metadata
        additional = {k: v for k, v in data.items() if k not in known_fields}
        if additional:
            schema_data["additional_metadata"] = additional

        return cls(**schema_data)

    def has_download_url(self) -> bool:
        """
        Check if at least one download URL is available.

        Returns:
            bool: True if any download URL is present

        Examples:
            >>> sample = SRASampleSchema(...)
            >>> sample.has_download_url()
            True
        """
        return bool(self.public_url or self.ncbi_url or self.aws_url or self.gcp_url)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation, including additional_metadata.

        Returns:
            Dict[str, Any]: Complete sample dictionary with all 71 fields

        Examples:
            >>> validated = SRASampleSchema.from_dict(sample_dict)
            >>> reconstructed = validated.to_dict()
            >>> len(reconstructed)
            71  # All original fields preserved
        """
        base_dict = self.model_dump(exclude={"additional_metadata"}, exclude_none=True)
        if self.additional_metadata:
            base_dict.update(self.additional_metadata)
        return base_dict


def validate_sra_sample(sample: Dict[str, Any]) -> ValidationResult:
    """
    Validate single SRA sample using Pydantic schema.

    Returns ValidationResult following existing pattern from validation.py.
    Uses errors for critical issues, warnings for non-critical issues.

    Args:
        sample: SRA sample dictionary (71 fields from NCBI API)

    Returns:
        ValidationResult with errors/warnings/info

    Examples:
        >>> sample = {"run_accession": "SRR001", "library_strategy": "AMPLICON", ...}
        >>> result = validate_sra_sample(sample)
        >>> result.is_valid  # True if no errors
        True
        >>> len(result.warnings)  # May have warnings
        1
        >>> result.summary()
        'Validation completed with 1 warning(s)'
    """
    result = ValidationResult()

    try:
        validated = SRASampleSchema.from_dict(sample)

        # Critical check: At least one download URL must be present
        if not validated.has_download_url():
            result.add_error(
                f"Sample {validated.run_accession}: No download URLs available. "
                f"At least one of (public_url, ncbi_url, aws_url, gcp_url) is required."
            )

        # Warn about missing environmental context (important for microbiome filtering)
        if validated.library_strategy == "AMPLICON":
            if not validated.env_medium:
                result.add_warning(
                    f"Sample {validated.run_accession}: Missing 'env_medium' field. "
                    f"This field is important for microbiome filtering (fecal vs tissue samples)."
                )

        # Informational: successful validation
        result.add_info(
            f"Sample {validated.run_accession} validated successfully "
            f"(library_strategy: {validated.library_strategy})"
        )

    except ValidationError as e:
        # Pydantic validation failed - critical errors
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            result.add_error(f"Field '{field}': {msg}")

    except Exception as e:
        # Unexpected error
        result.add_error(f"Unexpected validation error: {str(e)}")
        logger.error(f"Unexpected error validating SRA sample: {e}", exc_info=True)

    return result


def validate_sra_samples_batch(samples: List[Dict[str, Any]]) -> ValidationResult:
    """
    Validate list of SRA samples and return aggregated results.

    This function validates each sample individually and merges all
    ValidationResults into a single aggregated result with batch-level
    statistics.

    Args:
        samples: List of SRA sample dictionaries

    Returns:
        ValidationResult: Aggregated validation results with batch statistics

    Examples:
        >>> samples = [
        ...     {"run_accession": "SRR001", ...},
        ...     {"run_accession": "SRR002", ...},
        ...     {"run_accession": "SRR003", ...}  # Missing required field
        ... ]
        >>> result = validate_sra_samples_batch(samples)
        >>> result.metadata["total_samples"]
        3
        >>> result.metadata["valid_samples"]
        2
        >>> result.metadata["validation_rate"]
        66.67
    """
    aggregated = ValidationResult()

    # Validate each sample
    for idx, sample in enumerate(samples):
        sample_result = validate_sra_sample(sample)
        aggregated = aggregated.merge(sample_result)

    # Add batch-level statistics
    valid_samples = len(samples) - len(
        [e for e in aggregated.errors if "Field" in e or "No download URLs" in e]
    )

    aggregated.metadata["total_samples"] = len(samples)
    aggregated.metadata["valid_samples"] = valid_samples
    aggregated.metadata["validation_rate"] = (
        (valid_samples / len(samples) * 100) if samples else 0.0
    )
    aggregated.metadata["error_count"] = len(aggregated.errors)
    aggregated.metadata["warning_count"] = len(aggregated.warnings)

    # Add summary info message
    if samples:
        aggregated.add_info(
            f"Batch validation complete: {valid_samples}/{len(samples)} samples valid "
            f"({aggregated.metadata['validation_rate']:.1f}%)"
        )

    return aggregated


def is_valid_sra_sample_key(ws_key: str) -> tuple[bool, Optional[str]]:
    """
    Validate SRA sample workspace key format.

    Validates that workspace key follows the expected pattern:
    'sra_<bioproject_id>_samples' where bioproject_id matches PRJNA*/PRJEB*/PRJDB*.

    Args:
        ws_key: Workspace key string (e.g., "sra_PRJNA891765_samples")

    Returns:
        (is_valid, reason): True if valid, reason string if invalid

    Examples:
        >>> is_valid_sra_sample_key("sra_PRJNA891765_samples")
        (True, None)
        >>> is_valid_sra_sample_key("pub_queue_doi_123_metadata.json")
        (False, "does not start with 'sra_'")
        >>> is_valid_sra_sample_key("sra_INVALID_samples")
        (False, "invalid project ID format: INVALID")
    """
    if not ws_key.startswith("sra_"):
        return False, "does not start with 'sra_'"

    if not ws_key.endswith("_samples"):
        return False, "does not end with '_samples'"

    # Extract project ID (e.g., "PRJNA891765" from "sra_PRJNA891765_samples")
    parts = ws_key.split("_")
    if len(parts) < 3:
        return False, "invalid format (expected 'sra_<project>_samples')"

    project_id = "_".join(parts[1:-1])  # Handle multi-part IDs

    # Validate project ID format (PRJNA, PRJEB, PRJDB + digits)
    # Common patterns:
    # - PRJNA891765 (NCBI)
    # - PRJEB12345 (ENA/EBI)
    # - PRJDB6789 (DDBJ)
    if not re.match(r"^PRJ[NED][A-Z]\d+$", project_id):
        return False, f"invalid project ID format: {project_id}"

    return True, None
