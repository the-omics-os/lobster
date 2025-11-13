"""
Metadata Standardization Service for Sample-Level Metadata Operations.

This service provides standardization, validation, and reading of sample-level
metadata using Pydantic schemas for cross-dataset harmonization.

Used by metadata_assistant agent for metadata operations.

Phase 3 implementation for research agent refactoring.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.proteomics import ProteomicsMetadataSchema
from lobster.core.schemas.transcriptomics import TranscriptomicsMetadataSchema
from lobster.tools.metadata_validation_service import MetadataValidationService

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for Standardization Results
# =============================================================================


class StandardizationResult(BaseModel):
    """Result of metadata standardization operation.

    Attributes:
        standardized_metadata: List of standardized Pydantic schema instances
        validation_errors: Dict mapping sample IDs to validation error messages
        field_coverage: Dict showing % of samples with each field
        warnings: List of warnings encountered during standardization
    """

    standardized_metadata: List[
        Union[TranscriptomicsMetadataSchema, ProteomicsMetadataSchema]
    ] = Field(default_factory=list, description="Standardized metadata schemas")
    validation_errors: Dict[str, str] = Field(
        default_factory=dict, description="Sample ID -> validation error message"
    )
    field_coverage: Dict[str, float] = Field(
        default_factory=dict, description="Field -> % of samples with field"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")


class DatasetValidationResult(BaseModel):
    """Result of dataset content validation.

    Attributes:
        has_required_samples: Whether dataset has minimum required samples
        missing_conditions: List of expected conditions not found
        control_issues: List of control-related warnings
        duplicate_ids: List of duplicate sample IDs
        platform_consistency: Whether platform metadata is consistent
        summary: Summary of validation results
        warnings: List of validation warnings
    """

    has_required_samples: bool = Field(
        ..., description="Whether dataset has minimum required samples"
    )
    missing_conditions: List[str] = Field(
        default_factory=list, description="Expected conditions not found"
    )
    control_issues: List[str] = Field(
        default_factory=list, description="Control-related warnings"
    )
    duplicate_ids: List[str] = Field(
        default_factory=list, description="Duplicate sample IDs"
    )
    platform_consistency: bool = Field(
        ..., description="Whether platform metadata is consistent"
    )
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary stats")
    warnings: List[str] = Field(default_factory=list, description="Warnings")


# =============================================================================
# MetadataStandardizationService
# =============================================================================


class MetadataStandardizationService:
    """Service for sample-level metadata standardization and validation.

    This service converts raw metadata to Pydantic schemas, reads sample metadata
    in different formats, and validates dataset completeness.

    Integrates with MetadataValidationService for field normalization.

    Follows Lobster architecture:
    - Receives DataManagerV2 in __init__
    - Stateless operations
    - Returns Pydantic result models
    """

    def __init__(self, data_manager: DataManagerV2) -> None:
        """Initialize MetadataStandardizationService.

        Args:
            data_manager: DataManagerV2 instance for modality access
        """
        self.data_manager = data_manager
        self.metadata_validator = MetadataValidationService(data_manager)

        # Schema registry mapping modality types to Pydantic schemas
        self.schema_registry = {
            "transcriptomics": TranscriptomicsMetadataSchema,
            "single_cell": TranscriptomicsMetadataSchema,
            "bulk_rna_seq": TranscriptomicsMetadataSchema,
            "proteomics": ProteomicsMetadataSchema,
            "mass_spectrometry": ProteomicsMetadataSchema,
            "affinity": ProteomicsMetadataSchema,
        }

        logger.info("MetadataStandardizationService initialized")

    # =========================================================================
    # Core Methods
    # =========================================================================

    def standardize_metadata(
        self,
        identifier: str,
        target_schema: str,
        controlled_vocabularies: Optional[Dict[str, List[str]]] = None,
    ) -> StandardizationResult:
        """Convert raw metadata to Pydantic schema with validation.

        Args:
            identifier: Dataset identifier
            target_schema: Target schema type ("transcriptomics" or "proteomics")
            controlled_vocabularies: Optional controlled vocabularies to enforce

        Returns:
            StandardizationResult with standardized metadata and validation errors

        Raises:
            ValueError: If dataset not found or schema type unknown
        """
        # Validate dataset exists
        if identifier not in self.data_manager.list_modalities():
            raise ValueError(f"Dataset '{identifier}' not found")

        # Get schema class
        if target_schema not in self.schema_registry:
            raise ValueError(
                f"Unknown schema type '{target_schema}'. "
                f"Available: {list(self.schema_registry.keys())}"
            )

        schema_class = self.schema_registry[target_schema]
        adata = self.data_manager.get_modality(identifier)

        logger.info(
            f"Standardizing metadata for {identifier} with {target_schema} schema"
        )

        # Extract sample metadata
        if not hasattr(adata, "obs") or adata.obs.empty:
            raise ValueError(f"Dataset '{identifier}' has no sample metadata")

        standardized = []
        validation_errors = {}
        field_coverage = {}
        warnings = []

        # Count field presence
        field_counts = adata.obs.notna().sum().to_dict()
        total_samples = len(adata.obs)
        field_coverage = {
            field: (count / total_samples) * 100
            for field, count in field_counts.items()
        }

        # Standardize each sample
        for sample_id, row in adata.obs.iterrows():
            try:
                # Convert row to dict, normalize field names
                raw_metadata = row.dropna().to_dict()
                normalized_metadata = {}

                for field, value in raw_metadata.items():
                    # Normalize field name using MetadataValidationService
                    norm_field = self.metadata_validator.normalize_field_name(field)
                    # Normalize value if it's a string
                    if isinstance(value, str):
                        norm_value = self.metadata_validator.normalize_field_value(
                            field, value
                        )
                    else:
                        norm_value = value

                    normalized_metadata[norm_field] = norm_value

                # Add sample_id if not present
                if "sample_id" not in normalized_metadata:
                    normalized_metadata["sample_id"] = str(sample_id)

                # Apply controlled vocabularies if provided
                if controlled_vocabularies:
                    for field, allowed_values in controlled_vocabularies.items():
                        if field in normalized_metadata:
                            value = normalized_metadata[field]
                            if value not in allowed_values:
                                warnings.append(
                                    f"Sample {sample_id}: '{field}' value '{value}' "
                                    f"not in controlled vocabulary {allowed_values}"
                                )

                # Create Pydantic schema instance
                schema_instance = schema_class.from_dict(normalized_metadata)
                standardized.append(schema_instance)

            except ValidationError as e:
                error_msg = "; ".join(
                    [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
                )
                validation_errors[str(sample_id)] = error_msg
                logger.warning(f"Validation error for sample {sample_id}: {error_msg}")

            except Exception as e:
                validation_errors[str(sample_id)] = str(e)
                logger.error(f"Unexpected error for sample {sample_id}: {e}")

        logger.info(
            f"Standardization complete: {len(standardized)} valid, "
            f"{len(validation_errors)} errors"
        )

        return StandardizationResult(
            standardized_metadata=standardized,
            validation_errors=validation_errors,
            field_coverage=field_coverage,
            warnings=warnings,
        )

    def read_sample_metadata(
        self,
        identifier: str,
        fields: Optional[List[str]] = None,
        return_format: str = "summary",
    ) -> Union[str, Dict[str, Any], pd.DataFrame]:
        """Extract and format sample metadata in different modes.

        Args:
            identifier: Dataset identifier
            fields: Optional list of fields to extract (None = all fields)
            return_format: Output format ("summary", "detailed", "schema")

        Returns:
            - "summary": Human-readable string summary
            - "detailed": Dict with complete metadata
            - "schema": DataFrame with requested fields

        Raises:
            ValueError: If dataset not found or invalid format
        """
        # Validate dataset exists
        if identifier not in self.data_manager.list_modalities():
            raise ValueError(f"Dataset '{identifier}' not found")

        adata = self.data_manager.get_modality(identifier)

        if not hasattr(adata, "obs") or adata.obs.empty:
            raise ValueError(f"Dataset '{identifier}' has no sample metadata")

        logger.info(
            f"Reading sample metadata for {identifier} (format: {return_format})"
        )

        # Filter fields if specified
        if fields:
            available_fields = [f for f in fields if f in adata.obs.columns]
            if not available_fields:
                raise ValueError(f"None of the requested fields found in dataset")
            metadata_df = adata.obs[available_fields]
        else:
            metadata_df = adata.obs

        # Format output based on return_format
        if return_format == "summary":
            # Generate human-readable summary
            total_samples = len(metadata_df)
            field_count = len(metadata_df.columns)
            field_coverage = (
                (metadata_df.notna().sum() / total_samples * 100).round(1).to_dict()
            )

            summary_lines = [
                f"Dataset: {identifier}",
                f"Total Samples: {total_samples}",
                f"Fields: {field_count}",
                "\nField Coverage:",
            ]
            for field, coverage in sorted(
                field_coverage.items(), key=lambda x: x[1], reverse=True
            ):
                summary_lines.append(f"  - {field}: {coverage}%")

            return "\n".join(summary_lines)

        elif return_format == "detailed":
            # Return complete metadata as dict
            return {
                "identifier": identifier,
                "total_samples": len(metadata_df),
                "fields": list(metadata_df.columns),
                "metadata": metadata_df.to_dict(orient="index"),
            }

        elif return_format == "schema":
            # Return DataFrame directly
            return metadata_df

        else:
            raise ValueError(
                f"Invalid return_format '{return_format}'. "
                f"Must be 'summary', 'detailed', or 'schema'"
            )

    def validate_dataset_content(
        self,
        identifier: str,
        expected_samples: Optional[int] = None,
        required_conditions: Optional[List[str]] = None,
        check_controls: bool = True,
        check_duplicates: bool = True,
    ) -> DatasetValidationResult:
        """Validate dataset completeness and metadata quality.

        Args:
            identifier: Dataset identifier
            expected_samples: Minimum expected sample count (None = no check)
            required_conditions: List of required condition values (None = no check)
            check_controls: Whether to check for control samples
            check_duplicates: Whether to check for duplicate sample IDs

        Returns:
            DatasetValidationResult with validation results

        Raises:
            ValueError: If dataset not found
        """
        # Validate dataset exists
        if identifier not in self.data_manager.list_modalities():
            raise ValueError(f"Dataset '{identifier}' not found")

        adata = self.data_manager.get_modality(identifier)

        logger.info(f"Validating dataset content for {identifier}")

        warnings = []

        # Check 1: Sample count verification
        actual_samples = len(adata.obs)
        has_required_samples = True

        if expected_samples is not None:
            has_required_samples = actual_samples >= expected_samples
            if not has_required_samples:
                warnings.append(
                    f"Expected at least {expected_samples} samples, found {actual_samples}"
                )

        # Check 2: Condition presence check
        missing_conditions = []
        if required_conditions and "condition" in adata.obs.columns:
            available_conditions = set(adata.obs["condition"].dropna().unique())
            for condition in required_conditions:
                if condition not in available_conditions:
                    missing_conditions.append(condition)
                    warnings.append(f"Required condition '{condition}' not found")

        # Check 3: Control sample detection
        control_issues = []
        if check_controls and "condition" in adata.obs.columns:
            conditions = adata.obs["condition"].str.lower().dropna()
            control_samples = conditions[
                conditions.str.contains("control", case=False, na=False)
            ]
            if len(control_samples) == 0:
                control_issues.append("No control samples found")
                warnings.append("No control samples detected in 'condition' field")

        # Check 4: Duplicate ID check
        duplicate_ids = []
        if check_duplicates:
            duplicates = adata.obs_names[adata.obs_names.duplicated()].tolist()
            duplicate_ids = list(set(duplicates))
            if duplicate_ids:
                warnings.append(f"Found {len(duplicate_ids)} duplicate sample IDs")

        # Check 5: Platform consistency check
        platform_consistency = True
        if "platform" in adata.obs.columns:
            unique_platforms = adata.obs["platform"].dropna().unique()
            if len(unique_platforms) > 1:
                platform_consistency = False
                warnings.append(
                    f"Inconsistent platforms found: {list(unique_platforms)}"
                )

        # Summary statistics
        summary = {
            "total_samples": actual_samples,
            "unique_conditions": (
                len(adata.obs["condition"].dropna().unique())
                if "condition" in adata.obs.columns
                else 0
            ),
            "has_condition_field": "condition" in adata.obs.columns,
            "has_platform_field": "platform" in adata.obs.columns,
            "has_batch_field": "batch" in adata.obs.columns,
        }

        logger.info(
            f"Validation complete: {len(warnings)} warnings, "
            f"{len(duplicate_ids)} duplicates"
        )

        return DatasetValidationResult(
            has_required_samples=has_required_samples,
            missing_conditions=missing_conditions,
            control_issues=control_issues,
            duplicate_ids=duplicate_ids,
            platform_consistency=platform_consistency,
            summary=summary,
            warnings=warnings,
        )
