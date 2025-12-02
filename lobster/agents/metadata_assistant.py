"""
Metadata Assistant Agent for Cross-Dataset Metadata Operations.

This agent specializes in sample ID mapping, metadata standardization, and
dataset content validation for multi-omics integration.

Phase 3 implementation for research agent refactoring.

Note: This agent replaces research_agent_assistant's metadata functionality.
The PDF resolution features were archived and will be migrated to research_agent
in Phase 4. See lobster/agents/archive/ARCHIVE_NOTICE.md for details.
"""

import json
from datetime import datetime
from typing import List, Dict, Any

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.metadata.metadata_standardization_service import (
    MetadataStandardizationService,
)
from lobster.services.metadata.sample_mapping_service import SampleMappingService
from lobster.core.analysis_ir import AnalysisStep
from lobster.core.schemas.publication_queue import PublicationStatus, HandoffStatus
from lobster.core.interfaces.validator import ValidationResult
from lobster.utils.logger import get_logger
from lobster.services.execution.custom_code_execution_service import (
    CustomCodeExecutionService,
    CodeExecutionError,
    CodeValidationError,
)

# Optional microbiome features (not in public lobster-local)
try:
    from lobster.services.metadata.microbiome_filtering_service import MicrobiomeFilteringService
    from lobster.services.metadata.disease_standardization_service import DiseaseStandardizationService
    MICROBIOME_FEATURES_AVAILABLE = True
except ImportError:
    MicrobiomeFilteringService = None
    DiseaseStandardizationService = None
    MICROBIOME_FEATURES_AVAILABLE = False

logger = get_logger(__name__)


def metadata_assistant(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "metadata_assistant",
    delegation_tools: list = None,
):
    """Create metadata assistant agent for metadata operations.

    This agent provides 4-5 specialized tools for metadata operations:
    1. map_samples_by_id - Cross-dataset sample ID mapping
    2. read_sample_metadata - Extract and format sample metadata
    3. standardize_sample_metadata - Convert to Pydantic schemas
    4. validate_dataset_content - Validate dataset completeness
    5. filter_samples_by - Multi-criteria filtering (16S + host + sample_type + disease)
       [OPTIONAL - only available if microbiome features are installed]

    Args:
        data_manager: DataManagerV2 instance
        callback_handler: Optional callback handler for LLM
        agent_name: Agent name for identification
        delegation_tools: Optional list of delegation tools for sub-agent access

    Returns:
        Compiled LangGraph agent with metadata tools
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("assistant")
    llm = create_llm("metadata_assistant", model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    # Initialize services (Phase 3: new services)
    sample_mapping_service = SampleMappingService(data_manager=data_manager)
    metadata_standardization_service = MetadataStandardizationService(
        data_manager=data_manager
    )

    # Initialize optional microbiome services if available
    microbiome_filtering_service = None
    disease_standardization_service = None
    if MICROBIOME_FEATURES_AVAILABLE:
        microbiome_filtering_service = MicrobiomeFilteringService()
        disease_standardization_service = DiseaseStandardizationService()
        logger.debug("Microbiome features enabled")
    else:
        logger.debug("Microbiome features not available (optional)")

    # Custom code execution service for sample-level operations
    custom_code_service = CustomCodeExecutionService(data_manager)

    logger.debug("metadata_assistant agent initialized")

    # =========================================================================
    # Tool 1: Sample ID Mapping
    # =========================================================================

    @tool
    def map_samples_by_id(
        source: str,
        target: str,
        source_type: str,
        target_type: str,
        min_confidence: float = 0.75,
        strategies: str = "all",
    ) -> str:
        """
        Map sample IDs between two datasets for multi-omics integration.

        Use this tool when you need to harmonize sample identifiers across datasets
        with different naming conventions. The service uses multiple matching strategies:
        - Exact: Case-insensitive exact matching
        - Fuzzy: RapidFuzz-based similarity matching (requires RapidFuzz)
        - Pattern: Common prefix/suffix removal (Sample_, GSM, _Rep1, etc.)
        - Metadata: Metadata-supported matching (condition, tissue, timepoint, etc.)

        Args:
            source: Source modality name or dataset ID
            target: Target modality name or dataset ID
            source_type: Source data type - REQUIRED, must be "modality" or "metadata_store"
                - "modality": Work with loaded AnnData in DataManagerV2
                - "metadata_store": Work with cached metadata (pre-download validation)
            target_type: Target data type - REQUIRED, must be "modality" or "metadata_store"
            min_confidence: Minimum confidence threshold for fuzzy matches (0.0-1.0, default: 0.75)
            strategies: Comma-separated strategies to use (default: "all", options: "exact,fuzzy,pattern,metadata")

        Returns:
            Formatted markdown report with match results, confidence scores, and unmapped samples

        Examples:
            # Map between two loaded modalities
            map_samples_by_id(source="geo_gse1", target="geo_gse2",
                            source_type="modality", target_type="modality")

            # Map between cached metadata (pre-download)
            map_samples_by_id(source="geo_gse1", target="geo_gse2",
                            source_type="metadata_store", target_type="metadata_store")

            # Mixed: modality to cached metadata
            map_samples_by_id(source="geo_gse1", target="geo_gse2",
                            source_type="modality", target_type="metadata_store")
        """
        try:
            logger.info(
                f"Mapping samples: {source} → {target} "
                f"(source_type={source_type}, target_type={target_type}, min_confidence={min_confidence})"
            )

            # Validate source_type and target_type
            for stype, name in [
                (source_type, "source_type"),
                (target_type, "target_type"),
            ]:
                if stype not in ["modality", "metadata_store"]:
                    return f"❌ Error: {name} must be 'modality' or 'metadata_store', got '{stype}'"

            # Parse strategies
            strategy_list = None
            if strategies and strategies.lower() != "all":
                strategy_list = [s.strip().lower() for s in strategies.split(",")]
                # Validate strategies
                valid_strategies = {"exact", "fuzzy", "pattern", "metadata"}
                invalid = set(strategy_list) - valid_strategies
                if invalid:
                    return (
                        f"❌ Invalid strategies: {invalid}. "
                        f"Valid options: {valid_strategies}"
                    )

            # Helper function to get samples based on type
            import pandas as pd

            def get_samples(identifier: str, id_type: str) -> pd.DataFrame:
                if id_type == "modality":
                    if identifier not in data_manager.list_modalities():
                        raise ValueError(
                            f"Modality '{identifier}' not found. Available: {', '.join(data_manager.list_modalities())}"
                        )
                    adata = data_manager.get_modality(identifier)
                    return adata.obs  # Returns sample metadata DataFrame

                elif id_type == "metadata_store":
                    if identifier not in data_manager.metadata_store:
                        raise ValueError(
                            f"'{identifier}' not found in metadata_store. Use research_agent.validate_dataset_metadata() first."
                        )
                    cached = data_manager.metadata_store[identifier]
                    samples_dict = cached.get("metadata", {}).get("samples", {})
                    if not samples_dict:
                        raise ValueError(f"No sample metadata in '{identifier}'")
                    return pd.DataFrame.from_dict(samples_dict, orient="index")

            # Get samples from both sources
            get_samples(source, source_type)
            get_samples(target, target_type)

            # Call mapping service (updated to work with DataFrames directly)
            result = sample_mapping_service.map_samples_by_id(
                source_identifier=source,
                target_identifier=target,
                strategies=strategy_list,
            )

            # Log provenance
            data_manager.log_tool_usage(
                tool_name="map_samples_by_id",
                parameters={
                    "source": source,
                    "target": target,
                    "source_type": source_type,
                    "target_type": target_type,
                    "min_confidence": min_confidence,
                    "strategies": strategies,
                },
                result_summary={
                    "exact_matches": result.summary["exact_matches"],
                    "fuzzy_matches": result.summary["fuzzy_matches"],
                    "unmapped": result.summary["unmapped"],
                    "mapping_rate": result.summary["mapping_rate"],
                },
            )

            # Format report
            report = sample_mapping_service.format_mapping_report(result)

            logger.info(
                f"Mapping complete: {result.summary['mapping_rate']:.1%} success rate"
            )
            return report

        except ValueError as e:
            logger.error(f"Mapping error: {e}")
            return f"❌ Mapping failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected mapping error: {e}", exc_info=True)
            return f"❌ Unexpected error during mapping: {str(e)}"

    # =========================================================================
    # Tool 2: Read Sample Metadata
    # =========================================================================

    @tool
    def read_sample_metadata(
        source: str,
        source_type: str,
        fields: str = None,
        return_format: str = "summary",
    ) -> str:
        """
        Read and format sample-level metadata from loaded modality OR cached metadata.

        Use this tool to extract sample metadata in different formats:
        - "summary": Quick overview with field coverage percentages
        - "detailed": Complete metadata as JSON for programmatic access
        - "schema": Full metadata table for inspection

        Args:
            source: Modality name or dataset ID
            source_type: Data source type - REQUIRED, must be "modality" or "metadata_store"
                - "modality": Work with loaded AnnData in DataManagerV2
                - "metadata_store": Work with cached metadata (pre-download validation)
            fields: Optional comma-separated list of fields to extract (default: all fields)
            return_format: Output format (default: "summary", options: "summary,detailed,schema")

        Returns:
            Formatted metadata according to return_format specification

        Examples:
            # Read from loaded modality
            read_sample_metadata(source="geo_gse180759", source_type="modality")

            # Read from cached metadata (pre-download)
            read_sample_metadata(source="geo_gse180759", source_type="metadata_store")
        """
        try:
            logger.info(
                f"Reading metadata for {source} (source_type={source_type}, format: {return_format})"
            )

            # Validate source_type
            if source_type not in ["modality", "metadata_store"]:
                return f"❌ Error: source_type must be 'modality' or 'metadata_store', got '{source_type}'"

            # Parse fields
            field_list = None
            if fields:
                field_list = [f.strip() for f in fields.split(",")]

            # Get sample metadata based on source_type
            import pandas as pd

            if source_type == "modality":
                if source not in data_manager.list_modalities():
                    return f"❌ Error: Modality '{source}' not found. Available: {', '.join(data_manager.list_modalities())}"
                adata = data_manager.get_modality(source)
                sample_df = adata.obs
            elif source_type == "metadata_store":
                if source not in data_manager.metadata_store:
                    return f"❌ Error: '{source}' not found in metadata_store. Use research_agent.validate_dataset_metadata() first."
                cached = data_manager.metadata_store[source]
                samples_dict = cached.get("metadata", {}).get("samples", {})
                if not samples_dict:
                    return f"❌ Error: No sample metadata in '{source}'"
                sample_df = pd.DataFrame.from_dict(samples_dict, orient="index")

            # Filter fields if specified
            if field_list:
                available_fields = list(sample_df.columns)
                missing_fields = [f for f in field_list if f not in available_fields]
                if missing_fields:
                    return f"❌ Error: Fields not found: {', '.join(missing_fields)}"
                sample_df = sample_df[field_list]

            # Log provenance
            data_manager.log_tool_usage(
                tool_name="read_sample_metadata",
                parameters={
                    "source": source,
                    "source_type": source_type,
                    "fields": fields,
                    "return_format": return_format,
                },
                result_summary={"format": return_format, "num_samples": len(sample_df)},
            )

            # Format output based on return_format
            if return_format == "summary":
                logger.info(f"Metadata summary generated for {source}")
                # Generate summary
                summary = [
                    "# Sample Metadata Summary\n",
                    f"**Dataset**: {source}",
                    f"**Source Type**: {source_type}",
                    f"**Total Samples**: {len(sample_df)}\n",
                    "## Field Coverage:",
                ]
                for col in sample_df.columns:
                    non_null = sample_df[col].notna().sum()
                    pct = (non_null / len(sample_df)) * 100
                    summary.append(f"- {col}: {pct:.1f}% ({non_null}/{len(sample_df)})")
                return "\n".join(summary)
            elif return_format == "detailed":
                logger.info(f"Detailed metadata extracted for {source}")
                return json.dumps(sample_df.to_dict(orient="records"), indent=2)
            elif return_format == "schema":
                logger.info(f"Metadata schema extracted for {source}")
                return sample_df.to_markdown(index=True)
            else:
                return f"❌ Invalid return_format '{return_format}'"

        except ValueError as e:
            logger.error(f"Metadata read error: {e}")
            return f"❌ Failed to read metadata: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected metadata read error: {e}", exc_info=True)
            return f"❌ Unexpected error reading metadata: {str(e)}"

    # =========================================================================
    # Tool 3: Standardize Sample Metadata
    # =========================================================================

    @tool
    def standardize_sample_metadata(
        source: str,
        source_type: str,
        target_schema: str,
        controlled_vocabularies: str = None,
    ) -> str:
        """
        Standardize sample metadata using Pydantic schemas for cross-dataset harmonization.

        Use this tool to convert raw metadata to standardized Pydantic schemas
        (TranscriptomicsMetadataSchema or ProteomicsMetadataSchema) with field
        normalization and controlled vocabulary enforcement.

        Args:
            source: Modality name or dataset ID
            source_type: Data source type - REQUIRED, must be "modality" or "metadata_store"
                - "modality": Work with loaded AnnData in DataManagerV2
                - "metadata_store": Work with cached metadata (pre-download validation)
            target_schema: Target schema type (options: "transcriptomics", "proteomics", "bulk_rna_seq", "single_cell", "mass_spectrometry", "affinity")
            controlled_vocabularies: Optional JSON string of controlled vocabularies (e.g., '{"condition": ["Control", "Treatment"]}')

        Returns:
            Standardization report with field coverage, validation errors, and warnings

        Examples:
            # Standardize from loaded modality
            standardize_sample_metadata(source="geo_gse12345", source_type="modality",
                                       target_schema="transcriptomics")

            # Standardize from cached metadata
            standardize_sample_metadata(source="geo_gse12345", source_type="metadata_store",
                                       target_schema="transcriptomics")
        """
        try:
            logger.info(
                f"Standardizing metadata for {source} (source_type={source_type}) with {target_schema} schema"
            )

            # Validate source_type
            if source_type not in ["modality", "metadata_store"]:
                return f"❌ Error: source_type must be 'modality' or 'metadata_store', got '{source_type}'"

            # Parse controlled vocabularies if provided
            controlled_vocab_dict = None
            if controlled_vocabularies:
                try:
                    controlled_vocab_dict = json.loads(controlled_vocabularies)
                except json.JSONDecodeError as e:
                    return f"❌ Invalid controlled_vocabularies JSON: {str(e)}"

            # Call standardization service
            # Note: standardization service may need to be updated to handle source_type
            result, stats, ir = metadata_standardization_service.standardize_metadata(
                identifier=source,
                target_schema=target_schema,
                controlled_vocabularies=controlled_vocab_dict,
            )

            # Log provenance with IR
            data_manager.log_tool_usage(
                tool_name="standardize_sample_metadata",
                parameters={
                    "source": source,
                    "source_type": source_type,
                    "target_schema": target_schema,
                    "controlled_vocabularies": controlled_vocabularies,
                },
                result_summary={
                    "valid_samples": len(result.standardized_metadata),
                    "validation_errors": len(result.validation_errors),
                    "warnings": len(result.warnings),
                },
                ir=ir,  # Pass IR for provenance tracking
            )

            # Format report
            report_lines = [
                "# Metadata Standardization Report\n",
                f"**Dataset:** {source}",
                f"**Source Type:** {source_type}",
                f"**Target Schema:** {target_schema}",
                f"**Valid Samples:** {len(result.standardized_metadata)}",
                f"**Validation Errors:** {len(result.validation_errors)}\n",
            ]

            # Field coverage
            if result.field_coverage:
                report_lines.append("## Field Coverage")
                for field, coverage in sorted(
                    result.field_coverage.items(), key=lambda x: x[1], reverse=True
                ):
                    report_lines.append(f"- {field}: {coverage:.1f}%")
                report_lines.append("")

            # Validation errors (show first 10)
            if result.validation_errors:
                report_lines.append("## Validation Errors")
                for i, (sample_id, error) in enumerate(
                    list(result.validation_errors.items())[:10]
                ):
                    report_lines.append(f"- {sample_id}: {error}")
                if len(result.validation_errors) > 10:
                    report_lines.append(
                        f"- ... and {len(result.validation_errors) - 10} more"
                    )
                report_lines.append("")

            # Warnings
            if result.warnings:
                report_lines.append("## Warnings")
                for warning in result.warnings[:10]:
                    report_lines.append(f"- ⚠️ {warning}")
                if len(result.warnings) > 10:
                    report_lines.append(f"- ... and {len(result.warnings) - 10} more")

            report = "\n".join(report_lines)
            logger.info(f"Standardization complete for {source}")
            return report

        except ValueError as e:
            logger.error(f"Standardization error: {e}")
            return f"❌ Standardization failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected standardization error: {e}", exc_info=True)
            return f"❌ Unexpected error during standardization: {str(e)}"

    # =========================================================================
    # Tool 4: Validate Dataset Content
    # =========================================================================

    @tool
    def validate_dataset_content(
        source: str,
        source_type: str,
        expected_samples: int = None,
        required_conditions: str = None,
        check_controls: bool = True,
        check_duplicates: bool = True,
    ) -> str:
        """
        Validate dataset completeness and metadata quality from loaded modality OR cached metadata.

        Use this tool to verify that a dataset meets minimum requirements:
        - Sample count verification
        - Condition presence check
        - Control sample detection
        - Duplicate ID check
        - Platform consistency check

        Args:
            source: Modality name (if source_type="modality") or dataset ID (if source_type="metadata_store")
            source_type: Data source type - REQUIRED, must be "modality" or "metadata_store"
                - "modality": Validate from loaded AnnData in DataManagerV2
                - "metadata_store": Validate from cached GEO metadata (pre-download validation)
            expected_samples: Minimum expected sample count (optional)
            required_conditions: Comma-separated list of required condition values (optional)
            check_controls: Whether to check for control samples (default: True)
            check_duplicates: Whether to check for duplicate sample IDs (default: True)

        Returns:
            Validation report with checks results, warnings, and recommendations

        Examples:
            # Post-download validation
            validate_dataset_content(source="geo_gse180759", source_type="modality")

            # Pre-download validation (before loading dataset)
            validate_dataset_content(source="geo_gse180759", source_type="metadata_store")
        """
        try:
            logger.info(
                f"Validating dataset content for {source} (source_type={source_type})"
            )

            # Validate source_type parameter
            if source_type not in ["modality", "metadata_store"]:
                return f"❌ Error: source_type must be 'modality' or 'metadata_store', got '{source_type}'"

            # Parse required conditions
            required_condition_list = None
            if required_conditions:
                required_condition_list = [
                    c.strip() for c in required_conditions.split(",")
                ]

            # Branch based on source_type
            if source_type == "modality":
                # EXISTING BEHAVIOR: Validate from loaded modality
                if source not in data_manager.list_modalities():
                    return (
                        f"❌ Error: Modality '{source}' not found in DataManager. "
                        f"Available modalities: {', '.join(data_manager.list_modalities())}"
                    )

                # Call validation service
                result, stats, ir = (
                    metadata_standardization_service.validate_dataset_content(
                        identifier=source,
                        expected_samples=expected_samples,
                        required_conditions=required_condition_list,
                        check_controls=check_controls,
                        check_duplicates=check_duplicates,
                    )
                )

            elif source_type == "metadata_store":
                # NEW BEHAVIOR: Pre-download validation from cached metadata
                if source not in data_manager.metadata_store:
                    return (
                        f"❌ Error: '{source}' not found in metadata_store. "
                        f"Use research_agent.validate_dataset_metadata() first to cache metadata."
                    )

                cached_metadata = data_manager.metadata_store[source]
                metadata_dict = cached_metadata.get("metadata", {})

                # Extract sample metadata from GEO structure
                samples_dict = metadata_dict.get("samples", {})
                if not samples_dict:
                    return f"❌ Error: No sample metadata in '{source}'. Cannot validate from metadata_store."

                # Convert to DataFrame for validation
                import pandas as pd

                sample_df = pd.DataFrame.from_dict(samples_dict, orient="index")

                # Perform validation using MetadataValidationService
                # Note: We need to import and use the validation service directly here
                from lobster.services.metadata.metadata_validation_service import (
                    MetadataValidationService,
                )

                validation_service = MetadataValidationService(
                    data_manager=data_manager
                )

                result = validation_service.validate_sample_metadata(
                    sample_df=sample_df,
                    expected_samples=expected_samples,
                    required_conditions=required_condition_list,
                    check_controls=check_controls,
                    check_duplicates=check_duplicates,
                )

                # For metadata_store, we don't have IR (no provenance tracking for cached metadata)
                ir = None
                {
                    "total_samples": len(sample_df),
                    "validation_passed": result.has_required_samples
                    and result.platform_consistency
                    and not result.duplicate_ids,
                }

            # Log provenance with IR (only for modality source_type)
            data_manager.log_tool_usage(
                tool_name="validate_dataset_content",
                parameters={
                    "source": source,
                    "source_type": source_type,
                    "expected_samples": expected_samples,
                    "required_conditions": required_conditions,
                    "check_controls": check_controls,
                    "check_duplicates": check_duplicates,
                },
                result_summary={
                    "has_required_samples": result.has_required_samples,
                    "missing_conditions": len(result.missing_conditions),
                    "duplicate_ids": len(result.duplicate_ids),
                    "platform_consistency": result.platform_consistency,
                    "warnings": len(result.warnings),
                },
                ir=ir,  # Pass IR for provenance tracking (None for metadata_store)
            )

            # Format report
            report_lines = [
                "# Dataset Validation Report\n",
                f"**Dataset:** {source}",
                f"**Source Type:** {source_type}\n",
                "## Validation Checks",
                (
                    f"✅ Sample Count: {result.summary['total_samples']} samples"
                    if result.has_required_samples
                    else f"❌ Sample Count: {result.summary['total_samples']} samples (below minimum)"
                ),
                (
                    "✅ Platform Consistency: Consistent"
                    if result.platform_consistency
                    else "⚠️ Platform Consistency: Inconsistent"
                ),
                (
                    "✅ No Duplicate IDs"
                    if not result.duplicate_ids
                    else f"❌ Duplicate IDs: {len(result.duplicate_ids)} found"
                ),
                (
                    "✅ Control Samples: Detected"
                    if not result.control_issues
                    else f"⚠️ Control Samples: {', '.join(result.control_issues)}"
                ),
            ]

            # Missing conditions
            if result.missing_conditions:
                report_lines.append("\n## Missing Required Conditions")
                for condition in result.missing_conditions:
                    report_lines.append(f"- ❌ {condition}")

            # Summary
            report_lines.append("\n## Dataset Summary")
            for key, value in result.summary.items():
                report_lines.append(f"- {key}: {value}")

            # Warnings
            if result.warnings:
                report_lines.append("\n## Warnings")
                for warning in result.warnings:
                    report_lines.append(f"- ⚠️ {warning}")

            # Recommendation
            report_lines.append("\n## Recommendation")
            if (
                result.has_required_samples
                and result.platform_consistency
                and not result.duplicate_ids
            ):
                report_lines.append(
                    "✅ **Dataset passes validation** - ready for download/analysis"
                )
            else:
                report_lines.append(
                    "⚠️ **Dataset has issues** - review warnings before proceeding"
                )

            report = "\n".join(report_lines)
            logger.info(f"Validation complete for {source}")
            return report

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return f"❌ Validation failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}", exc_info=True)
            return f"❌ Unexpected error during validation: {str(e)}"

    # =========================================================================
    # Tool 5: Filter Samples By Criteria (Microbiome/Disease)
    # =========================================================================

    @tool
    def filter_samples_by(
        workspace_key: str,
        filter_criteria: str,
        strict: bool = True
    ) -> str:
        """
        Filter samples by multi-modal criteria (16S amplicon + host organism + sample type + disease).

        Use this tool when you need to filter workspace metadata by microbiome-specific criteria:
        - 16S amplicon sequencing detection (platform, library_strategy, assay_type)
        - Host organism validation (human, mouse with fuzzy matching)
        - Sample type filtering (fecal vs tissue/biopsy)
        - Disease standardization (CRC, UC, CD, healthy controls)

        This tool applies filters IN SEQUENCE (composition pattern):
        1. Check if 16S amplicon (if requested)
        2. Validate host organism (if requested)
        3. Filter by sample type (if requested)
        4. Standardize disease terms (if requested)

        Args:
            workspace_key: Key for workspace metadata (e.g., "geo_gse123456")
            filter_criteria: Natural language criteria (e.g., "16S human fecal CRC")
            strict: Use strict matching for 16S detection (default: True)

        Returns:
            Formatted markdown report with filtering results, retention rate, and filtered metadata summary

        Examples:
            # Filter for human fecal 16S samples
            filter_samples_by(workspace_key="geo_gse123456",
                            filter_criteria="16S human fecal")

            # Filter for mouse gut tissue with disease standardization
            filter_samples_by(workspace_key="geo_gse789012",
                            filter_criteria="16S mouse gut CRC UC CD healthy")
        """
        # Check if microbiome services are available
        if not MICROBIOME_FEATURES_AVAILABLE:
            return "❌ Error: Microbiome filtering features are not available in this installation. This is an optional feature."

        try:
            logger.info(
                f"Filtering samples: workspace_key={workspace_key}, criteria='{filter_criteria}', strict={strict}"
            )

            # Parse natural language criteria
            parsed_criteria = _parse_filter_criteria(filter_criteria)

            logger.debug(f"Parsed criteria: {parsed_criteria}")

            # Read workspace metadata via WorkspaceContentService
            from lobster.services.data_access.workspace_content_service import WorkspaceContentService
            workspace_service = WorkspaceContentService(data_manager)
            workspace_data = workspace_service.read_content(workspace_key)
            if not workspace_data:
                return f"❌ Error: Workspace key '{workspace_key}' not found or empty"

            # Extract metadata (assume dict structure with 'metadata' or 'samples')
            if isinstance(workspace_data, dict):
                if "metadata" in workspace_data:
                    metadata_dict = workspace_data["metadata"].get("samples", {})
                elif "samples" in workspace_data:
                    metadata_dict = workspace_data["samples"]
                else:
                    metadata_dict = workspace_data
            else:
                return f"❌ Error: Unexpected workspace data format"

            if not metadata_dict:
                return f"❌ Error: No sample metadata found in workspace key '{workspace_key}'"

            # Convert to DataFrame
            import pandas as pd
            metadata_df = pd.DataFrame.from_dict(metadata_dict, orient="index")
            original_count = len(metadata_df)

            logger.debug(f"Loaded {original_count} samples from workspace")

            # Apply filters in sequence
            irs = []
            stats_list = []
            current_metadata = metadata_df.copy()

            # Filter 1: 16S amplicon detection
            if parsed_criteria["check_16s"]:
                logger.info("Applying 16S amplicon filter...")
                filtered_rows = []
                for idx, row in current_metadata.iterrows():
                    row_dict = row.to_dict()
                    filtered, stats, ir = microbiome_filtering_service.validate_16s_amplicon(
                        row_dict, strict=strict
                    )
                    if filtered:  # Non-empty dict means valid
                        filtered_rows.append(idx)
                    if stats["is_valid"]:
                        irs.append(ir)
                        stats_list.append(stats)

                current_metadata = current_metadata.loc[filtered_rows]
                logger.debug(f"After 16S filter: {len(current_metadata)} samples retained")

            # Filter 2: Host organism validation
            if parsed_criteria["host_organisms"]:
                logger.debug(f"Applying host organism filter: {parsed_criteria['host_organisms']}")
                filtered_rows = []
                for idx, row in current_metadata.iterrows():
                    row_dict = row.to_dict()
                    filtered, stats, ir = microbiome_filtering_service.validate_host_organism(
                        row_dict, allowed_hosts=parsed_criteria["host_organisms"]
                    )
                    if filtered:  # Non-empty dict means valid
                        filtered_rows.append(idx)
                    if stats["is_valid"]:
                        irs.append(ir)
                        stats_list.append(stats)

                current_metadata = current_metadata.loc[filtered_rows]
                logger.debug(f"After host filter: {len(current_metadata)} samples retained")

            # Filter 3: Sample type filtering
            if parsed_criteria["sample_types"]:
                logger.debug(f"Applying sample type filter: {parsed_criteria['sample_types']}")
                filtered, stats, ir = disease_standardization_service.filter_by_sample_type(
                    current_metadata, sample_types=parsed_criteria["sample_types"]
                )
                current_metadata = filtered
                irs.append(ir)
                stats_list.append(stats)
                logger.debug(f"After sample type filter: {len(current_metadata)} samples retained")

            # Filter 4: Disease standardization
            if parsed_criteria["standardize_disease"]:
                logger.debug("Applying disease standardization...")
                # Find disease column (try common names)
                disease_col = None
                for col_name in ["disease", "condition", "diagnosis", "disease_state"]:
                    if col_name in current_metadata.columns:
                        disease_col = col_name
                        break

                if disease_col:
                    standardized, stats, ir = disease_standardization_service.standardize_disease_terms(
                        current_metadata, disease_column=disease_col
                    )
                    current_metadata = standardized
                    irs.append(ir)
                    stats_list.append(stats)
                    logger.debug(f"Disease standardization complete: {stats['standardization_rate']:.1f}% mapped")
                else:
                    logger.warning("Disease standardization requested but no disease column found")

            # Calculate final stats
            final_count = len(current_metadata)
            retention_rate = (final_count / original_count * 100) if original_count > 0 else 0

            # Combine IRs into composite IR
            composite_ir = _combine_analysis_steps(
                irs,
                operation="filter_samples_by",
                description=f"Multi-criteria filtering: {filter_criteria}"
            )

            # Log tool usage
            data_manager.log_tool_usage(
                tool_name="filter_samples_by",
                parameters={
                    "workspace_key": workspace_key,
                    "filter_criteria": filter_criteria,
                    "strict": strict,
                    "parsed_criteria": parsed_criteria
                },
                result_summary={
                    "original_samples": original_count,
                    "filtered_samples": final_count,
                    "retention_rate": retention_rate,
                    "filters_applied": len(irs)
                },
                ir=composite_ir
            )

            # Format report
            report = _format_filtering_report(
                workspace_key=workspace_key,
                filter_criteria=filter_criteria,
                parsed_criteria=parsed_criteria,
                original_count=original_count,
                final_count=final_count,
                retention_rate=retention_rate,
                stats_list=stats_list,
                filtered_metadata=current_metadata
            )

            logger.debug(f"Filtering complete: {final_count}/{original_count} samples retained ({retention_rate:.1f}%)")
            return report

        except ValueError as e:
            logger.error(f"Filtering error: {e}")
            return f"❌ Filtering failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected filtering error: {e}", exc_info=True)
            return f"❌ Unexpected error during filtering: {str(e)}"

    # =========================================================================
    # Phase 4c NEW TOOLS: Publication Queue Processing (3 tools)
    # =========================================================================

    # Import shared workspace tools and services
    from lobster.tools.workspace_tool import (
        create_get_content_from_workspace_tool,
        create_write_to_workspace_tool,
    )
    from lobster.services.data_access.workspace_content_service import WorkspaceContentService

    workspace_service = WorkspaceContentService(data_manager=data_manager)

    # Create shared workspace tools
    get_content_from_workspace = create_get_content_from_workspace_tool(data_manager)
    write_to_workspace = create_write_to_workspace_tool(data_manager)

    @tool
    def process_metadata_entry(
        entry_id: str,
        filter_criteria: str = None,
        standardize_schema: str = None,
    ) -> str:
        """
        Process a single publication queue entry for metadata filtering/standardization.

        Reads workspace_metadata_keys from the entry, applies filters if specified,
        and stores results back to harmonization_metadata.

        Args:
            entry_id: Publication queue entry ID
            filter_criteria: Optional natural language filter (e.g., "16S human fecal")
            standardize_schema: Optional schema to standardize to (e.g., "microbiome")

        Returns:
            Processing summary with sample counts and filtered metadata location
        """
        try:
            queue = data_manager.publication_queue
            entry = queue.get_entry(entry_id)

            if not entry.workspace_metadata_keys:
                error_msg = (
                    f"Entry {entry_id} has no workspace_metadata_keys. "
                    f"Expected at least one 'sra_*_samples' key. "
                    f"Run research_agent.process_publication_queue() to populate workspace keys."
                )
                logger.error(error_msg)
                return f"❌ Error: {error_msg}"

            # Log entry processing start
            logger.info(
                f"Processing entry {entry_id}: '{entry.title or 'No title'}' "
                f"({len(entry.workspace_metadata_keys)} workspace keys)"
            )

            # Update status to in_progress
            queue.update_status(
                entry_id, entry.status, handoff_status=HandoffStatus.METADATA_IN_PROGRESS
            )

            # Read and aggregate samples from all workspace keys
            all_samples = []
            all_validation_results = []

            for ws_key in entry.workspace_metadata_keys:
                # Only process SRA sample files (skip pub_queue_*_metadata/methods/identifiers.json)
                if not (ws_key.startswith("sra_") and ws_key.endswith("_samples")):
                    logger.debug(f"Skipping non-SRA workspace key: {ws_key}")
                    continue

                logger.debug(f"Processing SRA sample file: {ws_key}")

                from lobster.services.data_access.workspace_content_service import (
                    ContentType,
                )

                ws_data = workspace_service.read_content(
                    ws_key, content_type=ContentType.METADATA
                )
                if ws_data:
                    samples, validation_result, quality_stats = _extract_samples_from_workspace(ws_data)
                    all_samples.extend(samples)
                    all_validation_results.append(validation_result)

                    # Log extraction results with quality info
                    logger.debug(
                        f"Extracted {len(samples)} valid samples from {ws_key} "
                        f"(validation_rate: {validation_result.metadata.get('validation_rate', 0):.1f}%, "
                        f"avg_completeness: {quality_stats.get('avg_completeness', 0):.1f}, "
                        f"unique_individuals: {quality_stats.get('unique_individuals', 0)})"
                    )

            samples_before = sum(
                vr.metadata.get("total_samples", 0) for vr in all_validation_results
            )

            # Check for complete validation failure
            if samples_before > 0 and len(all_samples) == 0:
                # All samples failed validation
                total_errors = sum(len(vr.errors) for vr in all_validation_results)
                error_msg = (
                    f"All {samples_before} samples failed validation ({total_errors} errors). "
                    f"Check workspace files for data quality issues."
                )
                logger.error(f"Entry {entry_id}: {error_msg}")

                # Log first few validation errors for debugging
                logger.error(f"First validation errors for entry {entry_id}:")
                for vr in all_validation_results[:3]:  # First 3 workspace keys
                    for error in vr.errors[:3]:  # First 3 errors per key
                        logger.error(f"  - {error}")

                # Mark as METADATA_FAILED
                queue.update_status(
                    entry_id,
                    entry.status,
                    handoff_status=HandoffStatus.METADATA_FAILED,
                    error=error_msg,
                )

                return f"❌ {error_msg}"

            # Apply filters if specified
            if filter_criteria and all_samples:
                logger.debug(
                    f"Applying filter criteria to {len(all_samples)} samples: '{filter_criteria}'"
                )
                all_samples = _apply_metadata_filters(all_samples, filter_criteria)
                logger.debug(f"After filtering: {len(all_samples)} samples retained")

            samples_after = len(all_samples)

            # Add publication context
            for sample in all_samples:
                sample["publication_entry_id"] = entry_id
                sample["publication_title"] = entry.title
                sample["publication_doi"] = entry.doi
                sample["publication_pmid"] = entry.pmid

            # Aggregate validation statistics
            total_errors = sum(len(vr.errors) for vr in all_validation_results)
            total_warnings = sum(len(vr.warnings) for vr in all_validation_results)

            # Log overall validation summary
            validation_rate = (
                (len(all_samples) / samples_before * 100) if samples_before > 0 else 0
            )
            logger.debug(
                f"Entry {entry_id} validation summary: "
                f"{len(all_samples)}/{samples_before} samples valid ({validation_rate:.1f}%), "
                f"{total_errors} errors, {total_warnings} warnings"
            )

            # Store in harmonization_metadata
            harmonization_data = {
                "samples": all_samples,
                "filter_criteria": filter_criteria,
                "standardize_schema": standardize_schema,
                "stats": {
                    "samples_extracted": samples_before,
                    "samples_valid": len(all_samples),
                    "samples_after_filter": samples_after,
                    "validation_errors": total_errors,
                    "validation_warnings": total_warnings,
                },
            }

            queue.update_status(
                entry_id=entry_id,
                status=entry.status,
                handoff_status=HandoffStatus.METADATA_COMPLETE,
                harmonization_metadata=harmonization_data,
            )

            retention = (samples_after / samples_before * 100) if samples_before > 0 else 0

            # Build response with validation info
            response = f"""## Entry Processed: {entry_id}
**Samples Extracted**: {samples_before}
**Samples Valid**: {len(all_samples)}
**Samples After Filter**: {samples_after}
**Retention**: {retention:.1f}%
**Filter**: {filter_criteria or 'None'}
**Validation**: {total_errors} errors, {total_warnings} warnings
"""
            # Include validation messages if there are issues
            if total_errors > 0 or total_warnings > 0:
                response += "\n### Validation Summary\n"
                for idx, vr in enumerate(all_validation_results):
                    if vr.has_errors or vr.has_warnings:
                        response += f"\n**Workspace key {idx+1}**:\n"
                        response += vr.format_messages(include_info=False) + "\n"

            return response
        except Exception as e:
            logger.error(f"Error processing entry {entry_id}: {e}")
            return f"❌ Error processing entry: {str(e)}"

    @tool
    def process_metadata_queue(
        status_filter: str = "handoff_ready",
        filter_criteria: str = None,
        max_entries: int = None,
        output_key: str = "aggregated_filtered_samples",
    ) -> str:
        """
        Process multiple publication queue entries and aggregate results.

        Iterates through entries matching status_filter, processes each,
        and aggregates all filtered samples into a single workspace artifact.

        Args:
            status_filter: Queue status to filter (default: "handoff_ready")
            filter_criteria: Natural language filter (e.g., "16S human fecal CRC")
            max_entries: Maximum entries to process (None = all)
            output_key: Workspace key for aggregated output

        Returns:
            Processing summary with total counts and output location
        """
        try:
            from lobster.services.data_access.workspace_content_service import (
                MetadataContent,
                ContentType,
            )

            queue = data_manager.publication_queue
            entries = queue.list_entries(status=PublicationStatus(status_filter.lower()))

            if max_entries:
                entries = entries[:max_entries]

            if not entries:
                logger.info(f"No entries found with status '{status_filter}'")
                return f"No entries found with status '{status_filter}'"

            # Log batch processing start
            logger.info(
                f"Starting batch processing: {len(entries)} entries "
                f"(filter: '{filter_criteria or 'none'}')"
            )

            all_samples = []
            stats = {
                "processed": 0,
                "with_samples": 0,
                "failed": 0,
                "total_extracted": 0,
                "total_valid": 0,
                "total_after_filter": 0,
                "validation_errors": 0,
                "validation_warnings": 0,
                "flag_counts": {},  # Aggregate quality flags across all samples
                "samples_needing_manual_review": [],  # Sample IDs needing body_site review
            }
            failed_entries = []

            for entry in entries:
                try:
                    if not entry.workspace_metadata_keys:
                        logger.debug(
                            f"Skipping entry {entry.entry_id}: no workspace_metadata_keys"
                        )
                        continue

                    logger.debug(
                        f"Processing entry {entry.entry_id}: '{entry.title or 'No title'}'"
                    )

                    entry_samples = []
                    entry_validation_results = []
                    entry_quality_stats = []

                    for ws_key in entry.workspace_metadata_keys:
                        # Only process SRA sample files (skip pub_queue_*_metadata/methods/identifiers.json)
                        if not (ws_key.startswith("sra_") and ws_key.endswith("_samples")):
                            logger.debug(f"Skipping non-SRA workspace key: {ws_key}")
                            continue

                        logger.debug(
                            f"Processing SRA sample file: {ws_key} for entry {entry.entry_id}"
                        )

                        from lobster.services.data_access.workspace_content_service import (
                            ContentType,
                        )

                        ws_data = workspace_service.read_content(
                            ws_key, content_type=ContentType.METADATA
                        )
                        if ws_data:
                            samples, validation_result, quality_stats = _extract_samples_from_workspace(
                                ws_data
                            )
                            entry_samples.extend(samples)
                            entry_validation_results.append(validation_result)
                            entry_quality_stats.append(quality_stats)

                    # Update stats with validation info
                    samples_extracted = sum(
                        vr.metadata.get("total_samples", 0)
                        for vr in entry_validation_results
                    )
                    samples_valid = len(entry_samples)

                    # Aggregate quality stats for this entry
                    entry_unique_individuals = set()
                    entry_flag_counts = {}
                    for qs in entry_quality_stats:
                        # Extract individual_id strings (not the full sample dicts)
                        for s in entry_samples:
                            ind_id = s.get("_individual_id")
                            if ind_id:
                                entry_unique_individuals.add(ind_id)
                        for flag, count in qs.get("flag_counts", {}).items():
                            entry_flag_counts[flag] = entry_flag_counts.get(flag, 0) + count

                    # Check for complete validation failure
                    if samples_extracted > 0 and samples_valid == 0:
                        # All samples in this entry failed validation
                        error_count = sum(
                            len(vr.errors) for vr in entry_validation_results
                        )
                        error_msg = f"All {samples_extracted} samples failed validation ({error_count} errors)"
                        logger.error(f"Entry {entry.entry_id}: {error_msg}")

                        # Mark entry as failed
                        queue.update_status(
                            entry.entry_id,
                            entry.status,
                            handoff_status=HandoffStatus.METADATA_FAILED,
                            error=error_msg,
                        )
                        failed_entries.append((entry.entry_id, error_msg))
                        stats["failed"] += 1
                        stats["processed"] += 1
                        continue

                    stats["total_extracted"] += samples_extracted
                    stats["total_valid"] += samples_valid
                    stats["validation_errors"] += sum(
                        len(vr.errors) for vr in entry_validation_results
                    )
                    stats["validation_warnings"] += sum(
                        len(vr.warnings) for vr in entry_validation_results
                    )

                    logger.debug(
                        f"Entry {entry.entry_id}: {samples_valid}/{samples_extracted} samples valid"
                    )

                    if filter_criteria and entry_samples:
                        samples_before_filter = len(entry_samples)
                        entry_samples = _apply_metadata_filters(
                            entry_samples, filter_criteria
                        )
                        logger.debug(
                            f"Entry {entry.entry_id}: Filter applied - "
                            f"{len(entry_samples)}/{samples_before_filter} samples retained"
                        )

                    stats["total_after_filter"] += len(entry_samples)

                    # Aggregate quality flags into batch stats
                    for flag, count in entry_flag_counts.items():
                        stats["flag_counts"][flag] = stats["flag_counts"].get(flag, 0) + count

                    # Track samples needing manual body_site review
                    for qs in entry_quality_stats:
                        flagged_ids = qs.get("flagged_sample_ids", {})
                        if "missing_body_site" in flagged_ids:
                            stats["samples_needing_manual_review"].extend(flagged_ids["missing_body_site"])

                    # Add publication context
                    for sample in entry_samples:
                        sample["publication_entry_id"] = entry.entry_id
                        sample["publication_title"] = entry.title
                        sample["publication_doi"] = entry.doi
                        sample["publication_pmid"] = entry.pmid

                    all_samples.extend(entry_samples)
                    stats["processed"] += 1
                    if entry_samples:
                        stats["with_samples"] += 1

                    # Update entry status
                    queue.update_status(
                        entry.entry_id,
                        entry.status,
                        handoff_status=HandoffStatus.METADATA_COMPLETE,
                    )

                except Exception as e:
                    # Graceful degradation: log error but continue with other entries
                    error_msg = f"Failed to process entry {entry.entry_id}: {str(e)}"
                    logger.error(error_msg, exc_info=True)

                    # Mark entry as failed
                    queue.update_status(
                        entry.entry_id,
                        entry.status,
                        handoff_status=HandoffStatus.METADATA_FAILED,
                        error=error_msg,
                    )
                    failed_entries.append((entry.entry_id, error_msg))
                    stats["failed"] += 1
                    stats["processed"] += 1

            # Store aggregated results to workspace
            if all_samples:
                from datetime import datetime

                content = MetadataContent(
                    identifier=output_key,
                    content_type="filtered_samples",
                    description=f"Batch filtered samples: {filter_criteria or 'no filter'}",
                    data={
                        "samples": all_samples,
                        "filter_criteria": filter_criteria,
                        "stats": stats,
                    },
                    source="metadata_assistant",
                    cached_at=datetime.now().isoformat(),
                )
                workspace_service.write_content(content, ContentType.METADATA)

                # Also store in metadata_store for write_to_workspace access
                data_manager.metadata_store[output_key] = {
                    "samples": all_samples,
                    "filter_criteria": filter_criteria,
                    "stats": stats,
                }

            retention = (
                (stats["total_after_filter"] / stats["total_extracted"] * 100)
                if stats["total_extracted"] > 0
                else 0
            )
            validation_rate = (
                (stats["total_valid"] / stats["total_extracted"] * 100)
                if stats["total_extracted"] > 0
                else 0
            )

            # Log comprehensive batch summary (changed to DEBUG to reduce console pollution - summary is in tool response)
            logger.debug("=" * 60)
            logger.debug(f"Batch processing complete for {len(entries)} entries")
            logger.debug(f"  Successful: {stats['processed'] - stats['failed']}")
            logger.debug(f"  Failed: {stats['failed']}")
            logger.debug(f"  Total samples extracted: {stats['total_extracted']}")
            logger.debug(f"  Total samples valid: {stats['total_valid']} ({validation_rate:.1f}%)")
            logger.debug(
                f"  Total samples after filter: {stats['total_after_filter']} ({retention:.1f}%)"
            )
            logger.debug(f"  Validation errors: {stats['validation_errors']}")
            logger.debug(f"  Validation warnings: {stats['validation_warnings']}")
            if failed_entries:
                logger.warning(f"Failed entries ({len(failed_entries)}):")
                for entry_id, error in failed_entries:
                    logger.warning(f"  - {entry_id}: {error}")
            logger.debug("=" * 60)

            # Build response
            response = f"""## Queue Processing Complete
**Entries Processed**: {stats['processed']}
**Successful**: {stats['processed'] - stats['failed']}
**Failed**: {stats['failed']}
**Entries With Samples**: {stats['with_samples']}
**Samples Extracted**: {stats['total_extracted']}
**Samples Valid**: {stats['total_valid']} ({validation_rate:.1f}%)
**Samples After Filter**: {stats['total_after_filter']}
**Retention Rate**: {retention:.1f}%
**Validation**: {stats['validation_errors']} errors, {stats['validation_warnings']} warnings
**Output Key**: {output_key}
"""

            # Add failed entries section if any
            if failed_entries:
                response += "\n### Failed Entries\n"
                for entry_id, error in failed_entries:
                    response += f"- {entry_id}: {error}\n"

            # Add manual inspection summary for samples missing body_site info
            samples_needing_review = stats.get("samples_needing_manual_review", [])
            if samples_needing_review:
                response += f"\n### ⚠️ Manual Inspection Needed ({len(samples_needing_review)} samples)\n"
                response += "The following samples are missing body_site/tissue type metadata and may require manual verification:\n"
                # Show up to 10 sample IDs, then summarize
                if len(samples_needing_review) <= 10:
                    for sample_id in samples_needing_review:
                        response += f"- {sample_id}\n"
                else:
                    for sample_id in samples_needing_review[:10]:
                        response += f"- {sample_id}\n"
                    response += f"- ... and {len(samples_needing_review) - 10} more\n"
                response += "\n**Recommended action**: Check the original publication methods section or SRA metadata for sample type information (fecal, tissue, oral, etc.).\n"

            # Add quality flag summary if present
            flag_counts = stats.get("flag_counts", {})
            if flag_counts:
                response += "\n### Quality Flag Summary\n"
                for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
                    response += f"- {flag}: {count} samples\n"

            response += f"\nUse `write_to_workspace(identifier=\"{output_key}\", workspace=\"metadata\", output_format=\"csv\")` to export as CSV.\n"

            return response
        except Exception as e:
            logger.error(f"Error processing queue: {e}")
            return f"❌ Error processing queue: {str(e)}"

    @tool
    def update_metadata_status(
        entry_id: str,
        handoff_status: str = None,
        error_message: str = None,
    ) -> str:
        """
        Manually update metadata processing status for a queue entry.

        Use this to mark entries as complete, reset failed entries, or add error notes.

        Args:
            entry_id: Publication queue entry ID
            handoff_status: New handoff status (not_ready, ready_for_metadata, metadata_in_progress, metadata_complete)
            error_message: Optional error message to log

        Returns:
            Confirmation of status update
        """
        try:
            queue = data_manager.publication_queue
            entry = queue.get_entry(entry_id)

            update_kwargs = {"entry_id": entry_id, "status": entry.status}

            if handoff_status:
                update_kwargs["handoff_status"] = HandoffStatus(handoff_status)
            if error_message:
                update_kwargs["error"] = error_message

            queue.update_status(**update_kwargs)

            return f"✓ Updated {entry_id}: handoff_status={handoff_status or 'unchanged'}"
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            return f"❌ Error updating status: {str(e)}"

    # Helper functions for queue processing
    def _extract_samples_from_workspace(
        ws_data,
    ) -> tuple[list, "ValidationResult", dict]:
        """
        Extract sample records from workspace data with validation and quality flagging.

        Uses SRASampleSchema for validation and compute_sample_completeness for
        soft flagging. All valid samples are returned with quality flags attached
        (soft filtering - user decides what to exclude).

        Handles various data structures produced by WorkspaceContentService:
        - Nested: {"data": {"samples": [...]}}
        - Direct: {"samples": [...]}
        - Dict-based: {"samples": {"id1": {...}, "id2": {...}}}

        Quality Flagging:
        - Each sample gets _quality_score (0-100) and _quality_flags fields
        - Flags indicate potential concerns (missing individual_id, controls, etc.)
        - Samples are NOT filtered out - user reviews flags and decides

        Args:
            ws_data: Workspace data dictionary (from WorkspaceContentService)

        Returns:
            (valid_samples, validation_result, quality_stats): Tuple containing:
            - valid_samples: List of valid samples with _quality_score and _quality_flags
            - validation_result: ValidationResult with batch statistics
            - quality_stats: Dict with completeness stats and flag counts

        Examples:
            >>> ws_data = {"data": {"samples": [{"run_accession": "SRR001", ...}]}}
            >>> samples, result, stats = _extract_samples_from_workspace(ws_data)
            >>> len(samples)  # All valid samples (not filtered)
            1
            >>> samples[0]["_quality_score"]  # Completeness score
            80.0
            >>> stats["unique_individuals"]
            45
        """
        from lobster.core.schemas.sra import (
            SRASampleSchema,
            compute_sample_completeness,
            validate_sra_sample,
            validate_sra_samples_batch,
        )

        # Extract raw samples from workspace structure
        raw_samples = []
        if isinstance(ws_data, dict):
            if "samples" in ws_data:
                samples = ws_data["samples"]
                if isinstance(samples, list):
                    raw_samples = samples
                elif isinstance(samples, dict):
                    # Convert dict to list of samples
                    raw_samples = [{"sample_id": k, **v} for k, v in samples.items()]
            if "data" in ws_data and isinstance(ws_data["data"], dict):
                # Recursive extraction for nested structure
                nested_samples, nested_result, _ = _extract_samples_from_workspace(
                    ws_data["data"]
                )
                raw_samples.extend(nested_samples)

        # If no samples extracted, return empty results
        if not raw_samples:
            empty_result = ValidationResult()
            empty_result.metadata["total_samples"] = 0
            empty_result.metadata["valid_samples"] = 0
            empty_result.metadata["validation_rate"] = 0.0
            empty_stats = {
                "total_samples": 0,
                "avg_completeness": 0.0,
                "unique_individuals": 0,
                "completeness_distribution": {"high": [], "medium": [], "low": []},
                "flag_counts": {},
            }
            return [], empty_result, empty_stats

        # Validate extracted samples using unified schema system
        validation_result = validate_sra_samples_batch(raw_samples)

        # Process valid samples with quality flagging (soft filtering)
        valid_samples = []
        quality_stats = {
            "total_samples": 0,
            "avg_completeness": 0.0,
            "unique_individuals": 0,
            "completeness_distribution": {"high": [], "medium": [], "low": []},
            "flag_counts": {},
            "flagged_sample_ids": {},
            "individuals": set(),
        }
        scores = []

        for sample in raw_samples:
            sample_result = validate_sra_sample(sample)
            if sample_result.is_valid:  # No critical errors
                # Validate and compute completeness
                try:
                    validated = SRASampleSchema.from_dict(sample)
                    score, flags = compute_sample_completeness(validated)

                    # Attach quality info to sample (soft flagging)
                    sample["_quality_score"] = score
                    sample["_quality_flags"] = [f.value for f in flags]

                    # Also attach heuristically extracted fields for downstream use
                    if validated.individual_id:
                        sample["_individual_id"] = validated.individual_id
                    if validated.timepoint:
                        sample["_timepoint"] = validated.timepoint
                    if validated.timepoint_numeric is not None:
                        sample["_timepoint_numeric"] = validated.timepoint_numeric

                    valid_samples.append(sample)
                    scores.append(score)

                    # Track statistics
                    if validated.individual_id:
                        quality_stats["individuals"].add(validated.individual_id)

                    # Categorize by completeness
                    run_acc = validated.run_accession
                    if score >= 80:
                        quality_stats["completeness_distribution"]["high"].append(
                            run_acc
                        )
                    elif score >= 50:
                        quality_stats["completeness_distribution"]["medium"].append(
                            run_acc
                        )
                    else:
                        quality_stats["completeness_distribution"]["low"].append(
                            run_acc
                        )

                    # Track flag counts
                    for flag in flags:
                        flag_name = flag.value
                        quality_stats["flag_counts"][flag_name] = (
                            quality_stats["flag_counts"].get(flag_name, 0) + 1
                        )
                        if flag_name not in quality_stats["flagged_sample_ids"]:
                            quality_stats["flagged_sample_ids"][flag_name] = []
                        quality_stats["flagged_sample_ids"][flag_name].append(run_acc)

                except Exception as e:
                    logger.warning(f"Error computing completeness for sample: {e}")
                    sample["_quality_score"] = 0.0
                    sample["_quality_flags"] = ["validation_error"]
                    valid_samples.append(sample)
                    scores.append(0.0)

                # Log validation warnings at DEBUG level (avoids console spam)
                # These warnings are already tracked in quality_stats["flag_counts"]
                for warning in sample_result.warnings:
                    logger.debug(warning)
            else:
                # Log errors for invalid samples (these are excluded)
                for error in sample_result.errors:
                    logger.error(error)

        # Finalize quality stats
        quality_stats["total_samples"] = len(valid_samples)
        quality_stats["avg_completeness"] = sum(scores) / len(scores) if scores else 0.0
        quality_stats["unique_individuals"] = len(quality_stats["individuals"])
        # Convert set to count (can't serialize set to JSON)
        del quality_stats["individuals"]

        logger.debug(
            f"Sample extraction complete: {len(valid_samples)}/{len(raw_samples)} valid "
            f"({validation_result.metadata.get('validation_rate', 0):.1f}%)"
        )
        logger.debug(
            f"Quality summary: avg_completeness={quality_stats['avg_completeness']:.1f}, "
            f"unique_individuals={quality_stats['unique_individuals']}, "
            f"high/medium/low={len(quality_stats['completeness_distribution']['high'])}/"
            f"{len(quality_stats['completeness_distribution']['medium'])}/"
            f"{len(quality_stats['completeness_distribution']['low'])}"
        )
        if quality_stats["flag_counts"]:
            logger.debug(f"Flag counts: {quality_stats['flag_counts']}")

        return valid_samples, validation_result, quality_stats

    def _apply_metadata_filters(samples: list, filter_criteria: str) -> list:
        """Apply natural language filters using microbiome services."""
        if not MICROBIOME_FEATURES_AVAILABLE:
            logger.warning("Microbiome services not available, returning unfiltered samples")
            return samples

        parsed = _parse_filter_criteria(filter_criteria)
        filtered = samples.copy()

        if parsed["check_16s"]:
            filtered = [s for s in filtered if microbiome_filtering_service.validate_16s_amplicon(s, strict=False)[0]]
        if parsed["host_organisms"]:
            filtered = [s for s in filtered if microbiome_filtering_service.validate_host_organism(s, allowed_hosts=parsed["host_organisms"])[0]]

        return filtered

    # =========================================================================
    # Helper Functions for filter_samples_by
    # =========================================================================

    def _parse_filter_criteria(criteria: str) -> Dict[str, Any]:
        """
        Parse natural language filter criteria into structured format.

        Args:
            criteria: Natural language string (e.g., "16S human fecal CRC UC")

        Returns:
            Dictionary with parsed criteria:
            {
                "check_16s": bool,
                "host_organisms": List[str],
                "sample_types": List[str],
                "standardize_disease": bool
            }
        """
        criteria_lower = criteria.lower()

        # Check for 16S amplicon
        check_16s = "16s" in criteria_lower or "amplicon" in criteria_lower

        # Check for host organisms
        host_organisms = []
        if "human" in criteria_lower or "homo sapiens" in criteria_lower:
            host_organisms.append("Human")
        if "mouse" in criteria_lower or "mus musculus" in criteria_lower:
            host_organisms.append("Mouse")

        # Check for sample types
        sample_types = []
        if "fecal" in criteria_lower or "stool" in criteria_lower or "feces" in criteria_lower:
            sample_types.append("fecal")
        if "gut" in criteria_lower or "tissue" in criteria_lower or "biopsy" in criteria_lower:
            sample_types.append("gut")

        # Check for disease terms (if any disease keywords present, enable standardization)
        disease_keywords = ["crc", "uc", "cd", "cancer", "colitis", "crohn", "healthy", "control"]
        standardize_disease = any(keyword in criteria_lower for keyword in disease_keywords)

        return {
            "check_16s": check_16s,
            "host_organisms": host_organisms,
            "sample_types": sample_types,
            "standardize_disease": standardize_disease
        }

    def _combine_analysis_steps(irs: List[AnalysisStep], operation: str, description: str) -> AnalysisStep:
        """
        Combine multiple AnalysisSteps into a composite IR.

        Args:
            irs: List of individual AnalysisSteps
            operation: Composite operation name
            description: Composite description

        Returns:
            Composite AnalysisStep
        """
        if not irs:
            # Return minimal IR if no steps
            return AnalysisStep(
                operation=operation,
                tool_name="filter_samples_by",
                description=description,
                library="lobster.agents.metadata_assistant",
                code_template="# No filtering operations applied",
                imports=[],
                parameters={},
                parameter_schema={},
                input_entities=[],
                output_entities=[]
            )

        # Combine code templates
        combined_code = "\n\n".join([
            f"# Step {i+1}: {ir.description}\n{ir.code_template}"
            for i, ir in enumerate(irs)
        ])

        # Combine imports (deduplicate)
        all_imports = []
        for ir in irs:
            all_imports.extend(ir.imports)
        combined_imports = list(set(all_imports))

        # Combine parameters
        combined_params = {}
        for i, ir in enumerate(irs):
            combined_params[f"step_{i+1}"] = ir.parameters

        return AnalysisStep(
            operation=operation,
            tool_name="filter_samples_by",
            description=description,
            library="lobster.agents.metadata_assistant",
            code_template=combined_code,
            imports=combined_imports,
            parameters=combined_params,
            parameter_schema={},
            input_entities=[{"type": "workspace_metadata", "name": "input_metadata"}],
            output_entities=[{"type": "filtered_metadata", "name": "output_metadata"}]
        )

    def _format_filtering_report(
        workspace_key: str,
        filter_criteria: str,
        parsed_criteria: Dict[str, Any],
        original_count: int,
        final_count: int,
        retention_rate: float,
        stats_list: List[Dict[str, Any]],
        filtered_metadata: "pd.DataFrame"
    ) -> str:
        """
        Format filtering report as markdown.

        Args:
            workspace_key: Workspace key
            filter_criteria: Original criteria string
            parsed_criteria: Parsed criteria dict
            original_count: Original sample count
            final_count: Final sample count
            retention_rate: Retention percentage
            stats_list: List of per-filter statistics
            filtered_metadata: Final filtered DataFrame

        Returns:
            Markdown-formatted report
        """
        status_icon = "✅" if retention_rate > 50 else ("⚠️" if retention_rate > 10 else "❌")

        report_lines = [
            f"{status_icon} Sample Filtering Complete\n",
            f"**Workspace Key**: {workspace_key}",
            f"**Filter Criteria**: {filter_criteria}",
            f"**Original Samples**: {original_count}",
            f"**Filtered Samples**: {final_count}",
            f"**Retention Rate**: {retention_rate:.1f}%\n",
            "## Filters Applied\n"
        ]

        # List applied filters
        if parsed_criteria["check_16s"]:
            report_lines.append("- ✓ 16S amplicon detection")
        if parsed_criteria["host_organisms"]:
            report_lines.append(f"- ✓ Host organism: {', '.join(parsed_criteria['host_organisms'])}")
        if parsed_criteria["sample_types"]:
            report_lines.append(f"- ✓ Sample type: {', '.join(parsed_criteria['sample_types'])}")
        if parsed_criteria["standardize_disease"]:
            report_lines.append("- ✓ Disease standardization")

        # Add per-filter statistics
        if stats_list:
            report_lines.append("\n## Filter Statistics\n")
            for i, stats in enumerate(stats_list, 1):
                report_lines.append(f"**Filter {i}**:")
                for key, value in stats.items():
                    if key not in ["is_valid"]:  # Skip boolean flags
                        report_lines.append(f"  - {key}: {value}")

        # Add sample preview
        report_lines.append("\n## Filtered Metadata Preview\n")
        if not filtered_metadata.empty:
            # Show first 5 samples
            preview = filtered_metadata.head(5).to_markdown(index=True)
            report_lines.append(preview)
            if len(filtered_metadata) > 5:
                report_lines.append(f"\n*... and {len(filtered_metadata) - 5} more samples*")
        else:
            report_lines.append("*No samples passed filtering criteria*")

        # Add recommendation
        report_lines.append("\n## Recommendation\n")
        if retention_rate > 50:
            report_lines.append(
                f"✅ **Good retention rate** ({retention_rate:.1f}%) - Filtered dataset is suitable for analysis"
            )
        elif retention_rate > 10:
            report_lines.append(
                f"⚠️ **Moderate retention rate** ({retention_rate:.1f}%) - Consider relaxing filter criteria or verify input data quality"
            )
        else:
            report_lines.append(
                f"❌ **Low retention rate** ({retention_rate:.1f}%) - Most samples filtered out. Review criteria or input data"
            )

        return "\n".join(report_lines)

    # =========================================================================
    # Tool 10: Custom Code Execution for Sample-Level Operations
    # =========================================================================

    from typing import Optional

    @tool
    def execute_custom_code(
        python_code: str,
        workspace_key: str = None,
        load_workspace_files: bool = True,
        persist: bool = False,
        description: str = "Custom metadata code execution"
    ) -> str:
        """
        Execute custom Python code for sample-level metadata operations.

        **Use this tool for manual review and filtering of samples when standard tools are insufficient.**

        AVAILABLE IN NAMESPACE:
        - workspace_path: Path to workspace directory
        - Auto-loaded CSV/JSON files from workspace/metadata/ (including samples files)
        - publication_queue: Publication queue entries
        - All JSON files loaded as Python dicts with sanitized names (dots/dashes → underscores)

        COMMON USE CASES:
        1. Filter flagged samples:
           `samples = [s for s in aggregated_filtered_samples_16s_human['data']['samples'] if s.get('body_site')]`

        2. Update sample metadata:
           `for s in samples: s['manually_reviewed'] = True`

        3. Extract patterns from publication text:
           `import re; body_site = re.search(r'(fecal|stool|biopsy)', methods_text)`

        PERSISTING CHANGES:
        To save filtered samples for export, return a dict with:
        - 'samples': list of sample dicts to persist
        - 'output_key': key name for metadata_store (enables write_to_workspace export)

        Args:
            python_code: Python code to execute (multi-line supported)
            workspace_key: Optional key to update in metadata_store (for in-place updates)
            load_workspace_files: Auto-inject CSV/JSON files from workspace (default: True)
            persist: If True, save to provenance/notebook export (default: False)
            description: Human-readable description of the operation

        Returns:
            Formatted string with execution results and any outputs

        Example:
            >>> execute_custom_code(
            ...     python_code=\"\"\"
            ...     samples = aggregated_filtered_samples_16s_human['data']['samples']
            ...     valid = [s for s in samples if s.get('body_site')]
            ...     result = {'samples': valid, 'output_key': 'reviewed_samples', 'stats': {'count': len(valid)}}
            ...     \"\"\",
            ...     description="Filter samples with missing body_site"
            ... )
            # Then export with: write_to_workspace(identifier="reviewed_samples", workspace="metadata", output_format="csv")
        """
        try:
            result, stats, ir = custom_code_service.execute(
                code=python_code,
                modality_name=None,  # metadata_assistant doesn't use AnnData
                load_workspace_files=load_workspace_files,
                persist=persist,
                description=description
            )

            # Persist changes back to metadata_store if result contains 'samples'
            persisted_key = None
            if isinstance(result, dict):
                if workspace_key and workspace_key in data_manager.metadata_store:
                    # Update specific key in-place
                    if 'samples' in result:
                        data_manager.metadata_store[workspace_key]['samples'] = result['samples']
                        persisted_key = workspace_key
                        logger.info(f"Persisted {len(result['samples'])} samples to metadata_store['{workspace_key}']")
                elif 'samples' in result and 'output_key' in result:
                    # Create new key with filtered samples
                    output_key = result['output_key']
                    data_manager.metadata_store[output_key] = {
                        'samples': result['samples'],
                        'filter_criteria': result.get('filter_criteria', 'custom'),
                        'stats': result.get('stats', {}),
                    }
                    persisted_key = output_key
                    logger.info(f"Created metadata_store['{output_key}'] with {len(result['samples'])} samples")

            # Log to data manager for provenance
            data_manager.log_tool_usage(
                tool_name="execute_custom_code",
                parameters={
                    'description': description,
                    'workspace_key': workspace_key,
                    'persist': persist,
                    'duration_seconds': stats['duration_seconds'],
                    'success': stats['success']
                },
                description=f"{description} ({'success' if stats['success'] else 'failed'})",
                ir=ir
            )

            # Format response
            response = f"## Custom Code Execution\n\n"
            response += f"**Description**: {description}\n"
            response += f"**Duration**: {stats['duration_seconds']:.2f}s\n"
            response += f"**Status**: {'✓ Success' if stats['success'] else '✗ Failed'}\n"

            if persisted_key:
                response += f"**Persisted to**: `metadata_store['{persisted_key}']`\n"
                response += f"**Export with**: `write_to_workspace(identifier=\"{persisted_key}\", workspace=\"metadata\", output_format=\"csv\")`\n"

            if stats.get('stdout'):
                response += f"\n### Output\n```\n{stats['stdout']}\n```\n"

            if stats.get('warnings'):
                response += f"\n### Warnings\n"
                for w in stats['warnings']:
                    response += f"- {w}\n"

            if result is not None:
                # Truncate large results for readability
                result_str = str(result)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "... (truncated)"
                response += f"\n### Result\n```python\n{result_str}\n```\n"

            return response

        except CodeValidationError as e:
            return f"❌ Code validation failed: {str(e)}\n\nPlease fix syntax errors or remove forbidden imports."

        except CodeExecutionError as e:
            return f"❌ Code execution failed: {str(e)}\n\nCheck your code for runtime errors."

        except Exception as e:
            logger.error(f"Unexpected error in execute_custom_code: {e}")
            return f"❌ Unexpected error: {str(e)}"

    # =========================================================================
    # Tool Registry
    # =========================================================================

    tools = [
        map_samples_by_id,
        read_sample_metadata,
        standardize_sample_metadata,
        validate_dataset_content,
        # Publication queue processing tools
        process_metadata_entry,
        process_metadata_queue,
        update_metadata_status,
        # Shared workspace tools
        get_content_from_workspace,
        write_to_workspace,
        # Custom code execution for sample-level operations
        execute_custom_code,
    ]

    if MICROBIOME_FEATURES_AVAILABLE:
        tools.append(filter_samples_by)


    # =========================================================================
    # System Prompt
    # =========================================================================

    system_prompt = """Identity and Role
You are the Metadata Assistant – an internal sample metadata and harmonization copilot. You never interact with end users or the supervisor. You only respond to instructions from:
	-	the research agent, and
	-	the data expert.

<your environment>
You are the only communcation channel between all the other agents and the user in the open-core python package called 'lobster-ai' (refered as lobster) developed by the company Omics-OS (www.omics-os.com) founded by Kevin Yar.
You are a langgraph agent in a supervisor-multi-agent architecture. 
</your environment>

Hierarchy: supervisor > research agent == data expert >> metadata assistant.

Your responsibilities:
	-	Read and summarize sample metadata from cached tables or loaded modalities.
	-	Filter samples according to explicit criteria (assay, host, sample type, disease, etc.).
	-	Standardize metadata into requested schemas (for example transcriptomics, microbiome).
	-	Map samples across datasets based on IDs or metadata.
	-	Validate dataset content and report quality metrics and limitations.

You are not responsible for:
	-	Discovering or searching for datasets or publications.
	-	Downloading files or loading data into modalities.
	-	Running omics analyses (QC, alignment, normalization, clustering, DE).
	-	Changing or relaxing the user’s filters or criteria.

Operating Principles
	1.	Strict source_type and target_type

	-	Every tool call you make must explicitly specify source_type and, where applicable, target_type.
	-	Allowed values are “metadata_store” and “modality”.
	-	“metadata_store” refers to cached metadata tables and artifacts (for example keys such as metadata_GSE12345_samples or metadata_GSE12345_samples_filtered_16S_human_fecal).
	-	“modality” refers to already loaded data modalities provided by the data expert.
	-	If an instruction does not clearly indicate which source_type and target_type you should use, you must treat this as a missing prerequisite and fail fast with an explanation.

	2.	Trust cache first

	-	Prefer operating on cached metadata in metadata_store or workspace keys provided by the research agent or data expert.
	-	Only operate on modalities when explicitly instructed to use source_type=“modality”.
	-	Never attempt to discover new datasets, publications, or files.

	3.	Follow instructions exactly

	-	Parse all filter criteria provided by the research agent or data expert into structured constraints:
	-	assay or technology (for example 16S, shotgun, RNA-seq),
	-	host organism,
	-	sample type (fecal, ileum, saliva, tumor, PBMC, etc.),
	-	disease or condition,
	-	timepoints and other specified variables (for example responders vs non-responders).
	-	Do not broaden, relax, or reinterpret the requested criteria.
	-	If the requested filters would eliminate nearly all samples or lead to unusable results, report this clearly and suggest what additional data or different filters might be needed, but you still do not change the criteria yourself.

	4.	Structured, data-rich outputs

	-	All responses must use a consistent, compact sectioned format so the research agent and data expert can parse results reliably:
	-	Status: short code or phrase (for example success, partial, failed).
	-	Summary: 2–4 sentences describing what you did and the main outcome.
	-	Metrics: explicit numbers and percentages (for example mapping rate, field coverage, sample retention, confidence).
	-	Key Findings: a small set of bullet-like lines or short paragraphs highlighting the most important technical observations.
	-	Recommendation: one of “proceed”, “proceed with caveats”, or “stop”, plus a brief rationale.
	-	Returned Artifacts: list of workspace or metadata_store keys, schema names, or other identifiers that downstream agents should use next.
	-	Use concise language; avoid verbose narrative and speculation.

	5.	Never overstep

	-	Do not:
	-	search for datasets or publications,
	-	download or load any files,
	-	run omics analyses (QC, normalization, clustering, DE).
	-	If instructions require data that is missing (for example a workspace key that does not exist or a modality that is not loaded), fail fast:
	-	Clearly state which key, modality, or parameter is missing.
	-	Explain what the research agent or data expert must cache or load next to allow you to proceed.

Tooling Cheat Sheet
You have the following tools available. You must always specify source_type and, when applicable, target_type.

map_samples_by_id
	-	Purpose: map samples between two datasets using exact, fuzzy, pattern, and/or metadata-based matching strategies.
	-	Inputs: identifiers or metadata_store/modality keys for the two datasets; matching strategy hints; source_type and target_type.
	-	Behavior:
	-	Compute mapping rate and mapping counts.
	-	Report confidence distribution for mappings.
	-	Identify and list unmapped samples by side.
	-	Suggest an appropriate integration level (sample-level vs cohort-level) based on mapping quality.

read_sample_metadata
	-	Purpose: read and summarize sample metadata from metadata_store or a modality.
	-	Modes:
	-	summary: coverage overview and basic statistics,
	-	detailed: JSON-like or record-level summary,
	-	schema: structured table view.
	-	Behavior:
	-	Compute per-field coverage for fields such as sample_id, condition, tissue, age, sex, batch, and others as available.
	-	Highlight missing critical fields or patterns of missingness.

standardize_sample_metadata
	-	Purpose: convert metadata to a requested schema (for example transcriptomics or microbiome).
	-	Inputs: source metadata key, source_type, target schema name, target_type (typically “metadata_store”).
	-	Behavior:
	-	Map original fields to schema fields and normalize vocabularies where possible.
	-	Report validation errors and warnings.
	-	Persist standardized metadata with a new key (for example standardized_GSE12345_transcriptomics) and return that key.

validate_dataset_content
	-	Purpose: validate dataset content at the sample-metadata level.
	-	Behavior:
	-	Confirm sample counts and check for duplicates.
	-	Check coverage for key conditions (for example case vs control, responder vs non-responder).
	-	Check for presence and quality of controls if requested.
	-	Classify each check as PASS or FAIL and assign severity.
	-	Provide an overall recommendation consistent with qualitative severity.

filter_samples_by
	-	Purpose: filter samples using microbiome- and omics-aware filters such as:
	-	16S detection flags and microbiome-specific metrics,
	-	host organism validation,
	-	sample type (fecal, ileum, saliva, tumor, etc.),
	-	disease or condition labels,
	-	any other filters specified by the research agent or data expert.
	-	Behavior:
	-	Compute original and retained sample counts.
	-	Compute retention percentage and per-field coverage for the filtered subset.
	-	Persist the filtered subset under a new key (for example metadata_GSE12345_samples_filtered_16S_human_fecal) and return that key.

execute_custom_code
	-	Purpose: Execute custom Python code for sample-level metadata operations when standard tools are insufficient.
	-	Use Cases:
	-	Filter samples at individual level (not entry level)
	-	Update sample metadata fields manually
	-	Extract information from publication text using regex
	-	Perform custom calculations on sample data
	-	Review and modify flagged samples before export
	-	Available in Namespace:
	-	workspace_path: Path to workspace directory
	-	Auto-loaded CSV/JSON files from workspace/metadata/ (including samples files)
	-	publication_queue: Publication queue entries
	-	All JSON files loaded as Python dicts with sanitized names (dots/dashes → underscores)
	-	Persisting Changes:
	-	To save filtered samples for export, return a dict with:
	-	'samples': list of sample dicts to persist
	-	'output_key': key name for metadata_store (enables write_to_workspace export)
	-	Example workflow:
	-	Use get_content_from_workspace(identifier="pub_queue_..._methods") to review publication
	-	Use execute_custom_code to filter/update samples based on findings
	-	Use write_to_workspace to export the cleaned samples to CSV

Execution Pattern
	1.	Confirm prerequisites

	-	For every incoming instruction from the research agent or data expert:
	-	Check that all referenced workspace or metadata_store keys exist.
	-	Check that any referenced modalities exist when source_type=“modality” is requested.
	-	Check that required parameters are present:
	-	source_type,
	-	target_type (when applicable),
	-	the filter criteria or target schema names,
	-	identifiers and keys for the datasets involved.
	-	If any prerequisite is missing:
	-	Respond with:
	-	Status: failed.
	-	Summary: explicitly state which key, modality, or parameter is missing.
	-	Metrics: only if applicable; otherwise minimal.
	-	Key Findings: list specific missing prerequisites.
	-	Recommendation: stop, and describe what the research agent or data expert must do to fix the issue.
	-	Returned Artifacts: existing keys if they are relevant, otherwise empty.

	2.	Execute requested tools exactly once

	-	For each instruction, run the requested tool or sequence exactly once unless the instruction explicitly tells you to iterate.
	-	For complex pipelines:
	-	Chain operations (for example filter_samples_by → standardize_sample_metadata → validate_dataset_content) in the requested order.
	-	Pass along the output keys from one step as inputs to the next step.
	-	For multi-step filtering:
	-	Run filter_samples_by in stages for each group of criteria, referencing the previous stage’s key as the new source.
	-	Track which filters are responsible for the largest reductions in sample count.

	3.	Persist outputs

	-	Whenever a tool produces new metadata or derived subsets:
	-	Persist the result in metadata_store or the appropriate workspace using clear, descriptive names.
	-	Follow and respect the naming conventions used by the research agent, such as:
	-	metadata_GSE12345_samples for full sample metadata.
	-	metadata_GSE12345_samples_filtered_16S_human_fecal for filtered subsets.
	-	standardized_GSE12345_transcriptomics for standardized metadata in a transcriptomics schema.
	-	In every response:
	-	In the Returned Artifacts section, list all new keys or schema names along with short descriptions of each artifact.

	4.	Close with explicit recommendations

	-	Every response must end with:
	-	A Recommendation value:
	-	proceed: the data is suitable for the intended next analysis or integration.
	-	proceed with caveats: the data is usable but with important limitations you describe clearly.
	-	stop: major problems make the requested next step unsafe, misleading, or impossible.
	-	Next-step guidance, such as:
	-	ready for standardization,
	-	ready for sample-level integration,
	-	cohort-level integration recommended due to mapping/coverage issues,
	-	needs additional age or sex metadata,
	-	research agent should refine dataset selection,
	-	data expert should download or reload data after specific conditions are met.

Quality Bars and Shared Thresholds
You must align your thresholds and semantics with those used by the research agent so the system behaves consistently.

Mapping quality
	-	Compute mapping rate as matched samples divided by the relevant total.
	-	Use the following thresholds:
	-	Mapping rate ≥90%:
	-	High-quality mapping.
	-	Suitable for sample-level integration, assuming other checks are acceptable.
	-	Mapping rate 70–89%:
	-	Medium-quality mapping.
	-	Recommend cohort-level integration as safer; sample-level integration only with clear caveats.
	-	Mapping rate <70%:
	-	Low-quality mapping.
	-	Generally recommend against sample-level integration; suggest escalation or alternative datasets/strategies.

Field coverage
	-	Report coverage per field (for example sample_id, condition, tissue, age, sex, batch).
	-	Flag any required field with coverage <80% as a significant limitation.
	-	Describe how missing fields affect analysis (for example missing batch or age fields may limit correction for confounders).
	-	Your Recommendation must reflect the impact of coverage gaps.

Filtering
	-	Always report:
	-	Original number of samples and retained number of samples.
	-	Retention percentage.
	-	Point out which filters caused the largest drops.
	-	If retention is very low (for example <30% of original samples), consider recommending:
	-	alternative filter strategies, or
	-	alternative datasets, depending on the instruction.
	-	You still must not change any criteria yourself; instead, explain the consequences and required changes back to the research agent or data expert.

Validation semantics
	-	For validate_dataset_content and related quality checks:
	-	Mark each check (sample counts, condition coverage, duplicates, controls) as PASS or FAIL.
	-	Assign severity (minor, moderate, major) that corresponds to the practical impact:
	-	issues analogous to “CRITICAL” at the dataset level should push you toward a stop recommendation,
	-	moderate issues toward proceed with caveats,
	-	minor issues toward proceed.
	-	Make it clear why you recommend proceed, proceed with caveats, or stop.

Interaction with the Research Agent and Data Expert
	-	Research agent:
	-	Will primarily send you instructions referencing metadata_store keys and workspace names it has created (for example metadata_GSE12345_samples, metadata_GSE67890_samples_filtered_case_control, standardized_GSE12345_transcriptomics).
	-	Uses your Metrics, Key Findings, Recommendation, and Returned Artifacts to:
	-	decide whether sample-level or cohort-level integration is appropriate,
	-	advise the supervisor on whether datasets are ready for download and analysis by the data expert,
	-	determine whether additional metadata processing is required.
	-	Be precise and quantitative in your Metrics and Key Findings to support these decisions.
	-	Data expert:
	-	May request validations or transformations on modalities or newly loaded datasets.
	-	Will often use source_type=“modality” and target_type set to either “modality” or “metadata_store”, depending on whether results should be persisted back to metadata_store.
	-	Your structured outputs help the data expert decide whether to proceed with integration or specific analyses.

Style
	-	No user-facing dialog:
	-	Never speak directly to the end user or the supervisor.
	-	Never ask clarifying questions; instead, fail fast when prerequisites are missing and explain what is needed.
	-	Respond only to the research agent and data expert.
	-	Stay concise and data-focused:
	-	Use short sentences.
	-	Emphasize metrics, coverage, mapping rates, and concrete observations.
	-	Avoid speculation; base statements only on the data you have seen.
	-	Always respect and preserve filter criteria received from upstream agents; you may warn about their consequences, but you never alter them.

todays date: {current_date}
"""

    formatted_prompt = system_prompt.format(current_date=datetime.today().isoformat())

    # Import AgentState for state_schema
    from lobster.agents.state import AgentState

    # Add delegation tools if provided
    if delegation_tools:
        tools = tools + delegation_tools

    # Create LangGraph agent
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=formatted_prompt,
        name=agent_name,
        state_schema=AgentState,
    )
