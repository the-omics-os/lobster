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
from lobster.tools.metadata_standardization_service import (
    MetadataStandardizationService,
)
from lobster.tools.sample_mapping_service import SampleMappingService
from lobster.core.analysis_ir import AnalysisStep
from lobster.utils.logger import get_logger

# Optional microbiome features (not in public lobster-local)
try:
    from lobster.tools.microbiome_filtering_service import MicrobiomeFilteringService
    from lobster.tools.disease_standardization_service import DiseaseStandardizationService
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
                from lobster.tools.metadata_validation_service import (
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
            from lobster.tools.workspace_content_service import WorkspaceContentService
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

            logger.info(f"Loaded {original_count} samples from workspace")

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
                logger.info(f"After 16S filter: {len(current_metadata)} samples retained")

            # Filter 2: Host organism validation
            if parsed_criteria["host_organisms"]:
                logger.info(f"Applying host organism filter: {parsed_criteria['host_organisms']}")
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
                logger.info(f"After host filter: {len(current_metadata)} samples retained")

            # Filter 3: Sample type filtering
            if parsed_criteria["sample_types"]:
                logger.info(f"Applying sample type filter: {parsed_criteria['sample_types']}")
                filtered, stats, ir = disease_standardization_service.filter_by_sample_type(
                    current_metadata, sample_types=parsed_criteria["sample_types"]
                )
                current_metadata = filtered
                irs.append(ir)
                stats_list.append(stats)
                logger.info(f"After sample type filter: {len(current_metadata)} samples retained")

            # Filter 4: Disease standardization
            if parsed_criteria["standardize_disease"]:
                logger.info("Applying disease standardization...")
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
                    logger.info(f"Disease standardization complete: {stats['standardization_rate']:.1f}% mapped")
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

            logger.info(f"Filtering complete: {final_count}/{original_count} samples retained ({retention_rate:.1f}%)")
            return report

        except ValueError as e:
            logger.error(f"Filtering error: {e}")
            return f"❌ Filtering failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected filtering error: {e}", exc_info=True)
            return f"❌ Unexpected error during filtering: {str(e)}"

    # =========================================================================
    # Tool Registry
    # =========================================================================

    tools = [
        map_samples_by_id,
        read_sample_metadata,
        standardize_sample_metadata,
        validate_dataset_content,
    ]

    if MICROBIOME_FEATURES_AVAILABLE:
        tools.append(filter_samples_by)

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
    # System Prompt
    # =========================================================================

    system_prompt = """
Identity and Role
You are the Metadata Assistant – an internal sample metadata and harmonization copilot. You never interact with end users or the supervisor. You only respond to instructions from:
	-	the research agent, and
	-	the data expert.

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
