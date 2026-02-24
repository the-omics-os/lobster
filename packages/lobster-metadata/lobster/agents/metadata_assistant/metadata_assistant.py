"""
Metadata Assistant Agent for Cross-Dataset Metadata Operations.

This agent specializes in sample ID mapping, metadata standardization, and
dataset content validation for multi-omics integration.

Phase 3 implementation for research agent refactoring.

Note: This agent replaces research_agent_assistant's metadata functionality.
The PDF resolution features were archived and will be migrated to research_agent
in Phase 4. See lobster/agents/archive/ARCHIVE_NOTICE.md for details.
"""

# =============================================================================
# Heavy imports below (may have circular dependencies, but AGENT_CONFIG is already defined)
# =============================================================================
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.analysis_ir import AnalysisStep
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.interfaces.validator import ValidationResult

# Graceful fallback: Import premium DiseaseOntologyService if available
# Public version (lobster-local) will use hardcoded fallback
try:
    from lobster.services.metadata.disease_ontology_service import (
        DiseaseOntologyService,
    )

    HAS_ONTOLOGY_SERVICE = True
except ImportError:
    HAS_ONTOLOGY_SERVICE = False

# Import helper functions from config module
from lobster.agents.metadata_assistant.config import (
    convert_list_to_dict,
    detect_metadata_pattern,
    phase1_column_rescan,
    phase2_llm_abstract_extraction,
    phase3_llm_methods_extraction,
    phase4_manual_mappings,
    suppress_logs,
)

# Import prompt factory
from lobster.agents.metadata_assistant.prompts import create_metadata_assistant_prompt

# Publication queue schemas
from lobster.core.schemas.publication_queue import HandoffStatus, PublicationStatus
from lobster.services.execution.custom_code_execution_service import (
    CustomCodeExecutionService,
)
from lobster.services.metadata.metadata_standardization_service import (
    MetadataStandardizationService,
)

# Sample mapping service
from lobster.services.metadata.sample_mapping_service import SampleMappingService
from lobster.tools.custom_code_tool import (
    create_execute_custom_code_tool,
    metadata_store_post_processor,
)
from lobster.tools.knowledgebase_tools import create_cross_database_id_mapping_tool
from lobster.utils.logger import get_logger

# Optional microbiome features (not in public lobster-local)
try:
    from lobster.services.metadata.disease_standardization_service import (
        DiseaseStandardizationService,
    )
    from lobster.services.metadata.metadata_filtering_service import (
        MetadataFilteringService,
        extract_disease_with_fallback,
    )
    from lobster.services.metadata.microbiome_filtering_service import (
        MicrobiomeFilteringService,
    )

    MICROBIOME_FEATURES_AVAILABLE = True
except ImportError:
    MicrobiomeFilteringService = None
    DiseaseStandardizationService = None
    MetadataFilteringService = None
    MICROBIOME_FEATURES_AVAILABLE = False

# Optional Rich UI for progress visualization
try:
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from lobster.ui.components.parallel_workers_progress import (
        parallel_workers_progress,
    )

    PROGRESS_UI_AVAILABLE = True
except ImportError:
    parallel_workers_progress = None
    ThreadPoolExecutor = None
    PROGRESS_UI_AVAILABLE = False

# Optional vector search for semantic standardization
try:
    from lobster.services.vector.service import VectorSearchService  # noqa: F401

    HAS_VECTOR_SEARCH = True
except ImportError:
    HAS_VECTOR_SEARCH = False

logger = get_logger(__name__)


# =========================================================================
# Helper: Log Suppression for Progress UI
# =========================================================================


# =========================================================================
# Helper: Metadata Pattern Detection (Option D - Namespace Separation)
# =========================================================================


# =========================================================================
# Disease Enrichment Helpers (Phase 1)
# =========================================================================


def metadata_assistant(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "metadata_assistant",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
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
        workspace_path: Path to workspace directory for config resolution

    Returns:
        Compiled LangGraph agent with metadata tools
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("metadata_assistant")
    llm = create_llm(
        "metadata_assistant",
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Initialize services (Phase 3: new services)
    sample_mapping_service = SampleMappingService(data_manager=data_manager)
    metadata_standardization_service = MetadataStandardizationService(
        data_manager=data_manager
    )

    # Initialize optional microbiome services if available
    microbiome_filtering_service = None
    disease_standardization_service = None
    metadata_filtering_service = None
    if MICROBIOME_FEATURES_AVAILABLE:
        microbiome_filtering_service = MicrobiomeFilteringService()
        disease_standardization_service = DiseaseStandardizationService()
        # Create filtering service with dependencies (disease_extractor set later)
        metadata_filtering_service = MetadataFilteringService(
            microbiome_service=microbiome_filtering_service,
            disease_service=disease_standardization_service,
        )
        logger.debug("Microbiome features enabled")
    else:
        logger.debug("Microbiome features not available (optional)")

    # Custom code execution service for sample-level operations
    custom_code_service = CustomCodeExecutionService(data_manager)

    # Create shared execute_custom_code tool via factory (v2.7+: unified tool)
    execute_custom_code = create_execute_custom_code_tool(
        data_manager=data_manager,
        custom_code_service=custom_code_service,
        agent_name="metadata_assistant",
        post_processor=metadata_store_post_processor,
    )

    # Cross-database ID mapping tool (UniProt + Ensembl)
    map_cross_database_ids = create_cross_database_id_mapping_tool(data_manager)

    logger.debug("metadata_assistant agent initialized")

    # Import pandas at factory function level for type hints in helper functions
    import pandas as pd

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

                    # Detect pattern (Option D: Namespace Separation)
                    pattern = detect_metadata_pattern(cached)

                    if pattern == "geo":
                        samples_dict = cached["metadata"]["samples"]
                    elif pattern == "aggregated":
                        samples_dict = convert_list_to_dict(cached["samples"])
                    else:
                        raise ValueError(
                            f"Unrecognized metadata pattern in '{identifier}'. "
                            "Expected 'metadata.samples' dict or 'samples' list."
                        )

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
                description=f"Mapped {result.summary['exact_matches']} exact, {result.summary['fuzzy_matches']} fuzzy, {result.summary['unmapped']} unmapped ({result.summary['mapping_rate']:.1%} rate)",
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
                # Two-tier cache pattern: Check memory first, then workspace files
                cached = None

                # Tier 1: Check in-memory metadata_store (fast path)
                if source in data_manager.metadata_store:
                    cached = data_manager.metadata_store[source]
                    logger.debug(
                        f"Found '{source}' in metadata_store (Tier 1 - memory)"
                    )

                # Tier 2: Fallback to workspace files (persistent, survives session restart)
                else:
                    logger.debug(
                        f"'{source}' not in metadata_store, checking workspace files (Tier 2)"
                    )
                    try:
                        from lobster.services.data_access.workspace_content_service import (
                            ContentType,
                            WorkspaceContentService,
                        )

                        workspace_service = WorkspaceContentService(data_manager)
                        cached = workspace_service.read_content(
                            identifier=source,
                            content_type=ContentType.METADATA,
                            level=None,  # Full content
                        )

                        # Lazy loading: Promote to metadata_store for subsequent fast access
                        data_manager.metadata_store[source] = cached
                        logger.info(
                            f"Loaded '{source}' from workspace and promoted to metadata_store"
                        )

                    except FileNotFoundError:
                        return (
                            f"❌ Error: '{source}' not found in metadata_store or workspace files. "
                            f"Use research_agent.validate_dataset_metadata() or process_metadata_queue first."
                        )

                # Detect pattern (Option D: Namespace Separation)
                pattern = detect_metadata_pattern(cached)

                if pattern == "geo":
                    # GEO pattern: {"metadata": {"samples": {...}}}
                    samples_dict = cached["metadata"]["samples"]
                elif pattern == "aggregated":
                    # Aggregated pattern: {"samples": [...]} - convert list to dict
                    samples_dict = convert_list_to_dict(cached["samples"])
                    logger.debug(
                        f"Converted aggregated pattern ({len(cached['samples'])} samples) to dict for '{source}'"
                    )
                else:
                    return f"❌ Error: Unrecognized metadata pattern in '{source}'. Expected 'metadata.samples' dict or 'samples' list."

                if not samples_dict:
                    return f"❌ Error: No sample metadata in '{source}'."

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
                description=f"Read {len(sample_df)} samples in {return_format} format",
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
                # Cap output to prevent token explosion (50 samples max by default)
                MAX_SAMPLES = 50
                if len(sample_df) > MAX_SAMPLES:
                    truncated_df = sample_df.head(MAX_SAMPLES)
                    output = json.dumps(
                        truncated_df.to_dict(orient="records"), indent=2
                    )
                    return f"⚠️ Output truncated to {MAX_SAMPLES}/{len(sample_df)} samples. Use filter_samples_by to narrow results.\n\n{output}"
                return json.dumps(sample_df.to_dict(orient="records"), indent=2)
            elif return_format == "schema":
                logger.info(f"Metadata schema extracted for {source}")
                # Cap output to prevent token explosion (50 samples max by default)
                MAX_SAMPLES = 50
                if len(sample_df) > MAX_SAMPLES:
                    truncated_df = sample_df.head(MAX_SAMPLES)
                    output = truncated_df.to_markdown(index=True)
                    return f"⚠️ Output truncated to {MAX_SAMPLES}/{len(sample_df)} samples. Use filter_samples_by to narrow results.\n\n{output}"
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
                description=f"Standardized {len(result.standardized_metadata)} valid samples, {len(result.validation_errors)} errors, {len(result.warnings)} warnings",
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

                # Detect pattern (Option D: Namespace Separation)
                pattern = detect_metadata_pattern(cached_metadata)

                if pattern == "geo":
                    samples_dict = cached_metadata["metadata"]["samples"]
                elif pattern == "aggregated":
                    samples_dict = convert_list_to_dict(cached_metadata["samples"])
                else:
                    return (
                        f"❌ Error: Unrecognized metadata pattern in '{source}'. "
                        "Expected 'metadata.samples' dict or 'samples' list."
                    )

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
                description=f"Validated: samples={'✓' if result.has_required_samples else '✗'}, platform={'✓' if result.platform_consistency else '✗'}, {len(result.duplicate_ids)} duplicates, {len(result.warnings)} warnings",
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
    # Helper: Disease Extraction from Diverse SRA Fields
    # =========================================================================

    def _extract_disease_from_raw_fields(
        metadata: pd.DataFrame, study_context: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Extract disease information from diverse, study-specific SRA field names.

        SRA datasets have NO standardized disease field. Disease data appears in:
        - Free-text: host_phenotype, phenotype, disease_state, diagnosis
        - Boolean flags: crohns_disease, inflam_bowel_disease, parkinson_disease
        - Study title: Embedded in study_title or experiment_title

        This method consolidates diverse field patterns into a unified "disease" column.

        Extraction Strategies (applied in order):
        1. Existing unified column (disease, disease_state, condition, diagnosis)
        2. Free-text phenotype fields (host_phenotype → disease)
        3. Boolean disease flags (crohns_disease: Yes → disease: cd)
        4. Study context (publication-level disease inference)

        Args:
            metadata: DataFrame with SRA sample metadata
            study_context: Optional publication metadata for context

        Returns:
            Column name containing unified disease data, or None if no disease info found

        Example transformations:
            host_phenotype: "Parkinson's Disease" → disease: "Parkinson's Disease"
            crohns_disease: "Yes" → disease: "cd"
            inflam_bowel_disease: "Yes" → disease: "ibd"
            parkinson_disease: TRUE → disease: "parkinsons"
        """
        # Strategy 1: Check for existing unified disease column
        existing_disease_cols = ["disease", "disease_state", "condition", "diagnosis"]
        for col in existing_disease_cols:
            if col in metadata.columns and metadata[col].notna().sum() > 0:
                logger.debug(f"Found existing disease column: {col}")
                # Rename to standard "disease" if different
                if col != "disease":
                    metadata["disease"] = metadata[col]
                    metadata["disease_original"] = metadata[col]
                return "disease"

        # Strategy 2: Extract from free-text phenotype fields
        phenotype_cols = [
            "host_phenotype",
            "phenotype",
            "host_disease",
            "health_status",
        ]
        for col in phenotype_cols:
            if col in metadata.columns:
                # Count non-empty values
                populated_count = metadata[col].notna().sum()
                if populated_count > 0:
                    # Create unified disease column from phenotype
                    metadata["disease"] = metadata[col].fillna("unknown")
                    metadata["disease_original"] = metadata[col].fillna("unknown")
                    logger.debug(
                        f"Extracted disease from {col} "
                        f"({populated_count}/{len(metadata)} samples, "
                        f"{populated_count / len(metadata) * 100:.1f}%)"
                    )
                    return "disease"

        # Strategy 3: Consolidate boolean disease flags
        # Find columns ending with "_disease" (crohns_disease, inflam_bowel_disease, etc.)
        disease_flag_cols = [c for c in metadata.columns if c.endswith("_disease")]

        if disease_flag_cols:
            logger.debug(
                f"Found {len(disease_flag_cols)} disease flag columns: {disease_flag_cols}"
            )

            def extract_from_flags(row):
                """Extract disease from boolean flags."""
                active_diseases = []

                for flag_col in disease_flag_cols:
                    # Check if flag is TRUE (handles Yes, Y, TRUE, True, 1, "1")
                    flag_value = row.get(flag_col)
                    if flag_value in [
                        "Yes",
                        "YES",
                        "yes",
                        "Y",
                        "y",
                        "TRUE",
                        "True",
                        "true",
                        True,
                        1,
                        "1",
                    ]:
                        # Convert flag name to disease term
                        # Examples:
                        #   crohns_disease → cd
                        #   inflam_bowel_disease → ibd
                        #   ulcerative_colitis → uc
                        #   parkinson_disease → parkinsons
                        disease_name = flag_col.replace("_disease", "").replace("_", "")

                        # Map common patterns to standard terms
                        disease_map = {
                            "crohns": "cd",
                            "crohn": "cd",
                            "inflammbowel": "ibd",
                            "inflambowel": "ibd",  # Handle different spellings
                            "ulcerativecolitis": "uc",
                            "colitis": "uc",
                            "parkinson": "parkinsons",
                            "parkinsons": "parkinsons",
                        }

                        standardized = disease_map.get(disease_name, disease_name)
                        active_diseases.append(standardized)

                if active_diseases:
                    # If multiple diseases, join with semicolon
                    return ";".join(active_diseases)

                # Check for negative controls (all flags FALSE)
                all_false = all(
                    row.get(flag_col)
                    in [
                        "No",
                        "NO",
                        "no",
                        "N",
                        "n",
                        "FALSE",
                        "False",
                        "false",
                        False,
                        0,
                        "0",
                    ]
                    for flag_col in disease_flag_cols
                )
                if all_false:
                    return "healthy"

                return "unknown"

            # Apply extraction
            metadata["disease"] = metadata.apply(extract_from_flags, axis=1)
            metadata["disease_original"] = metadata.apply(
                lambda row: ";".join(
                    [
                        f"{col}={row[col]}"
                        for col in disease_flag_cols
                        if pd.notna(row.get(col))
                    ]
                ),
                axis=1,
            )

            # Count successful extractions
            extracted_count = (metadata["disease"] != "unknown").sum()
            logger.debug(
                f"Extracted disease from {len(disease_flag_cols)} boolean flags "
                f"({extracted_count}/{len(metadata)} samples, "
                f"{extracted_count / len(metadata) * 100:.1f}%)"
            )

            return "disease"

        # Strategy 4: Use study context (publication-level disease focus)
        if study_context and "disease_focus" in study_context:
            # All samples in this study share the publication's disease focus
            metadata["disease"] = study_context["disease_focus"]
            metadata["disease_original"] = (
                f"inferred from publication: {study_context['disease_focus']}"
            )
            logger.debug(
                f"Assigned disease from publication context: {study_context['disease_focus']}"
            )
            return "disease"

        logger.warning(
            "No disease information found in metadata fields or study context"
        )
        return None

    # =========================================================================
    # Tool 5: Filter Samples By Criteria (Microbiome/Disease)
    # =========================================================================

    @tool
    def filter_samples_by(
        workspace_key: str,
        filter_criteria: str,
        strict: bool = True,
        min_disease_coverage: float = 0.5,
        strict_disease_validation: bool = True,
    ) -> str:
        """
        Filter samples by microbiome criteria (16S/shotgun, host, sample type, disease).

        Filter Criteria Syntax (CANONICAL - other tools reference this):
            "16S human fecal CRC"           - Basic: amplicon + host + sample_type + disease
            "16S V4 human fecal CRC"        - With region: V4, V3-V4, V1-V9, full-length
            "shotgun human gut UC"          - WGS/WXS/METAGENOMIC (excludes WGA)
            "16S shotgun human fecal"       - Both types (OR logic)

        Filters apply in sequence: amplicon → region → host → sample_type → disease

        Args:
            workspace_key: Metadata workspace key (e.g., "geo_gse123456")
            filter_criteria: Natural language criteria string
            strict: Strict 16S matching (default: True)
            min_disease_coverage: Min disease rate 0.0-1.0 (default: 0.5)
            strict_disease_validation: Fail on low coverage (default: True)

        Returns:
            Filtering report with retention rate and summary

        Example:
            filter_samples_by(workspace_key="geo_gse123456", filter_criteria="16S V4 human fecal CRC")
        """
        # Check if microbiome services are available
        if not MICROBIOME_FEATURES_AVAILABLE:
            return "❌ Error: Microbiome filtering features are not available in this installation. This is an optional feature."

        try:
            logger.info(
                f"Filtering samples: workspace_key={workspace_key}, criteria='{filter_criteria}', strict={strict}"
            )

            # Parse natural language criteria using service
            if not metadata_filtering_service:
                return "❌ Error: Microbiome filtering service not available"
            parsed_criteria = metadata_filtering_service.parse_criteria(filter_criteria)

            logger.debug(f"Parsed criteria: {parsed_criteria}")

            # Read workspace metadata via WorkspaceContentService
            from lobster.services.data_access.workspace_content_service import (
                ContentType,
                WorkspaceContentService,
            )

            workspace_service = WorkspaceContentService(data_manager)
            workspace_data = workspace_service.read_content(
                workspace_key, content_type=ContentType.METADATA
            )
            if not workspace_data:
                return f"❌ Error: Workspace key '{workspace_key}' not found or empty"

            # Detect pattern (Option D: Namespace Separation)
            if not isinstance(workspace_data, dict):
                return "❌ Error: Unexpected workspace data format (expected dict)"

            pattern = detect_metadata_pattern(workspace_data)

            if pattern == "geo":
                # GEO pattern: {"metadata": {"samples": {...}}}
                metadata_dict = workspace_data["metadata"]["samples"]
            elif pattern == "aggregated":
                # Aggregated pattern: {"samples": [...]} - keep as list for filtering efficiency
                metadata_dict = workspace_data["samples"]
                logger.debug(
                    f"Using aggregated pattern ({len(metadata_dict)} samples) from '{workspace_key}'"
                )
            else:
                # Fallback: treat entire workspace_data as samples dict
                metadata_dict = workspace_data

            if not metadata_dict:
                return f"❌ Error: No sample metadata found in workspace key '{workspace_key}'"

            # Convert to DataFrame
            import pandas as pd

            # Handle both dict-of-dicts (orient="index") and list-of-dicts
            if isinstance(metadata_dict, list):
                metadata_df = pd.DataFrame(metadata_dict)
                # Set index to run_accession if available for consistency
                if "run_accession" in metadata_df.columns:
                    metadata_df = metadata_df.set_index("run_accession")
            else:
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
                    filtered, stats, ir = (
                        microbiome_filtering_service.validate_16s_amplicon(
                            row_dict, strict=strict
                        )
                    )
                    if filtered:  # Non-empty dict means valid
                        filtered_rows.append(idx)
                    if stats["is_valid"]:
                        irs.append(ir)
                        stats_list.append(stats)

                current_metadata = current_metadata.loc[filtered_rows]
                logger.debug(
                    f"After 16S filter: {len(current_metadata)} samples retained"
                )

            # Filter 2: Host organism validation
            if parsed_criteria["host_organisms"]:
                logger.debug(
                    f"Applying host organism filter: {parsed_criteria['host_organisms']}"
                )
                filtered_rows = []
                for idx, row in current_metadata.iterrows():
                    row_dict = row.to_dict()
                    filtered, stats, ir = (
                        microbiome_filtering_service.validate_host_organism(
                            row_dict, allowed_hosts=parsed_criteria["host_organisms"]
                        )
                    )
                    if filtered:  # Non-empty dict means valid
                        filtered_rows.append(idx)
                    if stats["is_valid"]:
                        irs.append(ir)
                        stats_list.append(stats)

                current_metadata = current_metadata.loc[filtered_rows]
                logger.debug(
                    f"After host filter: {len(current_metadata)} samples retained"
                )

            # Filter 3: Sample type filtering
            if parsed_criteria["sample_types"]:
                logger.debug(
                    f"Applying sample type filter: {parsed_criteria['sample_types']}"
                )
                filtered, stats, ir = (
                    disease_standardization_service.filter_by_sample_type(
                        current_metadata, sample_types=parsed_criteria["sample_types"]
                    )
                )
                current_metadata = filtered
                irs.append(ir)
                stats_list.append(stats)
                logger.debug(
                    f"After sample type filter: {len(current_metadata)} samples retained"
                )

            # Filter 4: Disease extraction + standardization
            if parsed_criteria["standardize_disease"]:
                logger.debug("Applying disease extraction + standardization...")

                # NEW: Extract disease from diverse field patterns (v1.2.0)
                # Handles: host_phenotype, boolean flags (crohns_disease, etc.)
                disease_col = _extract_disease_from_raw_fields(
                    current_metadata, study_context=None
                )

                if disease_col:
                    # Apply standardization to extracted disease column
                    try:
                        standardized, stats, ir = (
                            disease_standardization_service.standardize_disease_terms(
                                current_metadata, disease_column=disease_col
                            )
                        )
                        current_metadata = standardized
                        irs.append(ir)
                        stats_list.append(stats)
                        logger.debug(
                            f"Disease extraction + standardization complete: {stats['standardization_rate']:.1f}% mapped"
                        )

                        # Validate disease coverage if requested (pass parameters through)
                        if min_disease_coverage > 0:
                            disease_coverage = (
                                stats.get("standardization_rate", 0) / 100.0
                            )
                            if disease_coverage < min_disease_coverage:
                                error_msg = (
                                    f"Insufficient disease data coverage: {disease_coverage * 100:.1f}% "
                                    f"(required: {min_disease_coverage * 100:.1f}%). "
                                    f"Consider using min_disease_coverage=0.0 to disable validation."
                                )
                                if strict_disease_validation:
                                    return f"❌ Disease validation failed:\n{error_msg}"
                                else:
                                    logger.warning(
                                        f"Disease validation warning: {error_msg}"
                                    )
                    except ValueError as e:
                        if "Insufficient disease data" in str(e):
                            # Extract coverage percentage
                            coverage_match = re.search(r"(\d+\.\d+)% coverage", str(e))
                            coverage = (
                                float(coverage_match.group(1))
                                if coverage_match
                                else 0.0
                            )

                            return f"""❌ Disease validation failed: {coverage:.1f}% coverage < 50% threshold

I can attempt automatic enrichment to improve coverage. Would you like me to:

1. **Auto-enrich** (recommended): Extract disease from publication abstracts/methods using LLM
   Command: enrich_samples_with_disease(workspace_key="{workspace_key}", enrichment_mode="hybrid")

2. **Manual mapping**: Provide disease for each publication
   Command: enrich_samples_with_disease(
       workspace_key="{workspace_key}",
       enrichment_mode="manual",
       manual_mappings='{{"pub_queue_doi_10_1234": "crc", "pub_queue_pmid_5678": "uc"}}'
   )

3. **Preview first** (recommended): Test enrichment without saving
   Command: enrich_samples_with_disease(workspace_key="{workspace_key}", dry_run=True)

4. **Lower threshold** (not recommended): min_disease_coverage={coverage / 100:.2f}

Which approach would you prefer?"""
                        else:
                            raise
                else:
                    logger.warning(
                        "Disease standardization requested but no disease information found in metadata"
                    )

            # Calculate final stats
            final_count = len(current_metadata)
            retention_rate = (
                (final_count / original_count * 100) if original_count > 0 else 0
            )

            # Combine IRs into composite IR
            composite_ir = _combine_analysis_steps(
                irs,
                operation="filter_samples_by",
                description=f"Multi-criteria filtering: {filter_criteria}",
            )

            # Log tool usage
            data_manager.log_tool_usage(
                tool_name="filter_samples_by",
                parameters={
                    "workspace_key": workspace_key,
                    "filter_criteria": filter_criteria,
                    "strict": strict,
                    "min_disease_coverage": min_disease_coverage,
                    "strict_disease_validation": strict_disease_validation,
                    "parsed_criteria": parsed_criteria,
                },
                description=f"Filtered {original_count}→{final_count} samples ({retention_rate:.1f}% retention), {len(irs)} filters applied",
                ir=composite_ir,
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
                filtered_metadata=current_metadata,
            )

            logger.debug(
                f"Filtering complete: {final_count}/{original_count} samples retained ({retention_rate:.1f}%)"
            )
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
    from lobster.services.data_access.workspace_content_service import (
        WorkspaceContentService,
    )
    from lobster.tools.workspace_tool import (
        create_get_content_from_workspace_tool,
        create_write_to_workspace_tool,
    )

    workspace_service = WorkspaceContentService(data_manager=data_manager)

    # Create shared workspace tools
    get_content_from_workspace = create_get_content_from_workspace_tool(data_manager)
    write_to_workspace = create_write_to_workspace_tool(data_manager)

    @tool
    def process_metadata_entry(
        entry_id: str,
        filter_criteria: str = None,
        standardize_schema: str = None,
        min_disease_coverage: float = 0.5,
        strict_disease_validation: bool = True,
    ) -> str:
        """
        Process a single publication queue entry for metadata filtering/standardization.

        Args:
            entry_id: Publication queue entry ID
            filter_criteria: Filter string. Syntax: see filter_samples_by docstring. None=no filter.
            standardize_schema: Optional schema (e.g., "microbiome")
            min_disease_coverage: Min disease rate 0.0-1.0 (default: 0.5)
            strict_disease_validation: Fail on low coverage (default: True)

        Returns:
            Processing summary with sample counts and filtered metadata location

        Example:
            process_metadata_entry(entry_id="pub_123", filter_criteria="16S V4 human fecal CRC")
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
                entry_id,
                entry.status,
                handoff_status=HandoffStatus.METADATA_IN_PROGRESS,
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
                    samples, validation_result, quality_stats = (
                        _extract_samples_from_workspace(ws_data)
                    )
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
            if filter_criteria and all_samples and metadata_filtering_service:
                logger.debug(
                    f"Applying filter criteria to {len(all_samples)} samples: '{filter_criteria}'"
                )
                parsed = metadata_filtering_service.parse_criteria(filter_criteria)
                # Add disease validation parameters
                parsed["min_disease_coverage"] = min_disease_coverage
                parsed["strict_disease_validation"] = strict_disease_validation
                # Pass disease_extractor with fallback chain
                metadata_filtering_service.disease_extractor = (
                    extract_disease_with_fallback
                )
                try:
                    all_samples, filter_stats, _ = (
                        metadata_filtering_service.apply_filters(all_samples, parsed)
                    )
                    logger.debug(
                        f"After filtering: {len(all_samples)} samples retained "
                        f"({filter_stats.get('retention_rate', 0):.1f}%)"
                    )
                except ValueError as e:
                    if "Insufficient disease data" in str(e):
                        error_msg = f"Disease validation failed: {str(e)}"
                        logger.error(f"Entry {entry_id}: {error_msg}")
                        # Mark as METADATA_FAILED
                        queue.update_status(
                            entry_id,
                            entry.status,
                            handoff_status=HandoffStatus.METADATA_FAILED,
                            error=error_msg,
                        )
                        return f"❌ {error_msg}"
                    else:
                        raise

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
                status=PublicationStatus.COMPLETED,  # Terminal success state
                handoff_status=HandoffStatus.METADATA_COMPLETE,
                harmonization_metadata=harmonization_data,
            )

            retention = (
                (samples_after / samples_before * 100) if samples_before > 0 else 0
            )

            # Build response with validation info
            response = f"""## Entry Processed: {entry_id}
**Samples Extracted**: {samples_before}
**Samples Valid**: {len(all_samples)}
**Samples After Filter**: {samples_after}
**Retention**: {retention:.1f}%
**Filter**: {filter_criteria or "None"}
**Validation**: {total_errors} errors, {total_warnings} warnings
"""
            # Include validation messages if there are issues (cap at 5 to prevent token explosion)
            if total_errors > 0 or total_warnings > 0:
                response += "\n### Validation Summary\n"
                MAX_VALIDATION_KEYS = 5
                shown_count = 0
                for idx, vr in enumerate(all_validation_results):
                    if vr.has_errors or vr.has_warnings:
                        if shown_count >= MAX_VALIDATION_KEYS:
                            remaining = sum(
                                1
                                for v in all_validation_results[idx:]
                                if v.has_errors or v.has_warnings
                            )
                            response += f"\n⚠️ ...and {remaining} more workspace keys with issues (truncated)\n"
                            break
                        response += f"\n**Workspace key {idx + 1}**:\n"
                        response += vr.format_messages(include_info=False) + "\n"
                        shown_count += 1

            return response
        except Exception as e:
            logger.error(f"Error processing entry {entry_id}: {e}")
            return f"❌ Error processing entry: {str(e)}"

    def _process_single_entry_for_queue(
        entry, filter_criteria, min_disease_coverage, strict_disease_validation
    ):
        """
        Process a single entry and return (entry_samples, entry_stats, failed_reason).

        Extracted for reuse in both sequential and parallel processing.

        Args:
            entry: Publication queue entry
            filter_criteria: Filter criteria string
            min_disease_coverage: Minimum disease coverage threshold
            strict_disease_validation: Whether to fail on low coverage

        Returns:
            Tuple of (entry_samples: List, entry_stats: Dict, failed_reason: Optional[str])
        """
        try:
            if not entry.workspace_metadata_keys:
                logger.debug(
                    f"Skipping entry {entry.entry_id}: no workspace_metadata_keys"
                )
                return [], {}, None

            entry_samples = []
            entry_validation_results = []
            entry_quality_stats = []

            for ws_key in entry.workspace_metadata_keys:
                # Only process SRA sample files
                if not (ws_key.startswith("sra_") and ws_key.endswith("_samples")):
                    continue

                logger.debug(f"Processing {ws_key} for entry {entry.entry_id}")

                from lobster.services.data_access.workspace_content_service import (
                    ContentType,
                )

                ws_data = workspace_service.read_content(
                    ws_key, content_type=ContentType.METADATA
                )
                if ws_data:
                    samples, validation_result, quality_stats = (
                        _extract_samples_from_workspace(ws_data)
                    )
                    entry_samples.extend(samples)
                    entry_validation_results.append(validation_result)
                    entry_quality_stats.append(quality_stats)

            samples_extracted = sum(
                vr.metadata.get("total_samples", 0) for vr in entry_validation_results
            )
            samples_valid = len(entry_samples)

            # Check for complete validation failure
            if samples_extracted > 0 and samples_valid == 0:
                error_count = sum(len(vr.errors) for vr in entry_validation_results)
                error_msg = f"All {samples_extracted} samples failed validation ({error_count} errors)"
                logger.error(f"Entry {entry.entry_id}: {error_msg}")
                return [], {"extracted": samples_extracted, "valid": 0}, error_msg

            # Apply filters if specified
            if filter_criteria and entry_samples and metadata_filtering_service:
                samples_before_filter = len(entry_samples)
                parsed = metadata_filtering_service.parse_criteria(filter_criteria)
                # Add disease validation parameters
                parsed["min_disease_coverage"] = min_disease_coverage
                parsed["strict_disease_validation"] = strict_disease_validation
                metadata_filtering_service.disease_extractor = (
                    extract_disease_with_fallback
                )
                try:
                    entry_samples, _, _ = metadata_filtering_service.apply_filters(
                        entry_samples, parsed
                    )
                    logger.debug(
                        f"Entry {entry.entry_id}: Filter applied - "
                        f"{len(entry_samples)}/{samples_before_filter} samples retained"
                    )
                except ValueError as e:
                    if "Insufficient disease data" in str(e):
                        error_msg = f"Disease validation failed: {str(e)}"
                        logger.error(f"Entry {entry.entry_id}: {error_msg}")
                        return (
                            [],
                            {"extracted": samples_extracted, "valid": samples_valid},
                            error_msg,
                        )
                    else:
                        raise

            # Add publication context
            for sample in entry_samples:
                sample["publication_entry_id"] = entry.entry_id
                sample["publication_title"] = entry.title
                sample["publication_doi"] = entry.doi
                sample["publication_pmid"] = entry.pmid

            entry_stats = {
                "extracted": samples_extracted,
                "valid": samples_valid,
                "after_filter": len(entry_samples),
                "validation_errors": sum(
                    len(vr.errors) for vr in entry_validation_results
                ),
                "validation_warnings": sum(
                    len(vr.warnings) for vr in entry_validation_results
                ),
            }

            return entry_samples, entry_stats, None

        except Exception as e:
            error_msg = f"Failed to process entry {entry.entry_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [], {}, error_msg

    def _process_queue_with_progress(
        entries,
        filter_criteria,
        output_key,
        parallel_workers,
        min_disease_coverage,
        strict_disease_validation,
    ):
        """Process queue entries in parallel with Rich progress visualization."""
        if not PROGRESS_UI_AVAILABLE:
            logger.warning("Progress UI not available, cannot use parallel processing")
            return "❌ Progress UI not available. Use sequential processing instead."

        # Batch flush configuration - reduces 655 atomic writes to ~33 writes (20x improvement)
        BATCH_FLUSH_SIZE = 20

        # Access publication queue via data_manager (in closure scope)
        pub_queue = data_manager.publication_queue

        def _batch_flush_updates(updates):
            """Flush multiple status updates in a single atomic write.

            This dramatically reduces I/O by writing all pending updates at once
            instead of one-by-one, avoiding the O(N²) bottleneck where each update
            reads/writes the entire queue file.
            """
            if not updates:
                return

            with pub_queue._locked():
                entries_list = pub_queue._load_entries()
                entry_map = {e.entry_id: e for e in entries_list}

                for update in updates:
                    entry_id = update["entry_id"]
                    if entry_id in entry_map:
                        entry = entry_map[entry_id]
                        entry.update_status(
                            status=update["status"],
                            handoff_status=update["handoff_status"],
                            error=update.get("error"),
                        )

                pub_queue._write_entries_atomic(entries_list)

            logger.debug(f"Batch flushed {len(updates)} status updates")

        results = []
        results_lock = threading.Lock()
        all_samples = []
        samples_lock = threading.Lock()

        # Work-stealing queue
        entry_queue = list(enumerate(entries))
        queue_lock = threading.Lock()

        def get_next_entry():
            with queue_lock:
                if entry_queue:
                    return entry_queue.pop(0)
                return None

        start_time = time.time()
        effective_workers = min(parallel_workers, len(entries))

        with suppress_logs():
            with parallel_workers_progress(effective_workers, len(entries)) as progress:

                def worker_func(worker_id: int):
                    """Worker function that processes entries from queue."""
                    # Collect status updates in memory, flush in batches to avoid O(N²) I/O
                    pending_updates = []

                    while True:
                        next_item = get_next_entry()
                        if next_item is None:
                            # Flush remaining updates before exiting
                            if pending_updates:
                                _batch_flush_updates(pending_updates)
                            progress.worker_done(worker_id)
                            break

                        idx, entry = next_item

                        # Get title for display
                        title = (entry.title or entry.entry_id)[:35]
                        progress.worker_start(worker_id, title)

                        # Process entry
                        entry_start = time.time()
                        entry_samples, entry_stats, failed_reason = (
                            _process_single_entry_for_queue(
                                entry,
                                filter_criteria,
                                min_disease_coverage,
                                strict_disease_validation,
                            )
                        )

                        elapsed = time.time() - entry_start

                        # Determine status and queue update (batched, not immediate)
                        if failed_reason:
                            status = "failed"
                            pending_updates.append(
                                {
                                    "entry_id": entry.entry_id,
                                    "status": entry.status,
                                    "handoff_status": HandoffStatus.METADATA_FAILED,
                                    "error": failed_reason,
                                }
                            )
                        else:
                            status = "completed"
                            pending_updates.append(
                                {
                                    "entry_id": entry.entry_id,
                                    "status": PublicationStatus.COMPLETED,  # Terminal success state
                                    "handoff_status": HandoffStatus.METADATA_COMPLETE,
                                }
                            )

                        # Flush batch when threshold reached
                        if len(pending_updates) >= BATCH_FLUSH_SIZE:
                            _batch_flush_updates(pending_updates)
                            pending_updates = []

                        progress.worker_complete(worker_id, status, elapsed)

                        # Store results
                        with results_lock:
                            results.append(
                                {
                                    "entry_id": entry.entry_id,
                                    "status": status,
                                    "stats": entry_stats,
                                    "error": failed_reason,
                                }
                            )

                        with samples_lock:
                            all_samples.extend(entry_samples)

                # Launch workers
                executor = ThreadPoolExecutor(max_workers=effective_workers)
                try:
                    futures = [
                        executor.submit(worker_func, i)
                        for i in range(effective_workers)
                    ]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Worker error: {e}")
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)

        # Post-processing with progress feedback (prevents "654/655 hang" UX issue)
        from rich.console import Console

        console = Console()

        with console.status("[bold cyan]Finalizing: aggregating results...") as status:
            # Aggregate stats
            total_extracted = sum(r["stats"].get("extracted", 0) for r in results)
            total_valid = sum(r["stats"].get("valid", 0) for r in results)
            total_after_filter = len(all_samples)
            successful = sum(1 for r in results if r["status"] == "completed")
            failed = sum(1 for r in results if r["status"] == "failed")

            # Store aggregated results
            if all_samples:
                status.update("[bold cyan]Finalizing: writing to workspace...")

                from datetime import datetime

                from lobster.services.data_access.workspace_content_service import (
                    ContentType,
                    MetadataContent,
                )

                content = MetadataContent(
                    identifier=output_key,
                    content_type="filtered_samples",
                    description=f"Batch filtered samples: {filter_criteria or 'no filter'}",
                    data={
                        "samples": all_samples,
                        "filter_criteria": filter_criteria,
                        "stats": results,
                    },
                    source="metadata_assistant",
                    cached_at=datetime.now().isoformat(),
                )
                workspace_service.write_content(content, ContentType.METADATA)
                data_manager.metadata_store[output_key] = {
                    "samples": all_samples,
                    "filter_criteria": filter_criteria,
                    "stats": results,
                }

        retention = (
            (total_after_filter / total_extracted * 100) if total_extracted > 0 else 0
        )
        validation_rate = (
            (total_valid / total_extracted * 100) if total_extracted > 0 else 0
        )
        total_time = time.time() - start_time

        response = f"""## Queue Processing Complete (Parallel Mode)
**Entries Processed**: {len(results)}
**Successful**: {successful}
**Failed**: {failed}
**Samples Extracted**: {total_extracted}
**Samples Valid**: {total_valid} ({validation_rate:.1f}%)
**Samples After Filter**: {total_after_filter}
**Retention Rate**: {retention:.1f}%
**Processing Time**: {total_time:.1f}s ({len(results) / total_time * 60:.1f} entries/min)
**Output Key**: {output_key}

Use `write_to_workspace(identifier="{output_key}", workspace="metadata", output_format="csv")` to export as CSV.
"""
        return response

    @tool
    def process_metadata_queue(
        status_filter: str = "handoff_ready",
        filter_criteria: str = None,
        max_entries: int = None,
        output_key: str = "aggregated_filtered_samples",
        parallel_workers: int = None,
        min_disease_coverage: float = 0.5,
        strict_disease_validation: bool = True,
        reprocess_completed: bool = False,
    ) -> str:
        """
        Batch process publication queue entries and aggregate sample-level metadata.

        CRITICAL status_filter behavior:
        - 'handoff_ready' (DEFAULT): Entries with sra_*_samples workspace files
        - 'metadata_enriched': NO workspace files - will return 0 samples
        - 'completed': Use with reprocess_completed=True to re-process

        Args:
            status_filter: Queue status (default: 'handoff_ready')
            filter_criteria: Sample filter. Syntax: see filter_samples_by docstring. None=no filter.
            max_entries: Max entries (None=all)
            output_key: Workspace key for output CSV
            parallel_workers: >1 for parallel processing
            min_disease_coverage: Min coverage rate 0.0-1.0 (default 0.5)
            strict_disease_validation: Fail on low coverage (default True)
            reprocess_completed: Re-process completed entries (default False)

        Returns:
            Processing summary with sample counts and workspace output location

        Example:
            process_metadata_queue(filter_criteria="16S V4 human fecal CRC")
        """
        try:
            from lobster.services.data_access.workspace_content_service import (
                ContentType,
                MetadataContent,
            )

            queue = data_manager.publication_queue

            # Handle reprocess_completed logic
            status_lower = status_filter.lower()

            if status_lower == "completed" and not reprocess_completed:
                # User is trying to process completed entries without explicit flag
                return (
                    "⚠️ Cannot process 'completed' entries without reprocess_completed=True.\n\n"
                    "Completed entries have already been processed. To re-process them:\n"
                    "```\n"
                    "process_metadata_queue(\n"
                    '    status_filter="completed",\n'
                    "    reprocess_completed=True,\n"
                    f'    filter_criteria="{filter_criteria or "16S human fecal"}"\n'
                    ")\n"
                    "```"
                )

            # Get entries based on status_filter
            entries = queue.list_entries(status=PublicationStatus(status_lower))

            # If reprocess_completed=True and not already filtering by completed,
            # also include completed entries
            if reprocess_completed and status_lower != "completed":
                completed_entries = queue.list_entries(
                    status=PublicationStatus.COMPLETED
                )
                # Merge, avoiding duplicates
                entry_ids = {e.entry_id for e in entries}
                for ce in completed_entries:
                    if ce.entry_id not in entry_ids:
                        entries.append(ce)

                logger.info(
                    f"Reprocess mode enabled: {len(entries)} total entries "
                    f"(including {len(completed_entries)} completed)"
                )

            if max_entries:
                entries = entries[:max_entries]

            if not entries:
                if reprocess_completed:
                    return (
                        f"No entries found with status '{status_filter}' or 'completed'"
                    )
                logger.info(f"No entries found with status '{status_filter}'")
                return f"No entries found with status '{status_filter}'"

            # Route to parallel processing if requested and available
            if parallel_workers and parallel_workers > 1 and PROGRESS_UI_AVAILABLE:
                logger.info(
                    f"Using parallel processing: {len(entries)} entries, "
                    f"{parallel_workers} workers, Rich UI enabled"
                )
                return _process_queue_with_progress(
                    entries,
                    filter_criteria,
                    output_key,
                    parallel_workers,
                    min_disease_coverage,
                    strict_disease_validation,
                )
            elif (
                parallel_workers and parallel_workers > 1 and not PROGRESS_UI_AVAILABLE
            ):
                logger.warning(
                    "Parallel processing requested but Rich UI not available. "
                    "Falling back to sequential processing."
                )

            # Log batch processing start (sequential mode)
            logger.info(
                f"Starting batch processing: {len(entries)} entries "
                f"(filter: '{filter_criteria or 'none'}', mode: sequential)"
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
                        if not (
                            ws_key.startswith("sra_") and ws_key.endswith("_samples")
                        ):
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
                            samples, validation_result, quality_stats = (
                                _extract_samples_from_workspace(ws_data)
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
                            entry_flag_counts[flag] = (
                                entry_flag_counts.get(flag, 0) + count
                            )

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

                    if filter_criteria and entry_samples and metadata_filtering_service:
                        samples_before_filter = len(entry_samples)
                        parsed = metadata_filtering_service.parse_criteria(
                            filter_criteria
                        )
                        # Add disease validation parameters
                        parsed["min_disease_coverage"] = min_disease_coverage
                        parsed["strict_disease_validation"] = strict_disease_validation
                        metadata_filtering_service.disease_extractor = (
                            extract_disease_with_fallback
                        )
                        try:
                            entry_samples, _, _ = (
                                metadata_filtering_service.apply_filters(
                                    entry_samples, parsed
                                )
                            )
                            logger.debug(
                                f"Entry {entry.entry_id}: Filter applied - "
                                f"{len(entry_samples)}/{samples_before_filter} samples retained"
                            )
                        except ValueError as e:
                            if "Insufficient disease data" in str(e):
                                # Extract coverage percentage from error message
                                coverage_match = re.search(
                                    r"(\d+\.\d+)% coverage", str(e)
                                )
                                coverage = (
                                    float(coverage_match.group(1))
                                    if coverage_match
                                    else 0.0
                                )

                                # Count cached publications (for cost/time estimate)
                                pub_groups = defaultdict(set)
                                for sample in entry_samples:
                                    pub_id = sample.get("publication_entry_id")
                                    if pub_id:
                                        pub_groups[pub_id].add(pub_id)

                                num_pubs = len(pub_groups)
                                estimated_cost = num_pubs * 0.003
                                estimated_time = num_pubs * 2

                                # Return to supervisor with suggestion (DON'T auto-enrich)
                                return f"""❌ Disease validation failed: {coverage:.1f}% coverage < 50% threshold

I can attempt automatic enrichment to extract disease from publication abstracts/methods.

**Estimated improvement**: +30-60% coverage (based on {num_pubs} cached publications)
**Cost**: ~${estimated_cost:.2f} (LLM calls)
**Time**: ~{estimated_time}s

Should I proceed with automatic enrichment?"""
                            else:
                                raise

                    stats["total_after_filter"] += len(entry_samples)

                    # Aggregate quality flags into batch stats
                    for flag, count in entry_flag_counts.items():
                        stats["flag_counts"][flag] = (
                            stats["flag_counts"].get(flag, 0) + count
                        )

                    # Track samples needing manual body_site review
                    for qs in entry_quality_stats:
                        flagged_ids = qs.get("flagged_sample_ids", {})
                        if "missing_body_site" in flagged_ids:
                            stats["samples_needing_manual_review"].extend(
                                flagged_ids["missing_body_site"]
                            )

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

                    # Update entry status to terminal success state
                    queue.update_status(
                        entry.entry_id,
                        PublicationStatus.COMPLETED,  # Terminal success state
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

            # Compute study-level statistics for batch effect awareness
            study_stats = defaultdict(int)
            study_to_publications = defaultdict(set)

            for sample in all_samples:
                study_id = sample.get("study_accession", "unknown")
                pub_id = sample.get("publication_entry_id", "unknown")

                study_stats[study_id] += 1
                study_to_publications[study_id].add(pub_id)

            # Identify potential batch effect risks
            total_samples = len(all_samples)
            warnings = []

            # Skip warnings if no samples or no valid study IDs
            if total_samples == 0 or not study_stats:
                warnings = []
                study_stats = {}
                multi_pub_studies = {}
            else:
                # Warning 1: Studies spanning multiple publications
                multi_pub_studies = {
                    study_id: len(pubs)
                    for study_id, pubs in study_to_publications.items()
                    if len(pubs) > 1
                }

                if multi_pub_studies:
                    study_list = ", ".join(
                        [
                            f"{s} ({n} pubs)"
                            for s, n in list(multi_pub_studies.items())[:3]
                        ]
                    )
                    warnings.append(
                        f"⚠️ {len(multi_pub_studies)} studies appear in multiple publications: {study_list}. "
                        f"This may indicate overlapping datasets or batch effects."
                    )

                # Warning 2: Dominant study (>50% of samples)
                dominant_studies = {
                    study_id: count
                    for study_id, count in study_stats.items()
                    if count > total_samples * 0.5
                }

                if dominant_studies:
                    for study_id, count in dominant_studies.items():
                        pct = (count / total_samples) * 100
                        warnings.append(
                            f"⚠️ Study {study_id} dominates dataset: {count}/{total_samples} samples ({pct:.1f}%). "
                            f"Consider batch effect correction using study_accession field."
                        )

                # Warning 3: Highly imbalanced study sizes (coefficient of variation > 1.5)
                if len(study_stats) > 1:
                    counts = list(study_stats.values())
                    mean_count = sum(counts) / len(counts)
                    variance = sum((x - mean_count) ** 2 for x in counts) / len(counts)
                    std_dev = variance**0.5
                    cv = std_dev / mean_count if mean_count > 0 else 0

                    if cv > 1.5:
                        warnings.append(
                            f"⚠️ Highly imbalanced study sizes (CV={cv:.2f}). "
                            f"Range: {min(counts)}-{max(counts)} samples per study. "
                            f"Consider stratified analysis or batch correction."
                        )

                # Log all warnings
                for warning in warnings:
                    logger.warning(warning)

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
            logger.debug(
                f"  Total samples valid: {stats['total_valid']} ({validation_rate:.1f}%)"
            )
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

            # Build response - check for 0 samples scenario first
            if stats["total_extracted"] == 0:
                response = f"""⚠️ **0 Samples Extracted** (status='{status_filter}', {len(entries)} entries, {stats["with_samples"]} with metadata)

Likely cause: Wrong status_filter. Use `status_filter='handoff_ready'` (default) for entries ready for processing. 'completed' entries have no actionable metadata.

Fix: `process_metadata_queue(status_filter="handoff_ready", filter_criteria="{filter_criteria or ""}", output_key="{output_key}")`
"""
                return response

            # Check if samples were extracted but ALL filtered out
            if (
                stats["total_after_filter"] == 0
                and stats["total_extracted"] > 0
                and filter_criteria
            ):
                response = f"""⚠️ **All {stats["total_extracted"]} Samples Filtered Out** (filter: '{filter_criteria}')

{stats["total_valid"]}/{stats["total_extracted"]} valid, 0 after filter. Note: sample_type filters (fecal/gut) are NOT YET IMPLEMENTED - only 16S/shotgun and host filters work.

Fix: Run without filter first to inspect data: `process_metadata_queue(status_filter="{status_filter}", filter_criteria=None, output_key="unfiltered")`
"""
                return response

            # Normal response for successful processing
            response = f"""## Queue Processing Complete
**Entries Processed**: {stats["processed"]}
**Successful**: {stats["processed"] - stats["failed"]}
**Failed**: {stats["failed"]}
**Entries With Samples**: {stats["with_samples"]}
**Samples Extracted**: {stats["total_extracted"]}
**Samples Valid**: {stats["total_valid"]} ({validation_rate:.1f}%)
**Samples After Filter**: {stats["total_after_filter"]}
**Retention Rate**: {retention:.1f}%
**Validation**: {stats["validation_errors"]} errors, {stats["validation_warnings"]} warnings
**Output Key**: {output_key}
"""

            # Add study-level statistics section
            if study_stats:
                response += "\n## Study-Level Statistics\n"
                response += f"**Unique Studies**: {len(study_stats)}\n"
                if study_stats:
                    study_counts = list(study_stats.values())
                    avg_samples = sum(study_counts) / len(study_counts)
                    response += f"**Samples Per Study**: {min(study_counts)}-{max(study_counts)} (avg: {avg_samples:.1f})\n"
                response += (
                    f"**Studies in Multiple Publications**: {len(multi_pub_studies)}\n"
                )
                response += "\n## Batch Effect Warnings\n"

                # Append warnings if any
                if warnings:
                    for warning in warnings:
                        response += f"\n{warning}\n"
                else:
                    response += "\n✅ No major batch effect risks detected.\n"

                response += """
## Recommendations
- Use **study_accession** field for batch effect correction in downstream analysis
- Consider PERMANOVA or ComBat-seq for batch adjustment
- Stratify analyses by study if imbalanced
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

            response += f'\nUse `write_to_workspace(identifier="{output_key}", workspace="metadata", output_format="csv")` to export as CSV.\n'

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

            return (
                f"✓ Updated {entry_id}: handoff_status={handoff_status or 'unchanged'}"
            )
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            return f"❌ Error updating status: {str(e)}"

    @tool
    def enrich_samples_with_disease(
        workspace_key: str,
        enrichment_mode: str = "hybrid",
        manual_mappings: Optional[str] = None,
        confidence_threshold: float = 0.8,
        auto_retry_validation: bool = True,
        dry_run: bool = False,
    ) -> str:
        """
        Enrich samples with missing disease annotation from publication context.

        Use when disease validation fails (<50% coverage). Extracts from abstracts/methods
        or applies manual mappings.

        Args:
            workspace_key: Metadata workspace key (e.g., "aggregated_filtered_samples")
            enrichment_mode: "column_scan"|"llm_auto"|"manual"|"hybrid" (default: hybrid)
            manual_mappings: JSON mapping pub IDs to diseases (e.g., '{"pub_queue_doi_10_1234": "crc"}')
            confidence_threshold: Min LLM confidence 0.0-1.0 (default: 0.8)
            auto_retry_validation: Re-run validation after enrichment (default: True)
            dry_run: Preview without saving (default: False)

        Returns:
            Enrichment report: coverage improvement, per-phase results, validation status

        Example:
            enrich_samples_with_disease(workspace_key="aggregated_filtered_samples")
        """
        # Step 1: Load samples from workspace
        if workspace_key not in data_manager.metadata_store:
            return f"❌ Workspace key '{workspace_key}' not found in metadata_store"

        workspace_data = data_manager.metadata_store[workspace_key]
        samples = workspace_data.get("samples", [])

        if not samples:
            return f"❌ No samples found in workspace key '{workspace_key}'"

        # Step 2: Calculate initial coverage
        initial_count = sum(1 for s in samples if s.get("disease"))
        initial_coverage = (initial_count / len(samples)) * 100 if samples else 0

        # Step 3: Parse manual mappings
        mappings_dict = {}
        if manual_mappings:
            try:
                mappings_dict = json.loads(manual_mappings)
            except json.JSONDecodeError as e:
                return f"❌ Invalid manual_mappings JSON: {e}"

        # Step 4: Run enrichment phases
        report = [
            "## Disease Enrichment Report",
            f"**Workspace Key**: {workspace_key}",
            f"**Total Samples**: {len(samples)}",
            f"**Initial Coverage**: {initial_coverage:.1f}% ({initial_count}/{len(samples)})",
            f"**Mode**: {enrichment_mode}",
            f"**Dry Run**: {dry_run}",
            "",
            "### Enrichment Phases",
            "",
        ]

        total_enriched = 0

        # Make a copy of samples for dry_run to avoid modifying original
        working_samples = [s.copy() for s in samples] if dry_run else samples

        # Phase 1: Column re-scan (always run unless manual-only mode)
        if enrichment_mode in ["column_scan", "hybrid"]:
            report.append("#### Phase 1: Column Re-scan")
            count, log = phase1_column_rescan(working_samples)
            total_enriched += count
            report.extend(log)
            report.append(f"**Phase 1 Result**: +{count} samples")
            report.append("")

        # Phase 2: LLM abstract extraction (unless manual-only mode)
        if enrichment_mode in ["llm_auto", "hybrid"]:
            report.append("#### Phase 2: LLM Abstract Extraction")
            count, log = phase2_llm_abstract_extraction(
                working_samples, data_manager, confidence_threshold, llm
            )
            total_enriched += count
            report.extend(log)
            report.append(f"**Phase 2 Result**: +{count} samples")
            report.append("")

            # Phase 3: LLM methods extraction (only if hybrid and still low coverage)
            current_coverage = ((initial_count + total_enriched) / len(samples)) * 100
            if current_coverage < 50.0 and enrichment_mode == "hybrid":
                report.append(
                    "#### Phase 3: LLM Methods Extraction (triggered by low coverage)"
                )
                count, log = phase3_llm_methods_extraction(
                    working_samples, data_manager, confidence_threshold, llm
                )
                total_enriched += count
                report.extend(log)
                report.append(f"**Phase 3 Result**: +{count} samples")
                report.append("")

        # Phase 4: Manual mappings
        if enrichment_mode in ["manual", "hybrid"] and mappings_dict:
            report.append("#### Phase 4: Manual Mappings")
            count, log = phase4_manual_mappings(working_samples, mappings_dict)
            total_enriched += count
            report.extend(log)
            report.append(f"**Phase 4 Result**: +{count} samples")
            report.append("")

        # Step 5: Calculate final coverage
        final_count = sum(1 for s in working_samples if s.get("disease"))
        final_coverage = (final_count / len(samples)) * 100 if samples else 0
        improvement = final_coverage - initial_coverage

        report.append("### Summary")
        report.append(
            f"**Initial Coverage**: {initial_coverage:.1f}% ({initial_count} samples)"
        )
        report.append(
            f"**Final Coverage**: {final_coverage:.1f}% ({final_count} samples)"
        )
        report.append(
            f"**Improvement**: +{improvement:.1f}% (+{total_enriched} samples)"
        )
        report.append("**Validation Threshold**: 50.0%")
        report.append("")

        # Step 6: Save enriched samples (if not dry_run)
        if not dry_run and total_enriched > 0:
            # Update metadata_store with enriched samples
            data_manager.metadata_store[workspace_key] = workspace_data
            report.append("✅ **Changes saved** to metadata_store")

            # Also save to workspace file
            try:
                from lobster.services.data_access.workspace_content_service import (
                    ContentType,
                    MetadataContent,
                    WorkspaceContentService,
                )

                workspace_service = WorkspaceContentService(data_manager)
                content = MetadataContent(
                    identifier=workspace_key,
                    content_type="enriched_samples",
                    description=f"Disease-enriched samples: +{total_enriched} samples",
                    data=workspace_data,
                    source="metadata_assistant",
                    cached_at=datetime.now().isoformat(),
                )
                workspace_service.write_content(content, ContentType.METADATA)
                report.append("✅ **Changes saved** to workspace/metadata/")
            except Exception as e:
                logger.warning(f"Failed to save to workspace file: {e}")
                report.append(
                    f"⚠️ **Warning**: Saved to metadata_store but workspace save failed: {e}"
                )

            report.append("")
        elif dry_run:
            report.append("ℹ️ **Dry run**: Changes NOT saved (preview only)")
            report.append("")

        # Step 7: Auto-retry validation (if enabled and not dry_run)
        if auto_retry_validation and not dry_run and total_enriched > 0:
            report.append("### Re-validation")

            if final_coverage >= 50.0:
                report.append(
                    f"✅ **Validation PASSED**: {final_coverage:.1f}% coverage ≥ 50% threshold"
                )
                report.append(
                    "You can now proceed with filtering using this enriched dataset."
                )
            else:
                missing_count = len(samples) - final_count
                report.append(
                    f"❌ **Validation FAILED**: {final_coverage:.1f}% coverage < 50% threshold"
                )
                report.append(f"**Samples still missing disease**: {missing_count}")
                report.append("")
                report.append("**Next Steps**:")
                report.append("  1. Provide manual mappings for remaining publications")
                report.append(
                    f"  2. Lower threshold: min_disease_coverage={final_coverage / 100:.2f}"
                )
                report.append(
                    "  3. Skip disease filtering (omit disease terms from filter_criteria)"
                )

        return "\n".join(report)

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
            "validation_errors": [],  # Track errors for aggregated summary
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
                # Track validation errors by type (aggregated, not per-sample)
                # These are data quality issues, not system errors
                for error in sample_result.errors:
                    quality_stats["validation_errors"].append(error)
                    logger.debug(f"Sample validation failed: {error}")

        # Log aggregated validation error summary (once per batch, not per sample)
        if quality_stats["validation_errors"]:
            # Group errors by type for concise summary
            error_types: Dict[str, int] = {}
            for error in quality_stats["validation_errors"]:
                # Extract error type from message (e.g., "Field 'organism_name':" -> "organism_name")
                if "Field '" in error:
                    field = error.split("Field '")[1].split("'")[0]
                    error_types[f"missing_{field}"] = (
                        error_types.get(f"missing_{field}", 0) + 1
                    )
                elif "No download URLs" in error:
                    error_types["no_download_url"] = (
                        error_types.get("no_download_url", 0) + 1
                    )
                else:
                    error_types["other"] = error_types.get("other", 0) + 1

            # Log summary at WARNING level (data quality issue, not system error)
            invalid_count = len(raw_samples) - len(valid_samples)
            error_summary = ", ".join(
                f"{k}: {v}" for k, v in sorted(error_types.items())
            )
            logger.warning(
                f"Validation summary: {invalid_count} samples excluded "
                f"({error_summary})"
            )

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

    # =========================================================================
    # Helper Functions for filter_samples_by
    # =========================================================================
    # NOTE: _parse_filter_criteria and _apply_metadata_filters have been moved to
    # lobster/services/metadata/metadata_filtering_service.py (MetadataFilteringService)
    # =========================================================================

    def _combine_analysis_steps(
        irs: List[AnalysisStep], operation: str, description: str
    ) -> AnalysisStep:
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
                output_entities=[],
            )

        # Combine code templates
        combined_code = "\n\n".join(
            [
                f"# Step {i + 1}: {ir.description}\n{ir.code_template}"
                for i, ir in enumerate(irs)
            ]
        )

        # Combine imports (deduplicate)
        all_imports = []
        for ir in irs:
            all_imports.extend(ir.imports)
        combined_imports = list(set(all_imports))

        # Combine parameters
        combined_params = {}
        for i, ir in enumerate(irs):
            combined_params[f"step_{i + 1}"] = ir.parameters

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
            output_entities=[{"type": "filtered_metadata", "name": "output_metadata"}],
        )

    def _format_filtering_report(
        workspace_key: str,
        filter_criteria: str,
        parsed_criteria: Dict[str, Any],
        original_count: int,
        final_count: int,
        retention_rate: float,
        stats_list: List[Dict[str, Any]],
        filtered_metadata: "pd.DataFrame",
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
        status_icon = (
            "✅" if retention_rate > 50 else ("⚠️" if retention_rate > 10 else "❌")
        )

        report_lines = [
            f"{status_icon} Sample Filtering Complete\n",
            f"**Workspace Key**: {workspace_key}",
            f"**Filter Criteria**: {filter_criteria}",
            f"**Original Samples**: {original_count}",
            f"**Filtered Samples**: {final_count}",
            f"**Retention Rate**: {retention_rate:.1f}%\n",
            "## Filters Applied\n",
        ]

        # List applied filters
        if parsed_criteria["check_16s"]:
            report_lines.append("- ✓ 16S amplicon detection")
        if parsed_criteria["host_organisms"]:
            report_lines.append(
                f"- ✓ Host organism: {', '.join(parsed_criteria['host_organisms'])}"
            )
        if parsed_criteria["sample_types"]:
            report_lines.append(
                f"- ✓ Sample type: {', '.join(parsed_criteria['sample_types'])}"
            )
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
                report_lines.append(
                    f"\n*... and {len(filtered_metadata) - 5} more samples*"
                )
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

    # execute_custom_code is created via factory at line ~273 (v2.7+: unified tool)
    # See create_execute_custom_code_tool() in lobster/tools/custom_code_tool.py
    # Post-processor: metadata_store_post_processor handles sample persistence

    # =========================================================================
    # Tool 11: Semantic Tissue Term Standardization (optional, requires vector-search)
    # =========================================================================

    _vector_service = None

    def _get_vector_service():
        nonlocal _vector_service
        if _vector_service is None:
            from lobster.services.vector.service import VectorSearchService

            _vector_service = VectorSearchService()
        return _vector_service

    @tool
    def standardize_tissue_term(
        term: str,
        k: int = 5,
        min_confidence: float = 0.5,
    ) -> str:
        """Standardize a tissue term to Uberon ontology using semantic vector search.

        Returns ranked matches with confidence scores. Use this tool when you need
        to map free-text tissue descriptions to standardized Uberon ontology terms
        for metadata harmonization.

        Args:
            term: Free-text tissue term to standardize (e.g., "brain cortex", "liver tissue")
            k: Maximum number of matches to return (default: 5)
            min_confidence: Minimum confidence threshold for matches (0.0-1.0, default: 0.5)

        Returns:
            Formatted string with ranked Uberon ontology matches and confidence scores
        """
        try:
            matches = _get_vector_service().match_ontology(term, "uberon", k=k)

            # Filter by confidence threshold
            filtered = [m for m in matches if m.score >= min_confidence]

            # Build stats dict
            top_matches = [
                {"term": m.term, "ontology_id": m.ontology_id, "score": m.score}
                for m in filtered
            ]
            stats = {
                "term": term,
                "n_matches": len(filtered),
                "top_matches": top_matches,
                "best_match": top_matches[0] if top_matches else None,
            }

            # Create IR for provenance
            ir = AnalysisStep(
                operation="standardize_tissue_term",
                tool_name="standardize_tissue_term",
                library="lobster.services.vector",
                description="Semantic tissue term standardization via Uberon ontology",
                code_template=(
                    'from lobster.services.vector.service import VectorSearchService\n'
                    'service = VectorSearchService()\n'
                    'matches = service.match_ontology("{{ term }}", "uberon", k={{ k }})\n'
                    'filtered = [m for m in matches if m.score >= {{ min_confidence }}]'
                ),
                imports=["from lobster.services.vector.service import VectorSearchService"],
                parameters={"term": term, "k": k, "min_confidence": min_confidence},
                parameter_schema={},
            )

            # Log with mandatory IR
            data_manager.log_tool_usage(
                "standardize_tissue_term", {"term": term, "k": k}, stats, ir=ir
            )

            if not filtered:
                return f"No matches found above confidence {min_confidence} for '{term}'."

            lines = [f"Tissue standardization for '{term}':"]
            for i, m in enumerate(filtered, 1):
                lines.append(f"  {i}. {m.term} ({m.ontology_id}) - score: {m.score:.4f}")
            return "\n".join(lines)

        except Exception as e:
            return f"Error standardizing tissue term '{term}': {e}"

    # =========================================================================
    # Tool 12: Semantic Disease Term Standardization (optional, requires ontology service)
    # =========================================================================

    @tool
    def standardize_disease_term(
        term: str,
        k: int = 3,
        min_confidence: float = 0.7,
    ) -> str:
        """Standardize a disease term to MONDO ontology using DiseaseOntologyService.

        Routes through existing disease matching infrastructure for semantic or
        keyword matching. Use this tool when you need to map free-text disease
        descriptions to standardized MONDO ontology terms.

        Args:
            term: Free-text disease term to standardize (e.g., "colon cancer", "Crohn's disease")
            k: Maximum number of matches to return (default: 3)
            min_confidence: Minimum confidence threshold (0.0-1.0, default: 0.7)

        Returns:
            Formatted string with ranked MONDO ontology matches and confidence scores
        """
        if not HAS_ONTOLOGY_SERVICE:
            return "DiseaseOntologyService not available. Install lobster-metadata package."

        try:
            matches = DiseaseOntologyService.get_instance().match_disease(
                term, k=k, min_confidence=min_confidence
            )

            # Build stats dict
            match_dicts = [
                {
                    "disease_id": m.disease_id,
                    "name": m.name,
                    "confidence": m.confidence,
                }
                for m in matches
            ]
            stats = {
                "term": term,
                "n_matches": len(matches),
                "matches": match_dicts,
                "best_match": match_dicts[0] if match_dicts else None,
            }

            # Create IR for provenance
            ir = AnalysisStep(
                operation="standardize_disease_term",
                tool_name="standardize_disease_term",
                library="lobster.services.metadata.disease_ontology_service",
                description="Disease term standardization via MONDO ontology",
                code_template=(
                    'from lobster.services.metadata.disease_ontology_service import DiseaseOntologyService\n'
                    'service = DiseaseOntologyService.get_instance()\n'
                    'matches = service.match_disease("{{ term }}", k={{ k }}, min_confidence={{ min_confidence }})'
                ),
                imports=["from lobster.services.metadata.disease_ontology_service import DiseaseOntologyService"],
                parameters={"term": term, "k": k, "min_confidence": min_confidence},
                parameter_schema={},
            )

            # Log with mandatory IR
            data_manager.log_tool_usage(
                "standardize_disease_term", {"term": term, "k": k}, stats, ir=ir
            )

            if not matches:
                return f"No matches found above confidence {min_confidence} for '{term}'."

            lines = [f"Disease standardization for '{term}':"]
            for i, m in enumerate(matches, 1):
                lines.append(
                    f"  {i}. {m.name} ({m.disease_id}) - confidence: {m.confidence:.4f}"
                )
            return "\n".join(lines)

        except Exception as e:
            return f"Error standardizing disease term '{term}': {e}"

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
        enrich_samples_with_disease,
        # Shared workspace tools
        get_content_from_workspace,
        write_to_workspace,
        # Custom code execution for sample-level operations
        execute_custom_code,
        # Cross-database ID mapping (UniProt + Ensembl)
        map_cross_database_ids,
    ]

    if MICROBIOME_FEATURES_AVAILABLE:
        tools.append(filter_samples_by)

    # Semantic standardization tools (conditional on optional deps)
    if HAS_VECTOR_SEARCH:
        tools.append(standardize_tissue_term)
    if HAS_ONTOLOGY_SERVICE:
        tools.append(standardize_disease_term)

    # =========================================================================
    # System Prompt
    # =========================================================================

    system_prompt = create_metadata_assistant_prompt()

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
