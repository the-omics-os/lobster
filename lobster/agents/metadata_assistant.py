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
from typing import List

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.metadata_standardization_service import (
    MetadataStandardizationService,
)
from lobster.tools.sample_mapping_service import SampleMappingService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def metadata_assistant(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "metadata_assistant",
    handoff_tools: List = None,
):
    """Create metadata assistant agent for metadata operations.

    This agent provides 4 specialized tools for metadata operations:
    1. map_samples_by_id - Cross-dataset sample ID mapping
    2. read_sample_metadata - Extract and format sample metadata
    3. standardize_sample_metadata - Convert to Pydantic schemas
    4. validate_dataset_content - Validate dataset completeness

    Args:
        data_manager: DataManagerV2 instance
        callback_handler: Optional callback handler for LLM
        agent_name: Agent name for identification
        handoff_tools: Optional list of handoff tools for coordination

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

    logger.info("metadata_assistant agent initialized")

    # =========================================================================
    # Tool 1: Sample ID Mapping
    # =========================================================================

    @tool
    def map_samples_by_id(
        source_identifier: str,
        target_identifier: str,
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
            source_identifier: Source dataset identifier (e.g., "geo_gse12345")
            target_identifier: Target dataset identifier (e.g., "geo_gse67890")
            min_confidence: Minimum confidence threshold for fuzzy matches (0.0-1.0, default: 0.75)
            strategies: Comma-separated strategies to use (default: "all", options: "exact,fuzzy,pattern,metadata")

        Returns:
            Formatted markdown report with match results, confidence scores, and unmapped samples
        """
        try:
            logger.info(
                f"Mapping samples: {source_identifier} → {target_identifier} "
                f"(min_confidence={min_confidence})"
            )

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

            # Call mapping service
            result = sample_mapping_service.map_samples_by_id(
                source_identifier=source_identifier,
                target_identifier=target_identifier,
                strategies=strategy_list,
            )

            # Log provenance
            data_manager.log_tool_usage(
                tool_name="map_samples_by_id",
                parameters={
                    "source": source_identifier,
                    "target": target_identifier,
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
        identifier: str, fields: str = None, return_format: str = "summary"
    ) -> str:
        """
        Read and format sample-level metadata from a dataset.

        Use this tool to extract sample metadata in different formats:
        - "summary": Quick overview with field coverage percentages
        - "detailed": Complete metadata as JSON for programmatic access
        - "schema": Full metadata table for inspection

        Args:
            identifier: Dataset identifier (e.g., "geo_gse12345")
            fields: Optional comma-separated list of fields to extract (default: all fields)
            return_format: Output format (default: "summary", options: "summary,detailed,schema")

        Returns:
            Formatted metadata according to return_format specification
        """
        try:
            logger.info(f"Reading metadata for {identifier} (format: {return_format})")

            # Parse fields
            field_list = None
            if fields:
                field_list = [f.strip() for f in fields.split(",")]

            # Call standardization service
            result = metadata_standardization_service.read_sample_metadata(
                identifier=identifier, fields=field_list, return_format=return_format
            )

            # Log provenance
            data_manager.log_tool_usage(
                tool_name="read_sample_metadata",
                parameters={
                    "identifier": identifier,
                    "fields": fields,
                    "return_format": return_format,
                },
                result_summary={"format": return_format},
            )

            # Format output based on return_format
            if return_format == "summary":
                logger.info(f"Metadata summary generated for {identifier}")
                return result
            elif return_format == "detailed":
                logger.info(f"Detailed metadata extracted for {identifier}")
                return json.dumps(result, indent=2)
            elif return_format == "schema":
                logger.info(f"Metadata schema extracted for {identifier}")
                return result.to_markdown(index=True)
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
        identifier: str,
        target_schema: str,
        controlled_vocabularies: str = None,
    ) -> str:
        """
        Standardize sample metadata using Pydantic schemas for cross-dataset harmonization.

        Use this tool to convert raw metadata to standardized Pydantic schemas
        (TranscriptomicsMetadataSchema or ProteomicsMetadataSchema) with field
        normalization and controlled vocabulary enforcement.

        Args:
            identifier: Dataset identifier (e.g., "geo_gse12345")
            target_schema: Target schema type (options: "transcriptomics", "proteomics", "bulk_rna_seq", "single_cell", "mass_spectrometry", "affinity")
            controlled_vocabularies: Optional JSON string of controlled vocabularies (e.g., '{"condition": ["Control", "Treatment"]}')

        Returns:
            Standardization report with field coverage, validation errors, and warnings
        """
        try:
            logger.info(
                f"Standardizing metadata for {identifier} with {target_schema} schema"
            )

            # Parse controlled vocabularies if provided
            controlled_vocab_dict = None
            if controlled_vocabularies:
                try:
                    controlled_vocab_dict = json.loads(controlled_vocabularies)
                except json.JSONDecodeError as e:
                    return f"❌ Invalid controlled_vocabularies JSON: {str(e)}"

            # Call standardization service
            result = metadata_standardization_service.standardize_metadata(
                identifier=identifier,
                target_schema=target_schema,
                controlled_vocabularies=controlled_vocab_dict,
            )

            # Log provenance
            data_manager.log_tool_usage(
                tool_name="standardize_sample_metadata",
                parameters={
                    "identifier": identifier,
                    "target_schema": target_schema,
                    "controlled_vocabularies": controlled_vocabularies,
                },
                result_summary={
                    "valid_samples": len(result.standardized_metadata),
                    "validation_errors": len(result.validation_errors),
                    "warnings": len(result.warnings),
                },
            )

            # Format report
            report_lines = [
                "# Metadata Standardization Report\n",
                f"**Dataset:** {identifier}",
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
            logger.info(f"Standardization complete for {identifier}")
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
        identifier: str,
        expected_samples: int = None,
        required_conditions: str = None,
        check_controls: bool = True,
        check_duplicates: bool = True,
    ) -> str:
        """
        Validate dataset completeness and metadata quality before download or analysis.

        Use this tool to verify that a dataset meets minimum requirements:
        - Sample count verification
        - Condition presence check
        - Control sample detection
        - Duplicate ID check
        - Platform consistency check

        Args:
            identifier: Dataset identifier (e.g., "geo_gse12345")
            expected_samples: Minimum expected sample count (optional)
            required_conditions: Comma-separated list of required condition values (optional)
            check_controls: Whether to check for control samples (default: True)
            check_duplicates: Whether to check for duplicate sample IDs (default: True)

        Returns:
            Validation report with checks results, warnings, and recommendations
        """
        try:
            logger.info(f"Validating dataset content for {identifier}")

            # Parse required conditions
            required_condition_list = None
            if required_conditions:
                required_condition_list = [
                    c.strip() for c in required_conditions.split(",")
                ]

            # Call validation service
            result = metadata_standardization_service.validate_dataset_content(
                identifier=identifier,
                expected_samples=expected_samples,
                required_conditions=required_condition_list,
                check_controls=check_controls,
                check_duplicates=check_duplicates,
            )

            # Log provenance
            data_manager.log_tool_usage(
                tool_name="validate_dataset_content",
                parameters={
                    "identifier": identifier,
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
            )

            # Format report
            report_lines = [
                "# Dataset Validation Report\n",
                f"**Dataset:** {identifier}\n",
                "## Validation Checks",
                (
                    f"✅ Sample Count: {result.summary['total_samples']} samples"
                    if result.has_required_samples
                    else f"❌ Sample Count: {result.summary['total_samples']} samples (below minimum)"
                ),
                (
                    f"✅ Platform Consistency: Consistent"
                    if result.platform_consistency
                    else f"⚠️ Platform Consistency: Inconsistent"
                ),
                (
                    f"✅ No Duplicate IDs"
                    if not result.duplicate_ids
                    else f"❌ Duplicate IDs: {len(result.duplicate_ids)} found"
                ),
                (
                    f"✅ Control Samples: Detected"
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
            logger.info(f"Validation complete for {identifier}")
            return report

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return f"❌ Validation failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}", exc_info=True)
            return f"❌ Unexpected error during validation: {str(e)}"

    # =========================================================================
    # System Prompt
    # =========================================================================

    system_prompt = """
You are a metadata librarian and quality control specialist for multi-omics bioinformatics datasets, supporting pharmaceutical early research and drug discovery.

<Role>
Your expertise lies in cross-dataset metadata harmonization, sample ID mapping, metadata standardization, and dataset completeness validation.
You ensure datasets are ready for multi-omics integration by:
- Mapping sample IDs across datasets with different naming conventions
- Standardizing metadata to Pydantic schemas (Transcriptomics, Proteomics)
- Validating dataset completeness (samples, conditions, controls, duplicates)
- Detecting metadata quality issues early to prevent downstream failures

You work closely with:
- **Research Agent**: who discovers datasets and publications
- **Data Experts**: who download and preprocess datasets
- **Analysis Experts**: who need clean, harmonized metadata for multi-omics integration
</Role>

<Critical_Rules>
1. **METADATA OPERATIONS ONLY**: You do NOT download datasets, search literature, or perform analyses. Hand off to appropriate agents:
   - Literature search → research_agent
   - Dataset download → data_expert
   - Single-cell analysis → singlecell_expert
   - Bulk RNA-seq → bulk_rnaseq_expert
   - Proteomics → ms_proteomics_expert or affinity_proteomics_expert

2. **VALIDATE EARLY**: Before any metadata operation:
   - Check that the dataset exists in the workspace
   - Verify the dataset has sample metadata (obs)
   - Identify available metadata fields before standardization

3. **CONFIDENCE SCORES**: When mapping samples:
   - Exact matches: 100% confidence
   - Fuzzy matches: Report confidence score (0.75-1.0 typical)
   - Pattern matches: 90% confidence (normalized IDs)
   - Metadata-supported: 70-95% (based on field alignment)
   - Flag low-confidence matches (<0.75) for manual review

4. **CONTROLLED VOCABULARIES**: When standardizing metadata:
   - Enforce controlled vocabularies when provided
   - Normalize field names using MetadataValidationService
   - Flag non-standard values as warnings (not errors)
   - Report field coverage percentages

5. **DATASET COMPLETENESS**: When validating datasets:
   - Check minimum sample count requirements
   - Verify required conditions are present
   - Detect missing control samples (flag as warning)
   - Identify duplicate sample IDs (flag as error)
   - Check platform consistency across samples

6. **ACTIONABLE REPORTING**:
   - Always include confidence scores in mapping reports
   - Report unmapped samples with best candidates
   - Provide clear recommendations: "proceed", "manual review", or "skip"
   - Quantify issues: "15 of 50 samples missing required field 'condition'"

7. **HANDOFF TRIGGERS**:
   - Dataset not in workspace → data_expert (download first)
   - Metadata missing or incomplete → research_agent (verify dataset quality)
   - Complex multi-step operations → supervisor (coordinate workflow)
   - Ready for analysis → appropriate analysis expert
</Critical_Rules>

<Best_Practices>
- Use `read_sample_metadata()` first to understand available fields
- Prefer "summary" format for quick checks, "detailed" for programmatic access
- Use multiple mapping strategies (exact+fuzzy+pattern+metadata) for best results
- Validate datasets BEFORE download to avoid wasting time on poor-quality data
- Report validation issues clearly: "Dataset has 3 issues: ..."
- Include field coverage percentages when standardizing
- Flag low-confidence matches for manual review
</Best_Practices>

<Tools_Available>
1. **map_samples_by_id**: Cross-dataset sample ID mapping with 4 strategies
2. **read_sample_metadata**: Extract and format sample metadata (3 formats)
3. **standardize_sample_metadata**: Convert to Pydantic schemas with validation
4. **validate_dataset_content**: Check completeness and quality (5 checks)
</Tools_Available>

Today's date: {current_date}
"""

    # Combine tools
    tools = [
        map_samples_by_id,
        read_sample_metadata,
        standardize_sample_metadata,
        validate_dataset_content,
    ]

    if handoff_tools:
        tools.extend(handoff_tools)

    # Format system prompt with current date
    from datetime import date

    formatted_prompt = system_prompt.format(current_date=date.today().isoformat())

    # Create LangGraph agent
    return create_react_agent(
        model=llm, tools=tools, prompt=formatted_prompt, name=agent_name
    )
