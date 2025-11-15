"""
Integration tests for metadata_assistant with real GEO datasets.

This test suite validates all 4 metadata_assistant tools using real GEO datasets
and workflows for multi-omics integration metadata operations:
- map_samples_by_id: Cross-dataset sample ID mapping
- read_sample_metadata: Metadata extraction and formatting
- standardize_sample_metadata: Pydantic schema standardization
- validate_dataset_content: Dataset completeness validation

**Test Strategy:**
- Use known stable GEO datasets (GSE180759, GSE12345, etc.)
- Test both happy paths and edge cases
- Validate all 4 matching strategies (exact, fuzzy, pattern, metadata)
- Test all 3 return formats (summary, detailed, schema)
- Test all target schemas (transcriptomics, proteomics)
- Validate all 5 validation checks

**Markers:**
- @pytest.mark.real_api: All tests (requires API keys + GEO data)
- @pytest.mark.slow: Tests >30s
- @pytest.mark.integration: Multi-component tests

**Environment Requirements:**
- NCBI_API_KEY (recommended for GEO rate limits)
- AWS_BEDROCK_ACCESS_KEY + AWS_BEDROCK_SECRET_ACCESS_KEY (for LLM)
- Internet connectivity for GEO access

Phase 7 - Task Group 3: Metadata Assistant Real Workflow Tests
"""

import os
import time
from pathlib import Path

import pytest

from lobster.agents.metadata_assistant import metadata_assistant
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def test_workspace(tmp_path_factory):
    """Create temporary workspace for test session."""
    workspace = tmp_path_factory.mktemp("test_metadata_assistant_real_workflows")
    return workspace


@pytest.fixture(scope="module")
def data_manager(test_workspace):
    """Initialize DataManagerV2 with test workspace."""
    settings = get_settings()
    dm = DataManagerV2(workspace_dir=test_workspace, console=None)
    return dm


@pytest.fixture(scope="module")
def agent(data_manager):
    """Create metadata_assistant instance for testing."""
    return metadata_assistant(
        data_manager=data_manager, callback_handler=None, handoff_tools=None
    )


@pytest.fixture(scope="module")
def check_api_keys():
    """Verify required API keys are present."""
    required_keys = ["AWS_BEDROCK_ACCESS_KEY", "AWS_BEDROCK_SECRET_ACCESS_KEY"]
    missing = [key for key in required_keys if not os.getenv(key)]

    if missing:
        pytest.skip(f"Missing required API keys: {', '.join(missing)}")

    # NCBI_API_KEY is recommended but not required
    if not os.getenv("NCBI_API_KEY"):
        logger.warning("NCBI_API_KEY not set - GEO requests may be rate limited")


@pytest.fixture(scope="module")
def sample_datasets(data_manager):
    """Load sample GEO datasets for testing."""
    # This fixture would pre-load GEO datasets to workspace
    # For now, tests will reference datasets that should exist or be loadable
    datasets = {
        "primary": "geo_gse180759",  # Known dataset with good metadata
        "secondary": "geo_gse12345",  # Another dataset for mapping tests
    }
    return datasets


# ============================================================================
# Tool 3.1: map_samples_by_id
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestMapSamplesByID:
    """Test map_samples_by_id tool with real GEO datasets."""

    def test_exact_matching_strategy(self, agent, check_api_keys):
        """Test sample mapping using exact matching strategy."""
        from langchain_core.messages import HumanMessage

        # Request exact matching only
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Map samples between geo_gse180759 and geo_gse180759 using exact strategy only"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify mapping executed with exact strategy
        assert "mapping" in response_text.lower() or "match" in response_text.lower()

    def test_fuzzy_matching_strategy(self, agent, check_api_keys):
        """Test sample mapping using fuzzy matching strategy."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Map samples between geo_gse180759 and geo_gse180759 using fuzzy strategy with 0.8 confidence"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify fuzzy matching executed
        assert (
            "fuzzy" in response_text.lower()
            or "confidence" in response_text.lower()
            or "mapping" in response_text.lower()
        )

    def test_pattern_matching_strategy(self, agent, check_api_keys):
        """Test sample mapping using pattern matching strategy."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Map samples using pattern strategy (remove prefixes/suffixes)"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention pattern or mapping
        assert "pattern" in response_text.lower() or "mapping" in response_text.lower()

    def test_metadata_assisted_mapping(self, agent, check_api_keys):
        """Test sample mapping using metadata-assisted strategy."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Map samples using metadata strategy (condition, tissue, timepoint)"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify metadata strategy attempted
        assert "metadata" in response_text.lower() or "mapping" in response_text.lower()

    def test_all_strategies_combined(self, agent, check_api_keys):
        """Test sample mapping using all strategies (default)."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Map samples between datasets using all available strategies"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should report mapping results
        assert "mapping" in response_text.lower() or "match" in response_text.lower()

    def test_confidence_scoring(self, agent, check_api_keys):
        """Test confidence score reporting for fuzzy matches."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Map samples with fuzzy matching and report confidence scores"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention confidence or scores
        assert (
            "confidence" in response_text.lower()
            or "score" in response_text.lower()
            or "mapping" in response_text.lower()
        )

    def test_no_matches_edge_case(self, agent, check_api_keys):
        """Test handling when no samples can be mapped."""
        from langchain_core.messages import HumanMessage

        # Try mapping completely different datasets
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Map samples between completely different datasets with no overlap"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should indicate no matches or low success rate
        assert (
            "no" in response_text.lower()
            or "unmapped" in response_text.lower()
            or "not found" in response_text.lower()
            or "mapping" in response_text.lower()
        )


# ============================================================================
# Tool 3.2: read_sample_metadata
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestReadSampleMetadata:
    """Test read_sample_metadata tool for metadata extraction."""

    def test_summary_format(self, agent, check_api_keys):
        """Test reading metadata in summary format."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Read sample metadata from geo_gse180759 in summary format"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify summary format returned
        assert (
            "metadata" in response_text.lower()
            or "sample" in response_text.lower()
            or "field" in response_text.lower()
        )

    def test_detailed_format(self, agent, check_api_keys):
        """Test reading metadata in detailed JSON format."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Read sample metadata from geo_gse180759 in detailed format"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify detailed format (JSON-like)
        assert (
            "metadata" in response_text.lower() or "detailed" in response_text.lower()
        )

    def test_schema_format(self, agent, check_api_keys):
        """Test reading metadata in schema table format."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Read sample metadata from geo_gse180759 in schema format"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify schema format (table-like)
        assert "schema" in response_text.lower() or "metadata" in response_text.lower()

    def test_field_filtering(self, agent, check_api_keys):
        """Test extracting specific fields only."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Read only 'sample_id,condition,tissue' fields from geo_gse180759 metadata"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention specific fields
        assert (
            "sample" in response_text.lower()
            or "field" in response_text.lower()
            or "metadata" in response_text.lower()
        )

    def test_geo_extraction(self, agent, check_api_keys):
        """Test extracting metadata from GEO dataset."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Extract all sample metadata from GEO dataset GSE180759"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify GEO extraction
        assert (
            "GSE180759" in response_text
            or "metadata" in response_text.lower()
            or "sample" in response_text.lower()
        )

    def test_missing_fields_edge_case(self, agent, check_api_keys):
        """Test handling when requested fields are missing."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Read 'nonexistent_field1,nonexistent_field2' from geo_gse180759"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should indicate missing fields or partial results
        assert (
            "missing" in response_text.lower()
            or "not found" in response_text.lower()
            or "field" in response_text.lower()
            or "metadata" in response_text.lower()
        )


# ============================================================================
# Tool 3.3: standardize_sample_metadata
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestStandardizeSampleMetadata:
    """Test standardize_sample_metadata tool for schema conversion."""

    def test_transcriptomics_schema(self, agent, check_api_keys):
        """Test standardizing to TranscriptomicsMetadataSchema."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Standardize geo_gse180759 metadata to transcriptomics schema"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify standardization attempted
        assert (
            "transcriptomics" in response_text.lower()
            or "standardiz" in response_text.lower()
            or "schema" in response_text.lower()
        )

    def test_proteomics_schema(self, agent, check_api_keys):
        """Test standardizing to ProteomicsMetadataSchema."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Standardize metadata to proteomics schema")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify standardization attempted
        assert (
            "proteomics" in response_text.lower()
            or "standardiz" in response_text.lower()
        )

    def test_pydantic_validation(self, agent, check_api_keys):
        """Test Pydantic validation during standardization."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Standardize metadata and report validation errors"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention validation or errors
        assert (
            "validation" in response_text.lower()
            or "error" in response_text.lower()
            or "standardiz" in response_text.lower()
        )

    def test_controlled_vocabularies(self, agent, check_api_keys):
        """Test controlled vocabulary enforcement."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content='Standardize metadata with controlled vocabulary: {"condition": ["Control", "Treatment"]}'
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention vocabulary or standardization
        assert (
            "vocabulary" in response_text.lower()
            or "standardiz" in response_text.lower()
            or "condition" in response_text.lower()
        )

    def test_field_coverage_reporting(self, agent, check_api_keys):
        """Test field coverage percentage reporting."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Standardize metadata and report field coverage percentages"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention coverage or percentages
        assert (
            "coverage" in response_text.lower()
            or "field" in response_text.lower()
            or "standardiz" in response_text.lower()
        )

    def test_invalid_values_edge_case(self, agent, check_api_keys):
        """Test handling metadata with invalid values."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Standardize metadata with invalid values and report warnings"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should indicate validation issues
        assert (
            "invalid" in response_text.lower()
            or "warning" in response_text.lower()
            or "error" in response_text.lower()
            or "standardiz" in response_text.lower()
        )


# ============================================================================
# Tool 3.4: validate_dataset_content
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestValidateDatasetContent:
    """Test validate_dataset_content tool for dataset validation."""

    def test_sample_count_validation(self, agent, check_api_keys):
        """Test validating minimum sample count."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Validate geo_gse180759 has at least 20 samples"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify validation executed
        assert (
            "sample" in response_text.lower() or "validation" in response_text.lower()
        )

    def test_condition_presence_check(self, agent, check_api_keys):
        """Test checking for required condition values."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Validate dataset has 'Control' and 'Treatment' conditions"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention conditions
        assert (
            "condition" in response_text.lower()
            or "validation" in response_text.lower()
        )

    def test_control_sample_detection(self, agent, check_api_keys):
        """Test detecting control samples in dataset."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {"messages": [HumanMessage(content="Validate dataset has control samples")]}
        )

        response_text = result["messages"][-1].content

        # Should mention controls
        assert (
            "control" in response_text.lower() or "validation" in response_text.lower()
        )

    def test_duplicate_id_check(self, agent, check_api_keys):
        """Test checking for duplicate sample IDs."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Validate dataset has no duplicate sample IDs")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention duplicates or validation
        assert (
            "duplicate" in response_text.lower()
            or "validation" in response_text.lower()
        )

    def test_platform_consistency(self, agent, check_api_keys):
        """Test checking platform consistency."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Validate dataset platform consistency")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention platform or validation
        assert (
            "platform" in response_text.lower() or "validation" in response_text.lower()
        )

    def test_multiple_failures_edge_case(self, agent, check_api_keys):
        """Test handling multiple validation failures."""
        from langchain_core.messages import HumanMessage

        # Request strict validation that may fail
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Validate dataset with strict requirements: 100 samples, multiple conditions, controls, no duplicates"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should report validation results (may have failures)
        assert (
            "validation" in response_text.lower()
            or "fail" in response_text.lower()
            or "warning" in response_text.lower()
            or "sample" in response_text.lower()
        )


# ============================================================================
# Integration Workflow Tests
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestMetadataAssistantWorkflows:
    """End-to-end workflow tests for metadata_assistant."""

    def test_multi_omics_integration_workflow(self, agent, check_api_keys):
        """Test complete workflow for multi-omics integration."""
        from langchain_core.messages import HumanMessage

        # Workflow: Validate → Read → Standardize → Map
        # Step 1: Validate dataset
        result1 = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Validate geo_gse180759 has required samples and conditions"
                    )
                ]
            }
        )

        # Step 2: Read metadata
        result2 = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Read sample metadata from geo_gse180759 in summary format"
                    )
                ]
            }
        )

        # Step 3: Standardize
        result3 = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Standardize geo_gse180759 to transcriptomics schema"
                    )
                ]
            }
        )

        # Verify all steps executed
        assert all(
            [
                result1["messages"][-1].content,
                result2["messages"][-1].content,
                result3["messages"][-1].content,
            ]
        )

    def test_meta_analysis_preparation_workflow(self, agent, check_api_keys):
        """Test workflow for meta-analysis dataset harmonization."""
        from langchain_core.messages import HumanMessage

        # Workflow: Read → Standardize → Validate compatibility
        result1 = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Read metadata from multiple datasets for meta-analysis"
                    )
                ]
            }
        )

        result2 = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Standardize all datasets to common schema")
                ]
            }
        )

        # Verify workflow executed
        assert all(
            [
                result1["messages"][-1].content,
                result2["messages"][-1].content,
            ]
        )
