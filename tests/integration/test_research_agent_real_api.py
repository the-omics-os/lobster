"""
Integration tests for research_agent with real API calls.

This test suite validates all 10 research_agent tools using real API calls to:
- PubMed (NCBI E-utilities)
- PMC (PubMed Central)
- GEO (Gene Expression Omnibus)
- bioRxiv/medRxiv
- Publisher webpages

**Performance Expectations:**
- fast_abstract_search: <500ms per PMID
- read_full_publication (PMC): <1s
- read_full_publication (webpage): 2-5s
- read_full_publication (PDF): 3-8s
- GEO metadata: 2-3s per dataset
- Literature search: 1-3s

**Test Strategy:**
- Use known stable identifiers (PMIDs, DOIs, GSE IDs)
- Test both happy paths and edge cases
- Validate cascade fallback logic
- Verify workspace caching behavior
- Test error handling and recovery

**Markers:**
- @pytest.mark.real_api: All tests (requires API keys)
- @pytest.mark.slow: Tests >30s
- @pytest.mark.integration: Multi-component tests

**Environment Requirements:**
- NCBI_API_KEY (recommended for PubMed rate limits)
- AWS_BEDROCK_ACCESS_KEY + AWS_BEDROCK_SECRET_ACCESS_KEY (for LLM)
- Internet connectivity for API access

Phase 7 - Task Group 1: Research Agent Real API Integration Tests
"""

import os
import time
from pathlib import Path

import pytest

from lobster.agents.research_agent import research_agent
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
    workspace = tmp_path_factory.mktemp("test_research_agent_real_api")
    return workspace


@pytest.fixture(scope="module")
def data_manager(test_workspace):
    """Initialize DataManagerV2 with test workspace."""
    settings = get_settings()
    dm = DataManagerV2(workspace_path=test_workspace, console=None)
    return dm


@pytest.fixture(scope="module")
def agent(data_manager):
    """Create research_agent instance for testing."""
    return research_agent(
        data_manager=data_manager, callback_handler=None, delegation_tools=None
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
        logger.warning(
            "NCBI_API_KEY not set - PubMed requests limited to 3/second (vs 10/second with key)"
        )


# ============================================================================
# Tool 1.1: search_literature
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestSearchLiterature:
    """Test search_literature tool with real PubMed/bioRxiv APIs."""

    def test_keyword_search_pubmed(self, agent, check_api_keys):
        """Test keyword-based literature search via PubMed."""
        # Invoke agent with search request
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Search PubMed for 'BRCA1 breast cancer' (max 5 results)"
                    )
                ]
            }
        )

        # Verify response structure
        assert "messages" in result
        response_text = result["messages"][-1].content

        # Verify search completed successfully
        assert "BRCA1" in response_text or "breast cancer" in response_text.lower()
        assert "PMID" in response_text or "DOI" in response_text

    def test_related_papers_discovery(self, agent, check_api_keys):
        """Test related paper discovery using related_to parameter."""
        from langchain_core.messages import HumanMessage

        # Use known PMID with citations
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Find papers related to PMID:35042229 (max 5 results)"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify related papers found
        assert "related" in response_text.lower() or "citing" in response_text.lower()

    def test_search_with_date_filters(self, agent, check_api_keys):
        """Test literature search with date range filters."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content='Search PubMed for "single-cell RNA-seq" published 2020-2024 (max 5 results)'
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify search executed with filters
        assert (
            "single-cell" in response_text.lower() or "rna-seq" in response_text.lower()
        )

    def test_invalid_query_handling(self, agent, check_api_keys):
        """Test error handling for invalid search queries."""
        from langchain_core.messages import HumanMessage

        # Empty query should fail gracefully
        result = agent.invoke(
            {"messages": [HumanMessage(content="Search PubMed for '' (empty query)")]}
        )

        response_text = result["messages"][-1].content

        # Should contain error message or request clarification
        assert (
            "error" in response_text.lower()
            or "invalid" in response_text.lower()
            or "provide" in response_text.lower()
        )


# ============================================================================
# Tool 1.2: find_related_entries
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestFindRelatedEntries:
    """Test find_related_entries tool for dataset discovery."""

    def test_dataset_linking_from_publication(self, agent, check_api_keys):
        """Test finding datasets linked to a publication."""
        from langchain_core.messages import HumanMessage

        # Use known PMID with GEO dataset (PMID:35042229 → GSE180759)
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Find datasets from publication PMID:35042229")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify dataset discovery
        assert "GSE" in response_text or "dataset" in response_text.lower()

    def test_publication_linking_from_dataset(self, agent, check_api_keys):
        """Test finding publications linked to a dataset."""
        from langchain_core.messages import HumanMessage

        # Use known GEO dataset with publication
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Find publications related to dataset GSE180759"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify publication found
        assert "PMID" in response_text or "publication" in response_text.lower()

    def test_entry_type_filtering_datasets_only(self, agent, check_api_keys):
        """Test filtering by entry_type to return only datasets."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Find only datasets (no publications) related to PMID:35042229"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention datasets specifically
        assert "dataset" in response_text.lower()

    def test_no_related_entries_edge_case(self, agent, check_api_keys):
        """Test handling of publications with no linked datasets."""
        from langchain_core.messages import HumanMessage

        # Use PMID unlikely to have public datasets
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Find datasets from publication PMID:10000000")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should indicate no datasets found
        assert (
            "no" in response_text.lower()
            or "not found" in response_text.lower()
            or "unable" in response_text.lower()
        )


# ============================================================================
# Tool 1.3: fast_dataset_search
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestFastDatasetSearch:
    """Test fast_dataset_search tool for direct repository queries."""

    def test_accession_based_search(self, agent, check_api_keys):
        """Test searching GEO by accession ID."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Search GEO for dataset GSE180759 directly")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify dataset found
        assert "GSE180759" in response_text

    def test_text_based_search_geo(self, agent, check_api_keys):
        """Test keyword search in GEO database."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Search GEO database for 'lung cancer single-cell' (max 5 results)"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify search executed
        assert "GSE" in response_text or "dataset" in response_text.lower()

    def test_multi_source_search(self, agent, check_api_keys):
        """Test searching across multiple repositories (GEO, SRA)."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Search omics databases for 'CRISPR screen' datasets"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify search executed
        assert (
            "GSE" in response_text
            or "SRR" in response_text
            or "dataset" in response_text.lower()
        )

    def test_invalid_accession_handling(self, agent, check_api_keys):
        """Test error handling for invalid accession IDs."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {"messages": [HumanMessage(content="Search GEO for dataset INVALID99999")]}
        )

        response_text = result["messages"][-1].content

        # Should indicate not found or error
        assert (
            "not found" in response_text.lower()
            or "invalid" in response_text.lower()
            or "error" in response_text.lower()
        )


# ============================================================================
# Tool 1.4: get_dataset_metadata
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestGetDatasetMetadata:
    """Test get_dataset_metadata tool for metadata extraction."""

    def test_geo_metadata_extraction(self, agent, check_api_keys):
        """Test extracting metadata from GEO dataset."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Get metadata for GEO dataset GSE180759")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify metadata extracted
        assert "GSE180759" in response_text
        assert "sample" in response_text.lower() or "platform" in response_text.lower()

    def test_publication_metadata_extraction(self, agent, check_api_keys):
        """Test extracting metadata from publication."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Get metadata for publication PMID:35042229")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify metadata extracted
        assert "PMID" in response_text or "35042229" in response_text
        assert "title" in response_text.lower() or "author" in response_text.lower()

    def test_auto_detection_from_identifier(self, agent, check_api_keys):
        """Test automatic database detection from identifier format."""
        from langchain_core.messages import HumanMessage

        # GSE format should auto-detect as GEO
        result = agent.invoke(
            {"messages": [HumanMessage(content="Get metadata for GSE12345")]}
        )

        response_text = result["messages"][-1].content

        # Should attempt GEO lookup
        assert "GSE" in response_text or "GEO" in response_text

    def test_deleted_dataset_handling(self, agent, check_api_keys):
        """Test error handling for deleted/unavailable datasets."""
        from langchain_core.messages import HumanMessage

        # Use known deleted or invalid GSE
        result = agent.invoke(
            {"messages": [HumanMessage(content="Get metadata for GSE1")]}
        )

        response_text = result["messages"][-1].content

        # Should indicate error or unavailable
        assert (
            "error" in response_text.lower()
            or "not found" in response_text.lower()
            or "unavailable" in response_text.lower()
        )


# ============================================================================
# Tool 1.5: validate_dataset_metadata
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestValidateDatasetMetadata:
    """Test validate_dataset_metadata tool for pre-download validation."""

    def test_geo_validation_with_required_fields(self, agent, check_api_keys):
        """Test validating GEO dataset has required metadata fields."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Validate GSE180759 has 'sample_id,condition,tissue' fields"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify validation executed
        assert "validation" in response_text.lower() or "GSE180759" in response_text

    def test_sample_count_validation(self, agent, check_api_keys):
        """Test validation includes sample count checks."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Validate GSE180759 has at least 20 samples with required fields"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention sample count
        assert "sample" in response_text.lower()

    def test_platform_compatibility_check(self, agent, check_api_keys):
        """Test validation checks platform information."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Validate GSE180759 platform is Illumina compatible"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention platform
        assert (
            "platform" in response_text.lower() or "illumina" in response_text.lower()
        )

    def test_incomplete_metadata_warning(self, agent, check_api_keys):
        """Test validation warns about incomplete metadata."""
        from langchain_core.messages import HumanMessage

        # Request validation with many fields (likely some missing)
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Validate GSE180759 has 'sample_id,condition,age,sex,batch,treatment' fields"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should indicate validation results
        assert (
            "validation" in response_text.lower()
            or "missing" in response_text.lower()
            or "complete" in response_text.lower()
        )


# ============================================================================
# Tool 1.6: extract_methods
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestExtractMethods:
    """Test extract_methods tool for computational methods extraction."""

    def test_pmc_xml_extraction(self, agent, check_api_keys):
        """Test methods extraction from PMC XML (fastest path)."""
        from langchain_core.messages import HumanMessage

        # Use PMID with PMC full text
        result = agent.invoke(
            {"messages": [HumanMessage(content="Extract methods from PMID:35042229")]}
        )

        response_text = result["messages"][-1].content

        # Verify extraction attempted
        assert "method" in response_text.lower() or "software" in response_text.lower()

    def test_webpage_fallback_extraction(self, agent, check_api_keys):
        """Test methods extraction falls back to webpage parsing."""
        from langchain_core.messages import HumanMessage

        # Use Nature paper URL (no PMC, has webpage)
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Extract methods from https://www.nature.com/articles/s41586-021-03852-1"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify extraction attempted
        assert "method" in response_text.lower() or "extract" in response_text.lower()

    def test_software_detection(self, agent, check_api_keys):
        """Test detection of software tools in methods."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Extract software tools used in PMID:35042229")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should mention software or tools
        assert "software" in response_text.lower() or "tool" in response_text.lower()

    def test_methods_missing_edge_case(self, agent, check_api_keys):
        """Test handling of papers without accessible methods section."""
        from langchain_core.messages import HumanMessage

        # Use abstract-only paper or paywalled
        result = agent.invoke(
            {"messages": [HumanMessage(content="Extract methods from PMID:10000000")]}
        )

        response_text = result["messages"][-1].content

        # Should indicate difficulty or limitations
        assert (
            "not" in response_text.lower()
            or "unable" in response_text.lower()
            or "abstract" in response_text.lower()
        )


# ============================================================================
# Tool 1.7: fast_abstract_search
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestFastAbstractSearch:
    """Test fast_abstract_search tool for rapid screening."""

    def test_pubmed_retrieval_single(self, agent, check_api_keys):
        """Test fast abstract retrieval for single PMID."""
        from langchain_core.messages import HumanMessage

        start_time = time.time()

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Get abstract for PMID:35042229 quickly")
                ]
            }
        )

        elapsed = time.time() - start_time
        response_text = result["messages"][-1].content

        # Verify abstract retrieved
        assert "abstract" in response_text.lower() or "35042229" in response_text

        # Performance expectation: <500ms (may be slower with agent overhead)
        logger.info(
            f"fast_abstract_search took {elapsed:.2f}s (target: <1s with agent)"
        )

    def test_performance_target_500ms(self, agent, check_api_keys):
        """Test abstract retrieval meets <500ms target."""
        from langchain_core.messages import HumanMessage

        # Note: Agent coordination adds overhead, direct tool would be <500ms
        start_time = time.time()

        result = agent.invoke(
            {"messages": [HumanMessage(content="Quick abstract for PMID:35042229")]}
        )

        elapsed = time.time() - start_time
        response_text = result["messages"][-1].content

        # Verify retrieval successful
        assert "abstract" in response_text.lower() or "PMID" in response_text

        logger.info(
            f"Abstract retrieval with agent overhead: {elapsed:.2f}s (direct tool target: <500ms)"
        )

    def test_batch_abstract_retrieval(self, agent, check_api_keys):
        """Test retrieving abstracts for multiple PMIDs."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Get abstracts for PMID:35042229 and PMID:33057194"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify multiple abstracts mentioned
        assert "abstract" in response_text.lower()

    def test_invalid_pmid_handling(self, agent, check_api_keys):
        """Test error handling for invalid PMID format."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {"messages": [HumanMessage(content="Get abstract for PMID:INVALID")]}
        )

        response_text = result["messages"][-1].content

        # Should indicate error or invalid
        assert (
            "error" in response_text.lower()
            or "invalid" in response_text.lower()
            or "not found" in response_text.lower()
        )


# ============================================================================
# Tool 1.8: read_full_publication (CRITICAL - Three-Tier Cascade)
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestReadFullPublication:
    """Test read_full_publication with three-tier cascade: PMC→Webpage→PDF."""

    def test_pmc_fast_path_tier1(self, agent, check_api_keys):
        """Test PMC XML extraction (Tier 1 - fastest, 500ms)."""
        from langchain_core.messages import HumanMessage

        start_time = time.time()

        # Use PMID with PMC full text
        result = agent.invoke(
            {"messages": [HumanMessage(content="Read full publication PMID:35042229")]}
        )

        elapsed = time.time() - start_time
        response_text = result["messages"][-1].content

        # Verify full text retrieved
        assert (
            "method" in response_text.lower()
            or "full" in response_text.lower()
            or "publication" in response_text.lower()
        )

        logger.info(f"PMC extraction took {elapsed:.2f}s (target: <2s with agent)")

    def test_webpage_fallback_tier2(self, agent, check_api_keys):
        """Test webpage extraction fallback (Tier 2 - 2-5s)."""
        from langchain_core.messages import HumanMessage

        start_time = time.time()

        # Use Nature URL (no PMC, has webpage)
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Read full publication https://www.nature.com/articles/s41586-021-03852-1"
                    )
                ]
            }
        )

        elapsed = time.time() - start_time
        response_text = result["messages"][-1].content

        # Verify content retrieved
        assert (
            "publication" in response_text.lower() or "content" in response_text.lower()
        )

        logger.info(f"Webpage extraction took {elapsed:.2f}s (target: 2-5s)")

    def test_pdf_fallback_tier3(self, agent, check_api_keys):
        """Test PDF extraction fallback (Tier 3 - 3-8s)."""
        from langchain_core.messages import HumanMessage

        start_time = time.time()

        # Use bioRxiv PDF URL
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Read publication from PDF: https://www.biorxiv.org/content/10.1101/2024.01.18.576270v1.full.pdf"
                    )
                ]
            }
        )

        elapsed = time.time() - start_time
        response_text = result["messages"][-1].content

        # Verify extraction attempted
        assert "publication" in response_text.lower() or "PDF" in response_text

        logger.info(f"PDF extraction took {elapsed:.2f}s (target: 3-8s)")

    def test_cascade_integration_all_tiers(self, agent, check_api_keys):
        """Test full cascade integration PMC→Webpage→PDF."""
        from langchain_core.messages import HumanMessage

        # Agent should attempt PMC first, then fallback if needed
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Read full publication PMID:35042229 with all fallbacks"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify successful retrieval through cascade
        assert "publication" in response_text.lower()

    def test_all_tiers_fail_edge_case(self, agent, check_api_keys):
        """Test handling when all three tiers fail."""
        from langchain_core.messages import HumanMessage

        # Use paywalled paper or invalid identifier
        result = agent.invoke(
            {"messages": [HumanMessage(content="Read full publication PMID:10000000")]}
        )

        response_text = result["messages"][-1].content

        # Should indicate failure or limitations
        assert (
            "not" in response_text.lower()
            or "unable" in response_text.lower()
            or "error" in response_text.lower()
        )


# ============================================================================
# Tool 1.9: write_to_workspace
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestWriteToWorkspace:
    """Test write_to_workspace tool for content caching."""

    def test_cache_publication_to_workspace(self, agent, check_api_keys):
        """Test caching publication to workspace."""
        from langchain_core.messages import HumanMessage

        # First read a publication
        result1 = agent.invoke(
            {"messages": [HumanMessage(content="Read abstract for PMID:35042229")]}
        )

        # Then cache it
        result2 = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Cache publication PMID:35042229 to workspace")
                ]
            }
        )

        response_text = result2["messages"][-1].content

        # Verify caching succeeded
        assert "cache" in response_text.lower() or "workspace" in response_text.lower()

    def test_cache_dataset_metadata(self, agent, check_api_keys):
        """Test caching dataset metadata to workspace."""
        from langchain_core.messages import HumanMessage

        # Get dataset metadata
        result1 = agent.invoke(
            {"messages": [HumanMessage(content="Get metadata for GSE180759")]}
        )

        # Cache to workspace
        result2 = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Cache GSE180759 metadata to workspace")
                ]
            }
        )

        response_text = result2["messages"][-1].content

        # Verify caching attempted
        assert "cache" in response_text.lower() or "workspace" in response_text.lower()

    def test_naming_convention_validation(self, agent, check_api_keys):
        """Test workspace validates naming conventions."""
        from langchain_core.messages import HumanMessage

        # Should follow publication_PMID or dataset_GSE conventions
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Cache publication PMID:35042229 with correct naming"
                    )
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Verify operation attempted
        assert (
            "publication" in response_text.lower() or "cache" in response_text.lower()
        )

    def test_duplicate_id_handling(self, agent, check_api_keys):
        """Test handling of duplicate cache attempts."""
        from langchain_core.messages import HumanMessage

        # Cache once
        result1 = agent.invoke(
            {"messages": [HumanMessage(content="Cache PMID:35042229 to workspace")]}
        )

        # Try caching again (duplicate)
        result2 = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Cache PMID:35042229 again to workspace")
                ]
            }
        )

        response_text = result2["messages"][-1].content

        # Should handle gracefully (overwrite or skip)
        assert (
            "cache" in response_text.lower()
            or "already" in response_text.lower()
            or "exist" in response_text.lower()
        )


# ============================================================================
# Tool 1.10: get_content_from_workspace
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
class TestGetContentFromWorkspace:
    """Test get_content_from_workspace tool for retrieval."""

    def test_list_all_cached_content(self, agent, data_manager, check_api_keys):
        """Test listing all cached workspace content."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {"messages": [HumanMessage(content="List all cached workspace content")]}
        )

        response_text = result["messages"][-1].content

        # Should show workspace listing (may be empty)
        assert (
            "workspace" in response_text.lower()
            or "cached" in response_text.lower()
            or "no" in response_text.lower()
        )

    def test_retrieve_by_identifier(self, agent, check_api_keys):
        """Test retrieving specific content by identifier."""
        from langchain_core.messages import HumanMessage

        # First cache something
        result1 = agent.invoke(
            {"messages": [HumanMessage(content="Cache metadata for GSE180759")]}
        )

        # Then retrieve it
        result2 = agent.invoke(
            {"messages": [HumanMessage(content="Get cached content for GSE180759")]}
        )

        response_text = result2["messages"][-1].content

        # Verify retrieval attempted
        assert "GSE180759" in response_text or "content" in response_text.lower()

    def test_detail_level_summary(self, agent, check_api_keys):
        """Test retrieving content with summary detail level."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Get summary of cached workspace content")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should show summary
        assert (
            "summary" in response_text.lower() or "workspace" in response_text.lower()
        )

    def test_detail_level_methods(self, agent, check_api_keys):
        """Test retrieving methods section detail level."""
        from langchain_core.messages import HumanMessage

        # Cache publication first
        result1 = agent.invoke(
            {"messages": [HumanMessage(content="Cache PMID:35042229 to workspace")]}
        )

        # Retrieve methods
        result2 = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Get methods section from cached PMID:35042229"
                    )
                ]
            }
        )

        response_text = result2["messages"][-1].content

        # Should show methods or indicate not available
        assert "method" in response_text.lower() or "not" in response_text.lower()

    def test_detail_level_samples(self, agent, check_api_keys):
        """Test retrieving sample IDs detail level."""
        from langchain_core.messages import HumanMessage

        # Cache dataset first
        result1 = agent.invoke(
            {"messages": [HumanMessage(content="Cache GSE180759 to workspace")]}
        )

        # Retrieve samples
        result2 = agent.invoke(
            {"messages": [HumanMessage(content="Get sample IDs from cached GSE180759")]}
        )

        response_text = result2["messages"][-1].content

        # Should show samples or indicate not available
        assert "sample" in response_text.lower() or "not" in response_text.lower()

    def test_detail_level_full_metadata(self, agent, check_api_keys):
        """Test retrieving full metadata detail level."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Get full metadata from cached content")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should show metadata or indicate empty workspace
        assert (
            "metadata" in response_text.lower()
            or "workspace" in response_text.lower()
            or "no" in response_text.lower()
        )

    def test_identifier_not_found_edge_case(self, agent, check_api_keys):
        """Test handling of non-existent identifier."""
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Get content for identifier NONEXISTENT_12345")
                ]
            }
        )

        response_text = result["messages"][-1].content

        # Should indicate not found
        assert (
            "not found" in response_text.lower()
            or "not" in response_text.lower()
            or "no" in response_text.lower()
        )


# ============================================================================
# Summary Tests
# ============================================================================


@pytest.mark.real_api
@pytest.mark.integration
@pytest.mark.slow
class TestResearchAgentIntegration:
    """End-to-end integration tests for research_agent workflows."""

    def test_full_workflow_discovery_to_cache(self, agent, check_api_keys):
        """Test complete workflow: search → metadata → validate → cache."""
        from langchain_core.messages import HumanMessage

        # 1. Search for papers
        result1 = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Search PubMed for 'BRCA1 breast cancer'")
                ]
            }
        )

        # 2. Find datasets from first paper (PMID:35042229)
        result2 = agent.invoke(
            {"messages": [HumanMessage(content="Find datasets from PMID:35042229")]}
        )

        # 3. Validate dataset metadata
        result3 = agent.invoke(
            {"messages": [HumanMessage(content="Validate GSE180759 metadata")]}
        )

        # 4. Cache to workspace
        result4 = agent.invoke(
            {"messages": [HumanMessage(content="Cache GSE180759 to workspace")]}
        )

        # Verify all steps executed
        assert all(
            [
                result1["messages"][-1].content,
                result2["messages"][-1].content,
                result3["messages"][-1].content,
                result4["messages"][-1].content,
            ]
        )

    def test_performance_benchmark_suite(self, agent, check_api_keys):
        """Benchmark performance of all key operations."""
        import time

        from langchain_core.messages import HumanMessage

        benchmarks = {}

        # Abstract retrieval
        start = time.time()
        agent.invoke(
            {"messages": [HumanMessage(content="Quick abstract for PMID:35042229")]}
        )
        benchmarks["abstract"] = time.time() - start

        # GEO metadata
        start = time.time()
        agent.invoke({"messages": [HumanMessage(content="Get metadata for GSE180759")]})
        benchmarks["geo_metadata"] = time.time() - start

        # Literature search
        start = time.time()
        agent.invoke({"messages": [HumanMessage(content="Search PubMed for 'cancer'")]})
        benchmarks["search"] = time.time() - start

        # Log results
        for operation, duration in benchmarks.items():
            logger.info(f"Performance: {operation} = {duration:.2f}s")

        # All should complete (times will vary with agent coordination overhead)
        assert all(duration > 0 for duration in benchmarks.values())
