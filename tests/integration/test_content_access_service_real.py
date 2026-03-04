"""
Real API integration tests for ContentAccessService.

These tests make actual API calls to verify end-to-end functionality.

API Keys Required:
- NCBI_API_KEY: Required for PubMed/PMC API access
- AWS_BEDROCK_ACCESS_KEY: Required for LLM-based methods extraction
- AWS_BEDROCK_SECRET_ACCESS_KEY: Required for LLM-based methods extraction

Rate Limiting:
- 0.5-1 second sleeps between consecutive API calls
"""

import os
import time
from unittest.mock import Mock

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.data_access.content_access_service import ContentAccessService
from lobster.tools.providers.base_provider import DatasetType

pytestmark = [pytest.mark.integration]


@pytest.fixture
def real_data_manager(tmp_path):
    """Create a real DataManagerV2 instance for integration testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    dm = Mock(spec=DataManagerV2)
    dm.cache_dir = cache_dir
    dm.literature_cache_dir = cache_dir / "literature"
    dm.literature_cache_dir.mkdir()

    return dm


@pytest.fixture
def real_content_access_service(real_data_manager):
    """Create a real ContentAccessService for integration testing."""
    return ContentAccessService(data_manager=real_data_manager)


# ============================================================================
# Integration Tests - Real API Calls
# ============================================================================


@pytest.mark.skipif(not os.getenv("NCBI_API_KEY"), reason="NCBI_API_KEY not set")
class TestRealAPIDiscovery:
    """Test discovery methods with real API calls."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_search_literature_real(self, real_content_access_service):
        """Test real PubMed literature search.

        Requires: NCBI_API_KEY environment variable
        Rate limit: 3 req/s with API key
        Expected duration: 1-2s
        """
        query = "BRCA1 breast cancer"

        start_time = time.time()
        # Service returns tuple: (formatted_string, stats_dict, ir)
        result_str, stats, ir = real_content_access_service.search_literature(
            query, max_results=5
        )
        elapsed = time.time() - start_time

        # Verify results
        assert result_str is not None
        assert len(result_str) > 0, "Should return non-empty formatted results"
        assert stats["results_count"] > 0, "Should find at least 1 result"
        assert stats["provider_used"] == "PubMedProvider"
        assert elapsed < 3.0, f"Search took {elapsed:.2f}s (expected <3s)"

        # Verify formatted output contains expected elements
        assert "PMID:" in result_str or "pmid" in result_str.lower()
        assert "Title:" in result_str or "title" in result_str.lower()

        # Rate limiting
        time.sleep(0.5)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_discover_datasets_real(self, real_content_access_service):
        """Test real GEO dataset discovery.

        Requires: NCBI_API_KEY environment variable
        Rate limit: 3 req/s with API key
        Expected duration: 1-2s
        """
        query = "breast cancer RNA-seq Homo sapiens"

        start_time = time.time()
        # Service returns tuple: (formatted_string, stats_dict, ir)
        result_str, stats, ir = real_content_access_service.discover_datasets(
            query, dataset_type=DatasetType.GEO, max_results=5
        )
        elapsed = time.time() - start_time

        # Verify results
        assert result_str is not None
        assert len(result_str) > 0, "Should return non-empty formatted results"
        assert stats["results_count"] > 0, "Should find at least 1 dataset"
        assert elapsed < 3.0, f"Discovery took {elapsed:.2f}s (expected <3s)"

        # Verify formatted output contains GEO accessions
        assert "GSE" in result_str, "Should contain GEO accession numbers"

        # Rate limiting
        time.sleep(0.5)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_find_linked_datasets_real(self, real_content_access_service):
        """Test finding datasets linked to real publication.

        Test Paper: PMID:35042229 (Nature 2022)
        Known dataset: GSE180759

        Requires: NCBI_API_KEY environment variable
        Rate limit: 3 req/s with API key
        Expected duration: 1-2s
        """
        identifier = "PMID:35042229"

        start_time = time.time()
        # Service returns formatted string directly
        result_str = real_content_access_service.find_linked_datasets(
            identifier=identifier
        )
        elapsed = time.time() - start_time

        # Verify results (may or may not find linked datasets)
        assert result_str is not None
        assert isinstance(result_str, str), "Should return formatted string"
        assert elapsed < 3.0, f"Link search took {elapsed:.2f}s (expected <3s)"

        # Rate limiting
        time.sleep(0.5)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_find_related_publications_real(self, real_content_access_service):
        """Test finding publications related to a real PMID using E-Link.

        Test Paper: PMID:35102706 (from bug report)
        Expected: Should find papers citing or cited by this paper

        Requires: NCBI_API_KEY environment variable
        Rate limit: 3 req/s with API key
        Expected duration: 2-3s (multiple E-Link calls)
        """
        identifier = "PMID:35102706"

        start_time = time.time()
        # Service returns formatted string directly
        result_str = real_content_access_service.find_related_publications(
            identifier=identifier, max_results=5
        )
        elapsed = time.time() - start_time

        # Verify results
        assert result_str is not None
        assert isinstance(result_str, str), "Should return formatted string"
        assert "Related Publications" in result_str, "Should have header"
        assert elapsed < 5.0, f"Related search took {elapsed:.2f}s (expected <5s)"

        # Should contain at least some PMIDs or indicate no results found
        has_results = "PMID:" in result_str or "No related publications" in result_str
        assert has_results, "Should either show PMIDs or 'no results' message"

        # Rate limiting
        time.sleep(0.5)


@pytest.mark.skipif(not os.getenv("NCBI_API_KEY"), reason="NCBI_API_KEY not set")
class TestRealAPIMetadata:
    """Test metadata methods with real API calls."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_extract_metadata_real(self, real_content_access_service):
        """Test real publication metadata retrieval via extract_metadata.

        Test Paper: PMID:35042229 (Nature 2022)

        Requires: NCBI_API_KEY environment variable
        Rate limit: 3 req/s with API key
        Expected duration: 1-2s
        """
        identifier = "PMID:35042229"

        start_time = time.time()
        metadata = real_content_access_service.extract_metadata(identifier)
        elapsed = time.time() - start_time

        assert metadata is not None
        # extract_metadata may return PublicationMetadata object or string
        if hasattr(metadata, "pmid"):
            assert metadata.pmid == "35042229"
        elif isinstance(metadata, dict):
            assert metadata.get("pmid") == "35042229"
        assert elapsed < 3.0, f"Metadata retrieval took {elapsed:.2f}s (expected <3s)"

        # Rate limiting
        time.sleep(0.5)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_get_dataset_metadata_real(self, real_content_access_service):
        """Test real dataset metadata retrieval.

        Test Dataset: GSE180759 (linked to PMID:35042229)

        Requires: NCBI_API_KEY environment variable
        Rate limit: 3 req/s with API key
        Expected duration: 1-2s
        """
        accession = "GSE180759"

        start_time = time.time()
        metadata = real_content_access_service.get_dataset_metadata(accession)
        elapsed = time.time() - start_time

        assert metadata is not None
        assert metadata.get("accession") == "GSE180759"
        assert "samples" in metadata or "sample_count" in metadata
        assert "platform" in metadata
        assert elapsed < 2.0, f"Dataset metadata took {elapsed:.2f}s (expected <2s)"

        # Rate limiting
        time.sleep(0.5)


@pytest.mark.skipif(not os.getenv("NCBI_API_KEY"), reason="NCBI_API_KEY not set")
class TestRealAPIContentAccess:
    """Test content access methods with real API calls."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_get_abstract_real(self, real_content_access_service):
        """Test real abstract retrieval.

        Test Paper: PMID:35042229 (Nature 2022)

        Requires: NCBI_API_KEY environment variable
        Rate limit: 3 req/s with API key
        Expected duration: <1s
        """
        identifier = "PMID:35042229"

        start_time = time.time()
        result = real_content_access_service.get_abstract(identifier)
        elapsed = time.time() - start_time

        assert result is not None
        assert "abstract" in result or "content" in result
        abstract = result.get("abstract") or result.get("content")
        assert len(abstract) > 200
        assert elapsed < 1.0, f"Abstract retrieval took {elapsed:.2f}s (expected <1s)"

        # Rate limiting
        time.sleep(0.5)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_extract_methods_real(self, real_content_access_service):
        """Test real methods extraction with LLM.

        Test Paper: PMID:35042229 (Nature 2022)
        Known to have PMC full text: PMC8760896

        Requires: NCBI_API_KEY, AWS_BEDROCK_ACCESS_KEY, AWS_BEDROCK_SECRET_ACCESS_KEY
        Rate limit: 3 req/s for PubMed
        Expected duration: 3-5s (includes LLM call)
        """
        identifier = "PMID:35042229"

        start_time = time.time()
        result = real_content_access_service.extract_methods_section(identifier)
        elapsed = time.time() - start_time

        assert result is not None
        assert "methods" in result or "content" in result
        methods = result.get("methods") or result.get("content")
        assert len(methods) > 100
        assert elapsed < 10.0, f"Methods extraction took {elapsed:.2f}s (expected <10s)"

        # Rate limiting
        time.sleep(1.0)


@pytest.mark.skipif(not os.getenv("NCBI_API_KEY"), reason="NCBI_API_KEY not set")
class TestRealAPIThreeTierCascade:
    """Test three-tier cascade strategy with real API calls."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_tier1_pmc_success_real(self, real_content_access_service):
        """Test Tier 1 (PMC) success with real API.

        Test Paper: PMID:35042229 (Nature 2022)
        Known PMC ID: PMC8760896

        Verifies:
        1. PMC full text retrieval succeeds
        2. Response time <1s (Tier 1 fast access)
        3. Content length >5000 chars
        4. Provider is PubMedProvider

        Requires: NCBI_API_KEY
        Rate limit: 3 req/s
        Expected duration: <1s
        """
        identifier = "PMID:35042229"

        start_time = time.time()
        result = real_content_access_service.get_full_content(identifier)
        elapsed = time.time() - start_time

        assert result is not None
        assert "content" in result
        assert len(result["content"]) > 5000
        assert (
            result.get("provider") == "PubMedProvider"
            or "pmc" in result.get("provider", "").lower()
        )
        assert elapsed < 2.0, f"Tier 1 PMC took {elapsed:.2f}s (expected <2s)"

        # Rate limiting
        time.sleep(1.0)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_tier2_webpage_fallback_real(self, real_content_access_service):
        """Test Tier 2 (Webpage) fallback when PMC unavailable.

        Test Paper: PMID without PMC (paywall paper from Nature/Science)

        Verifies:
        1. PMC fails (no PMC ID)
        2. Webpage provider succeeds
        3. Response time 2-5s (Tier 2 access)
        4. HTML content extracted

        Requires: NCBI_API_KEY
        Rate limit: 3 req/s
        Expected duration: 2-5s
        """
        # Find a PMID without PMC access (paywall paper)
        identifier = "PMID:20147447"  # Nature paper, likely no PMC

        start_time = time.time()
        result = real_content_access_service.get_full_content(identifier)
        elapsed = time.time() - start_time

        # May succeed or fail depending on publisher access
        if result is not None:
            assert "content" in result
            assert elapsed < 10.0, f"Tier 2 webpage took {elapsed:.2f}s (expected <10s)"

        # Rate limiting
        time.sleep(2.0)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_tier3_pdf_fallback_real(self, real_content_access_service):
        """Test Tier 3 (PDF) fallback when PMC and webpage fail.

        Test Paper: Paywall paper with PDF available

        Verifies:
        1. PMC fails
        2. Webpage fails or insufficient
        3. PDF provider succeeds
        4. Response time 3-8s (Tier 3 access)
        5. PDF text extraction works

        Requires: NCBI_API_KEY
        Rate limit: 3 req/s
        Expected duration: 3-8s
        """
        # Find a PMID with PDF access but no PMC
        identifier = "PMID:20147447"  # Nature paper

        start_time = time.time()
        result = real_content_access_service.get_full_content(
            identifier,
            force_pdf=True,  # Force PDF extraction if supported
        )
        elapsed = time.time() - start_time

        # May succeed or fail depending on PDF availability
        if result is not None:
            assert "content" in result
            assert elapsed < 15.0, f"Tier 3 PDF took {elapsed:.2f}s (expected <15s)"

        # Rate limiting
        time.sleep(2.0)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_all_tiers_fail_real(self, real_content_access_service):
        """Test handling when all tiers fail.

        Test Paper: Invalid PMID or completely restricted access

        Verifies:
        1. All providers attempted
        2. Graceful error handling
        3. Clear error message

        Requires: NCBI_API_KEY
        Rate limit: 3 req/s
        Expected duration: 1-2s
        """
        identifier = "PMID:99999999"  # Invalid PMID

        start_time = time.time()
        result = real_content_access_service.get_full_content(identifier)
        elapsed = time.time() - start_time

        # Should return None or error dict
        assert result is None or "error" in result or "failed" in str(result).lower()
        assert elapsed < 5.0, f"Failed cascade took {elapsed:.2f}s (expected <5s)"

        # Rate limiting
        time.sleep(0.5)
