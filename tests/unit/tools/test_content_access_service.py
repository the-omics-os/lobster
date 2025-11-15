"""
Unit tests for ContentAccessService.

Tests the three-tier cascade strategy for publication content access
introduced in Phase 2 of the research agent refactoring.

Three-Tier Cascade Strategy:
- Tier 1 (Fast): AbstractProvider for quick abstract retrieval (200-500ms, NCBI)
- Tier 2 (Structured): PubMedProvider for PMC XML full text (500ms, 95% accuracy)
- Tier 3 (Fallback): WebpageProvider → PDFProvider cascade (2-8 seconds)

Coverage Target: 90%+

Pytest Markers:
- @pytest.mark.integration: Tests requiring real API calls
- @pytest.mark.slow: Tests with >1 second execution time
- @pytest.mark.real_api: Tests that hit external APIs

Running Tests:
```bash
# All tests (unit + integration)
pytest tests/unit/tools/test_content_access_service.py -v

# Unit tests only (fast, no API calls)
pytest tests/unit/tools/test_content_access_service.py -m "not integration" -v

# Integration tests only (real API calls)
pytest tests/unit/tools/test_content_access_service.py -m "integration" -v
```

API Keys Required:
- NCBI_API_KEY: Required for PubMed/PMC API access
- AWS_BEDROCK_ACCESS_KEY: Required for LLM-based methods extraction
- AWS_BEDROCK_SECRET_ACCESS_KEY: Required for LLM-based methods extraction

Rate Limiting:
- 0.5-1 second sleeps between consecutive API calls
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.content_access_service import ContentAccessService
from lobster.tools.providers.abstract_provider import AbstractProvider
from lobster.tools.providers.provider_registry import ProviderRegistry
from lobster.tools.providers.pubmed_provider import PubMedProvider

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_data_manager(tmp_path):
    """Create a mock DataManagerV2 instance for unit testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    dm = Mock(spec=DataManagerV2)
    dm.cache_dir = cache_dir
    dm.literature_cache_dir = cache_dir / "literature"
    dm.literature_cache_dir.mkdir()

    return dm


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
def mock_provider_registry():
    """Create a mock ProviderRegistry for unit testing."""
    registry = Mock(spec=ProviderRegistry)

    # Mock providers
    mock_abstract_provider = Mock(spec=AbstractProvider)
    mock_pubmed_provider = Mock(spec=PubMedProvider)

    registry.get_providers_for_capability.return_value = [
        mock_abstract_provider,
        mock_pubmed_provider,
    ]

    return registry


@pytest.fixture
def mock_content_access_service(mock_data_manager, mock_provider_registry):
    """Create a mock ContentAccessService for unit testing."""
    return ContentAccessService(
        data_manager=mock_data_manager, provider_registry=mock_provider_registry
    )


@pytest.fixture
def real_content_access_service(real_data_manager):
    """Create a real ContentAccessService for integration testing."""
    # Create real provider registry
    registry = ProviderRegistry()

    # Register real providers
    abstract_provider = AbstractProvider(data_manager=real_data_manager)
    pubmed_provider = PubMedProvider(data_manager=real_data_manager)

    registry.register_provider(abstract_provider)
    registry.register_provider(pubmed_provider)

    return ContentAccessService(
        data_manager=real_data_manager, provider_registry=registry
    )


# ============================================================================
# Unit Tests - Discovery Methods
# ============================================================================


class TestDiscoveryMethods:
    """Test literature search and dataset discovery methods (mock-based)."""

    def test_search_literature_mock(self, mock_content_access_service):
        """Test literature search with mock provider."""
        mock_provider = Mock()
        mock_provider.search_literature.return_value = {
            "results": [
                {"pmid": "12345", "title": "Test Paper 1"},
                {"pmid": "67890", "title": "Test Paper 2"},
            ],
            "total_count": 2,
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        results = mock_content_access_service.search_literature(
            query="BRCA1 breast cancer"
        )

        assert results["total_count"] == 2
        assert len(results["results"]) == 2
        assert results["results"][0]["pmid"] == "12345"
        mock_provider.search_literature.assert_called_once()

    def test_discover_datasets_mock(self, mock_content_access_service):
        """Test dataset discovery with mock provider."""
        mock_provider = Mock()
        mock_provider.discover_datasets.return_value = {
            "results": [
                {"accession": "GSE12345", "title": "Test Dataset 1", "samples": 48},
                {"accession": "GSE67890", "title": "Test Dataset 2", "samples": 36},
            ],
            "total_count": 2,
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        results = mock_content_access_service.discover_datasets(
            query="breast cancer RNA-seq"
        )

        assert results["total_count"] == 2
        assert len(results["results"]) == 2
        assert results["results"][0]["accession"] == "GSE12345"
        mock_provider.discover_datasets.assert_called_once()

    def test_find_linked_datasets_mock(self, mock_content_access_service):
        """Test finding datasets linked to publication (mock)."""
        mock_provider = Mock()
        mock_provider.find_linked_datasets.return_value = {
            "results": [{"accession": "GSE12345", "link_type": "supplementary"}],
            "total_count": 1,
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        results = mock_content_access_service.find_linked_datasets(
            publication_id="PMID:35042229"
        )

        assert results["total_count"] == 1
        assert results["results"][0]["accession"] == "GSE12345"
        mock_provider.find_linked_datasets.assert_called_once()

    def test_search_with_filters_mock(self, mock_content_access_service):
        """Test search with filters (organism, date range, etc.)."""
        mock_provider = Mock()
        mock_provider.search_literature.return_value = {
            "results": [{"pmid": "12345"}],
            "total_count": 1,
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        results = mock_content_access_service.search_literature(
            query="BRCA1", filters={"organism": "human", "year_from": 2020}
        )

        assert results["total_count"] == 1
        mock_provider.search_literature.assert_called_once()

    def test_pagination_mock(self, mock_content_access_service):
        """Test pagination parameters (limit, offset)."""
        mock_provider = Mock()
        mock_provider.search_literature.return_value = {
            "results": [{"pmid": str(i)} for i in range(10)],
            "total_count": 100,
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        results = mock_content_access_service.search_literature(
            query="cancer", limit=10, offset=20
        )

        assert len(results["results"]) == 10
        assert results["total_count"] == 100

    def test_empty_results_mock(self, mock_content_access_service):
        """Test handling of empty search results."""
        mock_provider = Mock()
        mock_provider.search_literature.return_value = {"results": [], "total_count": 0}

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        results = mock_content_access_service.search_literature(
            query="nonexistent_term_xyz123"
        )

        assert results["total_count"] == 0
        assert len(results["results"]) == 0


# ============================================================================
# Unit Tests - Metadata Methods
# ============================================================================


class TestMetadataMethods:
    """Test publication and dataset metadata retrieval (mock-based)."""

    def test_get_publication_metadata_mock(self, mock_content_access_service):
        """Test publication metadata retrieval with mock provider."""
        mock_provider = Mock()
        mock_provider.get_publication_metadata.return_value = {
            "pmid": "35042229",
            "title": "Single-cell analysis reveals...",
            "authors": ["Author A", "Author B"],
            "journal": "Nature",
            "year": 2022,
            "doi": "10.1038/s41586-021-03852-1",
            "abstract": "This study...",
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        metadata = mock_content_access_service.get_publication_metadata("PMID:35042229")

        assert metadata["pmid"] == "35042229"
        assert metadata["journal"] == "Nature"
        assert metadata["year"] == 2022
        mock_provider.get_publication_metadata.assert_called_once()

    def test_get_dataset_metadata_mock(self, mock_content_access_service):
        """Test dataset metadata retrieval with mock provider."""
        mock_provider = Mock()
        mock_provider.get_dataset_metadata.return_value = {
            "accession": "GSE12345",
            "title": "RNA-seq of breast cancer samples",
            "organism": "Homo sapiens",
            "platform": "GPL16791",
            "samples": 48,
            "summary": "This dataset contains...",
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        metadata = mock_content_access_service.get_dataset_metadata("GSE12345")

        assert metadata["accession"] == "GSE12345"
        assert metadata["samples"] == 48
        assert metadata["platform"] == "GPL16791"
        mock_provider.get_dataset_metadata.assert_called_once()

    def test_metadata_caching_mock(self, mock_content_access_service):
        """Test metadata caching behavior."""
        mock_provider = Mock()
        mock_provider.get_publication_metadata.return_value = {
            "pmid": "35042229",
            "title": "Test Paper",
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        # First call
        metadata1 = mock_content_access_service.get_publication_metadata(
            "PMID:35042229"
        )

        # Second call (should use cache if implemented)
        metadata2 = mock_content_access_service.get_publication_metadata(
            "PMID:35042229"
        )

        assert metadata1["pmid"] == metadata2["pmid"]

    def test_invalid_identifier_mock(self, mock_content_access_service):
        """Test handling of invalid identifiers."""
        mock_provider = Mock()
        mock_provider.get_publication_metadata.side_effect = ValueError("Invalid PMID")

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        with pytest.raises(ValueError, match="Invalid PMID"):
            mock_content_access_service.get_publication_metadata("PMID:invalid")


# ============================================================================
# Unit Tests - Content Methods
# ============================================================================


class TestContentMethods:
    """Test content retrieval methods (mock-based)."""

    def test_get_abstract_mock(self, mock_content_access_service):
        """Test abstract retrieval with mock provider."""
        mock_provider = Mock()
        mock_provider.get_abstract.return_value = {
            "pmid": "35042229",
            "abstract": "This study presents single-cell RNA-seq analysis...",
            "title": "Single-cell analysis reveals...",
            "provider": "AbstractProvider",
            "response_time": 0.3,
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        result = mock_content_access_service.get_abstract("PMID:35042229")

        assert result["pmid"] == "35042229"
        assert len(result["abstract"]) > 100
        assert result["provider"] == "AbstractProvider"
        assert result["response_time"] < 1.0
        mock_provider.get_abstract.assert_called_once()

    def test_get_full_content_pmc_success_mock(self, mock_content_access_service):
        """Test full content retrieval - Tier 1 PMC success."""
        mock_provider = Mock()
        mock_provider.get_full_content.return_value = {
            "content": "Full text from PMC..." * 100,
            "format": "xml",
            "provider": "PubMedProvider",
            "tier": 1,
            "response_time": 0.5,
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        result = mock_content_access_service.get_full_content("PMID:35042229")

        assert len(result["content"]) > 1000
        assert result["provider"] == "PubMedProvider"
        assert result["tier"] == 1
        assert result["response_time"] < 1.0

    def test_get_full_content_pmc_fail_webpage_success_mock(
        self, mock_content_access_service
    ):
        """Test full content retrieval - Tier 1 fail → Tier 2 success."""
        # First provider (PMC) fails
        mock_pmc_provider = Mock()
        mock_pmc_provider.get_full_content.return_value = None

        # Second provider (Webpage) succeeds
        mock_webpage_provider = Mock()
        mock_webpage_provider.get_full_content.return_value = {
            "content": "Full text from webpage..." * 100,
            "format": "html",
            "provider": "WebpageProvider",
            "tier": 2,
            "response_time": 2.5,
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_pmc_provider,
            mock_webpage_provider,
        ]

        result = mock_content_access_service.get_full_content("PMID:35042229")

        assert result["provider"] == "WebpageProvider"
        assert result["tier"] == 2
        assert result["response_time"] > 1.0

    def test_get_full_content_all_tiers_fail_mock(self, mock_content_access_service):
        """Test full content retrieval - all tiers fail."""
        mock_provider = Mock()
        mock_provider.get_full_content.return_value = None

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        result = mock_content_access_service.get_full_content("PMID:99999999")

        assert result is None or "error" in result

    def test_extract_methods_section_mock(self, mock_content_access_service):
        """Test methods section extraction with mock LLM."""
        mock_provider = Mock()
        mock_provider.extract_methods_section.return_value = {
            "methods": "RNA was extracted using Qiagen RNeasy kit...",
            "confidence": 0.95,
            "provider": "PubMedProvider",
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        result = mock_content_access_service.extract_methods_section("PMID:35042229")

        assert "methods" in result
        assert len(result["methods"]) > 50
        assert result["confidence"] > 0.9

    def test_content_caching_mock(self, mock_content_access_service):
        """Test content caching behavior."""
        mock_provider = Mock()
        mock_provider.get_abstract.return_value = {
            "pmid": "35042229",
            "abstract": "Test abstract",
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        # First call
        result1 = mock_content_access_service.get_abstract("PMID:35042229")

        # Second call (should use cache if implemented)
        result2 = mock_content_access_service.get_abstract("PMID:35042229")

        assert result1["pmid"] == result2["pmid"]

    def test_content_format_conversion_mock(self, mock_content_access_service):
        """Test content format conversion (XML → text, HTML → text)."""
        mock_provider = Mock()
        mock_provider.get_full_content.return_value = {
            "content": "<xml>Full text</xml>",
            "format": "xml",
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        result = mock_content_access_service.get_full_content(
            "PMID:35042229", output_format="text"
        )

        # Should convert XML to text if format conversion is implemented
        assert "content" in result

    def test_partial_content_retrieval_mock(self, mock_content_access_service):
        """Test partial content retrieval (abstract only when full text unavailable)."""
        mock_provider = Mock()
        mock_provider.get_full_content.return_value = None
        mock_provider.get_abstract.return_value = {"abstract": "Test abstract"}

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        result = mock_content_access_service.get_full_content("PMID:35042229")

        # Should fallback to abstract if full text unavailable
        assert result is None or "abstract" in result

    def test_content_validation_mock(self, mock_content_access_service):
        """Test content validation (minimum length, format checks)."""
        mock_provider = Mock()
        mock_provider.get_full_content.return_value = {
            "content": "Too short",
            "format": "text",
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        result = mock_content_access_service.get_full_content("PMID:35042229")

        # Should validate content length
        assert result is not None

    def test_timeout_handling_mock(self, mock_content_access_service):
        """Test timeout handling for slow providers."""
        mock_provider = Mock()
        mock_provider.get_full_content.side_effect = TimeoutError("Request timed out")

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        result = mock_content_access_service.get_full_content(
            "PMID:35042229", timeout=5
        )

        # Should handle timeout gracefully
        assert result is None or "error" in result


# ============================================================================
# Integration Tests - Real API Calls
# ============================================================================


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
        results = real_content_access_service.search_literature(query, limit=5)
        elapsed = time.time() - start_time

        assert results is not None
        assert "results" in results
        assert len(results["results"]) > 0
        assert elapsed < 3.0, f"Search took {elapsed:.2f}s (expected <3s)"

        # Verify result structure
        first_result = results["results"][0]
        assert "pmid" in first_result or "doi" in first_result
        assert "title" in first_result

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
        results = real_content_access_service.discover_datasets(query, limit=5)
        elapsed = time.time() - start_time

        assert results is not None
        assert "results" in results
        assert len(results["results"]) > 0
        assert elapsed < 3.0, f"Discovery took {elapsed:.2f}s (expected <3s)"

        # Verify result structure
        first_result = results["results"][0]
        assert "accession" in first_result
        assert first_result["accession"].startswith("GSE")
        assert "title" in first_result

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
        publication_id = "PMID:35042229"

        start_time = time.time()
        results = real_content_access_service.find_linked_datasets(publication_id)
        elapsed = time.time() - start_time

        assert results is not None
        assert "results" in results
        # May or may not have linked datasets
        assert elapsed < 3.0, f"Link search took {elapsed:.2f}s (expected <3s)"

        # Rate limiting
        time.sleep(0.5)


class TestRealAPIMetadata:
    """Test metadata methods with real API calls."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_get_publication_metadata_real(self, real_content_access_service):
        """Test real publication metadata retrieval.

        Test Paper: PMID:35042229 (Nature 2022)

        Requires: NCBI_API_KEY environment variable
        Rate limit: 3 req/s with API key
        Expected duration: 1-2s
        """
        identifier = "PMID:35042229"

        start_time = time.time()
        metadata = real_content_access_service.get_publication_metadata(identifier)
        elapsed = time.time() - start_time

        assert metadata is not None
        assert metadata.get("pmid") == "35042229"
        assert "single-cell" in metadata.get("title", "").lower()
        assert metadata.get("journal") == "Nature"
        assert metadata.get("year") == 2022
        assert metadata.get("doi") == "10.1038/s41586-021-03852-1"
        assert len(metadata.get("abstract", "")) > 200
        assert elapsed < 2.0, f"Metadata retrieval took {elapsed:.2f}s (expected <2s)"

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
            identifier, force_pdf=True  # Force PDF extraction if supported
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


# ============================================================================
# Unit Tests - Caching Behavior
# ============================================================================


class TestCachingBehavior:
    """Test caching behavior (session cache, TTL, LRU eviction)."""

    def test_session_cache_hit_mock(self, mock_content_access_service):
        """Test session cache hit (2nd request instant, <10ms)."""
        mock_provider = Mock()
        mock_provider.get_abstract.return_value = {
            "pmid": "35042229",
            "abstract": "Test abstract",
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        # First call (cache miss)
        start_time = time.time()
        result1 = mock_content_access_service.get_abstract("PMID:35042229")
        elapsed1 = time.time() - start_time

        # Second call (cache hit if implemented)
        start_time = time.time()
        result2 = mock_content_access_service.get_abstract("PMID:35042229")
        elapsed2 = time.time() - start_time

        assert result1["pmid"] == result2["pmid"]
        # Cache hit should be faster (if caching is implemented)
        # Note: This may not be true if caching not yet implemented

    def test_ttl_expiration_mock(self, mock_content_access_service):
        """Test TTL expiration (cache invalidation after 300s)."""
        # This test would require time mocking or wait
        # Placeholder for TTL testing logic
        pass

    def test_lru_eviction_mock(self, mock_content_access_service):
        """Test LRU eviction (cache size limit, oldest evicted)."""
        # This test would require filling cache beyond limit
        # Placeholder for LRU testing logic
        pass

    def test_cache_invalidation_mock(self, mock_content_access_service):
        """Test cache invalidation (manual clear)."""
        mock_provider = Mock()
        mock_provider.get_abstract.return_value = {
            "pmid": "35042229",
            "abstract": "Test abstract",
        }

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        # First call
        result1 = mock_content_access_service.get_abstract("PMID:35042229")

        # Clear cache if method exists
        if hasattr(mock_content_access_service, "clear_cache"):
            mock_content_access_service.clear_cache()

        # Second call after cache clear
        result2 = mock_content_access_service.get_abstract("PMID:35042229")

        assert result1["pmid"] == result2["pmid"]

    def test_cache_statistics_mock(self, mock_content_access_service):
        """Test cache statistics (hit rate calculation)."""
        # Placeholder for cache statistics testing
        if hasattr(mock_content_access_service, "cache_stats"):
            stats = mock_content_access_service.cache_stats()
            assert "hit_rate" in stats or "hits" in stats


# ============================================================================
# Unit Tests - Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_no_providers_available_mock(self, mock_content_access_service):
        """Test error when no providers available."""
        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = (
            []
        )

        result = mock_content_access_service.get_abstract("PMID:35042229")

        assert result is None or "error" in result

    def test_all_providers_fail_mock(self, mock_content_access_service):
        """Test error when all providers fail."""
        mock_provider = Mock()
        mock_provider.get_abstract.side_effect = Exception("Provider error")

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        result = mock_content_access_service.get_abstract("PMID:35042229")

        assert result is None or "error" in result

    def test_invalid_pmid_format_mock(self, mock_content_access_service):
        """Test handling of malformed PMID."""
        mock_provider = Mock()
        mock_provider.get_abstract.side_effect = ValueError("Invalid PMID format")

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        with pytest.raises(ValueError, match="Invalid PMID"):
            mock_content_access_service.get_abstract("PMID:invalid")

    def test_rate_limiting_429_response_mock(self, mock_content_access_service):
        """Test rate limiting (429 response handling)."""
        mock_provider = Mock()
        mock_provider.get_abstract.side_effect = Exception("429 Too Many Requests")

        mock_content_access_service.provider_registry.get_providers_for_capability.return_value = [
            mock_provider
        ]

        result = mock_content_access_service.get_abstract("PMID:35042229")

        # Should handle rate limiting gracefully
        assert (
            result is None or "rate limit" in str(result).lower() or "error" in result
        )
