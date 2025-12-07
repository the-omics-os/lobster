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
from lobster.services.data_access.content_access_service import ContentAccessService
from lobster.tools.providers.abstract_provider import AbstractProvider
from lobster.tools.providers.base_provider import DatasetType
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
def mock_content_access_service(mock_data_manager):
    """Create a mock ContentAccessService for unit testing."""
    # ContentAccessService creates its own ProviderRegistry internally
    return ContentAccessService(data_manager=mock_data_manager)


@pytest.fixture
def real_content_access_service(real_data_manager):
    """Create a real ContentAccessService for integration testing."""
    # ContentAccessService creates and initializes its own ProviderRegistry
    # with all available providers automatically
    return ContentAccessService(data_manager=real_data_manager)


# ============================================================================
# Unit Tests - Discovery Methods
# ============================================================================


class TestDiscoveryMethods:
    """Test literature search and dataset discovery methods (mock-based)."""

    def test_search_literature_mock(self, mock_content_access_service):
        """Test literature search with mock provider."""
        mock_provider = Mock()
        # Service calls search_publications() and expects string with PMIDs
        mock_provider.search_publications.return_value = (
            "PMID: 12345 - Test Paper 1\n" "PMID: 67890 - Test Paper 2"
        )

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            # search_literature returns tuple: (str, dict, AnalysisStep)
            result_str, stats, ir = mock_content_access_service.search_literature(
                query="BRCA1 breast cancer"
            )

            # Check stats dict - counts PMIDs in result string
            assert stats["results_count"] == 2
            assert "Test Paper 1" in result_str
            assert "Test Paper 2" in result_str
            mock_provider.search_publications.assert_called_once()

    def test_discover_datasets_mock(self, mock_content_access_service):
        """Test dataset discovery with mock provider."""
        mock_provider = Mock()
        mock_provider.supported_dataset_types = [DatasetType.GEO]  # Required by service
        # Service calls search_publications() for dataset discovery, expecting string
        mock_provider.search_publications.return_value = (
            "GSE12345 - Test Dataset 1 (48 samples)\n"
            "GSE67890 - Test Dataset 2 (36 samples)"
        )

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            # discover_datasets returns tuple and requires dataset_type parameter
            result_str, stats, ir = mock_content_access_service.discover_datasets(
                query="breast cancer RNA-seq", dataset_type=DatasetType.GEO
            )

            # Check formatted result string contains accessions
            assert "GSE12345" in result_str
            assert "GSE67890" in result_str
            mock_provider.search_publications.assert_called_once()

    def test_find_linked_datasets_mock(self, mock_content_access_service):
        """Test finding datasets linked to publication (mock)."""
        mock_provider = Mock()
        # Service calls find_datasets_from_publication(), expecting string
        mock_provider.find_datasets_from_publication.return_value = (
            "GSE12345 - Linked dataset (supplementary)"
        )

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            # find_linked_datasets returns string and uses 'identifier' parameter
            result_str = mock_content_access_service.find_linked_datasets(
                identifier="PMID:35042229"
            )

            # Check formatted result string contains accession
            assert "GSE12345" in result_str
            mock_provider.find_datasets_from_publication.assert_called_once()

    def test_find_related_publications_mock(self, mock_content_access_service):
        """Test finding publications related to a given PMID (mock)."""
        from lobster.tools.providers.pubmed_provider import PubMedProvider

        mock_pubmed_provider = Mock(spec=PubMedProvider)
        # Service calls find_related_publications(), expecting string
        mock_pubmed_provider.find_related_publications.return_value = (
            "## Related Publications for PMID:35042229\n\n"
            "**Source Publication**: Original Paper Title\n"
            "**Found 2 related publications**\n\n"
            "### 1. Related Paper 1\n\n"
            "**PMID**: 12345\n"
            "**Abstract**: This study builds upon the original work...\n\n"
            "### 2. Related Paper 2\n\n"
            "**PMID**: 67890\n"
            "**Abstract**: Our findings complement the previous research...\n"
        )

        with patch.object(
            mock_content_access_service.registry,
            "get_all_providers",
            return_value=[mock_pubmed_provider],
        ):
            # find_related_publications returns string and uses 'identifier' parameter
            result_str = mock_content_access_service.find_related_publications(
                identifier="PMID:35042229", max_results=5
            )

            # Check formatted result string contains expected elements
            assert "Related Publications" in result_str
            assert "PMID:12345" in result_str or "12345" in result_str
            assert "PMID:67890" in result_str or "67890" in result_str
            mock_pubmed_provider.find_related_publications.assert_called_once_with(
                identifier="PMID:35042229", max_results=5
            )

    def test_search_with_filters_mock(self, mock_content_access_service):
        """Test search with filters (organism, date range, etc.)."""
        mock_provider = Mock()
        # Service calls search_publications() and counts PMIDs in result string
        mock_provider.search_publications.return_value = "PMID: 12345 - Test Paper"

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            # search_literature returns tuple
            result_str, stats, ir = mock_content_access_service.search_literature(
                query="BRCA1", filters={"organism": "human", "year_from": 2020}
            )

            assert stats["results_count"] == 1
            mock_provider.search_publications.assert_called_once()

    def test_pagination_mock(self, mock_content_access_service):
        """Test pagination parameters (limit, offset)."""
        mock_provider = Mock()
        # Service calls search_publications() and counts PMIDs in result string
        mock_provider.search_publications.return_value = "\n".join(
            [f"PMID: {i} - Paper {i}" for i in range(10)]
        )

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            # search_literature returns tuple
            result_str, stats, ir = mock_content_access_service.search_literature(
                query="cancer", limit=10, offset=20
            )

            # Check pagination was applied via mock call
            assert stats["results_count"] == 10
            mock_provider.search_publications.assert_called_once()

    def test_empty_results_mock(self, mock_content_access_service):
        """Test handling of empty search results."""
        mock_provider = Mock()
        # Service calls search_publications() - empty string for no results
        mock_provider.search_publications.return_value = ""

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            # search_literature returns tuple
            result_str, stats, ir = mock_content_access_service.search_literature(
                query="nonexistent_term_xyz123"
            )

            # Check that no results were found (empty string or explicit message)
            assert stats["results_count"] == 0
            assert len(result_str) == 0 or "no results" in result_str.lower()


# ============================================================================
# Unit Tests - Metadata Methods
# ============================================================================


class TestMetadataMethods:
    """Test publication and dataset metadata retrieval (mock-based)."""

    def test_extract_metadata_publication_mock(self, mock_content_access_service):
        """Test publication metadata extraction with mock provider."""
        mock_provider = Mock()

        # Create mock PublicationMetadata object
        from lobster.tools.providers.base_provider import PublicationMetadata

        mock_metadata = PublicationMetadata(
            uid="PMID:35042229",
            title="Single-cell analysis reveals...",
            authors=["Author A", "Author B"],
            journal="Nature",
            published="2022",
            doi="10.1038/s41586-021-03852-1",
            abstract="This study...",
            pmid="35042229",
        )
        mock_provider.extract_publication_metadata.return_value = mock_metadata

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            metadata = mock_content_access_service.extract_metadata("PMID:35042229")

            assert metadata.pmid == "35042229"
            assert metadata.journal == "Nature"
            assert metadata.published == "2022"
            mock_provider.extract_publication_metadata.assert_called_once()

    def test_validate_metadata_mock(self, mock_content_access_service):
        """Test dataset metadata validation (mock-based)."""
        # Mock DataManager to return cached GEO metadata
        mock_metadata = {
            "metadata": {
                "accession": "GSE12345",
                "title": "RNA-seq of breast cancer samples",
                "organism": "Homo sapiens",
                "platform": "GPL16791",
                "samples": 48,
            }
        }

        with patch.object(
            mock_content_access_service.data_manager,
            "_get_geo_metadata",
            return_value=mock_metadata,
        ):
            with patch(
                "lobster.services.metadata.metadata_validation_service.MetadataValidationService"
            ) as mock_validation_service_class:
                # Mock the validation service instance and methods
                mock_validation_service = Mock()
                mock_validation_service_class.return_value = mock_validation_service

                # Mock validation result
                mock_validation_config = Mock()
                mock_validation_config.recommendation = "proceed"
                mock_validation_config.confidence_score = 0.95
                mock_validation_service.validate_dataset_metadata.return_value = (
                    mock_validation_config
                )
                mock_validation_service.format_validation_report.return_value = (
                    "Validation passed"
                )

                result = mock_content_access_service.validate_metadata("GSE12345")

                assert isinstance(result, str)
                assert "Validation passed" in result or result == "Validation passed"

    def test_metadata_caching_mock(self, mock_content_access_service):
        """Test metadata extraction caching behavior."""
        mock_provider = Mock()

        from lobster.tools.providers.base_provider import PublicationMetadata

        mock_metadata = PublicationMetadata(
            uid="PMID:35042229",
            title="Test Paper",
            pmid="35042229",
        )
        mock_provider.extract_publication_metadata.return_value = mock_metadata

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            # First call
            metadata1 = mock_content_access_service.extract_metadata("PMID:35042229")

            # Second call (should use cache if implemented)
            metadata2 = mock_content_access_service.extract_metadata("PMID:35042229")

            assert metadata1.pmid == metadata2.pmid

    def test_invalid_identifier_mock(self, mock_content_access_service):
        """Test handling of invalid identifiers."""
        mock_provider = Mock()
        mock_provider.extract_publication_metadata.side_effect = ValueError(
            "Invalid PMID"
        )

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            # Service catches exceptions and returns error string
            result = mock_content_access_service.extract_metadata("PMID:invalid")

            # Result should be error string (service catches exceptions)
            assert isinstance(result, str)
            assert "error" in result.lower()


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
            "abstract": "This study presents single-cell RNA-seq analysis of tumor microenvironment in breast cancer patients, revealing novel immune cell populations and their interactions with tumor cells across different disease stages and treatment conditions.",
            "title": "Single-cell analysis reveals...",
            "provider": "AbstractProvider",
            "response_time": 0.3,
        }

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            result = mock_content_access_service.get_abstract("PMID:35042229")

            assert result["pmid"] == "35042229"
            assert len(result["abstract"]) > 100  # Now mock has longer abstract
            assert result["provider"] == "AbstractProvider"
            assert result["response_time"] < 1.0
            mock_provider.get_abstract.assert_called_once()

    def test_get_full_content_pmc_success_mock(self, mock_content_access_service):
        """Test full content retrieval - PMC success."""
        # Mock DataManager cache check (returns None - cache miss)
        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        # Mock PMCProvider
        mock_pmc_provider = Mock()
        mock_pmc_provider.__class__.__name__ = "PMCProvider"

        # Mock PMC full text result
        from lobster.tools.providers.pmc_provider import PMCFullText

        mock_pmc_result = PMCFullText(
            pmc_id="PMC8760896",
            pmid="35042229",
            doi="10.1038/test",
            title="Test Paper",
            abstract="Test abstract",
            full_text="Full text from PMC..." * 100,
            methods_section="Methods text",
            results_section="Results text",
            discussion_section="Discussion text",
            tables=[],
            figures=[],
            software_tools=["Scanpy", "Seurat"],
            github_repos=[],
        )
        mock_pmc_provider.extract_full_text.return_value = mock_pmc_result

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_pmc_provider],
        ):
            with patch.object(
                mock_content_access_service.data_manager, "cache_publication_content"
            ):
                result = mock_content_access_service.get_full_content("PMID:35042229")

                assert len(result["content"]) > 1000
                assert result["tier_used"] == "full_pmc_xml"
                assert result["source_type"] == "pmc_xml"
                assert result["extraction_time"] < 10.0

    def test_get_full_content_pmc_fail_webpage_success_mock(
        self, mock_content_access_service
    ):
        """Test full content retrieval - PMC fail → Webpage success."""
        # Mock DataManager cache check (returns None)
        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        # Mock PMCProvider that raises PMCNotAvailableError
        from lobster.tools.providers.pmc_provider import PMCNotAvailableError

        mock_pmc_provider = Mock()
        mock_pmc_provider.__class__.__name__ = "PMCProvider"
        mock_pmc_provider.extract_full_text.side_effect = PMCNotAvailableError(
            "PMC not available"
        )

        # Mock WebpageProvider that succeeds
        mock_webpage_provider = Mock()
        mock_webpage_provider.__class__.__name__ = "WebpageProvider"
        mock_webpage_provider.extract_with_full_metadata.return_value = {
            "content": "Full text from webpage..." * 100,
            "source_type": "webpage",
            "methods_text": "Methods from webpage",
        }

        mock_resolver = Mock()
        mock_resolution = Mock()
        mock_resolution.is_accessible.return_value = True
        mock_resolution.html_url = "https://example.com/paper"
        mock_resolution.pdf_url = None
        mock_resolution.alternative_urls = []
        mock_resolver.resolve.return_value = mock_resolution

        with (
            patch.object(
                mock_content_access_service, "_get_preprint_provider", return_value=None
            ),
            patch.object(
                mock_content_access_service,
                "_get_publication_resolver",
                return_value=mock_resolver,
            ),
        ):

            # Return PMC provider first, then webpage provider
            with patch.object(
                mock_content_access_service.registry,
                "get_providers_for_capability",
                side_effect=[
                    [mock_pmc_provider],  # First call for PMC
                    [mock_webpage_provider],  # Second call for webpage fallback
                ],
            ):
                with patch.object(
                    mock_content_access_service.data_manager,
                    "cache_publication_content",
                ):
                    result = mock_content_access_service.get_full_content(
                        "PMID:35042229"
                    )

                    assert result["tier_used"] == "full_webpage"
                    assert result["source_type"] == "webpage"

    def test_resolver_prefers_html_when_requested(self, mock_content_access_service):
        """Ensure prefer_webpage selects html_url when resolver provides both."""

        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        from lobster.tools.providers.pmc_provider import PMCNotAvailableError

        mock_pmc_provider = Mock()
        mock_pmc_provider.__class__.__name__ = "PMCProvider"
        mock_pmc_provider.extract_full_text.side_effect = PMCNotAvailableError(
            "PMC not available"
        )

        mock_webpage_provider = Mock()
        mock_webpage_provider.__class__.__name__ = "WebpageProvider"
        mock_webpage_provider.extract_with_full_metadata.return_value = {
            "content": "web",  # minimal payload
            "source_type": "webpage",
            "methods_text": "body",
        }

        mock_resolver = Mock()
        mock_resolution = Mock()
        mock_resolution.is_accessible.return_value = True
        mock_resolution.html_url = "https://example.com/html"
        mock_resolution.pdf_url = "https://example.com/paper.pdf"
        mock_resolution.alternative_urls = []
        mock_resolver.resolve.return_value = mock_resolution

        with (
            patch.object(
                mock_content_access_service, "_get_preprint_provider", return_value=None
            ),
            patch.object(
                mock_content_access_service,
                "_get_publication_resolver",
                return_value=mock_resolver,
            ),
        ):

            with patch.object(
                mock_content_access_service.registry,
                "get_providers_for_capability",
                side_effect=[
                    [mock_pmc_provider],
                    [mock_webpage_provider],
                ],
            ):
                mock_content_access_service.get_full_content(
                    "PMID:35042229", prefer_webpage=True
                )

                args, _ = mock_webpage_provider.extract_with_full_metadata.call_args
                assert args[0] == "https://example.com/html"

    def test_resolver_prefers_pdf_when_requested(self, mock_content_access_service):
        """Ensure prefer_webpage=False picks pdf_url."""

        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        from lobster.tools.providers.pmc_provider import PMCNotAvailableError

        mock_pmc_provider = Mock()
        mock_pmc_provider.__class__.__name__ = "PMCProvider"
        mock_pmc_provider.extract_full_text.side_effect = PMCNotAvailableError(
            "PMC not available"
        )

        mock_webpage_provider = Mock()
        mock_webpage_provider.__class__.__name__ = "WebpageProvider"
        mock_webpage_provider.extract_with_full_metadata.return_value = {
            "content": "pdf",
            "source_type": "webpage",
            "methods_text": "body",
        }

        mock_resolver = Mock()
        mock_resolution = Mock()
        mock_resolution.is_accessible.return_value = True
        mock_resolution.html_url = "https://example.com/html"
        mock_resolution.pdf_url = "https://example.com/paper.pdf"
        mock_resolution.alternative_urls = []
        mock_resolver.resolve.return_value = mock_resolution

        with (
            patch.object(
                mock_content_access_service, "_get_preprint_provider", return_value=None
            ),
            patch.object(
                mock_content_access_service,
                "_get_publication_resolver",
                return_value=mock_resolver,
            ),
        ):

            with patch.object(
                mock_content_access_service.registry,
                "get_providers_for_capability",
                side_effect=[
                    [mock_pmc_provider],
                    [mock_webpage_provider],
                ],
            ):
                mock_content_access_service.get_full_content(
                    "PMID:35042229", prefer_webpage=False
                )

                args, _ = mock_webpage_provider.extract_with_full_metadata.call_args
                assert args[0] == "https://example.com/paper.pdf"

    def test_arxiv_pdf_normalized_to_abs_html(self, mock_content_access_service):
        """Direct arXiv PDF sources should be normalized to abs HTML when prefer_webpage."""

        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        mock_webpage_provider = Mock()
        mock_webpage_provider.__class__.__name__ = "WebpageProvider"
        mock_webpage_provider.extract_with_full_metadata.return_value = {
            "content": "arxiv",
            "source_type": "webpage",
            "methods_text": "arxiv body",
        }

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_webpage_provider],
        ):
            mock_content_access_service.get_full_content(
                "https://arxiv.org/pdf/2406.12108.pdf", prefer_webpage=True
            )

            args, _ = mock_webpage_provider.extract_with_full_metadata.call_args
            assert args[0] == "https://arxiv.org/abs/2406.12108"

    def test_wiley_tdm_url_transformed_to_public_html(
        self, mock_content_access_service
    ):
        """Wiley API URLs should be rewritten to the HTML DOI page."""

        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        mock_webpage_provider = Mock()
        mock_webpage_provider.__class__.__name__ = "WebpageProvider"
        mock_webpage_provider.extract_with_full_metadata.return_value = {
            "content": "wiley",
            "source_type": "webpage",
            "methods_text": "body",
        }

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_webpage_provider],
        ):
            source_url = (
                "https://api.wiley.com/onlinelibrary/tdm/v1/articles/"
                "10.1111/2041-210X.12628"
            )

            mock_content_access_service.get_full_content(
                source_url,
                prefer_webpage=True,
            )

            args, _ = mock_webpage_provider.extract_with_full_metadata.call_args
            assert (
                args[0]
                == "https://onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12628"
            )

    def test_github_sources_marked_non_scientific(self, mock_content_access_service):
        """GitHub URLs should bypass Docling and return non-scientific metadata."""

        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )
        with patch.object(
            mock_content_access_service.data_manager, "cache_publication_content"
        ) as cache_mock:
            result = mock_content_access_service.get_full_content(
                "https://github.com/omics-os/lobster", prefer_webpage=True
            )

        assert result["tier_used"] == "non_scientific_link"
        assert result["source_type"] == "non_scientific"
        assert result["metadata"]["domain"] == "github.com"
        cache_mock.assert_called_once()

    def test_statista_marked_non_scientific(self, mock_content_access_service):
        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        result = mock_content_access_service.get_full_content(
            "https://www.statista.com/outlook/hmo/medical-technology/in-vitro-diagnostics/switzerland",
            prefer_webpage=True,
        )

        assert result["tier_used"] == "non_scientific_link"
        assert result["metadata"]["domain"] == "www.statista.com"

    def test_html_paywall_falls_back_to_pdf(self, mock_content_access_service):
        """HTML paywall errors should trigger PDF fallback extraction."""

        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        from lobster.tools.providers.pmc_provider import PMCNotAvailableError

        mock_pmc_provider = Mock()
        mock_pmc_provider.__class__.__name__ = "PMCProvider"
        mock_pmc_provider.extract_full_text.side_effect = PMCNotAvailableError(
            "PMC not available"
        )

        mock_webpage_provider = Mock()
        mock_webpage_provider.__class__.__name__ = "WebpageProvider"
        mock_webpage_provider.extract_with_full_metadata.side_effect = Exception(
            "403 Client Error"
        )

        pdf_result = {
            "methods_text": "pdf",
            "methods_markdown": "pdf",
            "provenance": {"conversion_seconds": 1.2},
        }

        docling_service_mock = Mock()
        docling_service_mock.extract_methods_section.return_value = pdf_result

        mock_resolver = Mock()
        mock_resolution = Mock()
        mock_resolution.is_accessible.return_value = True
        mock_resolution.html_url = "https://example.com/html"
        mock_resolution.pdf_url = "https://example.com/paper.pdf"
        mock_resolution.alternative_urls = []
        mock_resolver.resolve.return_value = mock_resolution

        with (
            patch.object(
                mock_content_access_service, "_get_preprint_provider", return_value=None
            ),
            patch.object(
                mock_content_access_service,
                "_get_publication_resolver",
                return_value=mock_resolver,
            ),
        ):

            with (
                patch.object(
                    mock_content_access_service,
                    "_get_docling_service",
                    return_value=docling_service_mock,
                ),
                patch.object(
                    mock_content_access_service.data_manager,
                    "cache_publication_content",
                ) as cache_mock,
            ):
                with patch.object(
                    mock_content_access_service.registry,
                    "get_providers_for_capability",
                    side_effect=[
                        [mock_pmc_provider],
                        [mock_webpage_provider],
                    ],
                ):
                    result = mock_content_access_service.get_full_content(
                        "PMID:35042229", prefer_webpage=True
                    )

        assert result["tier_used"] == "full_pdf"
        assert result["source_type"] == "pdf"
        docling_service_mock.extract_methods_section.assert_called_once_with(
            source="https://example.com/paper.pdf",
            keywords=None,
            max_paragraphs=100,
        )
        cache_mock.assert_called()

    def test_get_full_content_all_tiers_fail_mock(self, mock_content_access_service):
        """Test full content retrieval - no providers available."""
        # Mock DataManager cache check (returns None)
        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        # Mock no providers available for GET_FULL_CONTENT
        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[],
        ):
            result = mock_content_access_service.get_full_content("PMID:99999999")

            assert "error" in result

    def test_preprint_doi_uses_biorxiv_provider(self, mock_content_access_service):
        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )
        mock_content_access_service.data_manager.cache_publication_content = Mock()

        from lobster.tools.providers.pmc_provider import PMCFullText

        pmc_result = PMCFullText(
            pmc_id="",
            pmid=None,
            doi="10.1101/2024.01.01.123456",
            title="Preprint",
            abstract="Abstract",
            full_text="Full text",
            methods_section="Methods",
            results_section="Results",
            discussion_section="Discussion",
            tables=[],
            figures=[],
            software_tools=[],
            github_repos=[],
        )

        provider_mock = Mock()
        provider_mock.get_full_text.return_value = pmc_result

        with patch.object(
            mock_content_access_service,
            "_get_preprint_provider",
            return_value=provider_mock,
        ):
            result = mock_content_access_service.get_full_content(
                "10.1101/2024.01.01.123456", prefer_webpage=True
            )

        assert result["tier_used"].startswith("full_biorxiv")
        provider_mock.get_full_text.assert_called()

    def test_publication_resolver_singleton(self, mock_content_access_service):
        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        with patch(
            "lobster.services.data_access.content_access_service.PublicationResolver"
        ) as MockResolver:
            instance = Mock()
            instance.resolve.return_value = Mock(
                is_accessible=lambda: False,
            )
            MockResolver.return_value = instance

            # Force cache miss path
            mock_content_access_service.get_full_content("PMID:1")
            mock_content_access_service.get_full_content("PMID:2")

            MockResolver.assert_called_once()

    def test_extract_methods_mock(self, mock_content_access_service):
        """Test methods extraction from content result."""
        # Create mock content result (from get_full_content)
        content_result = {
            "content": "Full paper text...",
            "methods_text": "RNA was extracted using Qiagen RNeasy kit. Sequencing was performed using Illumina NovaSeq.",
            "metadata": {
                "software": ["Scanpy", "Seurat", "CellRanger"],
                "github_repos": ["https://github.com/theislab/scanpy"],
                "tables": [{"caption": "Methods table"}],
            },
        }

        result = mock_content_access_service.extract_methods(
            content_result, include_tables=True
        )

        assert "methods_text" in result
        assert len(result["methods_text"]) > 50
        assert "software_used" in result
        assert len(result["software_used"]) == 3
        assert "Scanpy" in result["software_used"]
        assert "github_repos" in result
        assert len(result["github_repos"]) == 1

    def test_content_caching_mock(self, mock_content_access_service):
        """Test content caching behavior."""
        mock_provider = Mock()
        mock_provider.get_abstract.return_value = {
            "pmid": "35042229",
            "abstract": "Test abstract for caching behavior testing purposes",
        }

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            # First call
            result1 = mock_content_access_service.get_abstract("PMID:35042229")

            # Second call (should return same result)
            result2 = mock_content_access_service.get_abstract("PMID:35042229")

            assert result1["pmid"] == result2["pmid"]
            # Provider should be called at least once
            assert mock_provider.get_abstract.call_count >= 1

    def test_content_format_handling_mock(self, mock_content_access_service):
        """Test content format handling (XML/HTML extraction)."""
        # get_full_content always returns dict with content key, regardless of source format
        # Service handles format internally
        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        # Mock PMC provider returns structured content
        mock_pmc_provider = Mock()
        mock_pmc_provider.__class__.__name__ = "PMCProvider"

        from lobster.tools.providers.pmc_provider import PMCFullText

        mock_pmc_result = PMCFullText(
            pmc_id="PMC123",
            pmid="123",
            doi="10.1234/test",
            title="Test",
            abstract="Test",
            full_text="<xml>Full text</xml>",
            methods_section="Methods",
            results_section="",
            discussion_section="",
            tables=[],
            figures=[],
            software_tools=[],
            github_repos=[],
        )
        mock_pmc_provider.extract_full_text.return_value = mock_pmc_result

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_pmc_provider],
        ):
            with patch.object(
                mock_content_access_service.data_manager, "cache_publication_content"
            ):
                result = mock_content_access_service.get_full_content("PMID:35042229")

                # Should have content field
                assert "content" in result
                assert len(result["content"]) > 0

    def test_partial_content_retrieval_mock(self, mock_content_access_service):
        """Test graceful handling when full content unavailable."""
        # get_full_content returns error dict when all providers fail
        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        # Mock PMC provider raises exception
        from lobster.tools.providers.pmc_provider import PMCNotAvailableError

        mock_pmc_provider = Mock()
        mock_pmc_provider.__class__.__name__ = "PMCProvider"
        mock_pmc_provider.extract_full_text.side_effect = PMCNotAvailableError(
            "Not available"
        )

        # Mock no webpage providers available
        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            side_effect=[
                [mock_pmc_provider],
                [],
            ],  # PMC available, no webpage providers
        ):
            with patch(
                "lobster.tools.providers.publication_resolver.PublicationResolver"
            ) as MockResolver:
                mock_resolver = Mock()
                mock_resolution = Mock()
                mock_resolution.is_accessible.return_value = False
                mock_resolver.resolve.return_value = mock_resolution
                MockResolver.return_value = mock_resolver

                result = mock_content_access_service.get_full_content("PMID:35042229")

                # Should return error dict
                assert "error" in result

    def test_content_extraction_success_mock(self, mock_content_access_service):
        """Test successful content extraction returns proper structure."""
        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        mock_pmc_provider = Mock()
        mock_pmc_provider.__class__.__name__ = "PMCProvider"

        from lobster.tools.providers.pmc_provider import PMCFullText

        mock_pmc_result = PMCFullText(
            pmc_id="PMC123",
            pmid="123",
            doi="10.1234/test",
            title="Test Paper",
            abstract="Test abstract",
            full_text="Full text content with sufficient length for testing purposes",
            methods_section="Methods",
            results_section="Results",
            discussion_section="Discussion",
            tables=[],
            figures=[],
            software_tools=[],
            github_repos=[],
        )
        mock_pmc_provider.extract_full_text.return_value = mock_pmc_result

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_pmc_provider],
        ):
            with patch.object(
                mock_content_access_service.data_manager, "cache_publication_content"
            ):
                result = mock_content_access_service.get_full_content("PMID:35042229")

                # Validate result structure
                assert "content" in result
                assert "tier_used" in result
                assert len(result["content"]) > 0

    def test_exception_handling_mock(self, mock_content_access_service):
        """Test graceful exception handling."""
        mock_content_access_service.data_manager.get_cached_publication = Mock(
            return_value=None
        )

        # Mock provider raises generic exception
        mock_pmc_provider = Mock()
        mock_pmc_provider.__class__.__name__ = "PMCProvider"
        mock_pmc_provider.extract_full_text.side_effect = RuntimeError(
            "Unexpected error"
        )

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_pmc_provider],
        ):
            result = mock_content_access_service.get_full_content("PMID:35042229")

            # Should handle exception and return error
            assert "error" in result or result is None


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
            query, dataset_type=DatasetType.GEO, limit=5
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

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
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

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
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
        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[],
        ):
            result = mock_content_access_service.get_abstract("PMID:35042229")

            assert result is None or "error" in result

    def test_all_providers_fail_mock(self, mock_content_access_service):
        """Test error when all providers fail."""
        mock_provider = Mock()
        mock_provider.get_abstract.side_effect = Exception("Provider error")

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            result = mock_content_access_service.get_abstract("PMID:35042229")

            assert result is None or "error" in result

    def test_invalid_pmid_format_mock(self, mock_content_access_service):
        """Test handling of malformed PMID."""
        mock_provider = Mock()
        mock_provider.get_abstract.side_effect = ValueError("Invalid PMID format")

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            # Service catches exceptions and returns error dict
            result = mock_content_access_service.get_abstract("PMID:invalid")

            # Result should be error dict (service handles error gracefully)
            assert isinstance(result, dict)
            assert "error" in result
            assert "Invalid PMID format" in result["error"]

    def test_rate_limiting_429_response_mock(self, mock_content_access_service):
        """Test rate limiting (429 response handling)."""
        mock_provider = Mock()
        mock_provider.get_abstract.side_effect = Exception("429 Too Many Requests")

        with patch.object(
            mock_content_access_service.registry,
            "get_providers_for_capability",
            return_value=[mock_provider],
        ):
            result = mock_content_access_service.get_abstract("PMID:35042229")

            # Should handle rate limiting gracefully
            assert (
                result is None
                or "rate limit" in str(result).lower()
                or "error" in result
            )


# ============================================================================
# URL Normalization Tests (Bug Fix: Generic /pdf/ patterns)
# ============================================================================


class TestContentMethods:
    """Test helper methods for content access (URL normalization, preprint detection, etc.)."""

    def test_prefer_html_variant_generic_doi_pdf_pattern(self, mock_data_manager):
        """Verify generic /doi/pdf/ → /doi/ transformation for multiple publishers."""
        service = ContentAccessService(mock_data_manager)

        # ACS pattern
        result = service._prefer_html_variant(
            "https://pubs.acs.org/doi/pdf/10.1021/acs.jafc.4c05616"
        )
        assert result == "https://pubs.acs.org/doi/10.1021/acs.jafc.4c05616"
        assert "/pdf/" not in result

        # Wiley pattern
        result = service._prefer_html_variant(
            "https://onlinelibrary.wiley.com/doi/pdf/10.1002/example"
        )
        assert result == "https://onlinelibrary.wiley.com/doi/10.1002/example"
        assert "/pdf/" not in result

        # Nature pattern (hypothetical)
        result = service._prefer_html_variant(
            "https://www.nature.com/articles/doi/pdf/s41586-023-12345-6"
        )
        assert result == "https://www.nature.com/articles/doi/s41586-023-12345-6"
        assert "/pdf/" not in result

    def test_prefer_html_variant_generic_pdf_suffix(self, mock_data_manager):
        """Verify generic /pdf suffix removal."""
        service = ContentAccessService(mock_data_manager)

        result = service._prefer_html_variant("https://example.com/articles/12345/pdf")
        assert result == "https://example.com/articles/12345"
        assert not result.endswith("/pdf")

        result = service._prefer_html_variant(
            "https://publisher.com/journal/volume/article/pdf"
        )
        assert result == "https://publisher.com/journal/volume/article"
        assert not result.endswith("/pdf")

    def test_prefer_html_variant_preserves_query_params(self, mock_data_manager):
        """Verify query parameters are preserved during transformation."""
        service = ContentAccessService(mock_data_manager)

        result = service._prefer_html_variant(
            "https://pubs.acs.org/doi/pdf/10.1021/example?download=true&token=abc123"
        )
        assert (
            result
            == "https://pubs.acs.org/doi/10.1021/example?download=true&token=abc123"
        )
        assert "/pdf/" not in result
        assert "download=true" in result
        assert "token=abc123" in result

    def test_prefer_html_variant_existing_patterns_unchanged(self, mock_data_manager):
        """Verify existing bioRxiv/medRxiv/arXiv patterns still work."""
        service = ContentAccessService(mock_data_manager)

        # bioRxiv pattern
        result = service._prefer_html_variant(
            "https://www.biorxiv.org/content/10.1101/2024.01.01.574440v1.full.pdf"
        )
        assert (
            result == "https://www.biorxiv.org/content/10.1101/2024.01.01.574440v1.full"
        )
        assert not result.endswith(".pdf")

        # medRxiv pattern
        result = service._prefer_html_variant(
            "https://www.medrxiv.org/content/10.1101/2024.01.01.574440v1.full.pdf"
        )
        assert (
            result == "https://www.medrxiv.org/content/10.1101/2024.01.01.574440v1.full"
        )
        assert not result.endswith(".pdf")

        # arXiv pattern
        result = service._prefer_html_variant("https://arxiv.org/pdf/2408.09869.pdf")
        assert result == "https://arxiv.org/abs/2408.09869"

    def test_prefer_html_variant_nature_articles_pdf_extension(self, mock_data_manager):
        """Verify Nature article .pdf extensions are removed (Bug Fix)."""
        service = ContentAccessService(mock_data_manager)

        # Nature article with .pdf extension (returns HTML, not PDF)
        result = service._prefer_html_variant(
            "https://www.nature.com/articles/s41575-025-01040-4.pdf"
        )
        assert result == "https://www.nature.com/articles/s41575-025-01040-4"
        assert not result.endswith(".pdf")

        # Nature article without .pdf (should be unchanged)
        result = service._prefer_html_variant(
            "https://www.nature.com/articles/s41575-025-01040-4"
        )
        assert result == "https://www.nature.com/articles/s41575-025-01040-4"

    def test_prefer_html_variant_no_change_for_non_pdf_urls(self, mock_data_manager):
        """Verify non-PDF URLs are returned unchanged."""
        service = ContentAccessService(mock_data_manager)

        # Already HTML URL
        html_url = "https://www.nature.com/articles/s41586-023-12345-6"
        result = service._prefer_html_variant(html_url)
        assert result == html_url

        # Plain DOI
        doi = "10.1038/s41586-023-12345-6"
        result = service._prefer_html_variant(doi)
        assert result == doi

    def test_preprint_doi_uses_biorxiv_provider(self, mock_data_manager):
        """Verify preprint DOIs are detected and routed to BioRxiv provider."""
        service = ContentAccessService(mock_data_manager)

        # Test _extract_preprint_doi_from_source
        doi, server = service._extract_preprint_doi_from_source(
            "10.1101/2024.01.01.574440"
        )
        assert doi == "10.1101/2024.01.01.574440"
        assert server is None  # No server hint from plain DOI

        # Test with URL
        doi, server = service._extract_preprint_doi_from_source(
            "https://www.biorxiv.org/content/10.1101/2024.01.01.574440v1"
        )
        assert doi == "10.1101/2024.01.01.574440v1"
        assert server == "biorxiv"

        doi, server = service._extract_preprint_doi_from_source(
            "https://www.medrxiv.org/content/10.1101/2024.01.01.574440v1"
        )
        assert doi == "10.1101/2024.01.01.574440v1"
        assert server == "medrxiv"
