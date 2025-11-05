"""
Unit tests for UnifiedContentService.

These tests verify the two-tier access strategy for publication content:
- Tier 1 (Fast): Quick abstract retrieval via NCBI (<500ms)
- Tier 2 (Full): Comprehensive content extraction (webpage → PDF fallback)

Tests cover:
- get_quick_abstract() (Tier 1 access)
- get_full_content() with webpage-first strategy
- extract_methods_section() (LLM-based extraction)
- get_cached_publication() (delegation to DataManager)
- Error handling (PaywalledError, ContentExtractionError)
- Cache integration with DataManagerV2
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from lobster.tools.unified_content_service import (
    ContentExtractionError,
    PaywalledError,
    UnifiedContentService,
)


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManagerV2 instance for testing."""
    mock_dm = Mock()
    mock_dm.cache_dir = Path("/tmp/test_cache")
    mock_dm.literature_cache_dir = Path("/tmp/test_cache/literature")
    mock_dm.get_cached_publication.return_value = None
    mock_dm.cache_publication_content.return_value = None
    mock_dm.tool_usage_history = []
    return mock_dm


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory for testing."""
    cache_dir = tmp_path / "literature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def unified_service(mock_data_manager, temp_cache_dir):
    """Create UnifiedContentService instance for testing."""
    return UnifiedContentService(
        cache_dir=temp_cache_dir,
        data_manager=mock_data_manager,
    )


@pytest.fixture
def sample_abstract_metadata():
    """Sample abstract metadata for testing."""
    from lobster.tools.providers.base_provider import PublicationMetadata

    return PublicationMetadata(
        uid="12345678",
        title="Sample Publication Title",
        authors=["John Doe", "Jane Smith", "Bob Johnson"],
        abstract="This is a sample abstract for testing purposes. It contains information about single-cell RNA sequencing analysis using scanpy with parameter min_genes=200 and resolution=0.5.",
        journal="Nature Methods",
        published="2024-01-15",
        pmid="12345678",
        doi="10.1038/s41592-024-12345-6",
        keywords=["single-cell", "RNA-seq", "analysis"],
    )


@pytest.fixture
def sample_full_content():
    """Sample full content result for testing."""
    return {
        "methods_markdown": "# Methods\n\n## Data Processing\n\nWe used scanpy version 1.9.3 for single-cell analysis...",
        "methods_text": "Data Processing: We used scanpy version 1.9.3 for single-cell analysis...",
        "tables": [
            {"caption": "Table 1: QC parameters", "data": []},
            {"caption": "Table 2: DE results", "data": []},
        ],
        "formulas": ["min_genes=200", "resolution=0.5"],
        "software_mentioned": ["scanpy", "seurat", "cellranger"],
        "sections": ["Methods", "Results", "Discussion"],
    }


# ==============================================================================
# Test Tier 1: Fast Abstract Access
# ==============================================================================


class TestQuickAbstract:
    """Test get_quick_abstract() method (Tier 1 access)."""

    @patch("lobster.tools.unified_content_service.AbstractProvider")
    def test_get_quick_abstract_success(
        self, mock_provider_class, unified_service, sample_abstract_metadata
    ):
        """Test successful quick abstract retrieval."""
        # Mock AbstractProvider
        mock_provider = Mock()
        mock_provider.get_abstract.return_value = sample_abstract_metadata
        mock_provider_class.return_value = mock_provider

        # Create new service with mocked provider
        service = UnifiedContentService(
            cache_dir=Path("/tmp/test"),
            data_manager=unified_service.data_manager,
        )
        service.abstract_provider = mock_provider

        # Test retrieval
        result = service.get_quick_abstract("PMID:12345678")

        # Verify result structure
        assert result["title"] == "Sample Publication Title"
        assert result["abstract"].startswith("This is a sample abstract")
        assert result["tier_used"] == "abstract"
        assert result["source"] == "pubmed"
        assert result["pmid"] == "12345678"
        assert result["doi"] == "10.1038/s41592-024-12345-6"
        assert "extraction_time" in result
        assert len(result["authors"]) == 3
        assert len(result["keywords"]) == 3

    @patch("lobster.tools.unified_content_service.AbstractProvider")
    def test_get_quick_abstract_with_doi(
        self, mock_provider_class, unified_service, sample_abstract_metadata
    ):
        """Test quick abstract retrieval with DOI identifier."""
        mock_provider = Mock()
        mock_provider.get_abstract.return_value = sample_abstract_metadata
        mock_provider_class.return_value = mock_provider

        service = UnifiedContentService(
            cache_dir=Path("/tmp/test"),
            data_manager=unified_service.data_manager,
        )
        service.abstract_provider = mock_provider

        result = service.get_quick_abstract("10.1038/s41592-024-12345-6")

        assert result["title"] == "Sample Publication Title"
        assert mock_provider.get_abstract.called

    @patch("lobster.tools.unified_content_service.AbstractProvider")
    def test_get_quick_abstract_error_handling(
        self, mock_provider_class, unified_service
    ):
        """Test error handling in quick abstract retrieval."""
        mock_provider = Mock()
        mock_provider.get_abstract.side_effect = Exception("API error")
        mock_provider_class.return_value = mock_provider

        service = UnifiedContentService(
            cache_dir=Path("/tmp/test"),
            data_manager=unified_service.data_manager,
        )
        service.abstract_provider = mock_provider

        with pytest.raises(ContentExtractionError) as exc_info:
            service.get_quick_abstract("PMID:invalid")

        assert "Failed to retrieve abstract" in str(exc_info.value)


# ==============================================================================
# Test Tier 2: Full Content Extraction
# ==============================================================================


class TestFullContent:
    """Test get_full_content() method (Tier 2 access)."""

    def test_get_full_content_cached(self, unified_service, sample_full_content):
        """Test full content retrieval from DataManager cache."""
        # Mock cached publication
        cached_data = {
            "identifier": "PMID:12345678",
            "methods_markdown": sample_full_content["methods_markdown"],
            "methods_text": sample_full_content["methods_text"],
            "tables": sample_full_content["tables"],
            "software_mentioned": sample_full_content["software_mentioned"],
        }
        unified_service.data_manager.get_cached_publication.return_value = cached_data

        result = unified_service.get_full_content("PMID:12345678")

        # Verify cache hit
        assert result["tier_used"] == "full_cached"
        assert result["methods_markdown"] == sample_full_content["methods_markdown"]
        assert "extraction_time" in result
        unified_service.data_manager.get_cached_publication.assert_called_once_with(
            "PMID:12345678"
        )

    @patch("lobster.tools.unified_content_service.WebpageProvider")
    def test_get_full_content_webpage_first(
        self, mock_webpage_provider_class, unified_service, sample_full_content
    ):
        """Test full content extraction with webpage-first strategy."""
        # Mock webpage provider
        mock_webpage_provider = Mock()
        mock_webpage_provider.can_handle.return_value = True
        mock_webpage_provider.extract_with_full_metadata.return_value = (
            sample_full_content
        )
        mock_webpage_provider_class.return_value = mock_webpage_provider

        service = UnifiedContentService(
            cache_dir=Path("/tmp/test"),
            data_manager=unified_service.data_manager,
        )
        service.webpage_provider = mock_webpage_provider

        url = "https://www.nature.com/articles/s41586-025-09686-5"
        result = service.get_full_content(url, prefer_webpage=True)

        # Verify webpage extraction
        assert result["tier_used"] == "full_webpage"
        assert result["source_type"] == "webpage"
        assert result["content"] == sample_full_content["methods_markdown"]
        assert len(result["metadata"]["software"]) == 3
        assert mock_webpage_provider.extract_with_full_metadata.called

    @patch("lobster.tools.unified_content_service.DoclingService")
    def test_get_full_content_pdf_extraction(
        self, mock_docling_class, unified_service, sample_full_content
    ):
        """Test full content extraction with PDF fallback."""
        # Mock docling service
        mock_docling = Mock()
        mock_docling.extract_methods_section.return_value = sample_full_content
        mock_docling_class.return_value = mock_docling

        service = UnifiedContentService(
            cache_dir=Path("/tmp/test"),
            data_manager=unified_service.data_manager,
        )
        service.docling_service = mock_docling

        pdf_url = "https://biorxiv.org/content/10.1101/2024.01.001.full.pdf"
        result = service.get_full_content(pdf_url)

        # Verify PDF extraction
        assert result["tier_used"] == "full_pdf"
        assert result["source_type"] == "pdf"
        assert result["parser"] == sample_full_content.get("parser", "docling")
        assert mock_docling.extract_methods_section.called

    @patch("lobster.tools.unified_content_service.WebpageProvider")
    @patch("lobster.tools.unified_content_service.DoclingService")
    def test_get_full_content_fallback_to_pdf(
        self,
        mock_docling_class,
        mock_webpage_provider_class,
        unified_service,
        sample_full_content,
    ):
        """Test fallback from webpage to PDF extraction."""
        # Mock webpage provider (fails)
        mock_webpage_provider = Mock()
        mock_webpage_provider.can_handle.return_value = True
        mock_webpage_provider.extract_with_full_metadata.side_effect = Exception(
            "Webpage extraction failed"
        )
        mock_webpage_provider_class.return_value = mock_webpage_provider

        # Mock docling service (succeeds)
        mock_docling = Mock()
        mock_docling.extract_methods_section.return_value = sample_full_content
        mock_docling_class.return_value = mock_docling

        service = UnifiedContentService(
            cache_dir=Path("/tmp/test"),
            data_manager=unified_service.data_manager,
        )
        service.webpage_provider = mock_webpage_provider
        service.docling_service = mock_docling

        url = "https://www.nature.com/articles/s41586-025-09686-5.pdf"
        result = service.get_full_content(url, prefer_webpage=True)

        # Verify fallback to PDF
        assert result["tier_used"] == "full_pdf"
        assert result["source_type"] == "pdf"
        assert mock_webpage_provider.extract_with_full_metadata.called
        assert mock_docling.extract_methods_section.called

    def test_get_full_content_invalid_source(self, unified_service):
        """Test error handling for invalid source."""
        with pytest.raises(ContentExtractionError) as exc_info:
            unified_service.get_full_content("invalid_source_format")

        assert "Cannot extract content from source" in str(exc_info.value)

    @patch("lobster.tools.unified_content_service.DoclingService")
    def test_get_full_content_pdf_extraction_failure(
        self, mock_docling_class, unified_service
    ):
        """Test error handling when PDF extraction fails."""
        mock_docling = Mock()
        mock_docling.extract_methods_section.side_effect = Exception(
            "PDF extraction failed"
        )
        mock_docling_class.return_value = mock_docling

        service = UnifiedContentService(
            cache_dir=Path("/tmp/test"),
            data_manager=unified_service.data_manager,
        )
        service.docling_service = mock_docling

        pdf_url = "https://example.com/paper.pdf"
        with pytest.raises(ContentExtractionError) as exc_info:
            service.get_full_content(pdf_url)

        assert "Failed to extract PDF content" in str(exc_info.value)


# ==============================================================================
# Test Methods Extraction
# ==============================================================================


class TestMethodsExtraction:
    """Test extract_methods_section() method."""

    def test_extract_methods_section_basic(self, unified_service, sample_full_content):
        """Test basic methods section extraction."""
        # Wrap sample_full_content in proper structure
        content_result = {
            "content": sample_full_content["methods_markdown"],
            "methods_text": sample_full_content["methods_text"],
            "metadata": {
                "software": sample_full_content["software_mentioned"],
                "tables": sample_full_content["tables"],
            },
        }

        result = unified_service.extract_methods_section(content_result)

        # Verify extraction result
        assert "software_used" in result
        assert "parameters" in result
        assert "statistical_methods" in result
        assert "extraction_confidence" in result
        assert result["software_used"] == sample_full_content["software_mentioned"]
        assert result["content_source"] == "unknown"

    def test_extract_methods_section_with_content_result(
        self, unified_service, sample_full_content
    ):
        """Test methods extraction from full content result."""
        content_result = {
            "content": sample_full_content["methods_markdown"],
            "methods_text": sample_full_content["methods_text"],
            "source_type": "pdf",
            "metadata": {
                "software": sample_full_content["software_mentioned"],
                "tables": sample_full_content["tables"],
            },
        }

        result = unified_service.extract_methods_section(content_result)

        assert result["software_used"] == sample_full_content["software_mentioned"]
        assert result["content_source"] == "pdf"
        assert result["methods_text"] == sample_full_content["methods_text"]

    def test_extract_methods_section_empty_content(self, unified_service):
        """Test methods extraction with empty content."""
        content_result = {
            "content": "",
            "metadata": {},
        }

        result = unified_service.extract_methods_section(content_result)

        # Verify defaults
        assert result["software_used"] == []
        assert result["parameters"] == {}
        assert result["extraction_confidence"] == 0.3  # Low confidence for no software


# ==============================================================================
# Test Cache Integration
# ==============================================================================


class TestCacheIntegration:
    """Test cache integration with DataManagerV2."""

    def test_get_cached_publication_success(self, unified_service):
        """Test successful cached publication retrieval."""
        cached_data = {
            "identifier": "PMID:12345678",
            "methods_markdown": "# Methods content",
            "timestamp": "2024-01-15T10:30:00",
        }
        unified_service.data_manager.get_cached_publication.return_value = cached_data

        result = unified_service.get_cached_publication("PMID:12345678")

        assert result == cached_data
        unified_service.data_manager.get_cached_publication.assert_called_once_with(
            "PMID:12345678"
        )

    def test_get_cached_publication_not_found(self, unified_service):
        """Test cached publication not found."""
        unified_service.data_manager.get_cached_publication.return_value = None

        result = unified_service.get_cached_publication("PMID:99999999")

        assert result is None

    def test_get_cached_publication_no_data_manager(self):
        """Test cache retrieval without DataManager."""
        service = UnifiedContentService(
            cache_dir=Path("/tmp/test"),
            data_manager=None,
        )

        result = service.get_cached_publication("PMID:12345678")

        assert result is None

    @patch("lobster.tools.unified_content_service.WebpageProvider")
    def test_cache_publication_after_extraction(
        self, mock_webpage_provider_class, unified_service, sample_full_content
    ):
        """Test that extracted content is cached in DataManager."""
        mock_webpage_provider = Mock()
        mock_webpage_provider.can_handle.return_value = True
        mock_webpage_provider.extract_with_full_metadata.return_value = (
            sample_full_content
        )
        mock_webpage_provider_class.return_value = mock_webpage_provider

        service = UnifiedContentService(
            cache_dir=Path("/tmp/test"),
            data_manager=unified_service.data_manager,
        )
        service.webpage_provider = mock_webpage_provider

        url = "https://www.nature.com/articles/test"
        service.get_full_content(url)

        # Verify caching was called
        service.data_manager.cache_publication_content.assert_called_once()
        call_args = service.data_manager.cache_publication_content.call_args
        assert call_args[1]["identifier"] == url
        assert call_args[1]["content"] == sample_full_content
        assert call_args[1]["format"] == "json"


# ==============================================================================
# Test Error Classes
# ==============================================================================


class TestErrorClasses:
    """Test custom error classes."""

    def test_content_extraction_error(self):
        """Test ContentExtractionError."""
        error = ContentExtractionError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_paywalled_error(self):
        """Test PaywalledError."""
        identifier = "PMID:12345678"
        suggestions = "Try PMC or bioRxiv"
        error = PaywalledError(identifier, suggestions)

        assert error.identifier == identifier
        assert error.suggestions == suggestions
        assert identifier in str(error)
        assert suggestions in str(error)
        assert isinstance(error, ContentExtractionError)


# ==============================================================================
# Test Helper Methods
# ==============================================================================


class TestHelperMethods:
    """Test internal helper methods."""

    def test_is_pdf_url_with_pdf_extension(self, unified_service):
        """Test PDF URL detection with .pdf extension."""
        assert unified_service._is_pdf_url("https://example.com/paper.pdf") is True
        assert unified_service._is_pdf_url("https://example.com/PAPER.PDF") is True

    def test_is_pdf_url_with_pdf_path(self, unified_service):
        """Test PDF URL detection with /pdf/ in path."""
        assert unified_service._is_pdf_url("https://example.com/pdf/paper") is True
        assert unified_service._is_pdf_url("https://example.com/PDF/paper") is True

    def test_is_pdf_url_negative(self, unified_service):
        """Test PDF URL detection with non-PDF URLs."""
        assert unified_service._is_pdf_url("https://example.com/article") is False
        assert unified_service._is_pdf_url("https://example.com/paper.html") is False

    def test_is_identifier_pmid(self, unified_service):
        """Test identifier detection for PMIDs."""
        assert unified_service._is_identifier("PMID:12345678") is True
        assert unified_service._is_identifier("pmid:12345678") is True
        assert unified_service._is_identifier("12345678") is True

    def test_is_identifier_doi(self, unified_service):
        """Test identifier detection for DOIs."""
        assert unified_service._is_identifier("10.1038/s41586-021-12345-6") is True
        assert unified_service._is_identifier("10.1126/science.abc1234") is True

    def test_is_identifier_negative(self, unified_service):
        """Test identifier detection for non-identifiers."""
        assert unified_service._is_identifier("https://example.com/article") is False
        assert unified_service._is_identifier("random_text") is False


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests for end-to-end workflows."""

    @patch("lobster.tools.unified_content_service.AbstractProvider")
    @patch("lobster.tools.unified_content_service.DoclingService")
    def test_two_tier_workflow(
        self,
        mock_docling_class,
        mock_provider_class,
        unified_service,
        sample_abstract_metadata,
        sample_full_content,
    ):
        """Test complete two-tier workflow: abstract then full content."""
        # Mock Tier 1 (abstract)
        mock_provider = Mock()
        mock_provider.get_abstract.return_value = sample_abstract_metadata
        mock_provider_class.return_value = mock_provider

        # Mock Tier 2 (full content)
        mock_docling = Mock()
        mock_docling.extract_methods_section.return_value = sample_full_content
        mock_docling_class.return_value = mock_docling

        service = UnifiedContentService(
            cache_dir=Path("/tmp/test"),
            data_manager=unified_service.data_manager,
        )
        service.abstract_provider = mock_provider
        service.docling_service = mock_docling

        # Step 1: Get quick abstract
        abstract = service.get_quick_abstract("PMID:12345678")
        assert abstract["tier_used"] == "abstract"
        assert abstract["title"] == "Sample Publication Title"

        # Step 2: Get full content
        full_content = service.get_full_content("https://example.com/paper.pdf")
        assert full_content["tier_used"] == "full_pdf"
        assert full_content["source_type"] == "pdf"

        # Step 3: Extract methods
        methods = service.extract_methods_section(full_content)
        assert len(methods["software_used"]) == 3

    def test_cache_first_strategy(self, unified_service, sample_full_content):
        """Test that cache is checked before extraction."""
        cached_data = {
            "methods_markdown": sample_full_content["methods_markdown"],
            "tier_used": "cached",
        }
        unified_service.data_manager.get_cached_publication.return_value = cached_data

        result = unified_service.get_full_content("PMID:12345678")

        # Verify cache was used (no extraction providers called)
        assert result["tier_used"] == "full_cached"
        unified_service.data_manager.get_cached_publication.assert_called_once()
