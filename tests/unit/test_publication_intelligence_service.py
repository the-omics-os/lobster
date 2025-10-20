"""
Unit tests for PublicationIntelligenceService.

Tests PDF extraction, URL content extraction, and supplementary material download.
"""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import PyPDF2

from lobster.tools.publication_intelligence_service import PublicationIntelligenceService


class TestPublicationIntelligenceService:
    """Test suite for PublicationIntelligenceService."""

    @pytest.fixture
    def service(self, tmp_path):
        """Create service instance with temporary cache directory."""
        service = PublicationIntelligenceService()
        # Override cache directory to use temp path
        service.cache_dir = tmp_path / "literature_cache"
        service.cache_dir.mkdir(parents=True, exist_ok=True)
        return service

    @pytest.fixture
    def mock_pdf_content(self):
        """Create a mock PDF with extractable text."""
        # Create a simple PDF in memory
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        try:
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.drawString(100, 750, "Test PDF Content")
            c.drawString(100, 735, "This is a test paper about bioinformatics.")
            c.save()
            buffer.seek(0)
            return buffer.getvalue()
        except ImportError:
            # Fallback if reportlab not available - create minimal PDF
            minimal_pdf = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj
4 0 obj << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> endobj
5 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test PDF Content) Tj ET
endstream endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
0000000304 00000 n
trailer << /Size 6 /Root 1 0 R >>
startxref
398
%%EOF"""
            return minimal_pdf

    @patch('requests.get')
    def test_extract_direct_pdf_url(self, mock_get, service, mock_pdf_content):
        """Test extracting from direct PDF URL."""
        # Mock PDF response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.content = mock_pdf_content
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Test extraction (no cache)
        result = service.extract_pdf_content("https://example.com/paper.pdf", use_cache=False)

        # Verify
        assert isinstance(result, str)
        assert len(result) > 0
        mock_get.assert_called_once()
        assert "paper.pdf" in mock_get.call_args[0][0]

    @patch('requests.get')
    def test_extract_webpage_with_pdf_link(self, mock_get, service, mock_pdf_content):
        """Test extracting from webpage that contains PDF link."""
        # First call: webpage with PDF link
        mock_webpage = Mock()
        mock_webpage.status_code = 200
        mock_webpage.text = '<html><a href="/download/paper.pdf">Download PDF</a></html>'
        mock_webpage.raise_for_status = Mock()

        # Second call: actual PDF
        mock_pdf = Mock()
        mock_pdf.status_code = 200
        mock_pdf.headers = {"Content-Type": "application/pdf"}
        mock_pdf.content = mock_pdf_content
        mock_pdf.raise_for_status = Mock()

        mock_get.side_effect = [mock_webpage, mock_pdf]

        # Test extraction
        result = service.extract_pdf_content("https://example.com/paper", use_cache=False)

        # Verify
        assert isinstance(result, str)
        assert len(result) > 0
        assert mock_get.call_count == 2

    @patch('requests.get')
    def test_extract_pdf_with_caching(self, mock_get, service, mock_pdf_content):
        """Test PDF caching functionality."""
        # Mock PDF response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.content = mock_pdf_content
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        url = "https://example.com/paper.pdf"

        # First call - should download
        result1 = service.extract_pdf_content(url, use_cache=True)
        assert mock_get.call_count == 1

        # Second call - should use cache
        result2 = service.extract_pdf_content(url, use_cache=True)
        assert mock_get.call_count == 1  # Should not call again
        assert result1 == result2  # Results should be identical

    @patch('requests.get')
    def test_extract_invalid_url(self, mock_get, service):
        """Test error handling for invalid URLs."""
        mock_get.side_effect = Exception("Connection failed")

        with pytest.raises(ValueError, match="Error downloading PDF"):
            service.extract_pdf_content("https://invalid-url.com/paper.pdf", use_cache=False)

    @patch('requests.get')
    def test_extract_non_pdf_content(self, mock_get, service):
        """Test error handling for non-PDF content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.content = b"<html>Not a PDF</html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="did not return a valid PDF"):
            service.extract_pdf_content("https://example.com/notpdf", use_cache=False)

    @patch('requests.get')
    def test_extract_url_content_html(self, mock_get, service):
        """Test extracting content from HTML webpage."""
        html_content = """
        <html>
        <body>
            <nav>Navigation</nav>
            <main>
                <h1>Article Title</h1>
                <p>This is the main content of the article.</p>
                <p>Another paragraph with important information.</p>
            </main>
            <footer>Footer content</footer>
        </body>
        </html>
        """

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = html_content
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = service.extract_url_content("https://example.com/article")

        # Verify content extracted
        assert "Article Title" in result
        assert "main content" in result
        # Verify unwanted elements removed
        assert "Navigation" not in result
        assert "Footer" not in result

    @patch('requests.get')
    def test_extract_url_content_plain_text(self, mock_get, service):
        """Test extracting plain text content."""
        plain_text = "This is plain text content from a URL."

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.text = plain_text
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = service.extract_url_content("https://example.com/text")

        assert result == plain_text

    @patch('requests.get')
    def test_extract_url_content_error(self, mock_get, service):
        """Test error handling for URL content extraction."""
        mock_get.side_effect = Exception("Connection failed")

        with pytest.raises(ValueError, match="Error extracting URL content"):
            service.extract_url_content("https://invalid-url.com")

    @patch('requests.get')
    def test_fetch_supplementary_info_success(self, mock_get, service, tmp_path):
        """Test successful supplementary material download."""
        # Mock DOI resolution
        mock_doi_response = Mock()
        mock_doi_response.status_code = 200
        mock_doi_response.url = "https://publisher.com/article/12345"

        # Mock publisher page with supplementary links
        html_with_links = """
        <html>
        <body>
            <a href="/supplementary/file1.pdf">Supplementary Material 1</a>
            <a href="/supplementary/file2.xlsx">Supplementary Data</a>
        </body>
        </html>
        """
        mock_page_response = Mock()
        mock_page_response.status_code = 200
        mock_page_response.content = html_with_links.encode()

        # Mock file downloads
        mock_file_response = Mock()
        mock_file_response.status_code = 200
        mock_file_response.content = b"File content"

        mock_get.side_effect = [
            mock_doi_response,  # DOI resolution
            mock_page_response,  # Publisher page
            mock_file_response,  # File 1
            mock_file_response,  # File 2
        ]

        output_dir = str(tmp_path / "supplements")
        result = service.fetch_supplementary_info_from_doi("10.1234/test", output_dir)

        # Verify
        assert "Successfully downloaded" in result
        assert "2 file(s)" in result

    @patch('requests.get')
    def test_fetch_supplementary_info_no_materials(self, mock_get, service):
        """Test DOI download when no supplementary materials found."""
        # Mock DOI resolution
        mock_doi_response = Mock()
        mock_doi_response.status_code = 200
        mock_doi_response.url = "https://publisher.com/article/12345"

        # Mock publisher page without supplementary links
        html_without_links = "<html><body><p>Article content</p></body></html>"
        mock_page_response = Mock()
        mock_page_response.status_code = 200
        mock_page_response.content = html_without_links.encode()

        mock_get.side_effect = [mock_doi_response, mock_page_response]

        result = service.fetch_supplementary_info_from_doi("10.1234/test")

        assert "No supplementary materials found" in result

    @patch('requests.get')
    def test_fetch_supplementary_info_doi_resolution_failed(self, mock_get, service):
        """Test DOI download when DOI resolution fails."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = service.fetch_supplementary_info_from_doi("10.1234/invalid")

        assert "Failed to resolve DOI" in result

    @patch('requests.get')
    @patch('lobster.tools.publication_intelligence_service.PublicationIntelligenceService.extract_pdf_content')
    def test_extract_methods_from_paper_with_llm(self, mock_extract, mock_get, service):
        """Test method extraction with LLM."""
        # Mock PDF extraction
        mock_extract.return_value = "This paper uses Scanpy for single-cell analysis with filtering criteria of min_genes=200."

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """```json
{
  "software_used": ["Scanpy"],
  "parameters": {"min_genes": "200"},
  "statistical_methods": ["Wilcoxon test"],
  "data_sources": ["GEO"],
  "sample_sizes": {"cells": "10000"},
  "normalization_methods": ["log1p"],
  "quality_control": ["min_genes filter"]
}
```"""
        mock_llm.invoke.return_value = mock_response

        result = service.extract_methods_from_paper(
            "https://example.com/paper.pdf",
            llm=mock_llm
        )

        # Verify
        assert isinstance(result, dict)
        assert "software_used" in result
        assert "Scanpy" in result["software_used"]
        assert mock_extract.called

    @patch('lobster.tools.publication_intelligence_service.PublicationIntelligenceService.extract_pdf_content')
    def test_extract_methods_from_paper_pmid_error(self, mock_extract, service):
        """Test that PMID input raises appropriate error."""
        with pytest.raises(ValueError, match="Please provide a direct PDF URL"):
            service.extract_methods_from_paper("PMID:12345678")

    def test_service_initialization(self, service):
        """Test service initialization and cache directory creation."""
        assert service.cache_dir.exists()
        assert service.cache_dir.is_dir()

    def test_service_with_data_manager(self, tmp_path):
        """Test service with DataManager for provenance tracking."""
        mock_data_manager = Mock()
        service = PublicationIntelligenceService(data_manager=mock_data_manager)

        assert service.data_manager == mock_data_manager
        assert service.cache_dir.exists()


class TestPDFExtractionEdgeCases:
    """Test edge cases for PDF extraction."""

    @pytest.fixture
    def service(self, tmp_path):
        """Create service instance."""
        service = PublicationIntelligenceService()
        service.cache_dir = tmp_path / "cache"
        service.cache_dir.mkdir(parents=True, exist_ok=True)
        return service

    @patch('requests.get')
    def test_pdf_with_no_text(self, mock_get, service):
        """Test PDF that contains no extractable text (image-based)."""
        # Create minimal PDF with no text
        minimal_pdf = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
xref
0 4
trailer << /Size 4 /Root 1 0 R >>
%%EOF"""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.content = minimal_pdf
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="did not contain any extractable text"):
            service.extract_pdf_content("https://example.com/image-pdf.pdf", use_cache=False)

    @patch('requests.get')
    def test_pdf_url_with_magic_bytes_check(self, mock_get, service):
        """Test PDF validation using magic bytes when content-type is wrong."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/octet-stream"}
        mock_response.content = b"%PDF-1.4\n...pdf content..."
        mock_response.raise_for_status = Mock()

        # This should not raise error because magic bytes indicate PDF
        # Note: Will still fail on actual extraction, but passes validation step
        mock_get.return_value = mock_response

        try:
            service.extract_pdf_content("https://example.com/file", use_cache=False)
        except ValueError as e:
            # Should fail on extraction, not validation
            assert "extracting text" in str(e) or "extractable text" in str(e)


class TestURLExtractionEdgeCases:
    """Test edge cases for URL content extraction."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return PublicationIntelligenceService()

    @patch('requests.get')
    def test_url_with_no_main_content(self, mock_get, service):
        """Test webpage with no main/article/body tags."""
        html_content = "<html><div>Some content</div></html>"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = html_content
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="No content found"):
            service.extract_url_content("https://example.com/page")

    @patch('requests.get')
    def test_url_with_json_content(self, mock_get, service):
        """Test URL returning JSON content."""
        json_content = '{"title": "Test", "content": "JSON data"}'

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.text = json_content
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = service.extract_url_content("https://api.example.com/data")
        assert result == json_content
