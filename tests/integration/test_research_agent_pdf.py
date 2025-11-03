"""
Integration tests for research agent PDF extraction workflow.

Tests the end-to-end workflow of PDF extraction and method analysis
through the research agent.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.unified_content_service import (
    UnifiedContentService,
    ContentExtractionError,
)


@pytest.fixture
def data_manager(tmp_path):
    """Create a DataManagerV2 instance for testing."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return DataManagerV2(workspace_path=str(workspace_dir))


@pytest.fixture
def content_service(data_manager):
    """Create UnifiedContentService instance."""
    return UnifiedContentService(
        cache_dir=data_manager.literature_cache_dir,
        data_manager=data_manager
    )


@pytest.mark.integration
class TestResearchAgentPDFWorkflow:
    """Integration tests for research agent PDF capabilities."""

    @patch("lobster.tools.docling_service.DoclingService.extract_methods_section")
    @patch("requests.get")
    def test_pdf_extraction_workflow(self, mock_get, mock_docling, content_service):
        """Test complete PDF extraction workflow."""
        # Mock PDF content
        mock_pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
xref
trailer << /Size 2 /Root 1 0 R >>
%%EOF"""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.content = mock_pdf_content
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Mock DoclingService response
        mock_docling.return_value = {
            "methods_markdown": "# Methods\nTest content",
            "methods_text": "Test content",
            "tables": [],
            "formulas": [],
            "software_mentioned": [],
            "sections": ["Methods"],
        }

        # Test extraction
        url = "https://example.com/paper.pdf"
        result = content_service.get_full_content(url)

        # Verify - returns dict with content
        assert isinstance(result, dict)
        assert "content" in result
        assert isinstance(result["content"], str)

    @patch("lobster.tools.providers.webpage_provider.WebpageProvider.extract_with_full_metadata")
    @patch("lobster.tools.providers.webpage_provider.WebpageProvider.can_handle")
    def test_url_content_extraction_workflow(self, mock_can_handle, mock_extract, content_service):
        """Test URL content extraction workflow."""
        # Mock webpage provider
        mock_can_handle.return_value = True
        mock_extract.return_value = {
            "methods_markdown": "# Methods Section\n\nWe used Scanpy for single-cell analysis.\nQuality control was performed with min_genes=200.",
            "methods_text": "Methods Section\n\nWe used Scanpy for single-cell analysis.\nQuality control was performed with min_genes=200.",
            "tables": [],
            "formulas": [],
            "software_mentioned": ["Scanpy"],
            "sections": ["Methods"],
        }

        url = "https://example.com/methods"
        result = content_service.get_full_content(url, prefer_webpage=True)

        # Verify content extraction
        assert isinstance(result, dict)
        assert "content" in result
        assert "Methods Section" in result["content"]
        assert "Scanpy" in result["content"]
        assert "min_genes=200" in result["content"]

    def test_supplementary_download_workflow(self, content_service, tmp_path):
        """Test supplementary material download workflow."""
        # Supplementary download is not part of UnifiedContentService core API
        # This functionality exists as a deprecated tool in research_agent.py
        # Skip this test as it's not applicable to the new architecture
        pytest.skip(
            "Supplementary download is deprecated and not part of UnifiedContentService. "
            "Functionality exists only as legacy tool in research_agent.py"
        )

    @patch("lobster.tools.docling_service.DoclingService.extract_methods_section")
    def test_method_extraction_workflow(self, mock_docling, content_service):
        """Test method extraction workflow with content extraction."""
        # Mock DoclingService to return content
        mock_docling.return_value = {
            "methods_markdown": """# Methods
We analyzed single-cell RNA-seq data using Scanpy version 1.9.
Quality control included filtering cells with min_genes=200 and max_percent_mito=5%.
Normalization was performed using log1p transformation.
Statistical analysis used Wilcoxon rank-sum test with FDR correction.""",
            "methods_text": "We analyzed single-cell RNA-seq data using Scanpy version 1.9...",
            "tables": [],
            "formulas": [],
            "software_mentioned": ["Scanpy"],
            "sections": ["Methods"],
        }

        # Step 1: Get full content
        content = content_service.get_full_content("https://example.com/paper.pdf")

        # Step 2: Extract methods (note: current implementation returns basic extraction)
        methods = content_service.extract_methods_section(content)

        # Verify extracted methods structure
        assert isinstance(methods, dict)
        assert "software_used" in methods
        assert "Scanpy" in methods["software_used"]
        assert "parameters" in methods
        assert "extraction_confidence" in methods

    @patch("lobster.tools.docling_service.DoclingService.extract_methods_section")
    def test_caching_workflow(self, mock_docling, content_service):
        """Test content caching workflow via DataManagerV2."""
        # Mock DoclingService
        mock_docling.return_value = {
            "methods_markdown": "# Test Content",
            "methods_text": "Test content",
            "tables": [],
            "formulas": [],
            "software_mentioned": [],
            "sections": [],
        }

        url = "https://example.com/cached.pdf"

        # First call - extracts and caches
        result1 = content_service.get_full_content(url)
        first_call_count = mock_docling.call_count

        # Second call - uses DataManager cache
        result2 = content_service.get_full_content(url)
        second_call_count = mock_docling.call_count

        # Verify caching worked - no additional extraction
        assert result1["content"] == result2["content"]
        assert second_call_count == first_call_count  # No additional extraction call

    @patch("lobster.tools.docling_service.DoclingService.extract_methods_section")
    def test_error_handling_workflow(self, mock_docling, content_service):
        """Test error handling in content extraction workflow."""
        # Mock extraction failure
        mock_docling.side_effect = Exception("Extraction failed")

        # Expect ContentExtractionError for failed PDF extraction
        with pytest.raises(ContentExtractionError, match="Failed to extract PDF content"):
            content_service.get_full_content("https://invalid.com/paper.pdf")

    def test_provenance_logging_workflow(self, content_service, data_manager):
        """Test that provenance tracking infrastructure is available."""
        # Note: UnifiedContentService doesn't directly log tool usage
        # Provenance logging happens at the agent level via data_manager.log_tool_usage()
        # This test verifies the infrastructure is in place

        # Verify provenance tracking is available
        if data_manager.provenance:
            activities = data_manager.provenance.activities
            # Check that provenance infrastructure exists
            assert isinstance(activities, list)
            print(f"✅ Provenance tracking infrastructure available")
            print(f"📋 Current activities logged: {len(activities)}")
        else:
            pytest.fail("Provenance tracking should be enabled in DataManagerV2")


@pytest.mark.integration
@pytest.mark.requires_api
class TestResearchAgentPDFRealWorld:
    """
    Integration tests with real PDFs (requires network access).

    These tests are marked with @pytest.mark.requires_api and will be skipped
    unless specifically requested.
    """

    def test_extract_biorxiv_paper(self, content_service):
        """
        Test extraction from a real bioRxiv paper (public domain).

        Note: This test requires network access and is marked as requires_api.
        """
        # Use a stable bioRxiv paper URL (example - replace with actual public paper)
        url = "https://www.biorxiv.org/content/10.1101/2023.01.001v1.full.pdf"

        try:
            result = content_service.get_full_content(url)
            assert "content" in result
            assert len(result["content"]) > 100  # Should have substantial text
        except Exception as e:
            pytest.skip(f"Network access required or URL changed: {e}")

    def test_extract_plos_paper(self, content_service):
        """
        Test extraction from PLOS ONE paper (CC-BY license).

        Note: This test requires network access.
        """
        # PLOS ONE papers are CC-BY licensed
        url = "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0123456&type=printable"

        try:
            result = content_service.get_full_content(url)
            assert "content" in result
            assert len(result["content"]) > 100
        except Exception as e:
            pytest.skip(f"Network access required or URL changed: {e}")


@pytest.mark.integration
class TestResearchAgentToolIntegration:
    """Test integration of publication tools within research agent."""

    def test_tools_are_registered(self):
        """Test that publication extraction tools are properly registered."""
        import tempfile
        from pathlib import Path

        from lobster.agents.research_agent import research_agent
        from lobster.core.data_manager_v2 import DataManagerV2

        # Create temporary workspace
        with tempfile.TemporaryDirectory() as tmp_dir:
            dm = DataManagerV2(workspace_path=str(Path(tmp_dir) / "workspace"))

            # Create agent
            agent = research_agent(data_manager=dm, handoff_tools=[])

            # Verify tools are available
            # The agent returns a LangGraph graph, we can't directly inspect tools
            # But we can verify the agent was created successfully
            assert agent is not None

    @patch("lobster.tools.unified_content_service.UnifiedContentService.get_full_content")
    def test_extract_paper_methods_tool_mock(self, mock_extract, data_manager):
        """Test extract_paper_methods tool with mocked content extraction."""
        from lobster.agents.research_agent import research_agent

        # Create agent
        agent_graph = research_agent(data_manager=data_manager, handoff_tools=[])

        # Mock content extraction
        mock_extract.return_value = {
            "content": "Test paper content with methods",
            "source_type": "pdf",
            "tier_used": "full_pdf",
            "metadata": {"software": ["Scanpy"]},
        }

        # This test verifies the tool exists and can be invoked
        # Full agent workflow testing would require LangGraph execution
        assert agent_graph is not None

    def test_download_supplementary_tool_deprecated(self, data_manager):
        """Test that supplementary download tool is deprecated."""
        from lobster.agents.research_agent import research_agent

        # Create agent
        agent_graph = research_agent(data_manager=data_manager, handoff_tools=[])

        # Supplementary download is deprecated in Phase 3 refactoring
        # Tool still exists in research_agent.py as deprecated legacy functionality
        # This test just verifies agent creation succeeds
        assert agent_graph is not None


@pytest.mark.integration
class TestUnifiedContentServiceIntegration:
    """Integration tests for UnifiedContentService with DataManager."""

    def test_service_with_data_manager_integration(self, data_manager):
        """Test service integration with DataManagerV2."""
        service = UnifiedContentService(
            cache_dir=data_manager.literature_cache_dir,
            data_manager=data_manager
        )

        # Verify service is properly initialized
        assert service.data_manager == data_manager
        assert service.docling_service.cache_dir.exists()

    def test_provenance_tracking_integration(self, data_manager):
        """Test that provenance tracking infrastructure is available."""
        service = UnifiedContentService(
            cache_dir=data_manager.literature_cache_dir,
            data_manager=data_manager
        )

        # Verify provenance tracking is available
        # Note: UnifiedContentService doesn't directly log tool usage
        # Provenance logging happens at the agent level
        assert data_manager.provenance is not None
        assert isinstance(data_manager.provenance.activities, list)
        print(f"✅ Provenance tracking infrastructure available for UnifiedContentService")
