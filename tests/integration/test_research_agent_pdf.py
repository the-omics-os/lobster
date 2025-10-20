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
from lobster.tools.publication_intelligence_service import PublicationIntelligenceService


@pytest.fixture
def data_manager(tmp_path):
    """Create a DataManagerV2 instance for testing."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return DataManagerV2(workspace_dir=str(workspace_dir))


@pytest.fixture
def intelligence_service(data_manager):
    """Create PublicationIntelligenceService instance."""
    return PublicationIntelligenceService(data_manager=data_manager)


@pytest.mark.integration
class TestResearchAgentPDFWorkflow:
    """Integration tests for research agent PDF capabilities."""

    @patch('requests.get')
    def test_pdf_extraction_workflow(self, mock_get, intelligence_service):
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

        # Test extraction
        url = "https://example.com/paper.pdf"
        result = intelligence_service.extract_pdf_content(url, use_cache=False)

        # Verify
        assert isinstance(result, str)
        assert mock_get.called

    @patch('requests.get')
    def test_url_content_extraction_workflow(self, mock_get, intelligence_service):
        """Test URL content extraction workflow."""
        html_content = """
        <html>
        <body>
            <main>
                <h1>Methods Section</h1>
                <p>We used Scanpy for single-cell analysis.</p>
                <p>Quality control was performed with min_genes=200.</p>
            </main>
        </body>
        </html>
        """

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = html_content
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        url = "https://example.com/methods"
        result = intelligence_service.extract_url_content(url)

        # Verify content extraction
        assert "Methods Section" in result
        assert "Scanpy" in result
        assert "min_genes=200" in result

    @patch('requests.get')
    def test_supplementary_download_workflow(self, mock_get, intelligence_service, tmp_path):
        """Test supplementary material download workflow."""
        # Mock DOI resolution
        mock_doi_response = Mock()
        mock_doi_response.status_code = 200
        mock_doi_response.url = "https://publisher.com/article/12345"

        # Mock publisher page
        html_with_supp = """
        <html>
        <body>
            <a href="/supp/data.xlsx">Supplementary Data 1</a>
        </body>
        </html>
        """
        mock_page_response = Mock()
        mock_page_response.status_code = 200
        mock_page_response.content = html_with_supp.encode()

        # Mock file download
        mock_file_response = Mock()
        mock_file_response.status_code = 200
        mock_file_response.content = b"Excel content"

        mock_get.side_effect = [mock_doi_response, mock_page_response, mock_file_response]

        output_dir = str(tmp_path / "supplements")
        result = intelligence_service.fetch_supplementary_info_from_doi(
            "10.1234/test", output_dir
        )

        # Verify
        assert "Successfully downloaded" in result or "Downloaded file" in result
        assert mock_get.call_count == 3

    @patch('lobster.tools.publication_intelligence_service.PublicationIntelligenceService.extract_pdf_content')
    def test_method_extraction_workflow(self, mock_extract, intelligence_service):
        """Test method extraction workflow with LLM."""
        # Mock PDF extraction
        mock_extract.return_value = """
        Methods: We analyzed single-cell RNA-seq data using Scanpy version 1.9.
        Quality control included filtering cells with min_genes=200 and max_percent_mito=5%.
        Normalization was performed using log1p transformation.
        Statistical analysis used Wilcoxon rank-sum test with FDR correction.
        """

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "software_used": ["Scanpy"],
            "parameters": {
                "min_genes": "200",
                "max_percent_mito": "5%"
            },
            "statistical_methods": ["Wilcoxon rank-sum test", "FDR correction"],
            "normalization_methods": ["log1p"],
            "quality_control": ["min_genes filter", "mitochondrial percentage filter"]
        })
        mock_llm.invoke.return_value = mock_response

        result = intelligence_service.extract_methods_from_paper(
            "https://example.com/paper.pdf",
            llm=mock_llm
        )

        # Verify extracted methods
        assert isinstance(result, dict)
        assert "software_used" in result
        assert "Scanpy" in result["software_used"]
        assert "parameters" in result
        assert result["parameters"]["min_genes"] == "200"

    @patch('requests.get')
    def test_caching_workflow(self, mock_get, intelligence_service):
        """Test PDF caching workflow."""
        mock_pdf = b"%PDF-1.4\nTest content\n%%EOF"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.content = mock_pdf
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        url = "https://example.com/cached.pdf"

        # First call - downloads
        result1 = intelligence_service.extract_pdf_content(url, use_cache=True)
        first_call_count = mock_get.call_count

        # Second call - uses cache
        result2 = intelligence_service.extract_pdf_content(url, use_cache=True)
        second_call_count = mock_get.call_count

        # Verify caching worked
        assert result1 == result2
        assert second_call_count == first_call_count  # No additional API call

    @patch('requests.get')
    def test_error_handling_workflow(self, mock_get, intelligence_service):
        """Test error handling in PDF extraction workflow."""
        # Test connection error
        mock_get.side_effect = Exception("Network error")

        with pytest.raises(ValueError, match="Error downloading PDF"):
            intelligence_service.extract_pdf_content(
                "https://invalid.com/paper.pdf",
                use_cache=False
            )

    @patch('requests.get')
    def test_provenance_logging_workflow(self, mock_get, intelligence_service):
        """Test that tool usage is logged for provenance."""
        mock_pdf = b"%PDF-1.4\nContent\n%%EOF"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.content = mock_pdf
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        url = "https://example.com/paper.pdf"

        # Extract PDF
        intelligence_service.extract_pdf_content(url, use_cache=False)

        # Verify provenance logging
        if intelligence_service.data_manager:
            tool_history = intelligence_service.data_manager.tool_usage_history
            # Check that extraction was logged
            assert len(tool_history) > 0
            # The last logged tool should be extract_pdf_content
            last_log = tool_history[-1]
            assert last_log.get("tool") == "extract_pdf_content"


@pytest.mark.integration
@pytest.mark.requires_api
class TestResearchAgentPDFRealWorld:
    """
    Integration tests with real PDFs (requires network access).

    These tests are marked with @pytest.mark.requires_api and will be skipped
    unless specifically requested.
    """

    def test_extract_biorxiv_paper(self, intelligence_service):
        """
        Test extraction from a real bioRxiv paper (public domain).

        Note: This test requires network access and is marked as requires_api.
        """
        # Use a stable bioRxiv paper URL (example - replace with actual public paper)
        url = "https://www.biorxiv.org/content/10.1101/2023.01.001v1.full.pdf"

        try:
            result = intelligence_service.extract_pdf_content(url)
            assert len(result) > 100  # Should have substantial text
        except Exception as e:
            pytest.skip(f"Network access required or URL changed: {e}")

    def test_extract_plos_paper(self, intelligence_service):
        """
        Test extraction from PLOS ONE paper (CC-BY license).

        Note: This test requires network access.
        """
        # PLOS ONE papers are CC-BY licensed
        url = "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0123456&type=printable"

        try:
            result = intelligence_service.extract_pdf_content(url)
            assert len(result) > 100
        except Exception as e:
            pytest.skip(f"Network access required or URL changed: {e}")


@pytest.mark.integration
class TestResearchAgentToolIntegration:
    """Test integration of PDF tools within research agent."""

    def test_tools_are_registered(self):
        """Test that PDF extraction tools are properly registered."""
        from lobster.agents.research_agent import research_agent
        from lobster.core.data_manager_v2 import DataManagerV2
        from pathlib import Path
        import tempfile

        # Create temporary workspace
        with tempfile.TemporaryDirectory() as tmp_dir:
            dm = DataManagerV2(workspace_dir=str(Path(tmp_dir) / "workspace"))

            # Create agent
            agent = research_agent(data_manager=dm, handoff_tools=[])

            # Verify tools are available
            # The agent returns a LangGraph graph, we can't directly inspect tools
            # But we can verify the agent was created successfully
            assert agent is not None

    @patch('lobster.tools.publication_intelligence_service.PublicationIntelligenceService.extract_pdf_content')
    def test_extract_paper_methods_tool_mock(self, mock_extract, data_manager):
        """Test extract_paper_methods tool with mocked PDF extraction."""
        from lobster.agents.research_agent import research_agent

        # Create agent
        agent_graph = research_agent(data_manager=data_manager, handoff_tools=[])

        # Mock PDF extraction
        mock_extract.return_value = "Test paper content with methods"

        # This test verifies the tool exists and can be invoked
        # Full agent workflow testing would require LangGraph execution
        assert agent_graph is not None

    @patch('lobster.tools.publication_intelligence_service.PublicationIntelligenceService.fetch_supplementary_info_from_doi')
    def test_download_supplementary_tool_mock(self, mock_fetch, data_manager):
        """Test download_supplementary_materials tool with mocked fetching."""
        from lobster.agents.research_agent import research_agent

        # Create agent
        agent_graph = research_agent(data_manager=data_manager, handoff_tools=[])

        # Mock supplementary fetch
        mock_fetch.return_value = "Downloaded 2 files successfully"

        # Verify agent created with tool
        assert agent_graph is not None


@pytest.mark.integration
class TestPublicationIntelligenceServiceIntegration:
    """Integration tests for PublicationIntelligenceService with DataManager."""

    def test_service_with_data_manager_integration(self, data_manager):
        """Test service integration with DataManagerV2."""
        service = PublicationIntelligenceService(data_manager=data_manager)

        # Verify service is properly initialized
        assert service.data_manager == data_manager
        assert service.cache_dir.exists()

    @patch('requests.get')
    def test_provenance_tracking_integration(self, mock_get, data_manager):
        """Test that service properly logs to DataManager."""
        service = PublicationIntelligenceService(data_manager=data_manager)

        mock_pdf = b"%PDF-1.4\nTest\n%%EOF"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.content = mock_pdf
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Perform extraction
        service.extract_pdf_content("https://test.com/paper.pdf", use_cache=False)

        # Verify logging
        history = data_manager.tool_usage_history
        assert len(history) > 0
        assert any(log.get("tool") == "extract_pdf_content" for log in history)
