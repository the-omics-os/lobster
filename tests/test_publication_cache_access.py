"""
Integration tests for publication cache access workflow.

Tests the complete workflow:
1. Extract methods from paper (creates cache entry)
2. list_session_publications (supervisor coordination)
3. read_cached_publication (worker execution)
4. Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
from pathlib import Path
import json

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.publication_intelligence_service import PublicationIntelligenceService


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManagerV2 with tool usage history."""
    dm = Mock(spec=DataManagerV2)
    dm.tool_usage_history = []
    dm.data_dir = Mock()
    dm.data_dir.__truediv__ = Mock(return_value=Mock())
    return dm


@pytest.fixture
def intelligence_service(mock_data_manager):
    """Create PublicationIntelligenceService with mock data manager."""
    return PublicationIntelligenceService(data_manager=mock_data_manager)


class TestPublicationCacheWorkflow:
    """Test complete publication cache access workflow."""

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    def test_extract_list_read_workflow(self, mock_read_text, mock_exists, mock_data_manager, intelligence_service):
        """Test complete workflow: extract -> list -> read."""

        # Step 1: Simulate publication extraction
        # Use correct tool name: "extract_methods_from_paper"
        mock_data_manager.tool_usage_history.append({
            "tool_name": "extract_methods_from_paper",
            "timestamp": datetime.now().isoformat(),
            "parameters": {"url_or_pmid": "PMID:12345678", "parser": "docling"},
            "description": "Extract methods from publication",
        })

        # Step 2: List session publications (supervisor tool)
        # Mock file existence for cache status check
        mock_exists.return_value = True

        publications = intelligence_service.list_session_publications(mock_data_manager)

        assert len(publications) == 1
        assert publications[0]["identifier"] == "PMID:12345678"
        assert publications[0]["tool_name"] == "extract_methods_from_paper"
        assert publications[0]["cache_status"] in ["markdown", "json", "both"]

        # Step 3: Read cached publication (worker tool)
        # Mock file reading for cache retrieval
        mock_methods_content = "# Methods\n\nWe used Scanpy for analysis..."
        mock_read_text.return_value = mock_methods_content

        cached_pub = intelligence_service.get_cached_publication("PMID:12345678")

        assert cached_pub is not None
        assert cached_pub["identifier"] == "PMID:12345678"
        assert "methods_markdown" in cached_pub
        assert cached_pub["methods_markdown"] == mock_methods_content
        assert cached_pub["cache_source"] == "markdown"

    def test_list_publications_empty(self, mock_data_manager, intelligence_service):
        """Test listing publications when none have been extracted."""

        # Empty tool usage history
        mock_data_manager.tool_usage_history = []

        publications = intelligence_service.list_session_publications(mock_data_manager)

        assert len(publications) == 0

    def test_read_publication_not_found(self, mock_data_manager, intelligence_service):
        """Test reading publication that doesn't exist in cache."""

        # Empty tool usage history
        mock_data_manager.tool_usage_history = []

        result = intelligence_service.get_cached_publication("PMID:99999999")

        assert result is None

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    def test_multiple_publications_in_session(self, mock_read_text, mock_exists, mock_data_manager, intelligence_service):
        """Test workflow with multiple publications extracted."""

        # Simulate multiple publication extractions
        mock_data_manager.tool_usage_history = [
            {
                "tool_name": "extract_methods_from_paper",
                "timestamp": "2025-01-01T10:00:00",
                "parameters": {"url_or_pmid": "PMID:11111111", "parser": "docling"},
                "description": "Extract methods from first publication",
            },
            {
                "tool_name": "extract_methods_from_paper",
                "timestamp": "2025-01-01T11:00:00",
                "parameters": {"url_or_pmid": "PMID:22222222", "parser": "pypdf"},
                "description": "Extract methods from second publication",
            },
        ]

        # Mock file existence
        mock_exists.return_value = True

        # List publications
        publications = intelligence_service.list_session_publications(mock_data_manager)

        assert len(publications) == 2
        assert publications[0]["identifier"] == "PMID:11111111"
        assert publications[1]["identifier"] == "PMID:22222222"

        # Mock file reading for first publication
        mock_read_text.return_value = "First paper methods"
        pub1 = intelligence_service.get_cached_publication("PMID:11111111")
        assert pub1 is not None
        assert pub1["cache_source"] == "markdown"

        # Mock file reading for second publication
        mock_read_text.return_value = "Second paper methods"
        pub2 = intelligence_service.get_cached_publication("PMID:22222222")
        assert pub2 is not None
        assert pub2["cache_source"] == "markdown"

    @patch('pathlib.Path.exists')
    def test_batch_extraction_workflow(self, mock_exists, mock_data_manager, intelligence_service):
        """Test that batch extraction creates proper tool history entries."""

        # Note: In actual usage, extract_methods_batch creates multiple individual entries
        # Simulate individual extractions from batch
        mock_data_manager.tool_usage_history = [
            {
                "tool_name": "extract_methods_from_paper",
                "timestamp": "2025-01-01T10:00:00",
                "parameters": {"url_or_pmid": "PMID:11111111", "parser": "docling"},
                "description": "Batch extract paper 1",
            },
            {
                "tool_name": "extract_methods_from_paper",
                "timestamp": "2025-01-01T10:01:00",
                "parameters": {"url_or_pmid": "PMID:22222222", "parser": "pypdf"},
                "description": "Batch extract paper 2",
            },
            {
                "tool_name": "extract_methods_from_paper",
                "timestamp": "2025-01-01T10:02:00",
                "parameters": {"url_or_pmid": "PMID:33333333", "parser": "docling"},
                "description": "Batch extract paper 3",
            },
        ]

        # Mock file existence
        mock_exists.return_value = True

        # List publications should show all 3
        publications = intelligence_service.list_session_publications(mock_data_manager)

        assert len(publications) == 3
        identifiers = [pub["identifier"] for pub in publications]
        assert "PMID:11111111" in identifiers
        assert "PMID:22222222" in identifiers
        assert "PMID:33333333" in identifiers

    def test_publication_with_minimal_history(self, mock_data_manager, intelligence_service):
        """Test handling publication with minimal tool history."""

        # Simulate extraction with minimal fields
        mock_data_manager.tool_usage_history.append({
            "tool_name": "extract_methods_from_paper",
            "timestamp": datetime.now().isoformat(),
            "parameters": {"url_or_pmid": "PMID:12345678"},
            "description": "Extract methods",
        })

        # Should still be listable
        publications = intelligence_service.list_session_publications(mock_data_manager)
        assert len(publications) == 1
        assert publications[0]["identifier"] == "PMID:12345678"


class TestSupervisorWorkerCoordination:
    """Test supervisor-worker coordination patterns."""

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    def test_supervisor_lists_worker_reads(self, mock_read_text, mock_exists, mock_data_manager, intelligence_service):
        """Test typical supervisor -> worker coordination flow."""

        # Setup: Publications extracted in session
        mock_data_manager.tool_usage_history = [
            {
                "tool_name": "extract_methods_from_paper",
                "timestamp": "2025-01-01T10:00:00",
                "parameters": {"url_or_pmid": "PMID:12345678", "parser": "docling"},
                "description": "Extract methods",
            }
        ]

        # Mock file existence
        mock_exists.return_value = True

        # Step 1: Supervisor lists publications
        publications = intelligence_service.list_session_publications(mock_data_manager)

        # Supervisor sees summary
        assert len(publications) == 1
        pub_summary = publications[0]
        assert pub_summary["identifier"] == "PMID:12345678"
        assert pub_summary["source"] == "docling"

        # Step 2: Supervisor hands off to worker with identifier
        # Worker reads full cached content
        mock_read_text.return_value = "Full methods section..."
        full_pub = intelligence_service.get_cached_publication("PMID:12345678")

        # Worker has access to complete extraction
        assert full_pub is not None
        assert full_pub["identifier"] == "PMID:12345678"
        assert full_pub["methods_markdown"] == "Full methods section..."
        assert full_pub["cache_source"] == "markdown"

    def test_error_handling_coordination(self, mock_data_manager, intelligence_service):
        """Test error handling when publication not found."""

        # Empty history
        mock_data_manager.tool_usage_history = []

        # Supervisor lists (empty)
        publications = intelligence_service.list_session_publications(mock_data_manager)
        assert len(publications) == 0

        # Worker attempts to read non-existent publication
        result = intelligence_service.get_cached_publication("PMID:99999999")
        assert result is None

        # This should trigger appropriate error message in agent tool
        # (tested in agent tool implementation, not service layer)


class TestCachePersistence:
    """Test cache persistence behavior."""

    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.mkdir")
    def test_persist_extraction_creates_file(self, mock_mkdir, mock_write_text, mock_data_manager):
        """Test that persist_extraction_as_markdown creates proper markdown file."""

        service = PublicationIntelligenceService(data_manager=mock_data_manager)

        identifier = "PMID:12345678"
        extraction_result = {
            "methods_markdown": "# Methods\n\nDetailed methods...",
            "methods_text": "Detailed methods...",
            "software_mentioned": ["Scanpy"],
            "tables": [],
            "formulas": [],
            "provenance": {"parser": "docling", "fallback_used": False},
        }

        # Call persist method with identifier and extraction_result
        result_path = service.persist_extraction_as_markdown(identifier, extraction_result)

        # Verify result is a Path-like object
        assert result_path is not None
        assert str(result_path).endswith("PMID_12345678.md")

        # Verify file operations were attempted
        # mkdir is called multiple times (cache dirs + publications dir)
        assert mock_mkdir.call_count >= 1
        mock_write_text.assert_called_once()


@patch('pathlib.Path.exists')
@patch('pathlib.Path.read_text')
def test_full_integration_workflow(mock_read_text, mock_exists):
    """
    Full end-to-end integration test simulating real usage.

    Workflow:
    1. User asks research agent to extract methods from paper
    2. Extraction occurs and is cached
    3. User asks supervisor "what papers did we analyze?"
    4. Supervisor uses list_session_publications
    5. User asks "show me details from PMID:12345678"
    6. Supervisor hands off to research_agent or data_expert
    7. Worker uses read_cached_publication
    8. Full methods returned to user
    """

    # Create real DataManagerV2 mock
    dm = Mock(spec=DataManagerV2)
    dm.tool_usage_history = []
    dm.data_dir = Mock()
    dm.data_dir.__truediv__ = Mock(return_value=Mock())

    service = PublicationIntelligenceService(data_manager=dm)

    # Step 1 & 2: Extract methods (simulated with correct tool name)
    dm.tool_usage_history.append({
        "tool_name": "extract_methods_from_paper",
        "timestamp": "2025-01-01T10:00:00",
        "parameters": {"url_or_pmid": "PMID:12345678", "parser": "docling"},
        "description": "5000 characters extracted from KRAS paper",
    })

    # Mock file existence for cache status
    mock_exists.return_value = True

    # Step 4: Supervisor lists publications
    publications = service.list_session_publications(dm)

    assert len(publications) == 1
    pub_summary = publications[0]
    assert pub_summary["identifier"] == "PMID:12345678"
    assert pub_summary["tool_name"] == "extract_methods_from_paper"
    assert pub_summary["methods_length"] == 5000
    assert pub_summary["source"] == "docling"

    # Step 7: Worker reads cached publication
    mock_methods_content = "# Computational Methods\n\nWe performed single-cell RNA-seq analysis using Scanpy (v1.9.1)..."
    mock_read_text.return_value = mock_methods_content

    full_pub = service.get_cached_publication("PMID:12345678")

    assert full_pub is not None
    assert full_pub["identifier"] == "PMID:12345678"
    assert "Scanpy" in full_pub["methods_markdown"]
    assert full_pub["cache_source"] == "markdown"

    print("✅ Full integration workflow test passed!")


if __name__ == "__main__":
    # Run the full integration test
    test_full_integration_workflow()
