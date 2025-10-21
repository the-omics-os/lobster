"""
Integration tests for Research Agent PDF resolution workflows.

These tests simulate real user sessions with proper mocking of external dependencies.
Tests are designed to run successfully with pytest and verify end-to-end workflows.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from lobster.agents.research_agent_assistant import ResearchAgentAssistant
from lobster.tools.providers.publication_resolver import PublicationResolutionResult
from lobster.tools.publication_intelligence_service import PublicationIntelligenceService
from lobster.core.data_manager_v2 import DataManagerV2


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManagerV2 for testing."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.log_tool_usage = Mock()
    mock_dm.metadata_store = {}
    return mock_dm


class TestResearchAgentAssistantResolution:
    """Test ResearchAgentAssistant PDF resolution capabilities."""

    def setup_method(self):
        """Setup test fixtures."""
        self.assistant = ResearchAgentAssistant()

    def test_resolve_publication_accessible(self):
        """Test successful resolution of accessible publication."""
        with patch("lobster.agents.research_agent_assistant.PublicationResolver") as mock_resolver_class:
            mock_resolver = Mock()
            mock_resolver.resolve.return_value = PublicationResolutionResult(
                identifier="PMID:12345678",
                pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC7891011/pdf/",
                source="pmc",
                access_type="open_access",
                metadata={"pmc_id": "PMC7891011"},
            )
            mock_resolver_class.return_value = mock_resolver

            result = self.assistant.resolve_publication_to_pdf("PMID:12345678")

            assert result.is_accessible() is True
            assert result.source == "pmc"
            assert "PMC7891011" in result.pdf_url
            mock_resolver.resolve.assert_called_once_with("PMID:12345678")

    def test_resolve_publication_paywalled(self):
        """Test resolution of paywalled publication."""
        with patch("lobster.agents.research_agent_assistant.PublicationResolver") as mock_resolver_class:
            mock_resolver = Mock()
            mock_resolver.resolve.return_value = PublicationResolutionResult(
                identifier="PMID:87654321",
                pdf_url=None,
                source="paywalled",
                access_type="paywalled",
                suggestions="## Alternative Access Options\n1. Try PMC\n2. Try bioRxiv",
                alternative_urls=["https://pmc/search", "https://biorxiv/search"],
            )
            mock_resolver_class.return_value = mock_resolver

            result = self.assistant.resolve_publication_to_pdf("PMID:87654321")

            assert result.is_accessible() is False
            assert result.source == "paywalled"
            assert "Alternative Access Options" in result.suggestions
            assert len(result.alternative_urls) > 0

    def test_resolve_publication_error_handling(self):
        """Test error handling during resolution."""
        with patch("lobster.agents.research_agent_assistant.PublicationResolver") as mock_resolver_class:
            mock_resolver = Mock()
            mock_resolver.resolve.side_effect = Exception("Network timeout")
            mock_resolver_class.return_value = mock_resolver

            result = self.assistant.resolve_publication_to_pdf("PMID:99999999")

            assert result.is_accessible() is False
            assert result.source == "error"
            assert result.access_type == "error"

    def test_batch_resolve_mixed_results(self):
        """Test batch resolution with mixed accessible/paywalled results."""
        with patch("lobster.agents.research_agent_assistant.PublicationResolver") as mock_resolver_class:
            mock_resolver = Mock()
            mock_resolver.batch_resolve.return_value = [
                PublicationResolutionResult(
                    identifier="PMID:111",
                    pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC111/pdf/",
                    source="pmc",
                    access_type="open_access",
                ),
                PublicationResolutionResult(
                    identifier="PMID:222",
                    pdf_url=None,
                    source="paywalled",
                    access_type="paywalled",
                    suggestions="Try alternatives",
                ),
                PublicationResolutionResult(
                    identifier="PMID:333",
                    pdf_url="https://biorxiv.org/content/test.pdf",
                    source="biorxiv",
                    access_type="preprint",
                ),
            ]
            mock_resolver_class.return_value = mock_resolver

            identifiers = ["PMID:111", "PMID:222", "PMID:333"]
            results = self.assistant.batch_resolve_publications(identifiers, max_batch=5)

            assert len(results) == 3
            assert results[0].is_accessible() is True
            assert results[1].is_accessible() is False
            assert results[2].is_accessible() is True

    def test_format_resolution_report_accessible(self):
        """Test formatting accessible resolution report."""
        result = PublicationResolutionResult(
            identifier="PMID:12345678",
            pdf_url="https://test.com/paper.pdf",
            source="pmc",
            access_type="open_access",
            alternative_urls=["https://alt1.com", "https://alt2.com"],
        )

        report = self.assistant.format_resolution_report(result)

        assert "PMID:12345678" in report
        assert "ACCESSIBLE" in report
        assert "https://test.com/paper.pdf" in report
        assert "PMC" in report.upper()

    def test_format_resolution_report_paywalled(self):
        """Test formatting paywalled resolution report."""
        result = PublicationResolutionResult(
            identifier="PMID:87654321",
            pdf_url=None,
            source="paywalled",
            access_type="paywalled",
            suggestions="## Alternative Options\n1. PMC\n2. bioRxiv\n3. Contact author",
        )

        report = self.assistant.format_resolution_report(result)

        assert "PMID:87654321" in report
        assert "NOT DIRECTLY ACCESSIBLE" in report or "paywalled" in report.lower()
        assert "Alternative" in report

    def test_format_batch_resolution_report(self):
        """Test formatting batch resolution report."""
        results = [
            PublicationResolutionResult(
                identifier="PMID:111",
                pdf_url="http://test1.pdf",
                source="pmc",
                access_type="open_access",
            ),
            PublicationResolutionResult(
                identifier="PMID:222",
                pdf_url=None,
                source="paywalled",
                access_type="paywalled",
                suggestions="Try alternatives",
            ),
            PublicationResolutionResult(
                identifier="PMID:333",
                pdf_url=None,
                source="error",
                access_type="error",
                suggestions="Network error occurred",
            ),
        ]

        report = self.assistant.format_batch_resolution_report(results)

        assert "**Total Papers:** 3" in report
        assert "Accessible:" in report
        assert "Paywalled:" in report
        assert "Errors:" in report
        assert "PMID:111" in report
        assert "PMID:222" in report
        assert "PMID:333" in report


class TestPublicationIntelligenceServiceIntegration:
    """Test PublicationIntelligenceService with resolution integration."""

    def test_extract_methods_with_direct_url(self, mock_data_manager):
        """Test extract_methods_from_paper with direct URL (no resolution needed)."""
        service = PublicationIntelligenceService(data_manager=mock_data_manager)

        with patch.object(service, "extract_pdf_content") as mock_extract:
            mock_extract.return_value = "Methods: We used Scanpy for analysis..."

            with patch("lobster.config.llm_factory.create_llm") as mock_llm_factory:
                mock_llm = Mock()
                mock_llm.invoke.return_value = Mock(
                    content='{"software_used": ["Scanpy"], "parameters": {"min_genes": "200"}}'
                )
                mock_llm_factory.return_value = mock_llm

                result = service.extract_methods_from_paper("https://test.com/paper.pdf")

                assert "software_used" in result
                assert result["software_used"] == ["Scanpy"]
                mock_extract.assert_called_once()

    def test_extract_methods_with_pmid_resolution(self, mock_data_manager):
        """Test extract_methods_from_paper with PMID requiring resolution."""
        service = PublicationIntelligenceService(data_manager=mock_data_manager)

        with patch("lobster.agents.research_agent_assistant.ResearchAgentAssistant") as mock_assistant_class:
            # Setup assistant mock
            mock_assistant = Mock()
            mock_resolution = PublicationResolutionResult(
                identifier="PMID:12345678",
                pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123/pdf/",
                source="pmc",
                access_type="open_access",
            )
            mock_assistant.resolve_publication_to_pdf.return_value = mock_resolution
            mock_assistant_class.return_value = mock_assistant

            with patch.object(service, "extract_pdf_content") as mock_extract:
                mock_extract.return_value = "Methods: Analysis with Seurat..."

                with patch("lobster.config.llm_factory.create_llm") as mock_llm_factory:
                    mock_llm = Mock()
                    mock_llm.invoke.return_value = Mock(
                        content='{"software_used": ["Seurat"], "parameters": {"resolution": "0.5"}}'
                    )
                    mock_llm_factory.return_value = mock_llm

                    result = service.extract_methods_from_paper("PMID:12345678")

                    assert "software_used" in result
                    assert "Seurat" in result["software_used"]
                    mock_assistant.resolve_publication_to_pdf.assert_called_once_with("PMID:12345678")

    def test_extract_methods_paywalled_raises_error(self, mock_data_manager):
        """Test extract_methods_from_paper raises informative error for paywalled papers."""
        service = PublicationIntelligenceService(data_manager=mock_data_manager)

        with patch("lobster.agents.research_agent_assistant.ResearchAgentAssistant") as mock_assistant_class:
            mock_assistant = Mock()
            mock_resolution = PublicationResolutionResult(
                identifier="PMID:87654321",
                pdf_url=None,
                source="paywalled",
                access_type="paywalled",
                suggestions="Try PMC or bioRxiv for alternatives",
            )
            mock_assistant.resolve_publication_to_pdf.return_value = mock_resolution
            mock_assistant_class.return_value = mock_assistant

            with pytest.raises(ValueError) as exc_info:
                service.extract_methods_from_paper("PMID:87654321")

            assert "not openly accessible" in str(exc_info.value)
            assert "Try PMC or bioRxiv" in str(exc_info.value)

    def test_resolve_and_extract_methods_success(self, mock_data_manager):
        """Test resolve_and_extract_methods combined workflow success."""
        service = PublicationIntelligenceService(data_manager=mock_data_manager)

        with patch("lobster.agents.research_agent_assistant.ResearchAgentAssistant") as mock_assistant_class:
            mock_assistant = Mock()
            mock_resolution = PublicationResolutionResult(
                identifier="PMID:12345678",
                pdf_url="https://test.com/paper.pdf",
                source="pmc",
                access_type="open_access",
            )
            mock_assistant.resolve_publication_to_pdf.return_value = mock_resolution
            mock_assistant_class.return_value = mock_assistant

            with patch.object(service, "extract_methods_from_paper") as mock_extract:
                mock_extract.return_value = {
                    "software_used": ["Scanpy", "pandas"],
                    "parameters": {"min_genes": "200"},
                }

                result = service.resolve_and_extract_methods("PMID:12345678")

                assert result["status"] == "success"
                assert "methods" in result
                assert result["methods"]["software_used"] == ["Scanpy", "pandas"]
                assert result["source"] == "pmc"
                assert result["metadata"]["identifier"] == "PMID:12345678"

    def test_resolve_and_extract_methods_paywalled(self, mock_data_manager):
        """Test resolve_and_extract_methods with paywalled paper."""
        service = PublicationIntelligenceService(data_manager=mock_data_manager)

        with patch("lobster.agents.research_agent_assistant.ResearchAgentAssistant") as mock_assistant_class:
            mock_assistant = Mock()
            mock_resolution = PublicationResolutionResult(
                identifier="PMID:87654321",
                pdf_url=None,
                source="paywalled",
                access_type="paywalled",
                suggestions="Alternative access suggestions here",
                metadata={"pmid": "87654321"},
            )
            mock_assistant.resolve_publication_to_pdf.return_value = mock_resolution
            mock_assistant_class.return_value = mock_assistant

            result = service.resolve_and_extract_methods("PMID:87654321")

            assert result["status"] == "paywalled"
            assert "suggestions" in result
            assert result["suggestions"] == "Alternative access suggestions here"
            assert result["source"] == "paywalled"


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_workflow_search_to_extraction_simulation(self):
        """Simulate: Literature search → Extract PMIDs → Batch resolve → Extract methods."""
        assistant = ResearchAgentAssistant()

        # Step 1: Simulate literature search results (would return PMIDs)
        pmids = ["PMID:12345678", "PMID:23456789", "PMID:34567890"]

        # Step 2: Batch resolve to check accessibility
        with patch("lobster.agents.research_agent_assistant.PublicationResolver") as mock_resolver_class:
            mock_resolver = Mock()
            mock_resolver.batch_resolve.return_value = [
                PublicationResolutionResult(
                    identifier="PMID:12345678",
                    pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC111/pdf/",
                    source="pmc",
                    access_type="open_access",
                ),
                PublicationResolutionResult(
                    identifier="PMID:23456789",
                    pdf_url=None,
                    source="paywalled",
                    access_type="paywalled",
                    suggestions="Try alternatives",
                ),
                PublicationResolutionResult(
                    identifier="PMID:34567890",
                    pdf_url="https://biorxiv.org/content/test.pdf",
                    source="biorxiv",
                    access_type="preprint",
                ),
            ]
            mock_resolver_class.return_value = mock_resolver

            results = assistant.batch_resolve_publications(pmids, max_batch=5)

            # Verify resolution results
            assert len(results) == 3

            # Step 3: Filter accessible papers
            accessible = [r for r in results if r.is_accessible()]
            assert len(accessible) == 2

            # Step 4: In real workflow, would extract methods from accessible papers
            for result in accessible:
                assert result.pdf_url is not None

    def test_workflow_diagnostic_before_extraction(self):
        """Simulate: Check accessibility → Decide → Extract or suggest alternatives."""
        assistant = ResearchAgentAssistant()

        with patch("lobster.agents.research_agent_assistant.PublicationResolver") as mock_resolver_class:
            mock_resolver = Mock()

            # Test scenario: accessible paper
            mock_resolver.resolve.return_value = PublicationResolutionResult(
                identifier="PMID:12345678",
                pdf_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/pdf/",
                source="pmc",
                access_type="open_access",
            )
            mock_resolver_class.return_value = mock_resolver

            # Step 1: Diagnostic check
            result = assistant.resolve_publication_to_pdf("PMID:12345678")

            # Step 2: Decision based on result
            if result.is_accessible():
                # Would proceed with extraction
                pdf_url = result.pdf_url
                assert pdf_url is not None
                assert "ncbi.nlm.nih.gov/pmc" in pdf_url
            else:
                # Would present suggestions
                pytest.fail("Expected accessible paper")

    def test_workflow_batch_with_partial_failures(self):
        """Simulate: Batch processing with some failures → Handle gracefully."""
        assistant = ResearchAgentAssistant()

        with patch("lobster.agents.research_agent_assistant.PublicationResolver") as mock_resolver_class:
            mock_resolver = Mock()
            mock_resolver.batch_resolve.return_value = [
                PublicationResolutionResult(
                    identifier="PMID:111",
                    pdf_url="https://test1.pdf",
                    source="pmc",
                    access_type="open_access",
                ),
                PublicationResolutionResult(
                    identifier="PMID:222",
                    pdf_url=None,
                    source="paywalled",
                    access_type="paywalled",
                    suggestions="Access suggestions here",
                ),
                PublicationResolutionResult(
                    identifier="PMID:333",
                    pdf_url=None,
                    source="error",
                    access_type="error",
                    suggestions="Network error",
                ),
            ]
            mock_resolver_class.return_value = mock_resolver

            identifiers = ["PMID:111", "PMID:222", "PMID:333"]
            results = assistant.batch_resolve_publications(identifiers)

            # Generate report
            report = assistant.format_batch_resolution_report(results)

            # Verify report contains summary
            assert "**Total Papers:** 3" in report
            assert "Accessible:" in report

            # Extract successful, paywalled, and failed
            successful = [r for r in results if r.is_accessible()]
            paywalled = [r for r in results if r.access_type == "paywalled"]
            errors = [r for r in results if r.access_type == "error"]

            assert len(successful) == 1
            assert len(paywalled) == 1
            assert len(errors) == 1

            # Would proceed to extract from successful papers only
            for result in successful:
                assert result.pdf_url is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
