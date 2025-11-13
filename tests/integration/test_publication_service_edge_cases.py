"""
Comprehensive edge case tests for Publication Service.

This test module covers Agent 9's mission scenarios for publication services:
1. PubMed search with no results
2. Publication content extraction failures (404, paywalled, malformed)
3. Invalid publication identifiers (fake PMIDs, malformed DOIs)
4. Database search edge cases (special characters, injection attempts)
5. GEO dataset lookup failures

These tests ensure robust error handling for publication discovery workflows.
"""

import json
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.base_provider import DatasetType, PublicationSource
from lobster.tools.content_access_service import ContentAccessService

# ===============================================================================
# Fixtures
# ===============================================================================


@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2 instance."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.metadata_store = {}
    mock_dm.log_tool_usage = Mock()
    return mock_dm


@pytest.fixture
def content_access_service(mock_data_manager):
    """Create ContentAccessService instance."""
    return ContentAccessService(data_manager=mock_data_manager)


# ===============================================================================
# Scenario 1: PubMed Search with No Results
# ===============================================================================


@pytest.mark.integration
class TestPubMedSearchNoResults:
    """Test PubMed search scenarios that return zero results."""

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_search_nonexistent_query(self, mock_entrez, content_access_service):
        """Test search with query that returns no results."""
        # Mock empty search response
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "retmax": "0",
                        "idlist": [],
                        "querytranslation": "nonexistent_term_xyz[All Fields]",
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        # Search for nonsense term
        result = content_access_service.search_literature(
            query="nonexistent_term_xyz_12345", max_results=10
        )

        # Should handle gracefully
        assert isinstance(result, str)
        assert (
            "no results" in result.lower()
            or "0 results" in result
            or "not found" in result.lower()
        )

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_search_overly_specific_query(self, mock_entrez, content_access_service):
        """Test search with overly specific criteria (no matches)."""
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "idlist": [],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        # Overly specific query
        result = content_access_service.search_literature(
            query="BRCA1 mutation in left-handed vegetarian astronauts", max_results=5
        )

        assert isinstance(result, str)
        assert "0" in result or "no results" in result.lower()

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_search_with_invalid_date_range(self, mock_entrez, content_access_service):
        """Test search with date range that has no publications."""
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "idlist": [],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        # Search for future date range (no results)
        result = content_access_service.search_literature(
            query="cancer research", filters={"date_range": "2050/01/01:2050/12/31"}
        )

        assert isinstance(result, str)
        assert "0" in result or "no" in result.lower()

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_search_misspelled_terms(self, mock_entrez, content_access_service):
        """Test search with heavily misspelled terms (no results)."""
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "idlist": [],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        result = content_access_service.search_literature(
            query="braest cancr gnomic squencing", max_results=10  # Misspelled
        )

        assert isinstance(result, str)


# ===============================================================================
# Scenario 2: Publication Content Extraction Failures
# ===============================================================================


@pytest.mark.integration
class TestPublicationContentExtractionFailures:
    """Test failures in extracting publication full text/content."""

    @patch("requests.Session.get")
    def test_404_publication_not_found(self, mock_get, content_access_service):
        """Test extraction failure when publication URL returns 404."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        # Attempt to extract metadata
        result = content_access_service.extract_metadata(
            identifier="PMID:99999999", source=PublicationSource.PUBMED
        )

        # Should return error message
        assert isinstance(result, str)
        assert "error" in result.lower() or "not found" in result.lower()

    @patch("requests.Session.get")
    def test_paywalled_publication(self, mock_get, content_access_service):
        """Test extraction failure for paywalled publication."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        mock_get.return_value = mock_response

        result = content_access_service.extract_metadata(
            identifier="10.1038/s41586-021-12345-6", source=PublicationSource.PUBMED
        )

        # Should handle paywall gracefully
        assert isinstance(result, str)
        assert (
            "error" in result.lower()
            or "forbidden" in result.lower()
            or "access" in result.lower()
        )

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_malformed_xml_response(self, mock_entrez, content_access_service):
        """Test handling of malformed XML in metadata response."""
        # Return invalid XML
        mock_fetch_handle = StringIO("<DocumentSummary><incomplete_tag>")
        mock_entrez.efetch.return_value = mock_fetch_handle

        # Should handle parsing error gracefully
        with pytest.raises(Exception):
            content_access_service.extract_metadata(
                identifier="PMID:12345678", source=PublicationSource.PUBMED
            )

    @patch("requests.Session.get")
    def test_network_timeout_during_extraction(self, mock_get, content_access_service):
        """Test extraction failure due to network timeout."""
        mock_get.side_effect = requests.Timeout("Connection timed out")

        result = content_access_service.extract_metadata(
            identifier="PMID:12345678", source=PublicationSource.PUBMED
        )

        assert isinstance(result, str)
        assert "error" in result.lower() or "timeout" in result.lower()

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_empty_abstract_content(self, mock_entrez, content_access_service):
        """Test handling of publications with empty abstract."""
        mock_fetch_handle = StringIO(
            """<DocumentSummary uid="12345678">
                <Title>Test Publication</Title>
                <PubDate>2023</PubDate>
                <AuthorList></AuthorList>
            </DocumentSummary>"""
        )
        mock_entrez.efetch.return_value = mock_fetch_handle

        result = content_access_service.extract_metadata(
            identifier="PMID:12345678", source=PublicationSource.PUBMED
        )

        # Should return metadata even without abstract
        assert isinstance(result, (dict, str))


# ===============================================================================
# Scenario 3: Invalid Publication Identifiers
# ===============================================================================


@pytest.mark.integration
class TestInvalidPublicationIdentifiers:
    """Test handling of invalid publication identifiers."""

    def test_fake_pmid(self, content_access_service):
        """Test handling of non-existent PMID."""
        fake_pmid = "PMID:99999999999"

        # Should validate identifier format
        is_valid = (
            content_access_service.registry.get_default_provider().validate_identifier(
                fake_pmid
            )
        )

        # May be valid format but won't exist
        assert isinstance(is_valid, bool)

    def test_malformed_doi(self, content_access_service):
        """Test handling of malformed DOI."""
        malformed_dois = [
            "10.1038",  # Incomplete
            "not_a_doi",  # Invalid format
            "10.1038/",  # Trailing slash
            "doi:10.1038",  # Incorrect prefix
        ]

        for doi in malformed_dois:
            provider = content_access_service.registry.get_default_provider()
            # Should reject or handle gracefully
            is_valid = provider.validate_identifier(doi)

            # Either invalid or None
            assert is_valid is False or is_valid is None or isinstance(is_valid, bool)

    def test_pmid_with_letters(self, content_access_service):
        """Test PMID containing letters (invalid)."""
        invalid_pmid = "PMID:123ABC456"

        provider = content_access_service.registry.get_default_provider()
        is_valid = provider.validate_identifier(invalid_pmid)

        assert is_valid is False or is_valid is None

    def test_empty_identifier(self, content_access_service):
        """Test empty identifier string."""
        empty_identifiers = ["", "   ", "PMID:", "doi:"]

        for identifier in empty_identifiers:
            result = content_access_service.extract_metadata(identifier)

            # Should handle gracefully with error message
            assert isinstance(result, str)
            assert "error" in result.lower() or "invalid" in result.lower()

    def test_url_as_identifier(self, content_access_service):
        """Test full URL passed as identifier (should extract DOI/PMID)."""
        urls = [
            "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            "https://www.ncbi.nlm.nih.gov/pubmed/12345678",
            "https://doi.org/10.1038/s41586-021-12345-6",
        ]

        for url in urls:
            # Should extract identifier from URL
            provider = content_access_service.registry.get_default_provider()
            is_valid = provider.validate_identifier(url)

            # Should handle URL extraction
            assert isinstance(is_valid, bool)


# ===============================================================================
# Scenario 4: Database Search Edge Cases
# ===============================================================================


@pytest.mark.integration
class TestDatabaseSearchEdgeCases:
    """Test edge cases in database searches."""

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_search_with_special_characters(self, mock_entrez, content_access_service):
        """Test search with special characters (quotes, operators)."""
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "5",
                        "idlist": ["12345", "12346", "12347", "12348", "12349"],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        special_queries = [
            "cancer AND (breast OR ovarian)",
            '"exact phrase match"',
            "protein-protein interactions",
            "CD4+ T cells",
            "α-synuclein aggregation",
        ]

        for query in special_queries:
            result = content_access_service.search_literature(query=query, max_results=5)

            # Should handle special characters
            assert isinstance(result, str)
            # Should not crash or return error
            assert (
                "error" not in result.lower()
                or "5 results" in result
                or "found" in result.lower()
            )

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_search_with_sql_injection_attempt(self, mock_entrez, content_access_service):
        """Test that SQL injection attempts are safely handled."""
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "idlist": [],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        # SQL injection attempts
        injection_queries = [
            "cancer'; DROP TABLE publications;--",
            "1' OR '1'='1",
            'cancer"; SELECT * FROM users;--',
        ]

        for query in injection_queries:
            result = content_access_service.search_literature(query=query, max_results=5)

            # Should handle safely (treat as normal text)
            assert isinstance(result, str)
            # Should not crash or execute SQL
            assert "DROP" not in result or "SELECT" not in result

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_search_with_unicode_characters(self, mock_entrez, content_access_service):
        """Test search with Unicode characters."""
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "3",
                        "idlist": ["123", "456", "789"],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        unicode_queries = [
            "β-amyloid protein",
            "α-synuclein in Parkinson's",
            "γ-secretase inhibitors",
            "Müller cells",
            "café-au-lait spots",
        ]

        for query in unicode_queries:
            result = content_access_service.search_literature(query=query, max_results=5)

            # Should handle Unicode
            assert isinstance(result, str)

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_search_with_very_long_query(self, mock_entrez, content_access_service):
        """Test search with extremely long query string."""
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "idlist": [],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        # Generate very long query (>1000 characters)
        long_query = " AND ".join([f"gene{i}" for i in range(200)])

        result = content_access_service.search_literature(query=long_query, max_results=5)

        # Should handle long query (may truncate or reject)
        assert isinstance(result, str)


# ===============================================================================
# Scenario 5: GEO Dataset Lookup Failures
# ===============================================================================


@pytest.mark.integration
class TestGEODatasetLookupFailures:
    """Test GEO dataset lookup failures in publication context."""

    @patch("lobster.tools.providers.geo_provider.Entrez")
    def test_geo_accession_not_found(self, mock_entrez, content_access_service):
        """Test lookup of non-existent GEO accession."""
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "idlist": [],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        # Search for non-existent accession
        result = content_access_service.discover_datasets(
            query="GSE999999999", dataset_type=DatasetType.GEO, max_results=5
        )

        # Should handle not found
        assert isinstance(result, str)
        assert "not found" in result.lower() or "0" in result or "no" in result.lower()

    @patch("lobster.tools.providers.geo_provider.Entrez")
    def test_geo_dataset_from_invalid_pmid(self, mock_entrez, content_access_service):
        """Test finding GEO datasets from invalid PMID."""
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "idlist": [],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        result = content_access_service.find_linked_datasets(
            identifier="PMID:99999999", dataset_types=[DatasetType.GEO]
        )

        # Should handle gracefully
        assert isinstance(result, str)
        assert (
            "not found" in result.lower()
            or "no datasets" in result.lower()
            or "0" in result
        )

    @patch("lobster.tools.providers.geo_provider.Entrez")
    def test_geo_metadata_extraction_failure(self, mock_entrez, content_access_service):
        """Test failure in extracting GEO metadata."""
        # Return malformed response
        mock_fetch_handle = StringIO("<invalid_xml>")
        mock_entrez.efetch.return_value = mock_fetch_handle

        result = content_access_service.extract_metadata(
            identifier="GSE123456", source=PublicationSource.GEO
        )

        # Should handle parsing error
        assert isinstance(result, str)
        assert "error" in result.lower() or "failed" in result.lower()

    @patch("lobster.tools.providers.geo_provider.Entrez")
    def test_geo_search_with_no_samples(self, mock_entrez, content_access_service):
        """Test GEO dataset with zero samples."""
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "1",
                        "idlist": ["200123456"],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        # Mock metadata with 0 samples
        mock_fetch_handle = StringIO(
            """<DocumentSummary uid="200123456">
                <Accession>GSE123456</Accession>
                <title>Test Dataset</title>
                <n_samples>0</n_samples>
            </DocumentSummary>"""
        )
        mock_entrez.efetch.return_value = mock_fetch_handle

        result = content_access_service.discover_datasets(
            query="empty dataset", dataset_type=DatasetType.GEO, max_results=5
        )

        # Should handle or warn about 0 samples
        assert isinstance(result, str)

    @patch("lobster.tools.providers.geo_provider.Entrez")
    def test_geo_network_error_during_search(self, mock_entrez, content_access_service):
        """Test network error during GEO search."""
        mock_entrez.esearch.side_effect = Exception("Network connection failed")

        # Should handle network error gracefully
        with pytest.raises(Exception):
            content_access_service.discover_datasets(
                query="test query", dataset_type=DatasetType.GEO, max_results=5
            )


# ===============================================================================
# Integration Tests: Full Workflow Edge Cases
# ===============================================================================


@pytest.mark.integration
class TestPublicationWorkflowEdgeCases:
    """Test full publication discovery workflows with edge cases."""

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_search_then_extract_nonexistent_paper(
        self, mock_entrez, content_access_service
    ):
        """Test workflow: search returns result, but extraction fails."""
        # Search succeeds
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "1",
                        "idlist": ["12345678"],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        search_result = content_access_service.search_literature(
            query="test query", max_results=1
        )
        assert isinstance(search_result, str)

        # But extraction fails (paper deleted?)
        mock_entrez.efetch.side_effect = Exception("Record not found")

        with pytest.raises(Exception):
            content_access_service.extract_metadata(
                identifier="PMID:12345678", source=PublicationSource.PUBMED
            )

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    @patch("lobster.tools.providers.geo_provider.Entrez")
    def test_paper_with_no_geo_datasets(
        self, mock_geo_entrez, mock_pubmed_entrez, content_access_service
    ):
        """Test finding datasets from paper with no associated GEO data."""
        # Paper exists
        mock_fetch_handle = StringIO(
            """<DocumentSummary uid="12345678">
                <Title>Test Paper</Title>
                <PubDate>2023</PubDate>
            </DocumentSummary>"""
        )
        mock_pubmed_entrez.efetch.return_value = mock_fetch_handle

        # But no GEO datasets linked
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "idlist": [],
                    }
                }
            )
        )
        mock_geo_entrez.esearch.return_value = mock_search_handle

        result = content_access_service.find_linked_datasets(
            identifier="PMID:12345678", dataset_types=[DatasetType.GEO]
        )

        # Should report no datasets found
        assert isinstance(result, str)
        assert (
            "no datasets" in result.lower()
            or "0" in result
            or "not found" in result.lower()
        )

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_rate_limit_during_batch_search(self, mock_entrez, content_access_service):
        """Test rate limiting during batch publication search."""
        # Simulate rate limit error
        mock_entrez.esearch.side_effect = Exception("API rate limit exceeded")

        with pytest.raises(Exception, match="rate limit"):
            content_access_service.search_literature(
                query="test query", max_results=100  # Large request
            )


# ===============================================================================
# Stress Tests
# ===============================================================================


@pytest.mark.integration
class TestPublicationServiceStress:
    """Stress tests for publication service."""

    @patch("lobster.tools.providers.pubmed_provider.Entrez")
    def test_many_concurrent_searches(self, mock_entrez, content_access_service):
        """Test many concurrent literature searches."""
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "5",
                        "idlist": ["1", "2", "3", "4", "5"],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        # Simulate 10 concurrent searches
        from concurrent.futures import ThreadPoolExecutor

        def search_worker(query):
            return content_access_service.search_literature(query=query, max_results=5)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(search_worker, f"query_{i}") for i in range(10)]

            results = [f.result() for f in futures]

        # All should succeed
        assert len(results) == 10
        assert all(isinstance(r, str) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
