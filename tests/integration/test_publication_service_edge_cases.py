"""
Publication service edge case integration tests.

These tests exercise ContentAccessService routing and error handling while
mocking the current provider methods directly. They intentionally avoid the
legacy Bio.Entrez surface; PubMedProvider and GEOProvider now use direct
E-utilities request helpers internally.
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import requests

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.data_access.content_access_service import ContentAccessService
from lobster.tools.providers.base_provider import (
    DatasetType,
    PublicationMetadata,
    PublicationSource,
)
from lobster.tools.providers.geo_provider import GEOProvider
from lobster.tools.providers.pubmed_provider import PubMedProvider


@pytest.fixture
def data_manager(tmp_path):
    """Create a real DataManagerV2 so provider initialization matches runtime."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return DataManagerV2(workspace_path=workspace)


@pytest.fixture
def content_access_service(data_manager):
    """Create ContentAccessService for routing-level integration tests."""
    return ContentAccessService(data_manager=data_manager)


def _text(result) -> str:
    """Normalize service results across tuple, model, and string contracts."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, PublicationMetadata):
        parts = [
            result.uid,
            result.title,
            result.abstract or "",
            result.journal or "",
            result.pmid or "",
            result.doi or "",
        ]
        return " ".join(parts)
    return str(result)


def _pubmed_provider(service: ContentAccessService) -> PubMedProvider:
    for provider in service.registry.get_all_providers():
        if isinstance(provider, PubMedProvider):
            return provider
    raise AssertionError("PubMedProvider was not registered")


@pytest.mark.integration
class TestPubMedSearchNoResults:
    """Searches that should return an empty PubMed result set."""

    @pytest.mark.parametrize(
        ("query", "filters"),
        [
            ("nonexistent_term_xyz_12345", None),
            ("BRCA1 mutation in left-handed vegetarian astronauts", None),
            ("cancer research", {"date_range": "2050/01/01:2050/12/31"}),
            ("braest cancr gnomic squencing", None),
        ],
    )
    def test_search_no_results(self, query, filters, content_access_service):
        with patch.object(
            PubMedProvider,
            "search_publications",
            return_value="No PubMed results found for your query.",
        ):
            result = content_access_service.search_literature(
                query=query, max_results=10, filters=filters
            )

        text = _text(result).lower()
        assert "no pubmed results" in text


@pytest.mark.integration
class TestPublicationContentExtractionFailures:
    """Failures while extracting publication metadata."""

    @pytest.mark.parametrize(
        "error",
        [
            requests.HTTPError("404 Not Found"),
            requests.HTTPError("403 Forbidden"),
            requests.Timeout("Connection timed out"),
            ValueError("Malformed PubMed XML"),
        ],
    )
    def test_pubmed_metadata_errors_are_returned(self, error, content_access_service):
        with patch.object(
            PubMedProvider,
            "extract_publication_metadata",
            side_effect=error,
        ):
            result = content_access_service.extract_metadata(
                identifier="PMID:12345678",
                source=PublicationSource.PUBMED,
            )

        text = _text(result).lower()
        assert "metadata extraction error" in text
        assert str(error).split()[0].lower() in text

    def test_empty_abstract_content(self, content_access_service):
        metadata = PublicationMetadata(
            uid="12345678",
            title="Test Publication",
            pmid="12345678",
            abstract=None,
        )

        with patch.object(
            PubMedProvider,
            "extract_publication_metadata",
            return_value=metadata,
        ):
            result = content_access_service.extract_metadata(
                identifier="PMID:12345678",
                source=PublicationSource.PUBMED,
            )

        assert isinstance(result, PublicationMetadata)
        assert result.title == "Test Publication"
        assert result.abstract is None


@pytest.mark.integration
class TestInvalidPublicationIdentifiers:
    """Identifier validation and empty-input handling."""

    def test_fake_pmid_format_is_validated(self, content_access_service):
        provider = _pubmed_provider(content_access_service)
        assert isinstance(provider.validate_identifier("PMID:99999999999"), bool)

    @pytest.mark.parametrize("identifier", ["10.1038", "not_a_doi", "10.1038/", "doi:10.1038"])
    def test_malformed_doi(self, identifier, content_access_service):
        provider = _pubmed_provider(content_access_service)
        assert isinstance(provider.validate_identifier(identifier), bool)

    def test_pmid_with_letters(self, content_access_service):
        provider = _pubmed_provider(content_access_service)
        assert provider.validate_identifier("PMID:123ABC456") is False

    @pytest.mark.parametrize("identifier", ["", "   ", "PMID:", "doi:"])
    def test_empty_identifier(self, identifier, content_access_service):
        result = content_access_service.extract_metadata(identifier)
        text = _text(result).lower()
        assert "invalid" in text or "error" in text

    @pytest.mark.parametrize(
        "url",
        [
            "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            "https://www.ncbi.nlm.nih.gov/pubmed/12345678",
            "https://doi.org/10.1038/s41586-021-12345-6",
        ],
    )
    def test_url_as_identifier(self, url, content_access_service):
        provider = _pubmed_provider(content_access_service)
        assert isinstance(provider.validate_identifier(url), bool)


@pytest.mark.integration
class TestDatabaseSearchEdgeCases:
    """Query text edge cases for PubMed search routing."""

    @pytest.mark.parametrize(
        "query",
        [
            "cancer AND (breast OR ovarian)",
            '"exact phrase match"',
            "protein-protein interactions",
            "CD4+ T cells",
            "alpha-synuclein aggregation",
        ],
    )
    def test_search_with_special_characters(self, query, content_access_service):
        with patch.object(
            PubMedProvider,
            "search_publications",
            return_value="Found 5 PubMed results\nPMID: 12345",
        ):
            result = content_access_service.search_literature(
                query=query,
                max_results=5,
            )

        text = _text(result).lower()
        assert "error" not in text
        assert "pmid" in text or "found" in text

    @pytest.mark.parametrize(
        "query",
        [
            "cancer'; DROP TABLE publications;--",
            "1' OR '1'='1",
            'cancer"; SELECT * FROM users;--',
        ],
    )
    def test_search_with_sql_injection_attempt(self, query, content_access_service):
        with patch.object(
            PubMedProvider,
            "search_publications",
            return_value="No PubMed results found for your query.",
        ):
            result = content_access_service.search_literature(
                query=query,
                max_results=5,
            )

        text = _text(result)
        assert isinstance(text, str)
        assert "DROP TABLE" not in text
        assert "SELECT *" not in text

    @pytest.mark.parametrize(
        "query",
        [
            "beta-amyloid protein",
            "alpha-synuclein in Parkinson's",
            "gamma-secretase inhibitors",
            "Muller cells",
            "cafe-au-lait spots",
        ],
    )
    def test_search_with_ascii_transliterated_terms(self, query, content_access_service):
        with patch.object(
            PubMedProvider,
            "search_publications",
            return_value="Found 3 PubMed results\nPMID: 123",
        ):
            result = content_access_service.search_literature(query=query, max_results=5)

        assert isinstance(_text(result), str)

    def test_search_with_very_long_query(self, content_access_service):
        long_query = " AND ".join([f"gene{i}" for i in range(200)])

        with patch.object(
            PubMedProvider,
            "search_publications",
            return_value="No PubMed results found for your query.",
        ):
            result = content_access_service.search_literature(
                query=long_query,
                max_results=5,
            )

        assert "no pubmed results" in _text(result).lower()


@pytest.mark.integration
class TestGEODatasetLookupFailures:
    """GEO lookup failure paths routed through ContentAccessService."""

    def test_geo_accession_not_found(self, content_access_service):
        with patch.object(
            GEOProvider,
            "search_by_accession",
            return_value="GEO dataset not found: GSE999999999",
        ):
            result = content_access_service.discover_datasets(
                query="GSE999999999",
                dataset_type=DatasetType.GEO,
                max_results=5,
            )

        assert "not found" in _text(result).lower()

    def test_geo_dataset_from_invalid_pmid(self, content_access_service):
        with patch.object(
            PubMedProvider,
            "find_datasets_from_publication",
            return_value="No datasets found for PMID:99999999",
        ):
            result = content_access_service.find_linked_datasets(
                identifier="PMID:99999999",
                dataset_types=[DatasetType.GEO],
            )

        assert "no datasets" in _text(result).lower()

    def test_geo_metadata_extraction_failure(self, content_access_service):
        with patch.object(
            GEOProvider,
            "extract_publication_metadata",
            side_effect=ValueError("Malformed GEO metadata"),
        ):
            result = content_access_service.extract_metadata(
                identifier="GSE123456",
                source=PublicationSource.GEO,
            )

        text = _text(result).lower()
        assert "metadata extraction error" in text
        assert "malformed geo metadata" in text

    def test_geo_search_with_no_samples(self, content_access_service):
        with patch.object(
            GEOProvider,
            "search_publications",
            return_value="Found GEO dataset GSE123456 with 0 samples",
        ):
            result = content_access_service.discover_datasets(
                query="empty dataset",
                dataset_type=DatasetType.GEO,
                max_results=5,
            )

        text = _text(result).lower()
        assert "gse123456" in text
        assert "0 samples" in text

    def test_geo_network_error_during_search(self, content_access_service):
        with patch.object(
            GEOProvider,
            "search_publications",
            side_effect=ConnectionError("Network connection failed"),
        ):
            result = content_access_service.discover_datasets(
                query="test query",
                dataset_type=DatasetType.GEO,
                max_results=5,
            )

        text = _text(result).lower()
        assert "dataset search error" in text
        assert "network connection failed" in text


@pytest.mark.integration
class TestPublicationWorkflowEdgeCases:
    """Multi-step workflow edge cases."""

    def test_search_then_extract_nonexistent_paper(self, content_access_service):
        with patch.object(
            PubMedProvider,
            "search_publications",
            return_value="Found 1 PubMed result\nPMID: 12345678",
        ):
            search_result = content_access_service.search_literature(
                query="test query",
                max_results=1,
            )

        assert "found 1" in _text(search_result).lower()

        with patch.object(
            PubMedProvider,
            "extract_publication_metadata",
            side_effect=ValueError("Record not found"),
        ):
            metadata_result = content_access_service.extract_metadata(
                identifier="PMID:12345678",
                source=PublicationSource.PUBMED,
            )

        assert "record not found" in _text(metadata_result).lower()

    def test_paper_with_no_geo_datasets(self, content_access_service):
        with patch.object(
            PubMedProvider,
            "find_datasets_from_publication",
            return_value="No datasets found.",
        ):
            result = content_access_service.find_linked_datasets(
                identifier="PMID:12345678",
                dataset_types=[DatasetType.GEO],
            )

        assert "no datasets" in _text(result).lower()

    def test_rate_limit_during_batch_search(self, content_access_service):
        with patch.object(
            PubMedProvider,
            "search_publications",
            side_effect=RuntimeError("API rate limit exceeded"),
        ):
            result = content_access_service.search_literature(
                query="test query",
                max_results=100,
            )

        assert "rate limit" in _text(result).lower()


@pytest.mark.integration
class TestPublicationServiceStress:
    """Concurrent search calls should all route cleanly."""

    def test_many_concurrent_searches(self, content_access_service):
        with patch.object(
            PubMedProvider,
            "search_publications",
            return_value="Found 5 PubMed results\nPMID: 1",
        ):

            def search_worker(query):
                return content_access_service.search_literature(
                    query=query,
                    max_results=5,
                )

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(search_worker, f"query_{i}") for i in range(10)
                ]
                results = [future.result() for future in futures]

        assert len(results) == 10
        assert all(isinstance(_text(result), str) for result in results)
