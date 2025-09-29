"""
Comprehensive unit tests for publication providers.

This module provides thorough testing of publication providers including
PubMed literature search and GEO dataset discovery.

Test coverage target: 95%+ with meaningful tests for provider operations.
"""

import pytest
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import urllib.error
from datetime import datetime
import tempfile
from pathlib import Path

from lobster.tools.providers.base_provider import (
    BasePublicationProvider,
    PublicationSource,
    DatasetType,
    PublicationMetadata,
    DatasetMetadata
)
from lobster.tools.providers.pubmed_provider import PubMedProvider, PubMedProviderConfig
from lobster.tools.providers.geo_provider import GEOProvider, GEOProviderConfig
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def mock_data_manager():
    """Mock DataManagerV2 instance."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.log_tool_usage = Mock()
    return mock_dm


@pytest.fixture
def mock_pubmed_search_response():
    """Mock PubMed eSearch API response data."""
    return {
        "esearchresult": {
            "count": "3",
            "retmax": "20",
            "retstart": "0",
            "idlist": ["37123456", "37123457", "37123458"],
            "webenv": "MCID_123456789",
            "querykey": "1"
        }
    }


@pytest.fixture
def mock_pubmed_article_xml():
    """Mock PubMed eFetch XML response."""
    return """<?xml version="1.0" ?>
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <ArticleTitle>Single-cell RNA sequencing reveals novel T cell populations in cancer</ArticleTitle>
                    <Journal>
                        <Title>Nature</Title>
                        <JournalIssue>
                            <PubDate>
                                <Year>2023</Year>
                                <Month>06</Month>
                                <Day>15</Day>
                            </PubDate>
                        </JournalIssue>
                    </Journal>
                    <Abstract>
                        <AbstractText>Single-cell RNA sequencing has revolutionized our understanding of cellular heterogeneity in cancer. This study identifies novel T cell subpopulations with distinct functional profiles.</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>"""


@pytest.fixture
def mock_geo_search_response():
    """Mock GEO eSearch API response data."""
    return {
        "esearchresult": {
            "count": "2",
            "retmax": "20",
            "retstart": "0",
            "idlist": ["200123456", "200123457"],
            "webenv": "MCID_GEO_123",
            "querykey": "1"
        }
    }


@pytest.fixture
def mock_geo_summary_response():
    """Mock GEO eSummary API response data."""
    return {
        "result": {
            "200123456": {
                "uid": "200123456",
                "accession": "GSE123456",
                "title": "Single-cell RNA-seq of tumor-infiltrating T cells",
                "summary": "This dataset contains single-cell RNA sequencing data from tumor-infiltrating lymphocytes.",
                "taxon": "Homo sapiens",
                "GPL": "GPL24676",
                "n_samples": "20",
                "PDAT": "2023/06/15",
                "entryType": "GSE"
            }
        }
    }


@pytest.fixture
def pubmed_provider_config():
    """Configuration for PubMed provider testing."""
    return PubMedProviderConfig(
        email="test@example.com",
        api_key="test_api_key",
        top_k_results=5,
        max_retry=2
    )


@pytest.fixture
def geo_provider_config():
    """Configuration for GEO provider testing."""
    return GEOProviderConfig(
        email="test@example.com",
        api_key="test_api_key",
        max_results=10,
        max_retry=2
    )


# ===============================================================================
# Base Provider Tests
# ===============================================================================

@pytest.mark.unit
class TestBasePublicationProvider:
    """Test base publication provider functionality."""

    def test_abstract_methods_defined(self):
        """Test that BasePublicationProvider defines required abstract methods."""
        abstract_methods = BasePublicationProvider.__abstractmethods__
        expected_methods = {
            'source',
            'supported_dataset_types',
            'search_publications',
            'find_datasets_from_publication',
            'extract_publication_metadata'
        }
        assert expected_methods.issubset(abstract_methods)

    def test_cannot_instantiate_abstract_base(self):
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            BasePublicationProvider()

    def test_publication_metadata_structure(self):
        """Test PublicationMetadata model structure."""
        metadata = PublicationMetadata(
            uid="12345",
            title="Test Publication",
            journal="Test Journal",
            published="2023-01-01",
            doi="10.1000/test",
            pmid="12345",
            abstract="Test abstract",
            authors=["Author One", "Author Two"],
            keywords=["test", "publication"]
        )

        assert metadata.uid == "12345"
        assert metadata.title == "Test Publication"
        assert len(metadata.authors) == 2
        assert len(metadata.keywords) == 2

    def test_dataset_metadata_structure(self):
        """Test DatasetMetadata model structure."""
        metadata = DatasetMetadata(
            accession="GSE123456",
            title="Test Dataset",
            description="Test description",
            organism="Homo sapiens",
            platform="GPL12345",
            samples_count=10,
            date="2023-01-01",
            data_type=DatasetType.GEO,
            source_url="https://example.com"
        )

        assert metadata.accession == "GSE123456"
        assert metadata.data_type == DatasetType.GEO
        assert metadata.samples_count == 10

    def test_publication_source_enum(self):
        """Test PublicationSource enum values."""
        assert PublicationSource.PUBMED.value == "pubmed"
        assert PublicationSource.GEO.value == "geo"
        assert PublicationSource.BIORXIV.value == "biorxiv"

    def test_dataset_type_enum(self):
        """Test DatasetType enum values."""
        assert DatasetType.GEO.value == "geo"
        assert DatasetType.SRA.value == "sra"
        assert DatasetType.BIOPROJECT.value == "bioproject"


# ===============================================================================
# PubMed Provider Tests
# ===============================================================================

@pytest.mark.unit
class TestPubMedProvider:
    """Test PubMed provider functionality."""

    def test_pubmed_provider_initialization(self, mock_data_manager, pubmed_provider_config):
        """Test PubMedProvider initialization."""
        with patch('lobster.tools.providers.pubmed_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = PubMedProvider(mock_data_manager, pubmed_provider_config)

            assert provider.config.email == "test@example.com"
            assert provider.config.api_key == "test_api_key"
            assert provider.source == PublicationSource.PUBMED
            assert DatasetType.BIOPROJECT in provider.supported_dataset_types

    def test_pubmed_provider_properties(self, mock_data_manager, pubmed_provider_config):
        """Test PubMedProvider properties."""
        with patch('lobster.tools.providers.pubmed_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = PubMedProvider(mock_data_manager, pubmed_provider_config)

            assert provider.source == PublicationSource.PUBMED
            supported_types = provider.supported_dataset_types
            assert DatasetType.BIOPROJECT in supported_types
            assert DatasetType.BIOSAMPLE in supported_types
            assert DatasetType.DBGAP in supported_types

    def test_validate_identifier(self, mock_data_manager, pubmed_provider_config):
        """Test PubMed identifier validation."""
        with patch('lobster.tools.providers.pubmed_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = PubMedProvider(mock_data_manager, pubmed_provider_config)

            # Valid identifiers
            assert provider.validate_identifier("12345678") == True  # PMID
            assert provider.validate_identifier("PMID:12345678") == True  # PMID with prefix
            assert provider.validate_identifier("10.1038/nature12345") == True  # DOI

            # Invalid identifiers
            assert provider.validate_identifier("") == False
            assert provider.validate_identifier("invalid") == False

    def test_search_publications(self, mock_data_manager, pubmed_provider_config, mock_pubmed_search_response):
        """Test PubMed publication search."""
        with patch('lobster.tools.providers.pubmed_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = PubMedProvider(mock_data_manager, pubmed_provider_config)

            with patch.object(provider, '_load_with_params') as mock_load:
                mock_articles = [
                    {
                        'uid': '37123456',
                        'Title': 'Single-cell RNA sequencing reveals novel T cell populations',
                        'Journal': 'Nature',
                        'Published': '2023-06-15',
                        'Summary': 'Single-cell RNA sequencing has revolutionized...'
                    }
                ]
                mock_load.return_value = iter(mock_articles)

                result = provider.search_publications("single cell RNA seq", max_results=5)

                assert "Pubmed Search Results" in result
                assert "single cell RNA seq" in result
                assert "37123456" in result
                assert "Single-cell RNA sequencing" in result
                mock_data_manager.log_tool_usage.assert_called_once()

    def test_search_publications_no_results(self, mock_data_manager, pubmed_provider_config):
        """Test PubMed search with no results."""
        with patch('lobster.tools.providers.pubmed_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = PubMedProvider(mock_data_manager, pubmed_provider_config)

            with patch.object(provider, '_load_with_params') as mock_load:
                mock_load.return_value = iter([])  # No results

                result = provider.search_publications("nonexistent query")

                assert "No PubMed results found" in result

    def test_find_datasets_from_publication(self, mock_data_manager, pubmed_provider_config):
        """Test finding datasets from PubMed publication."""
        with patch('lobster.tools.providers.pubmed_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = PubMedProvider(mock_data_manager, pubmed_provider_config)

            mock_article = {
                'uid': '37123456',
                'Title': 'Test Publication with GSE123456 dataset',
                'Summary': 'This study used GSE123456 for analysis.'
            }

            with patch.object(provider, '_load_with_params') as mock_load, \
                 patch.object(provider, '_extract_dataset_accessions') as mock_extract, \
                 patch.object(provider, '_find_linked_datasets') as mock_linked:

                mock_load.return_value = iter([mock_article])
                mock_extract.return_value = {'GEO': ['GSE123456']}
                mock_linked.return_value = {}

                result = provider.find_datasets_from_publication("37123456")

                assert "Dataset Discovery Report" in result
                assert "GSE123456" in result
                mock_data_manager.log_tool_usage.assert_called_once()

    def test_extract_publication_metadata(self, mock_data_manager, pubmed_provider_config):
        """Test extracting publication metadata."""
        with patch('lobster.tools.providers.pubmed_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = PubMedProvider(mock_data_manager, pubmed_provider_config)

            mock_article = {
                'uid': '37123456',
                'Title': 'Test Publication',
                'Journal': 'Nature',
                'Published': '2023-06-15',
                'Summary': 'Test abstract'
            }

            with patch.object(provider, '_load_with_params') as mock_load:
                mock_load.return_value = iter([mock_article])

                metadata = provider.extract_publication_metadata("37123456")

                assert isinstance(metadata, PublicationMetadata)
                assert metadata.uid == "37123456"
                assert metadata.title == "Test Publication"
                assert metadata.journal == "Nature"

    def test_ncbi_request_with_retry(self, mock_data_manager, pubmed_provider_config):
        """Test NCBI request retry mechanism."""
        with patch('lobster.tools.providers.pubmed_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = PubMedProvider(mock_data_manager, pubmed_provider_config)

            with patch('urllib.request.urlopen') as mock_urlopen:
                # Mock successful response after retry
                mock_response = Mock()
                mock_response.read.return_value = b'{"test": "data"}'
                mock_urlopen.side_effect = [
                    urllib.error.HTTPError("url", 500, "Server Error", {}, None),
                    mock_response
                ]

                result = provider._make_ncbi_request("http://test.url", "test operation")

                assert result == b'{"test": "data"}'
                assert mock_urlopen.call_count == 2

    def test_rate_limiting(self, mock_data_manager, pubmed_provider_config):
        """Test request rate limiting."""
        with patch('lobster.tools.providers.pubmed_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = PubMedProvider(mock_data_manager, pubmed_provider_config)

            with patch('time.sleep') as mock_sleep, \
                 patch('time.time') as mock_time:

                # Mock time to simulate rapid requests that trigger rate limiting
                # Need more values since time.time() is called multiple times
                mock_time.side_effect = [0.0, 0.0, 0.0, 0.05, 0.05, 0.05]  # Fast successive calls
                # Set the provider to have no API key to trigger stricter limits
                provider.config.api_key = None

                provider._apply_request_throttling()
                provider._apply_request_throttling()

                # Should have applied rate limiting
                mock_sleep.assert_called()


# ===============================================================================
# GEO Provider Tests
# ===============================================================================

@pytest.mark.unit
class TestGEOProvider:
    """Test GEO provider functionality."""

    def test_geo_provider_initialization(self, mock_data_manager, geo_provider_config):
        """Test GEOProvider initialization."""
        with patch('lobster.tools.providers.geo_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = GEOProvider(mock_data_manager, geo_provider_config)

            assert provider.config.email == "test@example.com"
            assert provider.config.api_key == "test_api_key"
            assert provider.source == PublicationSource.GEO
            assert DatasetType.GEO in provider.supported_dataset_types

    def test_geo_provider_properties(self, mock_data_manager, geo_provider_config):
        """Test GEOProvider properties."""
        with patch('lobster.tools.providers.geo_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = GEOProvider(mock_data_manager, geo_provider_config)

            assert provider.source == PublicationSource.GEO
            assert provider.supported_dataset_types == [DatasetType.GEO]

    def test_validate_geo_identifier(self, mock_data_manager, geo_provider_config):
        """Test GEO identifier validation."""
        with patch('lobster.tools.providers.geo_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = GEOProvider(mock_data_manager, geo_provider_config)

            # Valid GEO identifiers
            assert provider.validate_identifier("GSE123456") == True
            assert provider.validate_identifier("GDS123456") == True
            assert provider.validate_identifier("GPL123456") == True
            assert provider.validate_identifier("GSM123456") == True

            # Invalid identifiers
            assert provider.validate_identifier("") == False
            assert provider.validate_identifier("invalid") == False
            assert provider.validate_identifier("12345678") == False

    def test_search_geo_datasets(self, mock_data_manager, geo_provider_config, mock_geo_search_response):
        """Test GEO dataset search."""
        with patch('lobster.tools.providers.geo_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = GEOProvider(mock_data_manager, geo_provider_config)

            with patch.object(provider, 'search_geo_datasets') as mock_search, \
                 patch.object(provider, 'get_dataset_summaries') as mock_summaries:
                from lobster.tools.providers.geo_provider import GEOSearchResult

                mock_result = GEOSearchResult(
                    count=1,
                    ids=["200123456"],
                    web_env="MCID_GEO_123",
                    query_key="1"
                )
                mock_search.return_value = mock_result
                mock_summaries.return_value = [{
                    'uid': '200123456',
                    'accession': 'GSE123456',
                    'title': 'Test GEO Dataset',
                    'summary': 'Test description',
                    'taxon': 'Homo sapiens'
                }]

                result = provider.search_publications("single cell", max_results=10)

                assert "GEO DataSets Search Results" in result
                mock_data_manager.log_tool_usage.assert_called_once()

    def test_find_datasets_by_accession(self, mock_data_manager, geo_provider_config):
        """Test finding GEO datasets by accession."""
        with patch('lobster.tools.providers.geo_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = GEOProvider(mock_data_manager, geo_provider_config)

            mock_summary = {
                'uid': '200123456',
                'accession': 'GSE123456',
                'title': 'Test Dataset',
                'summary': 'Test description',
                'taxon': 'Homo sapiens',
                'n_samples': '20'
            }

            with patch.object(provider, 'search_geo_datasets') as mock_search, \
                 patch.object(provider, 'get_dataset_summaries') as mock_summaries:

                from lobster.tools.providers.geo_provider import GEOSearchResult
                mock_search.return_value = GEOSearchResult(count=1, ids=["200123456"])
                mock_summaries.return_value = [mock_summary]

                result = provider.find_datasets_from_publication("GSE123456")

                assert "GSE123456" in result
                assert "Test Dataset" in result
                mock_data_manager.log_tool_usage.assert_called_once()

    def test_extract_geo_metadata(self, mock_data_manager, geo_provider_config):
        """Test extracting GEO dataset metadata."""
        with patch('lobster.tools.providers.geo_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = GEOProvider(mock_data_manager, geo_provider_config)

            mock_summary = {
                'title': 'Test GEO Dataset',
                'summary': 'Test description',
                'PDAT': '2023/06/15',
                'GPL': 'GPL123456',
                'taxon': 'Homo sapiens'
            }

            with patch.object(provider, 'search_geo_datasets') as mock_search, \
                 patch.object(provider, 'get_dataset_summaries') as mock_summaries:

                from lobster.tools.providers.geo_provider import GEOSearchResult
                mock_search.return_value = GEOSearchResult(count=1, ids=["200123456"])
                mock_summaries.return_value = [mock_summary]

                metadata = provider.extract_publication_metadata("GSE123456")

                assert isinstance(metadata, PublicationMetadata)
                assert metadata.uid == "GSE123456"
                assert metadata.title == "Test GEO Dataset"
                assert metadata.journal == "Gene Expression Omnibus (GEO)"

    def test_geo_request_retry(self, mock_data_manager, geo_provider_config):
        """Test GEO request retry mechanism."""
        with patch('lobster.tools.providers.geo_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = GEOProvider(mock_data_manager, geo_provider_config)

            with patch('urllib.request.urlopen') as mock_urlopen:
                # Mock successful response after retry
                mock_response = Mock()
                mock_response.read.return_value = b'{"result": {"test": "data"}}'
                mock_response.__enter__ = Mock(return_value=mock_response)
                mock_response.__exit__ = Mock(return_value=None)

                mock_urlopen.side_effect = [
                    urllib.error.HTTPError("url", 429, "Rate Limited", {}, None),
                    mock_response
                ]

                result = provider._execute_request_with_retry("http://test.url")

                assert '{"result": {"test": "data"}}' in result
                assert mock_urlopen.call_count == 2


# ===============================================================================
# Error Handling and Integration Tests
# ===============================================================================

@pytest.mark.unit
class TestProvidersErrorHandling:
    """Test provider error handling and integration."""

    def test_pubmed_network_error_handling(self, mock_data_manager, pubmed_provider_config):
        """Test PubMed network error handling."""
        with patch('lobster.tools.providers.pubmed_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = PubMedProvider(mock_data_manager, pubmed_provider_config)

            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_urlopen.side_effect = urllib.error.URLError("Network error")

                with pytest.raises(Exception, match="Network error"):
                    provider._make_ncbi_request("http://test.url", "test")

    def test_geo_invalid_response_handling(self, mock_data_manager, geo_provider_config):
        """Test GEO invalid response handling."""
        with patch('lobster.tools.providers.geo_provider.get_settings') as mock_settings:
            mock_settings.return_value.NCBI_API_KEY = "test_key"

            provider = GEOProvider(mock_data_manager, geo_provider_config)

            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_response = Mock()
                mock_response.read.return_value = b'invalid json'
                mock_response.__enter__ = Mock(return_value=mock_response)
                mock_response.__exit__ = Mock(return_value=None)
                mock_urlopen.return_value = mock_response

                with pytest.raises(Exception):
                    provider.search_geo_datasets("test query")

    def test_provider_integration_workflow(self, mock_data_manager, pubmed_provider_config, geo_provider_config):
        """Test integrated workflow using both providers."""
        with patch('lobster.tools.providers.pubmed_provider.get_settings') as mock_pubmed_settings, \
             patch('lobster.tools.providers.geo_provider.get_settings') as mock_geo_settings:

            mock_pubmed_settings.return_value.NCBI_API_KEY = "test_key"
            mock_geo_settings.return_value.NCBI_API_KEY = "test_key"

            pubmed_provider = PubMedProvider(mock_data_manager, pubmed_provider_config)
            geo_provider = GEOProvider(mock_data_manager, geo_provider_config)

            # Mock PubMed search finding a paper with GEO dataset
            mock_article = {
                'uid': '37123456',
                'Title': 'Study with GSE123456',
                'Summary': 'This study used GSE123456 dataset.'
            }

            # Mock GEO dataset lookup
            mock_geo_summary = {
                'title': 'Corresponding GEO Dataset',
                'accession': 'GSE123456',
                'summary': 'Dataset from the paper'
            }

            with patch.object(pubmed_provider, '_load_with_params') as mock_pubmed_load, \
                 patch.object(pubmed_provider, '_extract_dataset_accessions') as mock_extract, \
                 patch.object(geo_provider, 'search_geo_datasets') as mock_geo_search, \
                 patch.object(geo_provider, 'get_dataset_summaries') as mock_geo_summaries:

                mock_pubmed_load.return_value = iter([mock_article])
                mock_extract.return_value = {'GEO': ['GSE123456']}

                from lobster.tools.providers.geo_provider import GEOSearchResult
                mock_geo_search.return_value = GEOSearchResult(count=1, ids=["200123456"])
                mock_geo_summaries.return_value = [mock_geo_summary]

                # Workflow: Search PubMed → Find datasets → Lookup in GEO
                pubmed_result = pubmed_provider.find_datasets_from_publication("37123456")
                geo_result = geo_provider.find_datasets_from_publication("GSE123456")

                assert "GSE123456" in pubmed_result
                assert "GSE123456" in geo_result
                assert "Corresponding GEO Dataset" in geo_result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])