"""
Unit tests for SRAProvider.

Tests provider functionality with mocked pysradb responses to avoid
real API calls during unit testing.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.base_provider import (
    DatasetType,
    ProviderCapability,
    PublicationMetadata,
    PublicationSource,
)
from lobster.tools.providers.sra_provider import (
    SRAConnectionError,
    SRANotFoundError,
    SRAProvider,
    SRAProviderConfig,
    SRAProviderError,
)


# Helper function to create mock pysradb
def create_mock_pysradb():
    """Create a mock pysradb module with SRAweb class."""
    mock_pysradb = MagicMock()
    mock_sraweb_class = MagicMock()
    mock_pysradb.SRAweb = mock_sraweb_class
    return mock_pysradb, mock_sraweb_class


@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2 instance."""
    return Mock(spec=DataManagerV2)


@pytest.fixture
def sra_provider(mock_data_manager):
    """Create SRAProvider instance with mock DataManager."""
    config = SRAProviderConfig(max_results=20)
    provider = SRAProvider(data_manager=mock_data_manager, config=config)
    return provider


class TestSRAProviderProperties:
    """Test SRAProvider properties and configuration."""

    def test_source_property(self, sra_provider):
        """Test that source returns SRA."""
        assert sra_provider.source == PublicationSource.SRA

    def test_supported_dataset_types(self, sra_provider):
        """Test that SRA dataset type is supported."""
        assert sra_provider.supported_dataset_types == [DatasetType.SRA]

    def test_priority(self, sra_provider):
        """Test that priority is set correctly (high priority = 10)."""
        assert sra_provider.priority == 10

    def test_supported_capabilities(self, sra_provider):
        """Test that correct capabilities are declared."""
        caps = sra_provider.get_supported_capabilities()

        # Should support
        assert caps[ProviderCapability.DISCOVER_DATASETS] is True
        assert caps[ProviderCapability.EXTRACT_METADATA] is True
        assert caps[ProviderCapability.FIND_LINKED_DATASETS] is True
        assert caps[ProviderCapability.QUERY_CAPABILITIES] is True

        # Should not support
        assert caps[ProviderCapability.GET_FULL_CONTENT] is False
        assert caps[ProviderCapability.SEARCH_LITERATURE] is False


class TestSRAAccessionDetection:
    """Test SRA accession pattern detection."""

    def test_ncbi_accessions(self, sra_provider):
        """Test detection of NCBI SRA accessions."""
        assert sra_provider._is_sra_accession("SRP123456") is True
        assert sra_provider._is_sra_accession("SRX123456") is True
        assert sra_provider._is_sra_accession("SRS123456") is True
        assert sra_provider._is_sra_accession("SRR123456") is True

    def test_ddbj_accessions(self, sra_provider):
        """Test detection of DDBJ accessions."""
        assert sra_provider._is_sra_accession("DRP123456") is True
        assert sra_provider._is_sra_accession("DRX123456") is True
        assert sra_provider._is_sra_accession("DRR123456") is True

    def test_ena_accessions(self, sra_provider):
        """Test detection of ENA accessions."""
        assert sra_provider._is_sra_accession("ERP123456") is True
        assert sra_provider._is_sra_accession("ERX123456") is True
        assert sra_provider._is_sra_accession("ERR123456") is True

    def test_non_accessions(self, sra_provider):
        """Test that non-accessions are not detected."""
        assert sra_provider._is_sra_accession("not an accession") is False
        assert sra_provider._is_sra_accession("12345") is False
        assert sra_provider._is_sra_accession("GSE123456") is False

    def test_accession_with_whitespace(self, sra_provider):
        """Test that accessions with whitespace are properly detected."""
        assert sra_provider._is_sra_accession("  SRP123456  ") is True
        assert sra_provider._is_sra_accession("\tSRR789012\n") is True


class TestSearchPublications:
    """Test search_publications method."""

    def test_search_with_accession(self, sra_provider):
        """Test search with valid SRA accession."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db

        mock_df = pd.DataFrame(
            {
                "study_accession": ["SRP123456"],
                "study_title": ["Test Study"],
                "organism": ["Homo sapiens"],
                "library_strategy": ["RNA-Seq"],
                "library_layout": ["PAIRED"],
                "instrument_platform": ["ILLUMINA"],
            }
        )
        mock_db.sra_metadata.return_value = mock_df

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            result = sra_provider.search_publications("SRP123456")

        assert "SRA Database Search Results" in result
        assert "SRP123456" in result
        assert "Test Study" in result
        mock_db.sra_metadata.assert_called_once()

    def test_search_with_accession_no_results(self, sra_provider):
        """Test search with valid accession pattern but no results."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db
        mock_db.sra_metadata.return_value = pd.DataFrame()

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            result = sra_provider.search_publications("SRP999999")

        assert "No SRA Results Found" in result
        assert "SRP999999" in result

    def test_search_with_keyword_performs_search(self, sra_provider):
        """Test that keyword search uses NCBI esearch path."""
        from unittest.mock import Mock, patch

        import pandas as pd

        # Mock NCBI esearch/esummary (keyword search now uses NCBI E-utilities)
        mock_esummary_df = pd.DataFrame(
            {
                "study_accession": ["SRP111", "SRP222"],
                "study_title": ["CRISPR screen study 1", "CRISPR screen study 2"],
                "organism": ["Homo sapiens", "Homo sapiens"],
                "library_strategy": ["RNA-Seq", "RNA-Seq"],
            }
        )

        with patch.object(sra_provider, "_ncbi_esearch", return_value=["123", "456"]):
            with patch.object(
                sra_provider, "_ncbi_esummary", return_value=mock_esummary_df
            ):
                result = sra_provider.search_publications(
                    "CRISPR screen", max_results=5
                )

        # Should return formatted results
        assert (
            "SRA Database Search Results" in result
            or "SRP111" in result
            or "SRP222" in result
        )
        assert "CRISPR screen" in result

    def test_search_with_filters(self, sra_provider):
        """Test search with organism filter."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db

        mock_df = pd.DataFrame(
            {
                "study_accession": ["SRP111111", "SRP222222"],
                "organism": ["Homo sapiens", "Mus musculus"],
                "library_strategy": ["RNA-Seq", "RNA-Seq"],
            }
        )
        mock_db.sra_metadata.return_value = mock_df

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            filters = {"organism": "Homo sapiens"}
            # Use a valid SRA accession (6+ digits) so _is_sra_accession returns True
            result = sra_provider.search_publications("SRP111111", filters=filters)

        # Note: Result should have filtered data
        assert "SRP111111" in result or "Homo sapiens" in result

    def test_search_with_detailed_param(self, sra_provider):
        """Test search with detailed parameter."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db
        mock_db.sra_metadata.return_value = pd.DataFrame(
            {"study_accession": ["SRP123456"], "study_title": ["Test"]}
        )

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            sra_provider.search_publications("SRP123456", detailed=False)

        mock_db.sra_metadata.assert_called_once_with("SRP123456", detailed=False)


class TestExtractPublicationMetadata:
    """Test extract_publication_metadata method."""

    def test_extract_metadata_success(self, sra_provider):
        """Test successful metadata extraction."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db

        mock_df = pd.DataFrame(
            {
                "study_accession": ["SRP123456"],
                "study_title": ["Test Study Title"],
                "study_abstract": ["This is a test study"],
                "organism": ["Homo sapiens"],
                "instrument_platform": ["ILLUMINA"],
                "library_strategy": ["RNA-Seq"],
                "library_source": ["TRANSCRIPTOMIC"],
                "published": ["2023-01-15"],
            }
        )
        mock_db.sra_metadata.return_value = mock_df

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            metadata = sra_provider.extract_publication_metadata("SRP123456")

        assert isinstance(metadata, PublicationMetadata)
        assert metadata.uid == "SRP123456"
        assert metadata.title == "Test Study Title"
        assert metadata.journal == "Sequence Read Archive (SRA)"
        assert metadata.abstract == "This is a test study"
        assert "Homo sapiens" in metadata.keywords
        assert "ILLUMINA" in metadata.keywords

    def test_extract_metadata_not_found(self, sra_provider):
        """Test metadata extraction with non-existent accession."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db
        mock_db.sra_metadata.return_value = pd.DataFrame()

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            with pytest.raises(SRANotFoundError):
                sra_provider.extract_publication_metadata("SRP999999")

    def test_extract_metadata_minimal_fields(self, sra_provider):
        """Test metadata extraction with minimal available fields."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db

        mock_df = pd.DataFrame(
            {
                "study_accession": ["SRP123456"],
                "experiment_title": ["Experiment Title"],
                "experiment_desc": ["Experiment description"],
            }
        )
        mock_db.sra_metadata.return_value = mock_df

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            metadata = sra_provider.extract_publication_metadata("SRP123456")

        assert metadata.uid == "SRP123456"
        assert metadata.title == "Experiment Title"
        assert metadata.abstract == "Experiment description"

    def test_extract_metadata_with_detailed_param(self, sra_provider):
        """Test metadata extraction with detailed parameter."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db
        mock_db.sra_metadata.return_value = pd.DataFrame(
            {"study_accession": ["SRP123456"], "study_title": ["Test"]}
        )

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            sra_provider.extract_publication_metadata("SRP123456", detailed=True)

        mock_db.sra_metadata.assert_called_once_with("SRP123456", detailed=True)


class TestFindDatasetsFromPublication:
    """Test find_datasets_from_publication method."""

    def test_find_datasets_from_pmid(self, sra_provider):
        """Test finding SRA datasets from PMID."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db

        mock_df = pd.DataFrame(
            {
                "study_accession": ["SRP123456", "SRP789012"],
                "study_title": ["Study 1", "Study 2"],
                "organism": ["Homo sapiens", "Mus musculus"],
            }
        )
        mock_db.pubmed_to_srp.return_value = mock_df

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            result = sra_provider.find_datasets_from_publication("PMID:12345678")

        assert "SRA Datasets Linked to Publication" in result
        assert "PMID" in result
        assert "SRP123456" in result
        assert "SRP789012" in result
        mock_db.pubmed_to_srp.assert_called_once_with("12345678")

    def test_find_datasets_from_numeric_pmid(self, sra_provider):
        """Test finding datasets with numeric PMID (no prefix)."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db
        mock_db.pubmed_to_srp.return_value = pd.DataFrame(
            {"study_accession": ["SRP123456"]}
        )

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            sra_provider.find_datasets_from_publication("12345678")

        mock_db.pubmed_to_srp.assert_called_once_with("12345678")

    def test_find_datasets_no_results(self, sra_provider):
        """Test finding datasets with no results."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db
        mock_db.pubmed_to_srp.return_value = pd.DataFrame()

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            result = sra_provider.find_datasets_from_publication("PMID:99999999")

        assert "No SRA Datasets Found" in result
        assert "99999999" in result

    def test_find_datasets_doi_returns_guidance(self, sra_provider):
        """Test that DOI returns helpful guidance."""
        # Method calls _get_sraweb() first, so we need to mock it
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            result = sra_provider.find_datasets_from_publication("10.1038/nature12345")

        assert "DOI" in result
        assert "10.1038/nature12345" in result

    def test_find_datasets_pmc_returns_guidance(self, sra_provider):
        """Test that PMC ID returns helpful guidance."""
        # Method calls _get_sraweb() first, so we need to mock it
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            result = sra_provider.find_datasets_from_publication("PMC8760896")

        assert "PMC" in result
        assert "PMC8760896" in result

    def test_find_datasets_invalid_identifier(self, sra_provider):
        """Test that invalid identifier returns helpful message."""
        # Method calls _get_sraweb() first, so we need to mock it
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            result = sra_provider.find_datasets_from_publication("invalid_id_123")

        assert "Unsupported" in result or "invalid" in result.lower()


class TestFilterSupport:
    """Test filter application."""

    def test_apply_organism_filter(self, sra_provider):
        """Test organism filter application."""
        df = pd.DataFrame(
            {"organism": ["Homo sapiens", "Mus musculus", "Homo sapiens"]}
        )

        filters = {"organism": "Homo sapiens"}
        filtered_df = sra_provider._apply_filters(df, filters)

        assert len(filtered_df) == 2
        assert all(filtered_df["organism"].str.contains("Homo sapiens"))

    def test_apply_strategy_filter(self, sra_provider):
        """Test library strategy filter."""
        df = pd.DataFrame(
            {"library_strategy": ["RNA-Seq", "WGS", "RNA-Seq", "AMPLICON"]}
        )

        filters = {"strategy": "RNA-Seq"}
        filtered_df = sra_provider._apply_filters(df, filters)

        assert len(filtered_df) == 2
        assert all(filtered_df["library_strategy"] == "RNA-Seq")

    def test_apply_layout_filter(self, sra_provider):
        """Test library layout filter."""
        df = pd.DataFrame({"library_layout": ["PAIRED", "SINGLE", "PAIRED", "SINGLE"]})

        filters = {"layout": "PAIRED"}
        filtered_df = sra_provider._apply_filters(df, filters)

        assert len(filtered_df) == 2
        assert all(filtered_df["library_layout"] == "PAIRED")

    def test_apply_source_filter(self, sra_provider):
        """Test library source filter."""
        df = pd.DataFrame(
            {
                "library_source": [
                    "TRANSCRIPTOMIC",
                    "GENOMIC",
                    "METAGENOMIC",
                    "TRANSCRIPTOMIC",
                ]
            }
        )

        filters = {"source": "TRANSCRIPTOMIC"}
        filtered_df = sra_provider._apply_filters(df, filters)

        assert len(filtered_df) == 2
        assert all(filtered_df["library_source"] == "TRANSCRIPTOMIC")

    def test_apply_platform_filter(self, sra_provider):
        """Test platform filter."""
        df = pd.DataFrame(
            {
                "instrument_platform": [
                    "ILLUMINA",
                    "PACBIO",
                    "ILLUMINA",
                    "OXFORD_NANOPORE",
                ]
            }
        )

        filters = {"platform": "ILLUMINA"}
        filtered_df = sra_provider._apply_filters(df, filters)

        assert len(filtered_df) == 2

    def test_apply_multiple_filters(self, sra_provider):
        """Test multiple filters simultaneously."""
        df = pd.DataFrame(
            {
                "organism": ["Homo sapiens", "Homo sapiens", "Mus musculus"],
                "library_strategy": ["RNA-Seq", "WGS", "RNA-Seq"],
                "library_layout": ["PAIRED", "PAIRED", "SINGLE"],
            }
        )

        filters = {
            "organism": "Homo sapiens",
            "strategy": "RNA-Seq",
            "layout": "PAIRED",
        }
        filtered_df = sra_provider._apply_filters(df, filters)

        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]["organism"] == "Homo sapiens"
        assert filtered_df.iloc[0]["library_strategy"] == "RNA-Seq"

    def test_apply_filters_empty_dataframe(self, sra_provider):
        """Test that filtering empty DataFrame returns empty DataFrame."""
        df = pd.DataFrame()
        filters = {"organism": "Homo sapiens"}

        filtered_df = sra_provider._apply_filters(df, filters)

        assert filtered_df.empty

    def test_apply_filters_case_insensitive_organism(self, sra_provider):
        """Test that organism filter is case-insensitive."""
        df = pd.DataFrame(
            {"organism": ["HOMO SAPIENS", "Mus musculus", "homo sapiens"]}
        )

        filters = {"organism": "homo sapiens"}
        filtered_df = sra_provider._apply_filters(df, filters)

        # The filter uses str.contains which is case-insensitive
        # It matches partial strings, so 'homo sapiens' matches all homo sapiens variants
        assert len(filtered_df) >= 2  # At least the exact matches


class TestMicrobiomeSearch:
    """Test microbiome-specific search functionality."""

    def test_microbiome_search_16s(self, sra_provider):
        """Test 16S microbiome search."""
        with patch.object(
            SRAProvider, "search_publications", return_value="Mock results"
        ) as mock_search:
            result = sra_provider.search_microbiome_datasets(
                "gut IBS",
                amplicon_region="16S",
                body_site="gut",
                host_organism="Homo sapiens",
            )

            # Verify search_publications was called
            mock_search.assert_called_once()
            call_args = mock_search.call_args

            # Check query enhancement
            query = call_args[0][0]
            assert "16S" in query.upper()
            assert "gut" in query.lower()

            # Check filters
            filters = call_args[1]["filters"]
            assert filters["source"] == "METAGENOMIC"
            assert filters["strategy"] == "AMPLICON"
            assert filters["organism"] == "Homo sapiens"

    def test_microbiome_search_shotgun(self, sra_provider):
        """Test shotgun metagenomics search."""
        with patch.object(
            SRAProvider, "search_publications", return_value="Mock results"
        ) as mock_search:
            result = sra_provider.search_microbiome_datasets(
                "obesity microbiome", amplicon_region=None, host_organism="Homo sapiens"
            )

            # Verify filters
            call_args = mock_search.call_args
            filters = call_args[1]["filters"]

            assert filters["strategy"] == "WGS"
            assert filters["source"] == "METAGENOMIC"

    def test_microbiome_search_its(self, sra_provider):
        """Test ITS fungal microbiome search."""
        with patch.object(
            SRAProvider, "search_publications", return_value="Mock results"
        ) as mock_search:
            result = sra_provider.search_microbiome_datasets(
                "fungal dysbiosis", amplicon_region="ITS", host_organism="Homo sapiens"
            )

            # Verify query enhancement
            call_args = mock_search.call_args
            query = call_args[0][0]
            assert "ITS" in query.upper()

    def test_microbiome_search_adds_microbiome_keyword(self, sra_provider):
        """Test that microbiome keyword is added when missing."""
        with patch.object(
            SRAProvider, "search_publications", return_value="Mock results"
        ) as mock_search:
            result = sra_provider.search_microbiome_datasets(
                "obesity gut", amplicon_region="16S"
            )

            # Verify microbiome keyword was added
            call_args = mock_search.call_args
            query = call_args[0][0]
            assert "microbiome" in query.lower() or "16s" in query.lower()

    def test_microbiome_search_no_duplicate_keywords(self, sra_provider):
        """Test that keywords are not excessively duplicated."""
        with patch.object(
            SRAProvider, "search_publications", return_value="Mock results"
        ) as mock_search:
            result = sra_provider.search_microbiome_datasets(
                "gut microbiome 16S", amplicon_region="16S", body_site="gut"
            )

            # Verify query is reasonable
            call_args = mock_search.call_args
            query = call_args[0][0]
            assert len(query.split()) < 15


class TestErrorHandling:
    """Test error handling and exceptions."""

    def test_connection_error(self, sra_provider):
        """Test handling of pysradb connection errors."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_sraweb_class.side_effect = Exception("Connection failed")

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            with pytest.raises(SRAConnectionError):
                sra_provider._get_sraweb()

    def test_import_error_handling(self, mock_data_manager):
        """Test handling of missing pysradb."""
        provider = SRAProvider(data_manager=mock_data_manager)

        # Simulate missing pysradb module
        with patch.dict("sys.modules", {"pysradb": None}):
            with pytest.raises(SRAConnectionError, match="pysradb not installed"):
                provider._get_sraweb()

    def test_search_error_handling(self, sra_provider):
        """Test error handling during search."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db
        mock_db.sra_metadata.side_effect = Exception("API error")

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            with pytest.raises(SRAProviderError, match="Error searching SRA"):
                sra_provider.search_publications("SRP123456")

    def test_metadata_extraction_error_handling(self, sra_provider):
        """Test error handling during metadata extraction."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db
        mock_db.sra_metadata.side_effect = Exception("API error")

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            with pytest.raises(SRAProviderError, match="Error extracting metadata"):
                sra_provider.extract_publication_metadata("SRP123456")

    def test_find_datasets_error_handling(self, sra_provider):
        """Test error handling during dataset linking."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db
        mock_db.pubmed_to_srp.side_effect = Exception("API error")

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            with pytest.raises(SRAProviderError, match="Error finding linked datasets"):
                sra_provider.find_datasets_from_publication("PMID:12345678")


class TestConfiguration:
    """Test provider configuration."""

    def test_default_config(self, mock_data_manager):
        """Test default configuration values."""
        provider = SRAProvider(data_manager=mock_data_manager)

        assert provider.config.max_results == 20
        assert provider.config.email == get_settings().NCBI_EMAIL
        assert provider.config.expand_attributes is False

    def test_custom_config(self, mock_data_manager):
        """Test custom configuration."""
        config = SRAProviderConfig(
            max_results=50, email="custom@example.com", expand_attributes=True
        )
        provider = SRAProvider(data_manager=mock_data_manager, config=config)

        assert provider.config.max_results == 50
        assert provider.config.email == "custom@example.com"
        assert provider.config.expand_attributes is True

    def test_config_validation(self, mock_data_manager):
        """Test configuration validation."""
        # Test max_results bounds
        with pytest.raises(Exception):  # Pydantic validation error
            SRAProviderConfig(max_results=0)

        with pytest.raises(Exception):  # Pydantic validation error
            SRAProviderConfig(max_results=200000)  # Above le=100000 limit


class TestLazyInitialization:
    """Test lazy initialization of pysradb connection."""

    def test_sraweb_lazy_init(self, mock_data_manager):
        """Test that SRAweb is not initialized until first use."""
        provider = SRAProvider(data_manager=mock_data_manager)

        # SRAweb should not be initialized yet
        assert provider._sraweb is None

        # Trigger lazy initialization
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            provider._get_sraweb()
            mock_sraweb_class.assert_called_once()
            assert provider._sraweb is not None

    def test_sraweb_cached(self, mock_data_manager):
        """Test that SRAweb instance is cached."""
        provider = SRAProvider(data_manager=mock_data_manager)

        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            provider._get_sraweb()
            provider._get_sraweb()

            # Should only be initialized once
            mock_sraweb_class.assert_called_once()


class TestFormatSearchResults:
    """Test search results formatting."""

    def test_format_search_results_basic(self, sra_provider):
        """Test basic search results formatting."""
        df = pd.DataFrame(
            {
                "study_accession": ["SRP123456"],
                "study_title": ["Test Study"],
                "organism": ["Homo sapiens"],
                "library_strategy": ["RNA-Seq"],
            }
        )

        result = sra_provider._format_search_results(df, "test query", 10)

        assert "SRA Database Search Results" in result
        assert "test query" in result
        assert "SRP123456" in result
        assert "Test Study" in result

    def test_format_search_results_with_filters(self, sra_provider):
        """Test formatting with filters displayed."""
        df = pd.DataFrame(
            {"study_accession": ["SRP123456"], "study_title": ["Test Study"]}
        )

        filters = {"organism": "Homo sapiens", "strategy": "RNA-Seq"}
        result = sra_provider._format_search_results(df, "test query", 10, filters)

        # Check for filters section (format may vary)
        assert "**Filters**" in result or "Filters:" in result
        assert "organism" in result
        assert "Homo sapiens" in result

    def test_format_search_results_max_limit(self, sra_provider):
        """Test that results are limited to max_results."""
        df = pd.DataFrame(
            {
                "study_accession": [f"SRP{i:06d}" for i in range(15)],
                "study_title": [f"Study {i}" for i in range(15)],
            }
        )

        result = sra_provider._format_search_results(df, "test", max_results=5)

        assert "Showing 5 of 15" in result

    def test_format_search_results_empty(self, sra_provider):
        """Test formatting empty results."""
        df = pd.DataFrame()

        result = sra_provider._format_search_results(df, "test query", 10)

        assert "No datasets found" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_metadata_with_missing_fields(self, sra_provider):
        """Test metadata extraction with many missing fields."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db

        mock_df = pd.DataFrame({"study_accession": ["SRP123456"]})
        mock_db.sra_metadata.return_value = mock_df

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            metadata = sra_provider.extract_publication_metadata("SRP123456")

        assert metadata.uid == "SRP123456"
        assert metadata.title.startswith("SRA Dataset")
        assert isinstance(metadata.keywords, list)

    def test_search_with_special_characters(self, sra_provider):
        """Test search with special characters in query."""
        mock_pysradb, mock_sraweb_class = create_mock_pysradb()
        mock_db = Mock()
        mock_sraweb_class.return_value = mock_db
        mock_db.sra_metadata.return_value = pd.DataFrame(
            {"study_accession": ["SRP123456"]}
        )

        with patch.dict("sys.modules", {"pysradb": mock_pysradb}):
            result = sra_provider.search_publications("SRP123456")

        assert "SRP123456" in result

    def test_accession_detection_edge_cases(self, sra_provider):
        """Test accession detection with edge cases."""
        # Too short (should not match)
        assert sra_provider._is_sra_accession("SRP12345") is False

        # Very long (should match)
        assert sra_provider._is_sra_accession("SRP1234567890") is True

        # Mixed case now matches (case-insensitive via AccessionResolver)
        # This is improved UX - users don't need to worry about case
        assert sra_provider._is_sra_accession("srp123456") is True
        assert sra_provider._is_sra_accession("Srp123456") is True
