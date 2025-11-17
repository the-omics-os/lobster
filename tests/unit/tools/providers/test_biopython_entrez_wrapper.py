"""
Unit tests for Biopython Bio.Entrez wrapper.

Tests the wrapper utility that provides standardized NCBI API access
across all providers (SRA, PubMed, GEO).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from lobster.tools.providers.biopython_entrez_wrapper import (
    BioPythonEntrezWrapper,
    get_default_wrapper,
)


class TestBioPythonEntrezWrapperInit:
    """Test wrapper initialization."""

    @patch("lobster.tools.providers.biopython_entrez_wrapper._get_bio_entrez")
    @patch("lobster.tools.providers.biopython_entrez_wrapper.get_settings")
    def test_initialization_with_defaults(self, mock_settings, mock_get_entrez):
        """Test initialization uses settings when no params provided."""
        # Setup mocks
        mock_entrez = Mock()
        mock_get_entrez.return_value = mock_entrez

        mock_settings_obj = Mock()
        mock_settings_obj.NCBI_EMAIL = "test@example.com"
        mock_settings_obj.NCBI_API_KEY = "test-api-key"
        mock_settings.return_value = mock_settings_obj

        # Create wrapper
        wrapper = BioPythonEntrezWrapper()

        # Verify Bio.Entrez configured with settings values
        assert mock_entrez.email == "test@example.com"
        assert mock_entrez.api_key == "test-api-key"

    @patch("lobster.tools.providers.biopython_entrez_wrapper._get_bio_entrez")
    @patch("lobster.tools.providers.biopython_entrez_wrapper.get_settings")
    def test_initialization_with_explicit_params(self, mock_settings, mock_get_entrez):
        """Test initialization with explicit email and API key."""
        # Setup mocks
        mock_entrez = Mock()
        mock_get_entrez.return_value = mock_entrez

        mock_settings_obj = Mock()
        mock_settings_obj.NCBI_EMAIL = "default@example.com"
        mock_settings_obj.NCBI_API_KEY = "default-key"
        mock_settings.return_value = mock_settings_obj

        # Create wrapper with explicit params
        wrapper = BioPythonEntrezWrapper(
            email="custom@example.com",
            api_key="custom-key"
        )

        # Verify Bio.Entrez configured with explicit values
        assert mock_entrez.email == "custom@example.com"
        assert mock_entrez.api_key == "custom-key"

    @patch("lobster.tools.providers.biopython_entrez_wrapper._get_bio_entrez")
    @patch("lobster.tools.providers.biopython_entrez_wrapper.get_settings")
    def test_initialization_without_api_key(self, mock_settings, mock_get_entrez):
        """Test initialization without API key (3 req/s rate limit)."""
        # Setup mocks
        mock_entrez = Mock()
        mock_entrez.api_key = None
        mock_get_entrez.return_value = mock_entrez

        mock_settings_obj = Mock()
        mock_settings_obj.NCBI_EMAIL = "test@example.com"
        mock_settings_obj.NCBI_API_KEY = None
        mock_settings.return_value = mock_settings_obj

        # Create wrapper (should log 3 req/s rate limit message)
        wrapper = BioPythonEntrezWrapper()

        # Verify no API key set
        assert mock_entrez.api_key is None


class TestBioPythonEntrezWrapperEsearch:
    """Test esearch method."""

    @patch("lobster.tools.providers.biopython_entrez_wrapper._get_bio_entrez")
    @patch("lobster.tools.providers.biopython_entrez_wrapper.get_settings")
    def test_esearch_basic(self, mock_settings, mock_get_entrez):
        """Test basic esearch call."""
        # Setup mocks
        mock_entrez = Mock()
        mock_handle = Mock()
        mock_handle.read.return_value = {
            "Count": "100",
            "IdList": ["12345", "67890"],
            "QueryKey": "1",
            "WebEnv": "test-webenv"
        }
        mock_entrez.esearch.return_value = mock_handle
        mock_entrez.read.return_value = mock_handle.read.return_value
        mock_get_entrez.return_value = mock_entrez

        mock_settings_obj = Mock()
        mock_settings_obj.NCBI_EMAIL = "test@example.com"
        mock_settings_obj.NCBI_API_KEY = "test-key"
        mock_settings.return_value = mock_settings_obj

        # Create wrapper and call esearch
        wrapper = BioPythonEntrezWrapper()
        result = wrapper.esearch(db="sra", term="microbiome", retmax=20)

        # Verify esearch called with correct params
        mock_entrez.esearch.assert_called_once_with(
            db="sra",
            term="microbiome",
            retmax=20,
            retstart=0,
            usehistory="n"
        )

        # Verify result
        assert result["Count"] == "100"
        assert result["IdList"] == ["12345", "67890"]
        assert mock_handle.close.called

    @patch("lobster.tools.providers.biopython_entrez_wrapper._get_bio_entrez")
    @patch("lobster.tools.providers.biopython_entrez_wrapper.get_settings")
    def test_esearch_with_pagination(self, mock_settings, mock_get_entrez):
        """Test esearch with pagination parameters."""
        # Setup mocks
        mock_entrez = Mock()
        mock_handle = Mock()
        mock_handle.read.return_value = {
            "Count": "1000",
            "IdList": ["111", "222"],
        }
        mock_entrez.esearch.return_value = mock_handle
        mock_entrez.read.return_value = mock_handle.read.return_value
        mock_get_entrez.return_value = mock_entrez

        mock_settings_obj = Mock()
        mock_settings_obj.NCBI_EMAIL = "test@example.com"
        mock_settings_obj.NCBI_API_KEY = None
        mock_settings.return_value = mock_settings_obj

        # Create wrapper and call esearch with pagination
        wrapper = BioPythonEntrezWrapper()
        result = wrapper.esearch(db="sra", term="cancer", retmax=100, retstart=50)

        # Verify esearch called with pagination params
        mock_entrez.esearch.assert_called_once()
        call_args = mock_entrez.esearch.call_args
        assert call_args[1]["retmax"] == 100
        assert call_args[1]["retstart"] == 50


class TestBioPythonEntrezWrapperEsummary:
    """Test esummary method."""

    @patch("lobster.tools.providers.biopython_entrez_wrapper._get_bio_entrez")
    @patch("lobster.tools.providers.biopython_entrez_wrapper.get_settings")
    def test_esummary_single_id(self, mock_settings, mock_get_entrez):
        """Test esummary with single ID."""
        # Setup mocks
        mock_entrez = Mock()
        mock_handle = Mock()
        mock_summary = {"Id": "12345", "Title": "Test Dataset"}
        mock_handle.read.return_value = mock_summary
        mock_entrez.esummary.return_value = mock_handle
        mock_entrez.read.return_value = mock_summary
        mock_get_entrez.return_value = mock_entrez

        mock_settings_obj = Mock()
        mock_settings_obj.NCBI_EMAIL = "test@example.com"
        mock_settings_obj.NCBI_API_KEY = "test-key"
        mock_settings.return_value = mock_settings_obj

        # Create wrapper and call esummary
        wrapper = BioPythonEntrezWrapper()
        result = wrapper.esummary(db="sra", id="12345")

        # Verify esummary called
        mock_entrez.esummary.assert_called_once_with(db="sra", id="12345")

        # Verify result (should be list)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["Id"] == "12345"

    @patch("lobster.tools.providers.biopython_entrez_wrapper._get_bio_entrez")
    @patch("lobster.tools.providers.biopython_entrez_wrapper.get_settings")
    def test_esummary_multiple_ids(self, mock_settings, mock_get_entrez):
        """Test esummary with comma-separated IDs."""
        # Setup mocks
        mock_entrez = Mock()
        mock_handle = Mock()
        mock_summaries = [
            {"Id": "12345", "Title": "Dataset 1"},
            {"Id": "67890", "Title": "Dataset 2"}
        ]
        mock_handle.read.return_value = mock_summaries
        mock_entrez.esummary.return_value = mock_handle
        mock_entrez.read.return_value = mock_summaries
        mock_get_entrez.return_value = mock_entrez

        mock_settings_obj = Mock()
        mock_settings_obj.NCBI_EMAIL = "test@example.com"
        mock_settings_obj.NCBI_API_KEY = None
        mock_settings.return_value = mock_settings_obj

        # Create wrapper and call esummary
        wrapper = BioPythonEntrezWrapper()
        result = wrapper.esummary(db="sra", id="12345,67890")

        # Verify esummary called with comma-separated IDs
        mock_entrez.esummary.assert_called_once_with(db="sra", id="12345,67890")

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 2


class TestBioPythonEntrezWrapperEfetch:
    """Test efetch method."""

    @patch("lobster.tools.providers.biopython_entrez_wrapper._get_bio_entrez")
    @patch("lobster.tools.providers.biopython_entrez_wrapper.get_settings")
    def test_efetch_xml(self, mock_settings, mock_get_entrez):
        """Test efetch for XML content."""
        # Setup mocks
        mock_entrez = Mock()
        mock_handle = Mock()
        mock_xml = b"<root><data>test</data></root>"
        mock_handle.read.return_value = mock_xml
        mock_entrez.efetch.return_value = mock_handle
        mock_get_entrez.return_value = mock_entrez

        mock_settings_obj = Mock()
        mock_settings_obj.NCBI_EMAIL = "test@example.com"
        mock_settings_obj.NCBI_API_KEY = "test-key"
        mock_settings.return_value = mock_settings_obj

        # Create wrapper and call efetch
        wrapper = BioPythonEntrezWrapper()
        result = wrapper.efetch(db="sra", id="12345", rettype="xml")

        # Verify efetch called
        mock_entrez.efetch.assert_called_once_with(
            db="sra",
            id="12345",
            rettype="xml",
            retmode="xml"
        )

        # Verify result
        assert result == mock_xml
        assert mock_handle.close.called


class TestBioPythonEntrezWrapperElink:
    """Test elink method."""

    @patch("lobster.tools.providers.biopython_entrez_wrapper._get_bio_entrez")
    @patch("lobster.tools.providers.biopython_entrez_wrapper.get_settings")
    def test_elink_basic(self, mock_settings, mock_get_entrez):
        """Test elink to find linked records."""
        # Setup mocks
        mock_entrez = Mock()
        mock_handle = Mock()
        mock_links = [
            {
                "DbFrom": "gds",
                "LinkSetDb": [
                    {
                        "DbTo": "pubmed",
                        "Link": [{"Id": "111"}, {"Id": "222"}]
                    }
                ]
            }
        ]
        mock_handle.read.return_value = mock_links
        mock_entrez.elink.return_value = mock_handle
        mock_entrez.read.return_value = mock_links
        mock_get_entrez.return_value = mock_entrez

        mock_settings_obj = Mock()
        mock_settings_obj.NCBI_EMAIL = "test@example.com"
        mock_settings_obj.NCBI_API_KEY = None
        mock_settings.return_value = mock_settings_obj

        # Create wrapper and call elink
        wrapper = BioPythonEntrezWrapper()
        result = wrapper.elink(dbfrom="gds", db="pubmed", id="12345")

        # Verify elink called
        mock_entrez.elink.assert_called_once_with(
            dbfrom="gds",
            db="pubmed",
            id="12345"
        )

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 1


class TestDefaultWrapper:
    """Test module-level convenience instance."""

    @patch("lobster.tools.providers.biopython_entrez_wrapper.BioPythonEntrezWrapper")
    def test_get_default_wrapper_singleton(self, mock_wrapper_class):
        """Test that get_default_wrapper returns singleton instance."""
        # Reset singleton
        import lobster.tools.providers.biopython_entrez_wrapper as wrapper_module
        wrapper_module._default_wrapper = None

        # Create mock wrapper
        mock_instance = Mock()
        mock_wrapper_class.return_value = mock_instance

        # Get default wrapper twice
        wrapper1 = get_default_wrapper()
        wrapper2 = get_default_wrapper()

        # Verify same instance returned
        assert wrapper1 is wrapper2

        # Verify wrapper created only once
        assert mock_wrapper_class.call_count == 1


class TestErrorHandling:
    """Test error handling in wrapper methods."""

    @patch("lobster.tools.providers.biopython_entrez_wrapper._get_bio_entrez")
    @patch("lobster.tools.providers.biopython_entrez_wrapper.get_settings")
    def test_esearch_error_handling(self, mock_settings, mock_get_entrez):
        """Test that esearch errors are propagated."""
        # Setup mocks to raise error
        mock_entrez = Mock()
        mock_entrez.esearch.side_effect = Exception("NCBI API error")
        mock_get_entrez.return_value = mock_entrez

        mock_settings_obj = Mock()
        mock_settings_obj.NCBI_EMAIL = "test@example.com"
        mock_settings_obj.NCBI_API_KEY = None
        mock_settings.return_value = mock_settings_obj

        # Create wrapper and expect error
        wrapper = BioPythonEntrezWrapper()

        with pytest.raises(Exception) as exc_info:
            wrapper.esearch(db="sra", term="invalid query")

        assert "NCBI API error" in str(exc_info.value)


class TestLazyImport:
    """Test lazy import of Bio.Entrez."""

    @patch("lobster.tools.providers.biopython_entrez_wrapper._Bio_Entrez", None)
    def test_lazy_import_on_first_use(self):
        """Test that Bio.Entrez is imported lazily."""
        # Reset global
        import lobster.tools.providers.biopython_entrez_wrapper as wrapper_module
        wrapper_module._Bio_Entrez = None

        # Import should happen on first _get_bio_entrez call
        with patch("lobster.tools.providers.biopython_entrez_wrapper.Entrez", create=True) as mock_entrez_import:
            from lobster.tools.providers.biopython_entrez_wrapper import _get_bio_entrez

            # Reset again for clean test
            wrapper_module._Bio_Entrez = None

            # First call should import
            entrez1 = _get_bio_entrez()

            # Second call should use cached import
            entrez2 = _get_bio_entrez()

            # Verify same instance
            assert entrez1 is entrez2
