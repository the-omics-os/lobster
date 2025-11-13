"""
Unit tests for PublicationResolver.

These tests use realistic scenarios with proper mocking of external API calls.
Tests are designed to run successfully with pytest.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from lobster.tools.providers.publication_resolver import (
    PublicationResolutionResult,
    PublicationResolver,
)


class TestPublicationResolverIdentifierParsing:
    """Test identifier parsing logic."""

    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = PublicationResolver()

    def test_parse_identifier_pmid_with_prefix(self):
        """Test parsing PMID with PMID: prefix."""
        pmid, doi = self.resolver._parse_identifier("PMID:12345678")
        assert pmid == "12345678"
        assert doi is None

    def test_parse_identifier_pmid_numeric(self):
        """Test parsing numeric PMID."""
        pmid, doi = self.resolver._parse_identifier("12345678")
        assert pmid == "12345678"
        assert doi is None

    def test_parse_identifier_doi(self):
        """Test parsing DOI."""
        pmid, doi = self.resolver._parse_identifier("10.1038/s41586-021-12345-6")
        assert pmid is None
        assert doi == "10.1038/s41586-021-12345-6"

    def test_parse_identifier_doi_from_url(self):
        """Test parsing DOI from doi.org URL."""
        pmid, doi = self.resolver._parse_identifier(
            "https://doi.org/10.1038/s41586-021-12345-6"
        )
        assert pmid is None
        assert doi == "10.1038/s41586-021-12345-6"


class TestPublicationResolverPMC:
    """Test PMC resolution with realistic API responses."""

    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = PublicationResolver()

    def test_resolve_via_pmc_success(self):
        """Test successful PMC resolution with realistic API response."""
        with patch.object(self.resolver.session, "get") as mock_get:
            # Simulate realistic PMC API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "linksets": [
                    {
                        "dbfrom": "pubmed",
                        "linksetdbs": [
                            {
                                "dbto": "pmc",
                                "linkname": "pubmed_pmc",
                                "links": ["7891011"],
                            }
                        ],
                    }
                ]
            }
            mock_get.return_value = mock_response

            result = self.resolver._resolve_via_pmc("12345678")

            assert result.is_accessible() is True
            assert result.source == "pmc"
            assert result.access_type == "open_access"
            assert "PMC7891011" in result.pdf_url
            assert "ncbi.nlm.nih.gov/pmc" in result.pdf_url

    def test_resolve_via_pmc_not_in_pmc(self):
        """Test PMC resolution when paper not in PMC."""
        with patch.object(self.resolver.session, "get") as mock_get:
            # Simulate PMC API response with no links
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {"linksets": []}
            mock_get.return_value = mock_response

            result = self.resolver._resolve_via_pmc("12345678")

            assert result.is_accessible() is False
            assert result.source == "pmc"
            assert result.access_type == "not_in_pmc"

    def test_resolve_via_pmc_network_error(self):
        """Test PMC resolution with network error."""
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_get.side_effect = Exception("Network timeout")

            result = self.resolver._resolve_via_pmc("12345678")

            assert result.is_accessible() is False
            assert result.source == "pmc"
            assert result.access_type == "error"

    def test_resolve_via_pmc_uses_correct_linkname(self):
        """Test that PMC resolution uses correct linkname parameter.

        Regression test for bug where pubmed_pmc_refs (returns citing articles)
        was used instead of pubmed_pmc (returns PMC version of article).

        The bug caused PMID:37963457 to incorrectly resolve to PMC12580505
        (an article that cites it, not the article itself).
        """
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {"linksets": []}
            mock_get.return_value = mock_response

            self.resolver._resolve_via_pmc("37963457")

            # Verify the correct elink URL was called
            called_url = mock_get.call_args[0][0]
            assert "linkname=pubmed_pmc" in called_url, (
                "Should use pubmed_pmc (PMC version of article), "
                "not pubmed_pmc_refs (articles that cite it)"
            )
            assert "linkname=pubmed_pmc_refs" not in called_url


class TestPublicationResolverPreprints:
    """Test preprint server resolution."""

    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = PublicationResolver()

    def test_resolve_via_preprint_biorxiv(self):
        """Test bioRxiv preprint resolution."""
        result = self.resolver._resolve_via_preprint_servers(
            "10.1101/2024.01.15.123456"
        )

        assert result.is_accessible() is True
        assert result.source == "biorxiv"
        assert result.access_type == "preprint"
        assert "biorxiv.org" in result.pdf_url
        assert "10.1101/2024.01.15.123456.full.pdf" in result.pdf_url

    def test_resolve_via_preprint_medrxiv(self):
        """Test medRxiv preprint resolution."""
        result = self.resolver._resolve_via_preprint_servers(
            "medrxiv.org/10.1101/2024.02.01.456789"
        )

        assert result.is_accessible() is True
        assert result.source == "medrxiv"
        assert result.access_type == "preprint"

    def test_resolve_via_preprint_not_preprint(self):
        """Test preprint resolution for non-preprint DOI."""
        result = self.resolver._resolve_via_preprint_servers(
            "10.1038/s41586-021-12345-6"
        )

        assert result.is_accessible() is False
        assert result.source == "preprint"
        assert result.access_type == "not_preprint"


class TestPublicationResolverSuggestions:
    """Test suggestion generation for paywalled papers."""

    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = PublicationResolver()

    def test_generate_suggestions_with_pmid(self):
        """Test generating suggestions with PMID."""
        result = self.resolver._generate_access_suggestions(
            "PMID:12345678", pmid="12345678", doi=None
        )

        assert result.source == "paywalled"
        assert result.access_type == "paywalled"
        assert "PubMed Central" in result.suggestions
        assert "bioRxiv" in result.suggestions
        assert "Author" in result.suggestions or "author" in result.suggestions.lower()
        assert len(result.alternative_urls) > 0

    def test_generate_suggestions_with_doi(self):
        """Test generating suggestions with DOI."""
        result = self.resolver._generate_access_suggestions(
            "10.1038/s41586-021-12345-6",
            pmid=None,
            doi="10.1038/s41586-021-12345-6",
        )

        assert result.source == "paywalled"
        assert result.access_type == "paywalled"
        assert (
            "Unpaywall" in result.suggestions
            or "unpaywall" in result.suggestions.lower()
        )
        assert "10.1038/s41586-021-12345-6" in result.suggestions

    def test_generate_suggestions_with_both(self):
        """Test generating suggestions with both PMID and DOI."""
        result = self.resolver._generate_access_suggestions(
            "PMID:12345678",
            pmid="12345678",
            doi="10.1038/s41586-021-12345-6",
        )

        assert result.source == "paywalled"
        assert len(result.alternative_urls) >= 3  # Should have multiple alternatives


class TestPublicationResolverFullWorkflow:
    """Test full resolution workflows."""

    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = PublicationResolver()

    def test_resolve_pmid_tries_pmc_first(self):
        """Test that PMID resolution prioritizes PMC."""
        with patch.object(self.resolver, "_resolve_via_pmc") as mock_pmc:
            mock_pmc.return_value = PublicationResolutionResult(
                identifier="PMID:12345678",
                pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123/pdf/",
                source="pmc",
                access_type="open_access",
            )

            result = self.resolver.resolve("PMID:12345678")

            assert result.is_accessible() is True
            assert result.source == "pmc"
            mock_pmc.assert_called_once_with("12345678")

    def test_resolve_doi_tries_preprint_after_pmc(self):
        """Test waterfall: PMC fails → tries preprint."""
        with (
            patch.object(self.resolver, "_resolve_via_pmc") as mock_pmc,
            patch.object(
                self.resolver, "_resolve_via_preprint_servers"
            ) as mock_preprint,
        ):

            # PMC fails
            mock_pmc.return_value = PublicationResolutionResult(
                identifier="10.1101/2024.01.001",
                source="pmc",
                access_type="not_in_pmc",
            )

            # Preprint succeeds
            mock_preprint.return_value = PublicationResolutionResult(
                identifier="10.1101/2024.01.001",
                pdf_url="https://biorxiv.org/content/10.1101/2024.01.001.full.pdf",
                source="biorxiv",
                access_type="preprint",
            )

            result = self.resolver.resolve("10.1101/2024.01.001")

            assert result.is_accessible() is True
            assert result.source == "biorxiv"

    def test_resolve_falls_back_to_suggestions(self):
        """Test full waterfall ending in suggestions."""
        with (
            patch.object(self.resolver, "_resolve_via_pmc") as mock_pmc,
            patch.object(self.resolver, "_resolve_via_linkout") as mock_linkout,
            patch.object(
                self.resolver, "_resolve_via_preprint_servers"
            ) as mock_preprint,
            patch.object(self.resolver, "_resolve_via_publisher") as mock_publisher,
        ):

            # All strategies fail
            mock_pmc.return_value = PublicationResolutionResult(
                identifier="PMID:123", source="pmc", access_type="not_in_pmc"
            )
            mock_linkout.return_value = PublicationResolutionResult(
                identifier="PMID:123", source="linkout", access_type="not_available"
            )
            mock_preprint.return_value = PublicationResolutionResult(
                identifier="10.1038/test", source="preprint", access_type="not_preprint"
            )
            mock_publisher.return_value = PublicationResolutionResult(
                identifier="10.1038/test",
                source="publisher",
                access_type="not_open_access",
            )

            result = self.resolver.resolve("PMID:123")

            assert result.is_accessible() is False
            assert result.source == "paywalled"
            assert len(result.suggestions) > 0


class TestPublicationResolverBatch:
    """Test batch resolution functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = PublicationResolver()

    def test_batch_resolve_respects_max_batch(self):
        """Test batch resolution respects max_batch limit."""
        identifiers = [f"PMID:{i}" for i in range(100, 120)]  # 20 identifiers

        with patch.object(self.resolver, "resolve") as mock_resolve:
            mock_resolve.return_value = PublicationResolutionResult(
                identifier="PMID:100",
                pdf_url="http://test.pdf",
                source="pmc",
                access_type="open_access",
            )

            results = self.resolver.batch_resolve(identifiers, max_batch=5)

            assert len(results) == 5
            assert mock_resolve.call_count == 5

    def test_batch_resolve_handles_individual_failures(self):
        """Test batch resolution handles individual failures gracefully."""
        identifiers = ["PMID:123", "PMID:456", "PMID:789"]

        with patch.object(self.resolver, "resolve") as mock_resolve:
            # Mix of success, failure, success
            mock_resolve.side_effect = [
                PublicationResolutionResult(
                    identifier="PMID:123",
                    pdf_url="http://test1.pdf",
                    source="pmc",
                    access_type="open_access",
                ),
                Exception("Network error"),
                PublicationResolutionResult(
                    identifier="PMID:789",
                    pdf_url="http://test3.pdf",
                    source="biorxiv",
                    access_type="preprint",
                ),
            ]

            results = self.resolver.batch_resolve(identifiers)

            assert len(results) == 3
            assert results[0].is_accessible() is True
            assert results[1].access_type == "error"
            assert results[2].is_accessible() is True

    def test_batch_resolve_empty_list(self):
        """Test batch resolution with empty list."""
        results = self.resolver.batch_resolve([])
        assert len(results) == 0


class TestPublicationResolutionResult:
    """Test PublicationResolutionResult data structure."""

    def test_result_to_dict(self):
        """Test result serialization to dict."""
        result = PublicationResolutionResult(
            identifier="PMID:12345678",
            pdf_url="http://test.com/paper.pdf",
            source="pmc",
            access_type="open_access",
            alternative_urls=["http://alt1.com", "http://alt2.com"],
            suggestions="Try these alternatives",
            metadata={"pmc_id": "PMC123"},
        )

        result_dict = result.to_dict()

        assert result_dict["identifier"] == "PMID:12345678"
        assert result_dict["pdf_url"] == "http://test.com/paper.pdf"
        assert result_dict["source"] == "pmc"
        assert result_dict["access_type"] == "open_access"
        assert len(result_dict["alternative_urls"]) == 2
        assert result_dict["suggestions"] == "Try these alternatives"
        assert result_dict["metadata"]["pmc_id"] == "PMC123"

    def test_is_accessible_true(self):
        """Test is_accessible returns True for accessible papers."""
        accessible = PublicationResolutionResult(
            identifier="PMID:123",
            pdf_url="http://test.com/paper.pdf",
            source="pmc",
            access_type="open_access",
        )
        assert accessible.is_accessible() is True

    def test_is_accessible_false_paywalled(self):
        """Test is_accessible returns False for paywalled papers."""
        paywalled = PublicationResolutionResult(
            identifier="PMID:456",
            pdf_url=None,
            source="paywalled",
            access_type="paywalled",
        )
        assert paywalled.is_accessible() is False

    def test_is_accessible_false_error(self):
        """Test is_accessible returns False for errors."""
        error = PublicationResolutionResult(
            identifier="PMID:789", pdf_url=None, source="error", access_type="error"
        )
        assert error.is_accessible() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
