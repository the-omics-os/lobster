"""
Comprehensive regression tests for Phase 1 publication resolver improvements.

This test suite specifically validates that Phase 1 changes (DOI resolution,
caching, LinkOut service, JSON validation) did not break existing functionality.

Tests cover:
- All consumer code paths (ResearchAgentAssistant, UnifiedContentService)
- Edge cases (invalid inputs, malformed responses, timeouts)
- Regression testing (previously working cases still work)
- Performance testing (caching effectiveness)
- Integration testing (full workflows)
"""

import time
from unittest.mock import Mock, patch

import pytest

from lobster.tools.providers.publication_resolver import (
    PublicationResolutionResult,
    PublicationResolver,
)


class TestPhase1RegressionSuite:
    """Comprehensive regression tests for Phase 1 improvements."""

    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = PublicationResolver()

    # ============================================================
    # Test 1: Consumer Code Paths - ResearchAgentAssistant Usage
    # ============================================================

    def test_research_agent_assistant_compatibility(self):
        """Test that ResearchAgentAssistant can still call resolver methods.

        ResearchAgentAssistant uses PublicationResolver.resolve() and expects
        PublicationResolutionResult objects. Phase 1 changes must preserve
        this interface.
        """
        # Simulate ResearchAgentAssistant calling resolve()
        with patch.object(self.resolver, "_resolve_via_pmc") as mock_pmc:
            mock_pmc.return_value = PublicationResolutionResult(
                identifier="PMID:12345678",
                pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
                source="pmc",
                access_type="open_access",
            )

            result = self.resolver.resolve("PMID:12345678")

            # Validate return type unchanged
            assert isinstance(result, PublicationResolutionResult)
            assert result.is_accessible() is True
            assert result.pdf_url is not None
            assert hasattr(result, "to_dict")  # Serialization method

    def test_batch_resolution_still_works(self):
        """Test batch_resolve() method unchanged by Phase 1.

        Research agent uses batch_resolve() for multiple papers.
        """
        with patch.object(self.resolver, "resolve") as mock_resolve:
            mock_resolve.return_value = PublicationResolutionResult(
                identifier="PMID:123",
                pdf_url="http://test.pdf",
                source="pmc",
                access_type="open_access",
            )

            results = self.resolver.batch_resolve(["PMID:123", "PMID:456"], max_batch=5)

            assert len(results) == 2
            assert all(isinstance(r, PublicationResolutionResult) for r in results)

    # ============================================================
    # Test 2: Edge Cases - Invalid Inputs
    # ============================================================

    def test_invalid_pmid_handling(self):
        """Test resolver handles invalid PMIDs gracefully."""
        with (
            patch.object(self.resolver, "_resolve_via_pmc") as mock_pmc,
            patch.object(self.resolver, "_resolve_via_linkout") as mock_linkout,
        ):
            # Simulate NCBI API returning error for invalid PMID
            mock_pmc.return_value = PublicationResolutionResult(
                identifier="PMID:INVALID", source="pmc", access_type="not_in_pmc"
            )
            mock_linkout.return_value = PublicationResolutionResult(
                identifier="PMID:INVALID", source="linkout", access_type="not_available"
            )

            result = self.resolver.resolve("PMID:INVALID")

            # Should fall through to suggestions
            assert result.source == "paywalled"
            assert len(result.suggestions) > 0

    def test_malformed_json_response_pmc(self):
        """Test PMC resolution handles malformed JSON gracefully.

        Phase 1 added JSON validation - ensure it catches malformed responses.
        """
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.side_effect = ValueError("Malformed JSON")
            mock_get.return_value = mock_response

            result = self.resolver._resolve_via_pmc("12345678")

            # Should return error result
            assert result.source == "pmc"
            assert result.access_type == "error"
            assert result.is_accessible() is False

    def test_malformed_json_response_publisher(self):
        """Test publisher resolution handles malformed JSON gracefully.

        Phase 1 added JSON validation - ensure it catches malformed responses.
        """
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.side_effect = ValueError("Malformed JSON")
            mock_get.return_value = mock_response

            result = self.resolver._resolve_via_publisher("10.1038/test")

            # Should return error result
            assert result.source == "publisher"
            assert result.access_type == "error"
            assert result.is_accessible() is False

    def test_unexpected_json_structure_pmc(self):
        """Test PMC handles unexpected JSON structure (non-dict response).

        Phase 1 added type checking - ensure it catches non-dict responses.
        """
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = ["unexpected", "list"]  # Not a dict
            mock_get.return_value = mock_response

            result = self.resolver._resolve_via_pmc("12345678")

            # Should return error result
            assert result.source == "pmc"
            assert result.access_type == "error"
            assert result.is_accessible() is False

    def test_invalid_pmc_id_type(self):
        """Test PMC handles invalid PMC ID types (non-numeric).

        Phase 1 added PMC ID validation - ensure it catches invalid types.
        """
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "linksets": [
                    {"linksetdbs": [{"links": ["INVALID_ID"]}]}  # Non-numeric PMC ID
                ]
            }
            mock_get.return_value = mock_response

            result = self.resolver._resolve_via_pmc("12345678")

            # Should return not_in_pmc (invalid ID treated as no ID)
            assert result.source == "pmc"
            assert result.access_type == "not_in_pmc"
            assert result.is_accessible() is False

    # ============================================================
    # Test 3: Regression Testing - Previously Working Cases
    # ============================================================

    def test_pmc_papers_still_resolve_to_pmc(self):
        """Test papers in PMC still resolve via PMC strategy.

        Critical regression: Phase 1 should not break PMC resolution.
        """
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "linksets": [
                    {"linksetdbs": [{"links": ["7891011"]}]}  # Valid numeric PMC ID
                ]
            }
            mock_get.return_value = mock_response

            result = self.resolver._resolve_via_pmc("12345678")

            # Should successfully resolve to PMC
            assert result.is_accessible() is True
            assert result.source == "pmc"
            assert result.access_type == "open_access"
            assert "PMC7891011" in result.pdf_url

    def test_biorxiv_papers_still_resolve_to_biorxiv(self):
        """Test bioRxiv papers still resolve via preprint strategy.

        Critical regression: Phase 1 should not break preprint resolution.
        """
        result = self.resolver._resolve_via_preprint_servers(
            "10.1101/2024.01.15.123456"
        )

        assert result.is_accessible() is True
        assert result.source == "biorxiv"
        assert result.access_type == "preprint"
        assert "biorxiv.org" in result.pdf_url

    def test_paywalled_papers_still_get_suggestions(self):
        """Test paywalled papers still receive helpful suggestions.

        Critical regression: Phase 1 should not break suggestion generation.
        """
        result = self.resolver._generate_access_suggestions(
            "PMID:12345678", pmid="12345678", doi="10.1038/test"
        )

        assert result.source == "paywalled"
        assert result.access_type == "paywalled"
        assert result.is_accessible() is False
        assert len(result.suggestions) > 0
        assert len(result.alternative_urls) > 0
        assert "PubMed Central" in result.suggestions

    def test_waterfall_strategy_order_preserved(self):
        """Test waterfall strategy still follows correct priority.

        Critical regression: PMC → LinkOut → DOI Resolution → Preprints → Publisher → Suggestions
        """
        with (
            patch.object(self.resolver, "_resolve_via_pmc") as mock_pmc,
            patch.object(self.resolver, "_resolve_via_linkout") as mock_linkout,
            patch.object(self.resolver, "_get_doi_from_pmid") as mock_get_doi,
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
            mock_get_doi.return_value = "10.1038/test"  # DOI found
            mock_preprint.return_value = PublicationResolutionResult(
                identifier="10.1038/test", source="preprint", access_type="not_preprint"
            )
            mock_publisher.return_value = PublicationResolutionResult(
                identifier="10.1038/test",
                source="publisher",
                access_type="not_open_access",
            )

            result = self.resolver.resolve("PMID:123")

            # Verify call order
            assert mock_pmc.called
            assert mock_linkout.called
            assert mock_get_doi.called  # New in Phase 1
            assert mock_preprint.called
            assert mock_publisher.called

            # Should end up with suggestions
            assert result.source == "paywalled"

    # ============================================================
    # Test 4: Performance Testing - Caching Effectiveness
    # ============================================================

    def test_caching_reduces_api_calls(self):
        """Test that caching reduces redundant API calls.

        Phase 1 added instance-level caching - verify it works.
        """
        with patch.object(self.resolver, "_resolve_via_pmc") as mock_pmc:
            mock_pmc.return_value = PublicationResolutionResult(
                identifier="PMID:12345678",
                pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
                source="pmc",
                access_type="open_access",
            )

            # First call - should hit API
            result1 = self.resolver.resolve("PMID:12345678")
            assert mock_pmc.call_count == 1

            # Second call - should use cache
            result2 = self.resolver.resolve("PMID:12345678")
            assert mock_pmc.call_count == 1  # No additional call

            # Results should be identical
            assert result1.pdf_url == result2.pdf_url
            assert result1.source == result2.source

    def test_cache_expiration_works(self):
        """Test that cache expires after TTL.

        Phase 1 caching has 300s TTL - verify expiration works.
        """
        # Create resolver with 1 second TTL for testing
        short_ttl_resolver = PublicationResolver(cache_ttl=1)

        with patch.object(short_ttl_resolver, "_resolve_via_pmc") as mock_pmc:
            mock_pmc.return_value = PublicationResolutionResult(
                identifier="PMID:12345678",
                pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
                source="pmc",
                access_type="open_access",
            )

            # First call - should hit API
            result1 = short_ttl_resolver.resolve("PMID:12345678")
            assert mock_pmc.call_count == 1

            # Wait for cache to expire
            time.sleep(1.1)

            # Second call - should hit API again (cache expired)
            result2 = short_ttl_resolver.resolve("PMID:12345678")
            assert mock_pmc.call_count == 2  # Additional call after expiration

    def test_cache_per_identifier(self):
        """Test that cache stores results per unique identifier.

        Phase 1 caching should cache each identifier separately.
        """
        with patch.object(self.resolver, "_resolve_via_pmc") as mock_pmc:
            mock_pmc.return_value = PublicationResolutionResult(
                identifier="PMID:12345678",
                pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
                source="pmc",
                access_type="open_access",
            )

            # Resolve two different PMIDs
            self.resolver.resolve("PMID:12345678")
            self.resolver.resolve("PMID:87654321")

            # Should have called API twice (different identifiers)
            assert mock_pmc.call_count == 2

            # Resolve first PMID again - should use cache
            self.resolver.resolve("PMID:12345678")
            assert mock_pmc.call_count == 2  # No additional call

    # ============================================================
    # Test 5: New Features - DOI Resolution and LinkOut
    # ============================================================

    def test_doi_resolution_enables_preprint_strategy(self):
        """Test DOI resolution unlocks preprint resolution for PMID-only inputs.

        Phase 1 feature: When only PMID provided, fetch DOI to enable preprints.
        """
        with (
            patch.object(self.resolver, "_resolve_via_pmc") as mock_pmc,
            patch.object(self.resolver, "_resolve_via_linkout") as mock_linkout,
            patch.object(self.resolver, "_get_doi_from_pmid") as mock_get_doi,
            patch.object(
                self.resolver, "_resolve_via_preprint_servers"
            ) as mock_preprint,
        ):
            # PMC and LinkOut fail
            mock_pmc.return_value = PublicationResolutionResult(
                identifier="PMID:123", source="pmc", access_type="not_in_pmc"
            )
            mock_linkout.return_value = PublicationResolutionResult(
                identifier="PMID:123", source="linkout", access_type="not_available"
            )

            # DOI resolution succeeds
            mock_get_doi.return_value = "10.1101/2024.01.001"

            # Preprint succeeds
            mock_preprint.return_value = PublicationResolutionResult(
                identifier="10.1101/2024.01.001",
                pdf_url="https://biorxiv.org/content/10.1101/2024.01.001.full.pdf",
                source="biorxiv",
                access_type="preprint",
            )

            result = self.resolver.resolve("PMID:123")

            # Should successfully resolve to preprint via DOI
            assert result.is_accessible() is True
            assert result.source == "biorxiv"
            assert mock_get_doi.called  # DOI resolution was used

    def test_linkout_provides_publisher_urls(self):
        """Test LinkOut service provides direct publisher URLs.

        Phase 1 feature: LinkOut adds publisher access before preprints.
        """
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "linksets": [
                    {
                        "idurllist": [
                            {
                                "objurls": [
                                    {
                                        "url": {
                                            "value": "https://www.cell.com/article/12345"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
            mock_get.return_value = mock_response

            result = self.resolver._resolve_via_linkout("37963457")

            # Should successfully resolve via LinkOut
            assert result.is_accessible() is True
            assert result.source == "linkout"
            assert result.access_type == "publisher"
            assert "cell.com" in result.pdf_url

    def test_linkout_handles_missing_urls(self):
        """Test LinkOut handles cases where no provider URL exists."""
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {"linksets": []}  # No URLs
            mock_get.return_value = mock_response

            result = self.resolver._resolve_via_linkout("12345678")

            # Should indicate not available
            assert result.is_accessible() is False
            assert result.source == "linkout"
            assert result.access_type == "not_available"

    # ============================================================
    # Test 6: Critical Bug Regression - PMID:37963457
    # ============================================================

    def test_original_bug_fix_verified(self):
        """Test that original bug (PMID:37963457 → PMC12580505) is fixed.

        CRITICAL REGRESSION: This bug must never return.

        Bug: Used pubmed_pmc_refs (citing articles) instead of pubmed_pmc.
        Fix: Changed linkname parameter to pubmed_pmc.
        """
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {"linksets": []}  # No PMC version
            mock_get.return_value = mock_response

            self.resolver._resolve_via_pmc("37963457")

            # Verify the correct elink URL was called
            called_url = mock_get.call_args[0][0]
            assert "linkname=pubmed_pmc" in called_url, (
                "REGRESSION: Should use pubmed_pmc (PMC version of article), "
                "not pubmed_pmc_refs (articles that cite it)"
            )
            assert (
                "linkname=pubmed_pmc_refs" not in called_url
            ), "REGRESSION: pubmed_pmc_refs linkname detected - original bug returned!"

    # ============================================================
    # Test 7: Network Error Handling
    # ============================================================

    def test_pmc_network_timeout_handling(self):
        """Test PMC resolution handles network timeouts gracefully."""
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_get.side_effect = Exception("Network timeout")

            result = self.resolver._resolve_via_pmc("12345678")

            # Should return error result
            assert result.source == "pmc"
            assert result.access_type == "error"
            assert result.is_accessible() is False

    def test_linkout_network_error_handling(self):
        """Test LinkOut resolution handles network errors gracefully."""
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            result = self.resolver._resolve_via_linkout("12345678")

            # Should return error result
            assert result.source == "linkout"
            assert result.access_type == "error"
            assert result.is_accessible() is False

    def test_doi_fetch_network_error_handling(self):
        """Test DOI fetching handles network errors gracefully."""
        with patch.object(self.resolver.session, "get") as mock_get:
            mock_get.side_effect = Exception("DNS resolution failed")

            result = self.resolver._get_doi_from_pmid("12345678")

            # Should return None (DOI not found)
            assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
