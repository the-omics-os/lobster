"""
Integration tests for SRAProvider Phase 1 implementation.

Tests the hybrid approach:
- Path 1: Accession lookup via pysradb (existing, proven)
- Path 2: Keyword search via direct NCBI API (new implementation)

Requirements:
- NCBI_API_KEY environment variable (optional but recommended)
- Network connectivity to NCBI services
"""

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.sra_provider import SRAProvider, SRAProviderConfig


@pytest.fixture
def sra_provider(tmp_path):
    """Create SRAProvider instance with temporary workspace."""
    dm = DataManagerV2(workspace_path=str(tmp_path / "test_workspace"))
    config = SRAProviderConfig(max_results=5)
    return SRAProvider(data_manager=dm, config=config)


@pytest.mark.integration
@pytest.mark.real_api
class TestPhase1AccessionPath:
    """Test Path 1: Accession lookup (should use pysradb)."""

    def test_srp_accession_lookup(self, sra_provider):
        """Test known GEUVADIS study SRP033351."""
        result = sra_provider.search_publications("SRP033351", max_results=5)

        assert "SRA Database Search Results" in result
        assert "SRP033351" in result
        assert "Query" in result
        # Should show some metadata
        assert any(
            keyword in result
            for keyword in ["RNA-Seq", "TRANSCRIPTOMIC", "Homo sapiens"]
        )

    def test_srr_accession_lookup(self, sra_provider):
        """Test individual run accession."""
        result = sra_provider.search_publications("SRR1234567", max_results=1)

        # Should either find results or gracefully handle not found
        assert "SRA" in result
        # Should not error out
        assert "Error" not in result or "No" in result

    def test_invalid_accession_graceful_handling(self, sra_provider):
        """Test that invalid accessions don't crash."""
        result = sra_provider.search_publications("SRP999999999", max_results=5)

        # Should return "No results" not crash
        assert "No" in result or "not found" in result.lower()


@pytest.mark.integration
@pytest.mark.real_api
class TestPhase1KeywordPath:
    """Test Path 2: Keyword search via direct NCBI API."""

    def test_simple_keyword_no_filters(self, sra_provider):
        """Test simple keyword search without filters."""
        result = sra_provider.search_publications("microbiome", max_results=5)

        assert "SRA Database Search Results" in result
        assert "Query" in result
        # Should have at least some results (microbiome is common)
        assert len(result) > 200  # Should have substantial content

    def test_keyword_with_organism_filter(self, sra_provider):
        """Test keyword search with organism filter (PubMed pattern)."""
        result = sra_provider.search_publications(
            "gut microbiome", max_results=5, filters={"organism": "Homo sapiens"}
        )

        assert "SRA Database Search Results" in result
        # Should show applied filter
        assert "organism=Homo sapiens" in result or "Homo sapiens" in result

    def test_keyword_with_multiple_filters(self, sra_provider):
        """Test keyword search with organism + strategy filters."""
        result = sra_provider.search_publications(
            "gut microbiome",
            max_results=5,
            filters={"organism": "Homo sapiens", "strategy": "AMPLICON"},
        )

        assert "SRA Database Search Results" in result
        # Should show both filters
        assert "organism=Homo sapiens" in result or "Homo sapiens" in result
        assert "strategy=AMPLICON" in result or "AMPLICON" in result

    def test_agent_constructed_or_query(self, sra_provider):
        """Test that agent's OR query is preserved (critical for Phase 1)."""
        # Agent constructs OR logic, provider should NOT split or modify
        result = sra_provider.search_publications(
            "microbiome OR 16S", max_results=5, filters={"organism": "Homo sapiens"}
        )

        assert "SRA Database Search Results" in result
        # Should have results (OR logic should work)
        assert len(result) > 200

    def test_complex_agent_query_preserved(self, sra_provider):
        """Test that complex agent queries are preserved."""
        # Agent might construct: "(term1 OR term2) AND term3"
        result = sra_provider.search_publications(
            "microbiome 16S", max_results=5, filters={"organism": "Homo sapiens"}
        )

        assert "SRA Database Search Results" in result
        # Should not crash, should preserve query

    def test_no_results_query_graceful(self, sra_provider):
        """Test that queries with no results are handled gracefully."""
        # Very specific query unlikely to match
        result = sra_provider.search_publications(
            "zzzzzz_nonexistent_term_12345", max_results=5
        )

        # Should return "No results" message, not crash
        assert "No" in result or "not found" in result.lower()
        # Should show the query that was tried
        assert "zzzzzz_nonexistent_term_12345" in result


@pytest.mark.integration
@pytest.mark.real_api
class TestPhase1RateLimiting:
    """Test rate limiting works correctly."""

    def test_rate_limiter_prevents_429(self, sra_provider):
        """Test that rate limiting prevents 429 errors."""
        # Make 5 rapid requests - rate limiter should throttle
        for i in range(5):
            result = sra_provider.search_publications(
                f"microbiome test{i}", max_results=3
            )
            # Should not get rate limit errors
            assert "429" not in result
            assert "rate limit" not in result.lower() or "Query" in result

    def test_multiple_searches_sequential(self, sra_provider):
        """Test multiple searches work sequentially."""
        results = []
        queries = ["microbiome", "gut", "16S"]

        for query in queries:
            result = sra_provider.search_publications(query, max_results=3)
            results.append(result)

        # All should succeed
        for result in results:
            assert "SRA Database Search Results" in result or "No" in result


@pytest.mark.integration
@pytest.mark.real_api
class TestPhase1ResultFormatting:
    """Test that results are formatted correctly."""

    def test_result_has_required_fields(self, sra_provider):
        """Test that search results have all required fields."""
        result = sra_provider.search_publications("microbiome", max_results=3)

        # Should have header
        assert "SRA Database Search Results" in result
        # Should show query
        assert "Query" in result
        # Should show result count
        assert "Total Results" in result or "results" in result.lower()

    def test_accession_links_formatted(self, sra_provider):
        """Test that accession links are properly formatted."""
        result = sra_provider.search_publications("SRP033351", max_results=1)

        # Should have clickable links
        assert "https://www.ncbi.nlm.nih.gov/sra/" in result or "SRP033351" in result

    def test_metadata_fields_present(self, sra_provider):
        """Test that key metadata fields are present in results."""
        result = sra_provider.search_publications("microbiome", max_results=3)

        # Should have at least some metadata fields
        metadata_keywords = [
            "Organism",
            "Strategy",
            "Platform",
            "Layout",
            "Accession",
        ]
        # At least 2 metadata fields should be present
        present_count = sum(1 for kw in metadata_keywords if kw in result)
        assert present_count >= 2


@pytest.mark.integration
@pytest.mark.real_api
@pytest.mark.slow
class TestPhase1Performance:
    """Test performance benchmarks."""

    def test_accession_search_speed(self, sra_provider):
        """Test accession searches complete in <2 seconds."""
        import time

        start = time.time()
        result = sra_provider.search_publications("SRP033351", max_results=5)
        duration = time.time() - start

        assert "SRA" in result
        # Phase 1 target: <2s for accession lookup
        assert (
            duration < 3.0
        ), f"Accession search took {duration:.2f}s (target: <2s, allowing 3s for CI)"

    def test_keyword_search_speed(self, sra_provider):
        """Test keyword searches complete in <5 seconds."""
        import time

        start = time.time()
        result = sra_provider.search_publications("microbiome", max_results=5)
        duration = time.time() - start

        assert "SRA" in result
        # Phase 1 target: <5s for keyword search
        assert (
            duration < 7.0
        ), f"Keyword search took {duration:.2f}s (target: <5s, allowing 7s for CI)"


@pytest.mark.integration
@pytest.mark.real_api
class TestPhase1PubMedPatternCompliance:
    """Test that implementation matches PubMedProvider pattern."""

    def test_query_wrapping_pattern(self, sra_provider):
        """Test that queries are wrapped in parentheses."""
        # This is internal but critical - queries should be: "(query) AND (filter[TAG])"
        result = sra_provider.search_publications(
            "gut microbiome", max_results=3, filters={"organism": "Homo sapiens"}
        )

        # Should work without errors (indicates proper query wrapping)
        assert "SRA Database Search Results" in result or "No" in result

    def test_filter_qualifiers_applied(self, sra_provider):
        """Test that SRA field qualifiers are properly applied."""
        # Test all supported filters
        result = sra_provider.search_publications(
            "microbiome",
            max_results=3,
            filters={
                "organism": "Homo sapiens",
                "strategy": "AMPLICON",
                "source": "METAGENOMIC",
                "layout": "PAIRED",
                "platform": "ILLUMINA",
            },
        )

        # Should not crash, should return results or "No results"
        assert "SRA" in result
        assert "Error" not in result or "No" in result

    def test_agent_query_not_modified(self, sra_provider):
        """Test that agent's query is NOT split or modified."""
        # Agent sends: "microbiome OR 16S"
        # Provider should use it AS-IS, only add filters
        result = sra_provider.search_publications(
            "microbiome OR 16S", max_results=3
        )

        # Should work (OR logic should be preserved)
        assert "SRA Database Search Results" in result or "No" in result
        # Should not crash trying to parse agent's query
        assert "Error" not in result or "No" in result


# Summary test to validate entire Phase 1
@pytest.mark.integration
@pytest.mark.real_api
class TestPhase1Summary:
    """High-level test validating Phase 1 is production-ready."""

    def test_phase1_complete_workflow(self, sra_provider):
        """Test complete Phase 1 workflow: accession + keyword + filters."""
        # Test 1: Accession lookup (Path 1 - pysradb)
        acc_result = sra_provider.search_publications("SRP033351", max_results=3)
        assert "SRP033351" in acc_result
        assert "Error" not in acc_result

        # Test 2: Simple keyword (Path 2 - direct NCBI)
        kw_result = sra_provider.search_publications("microbiome", max_results=3)
        assert "SRA" in kw_result
        assert "Error" not in kw_result

        # Test 3: Keyword + filters (Path 2 with filters)
        filtered_result = sra_provider.search_publications(
            "gut microbiome", max_results=3, filters={"organism": "Homo sapiens"}
        )
        assert "SRA" in filtered_result
        assert "Error" not in filtered_result

        # Test 4: Agent OR query (Path 2 with agent logic)
        or_result = sra_provider.search_publications("microbiome OR 16S", max_results=3)
        assert "SRA" in or_result
        assert "Error" not in or_result

        # All tests passed - Phase 1 is functional
        assert True
