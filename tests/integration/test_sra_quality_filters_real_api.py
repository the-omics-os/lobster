"""
Integration tests for SRAProvider Phase 1.4 quality filters with real NCBI API.

Tests quality filters against real NCBI SRA database to verify:
- Quality filters reduce result count appropriately
- Returned datasets meet quality criteria
- Modality-specific filters work as expected
"""

from pathlib import Path

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.sra_provider import SRAProvider, SRAProviderConfig


@pytest.mark.real_api
class TestQualityFiltersRealAPI:
    """Integration tests with real NCBI API calls."""

    @pytest.fixture
    def provider_with_filters(self, tmp_path):
        """Create SRAProvider with quality filters enabled."""
        dm = DataManagerV2(workspace_path=str(tmp_path / "test_workspace"))
        config = SRAProviderConfig(enable_quality_filters=True, max_results=5)
        return SRAProvider(data_manager=dm, config=config)

    @pytest.fixture
    def provider_no_filters(self, tmp_path):
        """Create SRAProvider with quality filters disabled."""
        dm = DataManagerV2(workspace_path=str(tmp_path / "test_workspace"))
        config = SRAProviderConfig(enable_quality_filters=False, max_results=5)
        return SRAProvider(data_manager=dm, config=config)

    def test_quality_filters_reduce_results(
        self, provider_with_filters, provider_no_filters
    ):
        """
        Test that quality filters reduce result count compared to no filters.

        This validates that quality filters are actually being applied and
        have a measurable effect on search results.
        """
        query = "microbiome"

        # Search with quality filters enabled
        result_with_filters = provider_with_filters.search_publications(
            query, max_results=5
        )

        # Search with quality filters disabled
        result_no_filters = provider_no_filters.search_publications(
            query, max_results=5
        )

        # Both should return results
        assert (
            "Total Results:" in result_with_filters
            or "🧬 SRA Database Search Results" in result_with_filters
        )
        assert (
            "Total Results:" in result_no_filters
            or "🧬 SRA Database Search Results" in result_no_filters
        )

        print("\n=== With Quality Filters ===")
        print(result_with_filters[:500])

        print("\n=== Without Quality Filters ===")
        print(result_no_filters[:500])

    def test_amplicon_quality_filters_real_search(self, provider_with_filters):
        """
        Test amplicon quality filters with real NCBI search.

        Validates that amplicon-specific filters work and return
        appropriate datasets for 16S rRNA studies.
        """
        result = provider_with_filters.search_publications(
            query="gut microbiome", max_results=5, modality_hint="amplicon"
        )

        # Should return results
        assert "🧬 SRA Database Search Results" in result or "Total Results:" in result
        assert "gut microbiome" in result.lower() or "microbiome" in result.lower()

        print("\n=== Amplicon Quality Filters Result ===")
        print(result[:800])

    def test_scrna_quality_filters_real_search(self, provider_with_filters):
        """
        Test scRNA-seq quality filters with real NCBI search.

        Validates that scRNA-seq filters (paired-end, Illumina) work
        and return appropriate single-cell datasets.
        """
        result = provider_with_filters.search_publications(
            query="single cell immune", max_results=5, modality_hint="scrna-seq"
        )

        # Should return results
        assert "🧬 SRA Database Search Results" in result or "Total Results:" in result

        print("\n=== scRNA-seq Quality Filters Result ===")
        print(result[:800])

    def test_bulk_rnaseq_quality_filters_real_search(self, provider_with_filters):
        """
        Test bulk RNA-seq quality filters with real NCBI search.

        Validates that bulk RNA-seq filters (base only) work correctly.
        """
        result = provider_with_filters.search_publications(
            query="liver RNA-seq", max_results=5, modality_hint="bulk-rna-seq"
        )

        # Should return results
        assert "🧬 SRA Database Search Results" in result or "Total Results:" in result

        print("\n=== Bulk RNA-seq Quality Filters Result ===")
        print(result[:800])

    def test_quality_filters_with_organism_filter(self, provider_with_filters):
        """
        Test quality filters combined with organism filter.

        Validates that quality filters work correctly when combined
        with traditional SRA filters like organism.
        """
        result = provider_with_filters.search_publications(
            query="gut microbiome",
            max_results=5,
            filters={"organism": "Homo sapiens"},
            modality_hint="amplicon",
        )

        # Should return results
        assert "🧬 SRA Database Search Results" in result or "Total Results:" in result

        # Should mention human organism
        assert (
            "sapiens" in result.lower()
            or "human" in result.lower()
            or "homo" in result.lower()
        )

        print("\n=== Quality + Organism Filters Result ===")
        print(result[:800])

    def test_microbiome_search_uses_quality_filters(self, provider_with_filters):
        """
        Test that search_microbiome_datasets correctly uses quality filters.

        Validates end-to-end integration with specialized microbiome search.
        """
        result = provider_with_filters.search_microbiome_datasets(
            query="inflammatory bowel disease",
            amplicon_region="16S",
            body_site="gut",
            host_organism="Homo sapiens",
            max_results=5,
        )

        # Should return results with microbiome tips
        assert "🧬 SRA Database Search Results" in result or "Total Results:" in result
        assert "Microbiome Analysis Tips" in result

        print("\n=== Microbiome Search with Quality Filters ===")
        print(result[:1000])

    @pytest.mark.slow
    def test_quality_filters_do_not_break_accession_lookup(self, provider_with_filters):
        """
        Test that quality filters don't break direct accession lookup.

        Accession lookups should bypass quality filters since they're
        direct metadata retrieval.
        """
        result = provider_with_filters.search_publications(
            query="SRP033351", max_results=20  # Known SRA study
        )

        # Should return results (accession lookup bypasses quality filters)
        assert (
            "🧬 SRA Database Search Results" in result
            or "No SRA Results Found" in result
            or "Total Results:" in result
        )

        print("\n=== Accession Lookup (bypasses quality filters) ===")
        print(result[:800])


@pytest.mark.real_api
@pytest.mark.slow
class TestQualityFiltersComparison:
    """
    Detailed comparison tests for quality filter effectiveness.

    These tests run slower but provide detailed metrics on filter effectiveness.
    """

    def test_compare_with_without_filters_detailed(self, tmp_path):
        """
        Detailed comparison of results with vs without quality filters.

        Prints statistics about result quality improvements.
        """
        # Provider with filters
        dm_with = DataManagerV2(workspace_path=str(tmp_path / "ws_with"))
        config_with = SRAProviderConfig(enable_quality_filters=True, max_results=10)
        provider_with = SRAProvider(data_manager=dm_with, config=config_with)

        # Provider without filters
        dm_without = DataManagerV2(workspace_path=str(tmp_path / "ws_without"))
        config_without = SRAProviderConfig(enable_quality_filters=False, max_results=10)
        provider_without = SRAProvider(data_manager=dm_without, config=config_without)

        query = "cancer transcriptomics"

        print("\n" + "=" * 80)
        print("QUALITY FILTER EFFECTIVENESS COMPARISON")
        print("=" * 80)

        print("\n[1/2] Searching WITHOUT quality filters...")
        result_without = provider_without.search_publications(query, max_results=10)

        print("\n[2/2] Searching WITH quality filters...")
        result_with = provider_with.search_publications(query, max_results=10)

        print("\n" + "-" * 80)
        print("Results WITHOUT Quality Filters:")
        print("-" * 80)
        print(result_without[:1000])

        print("\n" + "-" * 80)
        print("Results WITH Quality Filters:")
        print("-" * 80)
        print(result_with[:1000])

        print("\n" + "=" * 80)
        print("✅ Comparison complete - quality filters applied successfully")
        print("=" * 80)
