"""
Unit tests for SRAProvider Phase 1.4 quality filters.

Tests quality filter application for different modalities:
- Base filters (public, has data) - all modalities
- scRNA-seq specific filters (paired-end, Illumina)
- Bulk RNA-seq (base filters only)
- Amplicon/16S (strategy filter)
"""

from unittest.mock import Mock, patch

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.sra_provider import SRAProvider, SRAProviderConfig


class TestQualityFilters:
    """Test quality filter application."""

    @pytest.fixture
    def provider(self, tmp_path):
        """Create SRAProvider with quality filters enabled."""
        dm = DataManagerV2(workspace_path=str(tmp_path / "test_workspace"))
        config = SRAProviderConfig(enable_quality_filters=True)
        return SRAProvider(data_manager=dm, config=config)

    @pytest.fixture
    def provider_no_filters(self, tmp_path):
        """Create SRAProvider with quality filters disabled."""
        dm = DataManagerV2(workspace_path=str(tmp_path / "test_workspace"))
        config = SRAProviderConfig(enable_quality_filters=False)
        return SRAProvider(data_manager=dm, config=config)

    def test_base_quality_filters(self, provider):
        """Test that base quality filters are applied (public, has data)."""
        query = "microbiome"
        result = provider._apply_quality_filters(query)

        assert 'AND "public"[Access]' in result
        assert 'AND "has data"[Properties]' in result
        assert result.startswith("microbiome")

    def test_quality_filters_disabled(self, provider_no_filters):
        """Test that quality filters can be disabled via config."""
        query = "microbiome"
        result = provider_no_filters._apply_quality_filters(query)

        # Should return query unchanged
        assert result == query
        assert "[Access]" not in result
        assert "[Properties]" not in result

    def test_scrna_quality_filters(self, provider):
        """Test scRNA-seq specific quality filters."""
        query = "cancer cells"

        result = provider._apply_quality_filters(query, modality_hint="scrna-seq")

        # Base filters
        assert 'AND "public"[Access]' in result
        assert 'AND "has data"[Properties]' in result

        # scRNA-seq specific
        assert 'AND "library layout paired"[Filter]' in result
        assert 'AND "platform illumina"[Filter]' in result

    def test_scrna_quality_filters_variants(self, provider):
        """Test scRNA-seq quality filters with different hint variants."""
        query = "cells"

        # Test different hint formats
        for hint in ["scrna-seq", "single-cell", "singlecell", "scRNA"]:
            result = provider._apply_quality_filters(query, modality_hint=hint)
            assert 'AND "library layout paired"[Filter]' in result
            assert 'AND "platform illumina"[Filter]' in result

    def test_bulk_rnaseq_quality_filters(self, provider):
        """Test bulk RNA-seq quality filters (base only)."""
        query = "liver tissue"

        result = provider._apply_quality_filters(query, modality_hint="bulk-rna-seq")

        # Base filters
        assert 'AND "public"[Access]' in result
        assert 'AND "has data"[Properties]' in result

        # No additional filters for bulk RNA-seq
        assert '"library layout paired"[Filter]' not in result
        assert '"strategy amplicon"[Filter]' not in result

    def test_amplicon_quality_filters(self, provider):
        """Test amplicon/16S quality filters."""
        query = "gut microbiome"

        result = provider._apply_quality_filters(query, modality_hint="amplicon")

        # Base filters
        assert 'AND "public"[Access]' in result
        assert 'AND "has data"[Properties]' in result

        # Amplicon specific
        assert 'AND "strategy amplicon"[Filter]' in result

    def test_16s_quality_filters(self, provider):
        """Test 16S rRNA quality filters (same as amplicon)."""
        query = "microbiome"

        result = provider._apply_quality_filters(query, modality_hint="16s")

        # Base filters
        assert 'AND "public"[Access]' in result
        assert 'AND "has data"[Properties]' in result

        # 16S/amplicon specific
        assert 'AND "strategy amplicon"[Filter]' in result

    def test_no_modality_hint(self, provider):
        """Test quality filters without modality hint (base only)."""
        query = "cancer"

        result = provider._apply_quality_filters(query, modality_hint=None)

        # Base filters
        assert 'AND "public"[Access]' in result
        assert 'AND "has data"[Properties]' in result

        # No modality-specific filters
        assert '"library layout paired"[Filter]' not in result
        assert '"strategy amplicon"[Filter]' not in result

    def test_unknown_modality_hint(self, provider):
        """Test quality filters with unknown modality hint (base only)."""
        query = "data"

        result = provider._apply_quality_filters(
            query, modality_hint="unknown-modality"
        )

        # Base filters
        assert 'AND "public"[Access]' in result
        assert 'AND "has data"[Properties]' in result

        # No modality-specific filters
        assert '"library layout paired"[Filter]' not in result
        assert '"strategy amplicon"[Filter]' not in result

    def test_quality_filters_with_existing_filters(self, provider):
        """Test quality filters applied to query that already has filters."""
        query = "(gut microbiome) AND (Homo sapiens[ORGN])"

        result = provider._apply_quality_filters(query, modality_hint="amplicon")

        # Original query preserved
        assert "(gut microbiome)" in result
        assert "(Homo sapiens[ORGN])" in result

        # Quality filters added
        assert 'AND "public"[Access]' in result
        assert 'AND "has data"[Properties]' in result
        assert 'AND "strategy amplicon"[Filter]' in result


class TestQualityFiltersIntegration:
    """Test quality filters integration with search_publications."""

    @pytest.fixture
    def provider(self, tmp_path):
        """Create SRAProvider with mocked NCBI calls."""
        dm = DataManagerV2(workspace_path=str(tmp_path / "test_workspace"))
        config = SRAProviderConfig(enable_quality_filters=True)
        return SRAProvider(data_manager=dm, config=config)

    @patch("lobster.tools.providers.sra_provider.SRAProvider._ncbi_esearch")
    @patch("lobster.tools.providers.sra_provider.SRAProvider._ncbi_esummary")
    def test_quality_filters_applied_in_search(
        self, mock_esummary, mock_esearch, provider
    ):
        """Test that quality filters are applied during search_publications."""
        import pandas as pd

        # Mock NCBI responses
        mock_esearch.return_value = ["12345"]
        mock_esummary.return_value = pd.DataFrame(
            {
                "study_accession": ["SRP123"],
                "study_title": ["Test Study"],
                "organism": ["Homo sapiens"],
                "library_strategy": ["AMPLICON"],
                "library_layout": ["PAIRED"],
                "instrument_platform": ["ILLUMINA"],
                "total_runs": ["10"],
                "total_size": ["1000000"],
            }
        )

        # Call search_publications with modality_hint
        result = provider.search_publications(
            query="gut microbiome", max_results=5, modality_hint="amplicon"
        )

        # Verify esearch was called with quality filters
        mock_esearch.assert_called_once()
        call_args = mock_esearch.call_args
        query_arg = call_args[0][0]  # First positional arg is query

        # Verify quality filters in query
        assert 'AND "public"[Access]' in query_arg
        assert 'AND "has data"[Properties]' in query_arg
        assert 'AND "strategy amplicon"[Filter]' in query_arg

    @patch("lobster.tools.providers.sra_provider.SRAProvider._ncbi_esearch")
    @patch("lobster.tools.providers.sra_provider.SRAProvider._ncbi_esummary")
    def test_scrna_quality_filters_in_search(
        self, mock_esummary, mock_esearch, provider
    ):
        """Test scRNA-seq quality filters in search_publications."""
        import pandas as pd

        # Mock NCBI responses
        mock_esearch.return_value = ["12345"]
        mock_esummary.return_value = pd.DataFrame(
            {
                "study_accession": ["SRP123"],
                "study_title": ["Test Study"],
                "organism": ["Mus musculus"],
                "library_strategy": ["RNA-Seq"],
                "library_layout": ["PAIRED"],
                "instrument_platform": ["ILLUMINA"],
                "total_runs": ["5"],
                "total_size": ["500000"],
            }
        )

        # Call search_publications with scRNA-seq modality
        result = provider.search_publications(
            query="brain cells", max_results=5, modality_hint="scrna-seq"
        )

        # Verify esearch was called with scRNA-seq quality filters
        mock_esearch.assert_called_once()
        call_args = mock_esearch.call_args
        query_arg = call_args[0][0]

        # Verify scRNA-seq specific filters
        assert 'AND "public"[Access]' in query_arg
        assert 'AND "has data"[Properties]' in query_arg
        assert 'AND "library layout paired"[Filter]' in query_arg
        assert 'AND "platform illumina"[Filter]' in query_arg

    def test_search_microbiome_datasets_uses_modality_hint(self, provider):
        """Test that search_microbiome_datasets passes modality_hint correctly."""
        with patch.object(provider, "search_publications") as mock_search:
            mock_search.return_value = "## Mock Results"

            # Call with amplicon_region
            provider.search_microbiome_datasets(
                query="gut health", amplicon_region="16S", max_results=10
            )

            # Verify modality_hint="amplicon" was passed
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["modality_hint"] == "amplicon"

    def test_search_microbiome_datasets_no_modality_hint_for_shotgun(self, provider):
        """Test search_microbiome_datasets passes None for shotgun metagenomics."""
        with patch.object(provider, "search_publications") as mock_search:
            mock_search.return_value = "## Mock Results"

            # Call without amplicon_region (shotgun)
            provider.search_microbiome_datasets(
                query="gut microbiome", amplicon_region=None, max_results=10
            )

            # Verify modality_hint=None was passed
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["modality_hint"] is None
