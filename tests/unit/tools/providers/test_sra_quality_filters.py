"""
Unit tests for SRAProvider Phase 1.4 quality filters.

Tests quality filter application for different modalities:
- scRNA-seq specific filters (paired-end, Illumina)
- Bulk RNA-seq (no additional filters)
- Amplicon/16S (strategy filter)

Note: Base filters (public, has data) were removed as [Access] and [Properties]
are not valid SRA field qualifiers. Only valid field qualifiers ([LAY], [PLAT],
[STRA], [ORGN], [SRC]) are used.
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
        """Test that base quality filters return query unchanged (no invalid Access/Properties)."""
        query = "microbiome"
        result = provider._apply_quality_filters(query)

        # No base filters added (Access/Properties are not valid SRA fields)
        assert result.startswith("microbiome")
        # Query should be unchanged when no modality hint is provided
        assert result == query

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

        # scRNA-seq specific (valid SRA field qualifiers)
        assert 'AND "PAIRED"[LAY]' in result
        assert 'AND "ILLUMINA"[PLAT]' in result
        assert result.startswith("cancer cells")

    def test_scrna_quality_filters_variants(self, provider):
        """Test scRNA-seq quality filters with different hint variants."""
        query = "cells"

        # Test different hint formats
        for hint in ["scrna-seq", "single-cell", "singlecell", "scRNA"]:
            result = provider._apply_quality_filters(query, modality_hint=hint)
            assert 'AND "PAIRED"[LAY]' in result
            assert 'AND "ILLUMINA"[PLAT]' in result

    def test_bulk_rnaseq_quality_filters(self, provider):
        """Test bulk RNA-seq quality filters (no additional filters)."""
        query = "liver tissue"

        result = provider._apply_quality_filters(query, modality_hint="bulk-rna-seq")

        # No additional filters for bulk RNA-seq
        assert '"PAIRED"[LAY]' not in result
        assert '"AMPLICON"[STRA]' not in result
        # Query should be unchanged (bulk RNA-seq only gets base filters, which are now empty)
        assert result == query

    def test_amplicon_quality_filters(self, provider):
        """Test amplicon/16S quality filters."""
        query = "gut microbiome"

        result = provider._apply_quality_filters(query, modality_hint="amplicon")

        # Amplicon specific (valid SRA field qualifier)
        assert 'AND "AMPLICON"[STRA]' in result
        assert result.startswith("gut microbiome")

    def test_16s_quality_filters(self, provider):
        """Test 16S rRNA quality filters (same as amplicon)."""
        query = "microbiome"

        result = provider._apply_quality_filters(query, modality_hint="16s")

        # 16S/amplicon specific
        assert 'AND "AMPLICON"[STRA]' in result

    def test_no_modality_hint(self, provider):
        """Test quality filters without modality hint (no additional filters)."""
        query = "cancer"

        result = provider._apply_quality_filters(query, modality_hint=None)

        # No modality-specific filters
        assert '"PAIRED"[LAY]' not in result
        assert '"AMPLICON"[STRA]' not in result
        # Query unchanged when no modality hint
        assert result == query

    def test_unknown_modality_hint(self, provider):
        """Test quality filters with unknown modality hint (no additional filters)."""
        query = "data"

        result = provider._apply_quality_filters(
            query, modality_hint="unknown-modality"
        )

        # No modality-specific filters
        assert '"PAIRED"[LAY]' not in result
        assert '"AMPLICON"[STRA]' not in result
        # Query unchanged for unknown modality
        assert result == query

    def test_quality_filters_with_existing_filters(self, provider):
        """Test quality filters applied to query that already has filters."""
        query = "(gut microbiome) AND (Homo sapiens[ORGN])"

        result = provider._apply_quality_filters(query, modality_hint="amplicon")

        # Original query preserved
        assert "(gut microbiome)" in result
        assert "(Homo sapiens[ORGN])" in result

        # Modality-specific filter added
        assert 'AND "AMPLICON"[STRA]' in result


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

        # Verify modality-specific quality filter in query
        assert 'AND "AMPLICON"[STRA]' in query_arg

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

        # Verify scRNA-seq specific filters (valid SRA field qualifiers)
        assert 'AND "PAIRED"[LAY]' in query_arg
        assert 'AND "ILLUMINA"[PLAT]' in query_arg

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
