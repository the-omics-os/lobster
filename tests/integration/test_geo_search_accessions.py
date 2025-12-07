"""
Integration tests for GEO search accession handling.

Tests that GEO searches return valid accessions (GSE/GDS/GPL/GSM) and not invalid
UID-based accessions.

Regression test for bug where fast_dataset_search returned invalid accessions like
"GDS200278021" instead of valid "GSE278021".
"""

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.geo_provider import GEOProvider, GEOSearchFilters


@pytest.fixture
def geo_provider(tmp_path):
    """Create GEOProvider with test data manager."""
    data_manager = DataManagerV2(workspace_path=tmp_path)
    return GEOProvider(data_manager=data_manager)


@pytest.mark.real_api
@pytest.mark.integration
def test_geo_search_returns_valid_gse_accessions(geo_provider):
    """
    Test that GEO search returns valid GSE accessions, not invalid GDS-prefixed UIDs.

    Regression test for BUGFIX_REPORT_GDS_GSE.md issue.
    """
    # Query from user bug report
    query = '"10x Genomics" single cell RNA-seq'
    filters = GEOSearchFilters(max_results=5)

    # Execute search
    search_result = geo_provider.search_geo_datasets(query, filters)

    # Should have results
    assert search_result.count > 0, "Should find matching datasets"
    assert len(search_result.ids) > 0, "Should return dataset IDs"

    # Get summaries
    summaries = geo_provider.get_dataset_summaries(search_result)

    # Should have summaries (not empty due to API error)
    assert len(summaries) > 0, (
        "Should retrieve summaries successfully. "
        "If this fails, check NCBI API status and rate limits."
    )

    # Verify each summary has valid accession
    for summary in summaries:
        uid = summary["uid"]
        accession = summary.get("accession", "")
        entry_type = summary.get("entrytype", "")

        # Accession should exist
        assert accession, f"UID {uid} should have accession field"

        # Accession should start with valid GEO prefix
        valid_prefixes = ["GSE", "GDS", "GPL", "GSM"]
        assert any(accession.startswith(prefix) for prefix in valid_prefixes), (
            f"Accession '{accession}' should start with valid GEO prefix "
            f"(GSE/GDS/GPL/GSM), not be a raw UID"
        )

        # Accession should NOT be UID prefixed with "GDS"
        # This was the bug: "GDS200278021" instead of "GSE278021"
        if entry_type == "GSE":
            assert accession.startswith("GSE"), (
                f"Entry type is {entry_type} but accession is {accession}. "
                f"Should start with GSE, not GDS"
            )
            assert not accession.startswith("GDS200"), (
                f"Accession {accession} looks like invalid UID-based format. "
                f"Should be GSE, not GDS200..."
            )


@pytest.mark.real_api
@pytest.mark.integration
def test_geo_search_formatted_output_no_invalid_accessions(geo_provider):
    """
    Test that formatted search results don't contain invalid GDS-prefixed UIDs.

    Verifies the format_geo_search_results() fix.
    """
    # Query from user bug report
    query = '"10x Genomics" single cell RNA-seq'
    filters = GEOSearchFilters(max_results=5)

    # Execute search
    search_result = geo_provider.search_geo_datasets(query, filters)

    # Get summaries
    search_result.summaries = geo_provider.get_dataset_summaries(search_result)

    # Format results
    formatted = geo_provider.format_geo_search_results(search_result, query)

    # Should contain valid GSE accessions
    assert "GSE" in formatted, "Formatted output should contain GSE accessions"

    # Should NOT contain invalid GDS-prefixed UIDs (the bug pattern)
    assert "GDS200" not in formatted, (
        "Formatted output should not contain invalid 'GDS200...' UIDs. "
        "These are internal NCBI IDs, not valid accessions."
    )

    # Should NOT have GDSbrowser URLs with raw UIDs (bug pattern)
    assert "GDSbrowser?acc=GDS200" not in formatted, (
        "Should not generate URLs with invalid GDS-prefixed UIDs"
    )

    # Should have proper GEO URLs
    assert (
        "geo/query/acc.cgi?acc=GSE" in formatted
        or "GDSbrowser?acc=GSE" in formatted
        or "GDSbrowser?acc=GDS" in formatted
    ), "Should have valid GEO URLs with proper accessions"


@pytest.mark.real_api
@pytest.mark.integration
def test_geo_search_handles_summary_failure_gracefully(geo_provider, monkeypatch):
    """
    Test that search handles summary fetch failures gracefully.

    When summaries cannot be retrieved, should show clear error instead of
    invalid accessions.
    """
    # Mock get_dataset_summaries to return empty list (simulating API failure)
    def mock_get_summaries(search_result):
        return []

    monkeypatch.setattr(geo_provider, "get_dataset_summaries", mock_get_summaries)

    # Execute search
    query = "test query"
    filters = GEOSearchFilters(max_results=5)
    search_result = geo_provider.search_geo_datasets(query, filters)

    # Set empty summaries (simulating failure)
    search_result.summaries = []

    # Format results
    formatted = geo_provider.format_geo_search_results(search_result, query)

    # Should contain error message
    assert "Error" in formatted or "⚠️" in formatted, (
        "Should show error message when summaries unavailable"
    )

    # Should NOT contain invalid GDS-prefixed UIDs
    assert "GDS200" not in formatted, (
        "Should not fall back to invalid UID-based accessions"
    )

    # Should suggest actions
    assert "Try again" in formatted or "Refine" in formatted, (
        "Should suggest remediation steps"
    )


@pytest.mark.real_api
@pytest.mark.integration
def test_geo_search_validates_accession_structure(geo_provider):
    """
    Test that retrieved accessions have valid structure.

    Valid patterns:
    - GSE followed by digits (series): GSE12345
    - GDS followed by digits (dataset): GDS5678
    - GPL followed by digits (platform): GPL96
    - GSM followed by digits (sample): GSM123456

    Invalid patterns:
    - GDS followed by 9+ digits: GDS200278021 (this was the bug)
    - Any accession that looks like a prefixed UID
    """
    query = '"10x Genomics" single cell RNA-seq'
    filters = GEOSearchFilters(max_results=5)

    # Execute search
    search_result = geo_provider.search_geo_datasets(query, filters)
    summaries = geo_provider.get_dataset_summaries(search_result)

    for summary in summaries:
        accession = summary.get("accession", "")

        # Check length (UIDs are 9 digits, accessions are typically shorter)
        if accession.startswith("GDS"):
            # GDS datasets typically have shorter numeric IDs
            numeric_part = accession[3:]  # Strip "GDS" prefix
            assert len(numeric_part) <= 7, (
                f"GDS accession {accession} has suspiciously long ID. "
                f"Might be a prefixed UID (bug pattern: GDS200278021)"
            )

        if accession.startswith("GSE"):
            # GSE series typically have shorter numeric IDs
            numeric_part = accession[3:]  # Strip "GSE" prefix
            assert len(numeric_part) <= 7, (
                f"GSE accession {accession} has suspiciously long ID. "
                f"Validate this is a real accession, not a prefixed UID."
            )
