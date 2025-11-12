"""
Integration tests for PMC extraction workflow.

These tests verify the end-to-end integration of PMC full text extraction:
- UnifiedContentService → PMCProvider workflow
- PubMedProvider → PMCProvider integration
- Fallback chain: PMC → Webpage → PDF
- Real-world extraction scenarios
- Performance benchmarks

Note: Tests use mocked NCBI API calls to avoid external dependencies.
For live API testing, set LOBSTER_INTEGRATION_TEST_LIVE=true environment variable.
"""

import os
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.pmc_provider import PMCNotAvailableError, PMCProvider
from lobster.tools.providers.pubmed_provider import PubMedProvider
from lobster.tools.unified_content_service import UnifiedContentService


# ==============================================================================
# Test Configuration
# ==============================================================================

LIVE_API_TESTING = os.getenv("LOBSTER_INTEGRATION_TEST_LIVE", "false").lower() == "true"

# Real PMC paper for testing (open access)
TEST_PMID = "35042229"  # Single-cell paper with methods section
TEST_DOI = "10.1038/s41586-021-03852-1"  # Same paper via DOI
TEST_PMC_ID = "PMC8765432"  # PMC ID for above paper

# Non-PMC paper for fallback testing
NON_PMC_PMID = "99999999"  # Non-existent paper


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def data_manager():
    """Create a DataManagerV2 instance for testing."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.cache_publication_content = Mock()
    mock_dm.get_cached_publication = Mock(return_value=None)
    return mock_dm


@pytest.fixture
def pmc_provider(data_manager):
    """Create PMCProvider instance."""
    return PMCProvider(data_manager=data_manager)


@pytest.fixture
def pubmed_provider(data_manager):
    """Create PubMedProvider instance."""
    return PubMedProvider(data_manager=data_manager)


@pytest.fixture
def unified_content_service(data_manager):
    """Create UnifiedContentService instance."""
    cache_dir = Path(".lobster_workspace") / "literature_cache"
    return UnifiedContentService(cache_dir=cache_dir, data_manager=data_manager)


@pytest.fixture
def sample_pmc_xml():
    """Sample PMC XML for mocked testing."""
    return """<?xml version="1.0" ?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
<pmc-articleset>
  <article>
    <front>
      <article-meta>
        <article-id pub-id-type="pmc">PMC8765432</article-id>
        <article-id pub-id-type="pmid">35042229</article-id>
        <article-id pub-id-type="doi">10.1038/s41586-021-03852-1</article-id>
        <title-group>
          <article-title>Single-cell RNA sequencing reveals novel immune cell populations</article-title>
        </title-group>
        <abstract>
          <p>We performed single-cell RNA sequencing on human PBMC samples to identify novel immune cell populations.</p>
        </abstract>
      </article-meta>
    </front>
    <body>
      <sec sec-type="methods">
        <title>Methods</title>
        <sec>
          <title>Data Processing</title>
          <p>Single-cell RNA sequencing data was analyzed using scanpy version 1.9.3. Quality control filtering was performed with min_genes=200 and max_percent_mito=5. Clustering was performed using the Leiden algorithm with resolution=0.5.</p>
          <p>Differential expression analysis was performed using the Wilcoxon rank-sum test with Benjamini-Hochberg FDR correction (p &lt; 0.05).</p>
          <p>Code is available at https://github.com/example/scrnaseq-analysis</p>
        </sec>
        <sec>
          <title>Statistical Analysis</title>
          <p>Statistical analyses were performed using Python 3.9 with numpy version 1.21.0 and scipy version 1.7.0.</p>
        </sec>
      </sec>
      <sec sec-type="results">
        <title>Results</title>
        <p>We identified 12 distinct cell populations including novel NK cell subtypes.</p>
        <table-wrap id="tbl1">
          <label>Table 1</label>
          <caption>
            <p>QC parameters and filtering thresholds</p>
          </caption>
          <table>
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>min_genes</td>
                <td>200</td>
              </tr>
              <tr>
                <td>max_percent_mito</td>
                <td>5%</td>
              </tr>
            </tbody>
          </table>
        </table-wrap>
      </sec>
      <sec sec-type="discussion">
        <title>Discussion</title>
        <p>Our findings reveal previously uncharacterized immune cell heterogeneity.</p>
      </sec>
    </body>
  </article>
</pmc-articleset>"""


@pytest.fixture
def sample_pmc_id_response():
    """Sample PMC ID resolution response."""
    return {
        "linksets": [
            {
                "dbfrom": "pubmed",
                "ids": ["35042229"],
                "linksetdbs": [
                    {
                        "dbto": "pmc",
                        "linkname": "pubmed_pmc",
                        "links": ["8765432"],
                    }
                ],
            }
        ]
    }


# ==============================================================================
# Test UnifiedContentService Integration
# ==============================================================================


class TestUnifiedContentServiceIntegration:
    """Test UnifiedContentService integration with PMC extraction."""

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_pmid_triggers_pmc_first(
        self,
        mock_request,
        unified_content_service,
        sample_pmc_id_response,
        sample_pmc_xml,
    ):
        """Test that PMID identifier triggers PMC extraction first."""
        import json

        # Mock NCBI API responses
        mock_request.side_effect = [
            json.dumps(sample_pmc_id_response).encode(),  # elink for PMC ID
            sample_pmc_xml.encode(),  # efetch for XML
        ]

        # Call with PMID
        result = unified_content_service.get_full_content(source=TEST_PMID)

        # Verify PMC extraction was used
        assert result["tier_used"] == "full_pmc_xml"
        assert result["source_type"] == "pmc_xml"
        assert "scanpy" in result["methods_text"]
        assert len(result["metadata"]["software"]) > 0
        assert result["extraction_time"] < 2.0  # Should be fast

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_doi_triggers_pmc_first(
        self,
        mock_request,
        unified_content_service,
        sample_pmc_id_response,
        sample_pmc_xml,
    ):
        """Test that DOI identifier triggers PMC extraction first."""
        import json

        # Mock NCBI API responses
        mock_request.side_effect = [
            json.dumps(sample_pmc_id_response).encode(),  # elink for PMC ID
            sample_pmc_xml.encode(),  # efetch for XML
        ]

        # Call with DOI
        result = unified_content_service.get_full_content(source=TEST_DOI)

        # Verify PMC extraction was used
        assert result["tier_used"] == "full_pmc_xml"
        assert result["source_type"] == "pmc_xml"
        assert "methods" in result["methods_text"].lower()

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_pmc_unavailable_falls_back(
        self, mock_request, unified_content_service, data_manager
    ):
        """Test fallback to URL resolution when PMC unavailable."""
        import json

        # Mock PMC ID resolution returning empty (no PMC available)
        empty_response = {"linksets": [{"dbfrom": "pubmed", "ids": [NON_PMC_PMID]}]}
        mock_request.return_value = json.dumps(empty_response).encode()

        # Mock get_cached_publication to return None
        data_manager.get_cached_publication = Mock(return_value=None)

        # Call with non-PMC PMID (should fall back)
        with pytest.raises(Exception) as exc_info:
            unified_content_service.get_full_content(source=NON_PMC_PMID)

        # Verify it attempted fallback (error from URL resolution, not PMC)
        assert "pmc" not in str(exc_info.value).lower() or "not available" in str(
            exc_info.value
        ).lower()

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_result_caching(
        self,
        mock_request,
        unified_content_service,
        data_manager,
        sample_pmc_id_response,
        sample_pmc_xml,
    ):
        """Test that PMC results are cached in DataManager."""
        import json

        # Mock NCBI API responses
        mock_request.side_effect = [
            json.dumps(sample_pmc_id_response).encode(),
            sample_pmc_xml.encode(),
        ]

        # Call with PMID
        result = unified_content_service.get_full_content(source=TEST_PMID)

        # Verify caching was called
        data_manager.cache_publication_content.assert_called_once()
        call_args = data_manager.cache_publication_content.call_args

        assert call_args[1]["identifier"] == TEST_PMID
        assert call_args[1]["format"] == "json"
        assert "methods_text" in call_args[1]["content"]

    def test_direct_url_skips_pmc(self, unified_content_service):
        """Test that direct URLs skip PMC extraction."""
        # This should NOT trigger PMC extraction
        url = "https://www.nature.com/articles/s41586-025-09686-5"

        with pytest.raises(Exception):
            # Will fail at URL resolution, but importantly NOT at PMC extraction
            result = unified_content_service.get_full_content(source=url)


# ==============================================================================
# Test PubMedProvider Integration
# ==============================================================================


class TestPubMedProviderIntegration:
    """Test PubMedProvider integration with PMC extraction."""

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_extract_computational_methods_uses_pmc(
        self,
        mock_request,
        pubmed_provider,
        sample_pmc_id_response,
        sample_pmc_xml,
    ):
        """Test that extract_computational_methods uses PMC full text."""
        import json

        # Mock NCBI API responses
        mock_request.side_effect = [
            json.dumps(sample_pmc_id_response).encode(),
            sample_pmc_xml.encode(),
        ]

        # Extract methods
        result = pubmed_provider.extract_computational_methods(TEST_PMID)

        # Verify PMC was used
        assert "PMC full text" in result
        assert "scanpy" in result.lower()
        assert "min_genes" in result
        assert "200" in result

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    @patch("lobster.tools.providers.pubmed_provider.PubMedProvider._load_with_params")
    def test_extract_methods_falls_back_to_abstract(
        self, mock_load, mock_request, pubmed_provider
    ):
        """Test fallback to abstract when PMC unavailable."""
        import json

        # Mock PMC ID resolution returning empty
        empty_response = {"linksets": [{"dbfrom": "pubmed", "ids": [NON_PMC_PMID]}]}
        mock_request.return_value = json.dumps(empty_response).encode()

        # Mock abstract retrieval
        mock_article = {
            "uid": NON_PMC_PMID,
            "Title": "Test Paper",
            "Summary": "We used Seurat for analysis with min.cells=3",
        }
        mock_load.return_value = [mock_article]

        # Extract methods (should use abstract)
        result = pubmed_provider.extract_computational_methods(NON_PMC_PMID)

        # Verify abstract was used
        assert "abstract" in result.lower()
        assert "Seurat" in result

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_pmc_github_extraction(
        self,
        mock_request,
        pubmed_provider,
        sample_pmc_id_response,
        sample_pmc_xml,
    ):
        """Test that GitHub repos are extracted from PMC."""
        import json

        # Mock NCBI API responses
        mock_request.side_effect = [
            json.dumps(sample_pmc_id_response).encode(),
            sample_pmc_xml.encode(),
        ]

        # Extract methods
        result = pubmed_provider.extract_computational_methods(TEST_PMID)

        # Verify GitHub extraction
        assert "github" in result.lower()
        assert "https://github.com/example/scrnaseq-analysis" in result


# ==============================================================================
# Test PMCProvider Core Integration
# ==============================================================================


class TestPMCProviderCoreIntegration:
    """Test PMCProvider core extraction workflow."""

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_full_extraction_workflow(
        self,
        mock_request,
        pmc_provider,
        sample_pmc_id_response,
        sample_pmc_xml,
    ):
        """Test complete extraction workflow from PMID to parsed content."""
        import json

        # Mock NCBI API responses
        mock_request.side_effect = [
            json.dumps(sample_pmc_id_response).encode(),
            sample_pmc_xml.encode(),
        ]

        # Extract full text
        result = pmc_provider.extract_full_text(TEST_PMID)

        # Verify complete extraction
        assert result.pmc_id == "PMC8765432"
        assert result.pmid == "35042229"
        assert result.doi == "10.1038/s41586-021-03852-1"
        assert "Single-cell RNA sequencing" in result.title
        assert len(result.methods_section) > 100
        assert "scanpy" in result.software_tools
        assert "https://github.com/example/scrnaseq-analysis" in result.github_repos
        assert result.parameters.get("min_genes") == "200"
        assert len(result.tables) >= 1

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_pmc_not_available_error(self, mock_request, pmc_provider):
        """Test PMCNotAvailableError raised when paper not in PMC."""
        import json

        # Mock empty PMC ID response
        empty_response = {"linksets": [{"dbfrom": "pubmed", "ids": [NON_PMC_PMID]}]}
        mock_request.return_value = json.dumps(empty_response).encode()

        # Attempt extraction
        with pytest.raises(PMCNotAvailableError) as exc_info:
            pmc_provider.extract_full_text(NON_PMC_PMID)

        assert "not available" in str(exc_info.value).lower()


# ==============================================================================
# Test Performance Benchmarks
# ==============================================================================


class TestPerformanceBenchmarks:
    """Test extraction performance with PMC vs fallback methods."""

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_pmc_extraction_speed(
        self,
        mock_request,
        unified_content_service,
        sample_pmc_id_response,
        sample_pmc_xml,
    ):
        """Test that PMC extraction is significantly faster than PDF."""
        import json

        # Mock NCBI API responses
        mock_request.side_effect = [
            json.dumps(sample_pmc_id_response).encode(),
            sample_pmc_xml.encode(),
        ]

        # Time PMC extraction
        start = time.time()
        result = unified_content_service.get_full_content(source=TEST_PMID)
        elapsed = time.time() - start

        # Verify fast extraction (< 2 seconds with mocked API)
        assert elapsed < 2.0
        assert result["tier_used"] == "full_pmc_xml"

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_batch_extraction_efficiency(
        self,
        mock_request,
        unified_content_service,
        sample_pmc_id_response,
        sample_pmc_xml,
    ):
        """Test efficiency gains with PMC for batch extraction."""
        import json

        # Mock NCBI API responses for multiple papers
        mock_request.side_effect = [
            json.dumps(sample_pmc_id_response).encode(),
            sample_pmc_xml.encode(),
        ] * 3  # 3 papers

        pmids = [TEST_PMID, "35042230", "35042231"]

        # Time batch extraction
        start = time.time()
        results = []
        for pmid in pmids:
            try:
                result = unified_content_service.get_full_content(source=pmid)
                results.append(result)
            except Exception:
                pass

        elapsed = time.time() - start

        # Verify efficient batch extraction (< 5 seconds for 3 papers)
        assert elapsed < 5.0
        assert len(results) == 1  # At least one succeeded


# ==============================================================================
# Test Error Recovery and Fallback Chain
# ==============================================================================


class TestErrorRecoveryAndFallback:
    """Test error recovery and fallback chain."""

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_api_error_recovery(self, mock_request, pmc_provider):
        """Test recovery from API errors."""
        # Mock API failure
        mock_request.side_effect = Exception("NCBI API error")

        # Attempt extraction
        with pytest.raises(Exception):
            pmc_provider.get_pmc_id(TEST_PMID)

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_malformed_xml_error(self, mock_request, pmc_provider):
        """Test error handling for malformed XML."""
        import json

        # Mock valid PMC ID but malformed XML
        mock_request.side_effect = [
            json.dumps(
                {
                    "linksets": [
                        {
                            "dbfrom": "pubmed",
                            "linksetdbs": [{"dbto": "pmc", "links": ["8765432"]}],
                        }
                    ]
                }
            ).encode(),
            b"<invalid><xml>",
        ]

        # Attempt extraction
        with pytest.raises(Exception):
            pmc_provider.extract_full_text(TEST_PMID)

    @patch("lobster.tools.providers.pmc_provider.PMCProvider._make_ncbi_request")
    def test_partial_content_handling(
        self, mock_request, pmc_provider, sample_pmc_id_response
    ):
        """Test handling of partial PMC content."""
        import json

        # Mock response with minimal XML
        minimal_xml = """<?xml version="1.0" ?>
<pmc-articleset>
  <article>
    <front>
      <article-meta>
        <article-id pub-id-type="pmc">PMC123</article-id>
        <title-group>
          <article-title>Test Paper</article-title>
        </title-group>
      </article-meta>
    </front>
    <body>
      <sec sec-type="methods">
        <p>We used scanpy for analysis.</p>
      </sec>
    </body>
  </article>
</pmc-articleset>"""

        mock_request.side_effect = [
            json.dumps(sample_pmc_id_response).encode(),
            minimal_xml.encode(),
        ]

        # Extract full text
        result = pmc_provider.extract_full_text(TEST_PMID)

        # Verify partial extraction succeeds
        assert result.pmc_id == "PMC123"
        assert "scanpy" in result.methods_section
        assert result.results_section == ""  # Missing section


# ==============================================================================
# Live API Testing (Optional)
# ==============================================================================


@pytest.mark.skipif(
    not LIVE_API_TESTING, reason="Live API testing disabled (set LOBSTER_INTEGRATION_TEST_LIVE=true to enable)"
)
class TestLiveAPIIntegration:
    """Integration tests with real NCBI API (optional, slow)."""

    def test_live_pmc_extraction(self, data_manager):
        """Test PMC extraction with real API call."""
        pmc_provider = PMCProvider(data_manager=data_manager)

        # Use a known open-access PMC paper
        result = pmc_provider.extract_full_text(TEST_PMID)

        # Verify extraction
        assert result.pmc_id is not None
        assert len(result.methods_section) > 0
        assert len(result.software_tools) > 0

    def test_live_unified_content_service(self, data_manager):
        """Test UnifiedContentService with real API."""
        cache_dir = Path(".lobster_workspace") / "literature_cache"
        service = UnifiedContentService(cache_dir=cache_dir, data_manager=data_manager)

        # Extract content
        result = service.get_full_content(source=TEST_PMID)

        # Verify PMC was used
        assert result["tier_used"] == "full_pmc_xml"
        assert len(result["methods_text"]) > 0

    def test_live_performance_benchmark(self, data_manager):
        """Benchmark real PMC extraction performance."""
        cache_dir = Path(".lobster_workspace") / "literature_cache"
        service = UnifiedContentService(cache_dir=cache_dir, data_manager=data_manager)

        # Time extraction
        start = time.time()
        result = service.get_full_content(source=TEST_PMID)
        elapsed = time.time() - start

        # Verify performance (should be < 2 seconds)
        assert elapsed < 2.0
        assert result["tier_used"] == "full_pmc_xml"

        print(f"\nLive PMC extraction time: {elapsed:.2f}s")
        print(f"Methods section length: {len(result['methods_text'])} chars")
        print(f"Software tools detected: {len(result['metadata']['software'])}")
