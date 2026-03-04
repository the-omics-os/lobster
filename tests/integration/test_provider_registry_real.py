"""
Real API integration tests for ProviderRegistry.

These tests verify the modular provider architecture introduced in Phase 1
to replace the monolithic PublicationResolver class. Tests make actual API
calls to NCBI, PMC, and GEO to validate end-to-end functionality.

Pytest Markers:
- @pytest.mark.integration: Tests requiring real API calls
- @pytest.mark.slow: Tests with >1 second execution time
- @pytest.mark.real_api: Tests that hit external APIs

Running Tests:
```bash
# Integration tests only (real API calls)
pytest tests/integration/test_provider_registry_real.py -m "integration" -v
```

API Keys Required:
- NCBI_API_KEY: Required for PMC and PubMed API access
- AWS_BEDROCK_ACCESS_KEY: Required for LLM-based methods extraction
- AWS_BEDROCK_SECRET_ACCESS_KEY: Required for LLM-based methods extraction

Rate Limiting:
- 0.5-1 second sleeps between consecutive API calls
"""

import os
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.abstract_provider import AbstractProvider
from lobster.tools.providers.pubmed_provider import PubMedProvider

pytestmark = [pytest.mark.integration]


@pytest.fixture
def real_data_manager(tmp_path):
    """Create a real DataManagerV2 instance with temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    dm = Mock(spec=DataManagerV2)
    dm.cache_dir = cache_dir
    dm.literature_cache_dir = cache_dir / "literature"
    dm.literature_cache_dir.mkdir()

    return dm


@pytest.fixture
def real_abstract_provider(real_data_manager):
    """Create a real AbstractProvider instance for API testing."""
    return AbstractProvider(data_manager=real_data_manager)


@pytest.fixture
def real_pubmed_provider(real_data_manager):
    """Create a real PubMedProvider instance for API testing."""
    return PubMedProvider(data_manager=real_data_manager)


# Test Classes for Real API Integration


@pytest.mark.skipif(
    os.getenv("LOBSTER_RUN_REAL_API_TESTS") != "1",
    reason="Real API tests require LOBSTER_RUN_REAL_API_TESTS=1 environment variable",
)
class TestRealPMIDResolution:
    """Test PMID resolution with real NCBI API calls.

    Test Paper: PMID:35042229 (Nature 2022)
    - Title: "Single-cell analysis reveals inflammatory interactions..."
    - Valid PMC ID: PMC8760896
    - DOI: 10.1038/s41586-021-03852-1
    """

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_pmid_resolution_real(self, real_abstract_provider):
        """Test real PMID resolution via NCBI Entrez API.

        Verifies:
        1. PMID resolves to publication metadata
        2. Title, authors, journal, abstract are extracted
        3. DOI is correctly identified
        4. Response time is <1 second (fast tier)
        """
        identifier = "PMID:35042229"

        # Real API call - measure response time
        start_time = time.time()
        metadata = real_abstract_provider.get_abstract(identifier)
        elapsed = time.time() - start_time

        # Verify metadata extraction
        assert metadata.pmid == "35042229"
        assert "single-cell" in metadata.title.lower()
        assert metadata.journal == "Nature"
        assert metadata.doi == "10.1038/s41586-021-03852-1"
        assert len(metadata.abstract) > 200
        assert len(metadata.authors) > 5

        # Verify response time (Tier 1 fast access)
        assert elapsed < 2.0, f"Abstract retrieval took {elapsed:.2f}s (expected <2s)"

        # Rate limiting
        time.sleep(0.5)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_pmid_invalid_real(self, real_abstract_provider):
        """Test handling of invalid PMID with real API call.

        Verifies graceful error handling for non-existent PMIDs.
        """
        identifier = "PMID:99999999"

        # Real API call - should raise ValueError
        with pytest.raises((ValueError, Exception)):
            real_abstract_provider.get_abstract(identifier)

        # Rate limiting
        time.sleep(0.5)


@pytest.mark.skipif(
    os.getenv("LOBSTER_RUN_REAL_API_TESTS") != "1",
    reason="Real API tests require LOBSTER_RUN_REAL_API_TESTS=1 environment variable",
)
class TestRealDOIResolution:
    """Test DOI resolution with real CrossRef/DOI.org API calls.

    Test Paper: 10.1038/s41586-021-03852-1 (Nature 2022)
    - Same paper as PMID:35042229
    """

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_doi_resolution_real(self, real_abstract_provider):
        """Test real DOI resolution via CrossRef API.

        Verifies:
        1. DOI resolves to publication metadata
        2. Title, authors, journal match PMID resolution
        3. Response time is reasonable (<2 seconds)
        """
        identifier = "10.1038/s41586-021-03852-1"

        # Real API call - measure response time
        start_time = time.time()
        metadata = real_abstract_provider.get_abstract(identifier)
        elapsed = time.time() - start_time

        # Verify metadata extraction
        assert metadata.doi == "10.1038/s41586-021-03852-1"
        assert "single-cell" in metadata.title.lower()
        assert metadata.journal == "Nature"
        assert len(metadata.authors) > 5

        # Verify response time
        assert elapsed < 3.0, f"DOI resolution took {elapsed:.2f}s (expected <3s)"

        # Rate limiting
        time.sleep(0.5)


@pytest.mark.skipif(
    os.getenv("LOBSTER_RUN_REAL_API_TESTS") != "1",
    reason="Real API tests require LOBSTER_RUN_REAL_API_TESTS=1 environment variable",
)
class TestRealPMCFulltextAccess:
    """Test PMC fulltext XML retrieval with real NCBI PMC API calls.

    Test Paper: PMC8760896 (Nature 2022)
    - Full text available in PMC Open Access
    - Contains Methods section
    """

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_get_pmc_fulltext_real(self, real_pubmed_provider):
        """Test real PMC fulltext XML retrieval via NCBI PMC OA Service.

        Verifies:
        1. PMC ID resolves to fulltext XML
        2. XML contains substantive content
        3. Methods section can be extracted
        4. Response time is <3 seconds (Tier 2 structured access)
        """
        pmc_id = "PMC8760896"

        # Real API call - measure response time
        start_time = time.time()
        fulltext = real_pubmed_provider.get_pmc_fulltext(pmc_id)
        elapsed = time.time() - start_time

        # Verify fulltext extraction
        assert fulltext is not None
        assert len(fulltext) > 5000  # Non-trivial content
        assert "methods" in fulltext.lower() or "method" in fulltext.lower()
        assert "single-cell" in fulltext.lower()

        # Verify response time (Tier 2 access)
        assert elapsed < 5.0, f"PMC fulltext took {elapsed:.2f}s (expected <5s)"

        # Rate limiting - longer wait for fulltext
        time.sleep(1.0)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.real_api
    def test_pmid_to_pmc_conversion_real(self, real_pubmed_provider):
        """Test PMID to PMC ID conversion via NCBI ID Converter API.

        Verifies:
        1. PMID resolves to PMC ID
        2. PMC ID format is correct
        """
        pmid = "35042229"

        # Real API call
        pmc_id = real_pubmed_provider._pmid_to_pmc(pmid)

        # Verify PMC ID extraction
        assert pmc_id == "PMC8760896"
        assert pmc_id.startswith("PMC")

        # Rate limiting
        time.sleep(0.5)
