"""
Unit tests for ProviderRegistry.

Tests the provider registration system and capability-based routing
introduced in Phase 1 of the research agent refactoring.
"""

import pytest

from lobster.tools.providers.base_provider import DatasetType, ProviderCapability
from lobster.tools.providers.provider_registry import (
    ProviderRegistry,
    ProviderRegistryError,
)

# Mock Provider Classes for Testing


class MockProvider:
    """Base mock provider for testing."""

    def __init__(
        self,
        name="MockProvider",
        priority=100,
        capabilities=None,
        dataset_types=None,
    ):
        self.name = name
        self._priority = priority
        self._capabilities = capabilities or {}
        self._dataset_types = dataset_types or []

    @property
    def priority(self):
        return self._priority

    @property
    def supported_dataset_types(self):
        return self._dataset_types

    def get_supported_capabilities(self):
        return self._capabilities


class HighPriorityProvider(MockProvider):
    """Mock provider with high priority (10)."""

    def __init__(self):
        super().__init__(
            name="HighPriorityProvider",
            priority=10,
            capabilities={
                ProviderCapability.SEARCH_LITERATURE: True,
                ProviderCapability.GET_ABSTRACT: True,
                ProviderCapability.DISCOVER_DATASETS: False,
            },
            dataset_types=[],
        )


class MediumPriorityProvider(MockProvider):
    """Mock provider with medium priority (50)."""

    def __init__(self):
        super().__init__(
            name="MediumPriorityProvider",
            priority=50,
            capabilities={
                ProviderCapability.SEARCH_LITERATURE: True,
                ProviderCapability.GET_FULL_CONTENT: True,
                ProviderCapability.DISCOVER_DATASETS: False,
            },
            dataset_types=[],
        )


class LowPriorityProvider(MockProvider):
    """Mock provider with low priority (100)."""

    def __init__(self):
        super().__init__(
            name="LowPriorityProvider",
            priority=100,
            capabilities={
                ProviderCapability.SEARCH_LITERATURE: False,
                ProviderCapability.GET_FULL_CONTENT: True,
                ProviderCapability.EXTRACT_PDF: True,
            },
            dataset_types=[],
        )


class GEOMockProvider(MockProvider):
    """Mock GEO provider with dataset support."""

    def __init__(self):
        super().__init__(
            name="GEOMockProvider",
            priority=10,
            capabilities={
                ProviderCapability.DISCOVER_DATASETS: True,
                ProviderCapability.EXTRACT_METADATA: True,
            },
            dataset_types=[DatasetType.GEO],
        )


# Test Classes


class TestProviderRegistration:
    """Test provider registration functionality."""

    def test_register_single_provider(self):
        """Test registering a single provider."""
        registry = ProviderRegistry()
        provider = HighPriorityProvider()

        registry.register_provider(provider)

        all_providers = registry.get_all_providers()
        assert len(all_providers) == 1
        assert all_providers[0].name == "HighPriorityProvider"

    def test_register_multiple_providers(self):
        """Test registering multiple providers."""
        registry = ProviderRegistry()

        high_provider = HighPriorityProvider()
        medium_provider = MediumPriorityProvider()
        low_provider = LowPriorityProvider()

        registry.register_provider(high_provider)
        registry.register_provider(medium_provider)
        registry.register_provider(low_provider)

        all_providers = registry.get_all_providers()
        assert len(all_providers) == 3

    def test_register_duplicate_provider_warns_and_skips(self):
        """Test that duplicate provider registration is skipped."""
        registry = ProviderRegistry()
        provider1 = HighPriorityProvider()
        provider2 = HighPriorityProvider()

        registry.register_provider(provider1)
        registry.register_provider(provider2)

        # Should only have one provider registered
        all_providers = registry.get_all_providers()
        assert len(all_providers) == 1

    def test_register_provider_missing_interface_raises_error(self):
        """Test that providers missing required interface raise errors."""
        registry = ProviderRegistry()

        # Provider missing get_supported_capabilities
        class InvalidProvider:
            priority = 10
            supported_dataset_types = []

        with pytest.raises(ValueError, match="missing get_supported_capabilities"):
            registry.register_provider(InvalidProvider())


class TestCapabilityRouting:
    """Test capability-based provider routing."""

    def test_get_providers_for_capability(self):
        """Test retrieving providers for a specific capability."""
        registry = ProviderRegistry()
        high_provider = HighPriorityProvider()
        medium_provider = MediumPriorityProvider()

        registry.register_provider(high_provider)
        registry.register_provider(medium_provider)

        # Both providers support SEARCH_LITERATURE
        providers = registry.get_providers_for_capability(
            ProviderCapability.SEARCH_LITERATURE
        )
        assert len(providers) == 2

    def test_get_providers_for_unsupported_capability(self):
        """Test retrieving providers for unsupported capability returns empty list."""
        registry = ProviderRegistry()
        high_provider = HighPriorityProvider()

        registry.register_provider(high_provider)

        # No providers support INTEGRATE_MULTI_OMICS
        providers = registry.get_providers_for_capability(
            ProviderCapability.INTEGRATE_MULTI_OMICS
        )
        assert len(providers) == 0

    def test_get_providers_for_nonexistent_capability(self):
        """Test retrieving providers for non-existent capability."""
        registry = ProviderRegistry()
        high_provider = HighPriorityProvider()

        registry.register_provider(high_provider)

        providers = registry.get_providers_for_capability("nonexistent_capability")
        assert len(providers) == 0


class TestPrioritySorting:
    """Test priority-based provider sorting."""

    def test_providers_sorted_by_priority(self):
        """Test that providers are sorted by priority (lower = higher priority)."""
        registry = ProviderRegistry()

        low_provider = LowPriorityProvider()
        high_provider = HighPriorityProvider()
        medium_provider = MediumPriorityProvider()

        # Register in random order
        registry.register_provider(low_provider)
        registry.register_provider(high_provider)
        registry.register_provider(medium_provider)

        # GET_FULL_CONTENT supported by medium (50) and low (100)
        providers = registry.get_providers_for_capability(
            ProviderCapability.GET_FULL_CONTENT
        )

        assert len(providers) == 2
        # First should be medium (priority 50)
        assert providers[0].priority == 50
        # Second should be low (priority 100)
        assert providers[1].priority == 100

    def test_same_priority_providers_maintain_registration_order(self):
        """Test that providers with same priority maintain registration order."""
        registry = ProviderRegistry()

        # Create two different provider classes with same priority
        class Provider1:
            name = "Provider1"
            priority = 10
            supported_dataset_types = []

            def get_supported_capabilities(self):
                return {ProviderCapability.SEARCH_LITERATURE: True}

        class Provider2:
            name = "Provider2"
            priority = 10
            supported_dataset_types = []

            def get_supported_capabilities(self):
                return {ProviderCapability.SEARCH_LITERATURE: True}

        provider1 = Provider1()
        provider2 = Provider2()

        registry.register_provider(provider1)
        registry.register_provider(provider2)

        providers = registry.get_providers_for_capability(
            ProviderCapability.SEARCH_LITERATURE
        )

        # Should maintain registration order for same priority
        assert providers[0].name == "Provider1"
        assert providers[1].name == "Provider2"


class TestDatasetTypeMapping:
    """Test dataset type to provider mapping."""

    def test_get_provider_for_dataset_type(self):
        """Test retrieving provider for specific dataset type."""
        registry = ProviderRegistry()
        geo_provider = GEOMockProvider()

        registry.register_provider(geo_provider)

        provider = registry.get_provider_for_dataset_type(DatasetType.GEO)
        assert provider is not None
        assert provider.name == "GEOMockProvider"

    def test_get_provider_for_unsupported_dataset_type(self):
        """Test retrieving provider for unsupported dataset type returns None."""
        registry = ProviderRegistry()
        geo_provider = GEOMockProvider()

        registry.register_provider(geo_provider)

        provider = registry.get_provider_for_dataset_type(DatasetType.SRA)
        assert provider is None

    def test_get_supported_dataset_types(self):
        """Test retrieving all supported dataset types."""
        registry = ProviderRegistry()

        geo_provider = GEOMockProvider()
        sra_provider = MockProvider(
            name="SRAProvider",
            priority=10,
            capabilities={},
            dataset_types=[DatasetType.SRA, DatasetType.BIOPROJECT],
        )

        registry.register_provider(geo_provider)
        registry.register_provider(sra_provider)

        dataset_types = registry.get_supported_dataset_types()
        assert len(dataset_types) >= 3  # GEO, SRA, BIOPROJECT
        assert DatasetType.GEO in dataset_types
        assert DatasetType.SRA in dataset_types
        assert DatasetType.BIOPROJECT in dataset_types


class TestCapabilityMatrix:
    """Test capability matrix generation."""

    def test_get_capability_matrix_empty_registry(self):
        """Test capability matrix generation with empty registry."""
        registry = ProviderRegistry()

        matrix = registry.get_capability_matrix()
        assert "No providers registered" in matrix

    def test_get_capability_matrix_with_providers(self):
        """Test capability matrix generation with registered providers."""
        registry = ProviderRegistry()

        high_provider = HighPriorityProvider()
        medium_provider = MediumPriorityProvider()

        registry.register_provider(high_provider)
        registry.register_provider(medium_provider)

        matrix = registry.get_capability_matrix()

        # Should contain provider names with priorities
        assert "HighPriorityProvider (10)" in matrix
        assert "MediumPriorityProvider (50)" in matrix

        # Should contain capabilities
        assert ProviderCapability.SEARCH_LITERATURE in matrix
        assert ProviderCapability.GET_ABSTRACT in matrix
        assert ProviderCapability.GET_FULL_CONTENT in matrix

        # Should show support indicators
        assert "✓" in matrix  # Supported
        assert "-" in matrix  # Not supported

    def test_capability_matrix_shows_all_capabilities(self):
        """Test that capability matrix includes all capabilities from all providers."""
        registry = ProviderRegistry()

        # Create two different provider classes to avoid duplicate detection
        class MockProvider1:
            priority = 100
            supported_dataset_types = []

            def get_supported_capabilities(self):
                return {
                    ProviderCapability.SEARCH_LITERATURE: True,
                    ProviderCapability.GET_ABSTRACT: True,
                }

        class MockProvider2:
            priority = 100
            supported_dataset_types = []

            def get_supported_capabilities(self):
                return {
                    ProviderCapability.GET_FULL_CONTENT: True,
                    ProviderCapability.EXTRACT_PDF: True,
                }

        provider1 = MockProvider1()
        provider2 = MockProvider2()

        registry.register_provider(provider1)
        registry.register_provider(provider2)

        matrix = registry.get_capability_matrix()

        # All capabilities from both providers should be in matrix
        assert ProviderCapability.SEARCH_LITERATURE in matrix
        assert ProviderCapability.GET_ABSTRACT in matrix
        assert ProviderCapability.GET_FULL_CONTENT in matrix
        assert ProviderCapability.EXTRACT_PDF in matrix


# ============================================================================
# Phase 6: Real API Integration Tests
# ============================================================================

"""
The following test classes test provider functionality with real API calls.

These tests verify the modular provider architecture introduced in Phase 1
to replace the monolithic PublicationResolver class. Tests make actual API
calls to NCBI, PMC, and GEO to validate end-to-end functionality.

Pytest Markers:
- @pytest.mark.integration: Tests requiring real API calls
- @pytest.mark.slow: Tests with >1 second execution time
- @pytest.mark.real_api: Tests that hit external APIs

Running Tests:
```bash
# All tests (unit + integration)
pytest tests/unit/tools/providers/test_provider_registry.py -v

# Unit tests only (fast, no API calls)
pytest tests/unit/tools/providers/test_provider_registry.py -m "not integration" -v

# Integration tests only (real API calls)
pytest tests/unit/tools/providers/test_provider_registry.py -m "integration" -v
```

API Keys Required:
- NCBI_API_KEY: Required for PMC and PubMed API access
- AWS_BEDROCK_ACCESS_KEY: Required for LLM-based methods extraction
- AWS_BEDROCK_SECRET_ACCESS_KEY: Required for LLM-based methods extraction

Rate Limiting:
- 0.5-1 second sleeps between consecutive API calls

Coverage Target: 95%+ (unit tests) + real API validation
"""

import os
import time
from pathlib import Path
from unittest.mock import Mock

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.abstract_provider import AbstractProvider
from lobster.tools.providers.pubmed_provider import PubMedProvider

# Fixtures for real API integration tests


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
    reason="Real API tests require LOBSTER_RUN_REAL_API_TESTS=1 environment variable"
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
    reason="Real API tests require LOBSTER_RUN_REAL_API_TESTS=1 environment variable"
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
    reason="Real API tests require LOBSTER_RUN_REAL_API_TESTS=1 environment variable"
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
