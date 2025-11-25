"""
Integration tests for ContentAccessService (Phase 1 stub).

These tests verify that the ContentAccessService correctly initializes
all providers and registers them with the ProviderRegistry. Uses real
provider implementations (not mocks) to validate the actual system behavior.
"""

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.data_access.content_access_service import ContentAccessService
from lobster.tools.providers.base_provider import DatasetType, ProviderCapability


class TestProviderRegistryInitialization:
    """Test provider registry initialization in ContentAccessService."""

    @pytest.fixture
    def data_manager(self):
        """Create DataManagerV2 instance for testing."""
        return DataManagerV2()

    @pytest.fixture
    def service(self, data_manager):
        """Create ContentAccessService instance for testing."""
        return ContentAccessService(data_manager)

    def test_all_providers_registered(self, service):
        """Test that all 5 providers are registered correctly."""
        all_providers = service.registry.get_all_providers()

        # Should have exactly 5 providers (Phase 1)
        assert len(all_providers) == 5, (
            f"Expected 5 providers, got {len(all_providers)}: "
            f"{[type(p).__name__ for p in all_providers]}"
        )

        # Verify provider types
        provider_names = {type(p).__name__ for p in all_providers}
        expected_providers = {
            "AbstractProvider",
            "PubMedProvider",
            "GEOProvider",
            "PMCProvider",
            "WebpageProvider",
        }

        assert provider_names == expected_providers, (
            f"Provider mismatch. Expected: {expected_providers}, "
            f"Got: {provider_names}"
        )

    def test_capability_matrix_correct(self, service):
        """Test that capability → provider mappings are correct."""
        # Test SEARCH_LITERATURE capability (should have PubMedProvider)
        search_providers = service.registry.get_providers_for_capability(
            ProviderCapability.SEARCH_LITERATURE
        )
        assert len(search_providers) > 0, "No providers for SEARCH_LITERATURE"
        assert any(
            type(p).__name__ == "PubMedProvider" for p in search_providers
        ), "PubMedProvider not found for SEARCH_LITERATURE"

        # Test DISCOVER_DATASETS capability (should have GEOProvider)
        discover_providers = service.registry.get_providers_for_capability(
            ProviderCapability.DISCOVER_DATASETS
        )
        assert len(discover_providers) > 0, "No providers for DISCOVER_DATASETS"
        assert any(
            type(p).__name__ == "GEOProvider" for p in discover_providers
        ), "GEOProvider not found for DISCOVER_DATASETS"

        # Test GET_FULL_CONTENT capability (should have PMCProvider and WebpageProvider)
        fulltext_providers = service.registry.get_providers_for_capability(
            ProviderCapability.GET_FULL_CONTENT
        )
        assert len(fulltext_providers) >= 2, (
            f"Expected at least 2 providers for GET_FULL_CONTENT, "
            f"got {len(fulltext_providers)}"
        )

        fulltext_names = {type(p).__name__ for p in fulltext_providers}
        assert (
            "PMCProvider" in fulltext_names
        ), "PMCProvider not found for GET_FULL_CONTENT"
        assert (
            "WebpageProvider" in fulltext_names
        ), "WebpageProvider not found for GET_FULL_CONTENT"

        # Test GET_ABSTRACT capability (should have AbstractProvider and PubMedProvider)
        abstract_providers = service.registry.get_providers_for_capability(
            ProviderCapability.GET_ABSTRACT
        )
        assert len(abstract_providers) >= 2, (
            f"Expected at least 2 providers for GET_ABSTRACT, "
            f"got {len(abstract_providers)}"
        )

        abstract_names = {type(p).__name__ for p in abstract_providers}
        assert (
            "AbstractProvider" in abstract_names
        ), "AbstractProvider not found for GET_ABSTRACT"

    def test_dataset_types_registered(self, service):
        """Test that GEO dataset type is properly registered."""
        # Get provider for GEO dataset type
        geo_provider = service.registry.get_provider_for_dataset_type(DatasetType.GEO)

        assert geo_provider is not None, "No provider found for GEO dataset type"
        assert type(geo_provider).__name__ == "GEOProvider", (
            f"Expected GEOProvider for GEO datasets, "
            f"got {type(geo_provider).__name__}"
        )

        # Verify supported dataset types list
        dataset_types = service.registry.get_supported_dataset_types()
        assert DatasetType.GEO in dataset_types, "GEO not in supported dataset types"

    def test_get_provider_for_capability_priority_ordering(self, service):
        """Test that providers are returned in correct priority order."""
        # Test GET_FULL_CONTENT capability
        # Should return PMCProvider (priority 10) before WebpageProvider (priority 50)
        fulltext_providers = service.registry.get_providers_for_capability(
            ProviderCapability.GET_FULL_CONTENT
        )

        # Verify we have multiple providers
        assert (
            len(fulltext_providers) >= 2
        ), "Need at least 2 providers to test priority ordering"

        # First provider should be PMCProvider (priority 10)
        assert type(fulltext_providers[0]).__name__ == "PMCProvider", (
            f"Expected PMCProvider first (priority 10), "
            f"got {type(fulltext_providers[0]).__name__} "
            f"(priority {fulltext_providers[0].priority})"
        )

        # Second provider should be WebpageProvider (priority 50)
        assert type(fulltext_providers[1]).__name__ == "WebpageProvider", (
            f"Expected WebpageProvider second (priority 50), "
            f"got {type(fulltext_providers[1]).__name__} "
            f"(priority {fulltext_providers[1].priority})"
        )

        # Verify priorities are correct
        assert (
            fulltext_providers[0].priority == 10
        ), "PMCProvider should have priority 10"
        assert (
            fulltext_providers[1].priority == 50
        ), "WebpageProvider should have priority 50"

        # Verify priorities are in ascending order (lower = higher priority)
        priorities = [p.priority for p in fulltext_providers]
        assert priorities == sorted(
            priorities
        ), f"Providers not sorted by priority: {priorities}"

    def test_high_priority_providers(self, service):
        """Test that high-priority providers (priority 10) are registered."""
        all_providers = service.registry.get_all_providers()

        # Count providers with priority 10 (high priority)
        high_priority_providers = [p for p in all_providers if p.priority == 10]

        # Should have 4 high-priority providers:
        # AbstractProvider, PubMedProvider, GEOProvider, PMCProvider
        assert len(high_priority_providers) == 4, (
            f"Expected 4 high-priority providers (priority 10), "
            f"got {len(high_priority_providers)}: "
            f"{[type(p).__name__ for p in high_priority_providers]}"
        )

        high_priority_names = {type(p).__name__ for p in high_priority_providers}
        expected_high_priority = {
            "AbstractProvider",
            "PubMedProvider",
            "GEOProvider",
            "PMCProvider",
        }

        assert high_priority_names == expected_high_priority, (
            f"High-priority provider mismatch. "
            f"Expected: {expected_high_priority}, Got: {high_priority_names}"
        )

    def test_medium_priority_provider(self, service):
        """Test that medium-priority provider (priority 50) is registered."""
        all_providers = service.registry.get_all_providers()

        # Find medium-priority provider (WebpageProvider)
        medium_priority_providers = [p for p in all_providers if p.priority == 50]

        # Should have exactly 1 medium-priority provider
        assert len(medium_priority_providers) == 1, (
            f"Expected 1 medium-priority provider (priority 50), "
            f"got {len(medium_priority_providers)}"
        )

        assert type(medium_priority_providers[0]).__name__ == "WebpageProvider", (
            f"Expected WebpageProvider as medium-priority provider, "
            f"got {type(medium_priority_providers[0]).__name__}"
        )

    def test_capability_matrix_generation(self, service):
        """Test that capability matrix can be generated without errors."""
        # This also serves as a smoke test for the matrix generation
        matrix = service.registry.get_capability_matrix()

        # Basic validation
        assert "Provider Capability Matrix" in matrix
        assert len(matrix) > 0

        # Should contain all registered provider names
        assert "AbstractProvider" in matrix
        assert "PubMedProvider" in matrix
        assert "GEOProvider" in matrix
        assert "PMCProvider" in matrix
        assert "WebpageProvider" in matrix

        # Should contain some capabilities
        assert "search_literature" in matrix or "SEARCH_LITERATURE" in matrix
        assert "get_abstract" in matrix or "GET_ABSTRACT" in matrix
        assert "get_full_content" in matrix or "GET_FULL_CONTENT" in matrix

    def test_query_capabilities_supported_by_all(self, service):
        """Test that all providers support QUERY_CAPABILITIES."""
        all_providers = service.registry.get_all_providers()

        for provider in all_providers:
            capabilities = provider.get_supported_capabilities()
            assert (
                ProviderCapability.QUERY_CAPABILITIES in capabilities
            ), f"{type(provider).__name__} missing QUERY_CAPABILITIES"
            assert (
                capabilities[ProviderCapability.QUERY_CAPABILITIES] is True
            ), f"{type(provider).__name__} has QUERY_CAPABILITIES=False"
