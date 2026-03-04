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
