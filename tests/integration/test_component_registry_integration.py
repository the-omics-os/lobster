"""
Integration tests for ComponentRegistry.

Tests that verify the ComponentRegistry works correctly with the rest of the system:
1. Backward compatibility with plugin_loader's discover_plugins()
2. Custom agent override behavior
3. Service class validity

These tests exercise real component interactions rather than mocked behavior.
"""

from typing import Any, Dict

import pytest

from lobster.config.agent_registry import AGENT_REGISTRY, AgentRegistryConfig
from lobster.core.component_registry import (
    ComponentConflictError,
    ComponentRegistry,
    component_registry,
)

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fresh_registry():
    """Create a fresh ComponentRegistry instance for testing."""
    registry = ComponentRegistry()
    yield registry
    # Cleanup
    registry.reset()


@pytest.fixture(autouse=True)
def reset_singleton_after_test():
    """Reset the singleton registry after each test."""
    yield
    component_registry.reset()


# =============================================================================
# Tests for Plugin Loader Backward Compatibility
# =============================================================================


class TestPluginLoaderBackwardCompatibility:
    """Test backward compatibility with plugin_loader."""

    def test_plugin_loader_backward_compatibility(self, fresh_registry):
        """discover_plugins() returns all agents including core via list_agents()."""
        # Load components
        fresh_registry.load_components()

        # list_agents() should include all core agents from AGENT_REGISTRY
        all_agents = fresh_registry.list_agents()

        # Verify core agents are present
        for agent_name in AGENT_REGISTRY.keys():
            assert (
                agent_name in all_agents
            ), f"Core agent '{agent_name}' should be in list_agents()"

        # Verify all returned agents are AgentRegistryConfig or similar
        for name, config in all_agents.items():
            # Should have required attributes that AgentRegistryConfig has
            assert hasattr(config, "name") or isinstance(
                config, dict
            ), f"Agent '{name}' should have a 'name' attribute or be a dict"

    def test_list_agents_includes_agent_registry_entries(self, fresh_registry):
        """list_agents() should include all entries from AGENT_REGISTRY."""
        fresh_registry.load_components()
        all_agents = fresh_registry.list_agents()

        # Get expected agents from AGENT_REGISTRY
        expected_core_agents = set(AGENT_REGISTRY.keys())

        # All expected agents should be present
        for agent_name in expected_core_agents:
            assert (
                agent_name in all_agents
            ), f"Expected core agent '{agent_name}' not found in list_agents()"

    def test_agent_registry_config_structure_preserved(self, fresh_registry):
        """Agent configs from AGENT_REGISTRY should preserve their structure."""
        fresh_registry.load_components()
        all_agents = fresh_registry.list_agents()

        # Check a known core agent
        research_agent = all_agents.get("research_agent")
        assert research_agent is not None, "research_agent should be in list_agents()"

        # Verify structure
        assert isinstance(
            research_agent, AgentRegistryConfig
        ), "Core agents should be AgentRegistryConfig instances"
        assert research_agent.name == "research_agent"
        assert research_agent.display_name is not None
        assert research_agent.factory_function is not None


# =============================================================================
# Tests for Custom Agent Override
# =============================================================================


class TestCustomAgentOverride:
    """Test custom agent collision detection and merging behavior."""

    def test_custom_agents_name_collision_raises_error(self, fresh_registry):
        """Custom agents with same name as core agents raise ComponentConflictError."""
        fresh_registry.load_components()

        # Simulate a custom agent with same name as core agent
        custom_agent_config = AgentRegistryConfig(
            name="research_agent",  # Collides with core agent
            display_name="Custom Research Agent",
            description="A custom override of research_agent",
            factory_function="custom.module:research_agent",
            handoff_tool_name="handoff_to_custom_research",
            handoff_tool_description="Custom handoff description",
        )
        fresh_registry._custom_agents["research_agent"] = custom_agent_config

        # Attempting to list agents should raise ComponentConflictError
        with pytest.raises(ComponentConflictError) as exc_info:
            fresh_registry.list_agents()

        # Verify error message contains the conflicting name
        assert "research_agent" in str(exc_info.value)
        assert "collision" in str(exc_info.value).lower()

    def test_list_custom_agents_only_returns_custom(self, fresh_registry):
        """list_custom_agents() should only return custom agents, not core."""
        fresh_registry.load_components()

        # Add a custom agent
        custom_config = AgentRegistryConfig(
            name="my_custom_agent",
            display_name="My Custom Agent",
            description="A custom agent",
            factory_function="custom.module:my_agent",
        )
        fresh_registry._custom_agents["my_custom_agent"] = custom_config

        custom_agents = fresh_registry.list_custom_agents()

        # Should contain custom agent
        assert "my_custom_agent" in custom_agents

        # Should NOT contain core agents
        for core_name in AGENT_REGISTRY.keys():
            assert (
                core_name not in custom_agents
            ), f"Core agent '{core_name}' should not be in list_custom_agents()"

    def test_custom_agent_preserves_core_agents(self, fresh_registry):
        """Adding a custom agent should not affect core agents."""
        fresh_registry.load_components()

        # Add a custom agent with UNIQUE name (no collision)
        custom_config = AgentRegistryConfig(
            name="my_custom_agent",  # Unique name
            display_name="Custom Agent",
            description="Custom",
            factory_function="custom:agent",
        )
        fresh_registry._custom_agents["my_custom_agent"] = custom_config

        all_agents = fresh_registry.list_agents()

        # All core agents should be present and unchanged
        for core_name in AGENT_REGISTRY.keys():
            assert (
                core_name in all_agents
            ), f"Core agent '{core_name}' should be present"
            assert (
                all_agents[core_name] is AGENT_REGISTRY[core_name]
            ), f"Core agent '{core_name}' should be unchanged"

        # Custom agent should also be present
        assert "my_custom_agent" in all_agents
        assert all_agents["my_custom_agent"] is custom_config


# =============================================================================
# Tests for Service Validity
# =============================================================================


class TestServiceValidity:
    """Test that discovered services are valid classes."""

    def test_services_are_callable(self, fresh_registry):
        """Discovered services should be actual classes (have __name__)."""
        fresh_registry.load_components()

        services = fresh_registry.list_services()

        # If there are any services discovered, verify they are valid
        for service_name, module_path in services.items():
            service_class = fresh_registry.get_service(service_name)
            assert (
                service_class is not None
            ), f"Service '{service_name}' should be retrievable"

            # Should be a class (has __name__)
            assert hasattr(
                service_class, "__name__"
            ), f"Service '{service_name}' should have __name__ attribute"

            # Module path should match
            expected_path = f"{service_class.__module__}.{service_class.__name__}"
            assert (
                module_path == expected_path
            ), f"Service '{service_name}' module path mismatch"

    def test_service_classes_are_instantiable_pattern(self, fresh_registry):
        """Service classes should follow callable pattern (can be instantiated)."""
        fresh_registry.load_components()

        services = fresh_registry.list_services()

        for service_name in services.keys():
            service_class = fresh_registry.get_service(service_name)
            # Verify it looks like a class (callable)
            assert callable(
                service_class
            ), f"Service '{service_name}' should be callable (a class)"


# =============================================================================
# Tests for Registry Info
# =============================================================================


class TestRegistryInfo:
    """Test get_info() returns accurate information."""

    def test_get_info_reflects_actual_state(self, fresh_registry):
        """get_info() should accurately reflect registry state."""
        fresh_registry.load_components()

        info = fresh_registry.get_info()

        # Service count should match actual services
        assert info["services"]["count"] == len(fresh_registry._services)
        assert set(info["services"]["names"]) == set(fresh_registry._services.keys())

        # Custom agent count should match actual custom agents
        assert info["custom_agents"]["count"] == len(fresh_registry._custom_agents)
        assert set(info["custom_agents"]["names"]) == set(
            fresh_registry._custom_agents.keys()
        )

        # Total agents should include core + custom
        all_agents = fresh_registry.list_agents()
        assert info["total_agents"] == len(all_agents)

    def test_get_info_total_agents_includes_core(self, fresh_registry):
        """total_agents in get_info() should include core agents."""
        fresh_registry.load_components()

        info = fresh_registry.get_info()

        # Total should be at least the number of core agents
        assert info["total_agents"] >= len(AGENT_REGISTRY)


# =============================================================================
# Tests for Real Entry Point Discovery
# =============================================================================


class TestRealEntryPointDiscovery:
    """Test actual entry point discovery (integration with importlib)."""

    def test_entry_point_discovery_does_not_crash(self, fresh_registry):
        """Entry point discovery should complete without errors."""
        # This tests the actual importlib.metadata integration
        try:
            fresh_registry.load_components()
        except Exception as e:
            pytest.fail(f"Entry point discovery crashed: {e}")

    def test_multiple_load_calls_stable(self, fresh_registry):
        """Multiple load_components() calls should produce stable results."""
        fresh_registry.load_components()
        first_services = dict(fresh_registry._services)
        first_agents = dict(fresh_registry._custom_agents)

        # Reset and reload
        fresh_registry.reset()
        fresh_registry.load_components()

        # Results should be the same
        assert set(fresh_registry._services.keys()) == set(first_services.keys())
        assert set(fresh_registry._custom_agents.keys()) == set(first_agents.keys())


# =============================================================================
# Tests for Singleton Behavior
# =============================================================================


class TestSingletonBehavior:
    """Test module-level singleton behavior."""

    def test_singleton_shares_state(self):
        """Multiple imports should return the same instance."""
        from lobster.core.component_registry import component_registry as registry1
        from lobster.core.component_registry import component_registry as registry2

        assert registry1 is registry2

    def test_singleton_persists_across_loads(self):
        """Singleton should maintain state across load calls."""
        component_registry.load_components()
        loaded_state = component_registry._loaded
        services_count = len(component_registry._services)

        # Import again and verify state persisted
        from lobster.core.component_registry import (
            component_registry as imported_registry,
        )

        assert imported_registry._loaded == loaded_state
        assert len(imported_registry._services) == services_count


# =============================================================================
# Tests for Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_required_service_error_includes_available(self, fresh_registry):
        """ValueError for required missing service should list available ones."""
        fresh_registry.load_components()

        # Add a test service
        test_service = type(
            "TestService", (), {"__name__": "TestService", "__module__": "test"}
        )
        fresh_registry._services["available_service"] = test_service

        with pytest.raises(ValueError) as exc_info:
            fresh_registry.get_service("nonexistent", required=True)

        error_msg = str(exc_info.value)
        assert "available_service" in error_msg

    def test_required_agent_error_includes_available(self, fresh_registry):
        """ValueError for required missing agent should list available custom ones."""
        fresh_registry.load_components()

        # Add a custom agent
        custom_config = AgentRegistryConfig(
            name="available_custom",
            display_name="Available Custom",
            description="Test",
            factory_function="test:agent",
        )
        fresh_registry._custom_agents["available_custom"] = custom_config

        with pytest.raises(ValueError) as exc_info:
            fresh_registry.get_agent("nonexistent", required=True)

        error_msg = str(exc_info.value)
        assert "available_custom" in error_msg
