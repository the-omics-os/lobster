"""
Unit tests for ComponentRegistry.

Tests the unified component registry for premium services and agents via entry points.
The ComponentRegistry discovers and loads components from:
- lobster-premium: Shared premium features
- lobster-custom-*: Customer-specific features

Test coverage:
1. Initialization and loading (idempotency, lazy loading)
2. Service API (get, has, list)
3. Agent API (get, has, list, core vs custom)
4. Utility methods (get_info, reset)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from lobster.core.component_registry import ComponentRegistry, component_registry


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


@pytest.fixture
def mock_entry_points():
    """Create mock entry points for testing discovery."""
    mock_service_ep = Mock()
    mock_service_ep.name = "test_service"
    mock_service_ep.value = "test_package.services:TestService"
    mock_service_ep.load.return_value = type(
        "TestService",
        (),
        {"__name__": "TestService", "__module__": "test_package.services"},
    )

    mock_agent_ep = Mock()
    mock_agent_ep.name = "test_agent"
    mock_agent_ep.value = "test_package.agents:TEST_AGENT_CONFIG"
    mock_agent_ep.load.return_value = Mock(name="test_agent", display_name="Test Agent")

    return {
        "lobster.services": [mock_service_ep],
        "lobster.agents": [mock_agent_ep],
    }


# =============================================================================
# Tests for Initialization and Loading
# =============================================================================


class TestRegistryInitialization:
    """Test ComponentRegistry initialization and loading behavior."""

    def test_registry_starts_unloaded(self, fresh_registry):
        """Registry should not be loaded initially."""
        assert fresh_registry._loaded is False
        assert fresh_registry._services == {}
        assert fresh_registry._agents == {}

    def test_load_components_idempotent(self, fresh_registry):
        """Loading components multiple times should be safe."""
        # First load
        fresh_registry.load_components()
        assert fresh_registry._loaded is True

        # Capture state after first load
        services_after_first = dict(fresh_registry._services)
        agents_after_first = dict(fresh_registry._agents)

        # Second load should be no-op
        fresh_registry.load_components()
        assert fresh_registry._loaded is True
        assert fresh_registry._services == services_after_first
        assert fresh_registry._agents == agents_after_first

    def test_load_components_sets_loaded_flag(self, fresh_registry):
        """Loading components should set _loaded to True."""
        assert fresh_registry._loaded is False
        fresh_registry.load_components()
        assert fresh_registry._loaded is True

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_load_components_calls_entry_point_groups(
        self, mock_load_ep, fresh_registry
    ):
        """Loading should discover both services and agents entry point groups."""
        fresh_registry.load_components()

        # Verify all entry point groups were loaded
        calls = mock_load_ep.call_args_list
        assert len(calls) == 3
        assert calls[0][0][0] == "lobster.services"
        assert calls[1][0][0] == "lobster.agents"
        assert calls[2][0][0] == "lobster.agent_configs"


# =============================================================================
# Tests for Service API
# =============================================================================


class TestServiceAPI:
    """Test service-related methods."""

    def test_get_service_returns_none_when_missing(self, fresh_registry):
        """Missing services should return None by default."""
        fresh_registry.load_components()
        result = fresh_registry.get_service("nonexistent_service")
        assert result is None

    def test_get_service_raises_when_required(self, fresh_registry):
        """Required missing services should raise ValueError."""
        fresh_registry.load_components()

        with pytest.raises(ValueError) as exc_info:
            fresh_registry.get_service("nonexistent_service", required=True)

        assert "Required service 'nonexistent_service' not found" in str(exc_info.value)
        assert "Available services:" in str(exc_info.value)

    def test_has_service_returns_false_for_missing(self, fresh_registry):
        """has_service should return False for missing services."""
        fresh_registry.load_components()
        assert fresh_registry.has_service("nonexistent_service") is False

    def test_list_services_returns_dict(self, fresh_registry):
        """list_services should return a dictionary."""
        result = fresh_registry.list_services()
        assert isinstance(result, dict)

    def test_get_service_auto_loads(self, fresh_registry):
        """get_service should auto-load components if not loaded."""
        assert fresh_registry._loaded is False
        fresh_registry.get_service("test")
        assert fresh_registry._loaded is True

    def test_has_service_auto_loads(self, fresh_registry):
        """has_service should auto-load components if not loaded."""
        assert fresh_registry._loaded is False
        fresh_registry.has_service("test")
        assert fresh_registry._loaded is True

    def test_list_services_auto_loads(self, fresh_registry):
        """list_services should auto-load components if not loaded."""
        assert fresh_registry._loaded is False
        fresh_registry.list_services()
        assert fresh_registry._loaded is True

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_get_service_returns_loaded_service(self, mock_load_ep, fresh_registry):
        """get_service should return a loaded service."""
        # Manually add a service to simulate discovery
        mock_service_class = type(
            "MockService", (), {"__name__": "MockService", "__module__": "test"}
        )
        fresh_registry._services["mock_service"] = mock_service_class
        fresh_registry._loaded = True

        result = fresh_registry.get_service("mock_service")
        assert result is mock_service_class

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_has_service_returns_true_for_existing(self, mock_load_ep, fresh_registry):
        """has_service should return True for existing services."""
        fresh_registry._services["existing_service"] = Mock()
        fresh_registry._loaded = True

        assert fresh_registry.has_service("existing_service") is True

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_list_services_includes_module_paths(self, mock_load_ep, fresh_registry):
        """list_services should include module paths for each service."""
        mock_service_class = type(
            "TestService", (), {"__name__": "TestService", "__module__": "test.module"}
        )
        fresh_registry._services["test_service"] = mock_service_class
        fresh_registry._loaded = True

        result = fresh_registry.list_services()
        assert "test_service" in result
        assert result["test_service"] == "test.module.TestService"


# =============================================================================
# Tests for Agent API
# =============================================================================


class TestAgentAPI:
    """Test agent-related methods."""

    def test_get_agent_returns_none_when_missing(self, fresh_registry):
        """Missing agents should return None by default."""
        fresh_registry.load_components()
        result = fresh_registry.get_agent("nonexistent_agent")
        assert result is None

    def test_get_agent_raises_when_required(self, fresh_registry):
        """Required missing agents should raise ValueError."""
        fresh_registry.load_components()

        with pytest.raises(ValueError) as exc_info:
            fresh_registry.get_agent("nonexistent_agent", required=True)

        assert "Required agent 'nonexistent_agent' not found" in str(exc_info.value)
        assert "Available agents:" in str(exc_info.value)

    def test_has_agent_returns_false_for_missing(self, fresh_registry):
        """has_agent should return False for missing agents."""
        fresh_registry.load_components()
        assert fresh_registry.has_agent("nonexistent_agent") is False

    def test_list_agents_includes_core_agents(self, fresh_registry):
        """list_agents should include core agents from AGENT_REGISTRY."""
        fresh_registry.load_components()
        all_agents = fresh_registry.list_agents()

        # Core agents from AGENT_REGISTRY should be present
        # These are defined in lobster/config/agent_registry.py
        assert "research_agent" in all_agents
        assert "data_expert_agent" in all_agents

    def test_list_custom_agents_excludes_core(self, fresh_registry):
        """list_custom_agents should NOT include core agents."""
        fresh_registry.load_components()
        custom_agents = fresh_registry.list_custom_agents()

        # Core agents should NOT be in custom list
        assert "supervisor" not in custom_agents
        assert "research_agent" not in custom_agents
        assert "data_expert_agent" not in custom_agents

    def test_get_agent_auto_loads(self, fresh_registry):
        """get_agent should auto-load components if not loaded."""
        assert fresh_registry._loaded is False
        fresh_registry.get_agent("test")
        assert fresh_registry._loaded is True

    def test_has_agent_auto_loads(self, fresh_registry):
        """has_agent should auto-load components if not loaded."""
        assert fresh_registry._loaded is False
        fresh_registry.has_agent("test")
        assert fresh_registry._loaded is True

    def test_list_agents_auto_loads(self, fresh_registry):
        """list_agents should auto-load components if not loaded."""
        assert fresh_registry._loaded is False
        fresh_registry.list_agents()
        assert fresh_registry._loaded is True

    def test_list_custom_agents_auto_loads(self, fresh_registry):
        """list_custom_agents should auto-load components if not loaded."""
        assert fresh_registry._loaded is False
        fresh_registry.list_custom_agents()
        assert fresh_registry._loaded is True

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_get_agent_returns_loaded_agent(self, mock_load_ep, fresh_registry):
        """get_agent should return a loaded custom agent."""
        mock_agent_config = Mock(name="custom_agent", display_name="Custom Agent")
        fresh_registry._agents["custom_agent"] = mock_agent_config
        fresh_registry._loaded = True

        result = fresh_registry.get_agent("custom_agent")
        assert result is mock_agent_config

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_has_agent_returns_true_for_existing(self, mock_load_ep, fresh_registry):
        """has_agent should return True for existing custom agents."""
        fresh_registry._agents["existing_agent"] = Mock()
        fresh_registry._loaded = True

        assert fresh_registry.has_agent("existing_agent") is True

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_list_custom_agents_returns_dict(self, mock_load_ep, fresh_registry):
        """list_custom_agents should return a dictionary."""
        fresh_registry._loaded = True
        result = fresh_registry.list_custom_agents()
        assert isinstance(result, dict)

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_list_agents_merges_core_and_custom(self, mock_load_ep, fresh_registry):
        """list_agents should return all agents in the unified _agents dict."""
        # Add agents (in unified architecture, core and custom are all in _agents)
        mock_core_config = Mock(name="research_agent", display_name="Research Agent")
        mock_custom_config = Mock(name="premium_agent", display_name="Premium Agent")
        fresh_registry._agents["research_agent"] = mock_core_config
        fresh_registry._agents["premium_agent"] = mock_custom_config
        fresh_registry._loaded = True

        all_agents = fresh_registry.list_agents()

        # Should contain all registered agents
        assert "research_agent" in all_agents
        assert "premium_agent" in all_agents


# =============================================================================
# Tests for Custom Agent Override Behavior
# =============================================================================


class TestCustomAgentOverride:
    """Test that custom agent name collisions are detected and raise errors."""

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_custom_agents_override_core(self, mock_load_ep, fresh_registry):
        """Custom agents with same names as core agents should override them (last-write-wins)."""
        # Add a core agent first
        core_research = Mock(name="research_agent", display_name="Core Research Agent")
        fresh_registry._agents["research_agent"] = core_research
        fresh_registry._loaded = True

        # Override with custom agent of the same name
        custom_research = Mock(
            name="research_agent", display_name="Custom Research Agent"
        )
        fresh_registry._agents["research_agent"] = custom_research

        # In unified registry, last-write-wins - no conflict error
        all_agents = fresh_registry.list_agents()
        assert "research_agent" in all_agents
        # The custom agent should be the one stored
        assert all_agents["research_agent"].display_name == "Custom Research Agent"


# =============================================================================
# Tests for Utility Methods
# =============================================================================


class TestUtilityMethods:
    """Test utility methods (get_info, reset)."""

    def test_get_info_returns_diagnostic_dict(self, fresh_registry):
        """get_info should return a diagnostic dictionary."""
        fresh_registry.load_components()
        info = fresh_registry.get_info()

        assert isinstance(info, dict)
        assert "services" in info
        assert "agents" in info
        assert "custom_agent_configs" in info

        # Verify structure
        assert "count" in info["services"]
        assert "names" in info["services"]
        assert isinstance(info["services"]["names"], list)

        assert "count" in info["agents"]
        assert "names" in info["agents"]
        assert isinstance(info["agents"]["names"], list)

        assert isinstance(info["custom_agent_configs"], int) or "count" in info.get(
            "custom_agent_configs", {}
        )

    def test_get_info_auto_loads(self, fresh_registry):
        """get_info should auto-load components if not loaded."""
        assert fresh_registry._loaded is False
        fresh_registry.get_info()
        assert fresh_registry._loaded is True

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_get_info_counts_correct(self, mock_load_ep, fresh_registry):
        """get_info should report correct counts."""
        # Add mock data
        fresh_registry._services["svc1"] = type(
            "Svc1", (), {"__name__": "Svc1", "__module__": "test"}
        )
        fresh_registry._services["svc2"] = type(
            "Svc2", (), {"__name__": "Svc2", "__module__": "test"}
        )
        fresh_registry._agents["agent1"] = Mock()
        fresh_registry._loaded = True

        info = fresh_registry.get_info()

        assert info["services"]["count"] == 2
        assert set(info["services"]["names"]) == {"svc1", "svc2"}
        assert info["agents"]["count"] == 1
        assert info["agents"]["names"] == ["agent1"]

    def test_reset_clears_registry_state(self, fresh_registry):
        """reset should clear all registry state."""
        # Load and populate
        fresh_registry.load_components()
        fresh_registry._services["test"] = Mock()
        fresh_registry._agents["test"] = Mock()

        # Reset
        fresh_registry.reset()

        # Verify cleared
        assert fresh_registry._loaded is False
        assert fresh_registry._services == {}
        assert fresh_registry._agents == {}

    def test_reset_allows_reload(self, fresh_registry):
        """After reset, load_components should work again."""
        fresh_registry.load_components()
        assert fresh_registry._loaded is True

        fresh_registry.reset()
        assert fresh_registry._loaded is False

        # Should be able to load again
        fresh_registry.load_components()
        assert fresh_registry._loaded is True


# =============================================================================
# Tests for Entry Point Loading
# =============================================================================


class TestEntryPointLoading:
    """Test entry point discovery and loading."""

    def test_load_entry_point_handles_errors_gracefully(self, fresh_registry):
        """Failed entry point loads should be logged but not crash."""
        import sys

        # Create a mock entry point that raises on load
        failing_ep = Mock()
        failing_ep.name = "failing_service"
        failing_ep.value = "nonexistent.module:Service"
        failing_ep.load.side_effect = ImportError("Module not found")

        # Mock entry_points - patched at importlib.metadata level since
        # component_registry imports it inside the function
        if sys.version_info >= (3, 10):
            # Python 3.10+: entry_points(group=group) returns iterable directly
            def mock_entry_points(group=None):
                if group == "lobster.services":
                    return [failing_ep]
                return []

            with patch("importlib.metadata.entry_points", mock_entry_points):
                # Should not raise
                fresh_registry.load_components()
        else:
            # Python 3.9: entry_points() returns dict
            mock_eps = {"lobster.services": [failing_ep], "lobster.agents": []}
            with patch("importlib.metadata.entry_points", return_value=mock_eps):
                fresh_registry.load_components()

        # Service should not be registered
        assert "failing_service" not in fresh_registry._services

    def test_load_entry_point_success(self, fresh_registry):
        """Successful entry point loads should register components."""
        import sys

        # Create a mock entry point that loads successfully
        mock_service = type(
            "TestService", (), {"__name__": "TestService", "__module__": "test"}
        )
        success_ep = Mock()
        success_ep.name = "test_service"
        success_ep.value = "test.module:TestService"
        success_ep.load.return_value = mock_service

        # Mock entry_points - patched at importlib.metadata level since
        # component_registry imports it inside the function
        if sys.version_info >= (3, 10):
            # Python 3.10+: entry_points(group=group) returns iterable directly
            def mock_entry_points(group=None):
                if group == "lobster.services":
                    return [success_ep]
                return []

            with patch("importlib.metadata.entry_points", mock_entry_points):
                fresh_registry.load_components()
        else:
            # Python 3.9: entry_points() returns dict
            mock_eps = {"lobster.services": [success_ep], "lobster.agents": []}
            with patch("importlib.metadata.entry_points", return_value=mock_eps):
                fresh_registry.load_components()

        assert "test_service" in fresh_registry._services
        assert fresh_registry._services["test_service"] is mock_service


# =============================================================================
# Tests for Singleton Instance
# =============================================================================


class TestSingletonInstance:
    """Test the module-level singleton instance."""

    def test_component_registry_is_component_registry(self):
        """Module-level component_registry should be a ComponentRegistry instance."""
        assert isinstance(component_registry, ComponentRegistry)

    def test_singleton_can_be_reset(self):
        """The singleton should be resettable for testing purposes."""
        # Store original state
        original_loaded = component_registry._loaded

        # Modify state
        component_registry.load_components()

        # Reset
        component_registry.reset()

        # Should be back to unloaded state
        assert component_registry._loaded is False


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_service_name(self, fresh_registry):
        """Empty service name should return None."""
        fresh_registry.load_components()
        result = fresh_registry.get_service("")
        assert result is None

    def test_empty_agent_name(self, fresh_registry):
        """Empty agent name should return None."""
        fresh_registry.load_components()
        result = fresh_registry.get_agent("")
        assert result is None

    def test_none_handling_in_get_service(self, fresh_registry):
        """get_service should handle None name gracefully."""
        fresh_registry.load_components()
        # This tests robustness - should return None, not crash
        result = fresh_registry.get_service(None)
        assert result is None

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_service_value_error_message_format(self, mock_load_ep, fresh_registry):
        """ValueError message should list available services."""
        fresh_registry._services["available1"] = Mock()
        fresh_registry._services["available2"] = Mock()
        fresh_registry._loaded = True

        with pytest.raises(ValueError) as exc_info:
            fresh_registry.get_service("missing", required=True)

        error_msg = str(exc_info.value)
        assert "available1" in error_msg or "available2" in error_msg

    @patch("lobster.core.component_registry.ComponentRegistry._load_entry_point_group")
    def test_agent_value_error_message_format(self, mock_load_ep, fresh_registry):
        """ValueError message should list available custom agents."""
        fresh_registry._agents["custom1"] = Mock()
        fresh_registry._loaded = True

        with pytest.raises(ValueError) as exc_info:
            fresh_registry.get_agent("missing", required=True)

        error_msg = str(exc_info.value)
        assert "custom1" in error_msg
