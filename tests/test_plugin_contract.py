"""Contract compliance tests for plugin packages.

These tests validate that agent packages follow the plugin API contract.
Run with: pytest tests/test_plugin_contract.py -v

This module serves two purposes:
1. Validates that the core lobster package follows its own contract
2. Provides examples for agent package authors to validate their packages
"""

import dataclasses
import inspect

import pytest


class TestFactorySignature:
    """Verify agent factories follow standardized signature.

    All agent factories must accept these standard parameters:
    - data_manager: DataManagerV2 instance
    - callback_handler: Optional callback for streaming
    - agent_name: Name for logging/attribution
    - delegation_tools: List of tools for child handoffs
    - workspace_path: Optional workspace path override

    Agent-specific params should use **kwargs.
    """

    STANDARD_PARAMS = [
        "data_manager",
        "callback_handler",
        "agent_name",
        "delegation_tools",
        "workspace_path",
    ]

    @pytest.mark.parametrize(
        "agent_module,factory_name",
        [
            ("lobster.agents.research.research_agent", "research_agent"),
            ("lobster.agents.data_expert.data_expert", "data_expert"),
            (
                "lobster.agents.transcriptomics.transcriptomics_expert",
                "transcriptomics_expert",
            ),
            ("lobster.agents.visualization_expert", "visualization_expert"),
            ("lobster.agents.genomics.genomics_expert", "genomics_expert"),
            ("lobster.agents.proteomics.proteomics_expert", "proteomics_expert"),
        ],
    )
    def test_factory_has_standard_params(self, agent_module, factory_name):
        """Factory must accept standard parameters."""
        module = __import__(agent_module, fromlist=[factory_name])
        factory = getattr(module, factory_name)
        sig = inspect.signature(factory)
        params = list(sig.parameters.keys())

        for param in self.STANDARD_PARAMS:
            assert param in params, f"{factory_name} missing parameter: {param}"

    @pytest.mark.parametrize(
        "agent_module,factory_name",
        [
            ("lobster.agents.research.research_agent", "research_agent"),
            ("lobster.agents.data_expert.data_expert", "data_expert"),
            (
                "lobster.agents.transcriptomics.transcriptomics_expert",
                "transcriptomics_expert",
            ),
            ("lobster.agents.visualization_expert", "visualization_expert"),
            ("lobster.agents.genomics.genomics_expert", "genomics_expert"),
            ("lobster.agents.proteomics.proteomics_expert", "proteomics_expert"),
        ],
    )
    def test_no_deprecated_handoff_tools_param(self, agent_module, factory_name):
        """Factory must NOT use deprecated 'handoff_tools' parameter.

        The 'handoff_tools' parameter was renamed to 'delegation_tools'
        in Phase 2 for semantic clarity about child agent delegation.
        """
        module = __import__(agent_module, fromlist=[factory_name])
        factory = getattr(module, factory_name)
        sig = inspect.signature(factory)
        params = list(sig.parameters.keys())

        assert "handoff_tools" not in params, (
            f"{factory_name} uses deprecated 'handoff_tools' parameter. "
            f"Use 'delegation_tools' instead."
        )


class TestAgentRegistryConfig:
    """Verify AgentRegistryConfig has required fields for plugin contract."""

    def test_has_tier_requirement_field(self):
        """AgentRegistryConfig must have tier_requirement field."""
        from lobster.config.agent_registry import AgentRegistryConfig

        fields = {f.name for f in dataclasses.fields(AgentRegistryConfig)}
        assert "tier_requirement" in fields

    def test_has_package_name_field(self):
        """AgentRegistryConfig must have package_name field."""
        from lobster.config.agent_registry import AgentRegistryConfig

        fields = {f.name for f in dataclasses.fields(AgentRegistryConfig)}
        assert "package_name" in fields

    def test_has_service_dependencies_field(self):
        """AgentRegistryConfig must have service_dependencies field."""
        from lobster.config.agent_registry import AgentRegistryConfig

        fields = {f.name for f in dataclasses.fields(AgentRegistryConfig)}
        assert "service_dependencies" in fields

    def test_tier_requirement_default(self):
        """tier_requirement should default to 'free'."""
        from lobster.config.agent_registry import AgentRegistryConfig

        config = AgentRegistryConfig(
            name="test",
            display_name="Test",
            description="Test",
            factory_function="test.module",
        )
        assert config.tier_requirement == "free"

    def test_package_name_default(self):
        """package_name should default to None (core lobster-ai)."""
        from lobster.config.agent_registry import AgentRegistryConfig

        config = AgentRegistryConfig(
            name="test",
            display_name="Test",
            description="Test",
            factory_function="test.module",
        )
        assert config.package_name is None

    def test_service_dependencies_default(self):
        """service_dependencies should default to None."""
        from lobster.config.agent_registry import AgentRegistryConfig

        config = AgentRegistryConfig(
            name="test",
            display_name="Test",
            description="Test",
            factory_function="test.module",
        )
        assert config.service_dependencies is None


class TestComponentConflictError:
    """Verify ComponentConflictError exists for duplicate handling."""

    def test_component_conflict_error_exists(self):
        """ComponentConflictError must be importable."""
        from lobster.core.component_registry import ComponentConflictError

        assert issubclass(ComponentConflictError, Exception)

    def test_component_conflict_error_message(self):
        """ComponentConflictError should accept a message."""
        from lobster.core.component_registry import ComponentConflictError

        error = ComponentConflictError("test conflict")
        assert "test conflict" in str(error)


class TestVersionCompatibility:
    """Verify version compatibility checking exists."""

    def test_check_plugin_compatibility_exists(self):
        """check_plugin_compatibility function must be importable."""
        from lobster.core.component_registry import check_plugin_compatibility

        assert callable(check_plugin_compatibility)

    def test_check_plugin_compatibility_returns_tuple(self):
        """check_plugin_compatibility must return (bool, str) tuple."""
        from lobster.core.component_registry import check_plugin_compatibility

        result = check_plugin_compatibility("nonexistent-package")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_check_plugin_compatibility_nonexistent_package(self):
        """check_plugin_compatibility returns False for nonexistent packages."""
        from lobster.core.component_registry import check_plugin_compatibility

        is_compatible, message = check_plugin_compatibility("nonexistent-package-xyz")
        assert is_compatible is False
        assert "not installed" in message.lower()


class TestComponentRegistry:
    """Verify ComponentRegistry core functionality."""

    def test_component_registry_singleton_exists(self):
        """component_registry singleton must be importable."""
        from lobster.core.component_registry import component_registry

        assert component_registry is not None

    def test_component_registry_has_load_components(self):
        """ComponentRegistry must have load_components method."""
        from lobster.core.component_registry import component_registry

        assert hasattr(component_registry, "load_components")
        assert callable(component_registry.load_components)

    def test_component_registry_has_list_agents(self):
        """ComponentRegistry must have list_agents method."""
        from lobster.core.component_registry import component_registry

        assert hasattr(component_registry, "list_agents")
        assert callable(component_registry.list_agents)

    def test_component_registry_has_get_service(self):
        """ComponentRegistry must have get_service method."""
        from lobster.core.component_registry import component_registry

        assert hasattr(component_registry, "get_service")
        assert callable(component_registry.get_service)
