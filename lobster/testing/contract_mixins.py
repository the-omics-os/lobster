"""
Contract test mixins for validating plugin API compliance.

This module provides test mixins that agent package authors can inherit
to validate their packages follow the plugin contract established in Phase 2.

Usage:
    from lobster.testing import AgentContractTestMixin

    class TestMyAgent(AgentContractTestMixin):
        agent_module = 'my_package.agents.my_agent'
        factory_name = 'my_agent'

    # Then run with pytest
"""

import importlib
import inspect
from typing import Any, ClassVar, Optional, Set


class AgentContractTestMixin:
    """
    Test mixin for validating agent package plugin API compliance.

    Subclasses must define:
        agent_module: str - Module path containing AGENT_CONFIG (entry point module)
        factory_name: str - Name of the factory function

    Optional:
        factory_module: str - Module containing the factory (defaults to agent_module)
        expected_tier: str - Expected tier requirement (default: None, skips check)

    Example (same module for config and factory):
        class TestAnnotationExpert(AgentContractTestMixin):
            agent_module = 'lobster_transcriptomics.annotation_expert'
            factory_name = 'annotation_expert'

    Example (separate modules - package __init__ has config, nested module has factory):
        class TestTranscriptomicsExpert(AgentContractTestMixin):
            agent_module = 'lobster_transcriptomics'  # AGENT_CONFIG in __init__.py
            factory_module = 'lobster_transcriptomics.transcriptomics_expert'  # factory here
            factory_name = 'transcriptomics_expert'
    """

    # Required attributes - subclasses must define these
    agent_module: ClassVar[str]
    factory_name: ClassVar[str]

    # Optional attributes
    factory_module: ClassVar[Optional[str]] = (
        None  # Defaults to agent_module if not set
    )
    expected_tier: ClassVar[Optional[str]] = None

    # Standard factory parameters as defined in Phase 2 plugin contract
    STANDARD_PARAMS: ClassVar[Set[str]] = {
        "data_manager",
        "callback_handler",
        "agent_name",
        "delegation_tools",
        "workspace_path",
    }

    # Deprecated parameters that should not be used
    DEPRECATED_PARAMS: ClassVar[Set[str]] = {
        "handoff_tools",  # Renamed to delegation_tools in Phase 2
    }

    # Required AGENT_CONFIG fields
    REQUIRED_CONFIG_FIELDS: ClassVar[Set[str]] = {
        "name",
        "tier_requirement",
    }

    def _get_module(self) -> Any:
        """Import and return the config module (contains AGENT_CONFIG)."""
        return importlib.import_module(self.agent_module)

    def _get_factory_module(self) -> Any:
        """Import and return the factory module (contains factory function)."""
        module_path = self.factory_module if self.factory_module else self.agent_module
        return importlib.import_module(module_path)

    def _get_factory(self) -> Any:
        """Get the factory function from the factory module."""
        module = self._get_factory_module()
        return getattr(module, self.factory_name)

    def _get_factory_params(self) -> Set[str]:
        """Get the set of parameter names from the factory signature."""
        factory = self._get_factory()
        sig = inspect.signature(factory)
        return set(sig.parameters.keys())

    def _get_agent_config(self) -> Any:
        """Get the AGENT_CONFIG from the module."""
        module = self._get_module()
        return getattr(module, "AGENT_CONFIG", None)

    def test_factory_has_standard_params(self) -> None:
        """
        Verify factory function has the standard parameters.

        The plugin contract requires these parameters:
        - data_manager: DataManagerV2 instance
        - callback_handler: Optional callback handler for LLM interactions
        - agent_name: Name identifier for the agent instance
        - delegation_tools: List of tools for delegating to child agents
        - workspace_path: Optional path to workspace directory
        """
        params = self._get_factory_params()
        missing = self.STANDARD_PARAMS - params

        assert not missing, (
            f"Factory '{self.factory_name}' in '{self.agent_module}' "
            f"is missing required parameters: {sorted(missing)}. "
            f"Factory must accept: {sorted(self.STANDARD_PARAMS)}"
        )

    def test_no_deprecated_handoff_tools(self) -> None:
        """
        Verify factory does NOT use deprecated handoff_tools parameter.

        The handoff_tools parameter was renamed to delegation_tools in Phase 2
        for semantic clarity. Factories should use delegation_tools instead.
        """
        params = self._get_factory_params()
        deprecated_used = self.DEPRECATED_PARAMS & params

        assert not deprecated_used, (
            f"Factory '{self.factory_name}' in '{self.agent_module}' "
            f"uses deprecated parameters: {sorted(deprecated_used)}. "
            f"Use 'delegation_tools' instead of 'handoff_tools'."
        )

    def test_agent_config_exists(self) -> None:
        """
        Verify AGENT_CONFIG exists at module top level.

        AGENT_CONFIG must be defined at the top of the module (before heavy imports)
        to enable fast entry point discovery (<50ms target).
        """
        config = self._get_agent_config()

        assert config is not None, (
            f"Module '{self.agent_module}' does not have AGENT_CONFIG defined. "
            f"Add: AGENT_CONFIG = AgentRegistryConfig(...) at module top, "
            f"before heavy imports."
        )

    def test_agent_config_has_name(self) -> None:
        """
        Verify AGENT_CONFIG has required 'name' field.

        The name field uniquely identifies the agent in the registry
        and is used for routing, configuration, and logging.
        """
        config = self._get_agent_config()

        assert (
            config is not None
        ), f"Cannot check name field: AGENT_CONFIG not found in '{self.agent_module}'"

        assert hasattr(
            config, "name"
        ), f"AGENT_CONFIG in '{self.agent_module}' is missing 'name' field."

        assert config.name, f"AGENT_CONFIG.name in '{self.agent_module}' is empty."

    def test_agent_config_has_tier_requirement(self) -> None:
        """
        Verify AGENT_CONFIG has required 'tier_requirement' field.

        The tier_requirement field controls runtime access based on
        user's subscription level. Valid values: "free", "premium", "enterprise".
        """
        config = self._get_agent_config()

        assert (
            config is not None
        ), f"Cannot check tier_requirement: AGENT_CONFIG not found in '{self.agent_module}'"

        assert hasattr(
            config, "tier_requirement"
        ), f"AGENT_CONFIG in '{self.agent_module}' is missing 'tier_requirement' field."

        valid_tiers = {"free", "premium", "enterprise"}
        tier = config.tier_requirement

        assert tier in valid_tiers, (
            f"AGENT_CONFIG.tier_requirement in '{self.agent_module}' has invalid value '{tier}'. "
            f"Must be one of: {sorted(valid_tiers)}"
        )

        # Optional: check expected tier if specified
        if self.expected_tier is not None:
            assert (
                tier == self.expected_tier
            ), f"AGENT_CONFIG.tier_requirement is '{tier}', expected '{self.expected_tier}'"

    def test_all_contract_requirements(self) -> None:
        """
        Run all contract validation tests together.

        This is a convenience method that runs all validation checks
        in a single test, useful for quick verification.
        """
        self.test_factory_has_standard_params()
        self.test_no_deprecated_handoff_tools()
        self.test_agent_config_exists()
        self.test_agent_config_has_name()
        self.test_agent_config_has_tier_requirement()
