"""
Integration tests for dynamic agent discovery (Phase 3).

These tests verify:
1. All agents are discovered via entry points (no hardcoded AGENT_REGISTRY)
2. ComponentRegistry is the single source of truth
3. Discovery completes within performance budget (<50ms)
4. Backward compatibility for existing code paths
"""

import time
import pytest


class TestDynamicAgentDiscovery:
    """Integration tests for the dynamic agent discovery system."""

    def test_discovery_via_entry_points(self):
        """Verify agents are discovered via entry points, not hardcoded dict."""
        from lobster.core.component_registry import component_registry
        import importlib.metadata

        component_registry.reset()

        eps = list(importlib.metadata.entry_points(group='lobster.agents'))
        ep_agent_names = {ep.name for ep in eps}

        registry_agents = component_registry.list_agents()
        registry_agent_names = set(registry_agents.keys())

        # Registry should contain at least all entry point agents
        assert ep_agent_names.issubset(registry_agent_names), (
            f"Entry point agents not in registry: "
            f"{ep_agent_names - registry_agent_names}"
        )

    def test_core_agents_present(self):
        """Verify all 7 core agents are discovered."""
        from lobster.core.component_registry import component_registry

        component_registry.reset()
        agents = component_registry.list_agents()

        core_agents = [
            'data_expert_agent',
            'research_agent',
            'transcriptomics_expert',
            'annotation_expert',
            'de_analysis_expert',
            'genomics_expert',
            'visualization_expert_agent',
        ]

        for agent_name in core_agents:
            assert agent_name in agents, f"Missing core agent: {agent_name}"
            config = agents[agent_name]
            assert config.name == agent_name
            assert config.display_name is not None
            assert config.factory_function is not None

    def test_discovery_performance(self):
        """Verify discovery completes in <50ms."""
        from lobster.core.component_registry import component_registry

        component_registry.reset()

        start = time.perf_counter()
        agents = component_registry.list_agents()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, (
            f"Discovery too slow: {elapsed_ms:.2f}ms > 50ms budget"
        )
        assert len(agents) >= 7, f"Expected 7+ agents, got {len(agents)}"

    def test_no_agent_registry_import(self):
        """Verify ComponentRegistry.list_agents() doesn't import AGENT_REGISTRY."""
        import inspect
        from lobster.core.component_registry import ComponentRegistry

        source = inspect.getsource(ComponentRegistry.list_agents)
        assert 'AGENT_REGISTRY' not in source, (
            "list_agents() should not reference AGENT_REGISTRY"
        )

    def test_tier_requirement_respected(self):
        """Verify tier_requirement field is present and respected."""
        from lobster.core.component_registry import component_registry
        from lobster.config.subscription_tiers import is_agent_available

        component_registry.reset()
        agents = component_registry.list_agents()

        genomics = agents.get('genomics_expert')
        assert genomics is not None
        assert genomics.tier_requirement == 'premium'

        assert not is_agent_available('genomics_expert', 'free')
        assert is_agent_available('genomics_expert', 'premium')

        research = agents.get('research_agent')
        assert research is not None
        assert research.tier_requirement == 'free'
        assert is_agent_available('research_agent', 'free')

    def test_agent_config_fields_complete(self):
        """Verify all AgentRegistryConfig fields are populated."""
        from lobster.core.component_registry import component_registry

        component_registry.reset()
        agents = component_registry.list_agents()

        # Check that essential fields are populated for all agents
        for name, config in agents.items():
            assert config.name == name, f"Agent {name}: name mismatch"
            assert config.display_name, f"Agent {name}: missing display_name"
            assert config.description, f"Agent {name}: missing description"
            assert config.factory_function, f"Agent {name}: missing factory_function"
            assert config.tier_requirement in ('free', 'premium', 'enterprise'), (
                f"Agent {name}: invalid tier_requirement '{config.tier_requirement}'"
            )

    def test_child_agents_relationship(self):
        """Verify child_agents relationships are preserved."""
        from lobster.core.component_registry import component_registry

        component_registry.reset()
        agents = component_registry.list_agents()

        # transcriptomics_expert should have child agents
        trans = agents.get('transcriptomics_expert')
        assert trans is not None
        assert trans.child_agents is not None
        assert 'annotation_expert' in trans.child_agents
        assert 'de_analysis_expert' in trans.child_agents


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_agent_registry_facade(self):
        """Verify agent_registry.py facade functions work."""
        from lobster.config.agent_registry import (
            get_worker_agents,
            get_agent_registry_config,
            get_all_agent_names,
            get_valid_handoffs,
            AgentRegistryConfig,
        )

        agents = get_worker_agents()
        assert len(agents) >= 7
        assert 'research_agent' in agents

        config = get_agent_registry_config('research_agent')
        assert config is not None
        assert isinstance(config, AgentRegistryConfig)

        names = get_all_agent_names()
        assert 'research_agent' in names

        handoffs = get_valid_handoffs()
        assert 'supervisor' in handoffs
        assert isinstance(handoffs['supervisor'], set)

    def test_graph_import_pattern(self):
        """Verify the import pattern used by graph.py still works."""
        from lobster.config.agent_registry import get_worker_agents, import_agent_factory

        agents = get_worker_agents()
        assert len(agents) >= 7

        factory = import_agent_factory(
            'lobster.agents.research.research_agent.research_agent'
        )
        assert callable(factory)

    def test_agent_config_dataclass(self):
        """Verify AgentRegistryConfig dataclass is still usable."""
        from lobster.config.agent_registry import AgentRegistryConfig

        config = AgentRegistryConfig(
            name="test_agent",
            display_name="Test Agent",
            description="Test description",
            factory_function="test.module.factory",
        )
        assert config.name == "test_agent"
        assert config.tier_requirement == "free"  # Default

    def test_is_valid_handoff_function(self):
        """Verify is_valid_handoff function works correctly."""
        from lobster.config.agent_registry import is_valid_handoff

        # Supervisor should be able to handoff to research_agent
        assert is_valid_handoff('supervisor', 'research_agent')

        # transcriptomics_expert should be able to handoff to annotation_expert
        assert is_valid_handoff('transcriptomics_expert', 'annotation_expert')

    def test_subscription_tiers_integration(self):
        """Verify subscription_tiers.py integrates with dynamic discovery."""
        from lobster.config.subscription_tiers import (
            is_agent_available,
            is_agent_available_dynamic,
            get_available_agents_dynamic,
        )

        # Dynamic checking should work
        assert is_agent_available_dynamic('research_agent', 'free')
        assert not is_agent_available_dynamic('genomics_expert', 'free')
        assert is_agent_available_dynamic('genomics_expert', 'premium')

        # is_agent_available should use dynamic checking
        assert is_agent_available('research_agent', 'free')
        assert is_agent_available('genomics_expert', 'premium')

        # get_available_agents_dynamic should return correct lists
        free_agents = get_available_agents_dynamic('free')
        assert 'research_agent' in free_agents
        assert 'genomics_expert' not in free_agents

        premium_agents = get_available_agents_dynamic('premium')
        assert 'research_agent' in premium_agents
        assert 'genomics_expert' in premium_agents


class TestPhase3SuccessCriteria:
    """Tests that directly verify Phase 3 success criteria."""

    def test_criterion_1_no_hardcoded_agent_registry(self):
        """Criterion 1: AGENT_REGISTRY dict removed from codebase.

        This test verifies that list_agents() in ComponentRegistry
        does NOT reference AGENT_REGISTRY at all.
        """
        import inspect
        from lobster.core.component_registry import ComponentRegistry

        source = inspect.getsource(ComponentRegistry.list_agents)
        assert 'AGENT_REGISTRY' not in source

    def test_criterion_2_self_registration_via_entry_points(self):
        """Criterion 2: lobster-ai self-registers core agents via entry points."""
        import importlib.metadata

        eps = list(importlib.metadata.entry_points(group='lobster.agents'))
        ep_names = {ep.name for ep in eps}

        core_agents = [
            'data_expert_agent',
            'research_agent',
            'transcriptomics_expert',
            'annotation_expert',
            'de_analysis_expert',
            'genomics_expert',
            'visualization_expert_agent',
        ]

        for agent in core_agents:
            assert agent in ep_names, f"Missing entry point for {agent}"

    def test_criterion_3_discovery_under_50ms(self):
        """Criterion 3: Plugin discovery completes in <50ms."""
        from lobster.core.component_registry import component_registry
        import time

        component_registry.reset()

        start = time.perf_counter()
        agents = component_registry.list_agents()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Discovery took {elapsed_ms:.2f}ms (>50ms budget)"

    def test_criterion_4_agent_groups_work(self):
        """Criterion 4: Agent groups and profiles work as registry concepts."""
        from lobster.core.component_registry import component_registry

        component_registry.reset()
        agents = component_registry.list_agents()

        trans = agents.get('transcriptomics_expert')
        assert trans is not None
        assert trans.child_agents is not None
        assert 'annotation_expert' in trans.child_agents
        assert 'de_analysis_expert' in trans.child_agents

    def test_criterion_5_single_source_of_truth(self):
        """Criterion 5: ComponentRegistry serves as single source of truth."""
        import inspect
        from lobster.core.component_registry import ComponentRegistry
        from lobster.config import agent_registry

        # ComponentRegistry.list_agents should NOT reference AGENT_REGISTRY
        source = inspect.getsource(ComponentRegistry.list_agents)
        assert 'AGENT_REGISTRY' not in source

        # agent_registry facade should delegate to component_registry
        source2 = inspect.getsource(agent_registry.get_worker_agents)
        assert 'component_registry' in source2
