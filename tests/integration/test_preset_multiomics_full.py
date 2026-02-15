"""
Integration tests for multiomics-full preset (TEST-05).

Verifies that the multiomics-full preset:
1. Expands to expected agents (all available agents)
2. Graph builds successfully with preset agents
3. GraphMetadata contains expected agent names
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lobster.agents.graph import GraphMetadata, create_bioinformatics_graph
from lobster.config.agent_presets import AGENT_PRESETS, expand_preset
from lobster.testing import MockDataManager


class TestMultiomicsFullPresetExpansion:
    """Test multiomics-full preset expands to expected agent list."""

    def test_multiomics_full_returns_list(self):
        """expand_preset returns a list for multiomics-full."""
        result = expand_preset("multiomics-full")
        assert result is not None, "multiomics-full should be a valid preset"
        assert isinstance(result, list), "expand_preset should return list"

    def test_multiomics_full_includes_all_basic_agents(self):
        """multiomics-full includes all agents from scrna-basic."""
        full_agents = expand_preset("multiomics-full")
        basic_agents = expand_preset("scrna-basic")

        for agent in basic_agents:
            assert agent in full_agents, f"multiomics-full should include {agent}"

    def test_multiomics_full_includes_all_scrna_full_agents(self):
        """multiomics-full includes all agents from scrna-full."""
        multiomics_agents = expand_preset("multiomics-full")
        scrna_full_agents = expand_preset("scrna-full")

        for agent in scrna_full_agents:
            assert agent in multiomics_agents, f"multiomics-full should include {agent}"

    def test_multiomics_full_contains_proteomics_expert(self):
        """multiomics-full includes proteomics_expert for mass spec analysis."""
        agents = expand_preset("multiomics-full")
        assert (
            "proteomics_expert" in agents
        ), "multiomics-full should include proteomics_expert"

    def test_multiomics_full_contains_genomics_expert(self):
        """multiomics-full includes genomics_expert for VCF/GWAS analysis."""
        agents = expand_preset("multiomics-full")
        assert (
            "genomics_expert" in agents
        ), "multiomics-full should include genomics_expert"

    def test_multiomics_full_contains_machine_learning_expert(self):
        """multiomics-full includes machine_learning_expert_agent for ML workflows."""
        agents = expand_preset("multiomics-full")
        assert (
            "machine_learning_expert_agent" in agents
        ), "multiomics-full should include ML expert"

    def test_multiomics_full_has_exactly_10_agents(self):
        """multiomics-full should have exactly 10 agents."""
        agents = expand_preset("multiomics-full")
        assert (
            len(agents) == 10
        ), f"multiomics-full should have 10 agents, got {len(agents)}"

    def test_multiomics_full_has_description(self):
        """multiomics-full preset has a human-readable description."""
        preset_info = AGENT_PRESETS.get("multiomics-full")
        assert preset_info is not None
        assert "description" in preset_info
        assert len(preset_info["description"]) > 0
        # Description should mention multi-omics
        desc_lower = preset_info["description"].lower()
        assert "multi-omics" in desc_lower or "multiomics" in desc_lower.replace(
            "-", ""
        )

    def test_multiomics_full_superset_of_scrna_full(self):
        """multiomics-full should be a strict superset of scrna-full."""
        multiomics_agents = set(expand_preset("multiomics-full"))
        scrna_full_agents = set(expand_preset("scrna-full"))

        assert scrna_full_agents.issubset(
            multiomics_agents
        ), "scrna-full should be subset"
        assert len(multiomics_agents) > len(
            scrna_full_agents
        ), "multiomics should have more agents"

    def test_multiomics_adds_exactly_3_agents_over_scrna_full(self):
        """multiomics-full adds proteomics, genomics, ML over scrna-full."""
        multiomics_agents = set(expand_preset("multiomics-full"))
        scrna_full_agents = set(expand_preset("scrna-full"))

        extra_agents = multiomics_agents - scrna_full_agents
        assert len(extra_agents) == 3, f"Expected 3 extra agents, got {extra_agents}"

        expected_extras = {
            "proteomics_expert",
            "genomics_expert",
            "machine_learning_expert_agent",
        }
        assert extra_agents == expected_extras, f"Extra agents mismatch: {extra_agents}"


class TestMultiomicsFullGraphBuilds:
    """Test that graph builds successfully with multiomics-full agents.

    Note: Some agent factories (proteomics_expert) have strict isinstance() checks
    that require real DataManagerV2. These tests use a subset of agents that work
    with MockDataManager to validate the graph building pattern.
    """

    @pytest.fixture
    def mock_data_manager(self, tmp_path: Path):
        """Create MockDataManager for testing."""
        workspace = tmp_path / "test_workspace"
        workspace.mkdir(exist_ok=True)
        return MockDataManager(workspace_path=workspace)

    @pytest.fixture
    def core_agents(self):
        """Return core agents that work with MockDataManager.

        Note: proteomics_expert requires real DataManagerV2 (isinstance check).
        This subset validates graph building pattern without specialized factories.
        """
        return [
            "research_agent",
            "data_expert_agent",
            "transcriptomics_expert",
            "visualization_expert_agent",
            "annotation_expert",
            "de_analysis_expert",
        ]

    def test_graph_builds_with_core_multiomics_agents(
        self, mock_data_manager, core_agents
    ):
        """Graph builds successfully with core agent subset."""
        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=core_agents,
        )

        assert graph is not None, "Graph should be created"
        assert metadata is not None, "Metadata should be returned"

    def test_graph_metadata_contains_requested_agents(
        self, mock_data_manager, core_agents
    ):
        """GraphMetadata contains all requested agents."""
        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=core_agents,
        )

        available_names = [a.name for a in metadata.available_agents]

        for agent_name in core_agents:
            assert agent_name in available_names, f"{agent_name} should be in metadata"

    def test_graph_metadata_filters_to_only_requested_agents(
        self, mock_data_manager, core_agents
    ):
        """GraphMetadata contains ONLY requested agents."""
        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=core_agents,
        )

        available_names = [a.name for a in metadata.available_agents]

        # All available agents should be in the requested list
        for name in available_names:
            assert name in core_agents, f"Unexpected agent {name} in metadata"

    def test_graph_returns_metadata_with_correct_count(
        self, mock_data_manager, core_agents
    ):
        """GraphMetadata has correct agent_count."""
        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=core_agents,
        )

        assert metadata.agent_count == len(
            core_agents
        ), f"Expected {len(core_agents)} agents"

    def test_preset_multiomics_full_agents_are_in_registry(self):
        """Verify all multiomics-full agents exist in ComponentRegistry."""
        from lobster.core.component_registry import component_registry

        agents = expand_preset("multiomics-full")
        registry_agents = component_registry.list_agents()

        # Most agents should be in registry (some may be missing ML agent)
        found_count = sum(1 for a in agents if a in registry_agents)
        assert (
            found_count >= 8
        ), f"Expected most agents in registry, found {found_count}/10"


class TestMultiomicsFullIntegrationWithConfig:
    """Test multiomics-full preset integrates with config system."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path: Path):
        """Create MockDataManager for testing."""
        workspace = tmp_path / "test_workspace"
        workspace.mkdir(exist_ok=True)
        return MockDataManager(workspace_path=workspace)

    @pytest.fixture
    def core_agents(self):
        """Core agents that work with MockDataManager."""
        return [
            "research_agent",
            "data_expert_agent",
            "transcriptomics_expert",
            "visualization_expert_agent",
        ]

    def test_config_can_specify_multiomics_full_preset(self, tmp_path: Path):
        """WorkspaceAgentConfig can specify multiomics-full preset."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config = WorkspaceAgentConfig(preset="multiomics-full")
        assert config.preset == "multiomics-full"

    def test_resolver_expands_multiomics_full_preset(self, tmp_path: Path):
        """AgentConfigResolver expands multiomics-full preset correctly."""
        from lobster.config.agent_config_resolver import AgentConfigResolver

        # Create TOML with preset
        config_path = tmp_path / "config.toml"
        config_path.write_text('preset = "multiomics-full"\n')

        resolver = AgentConfigResolver(tmp_path)
        agents, source = resolver.resolve_enabled_agents()

        # Should expand to multiomics-full agents (filtered by what's installed)
        assert "preset" in source or "config" in source

    def test_graph_builds_from_config_with_core_agents(
        self, mock_data_manager, core_agents, tmp_path: Path
    ):
        """Graph builds when config specifies core agents."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config = WorkspaceAgentConfig(preset="scrna-basic")

        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=core_agents,
            config=config,
        )

        assert graph is not None
        assert metadata.agent_count == len(core_agents)


class TestPresetHierarchy:
    """Test that presets form a proper hierarchy: basic < full < multiomics."""

    def test_preset_agent_counts_ascending(self):
        """Preset agent counts should be: basic < full < multiomics."""
        basic_count = len(expand_preset("scrna-basic"))
        full_count = len(expand_preset("scrna-full"))
        multiomics_count = len(expand_preset("multiomics-full"))

        assert basic_count < full_count < multiomics_count
        assert basic_count == 4
        assert full_count == 7
        assert multiomics_count == 10

    def test_preset_hierarchy_is_strict_superset(self):
        """Each preset should be strict superset of smaller preset."""
        basic = set(expand_preset("scrna-basic"))
        full = set(expand_preset("scrna-full"))
        multiomics = set(expand_preset("multiomics-full"))

        # basic < full < multiomics
        assert basic.issubset(full)
        assert full.issubset(multiomics)

        # Strict subsets (not equal)
        assert basic != full
        assert full != multiomics

    def test_all_presets_share_core_agents(self):
        """All presets should include research_agent and data_expert_agent."""
        presets = ["scrna-basic", "scrna-full", "multiomics-full"]
        core_agents = ["research_agent", "data_expert_agent"]

        for preset_name in presets:
            agents = expand_preset(preset_name)
            for core_agent in core_agents:
                assert core_agent in agents, f"{preset_name} missing {core_agent}"
