"""
Integration tests for scrna-full preset (TEST-05).

Verifies that the scrna-full preset:
1. Expands to expected agents (basic + annotation, DE, metadata)
2. Graph builds successfully with preset agents
3. GraphMetadata contains expected agent names
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lobster.agents.graph import GraphMetadata, create_bioinformatics_graph
from lobster.config.agent_presets import AGENT_PRESETS, expand_preset


class TestScrnaFullPresetExpansion:
    """Test scrna-full preset expands to expected agent list."""

    def test_scrna_full_returns_list(self):
        """expand_preset returns a list for scrna-full."""
        result = expand_preset("scrna-full")
        assert result is not None, "scrna-full should be a valid preset"
        assert isinstance(result, list), "expand_preset should return list"

    def test_scrna_full_includes_basic_agents(self):
        """scrna-full includes all agents from scrna-basic."""
        full_agents = expand_preset("scrna-full")
        basic_agents = expand_preset("scrna-basic")

        for agent in basic_agents:
            assert agent in full_agents, f"scrna-full should include {agent} from basic"

    def test_scrna_full_contains_annotation_expert(self):
        """scrna-full includes annotation_expert for cell type annotation."""
        agents = expand_preset("scrna-full")
        assert (
            "annotation_expert" in agents
        ), "scrna-full should include annotation_expert"

    def test_scrna_full_contains_de_analysis_expert(self):
        """scrna-full includes de_analysis_expert for differential expression."""
        agents = expand_preset("scrna-full")
        assert (
            "de_analysis_expert" in agents
        ), "scrna-full should include de_analysis_expert"

    def test_scrna_full_contains_metadata_assistant(self):
        """scrna-full includes metadata_assistant for metadata management."""
        agents = expand_preset("scrna-full")
        assert (
            "metadata_assistant" in agents
        ), "scrna-full should include metadata_assistant"

    def test_scrna_full_has_exactly_7_agents(self):
        """scrna-full should have exactly 7 agents."""
        agents = expand_preset("scrna-full")
        assert len(agents) == 7, f"scrna-full should have 7 agents, got {len(agents)}"

    def test_scrna_full_does_not_include_multiomics_agents(self):
        """scrna-full should not include proteomics/genomics/ML agents."""
        agents = expand_preset("scrna-full")

        # These should NOT be in full single-cell preset
        assert (
            "proteomics_expert" not in agents
        ), "scrna-full should not include proteomics"
        assert "genomics_expert" not in agents, "scrna-full should not include genomics"
        assert (
            "machine_learning_expert_agent" not in agents
        ), "scrna-full should not include ML"

    def test_scrna_full_has_description(self):
        """scrna-full preset has a human-readable description."""
        preset_info = AGENT_PRESETS.get("scrna-full")
        assert preset_info is not None
        assert "description" in preset_info
        assert len(preset_info["description"]) > 0
        # Description should mention annotation or differential expression
        desc_lower = preset_info["description"].lower()
        assert "annotation" in desc_lower or "differential" in desc_lower

    def test_scrna_full_superset_of_basic(self):
        """scrna-full should be a strict superset of scrna-basic."""
        full_agents = set(expand_preset("scrna-full"))
        basic_agents = set(expand_preset("scrna-basic"))

        assert basic_agents.issubset(full_agents), "basic should be subset of full"
        assert len(full_agents) > len(basic_agents), "full should have more agents"


class TestScrnaFullGraphBuilds:
    """Test that graph builds successfully with scrna-full agents."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path: Path):
        """Create mock DataManagerV2 with Path workspace."""
        dm = MagicMock()
        dm.workspace_path = tmp_path / "test_workspace"
        dm.workspace_path.mkdir(exist_ok=True)
        dm.get_modality_ids.return_value = []
        dm.modalities = {}
        return dm

    def test_graph_builds_with_scrna_full_agents(self, mock_data_manager):
        """Graph builds successfully when given scrna-full agent list."""
        agents = expand_preset("scrna-full")

        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=agents,
        )

        assert graph is not None, "Graph should be created"
        assert metadata is not None, "Metadata should be returned"

    def test_graph_metadata_contains_scrna_full_agents(self, mock_data_manager):
        """GraphMetadata contains all agents from scrna-full preset."""
        agents = expand_preset("scrna-full")

        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=agents,
        )

        available_names = [a.name for a in metadata.available_agents]

        for agent_name in agents:
            assert agent_name in available_names, f"{agent_name} should be in metadata"

    def test_graph_metadata_filters_to_only_preset_agents(self, mock_data_manager):
        """GraphMetadata contains ONLY scrna-full agents, not extras."""
        agents = expand_preset("scrna-full")

        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=agents,
        )

        available_names = [a.name for a in metadata.available_agents]

        # All available agents should be in the preset list
        for name in available_names:
            assert name in agents, f"Unexpected agent {name} in metadata"

    def test_graph_returns_metadata_with_agent_count(self, mock_data_manager):
        """GraphMetadata has correct agent_count for scrna-full."""
        agents = expand_preset("scrna-full")

        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=agents,
        )

        assert metadata.agent_count == len(agents), f"Expected {len(agents)} agents"

    def test_full_preset_includes_sub_agents(self, mock_data_manager):
        """Verify sub-agents (annotation, DE) are accessible in full preset."""
        agents = expand_preset("scrna-full")

        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=agents,
        )

        available_names = [a.name for a in metadata.available_agents]

        # Verify sub-agents are present
        assert "annotation_expert" in available_names
        assert "de_analysis_expert" in available_names


class TestScrnaFullIntegrationWithConfig:
    """Test scrna-full preset integrates with config system."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path: Path):
        """Create mock DataManagerV2 with Path workspace."""
        dm = MagicMock()
        dm.workspace_path = tmp_path / "test_workspace"
        dm.workspace_path.mkdir(exist_ok=True)
        dm.get_modality_ids.return_value = []
        dm.modalities = {}
        return dm

    def test_config_can_specify_scrna_full_preset(self, tmp_path: Path):
        """WorkspaceAgentConfig can specify scrna-full preset."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config = WorkspaceAgentConfig(preset="scrna-full")
        assert config.preset == "scrna-full"

    def test_resolver_expands_scrna_full_preset(self, tmp_path: Path):
        """AgentConfigResolver expands scrna-full preset correctly."""
        from lobster.config.agent_config_resolver import AgentConfigResolver

        # Create TOML with preset
        config_path = tmp_path / "config.toml"
        config_path.write_text('preset = "scrna-full"\n')

        resolver = AgentConfigResolver(tmp_path)
        agents, source = resolver.resolve_enabled_agents()

        # Should expand to scrna-full agents (filtered by what's installed)
        assert "preset" in source or "config" in source

    def test_graph_builds_from_config_preset(self, mock_data_manager, tmp_path: Path):
        """Graph builds when config specifies scrna-full preset."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config = WorkspaceAgentConfig(preset="scrna-full")
        expected_agents = expand_preset("scrna-full")

        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=expected_agents,
            config=config,
        )

        assert graph is not None
        assert metadata.agent_count == len(expected_agents)

    def test_scrna_full_vs_basic_agent_count_difference(self, mock_data_manager):
        """Verify scrna-full has 3 more agents than scrna-basic."""
        basic_agents = expand_preset("scrna-basic")
        full_agents = expand_preset("scrna-full")

        _, basic_metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=basic_agents,
        )

        _, full_metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=full_agents,
        )

        agent_diff = full_metadata.agent_count - basic_metadata.agent_count
        assert agent_diff == 3, f"Expected 3 more agents in full, got {agent_diff}"
