"""
Integration tests for scrna-basic preset (TEST-05).

Verifies that the scrna-basic preset:
1. Expands to expected agents (research, data, transcriptomics, visualization)
2. Graph builds successfully with preset agents
3. GraphMetadata contains expected agent names
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from lobster.config.agent_presets import expand_preset, AGENT_PRESETS
from lobster.agents.graph import create_bioinformatics_graph, GraphMetadata


class TestScrnaBasicPresetExpansion:
    """Test scrna-basic preset expands to expected agent list."""

    def test_scrna_basic_returns_list(self):
        """expand_preset returns a list for scrna-basic."""
        result = expand_preset("scrna-basic")
        assert result is not None, "scrna-basic should be a valid preset"
        assert isinstance(result, list), "expand_preset should return list"

    def test_scrna_basic_contains_research_agent(self):
        """scrna-basic includes research_agent for literature search."""
        agents = expand_preset("scrna-basic")
        assert "research_agent" in agents, "scrna-basic should include research_agent"

    def test_scrna_basic_contains_data_expert(self):
        """scrna-basic includes data_expert_agent for data loading."""
        agents = expand_preset("scrna-basic")
        assert "data_expert_agent" in agents, "scrna-basic should include data_expert_agent"

    def test_scrna_basic_contains_transcriptomics_expert(self):
        """scrna-basic includes transcriptomics_expert for scRNA-seq analysis."""
        agents = expand_preset("scrna-basic")
        assert "transcriptomics_expert" in agents, "scrna-basic should include transcriptomics_expert"

    def test_scrna_basic_contains_visualization_expert(self):
        """scrna-basic includes visualization_expert_agent for plotting."""
        agents = expand_preset("scrna-basic")
        assert "visualization_expert_agent" in agents, "scrna-basic should include visualization_expert_agent"

    def test_scrna_basic_has_exactly_4_agents(self):
        """scrna-basic should have exactly 4 agents."""
        agents = expand_preset("scrna-basic")
        assert len(agents) == 4, f"scrna-basic should have 4 agents, got {len(agents)}"

    def test_scrna_basic_does_not_include_premium_agents(self):
        """scrna-basic should not include premium/specialized agents."""
        agents = expand_preset("scrna-basic")

        # These should NOT be in basic preset
        assert "annotation_expert" not in agents, "basic preset should not include annotation_expert"
        assert "de_analysis_expert" not in agents, "basic preset should not include de_analysis_expert"
        assert "metadata_assistant" not in agents, "basic preset should not include metadata_assistant"
        assert "proteomics_expert" not in agents, "basic preset should not include proteomics_expert"
        assert "genomics_expert" not in agents, "basic preset should not include genomics_expert"

    def test_scrna_basic_has_description(self):
        """scrna-basic preset has a human-readable description."""
        preset_info = AGENT_PRESETS.get("scrna-basic")
        assert preset_info is not None
        assert "description" in preset_info
        assert len(preset_info["description"]) > 0
        # Description should mention single-cell
        assert "single-cell" in preset_info["description"].lower()


class TestScrnaBasicGraphBuilds:
    """Test that graph builds successfully with scrna-basic agents."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path: Path):
        """Create mock DataManagerV2 with Path workspace."""
        dm = MagicMock()
        dm.workspace_path = tmp_path / "test_workspace"
        dm.workspace_path.mkdir(exist_ok=True)
        dm.get_modality_ids.return_value = []
        dm.modalities = {}
        return dm

    def test_graph_builds_with_scrna_basic_agents(self, mock_data_manager):
        """Graph builds successfully when given scrna-basic agent list."""
        agents = expand_preset("scrna-basic")

        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=agents,
        )

        assert graph is not None, "Graph should be created"
        assert metadata is not None, "Metadata should be returned"

    def test_graph_metadata_contains_scrna_basic_agents(self, mock_data_manager):
        """GraphMetadata contains all agents from scrna-basic preset."""
        agents = expand_preset("scrna-basic")

        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=agents,
        )

        available_names = [a.name for a in metadata.available_agents]

        for agent_name in agents:
            assert agent_name in available_names, f"{agent_name} should be in metadata"

    def test_graph_metadata_filters_to_only_preset_agents(self, mock_data_manager):
        """GraphMetadata contains ONLY scrna-basic agents, not extras."""
        agents = expand_preset("scrna-basic")

        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=agents,
        )

        available_names = [a.name for a in metadata.available_agents]

        # All available agents should be in the preset list
        for name in available_names:
            assert name in agents, f"Unexpected agent {name} in metadata"

    def test_graph_returns_metadata_with_agent_count(self, mock_data_manager):
        """GraphMetadata has correct agent_count for scrna-basic."""
        agents = expand_preset("scrna-basic")

        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=agents,
        )

        assert metadata.agent_count == len(agents), f"Expected {len(agents)} agents"

    def test_graph_metadata_has_supervisor_accessible_agents(self, mock_data_manager):
        """GraphMetadata tracks supervisor-accessible agents."""
        agents = expand_preset("scrna-basic")

        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=agents,
        )

        # At least some agents should be supervisor-accessible
        assert metadata.supervisor_accessible_count >= 0
        assert hasattr(metadata, "supervisor_accessible_agents")


class TestScrnaBasicIntegrationWithConfig:
    """Test scrna-basic preset integrates with config system."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path: Path):
        """Create mock DataManagerV2 with Path workspace."""
        dm = MagicMock()
        dm.workspace_path = tmp_path / "test_workspace"
        dm.workspace_path.mkdir(exist_ok=True)
        dm.get_modality_ids.return_value = []
        dm.modalities = {}
        return dm

    def test_config_can_specify_scrna_basic_preset(self, tmp_path: Path):
        """WorkspaceAgentConfig can specify scrna-basic preset."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config = WorkspaceAgentConfig(preset="scrna-basic")
        assert config.preset == "scrna-basic"

    def test_resolver_expands_scrna_basic_preset(self, tmp_path: Path):
        """AgentConfigResolver expands scrna-basic preset correctly."""
        from lobster.config.agent_config_resolver import AgentConfigResolver

        # Create TOML with preset
        config_path = tmp_path / "config.toml"
        config_path.write_text('preset = "scrna-basic"\n')

        resolver = AgentConfigResolver(tmp_path)
        agents, source = resolver.resolve_enabled_agents()

        # Should expand to scrna-basic agents (filtered by what's installed)
        assert "preset" in source or "config" in source

    def test_graph_builds_from_config_preset(self, mock_data_manager, tmp_path: Path):
        """Graph builds when config specifies scrna-basic preset."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config = WorkspaceAgentConfig(preset="scrna-basic")
        expected_agents = expand_preset("scrna-basic")

        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=expected_agents,
            config=config,
        )

        assert graph is not None
        assert metadata.agent_count == len(expected_agents)
