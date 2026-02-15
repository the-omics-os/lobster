"""
Integration tests for Phase 4: Configuration System.

Tests validate all 5 success criteria:
1. Config file at .lobster_workspace/config.toml specifies enabled agents
2. Per-agent model and thinking settings configurable in TOML
3. Environment variables override config file values
4. Agent group presets (scrna-basic, scrna-full, multiomics-full) defined
5. Config validates against installed packages with helpful error messages
"""

import logging
from pathlib import Path

import pytest


class TestConfigFileTOML:
    """SC1: Config file at .lobster_workspace/config.toml specifies enabled agents."""

    def test_config_loads_from_toml(self, tmp_path: Path):
        """TOML config file is loaded correctly."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        # Create TOML config (top-level keys, no nested [agents] section)
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
config_version = "1.0"
enabled = ["research_agent", "data_expert_agent"]
""")

        config = WorkspaceAgentConfig.load(tmp_path)
        assert config.config_version == "1.0"
        assert "research_agent" in config.enabled_agents
        assert "data_expert_agent" in config.enabled_agents

    def test_config_defaults_when_no_file(self, tmp_path: Path):
        """Config returns defaults when no file exists."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config = WorkspaceAgentConfig.load(tmp_path)
        assert config.config_version == "1.0"
        assert config.enabled_agents == []

    def test_config_save_and_reload(self, tmp_path: Path):
        """Config can be saved and reloaded correctly."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config = WorkspaceAgentConfig(
            enabled=["research_agent", "transcriptomics_expert"]
        )
        config.save(tmp_path)

        loaded = WorkspaceAgentConfig.load(tmp_path)
        assert "research_agent" in loaded.enabled_agents
        assert "transcriptomics_expert" in loaded.enabled_agents


class TestPerAgentSettings:
    """SC2: Per-agent model and thinking settings configurable in TOML."""

    def test_per_agent_model_override(self, tmp_path: Path):
        """Per-agent model settings are parsed from TOML."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
config_version = "1.0"
enabled = ["research_agent"]

[agent_settings.research_agent]
model = "claude-4-5-sonnet"
thinking_preset = "standard"
temperature = 0.7
""")

        config = WorkspaceAgentConfig.load(tmp_path)
        settings = config.agent_settings.get("research_agent")
        assert settings is not None
        assert settings.model == "claude-4-5-sonnet"
        assert settings.thinking_preset == "standard"
        assert settings.temperature == 0.7

    def test_per_agent_settings_via_resolver(self, tmp_path: Path):
        """AgentConfigResolver returns per-agent settings."""
        from lobster.config.agent_config_resolver import AgentConfigResolver
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
config_version = "1.0"
enabled = ["research_agent"]

[agent_settings.research_agent]
model = "custom-model"
""")

        resolver = AgentConfigResolver(tmp_path)
        settings = resolver.get_agent_settings("research_agent")
        assert settings is not None
        assert settings["model"] == "custom-model"


class TestPriorityResolution:
    """SC3: Environment variables (runtime) override config file values."""

    def test_runtime_override_takes_priority(self, tmp_path: Path):
        """Runtime agents override TOML config."""
        from lobster.config.agent_config_resolver import AgentConfigResolver

        # Create TOML with different agents
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
enabled = ["transcriptomics_expert"]
""")

        resolver = AgentConfigResolver(tmp_path)

        # Runtime override should take priority
        agents, source = resolver.resolve_enabled_agents(
            runtime_agents=["research_agent"]
        )
        assert agents == ["research_agent"]
        assert "runtime" in source

    def test_runtime_preset_takes_priority(self, tmp_path: Path):
        """Runtime preset override takes priority over TOML config."""
        from lobster.config.agent_config_resolver import AgentConfigResolver

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
enabled = ["transcriptomics_expert"]
""")

        resolver = AgentConfigResolver(tmp_path)
        agents, source = resolver.resolve_enabled_agents(runtime_preset="scrna-basic")
        assert "research_agent" in agents
        assert "preset" in source or "runtime" in source

    def test_toml_config_priority_over_default(self, tmp_path: Path):
        """TOML config takes priority over default (all agents)."""
        from lobster.config.agent_config_resolver import AgentConfigResolver

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
enabled = ["research_agent"]
""")

        resolver = AgentConfigResolver(tmp_path)
        agents, source = resolver.resolve_enabled_agents()
        assert agents == ["research_agent"]
        assert "workspace" in source or "config" in source

    def test_default_returns_all_available(self, tmp_path: Path):
        """No config returns all available agents."""
        from lobster.config.agent_config_resolver import AgentConfigResolver

        resolver = AgentConfigResolver(tmp_path)
        agents, source = resolver.resolve_enabled_agents()
        assert len(agents) > 0  # Should have at least core agents
        assert "default" in source


class TestAgentPresets:
    """SC4: Agent group presets (scrna-basic, scrna-full, multiomics-full) defined."""

    def test_scrna_basic_preset(self):
        """scrna-basic preset has 4 core agents."""
        from lobster.config.agent_presets import expand_preset

        agents = expand_preset("scrna-basic")
        assert agents is not None
        assert len(agents) == 4
        assert "research_agent" in agents
        assert "data_expert_agent" in agents
        assert "transcriptomics_expert" in agents
        assert "visualization_expert_agent" in agents

    def test_scrna_full_preset(self):
        """scrna-full preset has 7 agents including premium."""
        from lobster.config.agent_presets import expand_preset

        agents = expand_preset("scrna-full")
        assert agents is not None
        assert len(agents) == 7
        assert "annotation_expert" in agents
        assert "de_analysis_expert" in agents
        assert "metadata_assistant" in agents

    def test_multiomics_full_preset(self):
        """multiomics-full preset has 10 agents."""
        from lobster.config.agent_presets import expand_preset

        agents = expand_preset("multiomics-full")
        assert agents is not None
        assert len(agents) == 10
        assert "proteomics_expert" in agents
        assert "genomics_expert" in agents
        assert "machine_learning_expert_agent" in agents

    def test_invalid_preset_returns_none(self):
        """Invalid preset name returns None."""
        from lobster.config.agent_presets import expand_preset

        result = expand_preset("invalid-preset-name")
        assert result is None

    def test_preset_in_toml_config(self, tmp_path: Path):
        """Preset specified in TOML is expanded correctly."""
        from lobster.config.agent_config_resolver import AgentConfigResolver
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
preset = "scrna-basic"
""")

        config = WorkspaceAgentConfig.load(tmp_path)
        assert config.preset == "scrna-basic"

        resolver = AgentConfigResolver(tmp_path)
        agents, source = resolver.resolve_enabled_agents()
        # Should expand to scrna-basic agents (filtered by installed)
        assert "preset" in source or "config" in source


class TestValidationErrorMessages:
    """SC5: Config validates against installed packages with helpful error messages."""

    def test_validates_against_installed_agents(self, tmp_path: Path, caplog):
        """Config validates enabled agents against ComponentRegistry."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config = WorkspaceAgentConfig(
            enabled=[
                "research_agent",  # Should be installed
                "nonexistent_agent_xyz",  # Not installed
            ]
        )

        with caplog.at_level(logging.WARNING):
            valid, missing = config.validate_enabled_agents()

        assert "research_agent" in valid
        assert "nonexistent_agent_xyz" in missing
        assert "nonexistent_agent_xyz" in caplog.text

    def test_missing_agent_warning_includes_install_hint(self, tmp_path: Path, caplog):
        """Warning for missing agents includes install suggestions."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config = WorkspaceAgentConfig(enabled=["fake_proteomics_agent"])

        with caplog.at_level(logging.WARNING):
            valid, missing = config.validate_enabled_agents()

        # Should suggest how to install or list agents
        assert "pip install" in caplog.text or "lobster" in caplog.text

    def test_valid_agents_passed_through(self, tmp_path: Path):
        """Valid agents are returned without error."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config = WorkspaceAgentConfig(enabled=["research_agent"])
        valid, missing = config.validate_enabled_agents()

        assert "research_agent" in valid
        assert len(missing) == 0


class TestListPresets:
    """Additional tests for preset listing functionality."""

    def test_list_presets_returns_all(self):
        """list_presets returns all available presets with counts."""
        from lobster.config.agent_presets import list_presets

        presets = list_presets()
        assert "scrna-basic" in presets
        assert "scrna-full" in presets
        assert "multiomics-full" in presets
        assert presets["scrna-basic"] == 4
        assert presets["scrna-full"] == 7
        assert presets["multiomics-full"] == 10

    def test_get_preset_description(self):
        """Presets have human-readable descriptions."""
        from lobster.config.agent_presets import get_preset_description

        desc = get_preset_description("scrna-basic")
        assert desc is not None
        assert "single-cell" in desc.lower()


class TestProviderSettings:
    """Test provider settings from TOML [provider] section (CONF-04)."""

    def test_provider_settings_parsed_from_toml(self, tmp_path: Path):
        """Provider settings are correctly parsed from TOML."""
        from lobster.config.workspace_agent_config import WorkspaceAgentConfig

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[lobster]
config_version = "1.0"

[provider]
default = "anthropic"
ollama_host = "http://localhost:11434"

[provider.models]
anthropic = "claude-sonnet-4-20250514"
ollama = "llama3:70b-instruct"
""")

        config = WorkspaceAgentConfig.load(tmp_path)
        assert config.has_provider_settings() is True
        assert config.provider_settings.default == "anthropic"
        assert config.provider_settings.ollama_host == "http://localhost:11434"
        assert (
            config.provider_settings.models["anthropic"] == "claude-sonnet-4-20250514"
        )

    def test_provider_settings_via_resolver(self, tmp_path: Path):
        """AgentConfigResolver exposes provider settings."""
        from lobster.config.agent_config_resolver import AgentConfigResolver

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[provider]
default = "ollama"
ollama_host = "http://gpu-server:11434"
""")

        resolver = AgentConfigResolver(tmp_path)
        provider = resolver.get_provider_settings()
        assert provider is not None
        assert provider["default"] == "ollama"
        assert provider["ollama_host"] == "http://gpu-server:11434"

    def test_no_provider_settings_returns_none(self, tmp_path: Path):
        """No [provider] section returns None."""
        from lobster.config.agent_config_resolver import AgentConfigResolver

        # Create TOML without provider section
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
enabled = ["research_agent"]
""")

        resolver = AgentConfigResolver(tmp_path)
        provider = resolver.get_provider_settings()
        assert provider is None
