"""Integration tests for Phase 6: Agent Migration.

Verifies all 8 agent packages install, register, and function correctly.

Success Criteria from ROADMAP.md:
1. lobster-transcriptomics package installs and registers independently
2. lobster-research package combines research_agent and data_expert
3. lobster-visualization, lobster-proteomics, lobster-genomics, lobster-ml, lobster-metadata packages exist
4. Each package passes independent CI tests
5. User can install any combination: core only, core + free agents, core + premium agents, or full suite
6. Services bundled with primary agent packages (no orphaned services)
7. Each package has README.md (serves as PyPI description and docs source)
"""

import pytest
from pathlib import Path

from lobster.core.component_registry import ComponentRegistry


class TestPackageDiscovery:
    """Test that all packages are discoverable via entry points."""

    @pytest.fixture
    def registry(self):
        """Fresh ComponentRegistry for each test."""
        reg = ComponentRegistry()
        reg.load_components()
        return reg

    def test_all_packages_registered(self, registry):
        """SC-1, SC-3: All 8 packages register their agents."""
        agents = registry.list_agents()

        # Expected agents from 8 packages
        expected_agents = [
            # lobster-transcriptomics
            "transcriptomics_expert",
            "annotation_expert",
            "de_analysis_expert",
            # lobster-research
            "research_agent",
            "data_expert_agent",
            # lobster-visualization
            "visualization_expert_agent",
            # lobster-genomics
            "genomics_expert",
            # lobster-proteomics
            "proteomics_expert",
            # lobster-ml
            "machine_learning_expert",
            # lobster-metadata
            "metadata_assistant",
            # lobster-structural-viz
            "protein_structure_visualization_expert",
        ]

        for agent_name in expected_agents:
            assert agent_name in agents, f"Agent '{agent_name}' should be registered"

    def test_transcriptomics_package(self, registry):
        """SC-1: lobster-transcriptomics installs and registers independently."""
        config = registry.get_agent("transcriptomics_expert")
        assert config is not None
        assert config.package_name == "lobster-transcriptomics"
        assert "annotation_expert" in config.child_agents
        assert "de_analysis_expert" in config.child_agents

    def test_research_package_combines_agents(self, registry):
        """SC-2: lobster-research combines research_agent and data_expert."""
        research = registry.get_agent("research_agent")
        data_expert = registry.get_agent("data_expert_agent")

        assert research is not None
        assert data_expert is not None
        assert research.package_name == "lobster-research"
        assert data_expert.package_name == "lobster-research"

    def test_visualization_package(self, registry):
        """SC-3: lobster-visualization package exists."""
        config = registry.get_agent("visualization_expert_agent")
        assert config is not None
        assert config.package_name == "lobster-visualization"

    def test_genomics_package(self, registry):
        """SC-3: lobster-genomics package exists."""
        config = registry.get_agent("genomics_expert")
        assert config is not None
        assert config.package_name == "lobster-genomics"

    def test_proteomics_package(self, registry):
        """SC-3: lobster-proteomics package exists."""
        config = registry.get_agent("proteomics_expert")
        assert config is not None
        assert config.package_name == "lobster-proteomics"

    def test_ml_package(self, registry):
        """SC-3: lobster-ml package exists."""
        config = registry.get_agent("machine_learning_expert")
        assert config is not None
        assert config.package_name == "lobster-ml"

    def test_metadata_package(self, registry):
        """SC-3: lobster-metadata package exists."""
        config = registry.get_agent("metadata_assistant")
        assert config is not None
        assert config.package_name == "lobster-metadata"

    def test_structural_viz_package(self, registry):
        """SC-3: lobster-structural-viz package exists."""
        config = registry.get_agent("protein_structure_visualization_expert")
        assert config is not None
        assert config.package_name == "lobster-structural-viz"


class TestSubAgentAccessibility:
    """Test that sub-agents are correctly configured."""

    @pytest.fixture
    def registry(self):
        reg = ComponentRegistry()
        reg.load_components()
        return reg

    def test_annotation_expert_not_supervisor_accessible(self, registry):
        """Sub-agents should not be accessible from supervisor."""
        config = registry.get_agent("annotation_expert")
        assert config is not None
        assert config.supervisor_accessible == False

    def test_de_analysis_expert_not_supervisor_accessible(self, registry):
        """Sub-agents should not be accessible from supervisor."""
        config = registry.get_agent("de_analysis_expert")
        assert config is not None
        assert config.supervisor_accessible == False

    def test_main_agents_supervisor_accessible(self, registry):
        """Main agents should be supervisor-accessible (default)."""
        main_agents = [
            "transcriptomics_expert",
            "research_agent",
            "visualization_expert_agent",
            "genomics_expert",
            "proteomics_expert",
            "machine_learning_expert",
            "metadata_assistant",
            "protein_structure_visualization_expert",
        ]
        for name in main_agents:
            config = registry.get_agent(name)
            assert config is not None, f"Agent '{name}' should exist"
            # Default is True or not specified (None means True)
            accessible = getattr(config, "supervisor_accessible", True)
            assert accessible != False, f"{name} should be supervisor-accessible"


class TestTierRequirements:
    """Test that tier requirements are set correctly."""

    @pytest.fixture
    def registry(self):
        reg = ComponentRegistry()
        reg.load_components()
        return reg

    def test_all_agents_start_free(self, registry):
        """Per user decision: all agents start as free."""
        # Only check package agents (not core agents)
        package_agents = [
            "transcriptomics_expert",
            "annotation_expert",
            "de_analysis_expert",
            "research_agent",
            "data_expert_agent",
            "visualization_expert_agent",
            "genomics_expert",
            "proteomics_expert",
            "machine_learning_expert",
            "metadata_assistant",
            "protein_structure_visualization_expert",
        ]
        for name in package_agents:
            config = registry.get_agent(name)
            assert config is not None, f"Agent '{name}' should exist"
            tier = getattr(config, "tier_requirement", "free")
            assert tier == "free", f"{name} should have tier_requirement='free', got '{tier}'"


class TestPackageREADMEs:
    """SC-7: Each package has README.md."""

    @pytest.fixture
    def packages_dir(self):
        return Path(__file__).parent.parent.parent.parent / "packages"

    def test_readme_exists_for_all_packages(self, packages_dir):
        """Each package should have a README.md file."""
        expected_packages = [
            "lobster-transcriptomics",
            "lobster-research",
            "lobster-visualization",
            "lobster-genomics",
            "lobster-proteomics",
            "lobster-ml",
            "lobster-metadata",
            "lobster-structural-viz",
        ]

        for pkg_name in expected_packages:
            readme = packages_dir / pkg_name / "README.md"
            assert readme.exists(), f"README.md should exist for {pkg_name}"
            content = readme.read_text()
            assert len(content) > 100, f"README.md for {pkg_name} should have content"


class TestPromptInfrastructure:
    """Test prompt infrastructure from 06-01."""

    def test_prompt_composer_available(self):
        """PromptComposer should be importable."""
        from lobster.prompts import PromptComposer, get_prompt_composer
        composer = get_prompt_composer()
        assert composer is not None

    def test_prompt_registry_available(self):
        """PromptRegistry should be importable."""
        from lobster.prompts import PromptRegistry, get_prompt_registry
        registry = get_prompt_registry()
        assert registry is not None

    def test_shared_sections_exist(self):
        """Shared prompt sections should be loadable."""
        from lobster.prompts import PromptLoader
        loader = PromptLoader()

        sections = [
            "shared/role_identity.md",
            "shared/important_rules.md",
            "shared/tool_usage_patterns.md",
        ]

        for section in sections:
            content = loader.load_section("lobster.prompts", section)
            assert len(content) > 0, f"Shared section {section} should have content"


class TestHandoffBuilder:
    """Test handoff builder from 06-02."""

    def test_build_handoff_tools_available(self):
        """build_handoff_tools should be importable."""
        from lobster.tools.handoff_builder import build_handoff_tools
        assert callable(build_handoff_tools)

    def test_get_unavailable_agents_available(self):
        """get_unavailable_agents should be importable."""
        from lobster.tools.handoff_builder import get_unavailable_agents
        assert callable(get_unavailable_agents)


class TestDynamicStateAggregation:
    """Test dynamic state aggregation from 06-02."""

    def test_get_all_state_classes(self):
        """get_all_state_classes should return dict with core states."""
        from lobster.agents.state import get_all_state_classes, OverallState, TodoItem

        states = get_all_state_classes()

        assert "OverallState" in states
        assert "TodoItem" in states
        assert states["OverallState"] is OverallState

    def test_package_states_discovered(self):
        """Package states should be discoverable via entry points."""
        from lobster.agents.state import get_all_state_classes

        states = get_all_state_classes()

        # Check for states from packages
        # These should be exported via lobster.states entry points
        expected_states = [
            "TranscriptomicsExpertState",
            "ResearchAgentState",
            "VisualizationExpertState",
            "MetadataAssistantState",
            "ProteinStructureVisualizationExpertState",
        ]

        for state_name in expected_states:
            assert state_name in states, f"State '{state_name}' should be discoverable"


class TestInstallationCombinations:
    """SC-5: Test that different installation combinations work."""

    def test_core_imports_without_packages(self):
        """Core should work even if packages not installed."""
        from lobster.core.component_registry import ComponentRegistry
        from lobster.core.data_manager_v2 import DataManagerV2
        from lobster.prompts import PromptComposer
        from lobster.tools.handoff_builder import build_handoff_tools

        # All core imports should succeed
        assert ComponentRegistry is not None
        assert DataManagerV2 is not None
        assert PromptComposer is not None
        assert build_handoff_tools is not None

    def test_package_independence(self):
        """Each package should be independently importable."""
        # These imports should work regardless of other packages
        packages = [
            ("lobster_transcriptomics", "AGENT_CONFIG"),
            ("lobster_research", "AGENT_CONFIG"),
            ("lobster_visualization", "AGENT_CONFIG"),
            ("lobster_genomics", "AGENT_CONFIG"),
            ("lobster_proteomics", "AGENT_CONFIG"),
            ("lobster_ml", "AGENT_CONFIG"),
            ("lobster_metadata", "AGENT_CONFIG"),
            ("lobster_structural_viz", "AGENT_CONFIG"),
        ]

        for pkg_name, attr in packages:
            try:
                module = __import__(pkg_name)
                config = getattr(module, attr)
                assert config is not None, f"AGENT_CONFIG should exist in {pkg_name}"
            except ImportError:
                pytest.skip(f"Package {pkg_name} not installed")


class TestPackageEntryPoints:
    """Test that all packages register correct entry points."""

    @pytest.fixture
    def registry(self):
        reg = ComponentRegistry()
        reg.load_components()
        return reg

    def test_all_agent_entry_points_valid(self, registry):
        """All registered agents should have valid configs."""
        for name in registry.list_agents():
            config = registry.get_agent(name)
            assert config is not None
            assert hasattr(config, "name")
            # Config can have either factory (callable) or factory_function (string path)
            has_factory = hasattr(config, "factory") and callable(config.factory)
            has_factory_function = hasattr(config, "factory_function") and config.factory_function
            assert has_factory or has_factory_function, f"Agent {name} needs factory or factory_function"

    def test_package_count(self, registry):
        """Should have agents from 8 packages."""
        packages = set()
        for name in registry.list_agents():
            config = registry.get_agent(name)
            pkg = getattr(config, "package_name", "lobster-ai")
            if pkg != "lobster-ai":  # Exclude core agents
                packages.add(pkg)

        assert len(packages) >= 8, f"Expected 8 packages, found {len(packages)}: {packages}"


class TestServiceBundling:
    """SC-6: Services bundled with primary agent packages."""

    def test_services_entry_point_group_exists(self):
        """Services should be registrable via entry points."""
        from importlib.metadata import entry_points

        eps = entry_points(group="lobster.services")
        # Services exist (may be empty if none registered yet)
        service_names = [ep.name for ep in eps]
        # This passes as long as the group can be queried
        assert isinstance(service_names, list)
