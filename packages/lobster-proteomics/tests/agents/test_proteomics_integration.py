"""Integration tests for proteomics sub-agent architecture."""

from unittest.mock import MagicMock

import pytest

from lobster.agents.proteomics.biomarker_discovery_expert import (
    AGENT_CONFIG as BIOMARKER_CONFIG,
)
from lobster.agents.proteomics.de_analysis_expert import AGENT_CONFIG as DE_CONFIG
from lobster.agents.proteomics.prompts import (
    create_biomarker_discovery_expert_prompt,
    create_de_analysis_expert_prompt,
    create_proteomics_expert_prompt,
)
from lobster.agents.proteomics.proteomics_expert import AGENT_CONFIG as PARENT_CONFIG
from lobster.agents.proteomics.shared_tools import create_shared_tools
from lobster.agents.proteomics.state import (
    BiomarkerDiscoveryExpertState,
    DEAnalysisExpertState,
    ProteomicsExpertState,
)


class TestParentAgentConfig:
    """Tests for parent agent AGENT_CONFIG."""

    def test_parent_has_child_agents(self):
        assert PARENT_CONFIG.child_agents == [
            "proteomics_de_analysis_expert",
            "biomarker_discovery_expert",
        ]

    def test_parent_is_supervisor_accessible(self):
        assert PARENT_CONFIG.supervisor_accessible is True

    def test_parent_tier_free(self):
        assert PARENT_CONFIG.tier_requirement == "free"

    def test_parent_name(self):
        assert PARENT_CONFIG.name == "proteomics_expert"

    def test_parent_has_handoff_tool(self):
        assert PARENT_CONFIG.handoff_tool_name == "handoff_to_proteomics_expert"


class TestDESubAgentConfig:
    """Tests for DE analysis sub-agent AGENT_CONFIG."""

    def test_de_supervisor_accessible(self):
        assert DE_CONFIG.supervisor_accessible is True

    def test_de_name(self):
        assert DE_CONFIG.name == "proteomics_de_analysis_expert"

    def test_de_has_handoff_tool(self):
        assert DE_CONFIG.handoff_tool_name == "handoff_to_proteomics_de_analysis_expert"

    def test_de_has_handoff_description(self):
        assert DE_CONFIG.handoff_tool_description is not None

    def test_de_tier_free(self):
        assert DE_CONFIG.tier_requirement == "free"

    def test_de_factory_function(self):
        assert (
            DE_CONFIG.factory_function
            == "lobster.agents.proteomics.de_analysis_expert.de_analysis_expert"
        )


class TestBiomarkerSubAgentConfig:
    """Tests for biomarker discovery sub-agent AGENT_CONFIG."""

    def test_biomarker_not_supervisor_accessible(self):
        assert BIOMARKER_CONFIG.supervisor_accessible is False

    def test_biomarker_name(self):
        assert BIOMARKER_CONFIG.name == "biomarker_discovery_expert"

    def test_biomarker_no_handoff_tool(self):
        assert BIOMARKER_CONFIG.handoff_tool_name is None

    def test_biomarker_no_handoff_description(self):
        assert BIOMARKER_CONFIG.handoff_tool_description is None

    def test_biomarker_tier_free(self):
        assert BIOMARKER_CONFIG.tier_requirement == "free"

    def test_biomarker_factory_function(self):
        assert (
            BIOMARKER_CONFIG.factory_function
            == "lobster.agents.proteomics.biomarker_discovery_expert.biomarker_discovery_expert"
        )


class TestPlatformConfigRegressions:
    """Regression tests for platform config handling."""

    def test_unknown_platform_defaults_to_mass_spec(self):
        """Regression: get_platform_config('unknown') raised ValueError.

        Bug: auto-detection sometimes returns 'unknown' for ambiguous data.
        Fix: map 'unknown' to 'mass_spec' as safe default.
        """
        from lobster.agents.proteomics.config import (
            PLATFORM_CONFIGS,
            get_platform_config,
        )

        config = get_platform_config("unknown")
        assert config == PLATFORM_CONFIGS["mass_spec"]

    def test_valid_platforms_still_work(self):
        from lobster.agents.proteomics.config import get_platform_config

        for platform in ["mass_spec", "affinity"]:
            config = get_platform_config(platform)
            assert config.platform_type == platform

    def test_invalid_platform_still_raises(self):
        from lobster.agents.proteomics.config import get_platform_config

        with pytest.raises(ValueError, match="Unknown platform type"):
            get_platform_config("nonexistent_platform")


class TestSharedTools:
    """Tests for shared tools factory."""

    def test_returns_list(self):
        tools = create_shared_tools(
            data_manager=MagicMock(),
            quality_service=MagicMock(),
            preprocessing_service=MagicMock(),
            analysis_service=MagicMock(),
        )
        assert isinstance(tools, list)

    def test_returns_8_tools(self):
        tools = create_shared_tools(
            data_manager=MagicMock(),
            quality_service=MagicMock(),
            preprocessing_service=MagicMock(),
            analysis_service=MagicMock(),
        )
        assert len(tools) == 8

    def test_tool_names(self):
        tools = create_shared_tools(
            data_manager=MagicMock(),
            quality_service=MagicMock(),
            preprocessing_service=MagicMock(),
            analysis_service=MagicMock(),
        )
        tool_names = [t.name for t in tools]
        expected = [
            "check_proteomics_status",
            "assess_proteomics_quality",
            "filter_proteomics_data",
            "normalize_proteomics_data",
            "analyze_proteomics_patterns",
            "impute_missing_values",
            "select_variable_proteins",
            "create_proteomics_summary",
        ]
        assert tool_names == expected


class TestStateClasses:
    """Tests for state class definitions."""

    def test_all_states_have_next_field(self):
        for cls in [
            ProteomicsExpertState,
            DEAnalysisExpertState,
            BiomarkerDiscoveryExpertState,
        ]:
            # AgentState subclasses use annotations
            assert "next" in cls.__annotations__ or hasattr(cls, "next")

    def test_de_state_has_de_fields(self):
        assert "de_results" in DEAnalysisExpertState.__annotations__
        assert "time_course_results" in DEAnalysisExpertState.__annotations__
        assert "correlation_results" in DEAnalysisExpertState.__annotations__

    def test_biomarker_state_has_biomarker_fields(self):
        assert "network_modules" in BiomarkerDiscoveryExpertState.__annotations__
        assert "survival_results" in BiomarkerDiscoveryExpertState.__annotations__
        assert "biomarker_candidates" in BiomarkerDiscoveryExpertState.__annotations__


class TestPrompts:
    """Tests for prompt functions."""

    def test_parent_prompt_contains_delegation(self):
        prompt = create_proteomics_expert_prompt()
        assert "handoff_to_proteomics_de_analysis_expert" in prompt
        assert "handoff_to_biomarker_discovery_expert" in prompt

    def test_parent_prompt_contains_mandatory_protocol(self):
        prompt = create_proteomics_expert_prompt()
        assert "MANDATORY DELEGATION PROTOCOL" in prompt

    def test_de_prompt_returns_string(self):
        prompt = create_de_analysis_expert_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_de_prompt_mentions_tools(self):
        prompt = create_de_analysis_expert_prompt()
        assert "find_differential_proteins" in prompt
        assert "run_time_course_analysis" in prompt
        assert "run_correlation_analysis" in prompt

    def test_biomarker_prompt_returns_string(self):
        prompt = create_biomarker_discovery_expert_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_biomarker_prompt_mentions_tools(self):
        prompt = create_biomarker_discovery_expert_prompt()
        assert "identify_coexpression_modules" in prompt
        assert "perform_survival_analysis" in prompt
        assert "find_survival_biomarkers" in prompt
