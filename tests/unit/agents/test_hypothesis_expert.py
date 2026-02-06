"""
Unit tests for HypothesisExpert agent.

Tests tool registration, prompt formatting, and agent configuration.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from lobster.agents.hypothesis_expert.config import AGENT_CONFIG
from lobster.agents.hypothesis_expert.prompts import (
    HYPOTHESIS_EXPERT_SYSTEM_PROMPT,
    create_hypothesis_expert_prompt,
)
from lobster.agents.hypothesis_expert.state import HypothesisExpertState
from lobster.config.agent_registry import AgentRegistryConfig


class TestAgentConfig:
    """Test agent configuration."""

    def test_agent_config_structure(self):
        """Test AGENT_CONFIG has correct structure."""
        assert isinstance(AGENT_CONFIG, AgentRegistryConfig)
        assert AGENT_CONFIG.name == "hypothesis_expert"
        assert AGENT_CONFIG.display_name == "Hypothesis Expert"

    def test_agent_config_factory_function(self):
        """Test factory function path is correct."""
        assert "hypothesis_expert.hypothesis_expert" in AGENT_CONFIG.factory_function

    def test_agent_config_handoff(self):
        """Test handoff configuration."""
        assert AGENT_CONFIG.handoff_tool_name == "handoff_to_hypothesis_expert"
        assert AGENT_CONFIG.handoff_tool_description is not None
        assert "hypothesis" in AGENT_CONFIG.handoff_tool_description.lower()

    def test_agent_config_supervisor_accessible(self):
        """Test agent is supervisor accessible."""
        assert AGENT_CONFIG.supervisor_accessible is True

    def test_agent_config_no_children(self):
        """Test agent has no child agents."""
        assert AGENT_CONFIG.child_agents is None


class TestAgentPrompts:
    """Test agent system prompts."""

    def test_system_prompt_has_identity(self):
        """Test system prompt defines agent identity."""
        assert "<Identity_And_Role>" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT
        assert "Hypothesis Expert" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT

    def test_system_prompt_has_tools_section(self):
        """Test system prompt documents available tools."""
        assert "<Your_Tools>" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT
        assert "generate_hypothesis" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT
        assert "get_current_hypothesis" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT
        assert "list_evidence_sources" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT

    def test_system_prompt_has_citation_rules(self):
        """Test system prompt includes citation format."""
        assert "<Citation_Rules>" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT
        assert "(claim)[DOI]" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT

    def test_system_prompt_has_workflow_guidelines(self):
        """Test system prompt includes workflow guidance."""
        assert "<Workflow_Guidelines>" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT

    def test_system_prompt_has_important_rules(self):
        """Test system prompt includes important rules."""
        assert "<Important_Rules>" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT

    def test_create_prompt_adds_date(self):
        """Test dynamic prompt creation adds date."""
        prompt = create_hypothesis_expert_prompt()
        assert "Today's date:" in prompt


class TestAgentState:
    """Test agent state class."""

    def test_state_class_attributes(self):
        """Test HypothesisExpertState class has expected attributes defined."""
        # HypothesisExpertState extends TypedDict with class-level defaults
        # TypedDict doesn't auto-populate defaults; verify class defines them
        assert hasattr(HypothesisExpertState, "__annotations__")

        annotations = HypothesisExpertState.__annotations__
        assert "next" in annotations
        assert "task_description" in annotations
        assert "research_objective" in annotations
        assert "current_hypothesis" in annotations
        assert "hypothesis_iteration" in annotations
        assert "evidence_sources" in annotations
        assert "key_insights" in annotations
        assert "methodology" in annotations

    def test_state_custom_values(self):
        """Test HypothesisExpertState accepts custom values."""
        # HypothesisExpertState extends TypedDict, so access via dict syntax
        state = HypothesisExpertState(
            messages=[],
            next="",
            task_description="",
            research_objective="Test objective",
            current_hypothesis="Test hypothesis",
            hypothesis_iteration=2,
            evidence_sources=[],
            key_insights=[],
            methodology=None,
            intermediate_outputs={},
        )

        assert state["research_objective"] == "Test objective"
        assert state["current_hypothesis"] == "Test hypothesis"
        assert state["hypothesis_iteration"] == 2


class TestAgentRegistry:
    """Test agent registry integration."""

    def test_hypothesis_expert_in_registry(self):
        """Test hypothesis_expert is registered."""
        from lobster.config.agent_registry import AGENT_REGISTRY

        assert "hypothesis_expert" in AGENT_REGISTRY

    def test_registry_config_matches_module(self):
        """Test registry config matches module config."""
        from lobster.config.agent_registry import AGENT_REGISTRY

        registry_config = AGENT_REGISTRY["hypothesis_expert"]

        assert registry_config.name == AGENT_CONFIG.name
        assert registry_config.factory_function == AGENT_CONFIG.factory_function


class TestAgentFactory:
    """Test agent factory function."""

    @pytest.fixture
    def mock_data_manager(self):
        """Create mock DataManagerV2."""
        dm = MagicMock()
        dm.workspace_path = Path("/tmp/test_workspace")
        dm.session_data = {}
        dm.list_modalities.return_value = []
        dm._save_session_metadata = MagicMock()
        dm.log_tool_usage = MagicMock()
        return dm

    @patch("lobster.agents.hypothesis_expert.hypothesis_expert.create_llm")
    @patch("lobster.agents.hypothesis_expert.hypothesis_expert.get_settings")
    def test_agent_factory_creates_agent(
        self, mock_settings, mock_create_llm, mock_data_manager
    ):
        """Test factory function creates agent."""
        from lobster.agents.hypothesis_expert.hypothesis_expert import hypothesis_expert

        mock_settings.return_value.get_agent_llm_params.return_value = {}
        mock_create_llm.return_value = MagicMock()

        agent = hypothesis_expert(
            data_manager=mock_data_manager,
            agent_name="test_hypothesis_expert",
        )

        assert agent is not None
        mock_create_llm.assert_called_once()

    @patch("lobster.agents.hypothesis_expert.hypothesis_expert.create_llm")
    @patch("lobster.agents.hypothesis_expert.hypothesis_expert.get_settings")
    def test_agent_factory_with_callback(
        self, mock_settings, mock_create_llm, mock_data_manager
    ):
        """Test factory function handles callback."""
        from lobster.agents.hypothesis_expert.hypothesis_expert import hypothesis_expert

        mock_settings.return_value.get_agent_llm_params.return_value = {}
        mock_llm = MagicMock()
        mock_llm.with_config.return_value = mock_llm
        mock_create_llm.return_value = mock_llm

        callback = MagicMock()
        agent = hypothesis_expert(
            data_manager=mock_data_manager,
            callback_handler=callback,
        )

        assert agent is not None
        mock_llm.with_config.assert_called_once()


class TestDelegationTriggers:
    """Test delegation trigger keywords."""

    @pytest.fixture
    def trigger_keywords(self):
        """Keywords that should trigger hypothesis_expert delegation."""
        return [
            "generate hypothesis",
            "synthesize findings",
            "propose research direction",
            "create testable hypothesis",
            "what hypothesis can we form",
            "formulate a hypothesis",
        ]

    def test_handoff_description_covers_triggers(self, trigger_keywords):
        """Test handoff description covers common trigger phrases."""
        description = AGENT_CONFIG.handoff_tool_description.lower()

        # Check key concepts are covered
        assert "hypothesis" in description
        assert "synthesize" in description or "findings" in description
        assert "research" in description or "direction" in description


class TestCitationFormat:
    """Test citation format handling."""

    def test_citation_format_in_prompts(self):
        """Test citation format documented correctly."""
        prompt = create_hypothesis_expert_prompt()

        # Check citation format examples
        assert "(claim)[DOI" in prompt or "(claim)[DOI]" in prompt

    def test_citation_rules_section(self):
        """Test citation rules section content."""
        assert "Cite DOIs/URLs from literature" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT
        assert "Reference analysis results directly" in HYPOTHESIS_EXPERT_SYSTEM_PROMPT
