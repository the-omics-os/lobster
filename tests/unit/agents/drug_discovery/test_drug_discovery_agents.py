"""
Contract tests for all 4 drug discovery agents.

Uses AgentContractTestMixin to validate plugin API compliance for the parent
drug_discovery_expert and its 3 child agents (cheminformatics_expert,
clinical_dev_expert, pharmacogenomics_expert).

Tests validate:
- Factory function signatures (standard params, no deprecated params)
- AGENT_CONFIG existence and required fields
- Parent/child hierarchy (supervisor_accessible, child_agents)
- Handoff tool configuration
- Tier requirements
"""

import pytest

from lobster.testing import AgentContractTestMixin

pytestmark = pytest.mark.unit


# =============================================================================
# CONTRACT TESTS (AgentContractTestMixin)
# =============================================================================


class TestDrugDiscoveryExpertContract(AgentContractTestMixin):
    """Contract tests for the drug_discovery_expert parent agent."""

    agent_module = "lobster.agents.drug_discovery.drug_discovery_expert"
    factory_name = "drug_discovery_expert"
    expected_tier = "free"


class TestCheminformaticsExpertContract(AgentContractTestMixin):
    """Contract tests for the cheminformatics_expert child agent."""

    agent_module = "lobster.agents.drug_discovery.cheminformatics_expert"
    factory_name = "cheminformatics_expert"
    expected_tier = "free"


class TestClinicalDevExpertContract(AgentContractTestMixin):
    """Contract tests for the clinical_dev_expert child agent."""

    agent_module = "lobster.agents.drug_discovery.clinical_dev_expert"
    factory_name = "clinical_dev_expert"
    expected_tier = "free"


class TestPharmacogenomicsExpertContract(AgentContractTestMixin):
    """Contract tests for the pharmacogenomics_expert child agent."""

    agent_module = "lobster.agents.drug_discovery.pharmacogenomics_expert"
    factory_name = "pharmacogenomics_expert"
    expected_tier = "free"


# =============================================================================
# AGENT_CONFIG FIELD TESTS
# =============================================================================


class TestDrugDiscoveryExpertConfig:
    """Detailed AGENT_CONFIG field tests for drug_discovery_expert."""

    def _get_config(self):
        from lobster.agents.drug_discovery.drug_discovery_expert import AGENT_CONFIG

        return AGENT_CONFIG

    def test_name_matches_expected(self):
        """AGENT_CONFIG.name must be 'drug_discovery_expert'."""
        config = self._get_config()
        assert config.name == "drug_discovery_expert"

    def test_display_name(self):
        """AGENT_CONFIG.display_name must be set."""
        config = self._get_config()
        assert config.display_name == "Drug Discovery Expert"

    def test_description_not_empty(self):
        """AGENT_CONFIG.description must be a non-empty string."""
        config = self._get_config()
        assert isinstance(config.description, str)
        assert len(config.description) > 10

    def test_factory_function_path(self):
        """AGENT_CONFIG.factory_function must point to the correct dotted path."""
        config = self._get_config()
        assert (
            config.factory_function
            == "lobster.agents.drug_discovery.drug_discovery_expert.drug_discovery_expert"
        )

    def test_supervisor_accessible_true(self):
        """Parent agent must be supervisor_accessible=True."""
        config = self._get_config()
        assert config.supervisor_accessible is True

    def test_child_agents_list(self):
        """Parent must declare exactly 3 child agents."""
        config = self._get_config()
        assert config.child_agents is not None
        assert isinstance(config.child_agents, list)
        assert len(config.child_agents) == 3
        assert "cheminformatics_expert" in config.child_agents
        assert "clinical_dev_expert" in config.child_agents
        assert "pharmacogenomics_expert" in config.child_agents

    def test_handoff_tool_name(self):
        """Parent must have handoff_to_drug_discovery_expert as handoff tool."""
        config = self._get_config()
        assert config.handoff_tool_name == "handoff_to_drug_discovery_expert"

    def test_handoff_tool_description_not_empty(self):
        """Parent handoff_tool_description must be a non-empty string."""
        config = self._get_config()
        assert isinstance(config.handoff_tool_description, str)
        assert len(config.handoff_tool_description) > 20

    def test_tier_requirement_free(self):
        """Parent must be free tier."""
        config = self._get_config()
        assert config.tier_requirement == "free"


class TestCheminformaticsExpertConfig:
    """Detailed AGENT_CONFIG field tests for cheminformatics_expert."""

    def _get_config(self):
        from lobster.agents.drug_discovery.cheminformatics_expert import AGENT_CONFIG

        return AGENT_CONFIG

    def test_name_matches_expected(self):
        config = self._get_config()
        assert config.name == "cheminformatics_expert"

    def test_display_name(self):
        config = self._get_config()
        assert config.display_name == "Cheminformatics Expert"

    def test_description_not_empty(self):
        config = self._get_config()
        assert isinstance(config.description, str)
        assert len(config.description) > 10

    def test_factory_function_path(self):
        config = self._get_config()
        assert (
            config.factory_function
            == "lobster.agents.drug_discovery.cheminformatics_expert.cheminformatics_expert"
        )

    def test_supervisor_accessible_false(self):
        """Child agent must NOT be supervisor_accessible."""
        config = self._get_config()
        assert config.supervisor_accessible is False

    def test_no_child_agents(self):
        """Child agent must NOT have its own children."""
        config = self._get_config()
        assert config.child_agents is None

    def test_handoff_tool_name_none(self):
        """Child agent must have handoff_tool_name=None (not routable from supervisor)."""
        config = self._get_config()
        assert config.handoff_tool_name is None

    def test_handoff_tool_description_none(self):
        """Child agent must have handoff_tool_description=None."""
        config = self._get_config()
        assert config.handoff_tool_description is None

    def test_tier_requirement_free(self):
        config = self._get_config()
        assert config.tier_requirement == "free"


class TestClinicalDevExpertConfig:
    """Detailed AGENT_CONFIG field tests for clinical_dev_expert."""

    def _get_config(self):
        from lobster.agents.drug_discovery.clinical_dev_expert import AGENT_CONFIG

        return AGENT_CONFIG

    def test_name_matches_expected(self):
        config = self._get_config()
        assert config.name == "clinical_dev_expert"

    def test_display_name(self):
        config = self._get_config()
        assert config.display_name == "Clinical Development Expert"

    def test_description_not_empty(self):
        config = self._get_config()
        assert isinstance(config.description, str)
        assert len(config.description) > 10

    def test_factory_function_path(self):
        config = self._get_config()
        assert (
            config.factory_function
            == "lobster.agents.drug_discovery.clinical_dev_expert.clinical_dev_expert"
        )

    def test_supervisor_accessible_false(self):
        """Child agent must NOT be supervisor_accessible."""
        config = self._get_config()
        assert config.supervisor_accessible is False

    def test_no_child_agents(self):
        """Child agent must NOT have its own children."""
        config = self._get_config()
        assert config.child_agents is None

    def test_handoff_tool_name_none(self):
        """Child agent must have handoff_tool_name=None."""
        config = self._get_config()
        assert config.handoff_tool_name is None

    def test_handoff_tool_description_none(self):
        """Child agent must have handoff_tool_description=None."""
        config = self._get_config()
        assert config.handoff_tool_description is None

    def test_tier_requirement_free(self):
        config = self._get_config()
        assert config.tier_requirement == "free"


class TestPharmacogenomicsExpertConfig:
    """Detailed AGENT_CONFIG field tests for pharmacogenomics_expert."""

    def _get_config(self):
        from lobster.agents.drug_discovery.pharmacogenomics_expert import AGENT_CONFIG

        return AGENT_CONFIG

    def test_name_matches_expected(self):
        config = self._get_config()
        assert config.name == "pharmacogenomics_expert"

    def test_display_name(self):
        config = self._get_config()
        assert config.display_name == "Pharmacogenomics Expert"

    def test_description_not_empty(self):
        config = self._get_config()
        assert isinstance(config.description, str)
        assert len(config.description) > 10

    def test_factory_function_path(self):
        config = self._get_config()
        assert (
            config.factory_function
            == "lobster.agents.drug_discovery.pharmacogenomics_expert.pharmacogenomics_expert"
        )

    def test_supervisor_accessible_false(self):
        """Child agent must NOT be supervisor_accessible."""
        config = self._get_config()
        assert config.supervisor_accessible is False

    def test_no_child_agents(self):
        """Child agent must NOT have its own children."""
        config = self._get_config()
        assert config.child_agents is None

    def test_handoff_tool_name_none(self):
        """Child agent must have handoff_tool_name=None."""
        config = self._get_config()
        assert config.handoff_tool_name is None

    def test_handoff_tool_description_none(self):
        """Child agent must have handoff_tool_description=None."""
        config = self._get_config()
        assert config.handoff_tool_description is None

    def test_tier_requirement_free(self):
        config = self._get_config()
        assert config.tier_requirement == "free"


# =============================================================================
# HIERARCHY CONSISTENCY TESTS
# =============================================================================


class TestAgentHierarchy:
    """Tests that parent-child relationships are correctly wired."""

    def test_parent_declares_all_children(self):
        """Parent's child_agents list must contain all 3 children."""
        from lobster.agents.drug_discovery.drug_discovery_expert import (
            AGENT_CONFIG as parent_config,
        )

        expected_children = {
            "cheminformatics_expert",
            "clinical_dev_expert",
            "pharmacogenomics_expert",
        }
        assert set(parent_config.child_agents) == expected_children

    def test_children_not_supervisor_accessible(self):
        """All child agents must have supervisor_accessible=False."""
        from lobster.agents.drug_discovery.cheminformatics_expert import (
            AGENT_CONFIG as chem_config,
        )
        from lobster.agents.drug_discovery.clinical_dev_expert import (
            AGENT_CONFIG as clin_config,
        )
        from lobster.agents.drug_discovery.pharmacogenomics_expert import (
            AGENT_CONFIG as pgx_config,
        )

        for config, name in [
            (chem_config, "cheminformatics_expert"),
            (clin_config, "clinical_dev_expert"),
            (pgx_config, "pharmacogenomics_expert"),
        ]:
            assert config.supervisor_accessible is False, (
                f"{name} must have supervisor_accessible=False"
            )

    def test_only_parent_has_handoff_tool(self):
        """Only the parent should have a handoff_tool_name set."""
        from lobster.agents.drug_discovery.cheminformatics_expert import (
            AGENT_CONFIG as chem_config,
        )
        from lobster.agents.drug_discovery.clinical_dev_expert import (
            AGENT_CONFIG as clin_config,
        )
        from lobster.agents.drug_discovery.drug_discovery_expert import (
            AGENT_CONFIG as parent_config,
        )
        from lobster.agents.drug_discovery.pharmacogenomics_expert import (
            AGENT_CONFIG as pgx_config,
        )

        # Parent has handoff tool
        assert parent_config.handoff_tool_name is not None
        assert parent_config.handoff_tool_name == "handoff_to_drug_discovery_expert"

        # Children do not
        assert chem_config.handoff_tool_name is None
        assert clin_config.handoff_tool_name is None
        assert pgx_config.handoff_tool_name is None

    def test_all_agents_free_tier(self):
        """All 4 agents must be free tier."""
        from lobster.agents.drug_discovery.cheminformatics_expert import (
            AGENT_CONFIG as chem_config,
        )
        from lobster.agents.drug_discovery.clinical_dev_expert import (
            AGENT_CONFIG as clin_config,
        )
        from lobster.agents.drug_discovery.drug_discovery_expert import (
            AGENT_CONFIG as parent_config,
        )
        from lobster.agents.drug_discovery.pharmacogenomics_expert import (
            AGENT_CONFIG as pgx_config,
        )

        for config in [parent_config, chem_config, clin_config, pgx_config]:
            assert config.tier_requirement == "free", (
                f"{config.name} must be free tier"
            )

    def test_child_agent_names_match_config_names(self):
        """Each child AGENT_CONFIG.name must match what the parent declares."""
        from lobster.agents.drug_discovery.cheminformatics_expert import (
            AGENT_CONFIG as chem_config,
        )
        from lobster.agents.drug_discovery.clinical_dev_expert import (
            AGENT_CONFIG as clin_config,
        )
        from lobster.agents.drug_discovery.drug_discovery_expert import (
            AGENT_CONFIG as parent_config,
        )
        from lobster.agents.drug_discovery.pharmacogenomics_expert import (
            AGENT_CONFIG as pgx_config,
        )

        child_configs = [chem_config, clin_config, pgx_config]
        child_names = {c.name for c in child_configs}
        declared_children = set(parent_config.child_agents)

        assert child_names == declared_children, (
            f"Mismatch between declared children {declared_children} "
            f"and actual AGENT_CONFIG names {child_names}"
        )
