"""
Semantic tests for all 4 drug discovery agent prompts.

Validates that each prompt:
- Contains required XML sections (Identity_And_Role, Your_Environment, etc.)
- Documents the correct capabilities and tool names
- Maintains clear agent boundary definitions (Not_Responsibilities)
- Does not claim other agents' responsibilities
- Tool names in prompts match the actual tool factory output
"""

import pytest

from lobster.agents.drug_discovery.prompts import (
    create_cheminformatics_expert_prompt,
    create_clinical_dev_expert_prompt,
    create_drug_discovery_expert_prompt,
    create_pharmacogenomics_expert_prompt,
)

pytestmark = pytest.mark.unit

# Required XML sections per the agent architecture guidelines
REQUIRED_SECTIONS = [
    "Identity_And_Role",
    "Your_Environment",
    "Your_Responsibilities",
    "Your_Not_Responsibilities",
    "Your_Tools",
    "Decision_Tree",
    "Important_Rules",
]


# =============================================================================
# SHARED SECTION TESTS
# =============================================================================


class TestRequiredSections:
    """Verify all 4 prompts have required XML sections."""

    @pytest.mark.parametrize(
        "prompt_fn,agent_name",
        [
            (create_drug_discovery_expert_prompt, "drug_discovery_expert"),
            (create_cheminformatics_expert_prompt, "cheminformatics_expert"),
            (create_clinical_dev_expert_prompt, "clinical_dev_expert"),
            (create_pharmacogenomics_expert_prompt, "pharmacogenomics_expert"),
        ],
    )
    def test_has_all_required_sections(self, prompt_fn, agent_name):
        """Every prompt must contain all 7 required XML sections."""
        prompt = prompt_fn()
        for section in REQUIRED_SECTIONS:
            assert f"<{section}>" in prompt, (
                f"{agent_name} prompt missing required section <{section}>"
            )

    @pytest.mark.parametrize(
        "prompt_fn,agent_name",
        [
            (create_drug_discovery_expert_prompt, "drug_discovery_expert"),
            (create_cheminformatics_expert_prompt, "cheminformatics_expert"),
            (create_clinical_dev_expert_prompt, "clinical_dev_expert"),
            (create_pharmacogenomics_expert_prompt, "pharmacogenomics_expert"),
        ],
    )
    def test_has_core_mission(self, prompt_fn, agent_name):
        """Every prompt should include a Core_Mission subsection."""
        prompt = prompt_fn()
        assert "<Core_Mission>" in prompt, (
            f"{agent_name} prompt missing <Core_Mission> subsection"
        )

    @pytest.mark.parametrize(
        "prompt_fn,agent_name",
        [
            (create_drug_discovery_expert_prompt, "drug_discovery_expert"),
            (create_cheminformatics_expert_prompt, "cheminformatics_expert"),
            (create_clinical_dev_expert_prompt, "clinical_dev_expert"),
            (create_pharmacogenomics_expert_prompt, "pharmacogenomics_expert"),
        ],
    )
    def test_contains_todays_date(self, prompt_fn, agent_name):
        """Prompt must include dynamic date via f-string."""
        from datetime import date

        prompt = prompt_fn()
        assert date.today().isoformat() in prompt, (
            f"{agent_name} prompt does not contain today's date"
        )


# =============================================================================
# DRUG DISCOVERY EXPERT (PARENT) PROMPT TESTS
# =============================================================================


class TestDrugDiscoveryExpertPrompt:
    """Semantic tests for the parent drug_discovery_expert prompt."""

    def _get_prompt(self):
        return create_drug_discovery_expert_prompt()

    def test_identifies_as_parent(self):
        """Prompt should identify itself as the primary orchestrator."""
        prompt = self._get_prompt()
        assert "orchestrator" in prompt.lower() or "primary" in prompt.lower()

    def test_mentions_all_child_agents(self):
        """Prompt must reference all 3 child agents."""
        prompt = self._get_prompt()
        assert "cheminformatics_expert" in prompt
        assert "clinical_dev_expert" in prompt
        assert "pharmacogenomics_expert" in prompt

    def test_describes_child_capabilities(self):
        """Prompt must describe what each child agent handles."""
        prompt = self._get_prompt()
        # Cheminformatics child capabilities
        assert "molecular" in prompt.lower() or "descriptor" in prompt.lower()
        assert "ADMET" in prompt or "admet" in prompt.lower()
        # Clinical child capabilities
        assert "synergy" in prompt.lower()
        assert "tractability" in prompt.lower()
        # Pharmacogenomics child capabilities
        assert "ESM2" in prompt or "mutation" in prompt.lower()

    def test_mentions_target_scoring_tools(self):
        """Prompt must document target scoring tools."""
        prompt = self._get_prompt()
        assert "search_drug_targets" in prompt
        assert "score_drug_target" in prompt
        assert "rank_targets" in prompt

    def test_mentions_compound_tools(self):
        """Prompt must document compound search and bioactivity tools."""
        prompt = self._get_prompt()
        assert "search_compounds" in prompt
        assert "get_compound_bioactivity" in prompt
        assert "get_target_compounds" in prompt
        assert "get_compound_properties" in prompt
        assert "get_drug_indications" in prompt

    def test_mentions_utility_tools(self):
        """Prompt must document status and database tools."""
        prompt = self._get_prompt()
        assert "check_drug_discovery_status" in prompt
        assert "list_available_databases" in prompt

    def test_tool_names_match_shared_tools(self):
        """All 10 shared tool names in the prompt must match shared_tools.py factory."""
        from unittest.mock import Mock

        from lobster.agents.drug_discovery.shared_tools import create_shared_tools
        from lobster.core.data_manager_v2 import DataManagerV2
        from lobster.services.drug_discovery.chembl_service import ChEMBLService
        from lobster.services.drug_discovery.opentargets_service import (
            OpenTargetsService,
        )
        from lobster.services.drug_discovery.pubchem_service import PubChemService
        from lobster.services.drug_discovery.target_scoring_service import (
            TargetScoringService,
        )

        mock_dm = Mock(spec=DataManagerV2)
        tools = create_shared_tools(
            mock_dm,
            ChEMBLService(),
            OpenTargetsService(),
            PubChemService(),
            TargetScoringService(),
        )
        tool_names = [t.name for t in tools]

        prompt = self._get_prompt()
        for name in tool_names:
            assert name in prompt, (
                f"Tool '{name}' from shared_tools.py not documented in parent prompt"
            )

    def test_decision_tree_mentions_delegation(self):
        """Decision tree should clearly list delegation paths."""
        prompt = self._get_prompt()
        decision_section_start = prompt.find("<Decision_Tree>")
        decision_section_end = prompt.find("</Decision_Tree>")
        decision_tree = prompt[decision_section_start:decision_section_end]

        assert "Handle directly" in decision_tree
        assert "Delegate to cheminformatics_expert" in decision_tree
        assert "Delegate to clinical_dev_expert" in decision_tree
        assert "Delegate to pharmacogenomics_expert" in decision_tree
        assert "Return to supervisor" in decision_tree

    def test_not_responsible_for_literature(self):
        """Parent should not claim literature search responsibilities."""
        prompt = self._get_prompt()
        not_resp_start = prompt.find("<Your_Not_Responsibilities>")
        not_resp_end = prompt.find("</Your_Not_Responsibilities>")
        not_resp = prompt[not_resp_start:not_resp_end]

        assert "literature" in not_resp.lower() or "research_agent" in not_resp

    def test_not_responsible_for_downloading(self):
        """Parent should not claim dataset downloading responsibilities."""
        prompt = self._get_prompt()
        not_resp_start = prompt.find("<Your_Not_Responsibilities>")
        not_resp_end = prompt.find("</Your_Not_Responsibilities>")
        not_resp = prompt[not_resp_start:not_resp_end]

        assert "download" in not_resp.lower() or "data_expert" in not_resp

    def test_mentions_lobster_ai_context(self):
        """Prompt must reference the Lobster AI multi-agent architecture."""
        prompt = self._get_prompt()
        assert "lobster-ai" in prompt.lower() or "Lobster AI" in prompt

    def test_mentions_supervisor_reporting(self):
        """Prompt must state it reports to the supervisor, not to users."""
        prompt = self._get_prompt()
        assert "supervisor" in prompt.lower()
        assert "never interact with end users" in prompt.lower() or (
            "report exclusively to the supervisor" in prompt.lower()
        )


# =============================================================================
# CHEMINFORMATICS EXPERT PROMPT TESTS
# =============================================================================


class TestCheminformaticsExpertPrompt:
    """Semantic tests for the cheminformatics_expert child prompt."""

    def _get_prompt(self):
        return create_cheminformatics_expert_prompt()

    def test_identifies_as_child(self):
        """Prompt should identify itself as a child of drug_discovery_expert."""
        prompt = self._get_prompt()
        assert "drug_discovery_expert" in prompt
        assert "child" in prompt.lower() or "parent" in prompt.lower()

    def test_not_supervisor_accessible(self):
        """Prompt should state it is not directly accessible by the supervisor."""
        prompt = self._get_prompt()
        assert "NOT directly accessible" in prompt

    def test_mentions_rdkit_capabilities(self):
        """Prompt should reference RDKit-based computations."""
        prompt = self._get_prompt()
        assert "RDKit" in prompt or "rdkit" in prompt.lower()

    def test_mentions_key_tools(self):
        """Prompt must document the core cheminformatics tools."""
        prompt = self._get_prompt()
        assert "calculate_descriptors" in prompt
        assert "lipinski_check" in prompt
        assert "fingerprint_similarity" in prompt
        assert "predict_admet" in prompt
        assert "prepare_molecule_3d" in prompt

    def test_mentions_additional_tools(self):
        """Prompt must document CAS conversion, similarity search, binding site, comparison."""
        prompt = self._get_prompt()
        assert "cas_to_smiles" in prompt
        assert "search_similar_compounds" in prompt
        assert "identify_binding_site" in prompt
        assert "compare_molecules" in prompt

    def test_tool_names_match_cheminformatics_tools(self):
        """All tool names in the prompt must match cheminformatics_tools.py factory."""
        from unittest.mock import Mock

        from lobster.agents.drug_discovery.cheminformatics_tools import (
            create_cheminformatics_tools,
        )
        from lobster.core.data_manager_v2 import DataManagerV2

        mock_dm = Mock(spec=DataManagerV2)
        mock_service = Mock()

        tools = create_cheminformatics_tools(
            data_manager=mock_dm,
            molecular_analysis_service=mock_service,
            admet_service=mock_service,
            pubchem_service=mock_service,
            compound_prep_service=mock_service,
        )
        tool_names = [t.name for t in tools]

        prompt = self._get_prompt()
        for name in tool_names:
            assert name in prompt, (
                f"Tool '{name}' from cheminformatics_tools.py not documented in prompt"
            )

    def test_not_responsible_for_target_scoring(self):
        """Cheminformatics should not claim target scoring as its responsibility."""
        prompt = self._get_prompt()
        not_resp_start = prompt.find("<Your_Not_Responsibilities>")
        not_resp_end = prompt.find("</Your_Not_Responsibilities>")
        not_resp = prompt[not_resp_start:not_resp_end]

        assert "target" in not_resp.lower() or "scoring" in not_resp.lower()

    def test_not_responsible_for_synergy(self):
        """Cheminformatics should not claim synergy scoring as its responsibility."""
        prompt = self._get_prompt()
        not_resp_start = prompt.find("<Your_Not_Responsibilities>")
        not_resp_end = prompt.find("</Your_Not_Responsibilities>")
        not_resp = prompt[not_resp_start:not_resp_end]

        assert "clinical" in not_resp.lower() or "synergy" in not_resp.lower()

    def test_smiles_validation_rule(self):
        """Important rules should mention SMILES validation."""
        prompt = self._get_prompt()
        rules_start = prompt.find("<Important_Rules>")
        rules_end = prompt.find("</Important_Rules>")
        rules = prompt[rules_start:rules_end]

        assert "SMILES" in rules or "smiles" in rules.lower()
        assert "validate" in rules.lower()


# =============================================================================
# CLINICAL DEVELOPMENT EXPERT PROMPT TESTS
# =============================================================================


class TestClinicalDevExpertPrompt:
    """Semantic tests for the clinical_dev_expert child prompt."""

    def _get_prompt(self):
        return create_clinical_dev_expert_prompt()

    def test_identifies_as_child(self):
        """Prompt should identify itself as a child of drug_discovery_expert."""
        prompt = self._get_prompt()
        assert "drug_discovery_expert" in prompt
        assert "child" in prompt.lower() or "parent" in prompt.lower()

    def test_not_supervisor_accessible(self):
        """Prompt should state it is not directly accessible by the supervisor."""
        prompt = self._get_prompt()
        assert "NOT directly accessible" in prompt

    def test_mentions_synergy_models(self):
        """Prompt should reference all 3 synergy models."""
        prompt = self._get_prompt()
        assert "Bliss" in prompt
        assert "Loewe" in prompt
        assert "HSA" in prompt

    def test_mentions_key_tools(self):
        """Prompt must document the core clinical tools."""
        prompt = self._get_prompt()
        assert "get_target_disease_evidence" in prompt
        assert "score_drug_synergy" in prompt
        assert "combination_matrix" in prompt
        assert "get_drug_safety_profile" in prompt
        assert "assess_clinical_tractability" in prompt

    def test_mentions_additional_tools(self):
        """Prompt must document trial search, indication mapping, comparison."""
        prompt = self._get_prompt()
        assert "search_clinical_trials" in prompt
        assert "indication_mapping" in prompt
        assert "compare_drug_candidates" in prompt

    def test_tool_names_match_clinical_tools(self):
        """All tool names in the prompt must match clinical_tools.py factory."""
        from unittest.mock import Mock

        from lobster.agents.drug_discovery.clinical_tools import create_clinical_tools
        from lobster.core.data_manager_v2 import DataManagerV2

        mock_dm = Mock(spec=DataManagerV2)
        mock_service = Mock()

        tools = create_clinical_tools(
            data_manager=mock_dm,
            opentargets_service=mock_service,
            synergy_service=mock_service,
            chembl_service=mock_service,
            target_scoring_service=mock_service,
        )
        tool_names = [t.name for t in tools]

        prompt = self._get_prompt()
        for name in tool_names:
            assert name in prompt, (
                f"Tool '{name}' from clinical_tools.py not documented in prompt"
            )

    def test_synergy_classification_thresholds_documented(self):
        """Important rules should document the synergy classification thresholds."""
        prompt = self._get_prompt()
        rules_start = prompt.find("<Important_Rules>")
        rules_end = prompt.find("</Important_Rules>")
        rules = prompt[rules_start:rules_end]

        # Bliss threshold documented
        assert "0.1" in rules
        # Loewe CI thresholds documented
        assert "0.9" in rules or "1.1" in rules

    def test_not_responsible_for_molecular_descriptors(self):
        """Clinical dev should not claim molecular descriptor calculation."""
        prompt = self._get_prompt()
        not_resp_start = prompt.find("<Your_Not_Responsibilities>")
        not_resp_end = prompt.find("</Your_Not_Responsibilities>")
        not_resp = prompt[not_resp_start:not_resp_end]

        assert "molecular" in not_resp.lower() or "cheminformatics" in not_resp.lower()

    def test_not_responsible_for_variants(self):
        """Clinical dev should not claim variant interaction analysis."""
        prompt = self._get_prompt()
        not_resp_start = prompt.find("<Your_Not_Responsibilities>")
        not_resp_end = prompt.find("</Your_Not_Responsibilities>")
        not_resp = prompt[not_resp_start:not_resp_end]

        assert "variant" in not_resp.lower() or "pharmacogenomics" in not_resp.lower()

    def test_mentions_open_targets(self):
        """Prompt should reference Open Targets as a data source."""
        prompt = self._get_prompt()
        assert "Open Targets" in prompt


# =============================================================================
# PHARMACOGENOMICS EXPERT PROMPT TESTS
# =============================================================================


class TestPharmacogenomicsExpertPrompt:
    """Semantic tests for the pharmacogenomics_expert child prompt."""

    def _get_prompt(self):
        return create_pharmacogenomics_expert_prompt()

    def test_identifies_as_child(self):
        """Prompt should identify itself as a child of drug_discovery_expert."""
        prompt = self._get_prompt()
        assert "drug_discovery_expert" in prompt
        assert "child" in prompt.lower() or "parent" in prompt.lower()

    def test_not_supervisor_accessible(self):
        """Prompt should state it is not directly accessible by the supervisor."""
        prompt = self._get_prompt()
        assert "NOT directly accessible" in prompt

    def test_mentions_esm2(self):
        """Prompt should reference ESM2 protein language model."""
        prompt = self._get_prompt()
        assert "ESM2" in prompt

    def test_mentions_key_tools(self):
        """Prompt must document the core pharmacogenomics tools."""
        prompt = self._get_prompt()
        assert "predict_mutation_effect" in prompt
        assert "extract_protein_embedding" in prompt
        assert "compare_variant_sequences" in prompt
        assert "get_variant_drug_interactions" in prompt
        assert "get_pharmacogenomic_evidence" in prompt

    def test_mentions_scoring_tools(self):
        """Prompt must document scoring and correlation tools."""
        prompt = self._get_prompt()
        assert "score_variant_impact" in prompt
        assert "expression_drug_sensitivity" in prompt
        assert "mutation_frequency_analysis" in prompt

    def test_tool_names_match_pharmacogenomics_tools(self):
        """All tool names in the prompt must match pharmacogenomics_tools.py factory."""
        from unittest.mock import Mock

        from lobster.agents.drug_discovery.pharmacogenomics_tools import (
            create_pharmacogenomics_tools,
        )
        from lobster.core.data_manager_v2 import DataManagerV2

        mock_dm = Mock(spec=DataManagerV2)
        mock_service = Mock()

        tools = create_pharmacogenomics_tools(
            data_manager=mock_dm,
            opentargets_service=mock_service,
            chembl_service=mock_service,
        )
        tool_names = [t.name for t in tools]

        prompt = self._get_prompt()
        for name in tool_names:
            assert name in prompt, (
                f"Tool '{name}' from pharmacogenomics_tools.py not documented in prompt"
            )

    def test_mentions_plm_extra_requirement(self):
        """Prompt should note [plm] extra is needed for PLM tools."""
        prompt = self._get_prompt()
        assert "[plm]" in prompt or "plm" in prompt.lower()

    def test_mentions_mutation_notation(self):
        """Important rules should reference mutation notation standard (A123G)."""
        prompt = self._get_prompt()
        rules_start = prompt.find("<Important_Rules>")
        rules_end = prompt.find("</Important_Rules>")
        rules = prompt[rules_start:rules_end]

        assert "A123G" in rules or "mutation" in rules.lower()

    def test_not_responsible_for_molecular_properties(self):
        """Pharmacogenomics should not claim molecular property calculations."""
        prompt = self._get_prompt()
        not_resp_start = prompt.find("<Your_Not_Responsibilities>")
        not_resp_end = prompt.find("</Your_Not_Responsibilities>")
        not_resp = prompt[not_resp_start:not_resp_end]

        assert (
            "molecular" in not_resp.lower()
            or "cheminformatics" in not_resp.lower()
        )

    def test_not_responsible_for_synergy(self):
        """Pharmacogenomics should not claim drug synergy scoring."""
        prompt = self._get_prompt()
        not_resp_start = prompt.find("<Your_Not_Responsibilities>")
        not_resp_end = prompt.find("</Your_Not_Responsibilities>")
        not_resp = prompt[not_resp_start:not_resp_end]

        assert "synergy" in not_resp.lower() or "clinical_dev" in not_resp.lower()

    def test_mentions_confidence_reporting(self):
        """Important rules should mention reporting confidence levels."""
        prompt = self._get_prompt()
        rules_start = prompt.find("<Important_Rules>")
        rules_end = prompt.find("</Important_Rules>")
        rules = prompt[rules_start:rules_end]

        assert "confidence" in rules.lower()


# =============================================================================
# CROSS-AGENT BOUNDARY TESTS
# =============================================================================


class TestAgentBoundaries:
    """Test that no agent claims another agent's responsibilities."""

    def test_parent_does_not_claim_rdkit_analysis(self):
        """Parent prompt should not list RDKit descriptor tools in Your_Tools."""
        prompt = create_drug_discovery_expert_prompt()
        tools_start = prompt.find("<Your_Tools>")
        tools_end = prompt.find("</Your_Tools>")
        tools_section = prompt[tools_start:tools_end]

        # Parent should not have cheminformatics tools
        assert "calculate_descriptors" not in tools_section
        assert "lipinski_check" not in tools_section
        assert "predict_admet" not in tools_section

    def test_parent_does_not_claim_synergy_tools(self):
        """Parent prompt should not list synergy scoring tools in Your_Tools."""
        prompt = create_drug_discovery_expert_prompt()
        tools_start = prompt.find("<Your_Tools>")
        tools_end = prompt.find("</Your_Tools>")
        tools_section = prompt[tools_start:tools_end]

        assert "score_drug_synergy" not in tools_section
        assert "combination_matrix" not in tools_section

    def test_parent_does_not_claim_plm_tools(self):
        """Parent prompt should not list PLM tools in Your_Tools."""
        prompt = create_drug_discovery_expert_prompt()
        tools_start = prompt.find("<Your_Tools>")
        tools_end = prompt.find("</Your_Tools>")
        tools_section = prompt[tools_start:tools_end]

        assert "predict_mutation_effect" not in tools_section
        assert "extract_protein_embedding" not in tools_section

    def test_cheminformatics_does_not_claim_target_tools(self):
        """Cheminformatics prompt should not list target scoring in Your_Tools."""
        prompt = create_cheminformatics_expert_prompt()
        tools_start = prompt.find("<Your_Tools>")
        tools_end = prompt.find("</Your_Tools>")
        tools_section = prompt[tools_start:tools_end]

        assert "search_drug_targets" not in tools_section
        assert "score_drug_target" not in tools_section
        assert "rank_targets" not in tools_section

    def test_clinical_does_not_claim_compound_search(self):
        """Clinical dev prompt should not list compound search in Your_Tools."""
        prompt = create_clinical_dev_expert_prompt()
        tools_start = prompt.find("<Your_Tools>")
        tools_end = prompt.find("</Your_Tools>")
        tools_section = prompt[tools_start:tools_end]

        assert "search_compounds" not in tools_section

    def test_pharmacogenomics_does_not_claim_synergy(self):
        """Pharmacogenomics prompt should not list synergy in Your_Tools."""
        prompt = create_pharmacogenomics_expert_prompt()
        tools_start = prompt.find("<Your_Tools>")
        tools_end = prompt.find("</Your_Tools>")
        tools_section = prompt[tools_start:tools_end]

        assert "score_drug_synergy" not in tools_section
        assert "combination_matrix" not in tools_section
