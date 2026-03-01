"""
AQUADIF contract tests for the lobster-proteomics package.

Validates that all tools in proteomics_expert, proteomics_de_analysis_expert,
and biomarker_discovery_expert have correct AQUADIF metadata.

Run with: pytest -m contract tests/agents/test_aquadif_proteomics.py -v
"""
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestAquadifProteomicsExpert(AgentContractTestMixin):
    """AQUADIF contract tests for proteomics_expert.

    proteomics_expert is the parent agent with 2 children:
    proteomics_de_analysis_expert and biomarker_discovery_expert.
    It owns IMPORT tools (import_proteomics_data, import_ptm_sites,
    import_affinity_data) and QUALITY tools (assess_proteomics_quality,
    assess_lod_quality, validate_antibody_specificity) — MVP parent check applies.
    """

    agent_module = "lobster.agents.proteomics.proteomics_expert"
    factory_name = "proteomics_expert"
    is_parent_agent = True  # Has children: proteomics_de_analysis_expert, biomarker_discovery_expert


@pytest.mark.contract
class TestAquadifProteomicsDeAnalysisExpert(AgentContractTestMixin):
    """AQUADIF contract tests for proteomics_de_analysis_expert.

    Note: factory_name is 'de_analysis_expert' (the Python function name),
    not 'proteomics_de_analysis_expert' (the entry point key).
    """

    agent_module = "lobster.agents.proteomics.de_analysis_expert"
    factory_name = "de_analysis_expert"
    is_parent_agent = False


@pytest.mark.contract
class TestAquadifBiomarkerDiscoveryExpert(AgentContractTestMixin):
    """AQUADIF contract tests for biomarker_discovery_expert."""

    agent_module = "lobster.agents.proteomics.biomarker_discovery_expert"
    factory_name = "biomarker_discovery_expert"
    is_parent_agent = False
