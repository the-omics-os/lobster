"""
AQUADIF contract tests for the lobster-drug-discovery package.

Validates that all tools in drug_discovery_expert, cheminformatics_expert,
clinical_dev_expert, and pharmacogenomics_expert have correct AQUADIF metadata.

Run with: pytest -m contract tests/agents/test_aquadif_drug_discovery.py -v
"""
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestAquadifDrugDiscoveryExpert(AgentContractTestMixin):
    """AQUADIF contract tests for drug_discovery_expert."""

    agent_module = "lobster.agents.drug_discovery.drug_discovery_expert"
    factory_name = "drug_discovery_expert"
    is_parent_agent = False  # No IMPORT/QUALITY lifecycle tools — query/analysis-centric parent (like machine_learning_expert)


@pytest.mark.contract
class TestAquadifCheminformaticsExpert(AgentContractTestMixin):
    """AQUADIF contract tests for cheminformatics_expert."""

    agent_module = "lobster.agents.drug_discovery.cheminformatics_expert"
    factory_name = "cheminformatics_expert"
    is_parent_agent = False


@pytest.mark.contract
class TestAquadifClinicalDevExpert(AgentContractTestMixin):
    """AQUADIF contract tests for clinical_dev_expert."""

    agent_module = "lobster.agents.drug_discovery.clinical_dev_expert"
    factory_name = "clinical_dev_expert"
    is_parent_agent = False


@pytest.mark.contract
class TestAquadifPharmacogenomicsExpert(AgentContractTestMixin):
    """AQUADIF contract tests for pharmacogenomics_expert."""

    agent_module = "lobster.agents.drug_discovery.pharmacogenomics_expert"
    factory_name = "pharmacogenomics_expert"
    is_parent_agent = False
