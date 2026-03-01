"""
AQUADIF contract tests for the lobster-metabolomics package.

Validates that all tools in metabolomics_expert have correct AQUADIF metadata.

Run with: pytest -m contract tests/agents/test_aquadif_metabolomics.py -v
"""
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestAquadifMetabolomicsExpert(AgentContractTestMixin):
    """AQUADIF contract tests for metabolomics_expert."""

    agent_module = "lobster.agents.metabolomics.metabolomics_expert"
    factory_name = "metabolomics_expert"
    is_parent_agent = False  # No child agents
