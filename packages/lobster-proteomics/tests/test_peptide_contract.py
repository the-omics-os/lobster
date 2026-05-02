"""
AQUADIF contract tests for peptide_expert agent.

Validates plugin API compliance via AgentContractTestMixin.
Run with: pytest packages/lobster-proteomics/tests/test_peptide_contract.py -v
"""

import pytest

from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestPeptideExpertContract(AgentContractTestMixin):
    """Contract compliance tests for peptide_expert child agent."""

    agent_module = "lobster.agents.proteomics.peptide_expert"
    factory_name = "peptide_expert"
    expected_tier = "free"
    is_parent_agent = False
    tools_required = True
