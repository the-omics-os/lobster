"""
AQUADIF contract tests for the lobster-genomics package.

Validates that all tools in genomics_expert and variant_analysis_expert
have correct AQUADIF metadata.

Run with: pytest -m contract tests/agents/test_aquadif_genomics.py -v
"""
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestAquadifGenomicsExpert(AgentContractTestMixin):
    """AQUADIF contract tests for genomics_expert."""

    agent_module = "lobster.agents.genomics.genomics_expert"
    factory_name = "genomics_expert"
    is_parent_agent = True  # Has child: variant_analysis_expert


@pytest.mark.contract
class TestAquadifVariantAnalysisExpert(AgentContractTestMixin):
    """AQUADIF contract tests for variant_analysis_expert."""

    agent_module = "lobster.agents.genomics.variant_analysis_expert"
    factory_name = "variant_analysis_expert"
    is_parent_agent = False
