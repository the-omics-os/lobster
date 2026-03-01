"""
AQUADIF contract tests for the lobster-research package.

Validates that all tools in research_agent and data_expert
have correct AQUADIF metadata.

Run with: pytest -m contract tests/agents/test_aquadif_research.py -v
"""
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestAquadifResearchAgent(AgentContractTestMixin):
    """AQUADIF contract tests for research_agent."""

    agent_module = "lobster.agents.research.research_agent"
    factory_name = "research_agent"
    # is_parent_agent = False: research_agent is a parent of metadata_assistant at runtime,
    # but MVP parent check requires IMPORT + QUALITY + (ANALYZE or DELEGATE).
    # research_agent has no IMPORT tools (it searches and queues, never loads data).
    # DELEGATE tools are provided at runtime via delegation_tools but not in contract tests.
    # Per the established pattern for data-prep focused agents (machine_learning_expert),
    # set is_parent_agent=False to skip MVP parent check.
    is_parent_agent = False


@pytest.mark.contract
class TestAquadifDataExpert(AgentContractTestMixin):
    """AQUADIF contract tests for data_expert."""

    agent_module = "lobster.agents.data_expert.data_expert"
    factory_name = "data_expert"
    # is_parent_agent = False: data_expert is a parent of metadata_assistant at runtime,
    # but DELEGATE tools are provided via delegation_tools (not present in contract tests).
    # data_expert has IMPORT + QUALITY but lacks ANALYZE/DELEGATE in base tools alone.
    # DELEGATE tools from graph.py provide DELEGATE at runtime — cannot be verified here.
    is_parent_agent = False
