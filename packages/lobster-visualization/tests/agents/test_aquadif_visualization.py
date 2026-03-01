"""
AQUADIF contract tests for the lobster-visualization package.

Validates that all tools in visualization_expert have correct AQUADIF metadata.

Run with: pytest -m contract tests/agents/test_aquadif_visualization.py -v
"""
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestAquadifVisualizationExpert(AgentContractTestMixin):
    """AQUADIF contract tests for visualization_expert."""

    agent_module = "lobster.agents.visualization_expert"
    factory_name = "visualization_expert"
    is_parent_agent = False
