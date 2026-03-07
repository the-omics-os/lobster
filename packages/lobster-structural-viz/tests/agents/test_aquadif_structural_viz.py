"""
AQUADIF contract tests for the lobster-structural-viz package.

Validates that all tools in protein_structure_visualization_expert have correct AQUADIF metadata.

Run with: pytest -m contract tests/agents/test_aquadif_structural_viz.py -v
"""
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestAquadifProteinStructureVisualizationExpert(AgentContractTestMixin):
    """AQUADIF contract tests for protein_structure_visualization_expert."""

    agent_module = "lobster.agents.protein_structure_visualization_expert"
    factory_name = "protein_structure_visualization_expert"
    is_parent_agent = False  # No child agents
