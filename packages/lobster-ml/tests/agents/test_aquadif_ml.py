"""
AQUADIF contract tests for the lobster-ml package.

Validates that all tools in machine_learning_expert, feature_selection_expert,
and survival_analysis_expert have correct AQUADIF metadata.

Run with: pytest -m contract tests/agents/test_aquadif_ml.py -v
"""
import pytest
from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestAquadifMachineLearningExpert(AgentContractTestMixin):
    """AQUADIF contract tests for machine_learning_expert.

    Note: is_parent_agent=False because machine_learning_expert is architecturally
    a parent (has feature_selection_expert and survival_analysis_expert children)
    but does not own IMPORT or QUALITY tools — it works exclusively on pre-loaded
    data. The MVP parent check (IMPORT + QUALITY + ANALYZE/DELEGATE) does not
    apply to this data-preparation-focused agent.
    """

    agent_module = "lobster.agents.machine_learning.machine_learning_expert"
    factory_name = "machine_learning_expert"
    is_parent_agent = False  # ML expert works on pre-loaded data; no IMPORT/QUALITY lifecycle


@pytest.mark.contract
class TestAquadifFeatureSelectionExpert(AgentContractTestMixin):
    """AQUADIF contract tests for feature_selection_expert."""

    agent_module = "lobster.agents.machine_learning.feature_selection_expert"
    factory_name = "feature_selection_expert"
    is_parent_agent = False


@pytest.mark.contract
class TestAquadifSurvivalAnalysisExpert(AgentContractTestMixin):
    """AQUADIF contract tests for survival_analysis_expert."""

    agent_module = "lobster.agents.machine_learning.survival_analysis_expert"
    factory_name = "survival_analysis_expert"
    is_parent_agent = False
