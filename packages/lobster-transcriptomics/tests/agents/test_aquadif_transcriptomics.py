"""
AQUADIF contract tests for the lobster-transcriptomics package.

Validates that all tools in transcriptomics_expert, annotation_expert,
and de_analysis_expert have correct AQUADIF metadata (categories, provenance,
AST-validated provenance calls).

Run with: pytest -m contract tests/agents/test_aquadif_transcriptomics.py
"""

import pytest

from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestTranscriptomicsExpertAquadif(AgentContractTestMixin):
    """Contract tests for transcriptomics_expert AQUADIF compliance."""

    agent_module = "lobster.agents.transcriptomics.transcriptomics_expert"
    factory_name = "transcriptomics_expert"
    is_parent_agent = True  # Has annotation_expert and de_analysis_expert children


@pytest.mark.contract
class TestAnnotationExpertAquadif(AgentContractTestMixin):
    """Contract tests for annotation_expert AQUADIF compliance."""

    agent_module = "lobster.agents.transcriptomics.annotation_expert"
    factory_name = "annotation_expert"
    is_parent_agent = False


@pytest.mark.contract
class TestDeAnalysisExpertAquadif(AgentContractTestMixin):
    """Contract tests for de_analysis_expert AQUADIF compliance."""

    agent_module = "lobster.agents.transcriptomics.de_analysis_expert"
    factory_name = "de_analysis_expert"
    is_parent_agent = False
