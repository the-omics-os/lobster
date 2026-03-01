"""
AQUADIF contract tests for the lobster-metadata package.

Validates that all tools in metadata_assistant have correct AQUADIF metadata
(categories, provenance, AST-validated provenance calls).

Run with: pytest -m contract tests/agents/test_aquadif_metadata.py
"""

import pytest

from lobster.testing.contract_mixins import AgentContractTestMixin


@pytest.mark.contract
class TestMetadataAssistantAquadif(AgentContractTestMixin):
    """Contract tests for metadata_assistant AQUADIF compliance."""

    agent_module = "lobster.agents.metadata_assistant"
    factory_module = "lobster.agents.metadata_assistant.metadata_assistant"
    factory_name = "metadata_assistant"
    is_parent_agent = False
