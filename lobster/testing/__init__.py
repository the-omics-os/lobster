"""
Lobster Testing Module - Shared testing utilities for all packages.

This module provides consistent mock objects, fixtures, and contract test
mixins that all agent packages and integration tests can import. It eliminates
duplication and ensures workspace_path is always Path (not str) per Phase 5
decision 05-02.

Public API:
    - MockDataManager: Mock DataManagerV2 with in-memory behavior
    - MockLLM: Mock LLM with configurable responses
    - MockProvenanceTracker: Mock provenance tracker for IR
    - AgentContractTestMixin: Test mixin for plugin API compliance validation

Example - Using mocks:
    >>> from lobster.testing import MockDataManager, MockLLM
    >>> from pathlib import Path
    >>>
    >>> # Create mock data manager
    >>> dm = MockDataManager(workspace_path=Path('/tmp/test'))
    >>> dm.add_modality('test', adata)
    >>>
    >>> # Create mock LLM with configurable responses
    >>> llm = MockLLM(default_response='Analysis complete')
    >>> llm.set_response_sequence(['First', 'Second', 'Third'])

Example - Contract testing:
    >>> from lobster.testing import AgentContractTestMixin
    >>>
    >>> class TestMyAgent(AgentContractTestMixin):
    ...     agent_module = 'my_package.my_agent'
    ...     factory_name = 'my_agent'
    ...
    >>> # Run with pytest - validates plugin API compliance
"""

from lobster.testing.mock_data_manager import MockDataManager, MockProvenanceTracker
from lobster.testing.mock_llm import MockLLM, MockLLMResponse
from lobster.testing.contract_mixins import AgentContractTestMixin
from lobster.testing.fixtures import create_test_workspace

__all__ = [
    # Mock objects
    "MockDataManager",
    "MockLLM",
    "MockLLMResponse",
    "MockProvenanceTracker",
    # Contract test mixins
    "AgentContractTestMixin",
    # Factory functions
    "create_test_workspace",
]
