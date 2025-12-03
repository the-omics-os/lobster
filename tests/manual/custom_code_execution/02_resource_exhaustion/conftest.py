"""
pytest configuration for resource exhaustion tests.

Fixtures:
- service: CustomCodeExecutionService with temporary workspace
"""

from pathlib import Path

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CustomCodeExecutionService,
)


@pytest.fixture
def service(tmp_path):
    """
    Create CustomCodeExecutionService instance with temporary workspace.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        CustomCodeExecutionService instance

    Example:
        def test_something(service):
            result, stats, ir = service.execute("result = 1 + 1", persist=False)
            assert result == 2
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_manager = DataManagerV2(workspace_path=workspace)
    return CustomCodeExecutionService(data_manager)
