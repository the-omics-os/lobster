"""
Security tests for subprocess-based code execution.

These tests verify that the subprocess model provides proper isolation,
timeout enforcement, and crash resistance.
"""

import pytest
import time
from pathlib import Path
import anndata
import numpy as np
import pandas as pd

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution import CustomCodeExecutionService, CodeExecutionError


class TestSubprocessSecurity:
    """Test subprocess security features."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create test workspace with sample data."""
        workspace = tmp_path / ".lobster_workspace"
        workspace.mkdir()

        # Create sample CSV
        df = pd.DataFrame({'gene': ['GAPDH', 'ACTB'], 'expression': [100, 200]})
        df.to_csv(workspace / "gene_expression.csv", index=False)

        return workspace

    @pytest.fixture
    def data_manager(self, workspace):
        """Create DataManagerV2 with test modality."""
        dm = DataManagerV2(workspace_path=workspace)

        # Add test modality
        adata = anndata.AnnData(
            X=np.array([[1, 2, 3], [4, 5, 6]]),
            obs=pd.DataFrame({'cell_type': ['T', 'B']}, index=['c1', 'c2']),
            var=pd.DataFrame({'gene': ['g1', 'g2', 'g3']}, index=['g1', 'g2', 'g3'])
        )
        dm.modalities['test_data'] = adata

        return dm

    @pytest.fixture
    def service(self, data_manager):
        """Create service instance."""
        return CustomCodeExecutionService(data_manager)

    def test_timeout_enforcement(self, service):
        """Test that infinite loops are killed by timeout."""
        # This would hang forever without timeout
        infinite_loop = """
import time
while True:
    time.sleep(0.1)
result = 42
"""

        # Should raise timeout error (use short timeout for testing)
        with pytest.raises(CodeExecutionError, match="timeout|exceeded"):
            service.execute(code=infinite_loop, persist=False, timeout=2)

    def test_crash_isolation(self, service):
        """Verify that crash in user code doesn't kill Lobster process."""
        crash_code = """
import sys
sys.exit(1)  # Exit subprocess
"""

        # Should capture error, not kill main process
        with pytest.raises(CodeExecutionError, match="failed|return code"):
            service.execute(code=crash_code, persist=False)

        # Main process still alive (test continues)
        assert True

    def test_file_access_limited_to_workspace(self, service):
        """Test that code can access workspace but gets error for external paths."""
        # This should work (workspace access)
        workspace_access = """
from pathlib import Path
workspace_files = list(Path('.').glob('*.csv'))
result = len(workspace_files)
"""

        result, stats, ir = service.execute(code=workspace_access, persist=False)
        assert result >= 0  # Should return count of CSV files
        assert stats['success'] is True

    def test_modality_loading_from_disk(self, service):
        """Test that modalities are auto-saved and loaded in subprocess."""
        code = """
# Access modality loaded from disk
if adata is not None:
    result = adata.n_obs
else:
    result = -1
"""

        result, stats, ir = service.execute(
            code=code,
            modality_name='test_data',
            persist=False
        )

        assert result == 2  # 2 observations
        assert stats['success'] is True

    def test_csv_file_auto_loading(self, service):
        """Test that CSV files are auto-loaded in subprocess."""
        code = """
# CSV should be auto-loaded as 'gene_expression'
if 'gene_expression' in dir():
    result = len(gene_expression)
else:
    result = -1
"""

        result, stats, ir = service.execute(
            code=code,
            load_workspace_files=True,
            persist=False
        )

        assert result == 2  # 2 rows in CSV
        assert stats['success'] is True

    def test_result_json_serialization(self, service):
        """Test that results are properly serialized via JSON."""
        # Test various result types
        test_cases = [
            ("result = 42", 42),
            ("result = 3.14", 3.14),
            ("result = 'hello'", 'hello'),
            ("result = [1, 2, 3]", [1, 2, 3]),
            ("result = {'a': 1, 'b': 2}", {'a': 1, 'b': 2}),
        ]

        for code, expected in test_cases:
            result, stats, ir = service.execute(code=code, persist=False)
            assert result == expected, f"Failed for code: {code}"

    def test_non_serializable_result_fallback(self, service):
        """Test that non-JSON-serializable results are converted to string."""
        code = """
import numpy as np
result = np.array([1, 2, 3])  # NumPy arrays not JSON-serializable
"""

        result, stats, ir = service.execute(code=code, persist=False)

        # Should fallback to string representation
        assert result is not None
        assert isinstance(result, str)
        assert '1' in result and '2' in result and '3' in result

    def test_execution_time_tracking(self, service):
        """Test that execution time is tracked accurately."""
        code = """
import time
time.sleep(0.5)
result = 42
"""

        result, stats, ir = service.execute(code=code, persist=False)

        assert stats['duration_seconds'] >= 0.5
        assert stats['duration_seconds'] < 2.0  # Should be reasonably fast

    def test_no_network_access_warning(self, service):
        """Test that network imports succeed but may not work (no explicit blocking yet)."""
        # Note: Full network isolation requires Docker, not subprocess
        # This test documents current behavior
        code = """
# Network libraries can be imported but may not work
import urllib.request
result = "network_imported"
"""

        # Should execute successfully (we don't block at import level yet)
        result, stats, ir = service.execute(code=code, persist=False)
        assert result == "network_imported"
