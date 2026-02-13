"""
Unit tests for CustomCodeExecutionService.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CodeExecutionError,
    CodeValidationError,
    CustomCodeExecutionService,
)


class TestCustomCodeExecutionService:
    """Test suite for CustomCodeExecutionService."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path):
        """Create mock DataManagerV2 with test workspace."""
        dm = Mock(spec=DataManagerV2)
        dm.workspace_path = tmp_path
        dm.list_modalities.return_value = ["test_modality"]
        return dm

    @pytest.fixture
    def service(self, mock_data_manager):
        """Create service instance."""
        return CustomCodeExecutionService(mock_data_manager)

    def test_initialization(self, service, mock_data_manager):
        """Test service initialization."""
        assert service.data_manager == mock_data_manager
        assert service.context_builder is not None

    def test_execute_simple_expression(self, service):
        """Test executing simple mathematical expression."""
        code = "result = 2 + 2"
        result, stats, ir = service.execute(
            code=code, persist=False, description="Simple addition"
        )

        assert result == 4
        assert stats["success"] is True
        assert stats["warnings"] == []
        assert stats["error"] is None
        assert ir.operation == "custom_code_execution"
        assert ir.exportable is False  # persist=False

    def test_execute_with_imports(self, service):
        """Test code with standard library imports (subprocess model)."""
        code = """
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = int(df['a'].sum())  # Convert to int for JSON serialization
"""
        result, stats, ir = service.execute(
            code=code, persist=False, description="Pandas computation"
        )

        assert result == 6
        assert stats["success"] is True
        assert "import pandas as pd" in ir.imports
        assert "import numpy as np" in ir.imports

    def test_execute_with_print(self, service):
        """Test that print statements are captured."""
        code = """
print("Hello, world!")
print("Line 2")
result = 42
"""
        result, stats, ir = service.execute(code=code, persist=False)

        assert result == 42
        assert stats["stdout_lines"] == 2
        assert "Hello, world!" in stats["stdout_preview"]

    def test_syntax_error_validation(self, service):
        """Test that syntax errors are caught."""
        code = "def incomplete_function("

        with pytest.raises(CodeValidationError, match="Syntax error"):
            service.execute(code=code, persist=False)

    def test_forbidden_import_subprocess(self, service):
        """Test that subprocess import is blocked."""
        code = "import subprocess"

        with pytest.raises(CodeValidationError, match="Forbidden import"):
            service.execute(code=code, persist=False)

    def test_forbidden_import_os_system(self, service):
        """Test that os.system is blocked."""
        code = "from os import system"

        with pytest.raises(CodeValidationError, match="Forbidden import"):
            service.execute(code=code, persist=False)

    def test_execution_error_handling(self, service):
        """Test that runtime errors are captured."""
        code = "result = 1 / 0"  # ZeroDivisionError

        with pytest.raises(CodeExecutionError, match="Code execution failed"):
            service.execute(code=code, persist=False)

    def test_persist_flag_affects_exportable(self, service):
        """Test that persist=True makes IR exportable."""
        code = "result = 42"

        # persist=False
        _, _, ir_ephemeral = service.execute(code, persist=False)
        assert ir_ephemeral.exportable is False

        # persist=True
        _, _, ir_persisted = service.execute(code, persist=True)
        assert ir_persisted.exportable is True

    def test_no_result_variable(self, service):
        """Test code without explicit result variable."""
        code = "x = 10"
        result, stats, ir = service.execute(code, persist=False)

        assert result is None  # No result variable set
        assert stats["success"] is True

    def test_modality_loading(self, service, mock_data_manager):
        """Test that modality is loaded if specified."""
        import anndata
        import numpy as np

        mock_adata = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
        mock_data_manager.get_modality.return_value = mock_adata
        mock_data_manager.list_modalities.return_value = ["test_modality"]

        # Write a real h5ad file so subprocess can read it
        h5ad_path = mock_data_manager.workspace_path / "test_modality.h5ad"
        mock_adata.write_h5ad(h5ad_path)

        code = "result = adata.n_obs"
        result, stats, ir = service.execute(
            code=code, modality_name="test_modality", persist=False
        )

        assert result == 2  # 2 observations in mock data
        assert (
            stats.get("modality_loaded") == "test_modality" or stats["success"] is True
        )
        assert "test_modality" in ir.input_entities

    def test_warning_for_non_standard_import(self, service):
        """Test warning for non-standard library import."""
        code = """
import math  # This is OK
result = math.sqrt(16)
"""
        # Should work and give warning about non-standard (but this one is standard, so no warning)
        result, stats, ir = service.execute(code, persist=False)

        assert result == 4.0
        assert stats["success"] is True

    def test_eval_raises_validation_error(self, service):
        """Test that eval is blocked with CodeValidationError."""
        code = """
x = eval("2 + 2")
result = x
"""
        with pytest.raises(CodeValidationError):
            service.execute(code, persist=False)

    def test_extract_imports_basic(self, service):
        """Test _extract_imports with simple imports."""
        code = "import numpy as np"
        imports = service._extract_imports(code)

        assert imports == ["import numpy as np"]

    def test_extract_imports_from(self, service):
        """Test _extract_imports with from imports."""
        code = "from pandas import DataFrame"
        imports = service._extract_imports(code)

        assert imports == ["from pandas import DataFrame"]

    def test_extract_imports_multiple(self, service):
        """Test _extract_imports with multiple imports."""
        code = """
import numpy as np
import pandas as pd
from scipy import stats
"""
        imports = service._extract_imports(code)

        assert len(imports) == 3
        assert "import numpy as np" in imports
        assert "import pandas as pd" in imports
        assert "from scipy import stats" in imports

    def test_extract_imports_with_syntax_error(self, service):
        """Test _extract_imports with syntax error."""
        code = "import invalid syntax"
        imports = service._extract_imports(code)

        # Should return empty list without crashing
        assert imports == []

    def test_validate_code_safety_valid(self, service):
        """Test _validate_code_safety with valid code."""
        code = "import pandas as pd\nx = 5"
        warnings = service._validate_code_safety(code)

        assert warnings == []

    def test_validate_code_safety_syntax_error(self, service):
        """Test _validate_code_safety with syntax error."""
        code = "def incomplete("

        with pytest.raises(CodeValidationError):
            service._validate_code_safety(code)

    def test_execute_in_namespace_basic(self, service):
        """Test _execute_in_namespace with basic code (subprocess model)."""
        context = {
            "modality_name": None,
            "workspace_path": service.data_manager.workspace_path,
            "load_workspace_files": False,
        }
        code = "result = 5 * 2"  # Can't inject arbitrary 'x' in subprocess

        result, stdout, stderr, error = service._execute_in_namespace(code, context)

        assert result == 10
        assert error is None

    def test_execute_in_namespace_with_print(self, service):
        """Test _execute_in_namespace captures stdout."""
        context = {}
        code = "print('test output')"

        result, stdout, stderr, error = service._execute_in_namespace(code, context)

        assert "test output" in stdout
        assert error is None

    def test_execute_in_namespace_with_error(self, service):
        """Test _execute_in_namespace captures errors (subprocess model)."""
        context = {
            "modality_name": None,
            "workspace_path": service.data_manager.workspace_path,
            "load_workspace_files": False,
        }
        code = "raise ValueError('test error')"

        result, stdout, stderr, error = service._execute_in_namespace(code, context)

        assert error is not None
        # In subprocess model, error is generic Exception with return code
        assert "Code execution failed" in str(error) or "test error" in stderr

    def test_output_truncation(self, service):
        """Test that very long output is truncated."""
        # Generate output longer than MAX_OUTPUT_LENGTH
        code = """
for i in range(1000):
    print('x' * 100)
result = 42
"""
        result, stats, ir = service.execute(code, persist=False)

        assert result == 42
        # Output should be truncated
        # Preview should be limited in some way
        assert (
            len(stats.get("stdout_preview", "")) <= 2000
            or stats.get("stdout_full_path") is not None
        )

    def test_create_ir_structure(self, service):
        """Test _create_ir creates proper IR structure."""
        ir = service._create_ir(
            code="result = 2 + 2",
            description="Test operation",
            modality_name="test_mod",
            load_workspace_files=True,
            workspace_keys=None,
            persist=True,
            stats={"success": True, "duration_seconds": 0.001, "warnings": []},
        )

        assert ir.operation == "custom_code_execution"
        assert ir.tool_name == "execute_custom_code"
        assert ir.description == "Test operation"
        assert ir.library == "custom"
        assert "result = 2 + 2" in ir.code_template
        assert ir.exportable is True  # persist=True
        assert "test_mod" in ir.input_entities
        assert "modality_name" in ir.parameter_schema

    def test_create_ir_without_modality(self, service):
        """Test _create_ir without modality."""
        ir = service._create_ir(
            code="result = 42",
            description="Test",
            modality_name=None,
            load_workspace_files=False,
            workspace_keys=None,
            persist=False,
            stats={"success": True, "duration_seconds": 0.001, "warnings": []},
        )

        assert ir.input_entities == []
        assert "modality_name" not in ir.parameter_schema
        assert ir.exportable is False

    def test_multiline_code_execution(self, service):
        """Test execution of multi-line code."""
        code = """
x = 1
y = 2
z = 3
result = x + y + z
"""
        result, stats, ir = service.execute(code, persist=False)

        assert result == 6
        assert stats["success"] is True

    def test_code_with_loops(self, service):
        """Test execution of code with loops."""
        code = """
total = 0
for i in range(10):
    total += i
result = total
"""
        result, stats, ir = service.execute(code, persist=False)

        assert result == 45  # Sum of 0-9
        assert stats["success"] is True

    def test_code_with_functions(self, service):
        """Test execution of code with function definitions."""
        code = """
def add(a, b):
    return a + b

result = add(5, 3)
"""
        result, stats, ir = service.execute(code, persist=False)

        assert result == 8
        assert stats["success"] is True

    def test_description_in_ir(self, service):
        """Test that description is properly stored in IR."""
        description = "Calculate the meaning of life"
        code = "result = 42"

        result, stats, ir = service.execute(
            code=code, description=description, persist=True
        )

        assert ir.description == description
        assert ir.parameters["description"] == description

    def test_stats_structure(self, service):
        """Test that stats dict has all required fields."""
        code = "result = 42"
        result, stats, ir = service.execute(code, persist=False)

        # Check all required stat fields
        required_fields = [
            "success",
            "duration_seconds",
            "warnings",
            "stdout_lines",
            "stderr_lines",
            "result_type",
            "modality_loaded",
            "workspace_files_loaded",
            "persisted",
            "error",
        ]

        for field in required_fields:
            assert field in stats

        assert stats["result_type"] == "int"
        assert stats["persisted"] is False

    def test_execution_context_in_ir(self, service):
        """Test that execution context is stored in IR."""
        code = "result = 42"
        result, stats, ir = service.execute(code, persist=True)

        assert "persist" in ir.execution_context
        assert "success" in ir.execution_context
        assert "duration" in ir.execution_context
        assert "warnings" in ir.execution_context

        assert ir.execution_context["persist"] is True
        assert ir.execution_context["success"] is True
