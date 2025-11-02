"""
Unit tests for notebook_validator.py.

Tests syntax validation, import checking, and common issue detection.
"""

import tempfile
from pathlib import Path

import nbformat
import pytest

from lobster.core.notebook_validator import (
    NotebookValidator,
    NotebookValidationResult,
    ValidationIssue,
)


def create_test_notebook(cells_code: list) -> Path:
    """
    Create a temporary test notebook with given code cells.

    Args:
        cells_code: List of code strings for cells

    Returns:
        Path to created notebook
    """
    nb = nbformat.v4.new_notebook()

    for code in cells_code:
        nb.cells.append(nbformat.v4.new_code_cell(code))

    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.ipynb', delete=False
    )
    nbformat.write(nb, temp_file)
    temp_file.close()

    return Path(temp_file.name)


class TestValidationIssue:
    """Test ValidationIssue dataclass."""

    def test_issue_creation(self):
        """Test creating a validation issue."""
        issue = ValidationIssue(
            severity='error',
            cell_index=0,
            line_number=5,
            message='Test error',
            code_snippet='x = y'
        )

        assert issue.severity == 'error'
        assert issue.cell_index == 0
        assert issue.line_number == 5
        assert 'Test error' in str(issue)
        assert 'x = y' in str(issue)

    def test_issue_without_line_number(self):
        """Test issue without line number."""
        issue = ValidationIssue(
            severity='warning',
            cell_index=2,
            message='Test warning'
        )

        assert 'Cell 2' in str(issue)
        assert 'WARNING' in str(issue)


class TestNotebookValidationResult:
    """Test NotebookValidationResult dataclass."""

    def test_empty_result_is_valid(self):
        """Test empty result is valid."""
        result = NotebookValidationResult()
        assert result.is_valid
        assert not result.has_errors
        assert not result.has_warnings

    def test_add_error_invalidates_result(self):
        """Test adding error sets is_valid to False."""
        result = NotebookValidationResult()
        result.add_error('Test error', cell_index=0)

        assert not result.is_valid
        assert result.has_errors
        assert result.error_count == 1

    def test_add_warning_keeps_valid(self):
        """Test adding warning doesn't invalidate result."""
        result = NotebookValidationResult()
        result.add_warning('Test warning', cell_index=0)

        # Initially valid, but has_warnings should be True
        assert result.is_valid  # No errors
        assert result.has_warnings
        assert result.warning_count == 1

    def test_multiple_issues(self):
        """Test handling multiple issues."""
        result = NotebookValidationResult()
        result.add_error('Error 1', cell_index=0)
        result.add_error('Error 2', cell_index=1)
        result.add_warning('Warning 1', cell_index=2)

        assert not result.is_valid
        assert result.error_count == 2
        assert result.warning_count == 1
        assert len(result.issues) == 3

    def test_string_representation(self):
        """Test string representation."""
        result = NotebookValidationResult()

        # Valid result
        assert '✓' in str(result)

        # Result with errors
        result.add_error('Test error', cell_index=0)
        result_str = str(result)
        assert '✗' in result_str
        assert 'error' in result_str.lower()


class TestNotebookValidator:
    """Test NotebookValidator class."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = NotebookValidator()
        assert validator is not None

        validator_strict = NotebookValidator(strict_imports=True)
        assert validator_strict.strict_imports

        validator_lenient = NotebookValidator(strict_imports=False)
        assert not validator_lenient.strict_imports

    def test_validate_valid_notebook(self):
        """Test validation of valid notebook."""
        code_cells = [
            "import numpy as np",
            "x = np.array([1, 2, 3])",
            "print(x.mean())"
        ]

        notebook_path = create_test_notebook(code_cells)

        try:
            validator = NotebookValidator()
            result = validator.validate(notebook_path)

            assert result.is_valid
            assert not result.has_errors
            assert 'numpy' in result.imports_found

        finally:
            notebook_path.unlink()

    def test_validate_syntax_error(self):
        """Test detection of syntax errors."""
        code_cells = [
            "import numpy as np",
            "x = y z  # Syntax error",
            "print(x)"
        ]

        notebook_path = create_test_notebook(code_cells)

        try:
            validator = NotebookValidator()
            result = validator.validate(notebook_path)

            assert not result.is_valid
            assert result.has_errors
            assert any('syntax' in issue.message.lower() for issue in result.issues)

        finally:
            notebook_path.unlink()

    def test_validate_missing_import(self):
        """Test detection of missing imports."""
        code_cells = [
            "import numpy as np",
            "import nonexistent_module_xyz123",  # This should fail
            "print('test')"
        ]

        notebook_path = create_test_notebook(code_cells)

        try:
            validator = NotebookValidator(strict_imports=True)
            result = validator.validate(notebook_path)

            # Should have error due to missing import
            assert not result.is_valid
            assert result.has_errors
            assert 'nonexistent_module_xyz123' in result.missing_imports

        finally:
            notebook_path.unlink()

    def test_validate_missing_import_lenient(self):
        """Test lenient mode treats missing imports as warnings."""
        code_cells = [
            "import numpy as np",
            "import nonexistent_module_xyz123",
            "print('test')"
        ]

        notebook_path = create_test_notebook(code_cells)

        try:
            validator = NotebookValidator(strict_imports=False)
            result = validator.validate(notebook_path)

            # Should be valid (warnings don't block)
            # But should have warnings
            assert result.has_warnings
            assert 'nonexistent_module_xyz123' in result.missing_imports

        finally:
            notebook_path.unlink()

    def test_extract_imports(self):
        """Test import extraction."""
        code_cells = [
            "import numpy as np",
            "from pandas import DataFrame",
            "import scipy.stats as stats",
            "from matplotlib.pyplot import plot"
        ]

        notebook_path = create_test_notebook(code_cells)

        try:
            validator = NotebookValidator()
            result = validator.validate(notebook_path)

            assert 'numpy' in result.imports_found
            assert 'pandas' in result.imports_found
            assert 'scipy' in result.imports_found
            assert 'matplotlib' in result.imports_found

        finally:
            notebook_path.unlink()

    def test_common_issues_detection(self):
        """Test detection of common code issues."""
        code_cells = [
            "import numpy as np",
            "try:\n    x = 1/0\nexcept:\n    pass  # Bare except",
            "result = eval('1 + 1')  # Dangerous eval"
        ]

        notebook_path = create_test_notebook(code_cells)

        try:
            validator = NotebookValidator()
            result = validator.validate(notebook_path)

            # Should have warnings for common issues
            assert result.has_warnings

            warning_messages = [issue.message.lower() for issue in result.issues if issue.severity == 'warning']

            # Check for bare except warning
            assert any('except' in msg for msg in warning_messages)

            # Check for eval warning
            assert any('eval' in msg or 'exec' in msg for msg in warning_messages)

        finally:
            notebook_path.unlink()

    def test_empty_cells_ignored(self):
        """Test that empty cells are ignored."""
        code_cells = [
            "import numpy as np",
            "",  # Empty cell
            "   ",  # Whitespace only
            "print('test')"
        ]

        notebook_path = create_test_notebook(code_cells)

        try:
            validator = NotebookValidator()
            result = validator.validate(notebook_path)

            assert result.is_valid
            assert not result.has_errors

        finally:
            notebook_path.unlink()

    def test_validate_quick_success(self):
        """Test quick validation with valid notebook."""
        code_cells = [
            "import numpy as np",
            "x = np.array([1, 2, 3])",
            "print(x)"
        ]

        notebook_path = create_test_notebook(code_cells)

        try:
            validator = NotebookValidator()
            is_valid = validator.validate_quick(notebook_path)

            assert is_valid

        finally:
            notebook_path.unlink()

    def test_validate_quick_failure(self):
        """Test quick validation with syntax error."""
        code_cells = [
            "import numpy as np",
            "x = y z  # Syntax error"
        ]

        notebook_path = create_test_notebook(code_cells)

        try:
            validator = NotebookValidator()
            is_valid = validator.validate_quick(notebook_path)

            assert not is_valid

        finally:
            notebook_path.unlink()

    def test_validate_nonexistent_notebook(self):
        """Test validation of nonexistent notebook."""
        validator = NotebookValidator()
        result = validator.validate(Path("/nonexistent/notebook.ipynb"))

        assert not result.is_valid
        assert result.has_errors
        assert any('cannot read' in issue.message.lower() for issue in result.issues)

    def test_multiple_syntax_errors(self):
        """Test handling of multiple syntax errors."""
        code_cells = [
            "import numpy as np",
            "x = y z  # Error 1",
            "def broken(:\n    pass  # Error 2"
        ]

        notebook_path = create_test_notebook(code_cells)

        try:
            validator = NotebookValidator()
            result = validator.validate(notebook_path)

            assert not result.is_valid
            assert result.error_count >= 2  # At least 2 syntax errors

        finally:
            notebook_path.unlink()
