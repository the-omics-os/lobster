"""
Unit tests for NotebookExecutor.

Tests notebook validation and execution functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import anndata
import nbformat
import numpy as np
import pandas as pd
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.notebook_executor import NotebookExecutor, ValidationResult


@pytest.fixture
def data_manager():
    """Create mock data manager."""
    dm = Mock(spec=DataManagerV2)
    dm.workspace_dir = Path(tempfile.mkdtemp())
    return dm


@pytest.fixture
def test_adata():
    """Create test AnnData object."""
    n_obs, n_vars = 100, 50
    X = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame(
        {"cell_type": ["TypeA"] * 50 + ["TypeB"] * 50},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_name": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def test_notebook():
    """Create test notebook."""
    nb = new_notebook()

    # Add header
    nb.cells.append(new_markdown_cell("# Test Notebook"))

    # Add parameters cell
    params_cell = new_code_cell("input_data = 'test.h5ad'\nrandom_seed = 42")
    params_cell.metadata["tags"] = ["parameters"]
    nb.cells.append(params_cell)

    # Add analysis cells
    nb.cells.append(new_code_cell("import scanpy as sc\nadata = sc.read_h5ad(input_data)"))
    nb.cells.append(new_code_cell("sc.pp.calculate_qc_metrics(adata)"))
    nb.cells.append(new_code_cell("print('Analysis complete')"))

    # Add metadata
    nb.metadata["lobster"] = {
        "min_cells": 100,
        "required_obs_columns": ["cell_type"],
    }

    return nb


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_init_empty(self):
        """Test empty validation result."""
        result = ValidationResult()
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert not result.has_errors
        assert not result.has_warnings
        assert result.is_valid

    def test_add_error(self):
        """Test adding errors."""
        result = ValidationResult()
        result.add_error("Test error")

        assert result.has_errors
        assert not result.is_valid
        assert "Test error" in result.errors

    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.has_warnings
        assert result.is_valid  # Warnings don't block
        assert "Test warning" in result.warnings

    def test_string_representation(self):
        """Test string representation."""
        result = ValidationResult()
        result.add_error("Error 1")
        result.add_warning("Warning 1")

        str_repr = str(result)
        assert "Error 1" in str_repr
        assert "Warning 1" in str_repr


class TestNotebookExecutor:
    """Test NotebookExecutor class."""

    def test_init(self, data_manager):
        """Test NotebookExecutor initialization."""
        executor = NotebookExecutor(data_manager)
        assert executor.data_manager == data_manager

    def test_validate_input_basic(self, data_manager, test_notebook, test_adata):
        """Test basic input validation."""
        executor = NotebookExecutor(data_manager)

        # Save test notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        # Save test data
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            input_path = Path(f.name)
            test_adata.write_h5ad(input_path)

        try:
            result = executor.validate_input(notebook_path, input_path)
            assert isinstance(result, ValidationResult)
        finally:
            notebook_path.unlink()
            input_path.unlink()

    def test_validate_input_missing_notebook(self, data_manager):
        """Test validation with missing notebook."""
        executor = NotebookExecutor(data_manager)

        result = executor.validate_input(Path("nonexistent.ipynb"), Path("test.h5ad"))

        assert result.has_errors
        assert "Cannot read notebook" in result.errors[0]

    def test_validate_input_missing_data(self, data_manager, test_notebook):
        """Test validation with missing input data."""
        executor = NotebookExecutor(data_manager)

        # Save test notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        try:
            result = executor.validate_input(notebook_path, Path("nonexistent.h5ad"))

            assert result.has_errors
            assert "Cannot read input data" in result.errors[0]
        finally:
            notebook_path.unlink()

    def test_validate_input_checks_shape(self, data_manager, test_notebook, test_adata):
        """Test validation checks data shape."""
        executor = NotebookExecutor(data_manager)

        # Modify notebook to expect more cells
        test_notebook.metadata["lobster"]["min_cells"] = 200

        # Save test notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        # Save test data (only 100 cells)
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            input_path = Path(f.name)
            test_adata.write_h5ad(input_path)

        try:
            result = executor.validate_input(notebook_path, input_path)

            # Should have warning about cell count
            assert result.has_warnings
            assert any("100 cells" in w for w in result.warnings)
        finally:
            notebook_path.unlink()
            input_path.unlink()

    def test_validate_input_checks_columns(
        self, data_manager, test_notebook, test_adata
    ):
        """Test validation checks required columns."""
        executor = NotebookExecutor(data_manager)

        # Require column that doesn't exist
        test_notebook.metadata["lobster"]["required_obs_columns"] = [
            "cell_type",
            "missing_column",
        ]

        # Save test notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        # Save test data
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            input_path = Path(f.name)
            test_adata.write_h5ad(input_path)

        try:
            result = executor.validate_input(notebook_path, input_path)

            # Should have error about missing column
            assert result.has_errors
            assert any("missing_column" in e for e in result.errors)
        finally:
            notebook_path.unlink()
            input_path.unlink()

    def test_dry_run(self, data_manager, test_notebook, test_adata):
        """Test dry run simulation."""
        executor = NotebookExecutor(data_manager)

        # Save test notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        # Save test data
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            input_path = Path(f.name)
            test_adata.write_h5ad(input_path)

        try:
            result = executor.dry_run(notebook_path, input_path)

            assert result["status"] == "dry_run_complete"
            assert "validation" in result
            assert "steps_to_execute" in result
            assert "estimated_duration_minutes" in result
            assert isinstance(result["steps_to_execute"], int)
            assert result["steps_to_execute"] > 0
        finally:
            notebook_path.unlink()
            input_path.unlink()

    @patch("lobster.core.notebook_executor.papermill")
    def test_execute_success(
        self, mock_papermill, data_manager, test_notebook, test_adata
    ):
        """Test successful notebook execution."""
        executor = NotebookExecutor(data_manager)

        # Save test notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        # Save test data
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            input_path = Path(f.name)
            test_adata.write_h5ad(input_path)

        try:
            # Mock papermill.execute_notebook to succeed
            mock_papermill.execute_notebook.return_value = None

            result = executor.execute(notebook_path, input_path)

            assert result["status"] == "success"
            assert "output_notebook" in result
            assert "execution_time" in result
            assert "parameters_used" in result

            # Verify papermill was called
            mock_papermill.execute_notebook.assert_called_once()
        finally:
            notebook_path.unlink()
            input_path.unlink()

    @patch("lobster.core.notebook_executor.papermill")
    def test_execute_with_parameters(
        self, mock_papermill, data_manager, test_notebook, test_adata
    ):
        """Test execution with custom parameters."""
        executor = NotebookExecutor(data_manager)

        # Save test notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        # Save test data
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            input_path = Path(f.name)
            test_adata.write_h5ad(input_path)

        try:
            # Mock papermill
            mock_papermill.execute_notebook.return_value = None

            custom_params = {"random_seed": 123, "custom_param": "value"}
            result = executor.execute(notebook_path, input_path, parameters=custom_params)

            assert result["status"] == "success"

            # Check parameters were passed
            call_args = mock_papermill.execute_notebook.call_args
            passed_params = call_args.kwargs["parameters"]
            assert passed_params["random_seed"] == 123
            assert passed_params["custom_param"] == "value"
        finally:
            notebook_path.unlink()
            input_path.unlink()

    @patch("lobster.core.notebook_executor.papermill")
    def test_execute_validation_failure(
        self, mock_papermill, data_manager, test_notebook, test_adata
    ):
        """Test execution blocked by validation failure."""
        executor = NotebookExecutor(data_manager)

        # Make notebook require missing column
        test_notebook.metadata["lobster"]["required_obs_columns"] = ["missing_column"]

        # Save test notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        # Save test data
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            input_path = Path(f.name)
            test_adata.write_h5ad(input_path)

        try:
            result = executor.execute(notebook_path, input_path)

            assert result["status"] == "validation_failed"
            assert "errors" in result
            assert len(result["errors"]) > 0

            # Papermill should not be called
            mock_papermill.execute_notebook.assert_not_called()
        finally:
            notebook_path.unlink()
            input_path.unlink()

    @patch("lobster.core.notebook_executor.papermill")
    def test_execute_execution_failure(
        self, mock_papermill, data_manager, test_notebook, test_adata
    ):
        """Test handling of execution failure."""
        executor = NotebookExecutor(data_manager)

        # Save test notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        # Save test data
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            input_path = Path(f.name)
            test_adata.write_h5ad(input_path)

        try:
            # Mock papermill to raise error
            mock_papermill.execute_notebook.side_effect = Exception("Execution failed")

            result = executor.execute(notebook_path, input_path)

            assert result["status"] == "failed"
            assert "error" in result
            assert "Execution failed" in result["error"]
        finally:
            notebook_path.unlink()
            input_path.unlink()

    def test_list_parameters(self, data_manager, test_notebook):
        """Test parameter extraction from notebook."""
        executor = NotebookExecutor(data_manager)

        # Save test notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        try:
            params = executor.list_parameters(notebook_path)

            assert "input_data" in params
            assert "random_seed" in params
            assert params["random_seed"] == "42"
        finally:
            notebook_path.unlink()

    def test_list_parameters_no_params_cell(self, data_manager):
        """Test parameter extraction with no parameters cell."""
        executor = NotebookExecutor(data_manager)

        # Create notebook without parameters cell
        nb = new_notebook()
        nb.cells.append(new_markdown_cell("# Test"))
        nb.cells.append(new_code_cell("import numpy as np"))

        # Save notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(nb, f)
            notebook_path = Path(f.name)

        try:
            params = executor.list_parameters(notebook_path)
            assert params == {}
        finally:
            notebook_path.unlink()

    def test_get_execution_summary(self, data_manager, test_notebook):
        """Test execution summary extraction."""
        executor = NotebookExecutor(data_manager)

        # Save test notebook (as if executed)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        try:
            summary = executor.get_execution_summary(notebook_path)

            assert "cells_executed" in summary
            assert "execution_time" in summary
            assert "has_errors" in summary
            assert isinstance(summary["cells_executed"], int)
        finally:
            notebook_path.unlink()

    def test_validate_papermill_availability(self, data_manager):
        """Test Papermill availability check."""
        executor = NotebookExecutor(data_manager)

        # Should be True if papermill is installed
        available = executor.validate_papermill_availability()
        assert isinstance(available, bool)

    def test_validate_papermill_not_available(self, data_manager):
        """Test Papermill not available."""
        executor = NotebookExecutor(data_manager)

        # Mock the PAPERMILL_AVAILABLE flag
        with patch("lobster.core.notebook_executor.PAPERMILL_AVAILABLE", False):
            available = executor.validate_papermill_availability()
            assert not available

    def test_execute_output_path(
        self, data_manager, test_notebook, test_adata
    ):
        """Test custom output path."""
        executor = NotebookExecutor(data_manager)

        # Save test notebook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
            nbformat.write(test_notebook, f)
            notebook_path = Path(f.name)

        # Save test data
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            input_path = Path(f.name)
            test_adata.write_h5ad(input_path)

        custom_output = notebook_path.parent / "custom_output.ipynb"

        try:
            with patch("lobster.core.notebook_executor.papermill") as mock_pm:
                mock_pm.execute_notebook.return_value = None

                result = executor.execute(
                    notebook_path, input_path, output_path=custom_output
                )

                assert result["status"] == "success"
                assert str(custom_output) in result["output_notebook"]

                # Verify custom path was used
                call_args = mock_pm.execute_notebook.call_args
                assert str(custom_output) == call_args.kwargs["output_path"]
        finally:
            notebook_path.unlink()
            input_path.unlink()
            if custom_output.exists():
                custom_output.unlink()
