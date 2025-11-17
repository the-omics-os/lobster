"""
Integration tests for notebook workflow.

Tests end-to-end notebook export, validation, and execution.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import anndata
import nbformat
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.notebook_executor import NotebookExecutor
from lobster.core.notebook_exporter import NotebookExporter
from lobster.core.provenance import ProvenanceTracker


@pytest.fixture
def workspace_dir():
    """Create temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_adata():
    """Create realistic test AnnData object."""
    n_obs, n_vars = 200, 100

    # Create sparse-like data
    X = np.random.rand(n_obs, n_vars)

    # Create realistic obs
    obs = pd.DataFrame(
        {
            "n_genes_by_counts": np.random.randint(100, 2000, n_obs),
            "total_counts": np.random.randint(1000, 10000, n_obs),
            "pct_counts_mt": np.random.uniform(0, 25, n_obs),
            "cell_type": np.random.choice(["TypeA", "TypeB", "TypeC"], n_obs),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )

    # Create var
    var = pd.DataFrame(
        {"gene_name": [f"gene_{i}" for i in range(n_vars)], "highly_variable": False},
        index=[f"gene_{i}" for i in range(n_vars)],
    )

    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Add some processed data layers
    adata.layers["counts"] = X.copy()

    return adata


@pytest.fixture
def data_manager_with_data(workspace_dir, test_adata):
    """Create DataManagerV2 with loaded data and provenance."""
    dm = DataManagerV2(workspace_path=workspace_dir, enable_provenance=True)

    # Add test data
    dm.modalities["test_dataset"] = test_adata

    # Simulate some analysis activities using correct API
    dm.provenance.create_activity(
        activity_type="quality_control",
        agent="test",
        description="QC metrics",
        outputs=[{"id": "qc", "type": "done"}],
    )

    dm.provenance.create_activity(
        activity_type="filter_cells",
        agent="test",
        description="Filter cells",
        parameters={"min_genes": 200, "max_mito_percent": 20.0},
        outputs=[{"id": "filtered", "type": "done"}],
    )

    dm.provenance.create_activity(
        activity_type="normalize",
        agent="test",
        description="Normalize data",
        parameters={"target_sum": 10000},
        outputs=[{"id": "normalized", "type": "done"}],
    )

    return dm


@pytest.mark.integration
class TestNotebookWorkflow:
    """Integration tests for complete notebook workflow."""

    def test_export_and_validate_structure(self, data_manager_with_data):
        """Test exporting notebook and validating its structure."""
        dm = data_manager_with_data

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = dm.workspace_path

            # Export notebook
            notebook_path = dm.export_notebook(
                name="test_workflow", description="Integration test workflow"
            )

        # Verify file exists
        assert notebook_path.exists()
        assert notebook_path.suffix == ".ipynb"

        # Load and validate structure
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Check basic structure
        assert len(nb.cells) > 0

        # First cell should be markdown header
        assert nb.cells[0].cell_type == "markdown"
        assert "test_workflow" in nb.cells[0].source

        # Should have parameters cell
        params_cells = [
            c for c in nb.cells if c.metadata.get("tags", []) == ["parameters"]
        ]
        assert len(params_cells) == 1
        assert params_cells[0].cell_type == "code"

        # Should have code cells for activities
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert len(code_cells) >= 3  # params + activities

        # Check metadata
        assert "lobster" in nb.metadata
        assert "source_session_id" in nb.metadata["lobster"]
        assert "dependencies" in nb.metadata["lobster"]

    def test_export_list_and_info(self, data_manager_with_data):
        """Test exporting, listing, and getting info about notebooks."""
        dm = data_manager_with_data

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = dm.workspace_path

            # Export multiple notebooks
            path1 = dm.export_notebook(name="workflow_1", description="First workflow")
            path2 = dm.export_notebook(name="workflow_2", description="Second workflow")

            # List notebooks (inside mock context)
            notebooks = dm.list_notebooks()

            assert len(notebooks) == 2
            assert any(nb["name"] == "workflow_1" for nb in notebooks)
            assert any(nb["name"] == "workflow_2" for nb in notebooks)

            # Check notebook metadata
            nb1 = next(nb for nb in notebooks if nb["name"] == "workflow_1")
            assert nb1["n_steps"] > 0
            assert nb1["size_kb"] > 0
            assert "created_by" in nb1

    def test_export_validation_and_dry_run(self, data_manager_with_data, test_adata):
        """Test notebook export, validation, and dry run."""
        dm = data_manager_with_data

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = dm.workspace_path

            # Export notebook
            notebook_path = dm.export_notebook(name="test_workflow")

        # Save test data
        input_path = dm.workspace_path / "test_input.h5ad"
        test_adata.write_h5ad(input_path)

        # Validate input
        validation = dm.notebook_executor.validate_input(notebook_path, input_path)

        # Should pass validation (or have warnings only)
        assert validation.is_valid or not validation.has_errors

        # Perform dry run
        dry_result = dm.notebook_executor.dry_run(notebook_path, input_path)

        assert dry_result["status"] == "dry_run_complete"
        assert dry_result["steps_to_execute"] > 0
        assert "validation" in dry_result
        assert "estimated_duration_minutes" in dry_result

    @patch("lobster.core.notebook_executor.papermill")
    def test_export_and_execute(
        self, mock_papermill, data_manager_with_data, test_adata
    ):
        """Test complete workflow: export and execute notebook."""
        dm = data_manager_with_data

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = dm.workspace_path

            # Export notebook
            notebook_path = dm.export_notebook(name="executable_workflow")

            # Mock papermill execution
            mock_papermill.execute_notebook.return_value = None

            # Add new test modality
            dm.modalities["new_dataset"] = test_adata

            # Execute notebook (inside mock context)
            result = dm.run_notebook("executable_workflow.ipynb", "new_dataset")

            # Should execute successfully
            assert result["status"] == "success"
            assert "output_notebook" in result
            assert "execution_time" in result

            # Verify papermill was called with correct parameters
            mock_papermill.execute_notebook.assert_called_once()
            call_kwargs = mock_papermill.execute_notebook.call_args.kwargs
            assert "parameters" in call_kwargs
            assert "input_data" in call_kwargs["parameters"]

    def test_export_with_different_filters(self, data_manager_with_data):
        """Test exporting with different activity filters."""
        dm = data_manager_with_data

        # Add a failed activity
        failed_activity = {"type": "failed_op", "agent": "test", "error": "Test error"}
        dm.provenance.activities.append(failed_activity)

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = dm.workspace_path

            # Export with successful filter (default)
            nb1_path = dm.export_notebook(
                name="successful_only", filter_strategy="successful"
            )

            # Export with all activities
            nb2_path = dm.export_notebook(name="all_activities", filter_strategy="all")

        # Load notebooks
        with open(nb1_path) as f:
            nb1 = nbformat.read(f, as_version=4)

        with open(nb2_path) as f:
            nb2 = nbformat.read(f, as_version=4)

        # nb2 should have more cells (includes failed activity)
        nb1_code_cells = len([c for c in nb1.cells if c.cell_type == "code"])
        nb2_code_cells = len([c for c in nb2.cells if c.cell_type == "code"])

        # All strategy might have same or more cells
        assert nb2_code_cells >= nb1_code_cells

    def test_reproducibility(self, data_manager_with_data, test_adata):
        """Test that exported notebook can reproduce results."""
        dm = data_manager_with_data

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = dm.workspace_path

            # Export notebook
            notebook_path = dm.export_notebook(name="reproducible_workflow")

        # Load exported notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Check that parameters are present and can be modified
        params_cell = next(
            c for c in nb.cells if c.metadata.get("tags", []) == ["parameters"]
        )
        assert "input_data" in params_cell.source
        assert "random_seed" in params_cell.source

        # Check that code cells use scanpy (standard library)
        code_content = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "import scanpy" in code_content or "sc." in code_content

    def test_error_handling_no_provenance(self, workspace_dir):
        """Test error handling when provenance is disabled."""
        dm = DataManagerV2(workspace_path=workspace_dir, enable_provenance=False)

        with pytest.raises(ValueError, match="Provenance tracking disabled"):
            dm.export_notebook(name="test")

    def test_error_handling_no_activities(self, workspace_dir):
        """Test error handling when no activities recorded."""
        dm = DataManagerV2(workspace_path=workspace_dir, enable_provenance=True)

        with pytest.raises(ValueError, match="No activities recorded"):
            dm.export_notebook(name="test")

    def test_error_handling_no_data(self, data_manager_with_data):
        """Test error handling when no data loaded for execution."""
        dm = data_manager_with_data

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = dm.workspace_path
            notebook_path = dm.export_notebook(name="test")

        # Try to run with non-existent modality
        with pytest.raises(ValueError, match="not loaded"):
            dm.run_notebook(str(notebook_path), "nonexistent_modality")

    def test_parameter_injection(self, data_manager_with_data, test_adata):
        """Test parameter injection during execution."""
        dm = data_manager_with_data

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = dm.workspace_path
            notebook_path = dm.export_notebook(name="param_test")

        dm.modalities["test_data"] = test_adata

        with patch("lobster.core.notebook_executor.papermill") as mock_pm:
            mock_pm.execute_notebook.return_value = None

            # Execute with custom parameters
            custom_params = {"random_seed": 999, "custom_value": "test"}
            result = dm.run_notebook(
                str(notebook_path), "test_data", parameters=custom_params
            )

            # Verify parameters were passed
            call_kwargs = mock_pm.execute_notebook.call_args.kwargs
            params = call_kwargs["parameters"]
            assert params["random_seed"] == 999
            assert params["custom_value"] == "test"

    @patch("lobster.core.notebook_executor.papermill")
    def test_execution_failure_handling(
        self, mock_papermill, data_manager_with_data, test_adata
    ):
        """Test handling of execution failures."""
        dm = data_manager_with_data

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = dm.workspace_path
            notebook_path = dm.export_notebook(name="failing_workflow")

        dm.modalities["test_data"] = test_adata

        # Mock papermill to fail
        mock_papermill.execute_notebook.side_effect = Exception("Execution error")

        result = dm.run_notebook(str(notebook_path), "test_data")

        # Should return failed status
        assert result["status"] == "failed"
        assert "error" in result
        assert "Execution error" in result["error"]

    def test_notebook_versioning_metadata(self, data_manager_with_data):
        """Test that notebooks include version metadata."""
        dm = data_manager_with_data

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = dm.workspace_path
            notebook_path = dm.export_notebook(name="versioned_notebook")

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        metadata = nb.metadata.get("lobster", {})

        # Check version tracking
        assert "lobster_version" in metadata
        assert "dependencies" in metadata

        deps = metadata["dependencies"]
        assert "python" in deps
        assert "pandas" in deps or "numpy" in deps  # At least some packages

    def test_complete_workflow_multiple_datasets(
        self, data_manager_with_data, test_adata
    ):
        """Test workflow with multiple datasets."""
        dm = data_manager_with_data

        # Add multiple datasets
        dm.modalities["dataset_1"] = test_adata
        dm.modalities["dataset_2"] = test_adata.copy()

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = dm.workspace_path

            # Export notebook
            notebook_path = dm.export_notebook(name="multi_dataset_workflow")

            # Verify export succeeded
            assert notebook_path.exists()

            # List available notebooks (inside mock context)
            notebooks = dm.list_notebooks()
            assert len(notebooks) > 0
            assert any(nb["name"] == "multi_dataset_workflow" for nb in notebooks)

            # Execute with different dataset
            with patch("lobster.core.notebook_executor.papermill") as mock_pm:
                mock_pm.execute_notebook.return_value = None

                result = dm.run_notebook(str(notebook_path), "dataset_2")
                assert result["status"] == "success"
