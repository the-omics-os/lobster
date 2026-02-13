"""
Unit tests for NotebookExporter.

Tests notebook generation from provenance records.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import anndata
import nbformat
import numpy as np
import pandas as pd
import pytest

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.notebook_exporter import NotebookExporter
from lobster.core.provenance import ProvenanceTracker


def create_sample_ir(operation: str, tool_name: str, description: str) -> AnalysisStep:
    """Create a sample exportable IR for testing."""
    return AnalysisStep(
        operation=operation,
        tool_name=tool_name,
        description=description,
        library="scanpy",
        code_template=f"# {operation}\nprint('Running {tool_name}')",
        imports=["import scanpy as sc"],
        parameters={},
        parameter_schema={},
        input_entities=["adata"],
        output_entities=["adata"],
        exportable=True,
    )


@pytest.fixture
def provenance_tracker():
    """Create provenance tracker with sample activities including exportable IR."""
    tracker = ProvenanceTracker()

    # Add sample activities with exportable IR for notebook export testing
    tracker.create_activity(
        activity_type="quality_control",
        agent="singlecell_expert",
        description="Calculate QC metrics",
        outputs=[{"id": "qc_metrics", "type": "calculated"}],
        ir=create_sample_ir(
            "scanpy.pp.calculate_qc_metrics",
            "quality_control",
            "Calculate QC metrics",
        ),
    )

    tracker.create_activity(
        activity_type="filter_cells",
        agent="singlecell_expert",
        description="Filter low-quality cells",
        parameters={"min_genes": 200, "min_cells": 3, "max_mito_percent": 20.0},
        outputs=[{"id": "filtered_adata", "type": "ready"}],
        ir=create_sample_ir(
            "scanpy.pp.filter_cells", "filter_cells", "Filter low-quality cells"
        ),
    )

    tracker.create_activity(
        activity_type="normalize",
        agent="singlecell_expert",
        description="Normalize counts",
        parameters={"target_sum": 1e4},
        outputs=[{"id": "normalized_adata", "type": "ready"}],
        ir=create_sample_ir(
            "scanpy.pp.normalize_total", "normalize", "Normalize counts"
        ),
    )

    return tracker


@pytest.fixture
def provenance_tracker_no_ir():
    """Create provenance tracker with activities WITHOUT IR (for testing filtering)."""
    tracker = ProvenanceTracker()

    # Add sample activities WITHOUT IR (orchestration-style activities)
    tracker.create_activity(
        activity_type="process_publication_entry",
        agent="research_agent",
        description="Process publication entry",
        parameters={"entry_id": "pub_123"},
        # No IR - these should be filtered out
    )

    tracker.create_activity(
        activity_type="download_dataset",
        agent="data_expert",
        description="Download dataset",
        parameters={"accession": "GSE12345"},
        # No IR - these should be filtered out
    )

    return tracker


@pytest.fixture
def data_manager():
    """Create mock data manager with sample modalities."""
    dm = Mock(spec=DataManagerV2)
    dm.modalities = {"test_data": create_test_adata()}
    workspace = Path(tempfile.mkdtemp())
    dm.workspace_dir = workspace
    dm.workspace_path = workspace  # For integrity manifest hashing
    return dm


def create_test_adata():
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


class TestNotebookExporter:
    """Test NotebookExporter class."""

    def test_init(self, provenance_tracker, data_manager):
        """Test NotebookExporter initialization."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        assert exporter.provenance == provenance_tracker
        assert exporter.data_manager == data_manager

    def test_export_basic(self, provenance_tracker, data_manager):
        """Test basic notebook export."""
        exporter = NotebookExporter(provenance_tracker, data_manager)

        # Export notebook
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)
                path = exporter.export(
                    name="test_notebook", description="Test workflow"
                )

                # Verify file created (inside context while tmpdir exists)
                assert path.exists()
                assert path.suffix == ".ipynb"
                assert path.name == "test_notebook.ipynb"

    def test_export_validates_name(self, provenance_tracker, data_manager):
        """Test export validates notebook name."""
        exporter = NotebookExporter(provenance_tracker, data_manager)

        with pytest.raises(ValueError, match="Notebook name cannot be empty"):
            exporter.export(name="")

        with pytest.raises(ValueError, match="Notebook name cannot be empty"):
            exporter.export(name="   ")

    def test_export_requires_activities(self, data_manager):
        """Test export requires recorded activities."""
        empty_tracker = ProvenanceTracker()
        exporter = NotebookExporter(empty_tracker, data_manager)

        with pytest.raises(ValueError, match="No activities recorded"):
            exporter.export(name="test")

    def test_export_requires_exportable_irs(
        self, provenance_tracker_no_ir, data_manager
    ):
        """Test export raises error when no exportable IRs found."""
        exporter = NotebookExporter(provenance_tracker_no_ir, data_manager)

        with pytest.raises(ValueError, match="No exportable IR objects found"):
            exporter.export(name="test")

    def test_get_exportable_activity_ir_pairs(self, provenance_tracker, data_manager):
        """Test extraction of exportable (activity, IR) pairs."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        activities = exporter._filter_activities("successful")
        pairs = exporter._get_exportable_activity_ir_pairs(activities)

        # All activities have exportable IR in this fixture
        assert len(pairs) == 3
        for activity, ir in pairs:
            assert ir.exportable is True
            assert ir.operation is not None

    def test_get_exportable_activity_ir_pairs_filters_no_ir(
        self, provenance_tracker_no_ir, data_manager
    ):
        """Test that activities without IR are filtered out."""
        exporter = NotebookExporter(provenance_tracker_no_ir, data_manager)
        activities = exporter._filter_activities("successful")
        pairs = exporter._get_exportable_activity_ir_pairs(activities)

        # No activities have IR in this fixture
        assert len(pairs) == 0

    def test_ir_to_code_cell(self, provenance_tracker, data_manager):
        """Test IR to code cell conversion."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        ir = create_sample_ir("test_op", "test_tool", "Test description")

        cell = exporter._ir_to_code_cell(ir, validate=True)

        assert cell.cell_type == "code"
        assert "test_op" in cell.source  # From code_template

    def test_create_provenance_summary_cell(self, provenance_tracker, data_manager):
        """Test provenance summary cell creation."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        cell = exporter._create_provenance_summary_cell(
            total_activities=100, exportable_count=5
        )

        assert cell.cell_type == "markdown"
        assert "Executable Steps" in cell.source
        assert "5" in cell.source  # Exportable count
        assert "95" in cell.source  # Filtered count (100 - 5)

    def test_export_notebook_structure(self, provenance_tracker, data_manager):
        """Test exported notebook has correct structure."""
        exporter = NotebookExporter(provenance_tracker, data_manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)
                path = exporter.export(name="test_notebook")

                # Load and verify notebook (inside context)
                with open(path) as f:
                    nb = nbformat.read(f, as_version=4)

                # Check cells exist
                assert len(nb.cells) > 0

                # First cell should be markdown header
                assert nb.cells[0].cell_type == "markdown"
                assert "test_notebook" in nb.cells[0].source

                # Second cell should be integrity manifest (markdown)
                assert nb.cells[1].cell_type == "markdown"
                assert "Data Integrity Manifest" in nb.cells[1].source
                assert "sha256" in nb.cells[1].source

                # Third cell should be imports (code)
                assert nb.cells[2].cell_type == "code"
                assert "import" in nb.cells[2].source

                # Should have activity cells
                code_cells = [c for c in nb.cells if c.cell_type == "code"]
                assert len(code_cells) >= 2  # Imports + at least 1 activity cell

    def test_filter_activities_successful(self, provenance_tracker, data_manager):
        """Test filtering successful activities."""
        # Add a failed activity by adding error marker
        failed_activity = {"type": "failed_op", "agent": "test", "error": "Test error"}
        provenance_tracker.activities.append(failed_activity)

        exporter = NotebookExporter(provenance_tracker, data_manager)
        filtered = exporter._filter_activities("successful")

        # Should exclude failed activity
        assert len(filtered) == 3  # Only successful activities
        assert all(not a.get("error") for a in filtered)

    def test_filter_activities_all(self, provenance_tracker, data_manager):
        """Test including all activities."""
        # Add a failed activity
        failed_activity = {"type": "failed_op", "agent": "test", "error": "Test error"}
        provenance_tracker.activities.append(failed_activity)

        exporter = NotebookExporter(provenance_tracker, data_manager)
        filtered = exporter._filter_activities("all")

        # Should include all activities
        assert len(filtered) == 4  # All activities

    def test_create_header_cell(self, provenance_tracker, data_manager):
        """Test header cell creation."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        n_irs = 2
        cell = exporter._create_header_cell("test_notebook", "Test description", n_irs)

        assert cell.cell_type == "markdown"
        assert "test_notebook" in cell.source
        assert "Test description" in cell.source
        assert "Generated from Lobster AI Session" in cell.source

    def test_create_parameters_cell(self, provenance_tracker, data_manager):
        """Test Papermill parameters cell."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        cell = exporter._create_parameters_cell([])  # Empty IRs list for basic test

        assert cell.cell_type == "code"
        assert "parameters" in cell.metadata["tags"]
        assert "input_data" in cell.source
        assert "random_seed" in cell.source

    def test_create_doc_cell(self, provenance_tracker, data_manager):
        """Test documentation cell creation."""
        activity = {
            "type": "quality_control",
            "timestamp": "2025-10-27T10:00:00",
            "agent": "singlecell_expert",
            "description": "Calculate QC metrics",
            "parameters": {"min_genes": 200},
        }

        exporter = NotebookExporter(provenance_tracker, data_manager)
        cell = exporter._create_doc_cell(activity, step_number=1)

        assert cell.cell_type == "markdown"
        assert "Step 1" in cell.source
        assert "quality_control" in cell.source
        assert "min_genes" in cell.source
        assert "200" in cell.source

    def test_extract_irs_with_ir(self, data_manager):
        """Test IR extraction from activities with IR."""
        from lobster.core.analysis_ir import AnalysisStep, ParameterSpec

        # Create tracker with IR-enriched activities
        tracker = ProvenanceTracker()

        # Create IR with correct signature
        ir = AnalysisStep(
            operation="scanpy.pp.calculate_qc_metrics",
            tool_name="calculate_qc_metrics",
            description="Calculate QC metrics",
            library="scanpy",
            code_template="sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], inplace=True)",
            imports=["import scanpy as sc"],
            parameters={"qc_vars": ["mt", "ribo"]},
            parameter_schema={
                "qc_vars": ParameterSpec(
                    param_type="list",
                    papermill_injectable=False,
                    default_value=["mt", "ribo"],
                    required=True,
                    description="QC variables",
                )
            },
        )

        # Add activity with IR
        tracker.create_activity(
            activity_type="quality_control",
            agent="singlecell_expert",
            description="Calculate QC metrics",
            ir=ir,
        )

        exporter = NotebookExporter(tracker, data_manager)
        irs = exporter._extract_irs(tracker.activities)

        assert len(irs) == 1
        assert irs[0].operation == "scanpy.pp.calculate_qc_metrics"
        assert irs[0].parameters == {"qc_vars": ["mt", "ribo"]}

    def test_extract_irs_without_ir(self, provenance_tracker_no_ir, data_manager):
        """Test IR extraction from activities without IR."""
        exporter = NotebookExporter(provenance_tracker_no_ir, data_manager)
        irs = exporter._extract_irs(provenance_tracker_no_ir.activities)

        # None of the fixture activities have IR
        assert len(irs) == 0

    def test_activity_to_code_with_ir(self, data_manager):
        """Test activity to code conversion with IR present."""
        from lobster.core.analysis_ir import AnalysisStep, ParameterSpec

        # Create IR with correct signature
        ir = AnalysisStep(
            operation="scanpy.pp.filter_cells",
            tool_name="filter_cells",
            description="Filter cells by gene count",
            library="scanpy",
            code_template="sc.pp.filter_cells(adata, min_genes={{min_genes}})",
            imports=["import scanpy as sc"],
            parameters={"min_genes": 200},
            parameter_schema={
                "min_genes": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=200,
                    required=False,
                    description="Minimum genes per cell",
                )
            },
        )

        # Create activity with IR
        activity = {
            "type": "filter_cells",
            "parameters": {"min_genes": 200},
            "ir": ir.to_dict(),
        }

        tracker = ProvenanceTracker()
        exporter = NotebookExporter(tracker, data_manager)
        cell = exporter._activity_to_code(activity, validate=False)

        assert cell is not None
        assert cell.cell_type == "code"
        assert "sc.pp.filter_cells" in cell.source
        assert "min_genes=200" in cell.source

    def test_activity_to_code_without_ir(self, provenance_tracker, data_manager):
        """Test activity to code conversion without IR (fallback)."""
        activity = {
            "type": "unknown_tool",
            "parameters": {"test_param": "value"},
        }

        exporter = NotebookExporter(provenance_tracker, data_manager)
        cell = exporter._activity_to_code(activity)

        assert cell is not None
        assert cell.cell_type == "code"
        assert "TODO" in cell.source  # Placeholder for missing IR
        assert "No IR available" in cell.source

    def test_create_imports_cell_with_irs(self, data_manager):
        """Test imports cell creation with IRs."""
        from lobster.core.analysis_ir import AnalysisStep

        # Create IRs with imports using correct signature
        ir1 = AnalysisStep(
            operation="scanpy.pp.filter_cells",
            tool_name="filter_cells",
            description="Filter cells",
            library="scanpy",
            code_template="sc.pp.filter_cells(adata, min_genes=200)",
            imports=["import scanpy as sc"],
            parameters={},
            parameter_schema={},
        )

        ir2 = AnalysisStep(
            operation="scanpy.pp.normalize_total",
            tool_name="normalize",
            description="Normalize expression",
            library="scanpy",
            code_template="sc.pp.normalize_total(adata)",
            imports=["import scanpy as sc", "import numpy as np"],
            parameters={},
            parameter_schema={},
        )

        tracker = ProvenanceTracker()
        exporter = NotebookExporter(tracker, data_manager)
        cell = exporter._create_imports_cell([ir1, ir2])

        assert cell.cell_type == "code"
        assert "import scanpy as sc" in cell.source
        assert "import numpy as np" in cell.source
        # Should be deduplicated
        assert cell.source.count("import scanpy as sc") == 1

    def test_create_imports_cell_without_irs(self, provenance_tracker, data_manager):
        """Test imports cell creation without IRs (fallback)."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        cell = exporter._create_imports_cell([])

        assert cell.cell_type == "code"
        # Should have default imports
        assert "import numpy as np" in cell.source
        assert "import pandas as pd" in cell.source
        assert "import scanpy as sc" in cell.source

    def test_create_parameters_cell_with_irs(self, data_manager):
        """Test Papermill parameters cell creation with IRs."""
        from lobster.core.analysis_ir import AnalysisStep, ParameterSpec

        # Create IR with parameters
        ir = AnalysisStep(
            operation="filter_and_normalize",
            tool_name="filter_and_normalize",
            description="Filter and normalize data",
            library="scanpy",
            code_template="# code here",
            imports=["import scanpy as sc"],
            parameters={"min_genes": 200, "target_sum": 10000},
            parameter_schema={
                "min_genes": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=200,
                    required=False,
                    description="Minimum genes",
                ),
                "target_sum": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=10000,
                    required=False,
                    description="Target sum",
                ),
            },
        )

        tracker = ProvenanceTracker()
        exporter = NotebookExporter(tracker, data_manager)
        cell = exporter._create_parameters_cell([ir])

        assert cell.cell_type == "code"
        assert "parameters" in cell.metadata.get("tags", [])
        assert "input_data" in cell.source
        assert "random_seed" in cell.source

    def test_create_metadata(self, provenance_tracker, data_manager):
        """Test notebook metadata creation with IR counts."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        n_irs = 2
        n_activities = len(provenance_tracker.activities)

        metadata = exporter._create_metadata(n_irs, n_activities)

        assert "source_session_id" in metadata
        assert "created_by" in metadata
        assert "created_at" in metadata
        assert "dependencies" in metadata
        assert "source_provenance_summary" in metadata

        summary = metadata["source_provenance_summary"]
        assert summary["n_activities"] == n_activities
        assert summary["n_entities"] == len(provenance_tracker.entities)

        # IR statistics are in separate top-level key
        assert "ir_statistics" in metadata
        ir_stats = metadata["ir_statistics"]
        assert ir_stats["n_irs_extracted"] == n_irs

    def test_snapshot_dependencies(self, provenance_tracker, data_manager):
        """Test dependency snapshot."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        deps = exporter._snapshot_dependencies()

        assert "python" in deps
        assert "pandas" in deps  # Should capture pandas version
        assert isinstance(deps, dict)

    def test_export_creates_directory(self, provenance_tracker, data_manager):
        """Test export creates .lobster/notebooks/ directory."""
        exporter = NotebookExporter(provenance_tracker, data_manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)
                path = exporter.export(name="test")

        # Verify directory structure
        notebooks_dir = path.parent
        assert notebooks_dir.name == "notebooks"
        # Notebooks dir is inside workspace_path
        assert notebooks_dir.parent == data_manager.workspace_path

    def test_export_with_description(self, provenance_tracker, data_manager):
        """Test export includes description in header."""
        exporter = NotebookExporter(provenance_tracker, data_manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)
                path = exporter.export(
                    name="test", description="Custom analysis workflow"
                )

                with open(path) as f:
                    nb = nbformat.read(f, as_version=4)

                # Check description appears in header
                assert "Custom analysis workflow" in nb.cells[0].source

    def test_export_multiple_notebooks(self, provenance_tracker, data_manager):
        """Test exporting multiple notebooks."""
        exporter = NotebookExporter(provenance_tracker, data_manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)

                path1 = exporter.export(name="notebook1")
                path2 = exporter.export(name="notebook2")

                assert path1.exists()
                assert path2.exists()
                assert path1 != path2
