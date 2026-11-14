"""
Unit tests for NotebookExporter.

Tests notebook generation from provenance records.
"""

import ast
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import anndata
import nbformat
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.notebook_exporter import NotebookExporter
from lobster.core.notebooks.exporter import _guarded_save_code
from lobster.core.provenance import ProvenanceTracker
from lobster.core.provenance.analysis_ir import (
    AnalysisStep,
    create_data_loading_ir,
    create_data_saving_ir,
)


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
            total_activities=100,
            exportable_count=5,
            excluded_activity_types=["unrecorded_analysis"] * 95,
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

                # Second cell should be the export record (markdown)
                assert nb.cells[1].cell_type == "markdown"
                assert "Export record (hashes)" in nb.cells[1].source
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

    def test_export_record_does_not_overclaim(self, provenance_tracker, data_manager):
        """The export record must state its limits and claim no compliance or proof.

        The provenance hash covers only the session ID and counts, and nothing
        checks the hashes when the notebook runs, so the cell must not present
        them as tamper evidence, verification or regulatory compliance.
        """
        exporter = NotebookExporter(provenance_tracker, data_manager)
        source = exporter._create_integrity_manifest_cell().source
        lowered = source.lower()

        for phrase in (
            "alcoa",
            "cfr",
            "part 11",
            "gxp",
            "tamper-evident",
            "proof that",
            "invalidating",
            "integrity manifest",
        ):
            assert phrase not in lowered, phrase

        assert "Nothing in this notebook checks these hashes" in source
        assert "activity and entity counts only" in source
        assert "not tamper evidence and not proof of regulatory compliance" in source
        assert "does not show that running this notebook reproduces" in source
        # JSON keys stay stable so existing hash-checking scripts keep working.
        assert '"data_integrity_manifest"' in source
        assert '"provenance"' in source

    def test_export_record_hashes_the_saved_modality_files(
        self, provenance_tracker, tmp_path
    ):
        """Saved modalities live in data/; the record hashes them there."""
        import hashlib

        data = tmp_path / "data"
        data.mkdir()
        (data / "saved.h5ad").write_bytes(b"saved bytes")
        dm = SimpleNamespace(
            modalities={"saved": None, "unsaved": None},
            workspace_path=tmp_path,
            data_dir=data,
        )
        hashes = NotebookExporter(provenance_tracker, dm)._get_input_file_hashes()
        assert hashes == {
            "saved.h5ad": hashlib.sha256(b"saved bytes").hexdigest(),
            "unsaved.h5ad": "not_saved",
        }

    @staticmethod
    def _saved_modalities(tmp_path, *names):
        data = tmp_path / "data"
        data.mkdir(exist_ok=True)
        dm = SimpleNamespace(
            modalities={name: None for name in names},
            workspace_path=tmp_path,
            data_dir=data,
        )
        return data, dm

    def test_export_record_finds_a_modality_saved_only_as_autosave_or_raw(
        self, provenance_tracker, tmp_path
    ):
        """The data manager also saves ``<name>_autosave.h5ad`` and ``<name>_raw.h5ad``."""
        import hashlib

        data, dm = self._saved_modalities(
            tmp_path, "auto", "raw", "named_autosave", "none", "odd_autosave"
        )
        # Never double-suffixed: the data manager would not have written this one.
        (data / "odd_autosave_autosave.h5ad").write_bytes(b"odd")
        (data / "auto_autosave.h5ad").write_bytes(b"auto")
        (data / "raw_raw.h5ad").write_bytes(b"raw")
        (data / "named_autosave.h5ad").write_bytes(b"named")  # never double-suffixed
        exporter = NotebookExporter(provenance_tracker, dm)
        assert exporter._get_input_file_hashes() == {
            "auto.h5ad": hashlib.sha256(b"auto").hexdigest(),
            "raw.h5ad": hashlib.sha256(b"raw").hexdigest(),
            "named_autosave.h5ad": hashlib.sha256(b"named").hexdigest(),
            "none.h5ad": "not_saved",
            "odd_autosave.h5ad": "not_saved",
        }
        assert exporter._get_input_file_paths() == {
            "auto.h5ad": "data/auto_autosave.h5ad",
            "raw.h5ad": "data/raw_raw.h5ad",
            "named_autosave.h5ad": "data/named_autosave.h5ad",
        }

    def test_export_record_prefers_the_save_then_autosave_then_raw_then_root(
        self, provenance_tracker, tmp_path
    ):
        data, dm = self._saved_modalities(tmp_path, "x")
        files = {
            "data/x.h5ad": data / "x.h5ad",
            "data/x_autosave.h5ad": data / "x_autosave.h5ad",
            "data/x_raw.h5ad": data / "x_raw.h5ad",
            "x.h5ad": tmp_path / "x.h5ad",
        }
        for relative, path in files.items():
            path.write_bytes(relative.encode())
        exporter = NotebookExporter(provenance_tracker, dm)
        hashed = []
        for relative, path in files.items():
            hashed.append(exporter._get_input_file_paths()["x.h5ad"])
            path.unlink()  # the next in line is hashed once this one is gone
        assert hashed == list(files)
        assert exporter._get_input_file_hashes() == {"x.h5ad": "not_saved"}
        assert exporter._get_input_file_paths() == {}

    def test_export_record_does_not_hash_a_folder(self, provenance_tracker, tmp_path):
        data, dm = self._saved_modalities(tmp_path, "x", "y")
        (data / "x.h5ad").mkdir()  # a folder (a zarr-like store) is not a saved file
        (data / "y.h5ad").mkdir()
        (data / "y_autosave.h5ad").write_bytes(b"y")
        exporter = NotebookExporter(provenance_tracker, dm)
        hashes = exporter._get_input_file_hashes()
        assert hashes["x.h5ad"] == "not_saved"
        assert hashes["y.h5ad"] not in ("not_saved", "file_not_found", "hash_failed")
        assert exporter._get_input_file_paths() == {"y.h5ad": "data/y_autosave.h5ad"}

    def test_export_record_never_hashes_outside_the_workspace(
        self, provenance_tracker, tmp_path
    ):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data, dm = self._saved_modalities(workspace)
        (tmp_path / "outside.h5ad").write_bytes(b"outside")
        (workspace / "evil.h5ad").write_bytes(b"in the workspace folder")
        (data / "sub").mkdir()
        (data / "sub" / "x.h5ad").write_bytes(b"in a sub folder")
        if os.name == "posix":  # a file name there; a separator on Windows
            (data / "sub\\x.h5ad").write_bytes(b"a backslash in the name")
        names = ["../outside", "../evil", "sub/x", "sub\\x", str(tmp_path / "outside")]
        dm.modalities = {name: None for name in names}
        exporter = NotebookExporter(provenance_tracker, dm)
        assert exporter._get_input_file_hashes() == {
            f"{name}.h5ad": "not_saved" for name in names
        }
        assert exporter._get_input_file_paths() == {}

    @staticmethod
    def _saved_to(tracker, name, path):
        """What ``save_modality`` records: the path it wrote, with the modality name."""
        tracker.create_entity(
            entity_type="modality_data",
            uri=path,
            metadata={"modality_name": name, "shape": (3, 3)},
        )

    def test_export_record_finds_a_file_named_after_the_dataset(
        self, provenance_tracker, tmp_path
    ):
        """GEO saves ``geo_<gse>_sample_<gsm>`` as ``<gse>_<gsm>_raw.h5ad``."""
        import hashlib

        data, dm = self._saved_modalities(tmp_path, "geo_gse1_sample_gsm2", "none")
        (data / "gse1_gsm2_raw.h5ad").write_bytes(b"sample")
        self._saved_to(
            provenance_tracker, "geo_gse1_sample_gsm2", data / "gse1_gsm2_raw.h5ad"
        )
        exporter = NotebookExporter(provenance_tracker, dm)
        assert exporter._get_input_file_hashes() == {
            "geo_gse1_sample_gsm2.h5ad": hashlib.sha256(b"sample").hexdigest(),
            "none.h5ad": "not_saved",
        }
        assert exporter._get_input_file_paths() == {
            "geo_gse1_sample_gsm2.h5ad": "data/gse1_gsm2_raw.h5ad"
        }

    def test_export_record_takes_the_latest_save_before_the_name_chain(
        self, provenance_tracker, tmp_path
    ):
        data, dm = self._saved_modalities(tmp_path, "x")
        for name in ("x.h5ad", "first.h5ad", "last.h5ad"):
            (data / name).write_bytes(name.encode())
        self._saved_to(provenance_tracker, "x", data / "first.h5ad")
        self._saved_to(provenance_tracker, "x", data / "last.h5ad")
        # loading the modality again records an entity with no file
        provenance_tracker.create_entity(
            entity_type="modality_data", metadata={"modality_name": "x"}
        )
        exporter = NotebookExporter(provenance_tracker, dm)
        assert exporter._get_input_file_paths() == {"x.h5ad": "data/last.h5ad"}
        (data / "last.h5ad").unlink()  # gone: the names the data manager saves with
        assert exporter._get_input_file_paths() == {"x.h5ad": "data/x.h5ad"}

    def test_export_record_ignores_what_is_not_a_save_of_the_modality(
        self, provenance_tracker, tmp_path
    ):
        data, dm = self._saved_modalities(tmp_path, "x", "y")
        for name in ("a.h5ad", "b.h5ad", "c.h5ad", "d.h5ad"):
            (data / name).write_bytes(name.encode())
        # loading a modality records an entity with no file
        provenance_tracker.create_entity(
            entity_type="modality_data", metadata={"modality_name": "x"}
        )
        # another kind of entity, and another modality's save
        provenance_tracker.create_entity(
            entity_type="plot", uri=data / "a.h5ad", metadata={"modality_name": "x"}
        )
        self._saved_to(provenance_tracker, "y", data / "b.h5ad")
        provenance_tracker.create_entity(
            entity_type="modality_data", uri=data / "c.h5ad", metadata={}
        )
        provenance_tracker.entities["broken"] = "not an entity"
        exporter = NotebookExporter(provenance_tracker, dm)
        assert exporter._get_input_file_hashes()["x.h5ad"] == "not_saved"
        assert exporter._get_input_file_paths() == {"y.h5ad": "data/b.h5ad"}

    def test_export_record_never_takes_a_saved_path_outside_the_workspace(
        self, provenance_tracker, tmp_path
    ):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data, dm = self._saved_modalities(workspace, "x", "y", "z")
        outside = tmp_path / "outside.h5ad"
        outside.write_bytes(b"outside")
        self._saved_to(provenance_tracker, "x", outside)
        self._saved_to(provenance_tracker, "y", data / ".." / ".." / "outside.h5ad")
        if os.name == "posix":
            (data / "link.h5ad").symlink_to(outside)
            self._saved_to(provenance_tracker, "z", data / "link.h5ad")
        exporter = NotebookExporter(provenance_tracker, dm)
        assert set(exporter._get_input_file_hashes().values()) == {"file_not_found"}
        assert exporter._get_input_file_paths() == {}

    def test_export_record_takes_a_saved_path_for_a_name_with_a_separator(
        self, provenance_tracker, tmp_path
    ):
        """The name only keys the lookup: the path is the one the data manager wrote."""
        data, dm = self._saved_modalities(tmp_path, "sub/x")
        (data / "sub").mkdir()
        (data / "sub" / "x.h5ad").write_bytes(b"x")
        self._saved_to(provenance_tracker, "sub/x", data / "sub" / "x.h5ad")
        exporter = NotebookExporter(provenance_tracker, dm)
        assert exporter._get_input_file_paths() == {"sub/x.h5ad": "data/sub/x.h5ad"}

    @pytest.mark.skipif(os.name != "posix", reason="needs a symlink")
    def test_export_record_names_a_saved_file_as_the_workspace_is_reached(
        self, provenance_tracker, tmp_path
    ):
        """A path through another name for the workspace is not named relative to it."""
        real = tmp_path / "real"
        (real / "data").mkdir(parents=True)
        link = tmp_path / "link"
        link.symlink_to(real)
        data, dm = self._saved_modalities(link, "geo", "other")
        (real / "data" / "gse_raw.h5ad").write_bytes(b"geo")
        (real / "data" / "other.h5ad").write_bytes(b"other")
        self._saved_to(provenance_tracker, "geo", real / "data" / "gse_raw.h5ad")
        exporter = NotebookExporter(provenance_tracker, dm)
        # the one file is not named; the others are, and the record still has paths
        assert exporter._get_input_file_paths() == {"other.h5ad": "data/other.h5ad"}

    def test_export_record_finds_what_save_modality_wrote(self, tmp_path):
        """With the real data manager: the entity ``save_modality`` records is read."""
        dm = DataManagerV2(workspace_path=tmp_path)
        adata = anndata.AnnData(np.ones((3, 3), dtype=np.float32))
        dm.store_modality("geo_gse1_sample_gsm2", adata)
        dm.store_modality("unsaved", adata)
        dm.save_modality("geo_gse1_sample_gsm2", "gse1_gsm2_raw.h5ad")
        exporter = NotebookExporter(dm.provenance, dm)
        hashes = exporter._get_input_file_hashes()
        assert hashes["unsaved.h5ad"] == "not_saved"
        assert len(hashes["geo_gse1_sample_gsm2.h5ad"]) == 64
        assert exporter._get_input_file_paths() == {
            "geo_gse1_sample_gsm2.h5ad": "data/gse1_gsm2_raw.h5ad"
        }

    def test_export_record_names_the_files_it_hashed(
        self, provenance_tracker, tmp_path
    ):
        import json

        data, dm = self._saved_modalities(tmp_path, "auto", "none")
        (data / "auto_autosave.h5ad").write_bytes(b"auto")
        source = (
            NotebookExporter(provenance_tracker, dm)
            ._create_integrity_manifest_cell()
            .source
        )
        record = json.loads(source.split("```json\n")[1].split("\n```")[0])
        manifest = record["data_integrity_manifest"]
        assert manifest["input_file_paths"] == {"auto.h5ad": "data/auto_autosave.h5ad"}
        assert manifest["input_files"]["none.h5ad"] == "not_saved"
        assert "`input_file_paths`" in source
        assert "the file Lobster last saved it to" in source
        assert (
            "`not_saved`: the session never saved that modality; `file_not_found`: it saved it and the file is gone"
            in source
        )

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
        assert "output_prefix" in cell.source
        # Nothing reads random_seed, so it is not offered as a parameter.
        assert "random_seed" not in cell.source

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
        from lobster.core.provenance.analysis_ir import AnalysisStep, ParameterSpec

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

    def test_create_imports_cell_with_irs(self, data_manager):
        """Test imports cell creation with IRs."""
        from lobster.core.provenance.analysis_ir import AnalysisStep

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
        from lobster.core.provenance.analysis_ir import AnalysisStep, ParameterSpec

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
        cell = exporter._create_parameters_cell([ir], read_names={"target_sum"})

        assert cell.cell_type == "code"
        assert "parameters" in cell.metadata.get("tags", [])
        assert "input_data" in cell.source
        # Offered only when the notebook code reads the name.
        assert "target_sum = 10000" in cell.source
        assert "min_genes" not in cell.source

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

    def test_has_download_activity_detects_geo(self, provenance_tracker, data_manager):
        """Test _has_download_activity detects GEO downloads."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        geo_ir = AnalysisStep(
            operation="geo_service.download_dataset",
            tool_name="GEODownloadService.download_dataset",
            description="Load GEO dataset",
            library="anndata",
            code_template='adata = ad.read_h5ad("{{ raw_file }}")',
            imports=["import anndata as ad"],
            parameters={"dataset_id": "GSE12345", "raw_file": "data/test_raw.h5ad"},
            parameter_schema={},
        )
        pairs = [({"type": "download_dataset"}, geo_ir)]
        assert exporter._has_download_activity(pairs) is True

    def test_has_download_activity_detects_sra(self, provenance_tracker, data_manager):
        """Test _has_download_activity detects SRA downloads."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        sra_ir = AnalysisStep(
            operation="sra_download",
            tool_name="SRADownloadService.download_dataset",
            description="List FASTQ files",
            library="pathlib",
            code_template='fastq_dir = Path("{{ fastq_dir }}")',
            imports=["from pathlib import Path"],
            parameters={"dataset_id": "SRR123", "fastq_dir": "downloads/sra/SRR123"},
            parameter_schema={},
        )
        pairs = [({"type": "download_dataset"}, sra_ir)]
        assert exporter._has_download_activity(pairs) is True

    def test_has_download_activity_false_for_analysis(
        self, provenance_tracker, data_manager
    ):
        """Test _has_download_activity returns False for non-download workflows."""
        exporter = NotebookExporter(provenance_tracker, data_manager)
        analysis_ir = create_sample_ir(
            "scanpy.pp.normalize_total", "normalize", "Normalize"
        )
        pairs = [({"type": "normalize"}, analysis_ir)]
        assert exporter._has_download_activity(pairs) is False

    def test_export_skips_generic_load_for_download_workflow(self, data_manager):
        """Test that download workflows don't get a generic data loading cell."""
        tracker = ProvenanceTracker()

        # Add a GEO download IR activity
        geo_ir = AnalysisStep(
            operation="geo_service.download_dataset",
            tool_name="GEODownloadService.download_dataset",
            description="Load GEO dataset GSE12345",
            library="anndata",
            code_template='# Load GEO dataset GSE12345\nadata = ad.read_h5ad("{{ raw_file }}")\nprint(f"Loaded: {adata.shape}")',
            imports=["import anndata as ad"],
            parameters={"dataset_id": "GSE12345", "raw_file": "data/test_raw.h5ad"},
            parameter_schema={},
            exportable=True,
        )
        tracker.create_activity(
            activity_type="download_dataset",
            agent="data_expert",
            description="Download GEO dataset",
            ir=geo_ir,
        )

        # Add a QC IR activity
        qc_ir = create_sample_ir("scanpy.pp.calculate_qc_metrics", "qc", "Calculate QC")
        tracker.create_activity(
            activity_type="quality_control",
            agent="singlecell_expert",
            description="QC metrics",
            ir=qc_ir,
        )

        exporter = NotebookExporter(tracker, data_manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)
                path = exporter.export(name="download_test")

                with open(path) as f:
                    nb = nbformat.read(f, as_version=4)

                # The generic load cell contains "adata = ad.read_h5ad(input_data)"
                # It should NOT be present in download workflows
                code_sources = [c.source for c in nb.cells if c.cell_type == "code"]
                generic_load_present = any(
                    "ad.read_h5ad(input_data)" in s for s in code_sources
                )
                assert (
                    not generic_load_present
                ), "Generic data loading cell should be skipped in download workflows"

    def test_export_keeps_generic_load_for_local_workflow(
        self, provenance_tracker, data_manager
    ):
        """Test that local file workflows keep the generic data loading cell."""
        exporter = NotebookExporter(provenance_tracker, data_manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)
                path = exporter.export(name="local_test")

                with open(path) as f:
                    nb = nbformat.read(f, as_version=4)

                # Generic load cell SHOULD be present for local file workflows
                code_sources = [c.source for c in nb.cells if c.cell_type == "code"]
                generic_load_present = any(
                    "ad.read_h5ad(input_data)" in s for s in code_sources
                )
                assert (
                    generic_load_present
                ), "Generic data loading cell should be present in local file workflows"

    def test_helper_cell_injected_when_helpers_present(self, data_manager):
        """Test helper cell is injected when IRs have helper_code."""
        tracker = ProvenanceTracker()

        helper = "def annotate_qc_genes(adata):\n    pass"
        ir_with_helper = AnalysisStep(
            operation="scanpy.pp.calculate_qc_metrics",
            tool_name="qc",
            description="QC with helper",
            library="scanpy",
            code_template="annotate_qc_genes(adata)\nprint('done')",
            imports=["import scanpy as sc"],
            parameters={},
            parameter_schema={},
            exportable=True,
            helper_code=[helper],
        )
        tracker.create_activity(
            activity_type="quality_control",
            agent="singlecell_expert",
            description="QC",
            ir=ir_with_helper,
        )

        exporter = NotebookExporter(tracker, data_manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)
                path = exporter.export(name="helper_test")

                with open(path) as f:
                    nb = nbformat.read(f, as_version=4)

                # Find helper cell (should be between imports and parameters)
                code_cells = [c for c in nb.cells if c.cell_type == "code"]
                helper_cells = [c for c in code_cells if "Helper functions" in c.source]
                assert len(helper_cells) == 1
                assert "annotate_qc_genes" in helper_cells[0].source

    def test_no_helper_cell_when_helpers_empty(self, provenance_tracker, data_manager):
        """Test no helper cell when IRs have no helper_code."""
        exporter = NotebookExporter(provenance_tracker, data_manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)
                path = exporter.export(name="no_helper_test")

                with open(path) as f:
                    nb = nbformat.read(f, as_version=4)

                code_cells = [c for c in nb.cells if c.cell_type == "code"]
                helper_cells = [c for c in code_cells if "Helper functions" in c.source]
                assert len(helper_cells) == 0

    def test_helper_dedup_across_irs(self, data_manager):
        """Test helpers are deduplicated across multiple IRs."""
        tracker = ProvenanceTracker()

        helper = "def annotate_qc_genes(adata):\n    pass"

        # Two activities with same helper
        for i, op in enumerate(["qc", "filter"]):
            ir = AnalysisStep(
                operation=f"scanpy.pp.{op}",
                tool_name=op,
                description=f"Step {i}",
                library="scanpy",
                code_template=f"print('{op}')",
                imports=["import scanpy as sc"],
                parameters={},
                parameter_schema={},
                exportable=True,
                helper_code=[helper],
            )
            tracker.create_activity(
                activity_type=op,
                agent="singlecell_expert",
                description=f"Step {i}",
                ir=ir,
            )

        exporter = NotebookExporter(tracker, data_manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)
                path = exporter.export(name="dedup_test")

                with open(path) as f:
                    nb = nbformat.read(f, as_version=4)

                # Helper function should appear exactly once in the notebook
                all_sources = "\n".join(c.source for c in nb.cells)
                assert all_sources.count("def annotate_qc_genes") == 1

    def test_non_exportable_irs_filtered(self, data_manager):
        """Test that IRs with exportable=False are filtered from notebook export."""
        tracker = ProvenanceTracker()

        # Add a non-exportable IR (modality management style)
        non_exportable_ir = AnalysisStep(
            operation="modality_management.list_modalities",
            tool_name="ModalityManagementService.list_modalities",
            description="List modalities",
            library="lobster",
            code_template="service = ModalityManagementService(data_manager)",
            imports=[
                "from lobster.services.data_management.modality_management_service import ModalityManagementService"
            ],
            parameters={},
            parameter_schema={},
            exportable=False,
        )
        tracker.create_activity(
            activity_type="list_modalities",
            agent="data_expert",
            description="List modalities",
            ir=non_exportable_ir,
        )

        # Add an exportable IR
        exportable_ir = create_sample_ir(
            "scanpy.pp.normalize_total", "normalize", "Normalize"
        )
        tracker.create_activity(
            activity_type="normalize",
            agent="singlecell_expert",
            description="Normalize",
            ir=exportable_ir,
        )

        exporter = NotebookExporter(tracker, data_manager)
        activities = exporter._filter_activities("successful")
        pairs = exporter._get_exportable_activity_ir_pairs(activities)

        # Only the exportable IR should remain
        assert len(pairs) == 1
        assert pairs[0][1].operation == "scanpy.pp.normalize_total"


@pytest.fixture
def ledger_exporter(tmp_path):
    """Export synthetic activities without a dataset or real tool invocation."""
    tracker = ProvenanceTracker(namespace="export-characterization")
    data_manager = SimpleNamespace(workspace_path=tmp_path, modalities={})
    return NotebookExporter(tracker, data_manager)


@pytest.fixture
def notebook_with_only_save_ir(ledger_exporter):
    # Use the same IR factory as DataManagerV2.save_modality, without saving data.
    ledger_exporter.provenance.activities = [
        {"id": "analysis", "type": "unrecorded_analysis", "ir": None},
        {
            "id": "save",
            "type": "save_dataset",
            "ir": create_data_saving_ir().to_dict(),
        },
    ]
    path = ledger_exporter.export(name="save_only_ir")
    return nbformat.read(path, as_version=4)


class TestNotebookExportCharacterization:
    """Characterize export behavior and the limits reported in notebook text."""

    @pytest.mark.parametrize("load_ir", [False, True], ids=["fallback", "recorded_ir"])
    def test_input_load_resets_ml_stopgap_once_per_run(
        self, ledger_exporter, tmp_path, load_ir
    ):
        input_path = tmp_path / "input.h5ad"
        create_test_adata().write_h5ad(input_path)
        if load_ir:
            ledger_exporter.provenance.activities = [
                {"type": "load_dataset", "ir": create_data_loading_ir().to_dict()}
            ]
        source = ledger_exporter._create_data_loading_cell().source
        scope = {"input_data": str(input_path)}
        for _ in range(2):
            scope.update(
                lobster_modalities={"old": object()},
                lobster_registered={"old": 99},
                lobster_epoch=99,
                lobster_last_output=object(),
            )
            # Execute a loader generated from controlled local test fixtures.
            exec(source, scope)  # nosec B102
            assert scope["adata"].shape == (100, 50)
            assert scope["lobster_modalities"] == {}
            assert scope["lobster_registered"] == {}
            assert scope["lobster_epoch"] == 0
            assert "lobster_last_output" not in scope

    @pytest.mark.parametrize("strategy", ["successful", "replay"])
    @pytest.mark.parametrize(
        "root_operation",
        [
            "peptidomics.import_sequences",
            "geo_service.download_dataset",
            "sra_download",
            "pride.download.download_dataset",
            "metabolights.download.download_dataset",
            "fallback_load",
            "recorded_load",
            "later_peptide_import",
        ],
    )
    def test_each_export_path_has_one_initial_ml_reset(
        self, ledger_exporter, monkeypatch, strategy, root_operation
    ):
        """Inspect download IR without fetching data; no later import resets ML."""
        monkeypatch.setattr(NotebookExporter, "_get_system_info", lambda self: {})
        monkeypatch.setattr(NotebookExporter, "_snapshot_dependencies", lambda self: {})
        dedicated_load = root_operation in {
            "fallback_load",
            "recorded_load",
            "later_peptide_import",
        }
        first_ir = create_sample_ir(
            "initial_analysis" if dedicated_load else root_operation,
            "initial_step",
            "Initial step",
        )
        first_ir.code_template = "initial_step_marker = True"
        later_import = create_sample_ir(
            "peptidomics.import_sequences", "later_import", "Later import"
        )
        later_import.code_template = "later_import_marker = True"
        analysis = create_sample_ir("later_analysis", "analysis", "Later analysis")
        analysis.code_template = "analysis_marker = True"
        activities = []
        if root_operation == "recorded_load":
            activities.append({"type": "load_dataset", "ir": create_data_loading_ir()})
        activities.extend(
            {"type": ir.tool_name, "ir": ir}
            for ir in (first_ir, later_import, analysis)
        )
        ledger_exporter.provenance.activities = activities
        path = ledger_exporter.export("run_boundary", filter_strategy=strategy)
        notebook = nbformat.read(path, as_version=4)
        sources = [c.source for c in notebook.cells if c.cell_type == "code"]
        reset_indices = []
        for name in ("lobster_modalities", "lobster_registered", "lobster_epoch"):
            assignments = [
                i
                for i, source in enumerate(sources)
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            ]
            assert len(assignments) == 1, (name, assignments)
            reset_indices.append(assignments[0])
        assert len(set(reset_indices)) == 1  # all reset state in one boundary
        reset_index = reset_indices[0]
        assert (
            sum("globals().pop('lobster_last_output', None)" in s for s in sources) == 1
        )
        assert reset_index < sources.index(first_ir.render())
        assert sources.count(later_import.render()) == 1  # no appended reset
        assert sources.count(analysis.render()) == 1
        load_indices = [
            i
            for i, source in enumerate(sources)
            if "ad.read_h5ad(input_data)" in source
        ]
        if dedicated_load:
            assert load_indices == [reset_index]  # not unconditional + loader reset
        else:
            assert load_indices == []
            # The self-loading path's sole initializer resets stale bindings on
            # each run before the root/analytical IR, not in every ML/import cell.
            scope = {}
            for _ in range(2):
                scope.update(
                    lobster_modalities={"old": object()},
                    lobster_registered={"old": 99},
                    lobster_epoch=99,
                    lobster_last_output=object(),
                )
                # Execute the fixture-generated run initializer, not user code.
                exec(sources[reset_index], scope)  # nosec B102
                assert scope["lobster_modalities"] == {}
                assert scope["lobster_registered"] == {}
                assert scope["lobster_epoch"] == 0
                assert "lobster_last_output" not in scope

    @pytest.mark.parametrize("selected_saving_split", [False, True])
    def test_final_saver_suppresses_only_explicitly_handled_split_save(
        self, ledger_exporter, selected_saving_split
    ):
        split_ir = create_sample_ir("ml.split", "create_ml_splits", "Split")
        split_ir.parameters = {
            "modality_name": "data",
            "validation_size": 0.1,
            "save_result": selected_saving_split,
        }
        saving_ir = create_data_saving_ir(filename_suffix="data_validation")

        def guarded(ir):
            path_line = next(
                line
                for line in ir.render().splitlines()
                if line.startswith("output_path = ")
            )
            return _guarded_save_code(path_line, ir.parameters["compression"])

        ledger_exporter.provenance.activities = [
            {
                "type": "save_dataset",
                "parameters": {
                    "name": "data_validation",
                    "path": "/workspace/data/data_validation.h5ad",
                },
                "ir": saving_ir.to_dict(),
            }
        ]
        source = ledger_exporter._create_data_saving_cell([split_ir]).source
        if selected_saving_split:
            assert 'f"{output_prefix}_processed.h5ad"' in source
            assert 'f"{output_prefix}_data_validation.h5ad"' not in source
        else:
            assert source == guarded(saving_ir)
        # Backward compatible direct call or unrelated saved modality: no
        # generic saver string surgery, no change to arbitrary legacy save IR.
        assert ledger_exporter._create_data_saving_cell().source == guarded(saving_ir)
        ledger_exporter.provenance.activities[0]["parameters"][
            "path"
        ] = "alternate.h5ad"
        assert ledger_exporter._create_data_saving_cell([split_ir]).source == guarded(
            saving_ir
        )
        ledger_exporter.provenance.activities[0]["parameters"]["name"] = "other"
        assert ledger_exporter._create_data_saving_cell([split_ir]).source == guarded(
            saving_ir
        )

    def test_save_ir_alone_bypasses_zero_ir_guard(self, notebook_with_only_save_ir):
        notebook = notebook_with_only_save_ir

        assert notebook.metadata.lobster.ir_statistics == {
            "n_irs_extracted": 1,
            "n_activities": 2,
            "coverage_percent": 50.0,
        }
        assert any(
            "adata.write_h5ad" in cell.source
            for cell in notebook.cells
            if cell.cell_type == "code"
        )
        # Export succeeds even though there is no analysis step in the notebook.
        assert not any(cell.source.startswith("## Step ") for cell in notebook.cells)

    @pytest.mark.parametrize("activity_type", ["load_dataset", "save_dataset"])
    @pytest.mark.parametrize("serialized", [False, True], ids=["object", "dict"])
    def test_dedicated_io_renders_non_exportable_ir(
        self, ledger_exporter, activity_type, serialized
    ):
        hidden_ir = create_sample_ir("ledger.hidden_io", activity_type, "Hidden IO")
        hidden_ir.exportable = False
        hidden_ir.code_template = f'io_bypass_marker = "{activity_type}"'
        visible_ir = create_sample_ir(
            "ledger.visible_analysis", "visible_analysis", "Positive control"
        )
        ledger_exporter.provenance.activities = [
            {
                "id": "hidden-io",
                "type": activity_type,
                "ir": hidden_ir.to_dict() if serialized else hidden_ir,
            },
            {
                "id": "visible-analysis",
                "type": "visible_analysis",
                "ir": visible_ir.to_dict(),
            },
        ]

        path = ledger_exporter.export(name="non_exportable_io")
        notebook = nbformat.read(path, as_version=4)
        code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]

        # Only the analysis qualifies, but the dedicated IO cell still renders.
        assert notebook.metadata.lobster.ir_statistics["n_irs_extracted"] == 1
        assert notebook.metadata.lobster.ir_statistics["n_activities"] == 2
        assert code.count(visible_ir.render()) == 1
        if activity_type == "load_dataset":
            assert sum(source.startswith(hidden_ir.render()) for source in code) == 1
        else:
            assert code.count(hidden_ir.render()) == 1

    def test_comment_only_ir_exports_a_valid_cell_without_executable_statements(
        self, ledger_exporter
    ):
        ir = AnalysisStep(
            operation="ledger.suggest_de_formula",
            tool_name="suggest_de_formula",
            description="A suggested formula represented only by comments",
            library="python",
            code_template=(
                "# Formula: {{ formula }}\n"
                "# Main variable: {{ groupby }}\n"
                "# Covariates: {{ covariates }}"
            ),
            imports=[],
            parameters={
                "formula": "~ condition",
                "groupby": "condition",
                "covariates": [],
            },
            parameter_schema={},
        )
        ledger_exporter.provenance.activities = [
            {"id": "comment-only", "type": "suggest_de_formula", "ir": ir.to_dict()}
        ]

        path = ledger_exporter.export(name="comment_only_ir", validate_syntax=True)
        notebook = nbformat.read(path, as_version=4)
        matching_cells = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code" and cell.source == ir.render()
        ]

        assert ir.exportable and ir.validates_on_export
        assert notebook.metadata.lobster.ir_statistics["n_irs_extracted"] == 1
        assert len(matching_cells) == 1
        assert ast.parse(matching_cells[0].source).body == []

    def test_footer_reports_excluded_activities_without_claiming_completeness(
        self, notebook_with_only_save_ir
    ):
        notebook = notebook_with_only_save_ir
        headers = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "markdown" and "**IR Coverage:**" in cell.source
        ]
        footers = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "markdown" and "## Results Export" in cell.source
        ]

        assert len(headers) == len(footers) == 1
        assert "**IR Coverage:** 1/2 activities (50.0%)" in headers[0].source
        assert "This analysis is now complete." not in footers[0].source
        assert "Notebook export is complete." not in footers[0].source
        assert "unrecorded_analysis" in footers[0].source
        assert "⚠ No IR" not in footers[0].source
        assert "IR-enabled steps are fully reproducible." not in footers[0].source
        assert (
            "IR coverage alone does not show that running it gives the same results."
            in footers[0].source
        )

    def test_footer_scopes_complete_export_to_ir_coverage(self, ledger_exporter):
        ledger_exporter.provenance.activities = [
            {"type": "save_dataset", "ir": create_data_saving_ir().to_dict()}
        ]
        path = ledger_exporter.export(name="all_activities_have_ir")
        notebook = nbformat.read(path, as_version=4)
        footers = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "markdown" and "## Results Export" in cell.source
        ]

        assert len(footers) == 1
        assert (
            "Notebook export is complete. All selected activities have exportable IR."
            in footers[0].source
        )
        assert "This analysis is now complete." not in footers[0].source

    def test_excluded_activity_types_are_unique_and_keep_first_seen_order(
        self, ledger_exporter
    ):
        ledger_exporter.provenance.activities = [
            {"type": "second_step", "ir": None},
            {"type": "first_step", "ir": None},
            {"type": "second_step", "ir": None},
            {
                # A qualifying activity must not hide excluded ones of the same type.
                "type": "second_step",
                "ir": create_sample_ir(
                    "ledger.second_step", "second_step", "Qualifying activity"
                ).to_dict(),
            },
        ]
        path = ledger_exporter.export(name="repeated_excluded_types")
        notebook = nbformat.read(path, as_version=4)

        assert notebook.metadata.lobster.ir_statistics == {
            "n_irs_extracted": 1,
            "n_activities": 4,
            "coverage_percent": 25.0,
        }
        for marker in ("## Results Export", "## Provenance\n"):
            cells = [
                cell
                for cell in notebook.cells
                if cell.cell_type == "markdown" and marker in cell.source
            ]
            assert len(cells) == 1
            assert cells[0].source.count("`second_step`") == 1
            assert cells[0].source.count("`first_step`") == 1
            assert "`second_step`, `first_step`" in cells[0].source
            if marker == "## Provenance\n":
                assert "| **Provenance-Only Activities** | 3 |" in cells[0].source

    def test_summary_names_excluded_activities_without_attributing_a_cause(
        self, notebook_with_only_save_ir
    ):
        summaries = [
            cell
            for cell in notebook_with_only_save_ir.cells
            if cell.cell_type == "markdown" and "## Provenance\n" in cell.source
        ]

        assert len(summaries) == 1
        assert "orchestration activities" not in summaries[0].source
        assert "unrecorded_analysis" in summaries[0].source
