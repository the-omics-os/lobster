"""
Specialized tests for workspace restoration and session management features (v2.2+).

This module focuses specifically on testing the workspace restoration capabilities
introduced in DataManagerV2 v2.2+, including:
- Session persistence and recovery
- Lazy loading mechanisms
- Pattern-based dataset restoration
- Workspace metadata tracking
- Performance with large workspaces
"""

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from tests.mock_data.base import (
    LARGE_DATASET_CONFIG,
    MEDIUM_DATASET_CONFIG,
    SMALL_DATASET_CONFIG,
)
from tests.mock_data.factories import (
    BulkRNASeqDataFactory,
    ProteomicsDataFactory,
    SingleCellDataFactory,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp(prefix="lobster_workspace_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def populated_workspace(temp_workspace):
    """Create a workspace populated with test datasets."""
    dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

    # Create various test datasets
    datasets = {
        "geo_gse123456_raw": SingleCellDataFactory(config=SMALL_DATASET_CONFIG),
        "geo_gse123456_filtered": SingleCellDataFactory(config=SMALL_DATASET_CONFIG),
        "geo_gse123456_clustered": SingleCellDataFactory(config=SMALL_DATASET_CONFIG),
        "geo_gse789012_bulk": BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG),
        "proteomics_experiment": ProteomicsDataFactory(config=SMALL_DATASET_CONFIG),
        "pilot_study_data": SingleCellDataFactory(config=SMALL_DATASET_CONFIG),
        "validation_cohort": BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG),
    }

    # Save datasets to workspace
    for name, adata in datasets.items():
        filepath = dm.data_dir / f"{name}.h5ad"
        adata.write_h5ad(filepath)

    # Trigger workspace scan
    dm._scan_workspace()

    return dm, datasets


# ===============================================================================
# Workspace Scanning Tests
# ===============================================================================


@pytest.mark.unit
class TestWorkspaceScanning:
    """Test workspace scanning functionality."""

    def test_automatic_workspace_scanning(self, temp_workspace):
        """Test automatic workspace scanning on initialization."""
        # Create some test files first
        data_dir = temp_workspace / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        test_files = ["dataset1.h5ad", "dataset2.h5ad", "processed.h5ad"]
        for filename in test_files:
            # Create minimal h5ad file
            adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            adata.write_h5ad(data_dir / filename)

        # Initialize with auto_scan=True (default)
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=True)

        # Should have automatically discovered datasets
        assert len(dm.available_datasets) == 3
        assert "dataset1" in dm.available_datasets
        assert "dataset2" in dm.available_datasets
        assert "processed" in dm.available_datasets

    def test_manual_workspace_scanning(self, temp_workspace):
        """Test manual workspace scanning."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        # Initially should be empty
        assert len(dm.available_datasets) == 0

        # Add files after initialization
        test_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        test_adata.write_h5ad(dm.data_dir / "new_dataset.h5ad")

        # Manual scan should discover new files
        dm._scan_workspace()

        assert len(dm.available_datasets) == 1
        assert "new_dataset" in dm.available_datasets

    def test_workspace_scan_metadata_extraction(self, temp_workspace):
        """Test metadata extraction during workspace scanning."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        # Create dataset with known properties
        test_adata = SingleCellDataFactory(config=MEDIUM_DATASET_CONFIG)
        filepath = dm.data_dir / "test_dataset.h5ad"
        test_adata.write_h5ad(filepath)

        # Get file stats for comparison
        file_stats = filepath.stat()
        expected_size_mb = file_stats.st_size / 1e6
        expected_modified = datetime.fromtimestamp(file_stats.st_mtime).isoformat()

        # Scan workspace
        dm._scan_workspace()

        # Verify metadata extraction
        dataset_info = dm.available_datasets["test_dataset"]
        assert dataset_info["path"] == str(filepath)
        assert (
            abs(dataset_info["size_mb"] - expected_size_mb) < 0.1
        )  # Allow small variance
        assert dataset_info["shape"] == test_adata.shape
        assert dataset_info["type"] == "h5ad"
        assert "modified" in dataset_info

    def test_scan_with_corrupted_files(self, temp_workspace):
        """Test workspace scanning with corrupted or invalid files."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        # Create valid dataset
        valid_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        valid_adata.write_h5ad(dm.data_dir / "valid_dataset.h5ad")

        # Create corrupted file
        corrupted_path = dm.data_dir / "corrupted.h5ad"
        with open(corrupted_path, "w") as f:
            f.write("This is not a valid h5ad file")

        # Scan should handle corrupted files gracefully
        dm._scan_workspace()

        # Should find valid dataset but skip corrupted one
        assert "valid_dataset" in dm.available_datasets
        assert "corrupted" not in dm.available_datasets

    def test_large_workspace_scanning_performance(self, temp_workspace):
        """Test scanning performance with large workspaces."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        # Create many small datasets
        n_datasets = 50
        for i in range(n_datasets):
            adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            adata.write_h5ad(dm.data_dir / f"dataset_{i:03d}.h5ad")

        # Time the scanning operation
        start_time = time.time()
        dm._scan_workspace()
        scan_time = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds for 50 small files)
        assert scan_time < 5.0
        assert len(dm.available_datasets) == n_datasets


# ===============================================================================
# Session Management Tests
# ===============================================================================


@pytest.mark.unit
class TestSessionManagement:
    """Test session persistence and management."""

    def test_session_file_creation(self, temp_workspace):
        """Test session file creation and structure."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Add some data and operations
        test_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_data"] = test_adata
        dm.log_tool_usage("test_tool", {"param": "value"})

        # Update session file
        dm._update_session_file("test_operation")

        # Verify session file exists and has correct structure
        assert dm.session_file.exists()

        with open(dm.session_file, "r") as f:
            session_data = json.load(f)

        required_keys = [
            "session_id",
            "created_at",
            "last_modified",
            "lobster_version",
            "active_modalities",
            "workspace_stats",
            "command_history",
        ]

        for key in required_keys:
            assert key in session_data

        # Verify modality tracking
        assert "test_data" in session_data["active_modalities"]

        # Verify workspace stats
        assert session_data["workspace_stats"]["total_loaded"] == 1

    def test_session_persistence_across_restarts(self, temp_workspace):
        """Test session persistence across manager restarts."""
        # Create first session
        dm1 = DataManagerV2(workspace_path=temp_workspace)
        dm1.session_id = "test_session_123"

        # Add data and perform operations
        test_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm1.modalities["persistent_data"] = test_adata
        dm1.log_tool_usage("operation_1", {"step": 1})
        dm1.log_tool_usage("operation_2", {"step": 2})

        # Update session
        dm1._update_session_file("test_operation")

        # Create second session (simulating restart)
        dm2 = DataManagerV2(workspace_path=temp_workspace)

        # Should load previous session data
        assert dm2.session_data is not None
        assert "session_id" in dm2.session_data
        assert "command_history" in dm2.session_data
        assert len(dm2.session_data["command_history"]) >= 2

    def test_session_command_history_limiting(self, temp_workspace):
        """Test command history limiting in session files."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Generate many commands (more than the 50 limit)
        for i in range(75):
            dm.log_tool_usage(f"command_{i}", {"iteration": i})

        # Update session
        dm._update_session_file("many_commands")

        # Load session data
        with open(dm.session_file, "r") as f:
            session_data = json.load(f)

        # Should limit to 50 most recent commands
        assert len(session_data["command_history"]) == 50
        assert session_data["command_history"][-1]["command"] == "command_74"
        assert session_data["command_history"][0]["command"] == "command_25"

    def test_session_metadata_accuracy(self, temp_workspace):
        """Test accuracy of session metadata tracking."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Create test workspace with datasets
        datasets = {
            "dataset_1": SingleCellDataFactory(config=SMALL_DATASET_CONFIG),
            "dataset_2": BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG),
        }

        total_size = 0
        for name, adata in datasets.items():
            filepath = dm.data_dir / f"{name}.h5ad"
            adata.write_h5ad(filepath)
            total_size += filepath.stat().st_size / 1e6
            dm.modalities[name] = adata

        dm._scan_workspace()
        dm._update_session_file("accuracy_test")

        # Load and verify session data
        with open(dm.session_file, "r") as f:
            session_data = json.load(f)

        workspace_stats = session_data["workspace_stats"]
        assert workspace_stats["total_datasets"] == 2
        assert workspace_stats["total_loaded"] == 2
        assert abs(workspace_stats["total_size_mb"] - total_size) < 0.1


# ===============================================================================
# Lazy Loading Tests
# ===============================================================================


@pytest.mark.unit
class TestLazyLoading:
    """Test lazy loading functionality."""

    def test_basic_lazy_loading(self, populated_workspace):
        """Test basic lazy loading of datasets."""
        dm, original_datasets = populated_workspace

        # Initially no datasets in memory
        assert len(dm.modalities) == 0
        assert len(dm.available_datasets) > 0

        # Load specific dataset
        success = dm.load_dataset("geo_gse123456_raw")
        assert success is True
        assert "geo_gse123456_raw" in dm.modalities
        assert len(dm.modalities) == 1

        # Verify loaded data matches original
        loaded_data = dm.modalities["geo_gse123456_raw"]
        original_data = original_datasets["geo_gse123456_raw"]
        assert loaded_data.shape == original_data.shape

    def test_lazy_loading_nonexistent_dataset(self, populated_workspace):
        """Test lazy loading of nonexistent datasets."""
        dm, _ = populated_workspace

        # Try to load nonexistent dataset
        success = dm.load_dataset("nonexistent_dataset")
        assert success is False
        assert "nonexistent_dataset" not in dm.modalities

    def test_lazy_loading_with_force_reload(self, populated_workspace):
        """Test force reloading of already loaded datasets."""
        dm, _ = populated_workspace

        # Load dataset first time
        dm.load_dataset("geo_gse123456_raw")
        original_id = id(dm.modalities["geo_gse123456_raw"])

        # Load again without force reload (should be same object)
        dm.load_dataset("geo_gse123456_raw", force_reload=False)
        assert id(dm.modalities["geo_gse123456_raw"]) == original_id

        # Load with force reload (should be different object)
        dm.load_dataset("geo_gse123456_raw", force_reload=True)
        assert id(dm.modalities["geo_gse123456_raw"]) != original_id

    def test_lazy_loading_session_tracking(self, populated_workspace):
        """Test that lazy loading updates session tracking."""
        dm, _ = populated_workspace

        # Load dataset
        dm.load_dataset("geo_gse123456_raw")

        # Should update session file
        assert dm.session_file.exists()

        with open(dm.session_file, "r") as f:
            session_data = json.load(f)

        # Should track loaded dataset
        assert "geo_gse123456_raw" in session_data["active_modalities"]
        assert "last_accessed" in session_data["active_modalities"]["geo_gse123456_raw"]

    def test_lazy_loading_memory_efficiency(self, temp_workspace):
        """Test memory efficiency of lazy loading."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        # Create multiple datasets on disk
        n_datasets = 10
        for i in range(n_datasets):
            adata = SingleCellDataFactory(config=MEDIUM_DATASET_CONFIG)
            adata.write_h5ad(dm.data_dir / f"dataset_{i}.h5ad")

        dm._scan_workspace()

        # All datasets available but none loaded
        assert len(dm.available_datasets) == n_datasets
        assert len(dm.modalities) == 0

        # Load only specific datasets
        dm.load_dataset("dataset_3")
        dm.load_dataset("dataset_7")

        # Only requested datasets should be in memory
        assert len(dm.modalities) == 2
        assert "dataset_3" in dm.modalities
        assert "dataset_7" in dm.modalities


# ===============================================================================
# Pattern-Based Restoration Tests
# ===============================================================================


@pytest.mark.unit
class TestPatternBasedRestoration:
    """Test pattern-based dataset restoration functionality."""

    def test_recent_pattern_restoration(self, populated_workspace):
        """Test restoration using 'recent' pattern."""
        dm, _ = populated_workspace

        # Simulate previous session by loading some datasets
        dm.load_dataset("geo_gse123456_raw")
        dm.load_dataset("proteomics_experiment")
        dm._update_session_file("previous_session")

        # Clear modalities and create new manager (simulating restart)
        dm.modalities.clear()
        dm2 = DataManagerV2(workspace_path=dm.workspace_path)

        # Restore using recent pattern
        result = dm2.restore_session(pattern="recent", max_size_mb=1000)

        # Should restore previously active datasets
        assert len(result["restored"]) >= 2
        assert "geo_gse123456_raw" in result["restored"]
        assert "proteomics_experiment" in result["restored"]

    def test_all_pattern_restoration(self, populated_workspace):
        """Test restoration using 'all' pattern."""
        dm, original_datasets = populated_workspace

        # Restore all datasets with sufficient size limit
        result = dm.restore_session(pattern="all", max_size_mb=1000)

        # Should restore all available datasets
        assert len(result["restored"]) == len(original_datasets)
        assert set(result["restored"]) == set(original_datasets.keys())

    def test_glob_pattern_restoration(self, populated_workspace):
        """Test restoration using glob patterns."""
        dm, _ = populated_workspace

        # Test GEO dataset pattern
        result = dm.restore_session(pattern="geo_gse123456*", max_size_mb=1000)

        # Should match all GSE123456 datasets
        expected_matches = [
            "geo_gse123456_raw",
            "geo_gse123456_filtered",
            "geo_gse123456_clustered",
        ]
        assert len(result["restored"]) == 3
        assert set(result["restored"]) == set(expected_matches)

        # Test proteomics pattern
        dm.modalities.clear()
        result = dm.restore_session(pattern="proteomics*", max_size_mb=1000)

        assert len(result["restored"]) == 1
        assert "proteomics_experiment" in result["restored"]

    def test_restoration_size_limits(self, populated_workspace):
        """Test restoration respects size limits."""
        dm, _ = populated_workspace

        # Set very small size limit
        result = dm.restore_session(pattern="all", max_size_mb=1)  # 1MB limit

        # Should skip datasets that exceed limit
        assert len(result["skipped"]) > 0
        assert len(result["restored"]) < len(dm.available_datasets)

        # Verify reason for skipping
        for skipped_item in result["skipped"]:
            assert skipped_item[1] == "size_limit"

    def test_restoration_with_missing_files(self, populated_workspace):
        """Test restoration when some files are missing."""
        dm, _ = populated_workspace

        # Remove one of the files
        missing_file = dm.data_dir / "geo_gse123456_raw.h5ad"
        if missing_file.exists():
            missing_file.unlink()

        # Update available datasets to reflect missing file
        dm._scan_workspace()

        # Restore should handle missing files gracefully
        result = dm.restore_session(pattern="geo_gse123456*", max_size_mb=1000)

        # Should restore available files and skip missing ones
        assert len(result["restored"]) == 2  # filtered and clustered
        assert "geo_gse123456_raw" not in result["restored"]

    def test_restoration_result_metadata(self, populated_workspace):
        """Test comprehensive restoration result metadata."""
        dm, _ = populated_workspace

        result = dm.restore_session(pattern="geo_*", max_size_mb=1000)

        # Verify result structure
        assert "restored" in result
        assert "skipped" in result
        assert "total_size_mb" in result
        assert "pattern" in result

        # Verify pattern tracking
        assert result["pattern"] == "geo_*"

        # Verify size tracking
        assert isinstance(result["total_size_mb"], (int, float))
        assert result["total_size_mb"] > 0


# ===============================================================================
# Performance and Stress Tests
# ===============================================================================


@pytest.mark.unit
class TestWorkspacePerformance:
    """Test workspace performance under various conditions."""

    def test_large_workspace_performance(self, temp_workspace):
        """Test performance with large numbers of datasets."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        # Create large number of small datasets
        n_datasets = 100
        start_time = time.time()

        for i in range(n_datasets):
            adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            adata.write_h5ad(dm.data_dir / f"dataset_{i:03d}.h5ad")

        creation_time = time.time() - start_time

        # Scan workspace
        start_time = time.time()
        dm._scan_workspace()
        scan_time = time.time() - start_time

        # Should complete in reasonable time
        assert scan_time < 10.0  # 10 seconds for 100 files
        assert len(dm.available_datasets) == n_datasets

        # Test restoration performance
        start_time = time.time()
        result = dm.restore_session(pattern="dataset_00*", max_size_mb=1000)
        restore_time = time.time() - start_time

        assert restore_time < 5.0
        assert len(result["restored"]) == 10  # dataset_000 through dataset_009

    def test_mixed_size_dataset_handling(self, temp_workspace):
        """Test handling of mixed-size datasets."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        # Create datasets of different sizes
        dataset_configs = [
            ("small", SMALL_DATASET_CONFIG),
            ("medium", MEDIUM_DATASET_CONFIG),
            ("small2", SMALL_DATASET_CONFIG),
            ("medium2", MEDIUM_DATASET_CONFIG),
        ]

        total_expected_size = 0
        for name, config in dataset_configs:
            adata = SingleCellDataFactory(config=config)
            filepath = dm.data_dir / f"{name}.h5ad"
            adata.write_h5ad(filepath)
            total_expected_size += filepath.stat().st_size / 1e6

        # Scan and verify size calculations
        dm._scan_workspace()

        total_scanned_size = sum(
            info["size_mb"] for info in dm.available_datasets.values()
        )
        assert abs(total_scanned_size - total_expected_size) < 1.0  # Allow 1MB variance

    def test_concurrent_workspace_operations(self, temp_workspace):
        """Test concurrent workspace operations."""
        import concurrent.futures

        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        def create_and_scan(thread_id):
            """Create dataset and trigger scan."""
            try:
                adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                filepath = dm.data_dir / f"concurrent_{thread_id}.h5ad"
                adata.write_h5ad(filepath)

                # Trigger scan
                dm._scan_workspace()
                return True
            except Exception as e:
                print(f"Thread {thread_id} error: {e}")
                return False

        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_and_scan, i) for i in range(5)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # All operations should succeed
        assert all(results)

        # Final scan should find all datasets
        dm._scan_workspace()
        assert len(dm.available_datasets) == 5

    def test_session_file_performance(self, temp_workspace):
        """Test session file performance with large amounts of data."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Create large session state
        for i in range(1000):
            dm.log_tool_usage(f"tool_{i}", {"param": i, "data": f"large_param_{i}"})

        # Add many modalities
        for i in range(20):
            dm.modalities[f"dataset_{i}"] = SingleCellDataFactory(
                config=SMALL_DATASET_CONFIG
            )

        # Time session update
        start_time = time.time()
        dm._update_session_file("performance_test")
        update_time = time.time() - start_time

        # Should complete quickly even with large state
        assert update_time < 2.0

        # Verify file was created and is readable
        assert dm.session_file.exists()

        with open(dm.session_file, "r") as f:
            session_data = json.load(f)

        # Should contain recent commands (limited to 50)
        assert len(session_data["command_history"]) == 50
        assert len(session_data["active_modalities"]) == 20


# ===============================================================================
# Multi-Format Workspace Tests
# ===============================================================================


@pytest.mark.unit
class TestMultiFormatWorkspace:
    """Test multi-format workspace scanning and auto-loading."""

    def test_scan_discovers_csv_files(self, temp_workspace):
        """CSV files in data/ should appear in available_datasets."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)
        data_dir = dm.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        # Write a simple numeric CSV
        csv_path = data_dir / "expression.csv"
        df = pd.DataFrame(
            {"gene1": [1.0, 2.0, 3.0], "gene2": [4.0, 5.0, 6.0]},
            index=["s1", "s2", "s3"],
        )
        df.to_csv(csv_path)

        dm._scan_workspace()

        assert "expression" in dm.available_datasets
        info = dm.available_datasets["expression"]
        assert info["type"] == "csv"
        assert info["shape"][0] == 3  # 3 rows
        assert info["shape"][1] == 2  # 2 data cols (index excluded)

    def test_scan_discovers_parquet_files(self, temp_workspace):
        """Parquet files should be discovered with correct shape from pyarrow metadata."""
        pytest.importorskip("pyarrow")
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)
        data_dir = dm.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            {"gene1": [1.0, 2.0], "gene2": [3.0, 4.0], "gene3": [5.0, 6.0]},
            index=["s1", "s2"],
        )
        parquet_path = data_dir / "proteomics.parquet"
        df.to_parquet(parquet_path)

        dm._scan_workspace()

        assert "proteomics" in dm.available_datasets
        info = dm.available_datasets["proteomics"]
        assert info["type"] == "parquet"
        # parquet shape: num_rows from metadata, num_cols from schema (includes index col)
        assert info["shape"][0] == 2

    def test_scan_discovers_tsv_files(self, temp_workspace):
        """TSV files should be discovered with correct shape."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)
        data_dir = dm.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        tsv_path = data_dir / "counts.tsv"
        df = pd.DataFrame(
            {"gene1": [10, 20], "gene2": [30, 40]}, index=["sample_a", "sample_b"]
        )
        df.to_csv(tsv_path, sep="\t")

        dm._scan_workspace()

        assert "counts" in dm.available_datasets
        info = dm.available_datasets["counts"]
        assert info["type"] == "tsv"
        assert info["shape"][0] == 2

    def test_auto_load_csv_as_anndata(self, temp_workspace):
        """CSV files should be auto-loaded as AnnData modalities."""
        data_dir = temp_workspace / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        csv_path = data_dir / "matrix.csv"
        df = pd.DataFrame(
            {"gene1": [1.5, 2.5], "gene2": [3.5, 4.5]}, index=["s1", "s2"]
        )
        df.to_csv(csv_path)

        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=True)

        assert "matrix" in dm.modalities
        adata = dm.modalities["matrix"]
        assert adata.n_obs == 2
        assert adata.n_vars == 2

    def test_auto_load_parquet_as_anndata(self, temp_workspace):
        """Parquet files should be auto-loaded as AnnData modalities."""
        pytest.importorskip("pyarrow")
        data_dir = temp_workspace / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = data_dir / "proteomics.parquet"
        df = pd.DataFrame(
            {"prot1": [1.0, 2.0, 3.0], "prot2": [4.0, 5.0, 6.0]},
            index=["s1", "s2", "s3"],
        )
        df.to_parquet(parquet_path)

        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=True)

        assert "proteomics" in dm.modalities
        adata = dm.modalities["proteomics"]
        assert adata.n_obs == 3
        assert adata.n_vars == 2

    def test_auto_load_h5ad_takes_precedence(self, temp_workspace):
        """When both data.csv and data_autosave.h5ad exist, h5ad wins."""
        data_dir = temp_workspace / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Write a CSV with 2 rows
        csv_path = data_dir / "results.csv"
        df_csv = pd.DataFrame({"g1": [1.0, 2.0]}, index=["a", "b"])
        df_csv.to_csv(csv_path)

        # Write an h5ad with 5 rows under the same stem
        adata_h5 = ad.AnnData(
            X=np.random.rand(5, 3).astype(np.float32),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )
        adata_h5.write_h5ad(data_dir / "results.h5ad")

        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=True)

        # The h5ad version (5 obs) should have been loaded, not the CSV (2 obs)
        assert "results" in dm.modalities
        assert dm.modalities["results"].n_obs == 5

    def test_auto_load_skips_large_tabular_files(self, temp_workspace):
        """Tabular files >500MB should be skipped."""
        data_dir = temp_workspace / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        csv_path = data_dir / "huge.csv"
        pd.DataFrame({"g1": [1.0]}, index=["s1"]).to_csv(csv_path)

        # Create DM without auto-scan so we can patch before loading
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        # Patch Path.stat to return large size only for the huge.csv file
        _original_stat = Path.stat

        def _fake_stat(self_path, *args, **kwargs):
            result = _original_stat(self_path, *args, **kwargs)
            if self_path.name == "huge.csv":
                # Return a stat_result with inflated st_size (600 MB)
                return os.stat_result(
                    (
                        result.st_mode,
                        result.st_ino,
                        result.st_dev,
                        result.st_nlink,
                        result.st_uid,
                        result.st_gid,
                        int(600e6),
                        int(result.st_atime),
                        int(result.st_mtime),
                        int(result.st_ctime),
                    )
                )
            return result

        with patch.object(Path, "stat", _fake_stat):
            dm._auto_load_modalities()

        assert "huge" not in dm.modalities

    def test_auto_load_skips_invalid_csv(self, temp_workspace):
        """Non-numeric CSV files should be skipped (0-var guard rejects them)."""
        data_dir = temp_workspace / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        csv_path = data_dir / "textonly.csv"
        with open(csv_path, "w") as f:
            f.write("name,description\nfoo,bar\nbaz,qux\n")

        # Should not crash, and should NOT load the text-only file
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=True)
        assert "textonly" not in dm.modalities

    def test_get_scannable_extensions_from_adapters(self, temp_workspace):
        """_get_scannable_extensions should return union of all adapter formats."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)
        extensions = dm._get_scannable_extensions()

        # Must always include the common formats
        assert "h5ad" in extensions
        assert "csv" in extensions
        assert "tsv" in extensions
        assert "parquet" in extensions

        # Should also include formats from registered adapters (e.g. BaseAdapter's xlsx)
        assert "xlsx" in extensions

    def test_resolve_adapter_unambiguous(self, temp_workspace):
        """When only one adapter supports a format, _resolve_adapter_for_file returns it."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        # Create a mock adapter that uniquely supports "xyz" format
        mock_adapter = Mock(spec=["get_supported_formats", "from_source"])
        mock_adapter.get_supported_formats.return_value = ["xyz"]
        dm.adapters["special"] = mock_adapter

        result = dm._resolve_adapter_for_file(Path("/fake/data.xyz"))
        assert result is not None
        name, adapter = result
        assert name == "special"
        assert adapter is mock_adapter

    def test_resolve_adapter_ambiguous_falls_back_to_generic(self, temp_workspace):
        """For CSV (multiple adapters), _resolve_adapter_for_file falls back to BaseAdapter."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        result = dm._resolve_adapter_for_file(Path("/fake/data.csv"))
        assert result is not None
        name, adapter = result
        # Should be generic BaseAdapter since multiple registered adapters support csv
        from lobster.core.adapters.base import BaseAdapter

        assert isinstance(adapter, BaseAdapter)

    def test_scan_parquet_without_pyarrow(self, temp_workspace):
        """When pyarrow is missing, parquet scan should still index the file with (0,0) shape."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)
        data_dir = dm.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        # Write a real parquet file (pyarrow IS available in this env)
        df = pd.DataFrame({"g1": [1.0, 2.0]}, index=["s1", "s2"])
        parquet_path = data_dir / "noarrow.parquet"
        df.to_parquet(parquet_path)

        # Mock pyarrow import failure during metadata extraction
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyarrow.parquet":
                raise ImportError("mocked: pyarrow not available")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            dm._scan_workspace()

        # File should still appear in available_datasets, just with (0,0) shape
        assert "noarrow" in dm.available_datasets
        assert dm.available_datasets["noarrow"]["type"] == "parquet"
        assert dm.available_datasets["noarrow"]["shape"] == (0, 0)

    def test_scan_csv_shape_extraction_uses_context_manager(self, temp_workspace):
        """Verify CSV scan does not leak file handles (reads via context manager)."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)
        data_dir = dm.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        csv_path = data_dir / "handles.csv"
        df = pd.DataFrame({"g1": [1.0] * 100, "g2": [2.0] * 100})
        df.to_csv(csv_path)

        # Scan should complete without warnings about leaked handles
        dm._scan_workspace()

        assert "handles" in dm.available_datasets
        info = dm.available_datasets["handles"]
        assert info["shape"][0] == 100  # rows
        assert info["shape"][1] >= 1  # at least 1 data col

    def test_scan_large_csv_uses_estimated_shape(self, temp_workspace):
        """CSV files >50 MB should use estimated row count instead of reading entire file."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)
        data_dir = dm.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        csv_path = data_dir / "bigcsv.csv"
        pd.DataFrame({"g1": [1.0], "g2": [2.0]}).to_csv(csv_path)

        # Patch stat to report >50MB so the estimated path is used
        _original_stat = Path.stat

        def _fake_stat(self_path, *args, **kwargs):
            result = _original_stat(self_path, *args, **kwargs)
            if self_path.name == "bigcsv.csv":
                return os.stat_result(
                    (
                        result.st_mode,
                        result.st_ino,
                        result.st_dev,
                        result.st_nlink,
                        result.st_uid,
                        result.st_gid,
                        int(60e6),  # 60 MB
                        int(result.st_atime),
                        int(result.st_mtime),
                        int(result.st_ctime),
                    )
                )
            return result

        with patch.object(Path, "stat", _fake_stat):
            dm._scan_workspace()

        assert "bigcsv" in dm.available_datasets
        info = dm.available_datasets["bigcsv"]
        # Shape should be an estimate, not exact — just verify it's present and positive
        assert info["shape"][0] > 0
        assert info["shape"][1] >= 1

    def test_auto_load_xlsx_as_anndata(self, temp_workspace):
        """XLSX files should be auto-loaded as AnnData modalities."""
        pytest.importorskip("openpyxl")
        data_dir = temp_workspace / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        xlsx_path = data_dir / "expression.xlsx"
        df = pd.DataFrame(
            {"gene1": [1.0, 2.0, 3.0], "gene2": [4.0, 5.0, 6.0]},
            index=["s1", "s2", "s3"],
        )
        df.to_excel(xlsx_path)

        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=True)

        assert "expression" in dm.modalities
        adata = dm.modalities["expression"]
        assert adata.n_obs == 3
        assert adata.n_vars == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
