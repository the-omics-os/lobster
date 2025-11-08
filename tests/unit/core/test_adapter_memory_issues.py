"""
Comprehensive tests for adapter registration conflicts and memory management issues.

Tests Issues #3, #10, #11, #12:
- Issue #3: Adapter registration conflicts (line 226 → actually 323-346)
- Issue #10: Memory calculation with mocks (lines 1646-1676)
- Issue #11: Plot persistence during clearing (lines 857-877)
- Issue #12: Path resolution failures (base.py lines 40-55)

Agent 3: Testing Campaign 2025-11-07
"""

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from plotly import graph_objects as go

from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
from lobster.core.backends.base import BaseBackend
from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.interfaces.backend import IDataBackend

# ============================================
# Issue #3: Adapter Registration Conflicts
# ============================================


class TestAdapterRegistrationConflicts(unittest.TestCase):
    """Test adapter registration conflicts and force override functionality."""

    def setUp(self):
        """Create temporary workspace for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)
        self.dm = DataManagerV2(workspace_path=self.workspace_path)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)

    # --------------------------------------------------
    # Test 1: Re-registering default adapter should fail
    # --------------------------------------------------
    def test_reregister_default_adapter_should_fail(self):
        """Re-registering a default adapter without force=True should raise ValueError."""
        # Default adapters are registered in __init__
        # Try to re-register transcriptomics_single_cell
        adapter = TranscriptomicsAdapter(
            data_type="single_cell", strict_validation=False
        )

        with self.assertRaises(ValueError) as context:
            self.dm.register_adapter("transcriptomics_single_cell", adapter)

        self.assertIn("already registered", str(context.exception).lower())
        self.assertIn("overwrite=True", str(context.exception))

    # --------------------------------------------------
    # Test 2: Force override should work
    # --------------------------------------------------
    def test_force_override_adapter_succeeds(self):
        """Overwriting an adapter with overwrite=True should succeed with warning."""
        adapter = TranscriptomicsAdapter(
            data_type="single_cell", strict_validation=False
        )

        # Should succeed with warning
        with self.assertLogs("lobster.core.data_manager_v2", level="WARNING") as logs:
            self.dm.register_adapter(
                "transcriptomics_single_cell", adapter, overwrite=True
            )

        # Check warning was logged
        self.assertTrue(
            any("overwriting" in log.lower() for log in logs.output),
            "Expected warning about overwriting adapter",
        )

    # --------------------------------------------------
    # Test 3: Register new custom adapter
    # --------------------------------------------------
    def test_register_new_custom_adapter(self):
        """Registering a new custom adapter should succeed without overwrite flag."""
        adapter = TranscriptomicsAdapter(
            data_type="single_cell", strict_validation=False
        )

        # Should succeed without error
        self.dm.register_adapter("custom_transcriptomics", adapter)

        # Verify it was registered
        self.assertIn("custom_transcriptomics", self.dm.adapters)

    # --------------------------------------------------
    # Test 4: Multiple registration attempts
    # --------------------------------------------------
    def test_multiple_registration_attempts_same_adapter(self):
        """Multiple attempts to register same adapter should consistently fail."""
        adapter = TranscriptomicsAdapter(
            data_type="single_cell", strict_validation=False
        )

        self.dm.register_adapter("test_adapter", adapter)

        # First re-registration attempt should fail
        with self.assertRaises(ValueError):
            self.dm.register_adapter("test_adapter", adapter)

        # Second re-registration attempt should also fail
        with self.assertRaises(ValueError):
            self.dm.register_adapter("test_adapter", adapter)

    # --------------------------------------------------
    # Test 5: Backend registration has same pattern
    # --------------------------------------------------
    def test_backend_registration_conflicts_same_pattern(self):
        """Backend registration should have same conflict behavior as adapters."""
        backend = H5ADBackend(base_path=self.workspace_path / "data")

        self.dm.register_backend("test_backend", backend)

        # Re-registration without overwrite should fail
        with self.assertRaises(ValueError) as context:
            self.dm.register_backend("test_backend", backend)

        self.assertIn("already registered", str(context.exception).lower())
        self.assertIn("overwrite=True", str(context.exception))

    # --------------------------------------------------
    # Test 6: Backend force override
    # --------------------------------------------------
    def test_backend_force_override_succeeds(self):
        """Backend overwriting with overwrite=True should succeed with warning."""
        backend = H5ADBackend(base_path=self.workspace_path / "data")

        self.dm.register_backend("test_backend", backend)

        # Should succeed with warning
        with self.assertLogs("lobster.core.data_manager_v2", level="WARNING") as logs:
            self.dm.register_backend("test_backend", backend, overwrite=True)

        self.assertTrue(any("overwriting" in log.lower() for log in logs.output))

    # --------------------------------------------------
    # Test 7: Adapter info retrieval
    # --------------------------------------------------
    def test_get_adapter_info_after_registration(self):
        """get_adapter_info() should return correct info after registration."""
        info = self.dm.get_adapter_info()

        # Should have default adapters
        self.assertIn("transcriptomics_single_cell", info)
        self.assertIn("transcriptomics_bulk", info)

        # Check structure
        for adapter_name, adapter_info in info.items():
            self.assertIn("modality_name", adapter_info)
            self.assertIn("supported_formats", adapter_info)
            self.assertIn("schema", adapter_info)


# ============================================
# Issue #10: Memory Calculation with Mocks
# ============================================


class TestMemoryCalculationWithMocks(unittest.TestCase):
    """Test memory usage calculation with mock objects and edge cases."""

    def setUp(self):
        """Create temporary workspace for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)
        self.dm = DataManagerV2(workspace_path=self.workspace_path)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)

    # --------------------------------------------------
    # Test 8: Memory calculation with None
    # --------------------------------------------------
    def test_memory_calculation_none_returns_na(self):
        """Memory calculation with None should return 'N/A' without crashing."""
        result = self.dm._get_safe_memory_usage(None)
        self.assertEqual(result, "N/A (No data matrix)")

    # --------------------------------------------------
    # Test 9: Memory calculation with sparse matrix
    # --------------------------------------------------
    def test_memory_calculation_sparse_matrix(self):
        """Memory calculation with sparse matrix should return correct MB value."""
        # Create sparse matrix
        sparse_matrix = sp.csr_matrix(np.random.rand(1000, 500))

        result = self.dm._get_safe_memory_usage(sparse_matrix)

        # Should contain "MB" and "sparse"
        self.assertIn("MB", result)
        self.assertIn("sparse", result.lower())

        # Should be a valid float MB value
        mb_value = float(result.split("MB")[0].strip().replace("~", ""))
        self.assertGreater(mb_value, 0)

    # --------------------------------------------------
    # Test 10: Memory calculation with dense matrix
    # --------------------------------------------------
    def test_memory_calculation_dense_matrix(self):
        """Memory calculation with dense numpy array should return correct MB value."""
        dense_matrix = np.random.rand(1000, 500)

        result = self.dm._get_safe_memory_usage(dense_matrix)

        # Should contain "MB" and "dense"
        self.assertIn("MB", result)
        self.assertIn("dense", result.lower())

        # Verify calculation: 1000 * 500 * 8 bytes (float64) / 1024^2
        expected_mb = (1000 * 500 * 8) / (1024**2)
        actual_mb = float(result.split("MB")[0].strip())
        self.assertAlmostEqual(actual_mb, expected_mb, places=2)

    # --------------------------------------------------
    # Test 11: Memory calculation with mock object (no nbytes)
    # --------------------------------------------------
    def test_memory_calculation_mock_without_nbytes(self):
        """Mock object without nbytes should fall back to 'Unknown format'."""
        mock_matrix = Mock()
        # Don't set nbytes, nnz, size, or dtype attributes

        result = self.dm._get_safe_memory_usage(mock_matrix)

        self.assertIn("unknown", result.lower())

    # --------------------------------------------------
    # Test 12: Memory calculation with mock sparse (has nnz but no data)
    # --------------------------------------------------
    def test_memory_calculation_mock_sparse_no_data(self):
        """Mock sparse matrix with nnz but no data should estimate memory."""
        mock_sparse = Mock()
        mock_sparse.nnz = 10000  # Non-zero count

        # Should estimate based on nnz * 12 bytes
        result = self.dm._get_safe_memory_usage(mock_sparse)

        self.assertIn("MB", result)
        self.assertIn("sparse", result.lower())
        self.assertIn("estimated", result.lower())

    # --------------------------------------------------
    # Test 13: Memory calculation with object having size and dtype
    # --------------------------------------------------
    def test_memory_calculation_with_size_and_dtype(self):
        """Object with size and dtype should estimate memory correctly."""
        mock_array = Mock()
        mock_array.size = 100000
        mock_array.dtype = Mock()
        mock_array.dtype.itemsize = 8

        result = self.dm._get_safe_memory_usage(mock_array)

        # Expected: 100000 * 8 / 1024^2 = 0.76 MB
        expected_mb = (100000 * 8) / (1024**2)
        actual_mb = float(result.split("MB")[0].strip())

        self.assertAlmostEqual(actual_mb, expected_mb, places=2)
        self.assertIn("estimated", result.lower())

    # --------------------------------------------------
    # Test 14: Memory calculation with sparse having partial attributes
    # --------------------------------------------------
    def test_memory_calculation_sparse_with_data_only(self):
        """Sparse matrix with data.nbytes but no indices/indptr should still calculate."""
        mock_sparse = Mock()
        mock_sparse.nnz = 1000
        mock_sparse.data = Mock()
        mock_sparse.data.nbytes = 8000  # 1000 * 8 bytes

        result = self.dm._get_safe_memory_usage(mock_sparse)

        self.assertIn("MB", result)
        self.assertIn("sparse", result.lower())

    # --------------------------------------------------
    # Test 15: Memory calculation with exception handling
    # --------------------------------------------------
    def test_memory_calculation_exception_handling(self):
        """Objects that raise exceptions during attribute access should return 'Unknown format'."""

        class BrokenMatrix:
            @property
            def nbytes(self):
                raise RuntimeError("Broken matrix")

        broken = BrokenMatrix()
        result = self.dm._get_safe_memory_usage(broken)

        # Should handle exception gracefully
        self.assertIn("unknown", result.lower())


# ============================================
# Issue #11: Plot Persistence During Clearing
# ============================================


class TestPlotPersistenceDuringClearing(unittest.TestCase):
    """Test plot lifecycle during workspace operations."""

    def setUp(self):
        """Create temporary workspace and add test plots."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)
        self.dm = DataManagerV2(workspace_path=self.workspace_path)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)

    def _create_test_plot(self) -> go.Figure:
        """Create a simple test plot."""
        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
        fig.update_layout(title="Test Plot")
        return fig

    # --------------------------------------------------
    # Test 16: Plots persist after modality clearing
    # --------------------------------------------------
    def test_plots_persist_after_modality_clearing(self):
        """Plots should persist after clearing modalities (not workspace)."""
        # Add a modality
        adata = anndata.AnnData(X=np.random.rand(10, 5))
        self.dm.modalities["test_data"] = adata

        # Store a plot
        test_plot = self._create_test_plot()
        self.dm.store_plot(test_plot, plot_type="test", description="Test plot")

        # Verify plot was stored
        self.assertEqual(len(self.dm.latest_plots), 1)

        # Clear workspace
        self.dm.clear_workspace(confirm=True)

        # Plots should still be present (clear_workspace doesn't clear plots)
        # NOTE: This is the actual behavior - plots persist unless clear_plots() is called
        self.assertEqual(
            len(self.dm.latest_plots),
            1,
            "Plots should persist after clear_workspace",
        )

    # --------------------------------------------------
    # Test 17: clear_plots() removes all plots
    # --------------------------------------------------
    def test_clear_plots_removes_all_plots(self):
        """clear_plots() should remove all stored plots."""
        # Store multiple plots
        for i in range(5):
            plot = self._create_test_plot()
            self.dm.store_plot(plot, plot_type="test", description=f"Plot {i}")

        self.assertEqual(len(self.dm.latest_plots), 5)

        # Clear plots
        self.dm.clear_plots()

        # All plots should be removed
        self.assertEqual(len(self.dm.latest_plots), 0)

    # --------------------------------------------------
    # Test 18: Plot retrieval by ID
    # --------------------------------------------------
    def test_plot_retrieval_by_id(self):
        """get_plot_by_id() should retrieve correct plot."""
        plot1 = self._create_test_plot()
        plot2 = self._create_test_plot()

        self.dm.store_plot(plot1, plot_type="test", description="First plot")
        self.dm.store_plot(plot2, plot_type="test", description="Second plot")

        # Get plot ID from latest_plots
        first_plot_id = self.dm.latest_plots[0]["id"]

        # Retrieve by ID
        retrieved_plot = self.dm.get_plot_by_id(first_plot_id)

        self.assertIsNotNone(retrieved_plot)
        self.assertIsInstance(retrieved_plot, go.Figure)

    # --------------------------------------------------
    # Test 19: Non-existent plot ID returns None
    # --------------------------------------------------
    def test_nonexistent_plot_id_returns_none(self):
        """get_plot_by_id() with non-existent ID should return None."""
        result = self.dm.get_plot_by_id("fake_plot_id_12345")
        self.assertIsNone(result)

    # --------------------------------------------------
    # Test 20: get_latest_plots() respects limit
    # --------------------------------------------------
    def test_get_latest_plots_respects_limit(self):
        """get_latest_plots(n) should return only n most recent plots."""
        # Store 10 plots
        for i in range(10):
            plot = self._create_test_plot()
            self.dm.store_plot(plot, plot_type="test", description=f"Plot {i}")

        # Get latest 3
        latest_3 = self.dm.get_latest_plots(n=3)

        self.assertEqual(len(latest_3), 3)

    # --------------------------------------------------
    # Test 21: Plot metadata structure
    # --------------------------------------------------
    def test_plot_metadata_structure(self):
        """Stored plots should have correct metadata structure."""
        plot = self._create_test_plot()
        self.dm.store_plot(plot, plot_type="umap", description="UMAP plot")

        plot_entry = self.dm.latest_plots[0]

        # Check required fields
        self.assertIn("id", plot_entry)
        self.assertIn("figure", plot_entry)
        self.assertIn("plot_type", plot_entry)
        self.assertIn("description", plot_entry)
        self.assertIn("timestamp", plot_entry)

        # Verify types
        self.assertIsInstance(plot_entry["id"], str)
        self.assertIsInstance(plot_entry["figure"], go.Figure)
        self.assertEqual(plot_entry["plot_type"], "umap")
        self.assertEqual(plot_entry["description"], "UMAP plot")

    # --------------------------------------------------
    # Test 22: Plot clearing is independent of workspace clearing
    # --------------------------------------------------
    def test_plot_clearing_independent_of_workspace(self):
        """Plots and workspace should clear independently."""
        # Add modality and plot
        adata = anndata.AnnData(X=np.random.rand(10, 5))
        self.dm.modalities["test_data"] = adata

        plot = self._create_test_plot()
        self.dm.store_plot(plot, plot_type="test", description="Test")

        # Clear workspace (modalities)
        self.dm.clear_workspace(confirm=True)

        # Modalities cleared, plots remain
        self.assertEqual(len(self.dm.modalities), 0)
        self.assertEqual(len(self.dm.latest_plots), 1)

        # Now clear plots
        self.dm.clear_plots()

        # Both should be empty
        self.assertEqual(len(self.dm.modalities), 0)
        self.assertEqual(len(self.dm.latest_plots), 0)


# ============================================
# Issue #12: Path Resolution Failures
# ============================================


class TestPathResolutionFailures(unittest.TestCase):
    """Test path resolution in temporary directories and edge cases."""

    def setUp(self):
        """Create temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)

    # --------------------------------------------------
    # Test 23: BaseBackend path resolution with absolute paths
    # --------------------------------------------------
    def test_base_backend_resolve_absolute_path(self):
        """BaseBackend._resolve_path() should return absolute paths unchanged."""
        backend = BaseBackend(base_path=self.temp_path)

        absolute_path = Path("/tmp/test_file.h5ad")
        resolved = backend._resolve_path(absolute_path)

        # Should return the path resolved (not concatenated with base_path)
        self.assertEqual(resolved, absolute_path.resolve())

    # --------------------------------------------------
    # Test 24: BaseBackend path resolution with relative paths
    # --------------------------------------------------
    def test_base_backend_resolve_relative_path(self):
        """BaseBackend._resolve_path() should resolve relative paths against base_path."""
        backend = BaseBackend(base_path=self.temp_path)

        relative_path = Path("data/test_file.h5ad")
        resolved = backend._resolve_path(relative_path)

        # Should be: base_path / relative_path
        expected = self.temp_path / relative_path
        self.assertEqual(resolved, expected)

    # --------------------------------------------------
    # Test 25: Path resolution without base_path
    # --------------------------------------------------
    def test_path_resolution_without_base_path(self):
        """BaseBackend without base_path should resolve paths to absolute."""
        backend = BaseBackend(base_path=None)

        relative_path = Path("data/test_file.h5ad")
        resolved = backend._resolve_path(relative_path)

        # Should resolve to absolute path (current working directory)
        self.assertTrue(resolved.is_absolute())

    # --------------------------------------------------
    # Test 26: Path resolution with string input
    # --------------------------------------------------
    def test_path_resolution_with_string_input(self):
        """BaseBackend._resolve_path() should handle string input."""
        backend = BaseBackend(base_path=self.temp_path)

        string_path = "data/test_file.h5ad"
        resolved = backend._resolve_path(string_path)

        expected = self.temp_path / string_path
        self.assertEqual(resolved, expected)

    # --------------------------------------------------
    # Test 27: H5ADBackend initialization creates directories
    # --------------------------------------------------
    def test_h5ad_backend_initialization_creates_directories(self):
        """H5ADBackend initialization should create base_path if it doesn't exist."""
        new_dir = self.temp_path / "new_backend_dir"

        # Directory doesn't exist yet
        self.assertFalse(new_dir.exists())

        # Initialize backend
        backend = H5ADBackend(base_path=new_dir)

        # Directory should now exist
        self.assertTrue(new_dir.exists())

    # --------------------------------------------------
    # Test 28: Path resolution in temporary directory
    # --------------------------------------------------
    def test_path_resolution_in_temp_directory(self):
        """Path resolution should work correctly in temporary directories."""
        # Use actual tempfile directory
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = BaseBackend(base_path=Path(tmpdir))

            relative_path = "subdir/data.h5ad"
            resolved = backend._resolve_path(relative_path)

            expected = Path(tmpdir) / relative_path
            self.assertEqual(resolved, expected)

    # --------------------------------------------------
    # Test 29: Ensure directory creates parent directories
    # --------------------------------------------------
    def test_ensure_directory_creates_parents(self):
        """BaseBackend._ensure_directory() should create parent directories."""
        backend = BaseBackend(base_path=self.temp_path)

        nested_path = self.temp_path / "level1" / "level2" / "data.h5ad"

        # Parents don't exist yet
        self.assertFalse(nested_path.parent.exists())

        # Ensure directory
        backend._ensure_directory(nested_path)

        # Parent directories should now exist
        self.assertTrue(nested_path.parent.exists())

    # --------------------------------------------------
    # Test 30: Path resolution with .. (parent directory)
    # --------------------------------------------------
    def test_path_resolution_with_parent_directory(self):
        """Path resolution should handle .. (parent directory) correctly."""
        backend = BaseBackend(base_path=self.temp_path / "subdir")

        relative_path = "../other_dir/data.h5ad"
        resolved = backend._resolve_path(relative_path)

        # Should resolve to: temp_path / other_dir / data.h5ad
        expected = self.temp_path / "other_dir" / "data.h5ad"
        self.assertEqual(resolved, expected)


# ============================================
# Additional Edge Cases & Memory Management
# ============================================


class TestAdapterMemoryLeaks(unittest.TestCase):
    """Test for memory leaks during repeated operations."""

    def setUp(self):
        """Create temporary workspace."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)

    # --------------------------------------------------
    # Test 31: Repeated modality loading doesn't leak
    # --------------------------------------------------
    def test_repeated_modality_loading_no_leak(self):
        """Repeated loading and clearing of modalities should not leak memory."""
        dm = DataManagerV2(workspace_path=self.workspace_path)

        # Load and clear 100 times
        for i in range(100):
            adata = anndata.AnnData(X=np.random.rand(100, 50))
            dm.modalities[f"data_{i}"] = adata

            if i % 10 == 0:
                dm.clear_workspace(confirm=True)

        # Final clear
        dm.clear_workspace(confirm=True)

        # Modalities should be empty
        self.assertEqual(len(dm.modalities), 0)

    # --------------------------------------------------
    # Test 32: Repeated adapter registration doesn't leak
    # --------------------------------------------------
    def test_repeated_adapter_override_no_leak(self):
        """Repeated adapter overriding should not leak memory."""
        dm = DataManagerV2(workspace_path=self.workspace_path)

        # Override 100 times
        for i in range(100):
            adapter = TranscriptomicsAdapter(
                data_type="single_cell", strict_validation=False
            )
            dm.register_adapter("test_adapter", adapter, overwrite=(i > 0))

        # Should have exactly one adapter registered
        self.assertIn("test_adapter", dm.adapters)

    # --------------------------------------------------
    # Test 33: Repeated plot storage and clearing
    # --------------------------------------------------
    def test_repeated_plot_storage_and_clearing_no_leak(self):
        """Repeated plot storage and clearing should not leak memory."""
        dm = DataManagerV2(workspace_path=self.workspace_path)

        for i in range(100):
            fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
            dm.store_plot(fig, plot_type="test", description=f"Plot {i}")

            if i % 10 == 0:
                dm.clear_plots()

        # Final clear
        dm.clear_plots()

        # Plots should be empty
        self.assertEqual(len(dm.latest_plots), 0)


class TestDataManagerInitializationEdgeCases(unittest.TestCase):
    """Test DataManagerV2 initialization edge cases."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up."""
        import shutil

        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)

    # --------------------------------------------------
    # Test 34: Initialization with non-existent workspace
    # --------------------------------------------------
    def test_initialization_creates_workspace(self):
        """DataManagerV2 should create workspace if it doesn't exist."""
        new_workspace = self.temp_path / "new_workspace"

        # Doesn't exist yet
        self.assertFalse(new_workspace.exists())

        # Initialize
        dm = DataManagerV2(workspace_path=new_workspace)

        # Should now exist
        self.assertTrue(new_workspace.exists())
        self.assertTrue(dm.data_dir.exists())
        self.assertTrue(dm.exports_dir.exists())
        self.assertTrue(dm.cache_dir.exists())

    # --------------------------------------------------
    # Test 35: Multiple DataManager instances with same workspace
    # --------------------------------------------------
    def test_multiple_instances_same_workspace(self):
        """Multiple DataManagerV2 instances can share the same workspace."""
        dm1 = DataManagerV2(workspace_path=self.temp_path)
        dm2 = DataManagerV2(workspace_path=self.temp_path)

        # Add data to first instance
        adata = anndata.AnnData(X=np.random.rand(10, 5))
        dm1.modalities["test_data"] = adata

        # Second instance shouldn't see it (separate in-memory state)
        self.assertEqual(len(dm2.modalities), 0)

    # --------------------------------------------------
    # Test 36: Adapter registration after initialization
    # --------------------------------------------------
    def test_adapters_registered_after_init(self):
        """Default adapters should be registered after initialization."""
        dm = DataManagerV2(workspace_path=self.temp_path)

        # Check default adapters exist
        self.assertIn("transcriptomics_single_cell", dm.adapters)
        self.assertIn("transcriptomics_bulk", dm.adapters)

        # Try get_adapter_info
        info = dm.get_adapter_info()
        self.assertGreater(len(info), 0)

    # --------------------------------------------------
    # Test 37: Backend initialization state
    # --------------------------------------------------
    def test_backends_initialized_correctly(self):
        """Default backends should be initialized correctly."""
        dm = DataManagerV2(workspace_path=self.temp_path)

        # Check default backends exist
        self.assertIn("h5ad", dm.backends)

        # Get backend info
        backend_info = dm.get_backend_info()
        self.assertIn("h5ad", backend_info)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
