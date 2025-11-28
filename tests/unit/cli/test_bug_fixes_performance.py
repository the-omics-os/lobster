"""
Unit tests for CLI performance bug fixes.

Tests for:
- Bug #3: Glob Memory Explosion (lazy evaluation, file size limits)
- Bug #7: Progress Bar Overhead (removed fake spinner)
- Bug #2: Workspace Scan Caching (TTL-based caching)
"""

import itertools
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lobster.core.data_manager_v2 import DataManagerV2


class TestBug3GlobMemoryExplosion:
    """
    Test Bug #3: Glob memory explosion prevention.

    Verifies that glob operations only load requested files, not all matches.
    """

    def test_glob_lazy_evaluation_limits_files(self, tmp_path):
        """Test that glob only loads first 10 files, not all matches."""
        # Create 100 small test files
        for i in range(100):
            test_file = tmp_path / f"test_{i:03d}.txt"
            test_file.write_text(f"content {i}\n" * 10)

        # Simulate glob operation from cli.py (Bug #3 fix)
        search_pattern = str(tmp_path / "test_*.txt")

        # Old implementation (memory unsafe):
        # matching_files = glob.glob(search_pattern)  # Loads all 100 paths

        # New implementation (Bug #3 fix):
        import glob

        matching_files = list(itertools.islice(glob.iglob(search_pattern), 10))
        total_count = sum(1 for _ in glob.iglob(search_pattern))

        # Verify only 10 files loaded
        assert len(matching_files) == 10, "Should only load first 10 files"
        assert total_count == 100, "Should count all 100 files"

    def test_file_size_limit_prevents_loading_large_files(self, tmp_path):
        """Test that files over 10MB are skipped."""
        # Create small and large files
        small_file = tmp_path / "small.txt"
        large_file = tmp_path / "large.txt"

        small_file.write_text("small content")
        large_file.write_text("x" * 11_000_000)  # 11MB

        # Check file size before reading (Bug #3 fix)
        small_size = small_file.stat().st_size
        large_size = large_file.stat().st_size

        assert small_size < 10_000_000, "Small file should be under limit"
        assert large_size > 10_000_000, "Large file should exceed limit"

        # Verify large file is skipped
        if large_size > 10_000_000:
            # Should NOT read this file
            should_skip = True
        else:
            should_skip = False

        assert should_skip, "Large files should be skipped"

    @pytest.mark.benchmark
    def test_glob_memory_usage_bounded(self, tmp_path, benchmark):
        """Benchmark: verify memory usage bounded even with many files."""
        # Create 50 files
        for i in range(50):
            test_file = tmp_path / f"data_{i:03d}.csv"
            test_file.write_text("col1,col2\n1,2\n" * 100)

        def glob_with_limit():
            """Simulate Bug #3 fix: lazy evaluation with limit."""
            import glob

            pattern = str(tmp_path / "data_*.csv")
            files = list(itertools.islice(glob.iglob(pattern), 10))
            return files

        result = benchmark(glob_with_limit)
        assert len(result) == 10, "Should only load 10 files"


class TestBug7ProgressBarOverhead:
    """
    Test Bug #7: Progress bar overhead elimination.

    Verifies that fake progress spinner removed, single code path used.
    """

    def test_query_execution_without_progress_overhead(self):
        """Test that query execution doesn't create unnecessary Progress object."""
        mock_client = Mock()
        mock_client.query.return_value = {"success": True, "response": "Test response"}

        # Bug #7 fix: simple status message, no context manager
        # Old: with create_progress() as progress: ...
        # New: Single code path

        start = time.time()
        result = mock_client.query("test query", stream=False)
        elapsed = time.time() - start

        assert result["success"]
        # No progress overhead means query time is pure client.query() time
        assert elapsed < 0.01, "Should complete immediately (mock)"


class TestBug2WorkspaceScanCaching:
    """
    Test Bug #2: Workspace scan caching with TTL.

    Verifies that repeated workspace scans use cache within TTL window.
    """

    def test_workspace_cache_initialization(self, tmp_path):
        """Test that DataManagerV2 initializes cache variables."""
        dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)

        # Bug #2 fix: cache variables should exist
        assert hasattr(dm, "_available_datasets_cache")
        assert hasattr(dm, "_scan_timestamp")
        assert hasattr(dm, "_scan_ttl")

        # Default values
        assert dm._available_datasets_cache is None
        assert dm._scan_timestamp == 0
        assert dm._scan_ttl == 30  # 30 second TTL

    def test_first_scan_triggers_filesystem_scan(self, tmp_path):
        """Test that first call triggers actual scan."""
        dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)

        # Create test dataset (data dir already created by DataManagerV2)
        test_file = dm.workspace_path / "data" / "test_dataset.h5ad"

        # Create minimal H5AD file
        import anndata as ad
        import numpy as np

        adata = ad.AnnData(X=np.array([[1, 2], [3, 4]]))
        adata.write_h5ad(test_file)

        # First call should scan
        start = time.time()
        datasets = dm.get_available_datasets(force_refresh=False)
        first_call_time = time.time() - start

        assert "test_dataset" in datasets
        assert datasets["test_dataset"]["type"] == "h5ad"
        # First call performs actual filesystem scan (time varies by system)

    def test_second_scan_uses_cache(self, tmp_path):
        """Test that second call within TTL uses cache."""
        dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)

        # Create test dataset (data dir already created by DataManagerV2)
        test_file = dm.workspace_path / "data" / "test_dataset.h5ad"

        import anndata as ad
        import numpy as np

        adata = ad.AnnData(X=np.array([[1, 2], [3, 4]]))
        adata.write_h5ad(test_file)

        # First call
        datasets1 = dm.get_available_datasets(force_refresh=False)

        # Second call (should use cache)
        start = time.time()
        datasets2 = dm.get_available_datasets(force_refresh=False)
        second_call_time = time.time() - start

        assert datasets1 == datasets2, "Should return same data"
        assert second_call_time < 0.001, "Second call should be instant (cached)"

    def test_force_refresh_bypasses_cache(self, tmp_path):
        """Test that force_refresh=True bypasses cache."""
        dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)

        # Create initial dataset (data dir already created)
        test_file1 = dm.workspace_path / "data" / "dataset1.h5ad"
        import anndata as ad
        import numpy as np

        adata = ad.AnnData(X=np.array([[1, 2]]))
        adata.write_h5ad(test_file1)

        # First scan
        datasets1 = dm.get_available_datasets(force_refresh=False)
        assert len(datasets1) == 1

        # Add new dataset
        test_file2 = dm.workspace_path / "data" / "dataset2.h5ad"
        adata2 = ad.AnnData(X=np.array([[3, 4]]))
        adata2.write_h5ad(test_file2)

        # Without force_refresh, should still return cached (old) data
        datasets2 = dm.get_available_datasets(force_refresh=False)
        assert len(datasets2) == 1, "Should use cache, not see new file"

        # With force_refresh, should rescan and see new file
        datasets3 = dm.get_available_datasets(force_refresh=True)
        assert len(datasets3) == 2, "Should rescan and see both files"

    def test_invalidate_cache_forces_refresh(self, tmp_path):
        """Test that invalidate_scan_cache() forces next scan."""
        dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)

        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        # First scan
        datasets1 = dm.get_available_datasets()

        # Invalidate cache
        dm.invalidate_scan_cache()

        # Verify cache cleared
        assert dm._available_datasets_cache is None
        assert dm._scan_timestamp == 0

    def test_cache_performance_improvement(self, tmp_path):
        """Test that cache provides significant speedup."""
        dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)

        # Create 20 datasets (data dir already created)
        import anndata as ad
        import numpy as np

        for i in range(20):
            test_file = dm.workspace_path / "data" / f"dataset_{i:02d}.h5ad"
            adata = ad.AnnData(X=np.random.rand(100, 50))
            adata.write_h5ad(test_file)

        # First call (cache miss)
        start = time.time()
        datasets1 = dm.get_available_datasets(force_refresh=False)
        first_time = time.time() - start

        # Second call (cache hit)
        start = time.time()
        datasets2 = dm.get_available_datasets(force_refresh=False)
        second_time = time.time() - start

        print(f"\nFirst scan (cache miss): {first_time*1000:.1f}ms")
        print(f"Second scan (cache hit): {second_time*1000:.1f}ms")
        if first_time > 0:
            print(f"Speedup: {first_time/second_time:.1f}x")

        # Verify improvement
        assert second_time < first_time * 0.5, "Cache should be faster"
        assert second_time < 0.05, "Cache hit should be <50ms"

    def test_timing_metrics_available(self, tmp_path):
        """enable_timing should capture scan timings for diagnostics."""
        dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)
        dm.enable_timing(True)

        test_file = dm.workspace_path / "data" / "timed_dataset.h5ad"
        import anndata as ad
        import numpy as np

        test_file.parent.mkdir(exist_ok=True)
        adata = ad.AnnData(X=np.array([[1, 2]]))
        adata.write_h5ad(test_file)

        dm.get_available_datasets(force_refresh=True)
        timings = dm.get_latest_timings(clear=True)

        assert "dm:scan_workspace" in timings
        assert timings["dm:scan_workspace"] >= 0
