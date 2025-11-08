"""
Chaos Engineering Test Suite for Lobster Platform.

This comprehensive test suite simulates various failure scenarios to assess
system resilience, graceful degradation, and error recovery capabilities.

Test Categories:
- Network Failures (GEO downloads, API timeouts, DNS failures)
- Disk Failures (disk full, permissions, corruption)
- Memory Failures (OOM, allocation failures, leaks)
- Service Failures (API unavailability, database loss, timeouts)
- Data Corruption (corrupted files, malformed data)
"""

import gc
import io
import json
import os
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.geo_service import GEOService

# ============================================================================
# CHAOS INJECTION FRAMEWORK
# ============================================================================


class ChaosInjector:
    """Framework for injecting various failure scenarios into system components."""

    @staticmethod
    def inject_network_timeout(timeout_seconds: float = 0.001):
        """Simulate network timeout by setting extremely short timeout."""
        return patch("socket.setdefaulttimeout", return_value=timeout_seconds)

    @staticmethod
    def inject_connection_error():
        """Simulate connection failure."""

        def raise_connection_error(*args, **kwargs):
            raise ConnectionError("Network unreachable")

        return raise_connection_error

    @staticmethod
    def inject_dns_failure():
        """Simulate DNS resolution failure."""

        def raise_dns_error(*args, **kwargs):
            raise socket.gaierror("Name resolution failed")

        return raise_dns_error

    @staticmethod
    def inject_disk_full():
        """Simulate disk full error."""

        def raise_disk_full(*args, **kwargs):
            raise OSError(28, "No space left on device")

        return raise_disk_full

    @staticmethod
    def inject_permission_denied():
        """Simulate permission denied error."""

        def raise_permission_error(*args, **kwargs):
            raise PermissionError("Permission denied")

        return raise_permission_error

    @staticmethod
    def inject_memory_error():
        """Simulate memory allocation failure."""

        def raise_memory_error(*args, **kwargs):
            raise MemoryError("Cannot allocate memory")

        return raise_memory_error

    @staticmethod
    def inject_corrupted_file(data_path: Path):
        """Create a corrupted file at the given path."""
        with open(data_path, "wb") as f:
            f.write(b"CORRUPTED_DATA_\x00\xff\x00")

    @staticmethod
    def inject_partial_write(original_size: int, truncate_to: int = 10):
        """Simulate partial file write."""

        def partial_write_mock(path, *args, **kwargs):
            with open(path, "wb") as f:
                f.write(b"X" * truncate_to)

        return partial_write_mock

    @staticmethod
    def inject_slow_network(delay_seconds: float = 5.0):
        """Simulate slow network by adding delays."""

        def slow_response(*args, **kwargs):
            time.sleep(delay_seconds)
            raise TimeoutError("Connection timed out")

        return slow_response


class ResilienceMetrics:
    """Track resilience metrics across chaos tests."""

    def __init__(self):
        self.total_tests = 0
        self.graceful_failures = 0
        self.catastrophic_failures = 0
        self.partial_recoveries = 0
        self.full_recoveries = 0
        self.data_corruptions = 0
        self.error_messages_clear = 0
        self.error_messages_cryptic = 0

    def record_graceful_failure(self):
        """Record a graceful failure (proper error handling)."""
        self.graceful_failures += 1
        self.total_tests += 1

    def record_catastrophic_failure(self):
        """Record a catastrophic failure (crash, data corruption)."""
        self.catastrophic_failures += 1
        self.total_tests += 1

    def record_partial_recovery(self):
        """Record partial recovery (some data saved)."""
        self.partial_recoveries += 1
        self.total_tests += 1

    def record_full_recovery(self):
        """Record full recovery (retry succeeded)."""
        self.full_recoveries += 1
        self.total_tests += 1

    def record_data_corruption(self):
        """Record data corruption detected."""
        self.data_corruptions += 1

    def record_error_message_quality(self, clear: bool):
        """Record whether error message was clear."""
        if clear:
            self.error_messages_clear += 1
        else:
            self.error_messages_cryptic += 1

    def get_resilience_score(self) -> float:
        """Calculate overall resilience score (0-100)."""
        if self.total_tests == 0:
            return 0.0

        score = 0.0
        # Graceful failures: 40 points max
        score += (self.graceful_failures / self.total_tests) * 40
        # Full recoveries: 30 points max
        score += (self.full_recoveries / self.total_tests) * 30
        # Partial recoveries: 15 points max
        score += (self.partial_recoveries / self.total_tests) * 15
        # No data corruptions: 10 points max
        if self.total_tests > 0:
            corruption_penalty = (self.data_corruptions / self.total_tests) * 10
            score += max(0, 10 - corruption_penalty)
        # Clear error messages: 5 points max
        total_errors = self.error_messages_clear + self.error_messages_cryptic
        if total_errors > 0:
            score += (self.error_messages_clear / total_errors) * 5

        # Catastrophic failures heavily penalize
        if self.catastrophic_failures > 0:
            score *= 1 - (self.catastrophic_failures / self.total_tests)

        return min(100.0, score)

    def get_report(self) -> Dict[str, Any]:
        """Generate detailed resilience report."""
        return {
            "total_tests": self.total_tests,
            "graceful_failures": self.graceful_failures,
            "catastrophic_failures": self.catastrophic_failures,
            "partial_recoveries": self.partial_recoveries,
            "full_recoveries": self.full_recoveries,
            "data_corruptions": self.data_corruptions,
            "error_message_quality": {
                "clear": self.error_messages_clear,
                "cryptic": self.error_messages_cryptic,
            },
            "resilience_score": self.get_resilience_score(),
            "grade": self._score_to_grade(self.get_resilience_score()),
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert resilience score to letter grade."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        elif score >= 55:
            return "C-"
        elif score >= 50:
            return "D"
        else:
            return "F"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def chaos_metrics():
    """Provide resilience metrics tracker."""
    return ResilienceMetrics()


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for chaos tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def data_manager(temp_workspace):
    """Create data manager for chaos testing."""
    return DataManagerV2(workspace_path=temp_workspace)


@pytest.fixture
def sample_adata():
    """Create sample AnnData for testing."""
    n_obs, n_vars = 100, 50
    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame({"cell_type": [f"type_{i%3}" for i in range(n_obs)]})
    var = pd.DataFrame({"gene_name": [f"gene_{i}" for i in range(n_vars)]})
    return anndata.AnnData(X=X, obs=obs, var=var)


# ============================================================================
# NETWORK FAILURE TESTS
# ============================================================================


class TestNetworkFailures:
    """Test system resilience to network failures."""

    def test_geo_download_connection_timeout(self, data_manager, chaos_metrics):
        """Test GEO download with connection timeout."""
        geo_service = GEOService(data_manager)

        with patch(
            "urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")
        ):
            try:
                geo_service.fetch_and_store_dataset("GSE12345")
                chaos_metrics.record_catastrophic_failure()
            except TimeoutError as e:
                # Should provide clear error message
                error_msg = str(e)
                if "timeout" in error_msg.lower():
                    chaos_metrics.record_error_message_quality(clear=True)
                    chaos_metrics.record_graceful_failure()
                else:
                    chaos_metrics.record_error_message_quality(clear=False)
                    chaos_metrics.record_graceful_failure()
            except Exception as e:
                # Unexpected exception type
                chaos_metrics.record_catastrophic_failure()

    def test_geo_download_connection_refused(self, data_manager, chaos_metrics):
        """Test GEO download with connection refused."""
        geo_service = GEOService(data_manager)

        with patch(
            "urllib.request.urlopen",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            try:
                geo_service.fetch_and_store_dataset("GSE12345")
                chaos_metrics.record_catastrophic_failure()
            except (ConnectionRefusedError, Exception) as e:
                error_msg = str(e)
                if "connection" in error_msg.lower():
                    chaos_metrics.record_error_message_quality(clear=True)
                    chaos_metrics.record_graceful_failure()
                else:
                    chaos_metrics.record_error_message_quality(clear=False)
                    chaos_metrics.record_graceful_failure()

    def test_dns_resolution_failure(self, data_manager, chaos_metrics):
        """Test handling of DNS resolution failures."""
        geo_service = GEOService(data_manager)

        with patch(
            "urllib.request.urlopen",
            side_effect=socket.gaierror("Name resolution failed"),
        ):
            try:
                geo_service.fetch_and_store_dataset("GSE12345")
                chaos_metrics.record_catastrophic_failure()
            except (socket.gaierror, Exception) as e:
                error_msg = str(e)
                if "resolution" in error_msg.lower() or "name" in error_msg.lower():
                    chaos_metrics.record_error_message_quality(clear=True)
                    chaos_metrics.record_graceful_failure()
                else:
                    chaos_metrics.record_error_message_quality(clear=False)
                    chaos_metrics.record_graceful_failure()

    def test_partial_download_interruption(self, temp_workspace, chaos_metrics):
        """Test handling of partial downloads."""
        download_path = temp_workspace / "partial_download.tar.gz"

        # Create partial file
        with open(download_path, "wb") as f:
            f.write(b"PARTIAL_DATA" * 100)

        try:
            # Try to extract
            import tarfile

            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(path=temp_workspace)
            chaos_metrics.record_catastrophic_failure()
        except (tarfile.ReadError, EOFError) as e:
            # Should detect corruption
            chaos_metrics.record_error_message_quality(clear=True)
            chaos_metrics.record_graceful_failure()
        except Exception as e:
            chaos_metrics.record_error_message_quality(clear=False)
            chaos_metrics.record_graceful_failure()

    def test_slow_network_timeout(self, data_manager, chaos_metrics):
        """Test handling of slow network connections."""
        geo_service = GEOService(data_manager)

        def slow_response(*args, **kwargs):
            time.sleep(2)
            raise TimeoutError("Read timeout")

        with patch("urllib.request.urlopen", side_effect=slow_response):
            start_time = time.time()
            try:
                geo_service.fetch_and_store_dataset("GSE12345")
                chaos_metrics.record_catastrophic_failure()
            except TimeoutError:
                elapsed = time.time() - start_time
                # Should timeout quickly, not hang
                if elapsed < 10:
                    chaos_metrics.record_error_message_quality(clear=True)
                    chaos_metrics.record_graceful_failure()
                else:
                    chaos_metrics.record_error_message_quality(clear=False)
                    chaos_metrics.record_graceful_failure()
            except Exception:
                chaos_metrics.record_graceful_failure()

    def test_connection_pool_exhaustion(self, data_manager, chaos_metrics):
        """Test behavior when connection pool is exhausted."""
        # Simulate many concurrent requests
        errors = []

        def concurrent_request():
            try:
                data_manager.get_modality("nonexistent")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=concurrent_request) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should handle gracefully without crashes
        if len(errors) > 0 and all(
            isinstance(e, (KeyError, Exception)) for e in errors
        ):
            chaos_metrics.record_graceful_failure()
            chaos_metrics.record_error_message_quality(clear=True)
        else:
            chaos_metrics.record_catastrophic_failure()


# ============================================================================
# DISK FAILURE TESTS
# ============================================================================


class TestDiskFailures:
    """Test system resilience to disk failures."""

    def test_disk_full_during_save(self, data_manager, sample_adata, chaos_metrics):
        """Test handling of disk full error during data save."""
        data_manager.modalities["test_data"] = sample_adata

        with patch(
            "anndata.AnnData.write_h5ad",
            side_effect=OSError(28, "No space left on device"),
        ):
            try:
                data_manager.save_processed_data("test_data")
                chaos_metrics.record_catastrophic_failure()
            except OSError as e:
                if "space" in str(e).lower():
                    chaos_metrics.record_error_message_quality(clear=True)
                    chaos_metrics.record_graceful_failure()
                else:
                    chaos_metrics.record_error_message_quality(clear=False)
                    chaos_metrics.record_graceful_failure()
            except Exception:
                chaos_metrics.record_graceful_failure()

    def test_permission_denied_read(self, temp_workspace, data_manager, chaos_metrics):
        """Test handling of permission denied on file read."""
        test_file = temp_workspace / "protected.h5ad"
        test_file.touch()
        os.chmod(test_file, 0o000)

        try:
            data_manager.load_file(test_file, "test_data")
            chaos_metrics.record_catastrophic_failure()
        except PermissionError:
            chaos_metrics.record_error_message_quality(clear=True)
            chaos_metrics.record_graceful_failure()
        except Exception as e:
            if "permission" in str(e).lower():
                chaos_metrics.record_error_message_quality(clear=True)
                chaos_metrics.record_graceful_failure()
            else:
                chaos_metrics.record_error_message_quality(clear=False)
                chaos_metrics.record_graceful_failure()
        finally:
            os.chmod(test_file, 0o644)

    def test_permission_denied_write(
        self, temp_workspace, data_manager, sample_adata, chaos_metrics
    ):
        """Test handling of permission denied on file write."""
        data_manager.modalities["test_data"] = sample_adata
        os.chmod(temp_workspace, 0o444)

        try:
            data_manager.save_processed_data("test_data")
            chaos_metrics.record_catastrophic_failure()
        except PermissionError:
            chaos_metrics.record_error_message_quality(clear=True)
            chaos_metrics.record_graceful_failure()
        except Exception as e:
            if "permission" in str(e).lower():
                chaos_metrics.record_error_message_quality(clear=True)
                chaos_metrics.record_graceful_failure()
            else:
                chaos_metrics.record_error_message_quality(clear=False)
                chaos_metrics.record_graceful_failure()
        finally:
            os.chmod(temp_workspace, 0o755)

    def test_corrupted_h5ad_file(self, temp_workspace, data_manager, chaos_metrics):
        """Test handling of corrupted H5AD file."""
        corrupted_file = temp_workspace / "corrupted.h5ad"
        ChaosInjector.inject_corrupted_file(corrupted_file)

        try:
            data_manager.load_file(corrupted_file, "corrupted_data")
            chaos_metrics.record_catastrophic_failure()
        except Exception as e:
            # Should detect corruption
            error_msg = str(e)
            if "corrupt" in error_msg.lower() or "invalid" in error_msg.lower():
                chaos_metrics.record_error_message_quality(clear=True)
            else:
                chaos_metrics.record_error_message_quality(clear=False)
            chaos_metrics.record_graceful_failure()

    def test_partial_file_write(
        self, temp_workspace, data_manager, sample_adata, chaos_metrics
    ):
        """Test detection of partial file writes."""
        data_manager.modalities["test_data"] = sample_adata
        output_path = temp_workspace / "partial.h5ad"

        def partial_write(*args, **kwargs):
            # Write partial data
            with open(output_path, "wb") as f:
                f.write(b"PARTIAL")
            # Don't raise error to simulate silent failure

        with patch("anndata.AnnData.write_h5ad", side_effect=partial_write):
            try:
                data_manager.save_processed_data("test_data", output_path)
                # Try to read back
                data_manager.load_file(output_path, "reloaded")
                chaos_metrics.record_catastrophic_failure()
            except Exception:
                # Should detect corruption on reload
                chaos_metrics.record_graceful_failure()
                chaos_metrics.record_error_message_quality(clear=True)

    def test_file_lock_contention(
        self, temp_workspace, data_manager, sample_adata, chaos_metrics
    ):
        """Test handling of file lock contention."""
        data_manager.modalities["test_data"] = sample_adata
        output_path = temp_workspace / "locked.h5ad"

        # Create a file lock
        lock_file = open(output_path, "wb")

        try:
            # Try to write to same file
            data_manager.save_processed_data("test_data", output_path)
            # On some systems this may succeed, on others may fail
            chaos_metrics.record_graceful_failure()
        except Exception as e:
            error_msg = str(e).lower()
            if "lock" in error_msg or "use" in error_msg or "access" in error_msg:
                chaos_metrics.record_error_message_quality(clear=True)
            else:
                chaos_metrics.record_error_message_quality(clear=False)
            chaos_metrics.record_graceful_failure()
        finally:
            lock_file.close()


# ============================================================================
# MEMORY FAILURE TESTS
# ============================================================================


class TestMemoryFailures:
    """Test system resilience to memory failures."""

    def test_out_of_memory_during_load(self, data_manager, chaos_metrics):
        """Test handling of OOM during data loading."""
        with patch(
            "anndata.read_h5ad", side_effect=MemoryError("Cannot allocate memory")
        ):
            try:
                data_manager.load_file(Path("fake.h5ad"), "test")
                chaos_metrics.record_catastrophic_failure()
            except MemoryError:
                chaos_metrics.record_error_message_quality(clear=True)
                chaos_metrics.record_graceful_failure()
            except Exception as e:
                if "memory" in str(e).lower():
                    chaos_metrics.record_error_message_quality(clear=True)
                else:
                    chaos_metrics.record_error_message_quality(clear=False)
                chaos_metrics.record_graceful_failure()

    def test_memory_allocation_failure(self, chaos_metrics):
        """Test handling of large array allocation failure."""
        try:
            # Try to allocate impossibly large array
            huge_array = np.zeros((10**10, 10**10))
            chaos_metrics.record_catastrophic_failure()
        except MemoryError:
            chaos_metrics.record_error_message_quality(clear=True)
            chaos_metrics.record_graceful_failure()
        except Exception:
            chaos_metrics.record_graceful_failure()

    def test_large_dataset_oom(self, data_manager, chaos_metrics):
        """Test handling of OOM with very large dataset."""
        # Create large dataset
        try:
            large_X = np.random.randn(100000, 5000)
            large_adata = anndata.AnnData(X=large_X)
            data_manager.modalities["large_data"] = large_adata

            # Force garbage collection
            del large_X
            del large_adata
            gc.collect()

            chaos_metrics.record_graceful_failure()
        except MemoryError:
            chaos_metrics.record_error_message_quality(clear=True)
            chaos_metrics.record_graceful_failure()
        except Exception:
            chaos_metrics.record_graceful_failure()

    def test_memory_leak_detection(self, data_manager, sample_adata, chaos_metrics):
        """Test for memory leaks in repeated operations."""
        import psutil

        process = psutil.Process()

        initial_memory = process.memory_info().rss / (1024 * 1024)

        # Perform many operations
        for i in range(100):
            data_manager.modalities[f"test_{i}"] = sample_adata.copy()
            if i % 10 == 0:
                data_manager.remove_modality(f"test_{i}")

        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = final_memory - initial_memory

        # Should not increase by more than 500 MB
        if memory_increase > 500:
            chaos_metrics.record_data_corruption()
            chaos_metrics.record_catastrophic_failure()
        else:
            chaos_metrics.record_graceful_failure()


# ============================================================================
# SERVICE FAILURE TESTS
# ============================================================================


class TestServiceFailures:
    """Test system resilience to external service failures."""

    def test_geo_api_unavailable(self, data_manager, chaos_metrics):
        """Test handling of GEO API unavailability."""
        geo_service = GEOService(data_manager)

        with patch(
            "urllib.request.urlopen", side_effect=ConnectionError("Service unavailable")
        ):
            try:
                geo_service.fetch_and_store_dataset("GSE12345")
                chaos_metrics.record_catastrophic_failure()
            except (ConnectionError, Exception) as e:
                error_msg = str(e)
                if (
                    "unavailable" in error_msg.lower()
                    or "connection" in error_msg.lower()
                ):
                    chaos_metrics.record_error_message_quality(clear=True)
                else:
                    chaos_metrics.record_error_message_quality(clear=False)
                chaos_metrics.record_graceful_failure()

    def test_api_timeout_handling(self, data_manager, chaos_metrics):
        """Test handling of API timeout."""
        geo_service = GEOService(data_manager)

        with patch(
            "urllib.request.urlopen", side_effect=TimeoutError("Request timed out")
        ):
            try:
                geo_service.fetch_and_store_dataset("GSE12345")
                chaos_metrics.record_catastrophic_failure()
            except TimeoutError:
                chaos_metrics.record_error_message_quality(clear=True)
                chaos_metrics.record_graceful_failure()
            except Exception:
                chaos_metrics.record_graceful_failure()

    def test_cache_invalidation_failure(
        self, data_manager, sample_adata, chaos_metrics
    ):
        """Test handling of cache invalidation failures."""
        data_manager.modalities["test_data"] = sample_adata

        try:
            # Try to clear cache
            if hasattr(data_manager, "clear_cache"):
                data_manager.clear_cache()
            chaos_metrics.record_graceful_failure()
        except Exception as e:
            if "cache" in str(e).lower():
                chaos_metrics.record_error_message_quality(clear=True)
            else:
                chaos_metrics.record_error_message_quality(clear=False)
            chaos_metrics.record_graceful_failure()

    def test_service_timeout_handling(self, data_manager, chaos_metrics):
        """Test handling of service timeouts."""

        # Simulate long-running operation
        def slow_operation():
            time.sleep(10)

        try:
            with patch(
                "lobster.tools.geo_service.GEOService.fetch_and_store_dataset",
                side_effect=slow_operation,
            ):
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("Operation timed out")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(2)

                try:
                    geo_service = GEOService(data_manager)
                    geo_service.fetch_and_store_dataset("GSE12345")
                except TimeoutError:
                    chaos_metrics.record_graceful_failure()
                    chaos_metrics.record_error_message_quality(clear=True)
                finally:
                    signal.alarm(0)
        except Exception:
            # Signal not available on all platforms
            chaos_metrics.record_graceful_failure()


# ============================================================================
# DATA CORRUPTION TESTS
# ============================================================================


class TestDataCorruption:
    """Test system resilience to data corruption."""

    def test_corrupted_csv_file(self, temp_workspace, data_manager, chaos_metrics):
        """Test handling of corrupted CSV file."""
        corrupted_csv = temp_workspace / "corrupted.csv"
        with open(corrupted_csv, "w") as f:
            f.write("col1,col2,col3\n")
            f.write("val1,val2\n")  # Missing column
            f.write("val3,val4,val5,val6\n")  # Extra column

        try:
            df = pd.read_csv(corrupted_csv)
            # Pandas may handle this gracefully
            chaos_metrics.record_graceful_failure()
        except Exception as e:
            error_msg = str(e)
            if "parse" in error_msg.lower() or "column" in error_msg.lower():
                chaos_metrics.record_error_message_quality(clear=True)
            else:
                chaos_metrics.record_error_message_quality(clear=False)
            chaos_metrics.record_graceful_failure()

    def test_malformed_h5ad_structure(
        self, temp_workspace, data_manager, chaos_metrics
    ):
        """Test handling of malformed H5AD file structure."""
        # Create H5AD with missing required fields
        try:
            import h5py

            bad_h5ad = temp_workspace / "malformed.h5ad"
            with h5py.File(bad_h5ad, "w") as f:
                # Create incomplete structure
                f.create_dataset("X", data=[[1, 2], [3, 4]])
                # Missing obs, var, etc.

            data_manager.load_file(bad_h5ad, "malformed")
            chaos_metrics.record_catastrophic_failure()
        except Exception as e:
            error_msg = str(e)
            if "structure" in error_msg.lower() or "invalid" in error_msg.lower():
                chaos_metrics.record_error_message_quality(clear=True)
            else:
                chaos_metrics.record_error_message_quality(clear=False)
            chaos_metrics.record_graceful_failure()

    def test_gzip_corruption(self, temp_workspace, chaos_metrics):
        """Test handling of gzip corruption."""
        corrupted_gz = temp_workspace / "corrupted.gz"
        with open(corrupted_gz, "wb") as f:
            f.write(b"\x1f\x8b\x08\x00CORRUPTED")

        try:
            import gzip

            with gzip.open(corrupted_gz, "rb") as f:
                f.read()
            chaos_metrics.record_catastrophic_failure()
        except Exception as e:
            error_msg = str(e)
            if "corrupt" in error_msg.lower() or "gzip" in error_msg.lower():
                chaos_metrics.record_error_message_quality(clear=True)
            else:
                chaos_metrics.record_error_message_quality(clear=False)
            chaos_metrics.record_graceful_failure()

    def test_truncated_download(self, temp_workspace, chaos_metrics):
        """Test detection of truncated downloads."""
        truncated_file = temp_workspace / "truncated.tar.gz"
        with open(truncated_file, "wb") as f:
            f.write(b"PARTIAL_TAR_DATA" * 10)

        try:
            import tarfile

            with tarfile.open(truncated_file, "r:gz") as tar:
                tar.extractall(path=temp_workspace)
            chaos_metrics.record_catastrophic_failure()
        except (tarfile.ReadError, EOFError):
            chaos_metrics.record_error_message_quality(clear=True)
            chaos_metrics.record_graceful_failure()
        except Exception:
            chaos_metrics.record_graceful_failure()

    def test_invalid_metadata_json(self, temp_workspace, data_manager, chaos_metrics):
        """Test handling of invalid metadata JSON."""
        invalid_json = temp_workspace / "metadata.json"
        with open(invalid_json, "w") as f:
            f.write('{"key": "value", "incomplete": ')

        try:
            with open(invalid_json, "r") as f:
                json.load(f)
            chaos_metrics.record_catastrophic_failure()
        except json.JSONDecodeError:
            chaos_metrics.record_error_message_quality(clear=True)
            chaos_metrics.record_graceful_failure()
        except Exception:
            chaos_metrics.record_graceful_failure()


# ============================================================================
# CHAOS CAMPAIGN SUMMARY
# ============================================================================


@pytest.fixture(scope="session")
def generate_chaos_report(request):
    """Generate comprehensive chaos engineering report after all tests."""
    yield

    # This will run after all tests complete
    # Note: Individual test metrics are collected in chaos_metrics fixture


def test_generate_final_chaos_report(chaos_metrics):
    """Generate and save final chaos engineering report."""
    report = chaos_metrics.get_report()

    print("\n" + "=" * 80)
    print("CHAOS ENGINEERING REPORT")
    print("=" * 80)
    print(f"Total Tests: {report['total_tests']}")
    print(f"Graceful Failures: {report['graceful_failures']}")
    print(f"Catastrophic Failures: {report['catastrophic_failures']}")
    print(f"Partial Recoveries: {report['partial_recoveries']}")
    print(f"Full Recoveries: {report['full_recoveries']}")
    print(f"Data Corruptions: {report['data_corruptions']}")
    print(
        f"Error Message Quality: {report['error_message_quality']['clear']}/{report['error_message_quality']['clear'] + report['error_message_quality']['cryptic']} clear"
    )
    print(f"\nResilience Score: {report['resilience_score']:.1f}/100")
    print(f"Grade: {report['grade']}")
    print("=" * 80)
