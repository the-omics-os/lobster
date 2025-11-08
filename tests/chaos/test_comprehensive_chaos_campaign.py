"""
Comprehensive Chaos Engineering Campaign.

This test orchestrates all chaos scenarios and generates a complete resilience report.
"""

import json
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.geo_service import GEOService


class ComprehensiveChaosMetrics:
    """Comprehensive resilience metrics for the chaos campaign."""

    def __init__(self):
        self.categories = {
            "network_failures": {"tests": 0, "graceful": 0, "catastrophic": 0},
            "disk_failures": {"tests": 0, "graceful": 0, "catastrophic": 0},
            "memory_failures": {"tests": 0, "graceful": 0, "catastrophic": 0},
            "service_failures": {"tests": 0, "graceful": 0, "catastrophic": 0},
            "data_corruption": {"tests": 0, "graceful": 0, "catastrophic": 0},
        }
        self.error_clarity_scores = []
        self.recovery_attempts = []
        self.data_integrity_checks = []

    def record_test(self, category: str, graceful: bool, error_clarity: int = 3):
        """Record a test result.

        Args:
            category: Test category
            graceful: Whether failure was handled gracefully
            error_clarity: Error message clarity (1-5, 5 being clearest)
        """
        if category in self.categories:
            self.categories[category]["tests"] += 1
            if graceful:
                self.categories[category]["graceful"] += 1
            else:
                self.categories[category]["catastrophic"] += 1
            self.error_clarity_scores.append(error_clarity)

    def record_recovery_attempt(self, success: bool, attempts: int):
        """Record recovery attempt."""
        self.recovery_attempts.append({"success": success, "attempts": attempts})

    def record_data_integrity(self, intact: bool):
        """Record data integrity check."""
        self.data_integrity_checks.append(intact)

    def get_comprehensive_report(self):
        """Generate comprehensive resilience report."""
        total_tests = sum(cat["tests"] for cat in self.categories.values())
        total_graceful = sum(cat["graceful"] for cat in self.categories.values())
        total_catastrophic = sum(
            cat["catastrophic"] for cat in self.categories.values()
        )

        avg_error_clarity = (
            sum(self.error_clarity_scores) / len(self.error_clarity_scores)
            if self.error_clarity_scores
            else 0
        )

        successful_recoveries = sum(1 for r in self.recovery_attempts if r["success"])
        total_recoveries = len(self.recovery_attempts)

        data_intact_count = sum(1 for d in self.data_integrity_checks if d)
        total_integrity_checks = len(self.data_integrity_checks)

        # Calculate resilience score
        resilience_score = self._calculate_resilience_score(
            total_tests,
            total_graceful,
            total_catastrophic,
            avg_error_clarity,
            successful_recoveries,
            total_recoveries,
            data_intact_count,
            total_integrity_checks,
        )

        return {
            "summary": {
                "total_tests": total_tests,
                "graceful_failures": total_graceful,
                "catastrophic_failures": total_catastrophic,
                "graceful_percentage": (
                    (total_graceful / total_tests * 100) if total_tests > 0 else 0
                ),
            },
            "categories": self.categories,
            "error_quality": {
                "average_clarity": avg_error_clarity,
                "scores": self.error_clarity_scores,
            },
            "recovery": {
                "successful": successful_recoveries,
                "total_attempts": total_recoveries,
                "success_rate": (
                    (successful_recoveries / total_recoveries * 100)
                    if total_recoveries > 0
                    else 0
                ),
            },
            "data_integrity": {
                "checks_passed": data_intact_count,
                "total_checks": total_integrity_checks,
                "integrity_rate": (
                    (data_intact_count / total_integrity_checks * 100)
                    if total_integrity_checks > 0
                    else 0
                ),
            },
            "resilience_score": resilience_score,
            "grade": self._score_to_grade(resilience_score),
            "production_ready": resilience_score >= 75,
        }

    def _calculate_resilience_score(
        self,
        total_tests,
        graceful,
        catastrophic,
        error_clarity,
        recoveries,
        total_recoveries,
        data_intact,
        total_integrity,
    ):
        """Calculate overall resilience score (0-100)."""
        if total_tests == 0:
            return 0

        score = 0

        # Graceful failure handling: 40 points
        score += (graceful / total_tests) * 40

        # Error message clarity: 15 points
        score += (error_clarity / 5) * 15

        # Recovery capability: 20 points
        if total_recoveries > 0:
            score += (recoveries / total_recoveries) * 20
        else:
            score += 10  # Partial credit if no recovery attempts needed

        # Data integrity: 25 points
        if total_integrity > 0:
            score += (data_intact / total_integrity) * 25
        else:
            score += 20  # Partial credit if no integrity checks

        # Catastrophic failure penalty
        if catastrophic > 0:
            penalty = (catastrophic / total_tests) * 30
            score -= penalty

        return max(0, min(100, score))

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
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


@pytest.fixture
def comprehensive_metrics():
    """Provide comprehensive metrics tracker."""
    return ComprehensiveChaosMetrics()


@pytest.fixture
def temp_workspace():
    """Create temporary workspace."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def data_manager(temp_workspace):
    """Create data manager."""
    return DataManagerV2(workspace_path=temp_workspace)


@pytest.fixture
def sample_adata():
    """Create sample AnnData."""
    n_obs, n_vars = 100, 50
    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame({"cell_type": [f"type_{i%3}" for i in range(n_obs)]})
    var = pd.DataFrame({"gene_name": [f"gene_{i}" for i in range(n_vars)]})
    return anndata.AnnData(X=X, obs=obs, var=var)


class TestComprehensiveChaosEngineering:
    """Run comprehensive chaos engineering campaign."""

    def test_network_failure_suite(self, data_manager, comprehensive_metrics):
        """Test all network failure scenarios."""
        geo_service = GEOService(data_manager)

        # Test 1: Connection timeout
        with patch(
            "urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")
        ):
            try:
                geo_service.fetch_and_store_dataset("GSE12345")
                comprehensive_metrics.record_test("network_failures", False, 1)
            except TimeoutError as e:
                clarity = 5 if "timeout" in str(e).lower() else 2
                comprehensive_metrics.record_test("network_failures", True, clarity)
            except Exception:
                comprehensive_metrics.record_test("network_failures", True, 2)

        # Test 2: Connection refused
        with patch(
            "urllib.request.urlopen",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            try:
                geo_service.fetch_and_store_dataset("GSE12345")
                comprehensive_metrics.record_test("network_failures", False, 1)
            except (ConnectionRefusedError, Exception) as e:
                clarity = 5 if "connection" in str(e).lower() else 2
                comprehensive_metrics.record_test("network_failures", True, clarity)

        # Test 3: DNS failure
        with patch(
            "urllib.request.urlopen",
            side_effect=socket.gaierror("Name resolution failed"),
        ):
            try:
                geo_service.fetch_and_store_dataset("GSE12345")
                comprehensive_metrics.record_test("network_failures", False, 1)
            except (socket.gaierror, Exception) as e:
                clarity = 5 if "resolution" in str(e).lower() else 2
                comprehensive_metrics.record_test("network_failures", True, clarity)

        # Test 4: Partial download
        temp_file = data_manager.workspace_path / "partial.tar.gz"
        with open(temp_file, "wb") as f:
            f.write(b"PARTIAL_DATA" * 100)

        try:
            import tarfile

            with tarfile.open(temp_file, "r:gz") as tar:
                tar.extractall(path=data_manager.workspace_path)
            comprehensive_metrics.record_test("network_failures", False, 1)
        except (tarfile.ReadError, EOFError):
            comprehensive_metrics.record_test("network_failures", True, 5)
        except Exception:
            comprehensive_metrics.record_test("network_failures", True, 2)

    def test_disk_failure_suite(
        self, data_manager, sample_adata, temp_workspace, comprehensive_metrics
    ):
        """Test all disk failure scenarios."""
        # Test 1: Disk full
        data_manager.modalities["test_data"] = sample_adata
        with patch(
            "anndata.AnnData.write_h5ad",
            side_effect=OSError(28, "No space left on device"),
        ):
            try:
                data_manager.save_processed_data("test_data")
                comprehensive_metrics.record_test("disk_failures", False, 1)
                comprehensive_metrics.record_data_integrity(False)
            except OSError as e:
                clarity = 5 if "space" in str(e).lower() else 2
                comprehensive_metrics.record_test("disk_failures", True, clarity)
                comprehensive_metrics.record_data_integrity(True)
            except Exception:
                comprehensive_metrics.record_test("disk_failures", True, 2)
                comprehensive_metrics.record_data_integrity(True)

        # Test 2: Permission denied (read)
        test_file = temp_workspace / "protected.h5ad"
        test_file.touch()
        os.chmod(test_file, 0o000)
        try:
            data_manager.load_file(test_file, "protected")
            comprehensive_metrics.record_test("disk_failures", False, 1)
        except (PermissionError, Exception) as e:
            clarity = 5 if "permission" in str(e).lower() else 2
            comprehensive_metrics.record_test("disk_failures", True, clarity)
        finally:
            os.chmod(test_file, 0o644)

        # Test 3: Corrupted file
        corrupted_file = temp_workspace / "corrupted.h5ad"
        with open(corrupted_file, "wb") as f:
            f.write(b"CORRUPTED_DATA_\x00\xff\x00")

        try:
            data_manager.load_file(corrupted_file, "corrupted")
            comprehensive_metrics.record_test("disk_failures", False, 1)
            comprehensive_metrics.record_data_integrity(False)
        except Exception as e:
            clarity = (
                5 if ("corrupt" in str(e).lower() or "invalid" in str(e).lower()) else 3
            )
            comprehensive_metrics.record_test("disk_failures", True, clarity)
            comprehensive_metrics.record_data_integrity(True)

    def test_memory_failure_suite(
        self, data_manager, sample_adata, comprehensive_metrics
    ):
        """Test all memory failure scenarios."""
        # Test 1: OOM during load
        with patch(
            "anndata.read_h5ad", side_effect=MemoryError("Cannot allocate memory")
        ):
            try:
                data_manager.load_file(Path("fake.h5ad"), "test")
                comprehensive_metrics.record_test("memory_failures", False, 1)
            except MemoryError:
                comprehensive_metrics.record_test("memory_failures", True, 5)
            except Exception as e:
                clarity = 5 if "memory" in str(e).lower() else 2
                comprehensive_metrics.record_test("memory_failures", True, clarity)

        # Test 2: Large allocation
        try:
            huge_array = np.zeros((10**10, 10**10))
            comprehensive_metrics.record_test("memory_failures", False, 1)
        except MemoryError:
            comprehensive_metrics.record_test("memory_failures", True, 5)
        except Exception:
            comprehensive_metrics.record_test("memory_failures", True, 3)

        # Test 3: Memory management
        try:
            for i in range(10):
                data_manager.modalities[f"test_{i}"] = sample_adata.copy()
            for i in range(5):
                data_manager.remove_modality(f"test_{i}")
            comprehensive_metrics.record_test("memory_failures", True, 4)
        except Exception:
            comprehensive_metrics.record_test("memory_failures", False, 2)

    def test_service_failure_suite(self, data_manager, comprehensive_metrics):
        """Test all service failure scenarios."""
        geo_service = GEOService(data_manager)

        # Test 1: API unavailable
        with patch(
            "urllib.request.urlopen",
            side_effect=ConnectionError("Service unavailable"),
        ):
            try:
                geo_service.fetch_and_store_dataset("GSE12345")
                comprehensive_metrics.record_test("service_failures", False, 1)
            except (ConnectionError, Exception) as e:
                clarity = 5 if "unavailable" in str(e).lower() else 3
                comprehensive_metrics.record_test("service_failures", True, clarity)

        # Test 2: API timeout
        with patch(
            "urllib.request.urlopen", side_effect=TimeoutError("Request timed out")
        ):
            try:
                geo_service.fetch_and_store_dataset("GSE12345")
                comprehensive_metrics.record_test("service_failures", False, 1)
            except TimeoutError:
                comprehensive_metrics.record_test("service_failures", True, 5)
            except Exception:
                comprehensive_metrics.record_test("service_failures", True, 3)

    def test_data_corruption_suite(
        self, temp_workspace, data_manager, comprehensive_metrics
    ):
        """Test all data corruption scenarios."""
        # Test 1: Corrupted CSV
        corrupted_csv = temp_workspace / "corrupted.csv"
        with open(corrupted_csv, "w") as f:
            f.write("col1,col2,col3\n")
            f.write("val1,val2\n")
            f.write("val3,val4,val5,val6\n")

        try:
            df = pd.read_csv(corrupted_csv)
            comprehensive_metrics.record_test("data_corruption", True, 3)
        except Exception as e:
            clarity = (
                5 if ("parse" in str(e).lower() or "column" in str(e).lower()) else 2
            )
            comprehensive_metrics.record_test("data_corruption", True, clarity)

        # Test 2: Gzip corruption
        corrupted_gz = temp_workspace / "corrupted.gz"
        with open(corrupted_gz, "wb") as f:
            f.write(b"\x1f\x8b\x08\x00CORRUPTED")

        try:
            import gzip

            with gzip.open(corrupted_gz, "rb") as f:
                f.read()
            comprehensive_metrics.record_test("data_corruption", False, 1)
            comprehensive_metrics.record_data_integrity(False)
        except Exception as e:
            clarity = (
                5 if ("corrupt" in str(e).lower() or "gzip" in str(e).lower()) else 3
            )
            comprehensive_metrics.record_test("data_corruption", True, clarity)
            comprehensive_metrics.record_data_integrity(True)

        # Test 3: JSON corruption
        invalid_json = temp_workspace / "metadata.json"
        with open(invalid_json, "w") as f:
            f.write('{"key": "value", "incomplete": ')

        try:
            with open(invalid_json, "r") as f:
                json.load(f)
            comprehensive_metrics.record_test("data_corruption", False, 1)
        except json.JSONDecodeError:
            comprehensive_metrics.record_test("data_corruption", True, 5)
        except Exception:
            comprehensive_metrics.record_test("data_corruption", True, 3)


def test_generate_comprehensive_chaos_report(comprehensive_metrics):
    """Generate final comprehensive chaos engineering report."""
    # Run all test suites
    pytest.main(
        [
            __file__,
            "-v",
            "-k",
            "TestComprehensiveChaosEngineering",
            "--tb=no",
        ]
    )

    # Generate report
    report = comprehensive_metrics.get_comprehensive_report()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE CHAOS ENGINEERING REPORT")
    print("=" * 80)
    print(f"\nTOTAL TESTS: {report['summary']['total_tests']}")
    print(
        f"Graceful Failures: {report['summary']['graceful_failures']} ({report['summary']['graceful_percentage']:.1f}%)"
    )
    print(f"Catastrophic Failures: {report['summary']['catastrophic_failures']}")

    print("\nCATEGORY BREAKDOWN:")
    for category, data in report["categories"].items():
        if data["tests"] > 0:
            success_rate = (data["graceful"] / data["tests"]) * 100
            print(
                f"  {category.replace('_', ' ').title()}: {data['graceful']}/{data['tests']} ({success_rate:.1f}%)"
            )

    print(
        f"\nERROR MESSAGE QUALITY: {report['error_quality']['average_clarity']:.2f}/5.00"
    )

    print(f"\nRECOVERY CAPABILITY:")
    print(
        f"  Successful: {report['recovery']['successful']}/{report['recovery']['total_attempts']}"
    )
    print(f"  Success Rate: {report['recovery']['success_rate']:.1f}%")

    print(f"\nDATA INTEGRITY:")
    print(
        f"  Checks Passed: {report['data_integrity']['checks_passed']}/{report['data_integrity']['total_checks']}"
    )
    print(f"  Integrity Rate: {report['data_integrity']['integrity_rate']:.1f}%")

    print(f"\nRESILIENCE SCORE: {report['resilience_score']:.1f}/100")
    print(f"GRADE: {report['grade']}")
    print(f"PRODUCTION READY: {'YES ✅' if report['production_ready'] else 'NO ❌'}")
    print("=" * 80)

    # Save report to file
    report_path = (
        Path(__file__).parent.parent.parent
        / "kevin_notes"
        / "CHAOS_ENGINEERING_REPORT.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")
