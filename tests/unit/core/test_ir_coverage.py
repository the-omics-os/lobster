"""
Unit tests for IR coverage analysis.

Tests the IRCoverageAnalyzer and related dataclasses.
"""

import tempfile
from pathlib import Path

import pytest

from lobster.core.ir_coverage import (
    CoverageReport,
    IRCoverageAnalyzer,
    ServiceCoverage,
)


class TestServiceCoverage:
    """Test ServiceCoverage dataclass."""

    def test_coverage_calculation_empty(self):
        """Test coverage calculation with no methods."""
        coverage = ServiceCoverage(
            service_name="test_service",
            service_path=Path("/fake/path.py"),
        )

        assert coverage.total_methods == 0
        assert coverage.coverage_percentage == 0.0

    def test_coverage_calculation_full(self):
        """Test coverage calculation with all methods having IR."""
        coverage = ServiceCoverage(
            service_name="test_service",
            service_path=Path("/fake/path.py"),
            has_ir_import=True,
            methods_with_ir={"method1", "method2", "method3"},
            methods_without_ir=set(),
        )

        assert coverage.total_methods == 3
        assert coverage.coverage_percentage == 100.0

    def test_coverage_calculation_partial(self):
        """Test coverage calculation with partial IR coverage."""
        coverage = ServiceCoverage(
            service_name="test_service",
            service_path=Path("/fake/path.py"),
            has_ir_import=True,
            methods_with_ir={"method1", "method2"},
            methods_without_ir={"method3", "method4"},
        )

        assert coverage.total_methods == 4
        assert coverage.coverage_percentage == 50.0

    def test_coverage_calculation_no_ir(self):
        """Test coverage calculation with no IR coverage."""
        coverage = ServiceCoverage(
            service_name="test_service",
            service_path=Path("/fake/path.py"),
            has_ir_import=False,
            methods_with_ir=set(),
            methods_without_ir={"method1", "method2"},
        )

        assert coverage.total_methods == 2
        assert coverage.coverage_percentage == 0.0


class TestCoverageReport:
    """Test CoverageReport dataclass."""

    def test_empty_report(self):
        """Test empty coverage report."""
        report = CoverageReport()

        assert report.total_services == 0
        assert report.services_with_ir == 0
        assert report.total_methods == 0
        assert report.methods_with_ir == 0
        assert report.overall_coverage == 0.0

    def test_report_with_services(self):
        """Test report with multiple services."""
        services = {
            "service1": ServiceCoverage(
                service_name="service1",
                service_path=Path("/fake/service1.py"),
                has_ir_import=True,
                methods_with_ir={"method1", "method2"},
                methods_without_ir=set(),
            ),
            "service2": ServiceCoverage(
                service_name="service2",
                service_path=Path("/fake/service2.py"),
                has_ir_import=True,
                methods_with_ir={"method3"},
                methods_without_ir={"method4"},
            ),
            "service3": ServiceCoverage(
                service_name="service3",
                service_path=Path("/fake/service3.py"),
                has_ir_import=False,
                methods_with_ir=set(),
                methods_without_ir={"method5", "method6"},
            ),
        }

        report = CoverageReport(services=services)

        assert report.total_services == 3
        assert report.services_with_ir == 2  # service1 and service2
        assert report.total_methods == 6  # 2 + 2 + 2
        assert report.methods_with_ir == 3  # 2 + 1 + 0
        assert report.overall_coverage == 50.0  # 3/6 = 50%

    def test_report_to_dict(self):
        """Test conversion of report to dictionary."""
        services = {
            "service1": ServiceCoverage(
                service_name="service1",
                service_path=Path("/fake/service1.py"),
                has_ir_import=True,
                methods_with_ir={"method1"},
                methods_without_ir={"method2"},
            ),
        }

        report = CoverageReport(services=services, timestamp="2025-01-01T00:00:00")

        report_dict = report.to_dict()

        assert report_dict["timestamp"] == "2025-01-01T00:00:00"
        assert report_dict["summary"]["total_services"] == 1
        assert report_dict["summary"]["overall_coverage"] == 50.0
        assert "service1" in report_dict["services"]
        assert report_dict["services"]["service1"]["has_ir_import"] is True


class TestIRCoverageAnalyzer:
    """Test IRCoverageAnalyzer class."""

    def test_analyzer_initialization_default(self):
        """Test analyzer initialization with default services directory."""
        analyzer = IRCoverageAnalyzer()

        # Should default to lobster/tools/
        assert analyzer.services_dir.name == "tools"

    def test_analyzer_initialization_custom(self):
        """Test analyzer initialization with custom services directory."""
        custom_dir = Path("/custom/services")
        analyzer = IRCoverageAnalyzer(services_dir=custom_dir)

        assert analyzer.services_dir == custom_dir

    def test_has_analysis_step_import_positive(self):
        """Test detection of AnalysisStep import."""
        source = """
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec

def some_function():
    pass
"""
        import ast

        tree = ast.parse(source)
        analyzer = IRCoverageAnalyzer()

        assert analyzer._has_analysis_step_import(tree) is True

    def test_has_analysis_step_import_negative(self):
        """Test detection when AnalysisStep is not imported."""
        source = """
from lobster.core.schemas import TranscriptomicsSchema

def some_function():
    pass
"""
        import ast

        tree = ast.parse(source)
        analyzer = IRCoverageAnalyzer()

        assert analyzer._has_analysis_step_import(tree) is False

    def test_method_returns_ir_positive(self):
        """Test detection of method returning 3-tuple with IR."""
        source = """
def process_data(adata, params):
    result = adata.copy()
    stats = {"processed": True}

    ir = AnalysisStep(
        operation="process",
        code_template="code here",
        parameters=params,
        parameter_schema={}
    )

    return result, stats, ir
"""
        import ast

        tree = ast.parse(source)
        func_node = tree.body[0]
        analyzer = IRCoverageAnalyzer()

        assert analyzer._method_returns_ir(func_node) is True

    def test_method_returns_ir_negative_2tuple(self):
        """Test detection fails for 2-tuple return."""
        source = """
def process_data(adata, params):
    result = adata.copy()
    stats = {"processed": True}
    return result, stats
"""
        import ast

        tree = ast.parse(source)
        func_node = tree.body[0]
        analyzer = IRCoverageAnalyzer()

        assert analyzer._method_returns_ir(func_node) is False

    def test_method_returns_ir_negative_wrong_var_name(self):
        """Test detection fails for 3-tuple without IR-like variable name."""
        source = """
def process_data(adata, params):
    result = adata.copy()
    stats = {"processed": True}
    metadata = {"extra": "info"}
    return result, stats, metadata
"""
        import ast

        tree = ast.parse(source)
        func_node = tree.body[0]
        analyzer = IRCoverageAnalyzer()

        assert analyzer._method_returns_ir(func_node) is False

    def test_analyze_methods(self):
        """Test method analysis in a service class."""
        source = """
from lobster.core.analysis_ir import AnalysisStep

class TestService:
    def method_with_ir(self, data):
        result = data
        stats = {}
        ir = AnalysisStep(operation="test", code_template="code", parameters={}, parameter_schema={})
        return result, stats, ir

    def method_without_ir(self, data):
        result = data
        stats = {}
        return result, stats

    def _private_method(self, data):
        return data
"""
        import ast

        tree = ast.parse(source)
        analyzer = IRCoverageAnalyzer()

        methods_with_ir, methods_without_ir = analyzer._analyze_methods(tree)

        assert "method_with_ir" in methods_with_ir
        assert "method_without_ir" in methods_without_ir
        assert "_private_method" not in methods_with_ir
        assert "_private_method" not in methods_without_ir

    def test_analyze_service_file(self):
        """Test analysis of a complete service file."""
        service_code = """
from lobster.core.analysis_ir import AnalysisStep

class QualityService:
    def assess_quality(self, adata):
        result = adata
        stats = {"quality": "good"}
        ir = AnalysisStep(operation="qc", code_template="scanpy.pp.calculate_qc_metrics", parameters={}, parameter_schema={})
        return result, stats, ir

    def compute_metrics(self, adata):
        result = adata
        stats = {"metrics": "computed"}
        return result, stats
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_service.py", delete=False
        ) as f:
            f.write(service_code)
            temp_path = Path(f.name)

        try:
            analyzer = IRCoverageAnalyzer()
            coverage = analyzer.analyze_service_file(temp_path)

            assert coverage.service_name == temp_path.stem
            assert coverage.has_ir_import is True
            assert "assess_quality" in coverage.methods_with_ir
            assert "compute_metrics" in coverage.methods_without_ir
            assert coverage.total_methods == 2
            assert coverage.coverage_percentage == 50.0

        finally:
            temp_path.unlink()

    def test_scan_services(self):
        """Test scanning multiple service files."""
        # Create temporary services directory
        with tempfile.TemporaryDirectory() as temp_dir:
            services_dir = Path(temp_dir)

            # Create service1 with IR
            service1_code = """
from lobster.core.analysis_ir import AnalysisStep

class Service1:
    def method1(self, data):
        ir = AnalysisStep(operation="op1", code_template="code", parameters={}, parameter_schema={})
        return data, {}, ir
"""
            service1_path = services_dir / "service1_service.py"
            service1_path.write_text(service1_code)

            # Create service2 without IR
            service2_code = """
class Service2:
    def method2(self, data):
        return data, {}
"""
            service2_path = services_dir / "service2_service.py"
            service2_path.write_text(service2_code)

            # Scan services
            analyzer = IRCoverageAnalyzer(services_dir=services_dir)
            services = analyzer.scan_services()

            assert len(services) == 2
            assert "service1_service" in services
            assert "service2_service" in services
            assert services["service1_service"].has_ir_import is True
            assert services["service2_service"].has_ir_import is False

    def test_generate_report(self):
        """Test generation of complete coverage report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            services_dir = Path(temp_dir)

            # Create test service
            service_code = """
from lobster.core.analysis_ir import AnalysisStep

class TestService:
    def with_ir(self, data):
        ir = AnalysisStep(operation="test", code_template="code", parameters={}, parameter_schema={})
        return data, {}, ir

    def without_ir(self, data):
        return data, {}
"""
            service_path = services_dir / "test_service.py"
            service_path.write_text(service_code)

            # Generate report
            analyzer = IRCoverageAnalyzer(services_dir=services_dir)
            report = analyzer.generate_report()

            assert report.total_services == 1
            assert report.services_with_ir == 1
            assert report.total_methods == 2
            assert report.methods_with_ir == 1
            assert report.overall_coverage == 50.0
            assert report.timestamp is not None

    def test_identify_gaps(self):
        """Test identification of methods without IR coverage."""
        services = {
            "service1": ServiceCoverage(
                service_name="service1",
                service_path=Path("/fake/service1.py"),
                has_ir_import=True,
                methods_with_ir={"method1"},
                methods_without_ir={"method2", "method3"},
            ),
            "service2": ServiceCoverage(
                service_name="service2",
                service_path=Path("/fake/service2.py"),
                has_ir_import=False,
                methods_with_ir=set(),
                methods_without_ir={"method4"},
            ),
        }

        report = CoverageReport(services=services)
        analyzer = IRCoverageAnalyzer()

        gaps = analyzer.identify_gaps(report)

        assert len(gaps) == 3
        assert ("service1", "method2") in gaps
        assert ("service1", "method3") in gaps
        assert ("service2", "method4") in gaps

    def test_print_report(self, capsys):
        """Test printing of coverage report."""
        services = {
            "test_service": ServiceCoverage(
                service_name="test_service",
                service_path=Path("/fake/test_service.py"),
                has_ir_import=True,
                methods_with_ir={"method1"},
                methods_without_ir={"method2"},
            ),
        }

        report = CoverageReport(services=services, timestamp="2025-01-01T00:00:00")
        analyzer = IRCoverageAnalyzer()

        analyzer.print_report(report, verbose=False)

        captured = capsys.readouterr()
        assert "IR COVERAGE REPORT" in captured.out
        assert "test_service" in captured.out
        assert "50.00%" in captured.out

    def test_print_report_verbose(self, capsys):
        """Test verbose printing of coverage report."""
        services = {
            "test_service": ServiceCoverage(
                service_name="test_service",
                service_path=Path("/fake/test_service.py"),
                has_ir_import=True,
                methods_with_ir={"method1"},
                methods_without_ir={"method2"},
            ),
        }

        report = CoverageReport(services=services, timestamp="2025-01-01T00:00:00")
        analyzer = IRCoverageAnalyzer()

        analyzer.print_report(report, verbose=True)

        captured = capsys.readouterr()
        assert "method1" in captured.out
        assert "method2" in captured.out
