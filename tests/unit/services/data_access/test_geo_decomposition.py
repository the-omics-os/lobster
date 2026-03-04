"""Structural verification tests for GEO Service Decomposition.

Verifies that the decomposition is complete: 5 domain modules exist,
each has the expected class, the facade is thin, no circular imports,
and SOFT pre-download code is fully deduplicated.
Part of Phase 4 Plan 03: GEO Service Decomposition.
"""

import inspect
import subprocess
from pathlib import Path


# Expected domain modules and their classes
DOMAIN_MODULES = {
    "metadata_fetch": "MetadataFetcher",
    "download_execution": "DownloadExecutor",
    "archive_processing": "ArchiveProcessor",
    "matrix_parsing": "MatrixParser",
    "concatenation": "SampleConcatenator",
}

GEO_MODULE_DIR = Path("lobster/services/data_access/geo")


class TestModuleExistence:
    """Verify all 5 domain module files exist."""

    def test_all_domain_modules_exist(self):
        """All 5 domain module files should exist in geo/ directory."""
        for module_name in DOMAIN_MODULES:
            module_path = GEO_MODULE_DIR / f"{module_name}.py"
            assert module_path.exists(), f"Missing domain module: {module_path}"

    def test_soft_download_module_exists(self):
        """soft_download.py should exist with shared SOFT download helpers."""
        soft_path = GEO_MODULE_DIR / "soft_download.py"
        assert soft_path.exists(), f"Missing module: {soft_path}"

    def test_helpers_module_exists(self):
        """helpers.py should exist with shared symbols."""
        helpers_path = GEO_MODULE_DIR / "helpers.py"
        assert helpers_path.exists(), f"Missing module: {helpers_path}"


class TestModuleClasses:
    """Verify each module defines the expected class with correct methods."""

    def test_metadata_fetcher_has_expected_methods(self):
        """MetadataFetcher should have fetch_metadata_only and key private methods."""
        from lobster.services.data_access.geo.metadata_fetch import MetadataFetcher

        methods = [m for m in dir(MetadataFetcher) if not m.startswith("__")]
        assert "fetch_metadata_only" in methods
        assert "_fetch_gse_metadata" in methods
        assert "_safely_extract_metadata_field" in methods
        assert "_validate_geo_metadata" in methods
        assert "_detect_sample_types" in methods

    def test_download_executor_has_expected_methods(self):
        """DownloadExecutor should have download_dataset and strategy methods."""
        from lobster.services.data_access.geo.download_execution import DownloadExecutor

        methods = [m for m in dir(DownloadExecutor) if not m.startswith("__")]
        assert "download_dataset" in methods
        assert "download_with_strategy" in methods
        assert "_get_processing_pipeline" in methods
        assert "_try_supplementary_first" in methods

    def test_archive_processor_has_expected_methods(self):
        """ArchiveProcessor should have _process_supplementary_files and tar methods."""
        from lobster.services.data_access.geo.archive_processing import ArchiveProcessor

        methods = [m for m in dir(ArchiveProcessor) if not m.startswith("__")]
        assert "_process_supplementary_files" in methods
        assert "_process_tar_file" in methods
        assert "_detect_kallisto_salmon_files" in methods

    def test_matrix_parser_has_expected_methods(self):
        """MatrixParser should have matrix validation and sample download methods."""
        from lobster.services.data_access.geo.matrix_parsing import MatrixParser

        methods = [m for m in dir(MatrixParser) if not m.startswith("__")]
        assert "_is_valid_expression_matrix" in methods
        assert "_download_single_sample" in methods
        assert "_determine_data_type_from_metadata" in methods

    def test_sample_concatenator_has_expected_methods(self):
        """SampleConcatenator should have storage and concatenation methods."""
        from lobster.services.data_access.geo.concatenation import SampleConcatenator

        methods = [m for m in dir(SampleConcatenator) if not m.startswith("__")]
        assert "_store_samples_as_anndata" in methods
        assert "_analyze_gene_coverage_and_decide_join" in methods
        assert "_concatenate_stored_samples" in methods


class TestFacadeSize:
    """Verify the facade is thin."""

    def test_geo_service_facade_under_400_lines(self):
        """geo_service.py facade should be under 400 lines (was ~3,200)."""
        facade_path = Path("lobster/services/data_access/geo_service.py")
        line_count = len(facade_path.read_text().splitlines())
        assert line_count < 400, (
            f"geo_service.py has {line_count} lines, expected < 400. "
            f"Logic should be in domain modules, not the facade."
        )


class TestNoCircularImports:
    """Verify no circular imports between domain modules."""

    def test_each_module_imports_independently(self):
        """Each domain module should be importable independently without circular errors."""
        import importlib

        modules = [
            "lobster.services.data_access.geo.metadata_fetch",
            "lobster.services.data_access.geo.download_execution",
            "lobster.services.data_access.geo.archive_processing",
            "lobster.services.data_access.geo.matrix_parsing",
            "lobster.services.data_access.geo.concatenation",
            "lobster.services.data_access.geo.soft_download",
            "lobster.services.data_access.geo.helpers",
        ]

        for module_name in modules:
            # Force re-import to detect circular import issues
            mod = importlib.import_module(module_name)
            assert mod is not None, f"Failed to import {module_name}"


class TestSoftDownloadDeduplication:
    """Verify SOFT pre-download code is fully deduplicated."""

    def test_soft_download_module_has_expected_functions(self):
        """soft_download.py should export build_soft_url and pre_download_soft_file."""
        from lobster.services.data_access.geo.soft_download import (
            build_soft_url,
            pre_download_soft_file,
        )

        assert callable(build_soft_url)
        assert callable(pre_download_soft_file)

    def test_helpers_has_all_shared_symbols(self):
        """helpers.py should export all shared symbols."""
        from lobster.services.data_access.geo.helpers import (
            ARCHIVE_EXTENSIONS,
            RetryOutcome,
            RetryResult,
            _is_archive_url,
            _is_data_valid,
            _retry_with_backoff,
            _score_expression_file,
        )

        assert isinstance(ARCHIVE_EXTENSIONS, tuple)
        assert callable(_is_archive_url)
        assert callable(_is_data_valid)
        assert callable(_retry_with_backoff)
        assert callable(_score_expression_file)

    def test_no_soft_blocks_remain_in_source_files(self):
        """Verify SOFT pre-download code is fully deduplicated (GDEC-03).

        The only file that should contain SOFT download logic is soft_download.py.
        All other files should use the shared helper via import.
        """
        files_to_check = [
            "lobster/services/data_access/geo_service.py",
            "lobster/services/data_access/geo/metadata_fetch.py",
            "lobster/services/data_access/geo/download_execution.py",
            "lobster/services/data_access/geo/matrix_parsing.py",
            "lobster/tools/providers/geo_provider.py",
            "lobster/services/data_access/geo_fallback_service.py",
        ]

        result = subprocess.run(
            ["grep", "-rn", "PRE-DOWNLOAD SOFT"] + files_to_check,
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.stdout.strip() == "", (
            f"SOFT pre-download blocks remain in source files:\n{result.stdout}"
        )

    def test_domain_modules_import_from_soft_download(self):
        """Domain modules should import pre_download_soft_file from soft_download.py."""
        files_that_should_import = [
            "lobster/services/data_access/geo/metadata_fetch.py",
            "lobster/services/data_access/geo/download_execution.py",
            "lobster/services/data_access/geo/matrix_parsing.py",
            "lobster/tools/providers/geo_provider.py",
            "lobster/services/data_access/geo_fallback_service.py",
        ]

        for filepath in files_that_should_import:
            content = Path(filepath).read_text()
            assert "from lobster.services.data_access.geo.soft_download import" in content, (
                f"{filepath} should import from soft_download.py"
            )
