"""
Unit tests for ContentDetector in lobster/core/archive_utils.py.

Tests verify:
1. V3 format detection (features.tsv)
2. V2 format detection (genes.tsv)
3. Compressed variants (.gz)
4. Nested directory structures
5. Incomplete/missing file handling
6. Non-10X archive handling (Kallisto, Salmon, GEO RAW)

These tests ensure correct detection of 10X Genomics V2 (genes.tsv) vs V3 (features.tsv)
formats, preventing regression of the zero-genes bug.
"""

import pytest
from pathlib import Path

from lobster.core.archive_utils import (
    ArchiveContentType,
    ArchiveExtractor,
    ArchiveInspector,
    ContentDetector,
)


class TestContentDetector10XFormats:
    """Tests for 10X Genomics V2/V3 format detection."""

    @pytest.fixture
    def v3_10x_directory(self, tmp_path: Path) -> Path:
        """Create V3 format 10X directory structure (uncompressed)."""
        mtx_dir = tmp_path / "filtered_feature_bc_matrix"
        mtx_dir.mkdir(parents=True)
        (mtx_dir / "matrix.mtx").write_bytes(b"mock mtx content")
        (mtx_dir / "features.tsv").write_bytes(b"mock features content")
        (mtx_dir / "barcodes.tsv").write_bytes(b"mock barcodes content")
        return tmp_path

    @pytest.fixture
    def v3_10x_directory_compressed(self, tmp_path: Path) -> Path:
        """Create V3 format 10X directory structure (compressed .gz)."""
        mtx_dir = tmp_path / "filtered_feature_bc_matrix"
        mtx_dir.mkdir(parents=True)
        (mtx_dir / "matrix.mtx.gz").write_bytes(b"mock compressed mtx")
        (mtx_dir / "features.tsv.gz").write_bytes(b"mock compressed features")
        (mtx_dir / "barcodes.tsv.gz").write_bytes(b"mock compressed barcodes")
        return tmp_path

    @pytest.fixture
    def v2_10x_directory(self, tmp_path: Path) -> Path:
        """Create V2 format 10X directory structure (uncompressed)."""
        mtx_dir = tmp_path / "filtered_gene_bc_matrices" / "GRCh38"
        mtx_dir.mkdir(parents=True)
        (mtx_dir / "matrix.mtx").write_bytes(b"mock mtx content")
        (mtx_dir / "genes.tsv").write_bytes(b"mock genes content")
        (mtx_dir / "barcodes.tsv").write_bytes(b"mock barcodes content")
        return tmp_path

    @pytest.fixture
    def v2_10x_directory_compressed(self, tmp_path: Path) -> Path:
        """Create V2 format 10X directory structure (compressed .gz)."""
        mtx_dir = tmp_path / "filtered_gene_bc_matrices" / "GRCh38"
        mtx_dir.mkdir(parents=True)
        (mtx_dir / "matrix.mtx.gz").write_bytes(b"mock compressed mtx")
        (mtx_dir / "genes.tsv.gz").write_bytes(b"mock compressed genes")
        (mtx_dir / "barcodes.tsv.gz").write_bytes(b"mock compressed barcodes")
        return tmp_path

    # --- V3 Format Tests ---

    def test_v3_format_detection_uncompressed(self, v3_10x_directory: Path):
        """V3 format (features.tsv) is correctly identified."""
        result = ContentDetector.detect_content_type(v3_10x_directory)
        assert result == ArchiveContentType.TEN_X_MTX, (
            f"V3 format (features.tsv) should be detected as TEN_X_MTX, got {result}"
        )

    def test_v3_format_detection_compressed(self, v3_10x_directory_compressed: Path):
        """V3 format with .gz compression is correctly identified."""
        result = ContentDetector.detect_content_type(v3_10x_directory_compressed)
        assert result == ArchiveContentType.TEN_X_MTX, (
            f"V3 compressed format should be detected as TEN_X_MTX, got {result}"
        )

    # --- V2 Format Tests ---

    def test_v2_format_detection_uncompressed(self, v2_10x_directory: Path):
        """V2 format (genes.tsv) is correctly identified."""
        result = ContentDetector.detect_content_type(v2_10x_directory)
        assert result == ArchiveContentType.TEN_X_MTX, (
            f"V2 format (genes.tsv) should be detected as TEN_X_MTX, got {result}"
        )

    def test_v2_format_detection_compressed(self, v2_10x_directory_compressed: Path):
        """V2 format with .gz compression is correctly identified."""
        result = ContentDetector.detect_content_type(v2_10x_directory_compressed)
        assert result == ArchiveContentType.TEN_X_MTX, (
            f"V2 compressed format should be detected as TEN_X_MTX, got {result}"
        )

    # --- Nested Directory Tests ---

    def test_nested_v2_geo_pattern(self, tmp_path: Path):
        """V2 format in nested GEO directory structure (common pattern)."""
        # Common GEO pattern: GSM123456_sample/filtered_feature_bc_matrix/
        nested_dir = tmp_path / "GSM123456_sample" / "filtered_feature_bc_matrix"
        nested_dir.mkdir(parents=True)
        (nested_dir / "matrix.mtx.gz").write_bytes(b"mock")
        (nested_dir / "genes.tsv.gz").write_bytes(b"mock")
        (nested_dir / "barcodes.tsv.gz").write_bytes(b"mock")

        result = ContentDetector.detect_content_type(tmp_path)
        assert result == ArchiveContentType.TEN_X_MTX, (
            f"Nested V2 GEO pattern should be detected, got {result}"
        )

    def test_nested_v3_sample_pattern(self, tmp_path: Path):
        """V3 format in nested sample directory structure."""
        nested_dir = tmp_path / "sample" / "filtered_feature_bc_matrix"
        nested_dir.mkdir(parents=True)
        (nested_dir / "matrix.mtx.gz").write_bytes(b"mock")
        (nested_dir / "features.tsv.gz").write_bytes(b"mock")
        (nested_dir / "barcodes.tsv.gz").write_bytes(b"mock")

        result = ContentDetector.detect_content_type(tmp_path)
        assert result == ArchiveContentType.TEN_X_MTX, (
            f"Nested V3 sample pattern should be detected, got {result}"
        )

    # --- Incomplete 10X Tests ---

    def test_incomplete_10x_missing_matrix(self, tmp_path: Path):
        """Incomplete 10X without matrix file is NOT detected as 10X."""
        mtx_dir = tmp_path / "incomplete"
        mtx_dir.mkdir()
        (mtx_dir / "features.tsv.gz").write_bytes(b"mock")
        (mtx_dir / "barcodes.tsv.gz").write_bytes(b"mock")
        # matrix.mtx is missing

        result = ContentDetector.detect_content_type(tmp_path)
        assert result != ArchiveContentType.TEN_X_MTX, (
            f"Incomplete 10X (missing matrix) should NOT be TEN_X_MTX, got {result}"
        )

    def test_incomplete_10x_missing_features(self, tmp_path: Path):
        """Incomplete 10X without features/genes file is NOT detected as 10X."""
        mtx_dir = tmp_path / "incomplete"
        mtx_dir.mkdir()
        (mtx_dir / "matrix.mtx.gz").write_bytes(b"mock")
        (mtx_dir / "barcodes.tsv.gz").write_bytes(b"mock")
        # features.tsv or genes.tsv is missing

        result = ContentDetector.detect_content_type(tmp_path)
        assert result != ArchiveContentType.TEN_X_MTX, (
            f"Incomplete 10X (missing features) should NOT be TEN_X_MTX, got {result}"
        )

    def test_incomplete_10x_missing_barcodes(self, tmp_path: Path):
        """Incomplete 10X without barcodes file is NOT detected as 10X."""
        mtx_dir = tmp_path / "incomplete"
        mtx_dir.mkdir()
        (mtx_dir / "matrix.mtx.gz").write_bytes(b"mock")
        (mtx_dir / "features.tsv.gz").write_bytes(b"mock")
        # barcodes.tsv is missing

        result = ContentDetector.detect_content_type(tmp_path)
        assert result != ArchiveContentType.TEN_X_MTX, (
            f"Incomplete 10X (missing barcodes) should NOT be TEN_X_MTX, got {result}"
        )


class TestContentDetectorOtherFormats:
    """Tests for non-10X format detection."""

    @pytest.fixture
    def kallisto_directory(self, tmp_path: Path) -> Path:
        """Create Kallisto quantification directory structure."""
        for i in range(3):
            sample_dir = tmp_path / f"sample_{i}"
            sample_dir.mkdir()
            (sample_dir / "abundance.tsv").write_bytes(b"mock kallisto output")
            (sample_dir / "abundance.h5").write_bytes(b"mock h5")
        return tmp_path

    @pytest.fixture
    def salmon_directory(self, tmp_path: Path) -> Path:
        """Create Salmon quantification directory structure."""
        for i in range(3):
            sample_dir = tmp_path / f"sample_{i}"
            sample_dir.mkdir()
            (sample_dir / "quant.sf").write_bytes(b"mock salmon output")
        return tmp_path

    @pytest.fixture
    def geo_raw_directory(self, tmp_path: Path) -> Path:
        """Create GEO RAW expression directory structure."""
        (tmp_path / "GSM123456_sample1.txt.gz").write_bytes(b"mock geo")
        (tmp_path / "GSM123457_sample2.txt.gz").write_bytes(b"mock geo")
        (tmp_path / "GSM123458_sample3.txt").write_bytes(b"mock geo")
        return tmp_path

    @pytest.fixture
    def generic_expression_directory(self, tmp_path: Path) -> Path:
        """Create generic expression matrix directory."""
        # Create a large CSV file (>100KB) to pass size threshold
        large_content = b"gene,sample1,sample2\n" + b"GENE,1.0,2.0\n" * 10000
        (tmp_path / "expression_matrix.csv").write_bytes(large_content)
        (tmp_path / "sample_metadata.csv").write_bytes(b"sample,condition\n")
        return tmp_path

    def test_kallisto_quant_detection(self, kallisto_directory: Path):
        """Kallisto quantification files are correctly identified."""
        result = ContentDetector.detect_content_type(kallisto_directory)
        assert result == ArchiveContentType.KALLISTO_QUANT, (
            f"Kallisto directory should be detected as KALLISTO_QUANT, got {result}"
        )

    def test_salmon_quant_detection(self, salmon_directory: Path):
        """Salmon quantification files are correctly identified."""
        result = ContentDetector.detect_content_type(salmon_directory)
        assert result == ArchiveContentType.SALMON_QUANT, (
            f"Salmon directory should be detected as SALMON_QUANT, got {result}"
        )

    def test_geo_raw_detection(self, geo_raw_directory: Path):
        """GEO RAW files (GSM*.txt.gz) are correctly identified."""
        result = ContentDetector.detect_content_type(geo_raw_directory)
        assert result == ArchiveContentType.GEO_RAW, (
            f"GEO RAW directory should be detected as GEO_RAW, got {result}"
        )

    def test_generic_expression_detection(self, generic_expression_directory: Path):
        """Generic expression matrix (CSV >100KB) is correctly identified."""
        result = ContentDetector.detect_content_type(generic_expression_directory)
        assert result == ArchiveContentType.GENERIC_EXPRESSION, (
            f"Generic expression should be detected as GENERIC_EXPRESSION, got {result}"
        )

    def test_unknown_format_detection(self, tmp_path: Path):
        """Unknown file collection returns UNKNOWN."""
        (tmp_path / "random.xyz").write_bytes(b"unknown")
        (tmp_path / "another.abc").write_bytes(b"unknown")

        result = ContentDetector.detect_content_type(tmp_path)
        assert result == ArchiveContentType.UNKNOWN, (
            f"Unknown files should be detected as UNKNOWN, got {result}"
        )

    def test_empty_directory_detection(self, tmp_path: Path):
        """Empty directory returns UNKNOWN."""
        result = ContentDetector.detect_content_type(tmp_path)
        assert result == ArchiveContentType.UNKNOWN, (
            f"Empty directory should be detected as UNKNOWN, got {result}"
        )


class TestArchiveInspectorManifestDetection:
    """Tests for ArchiveInspector content type detection from manifest."""

    def test_manifest_v3_detection(self):
        """V3 format detected from manifest (features.tsv)."""
        manifest = {
            "filenames": [
                "sample/matrix.mtx.gz",
                "sample/features.tsv.gz",
                "sample/barcodes.tsv.gz",
            ],
            "has_subdirectories": True,
            "extensions": {".gz": 3},
        }

        inspector = ArchiveInspector()
        result = inspector.detect_content_type_from_manifest(manifest)
        assert result == ArchiveContentType.TEN_X_MTX, (
            f"V3 manifest should be detected as TEN_X_MTX, got {result}"
        )

    def test_manifest_v2_detection(self):
        """V2 format detected from manifest (genes.tsv)."""
        manifest = {
            "filenames": [
                "sample/matrix.mtx.gz",
                "sample/genes.tsv.gz",
                "sample/barcodes.tsv.gz",
            ],
            "has_subdirectories": True,
            "extensions": {".gz": 3},
        }

        inspector = ArchiveInspector()
        result = inspector.detect_content_type_from_manifest(manifest)
        assert result == ArchiveContentType.TEN_X_MTX, (
            f"V2 manifest should be detected as TEN_X_MTX, got {result}"
        )

    def test_manifest_kallisto_detection(self):
        """Kallisto format detected from manifest."""
        manifest = {
            "filenames": [
                "sample1/abundance.tsv",
                "sample1/abundance.h5",
                "sample2/abundance.tsv",
                "sample2/abundance.h5",
            ],
            "has_subdirectories": True,
            "extensions": {".tsv": 2, ".h5": 2},
        }

        inspector = ArchiveInspector()
        result = inspector.detect_content_type_from_manifest(manifest)
        assert result == ArchiveContentType.KALLISTO_QUANT, (
            f"Kallisto manifest should be detected as KALLISTO_QUANT, got {result}"
        )

    def test_manifest_salmon_detection(self):
        """Salmon format detected from manifest."""
        manifest = {
            "filenames": [
                "sample1/quant.sf",
                "sample2/quant.sf",
            ],
            "has_subdirectories": True,
            "extensions": {".sf": 2},
        }

        inspector = ArchiveInspector()
        result = inspector.detect_content_type_from_manifest(manifest)
        assert result == ArchiveContentType.SALMON_QUANT, (
            f"Salmon manifest should be detected as SALMON_QUANT, got {result}"
        )

    def test_manifest_geo_raw_detection(self):
        """GEO RAW format detected from manifest."""
        manifest = {
            "filenames": [
                "GSM123456_sample1.txt.gz",
                "GSM123457_sample2.txt.gz",
            ],
            "has_subdirectories": False,
            "extensions": {".gz": 2},
        }

        inspector = ArchiveInspector()
        result = inspector.detect_content_type_from_manifest(manifest)
        assert result == ArchiveContentType.GEO_RAW, (
            f"GEO RAW manifest should be detected as GEO_RAW, got {result}"
        )


class TestContentDetectorKallistoSalmon:
    """Tests for detect_kallisto_salmon() method."""

    def test_detect_kallisto_files(self):
        """Kallisto abundance files are detected."""
        file_paths = [
            "/path/sample1/abundance.tsv",
            "/path/sample1/abundance.h5",
            "/path/sample2/abundance.tsv",
            "/path/sample2/abundance.h5",
        ]

        has_quant, tool_type, matched, n_samples = (
            ContentDetector.detect_kallisto_salmon(file_paths)
        )

        assert has_quant is True
        assert tool_type == "kallisto"
        assert len(matched) == 4
        assert n_samples == 4  # 2 .tsv + 2 .h5

    def test_detect_salmon_files(self):
        """Salmon quant files are detected."""
        file_paths = [
            "/path/sample1/quant.sf",
            "/path/sample2/quant.sf",
        ]

        has_quant, tool_type, matched, n_samples = (
            ContentDetector.detect_kallisto_salmon(file_paths)
        )

        assert has_quant is True
        assert tool_type == "salmon"
        assert len(matched) == 2
        assert n_samples == 2

    def test_detect_mixed_quantification(self):
        """Mixed Kallisto and Salmon files are detected."""
        file_paths = [
            "/path/sample1/abundance.tsv",
            "/path/sample2/quant.sf",
        ]

        has_quant, tool_type, matched, n_samples = (
            ContentDetector.detect_kallisto_salmon(file_paths)
        )

        assert has_quant is True
        assert tool_type == "mixed"
        assert len(matched) == 2

    def test_no_quantification_files(self):
        """Non-quantification files return negative result."""
        file_paths = [
            "/path/random.csv",
            "/path/other.txt",
        ]

        has_quant, tool_type, matched, n_samples = (
            ContentDetector.detect_kallisto_salmon(file_paths)
        )

        assert has_quant is False
        assert tool_type == ""
        assert len(matched) == 0
        assert n_samples == 0


class TestArchiveInspectorNestedArchives:
    """Tests for nested archive detection."""

    def test_detect_nested_archives(self):
        """Nested .tar.gz archives within parent archive are detected."""
        manifest = {
            "filenames": [
                "GSM4710689_PDAC_TISSUE_1.tar.gz",
                "GSM4710690_PDAC_TISSUE_2.tar.gz",
                "GSM4710691_PDAC_PBMC_1.tar.gz",
            ],
            "has_subdirectories": False,
        }

        inspector = ArchiveInspector()
        result = inspector.detect_nested_archives(manifest, "/path/to/archive.tar")

        assert result is not None
        assert result.total_count == 3
        assert len(result.nested_archives) == 3
        assert "PDAC_TISSUE" in result.groups
        assert "PDAC_PBMC" in result.groups
        assert len(result.groups["PDAC_TISSUE"]) == 2
        assert len(result.groups["PDAC_PBMC"]) == 1

    def test_no_nested_archives(self):
        """Returns None when no nested archives found."""
        manifest = {
            "filenames": [
                "matrix.mtx.gz",
                "features.tsv.gz",
                "barcodes.tsv.gz",
            ],
            "has_subdirectories": False,
        }

        inspector = ArchiveInspector()
        result = inspector.detect_nested_archives(manifest)

        assert result is None

    def test_single_nested_archive(self):
        """Returns None for single nested archive (not worth special handling)."""
        manifest = {
            "filenames": [
                "GSM4710689_PDAC_TISSUE_1.tar.gz",
            ],
            "has_subdirectories": False,
        }

        inspector = ArchiveInspector()
        result = inspector.detect_nested_archives(manifest)

        assert result is None  # Need >= 2 for special handling


class TestArchiveExtractor:
    """Tests for ArchiveExtractor security and functionality."""

    def test_cleanup_removes_temp_dirs(self, tmp_path: Path):
        """Cleanup removes all tracked temporary directories."""
        extractor = ArchiveExtractor()
        temp_dir = tmp_path / "temp_test"
        temp_dir.mkdir()
        extractor.temp_dirs.append(temp_dir)

        assert temp_dir.exists()
        extractor.cleanup()
        assert not temp_dir.exists()
        assert len(extractor.temp_dirs) == 0
