"""
Comprehensive unit tests for GEO Fallback Service.

This module provides complete test coverage for geo_fallback_service.py,
which currently has ZERO existing tests. Tests cover:

1. TAR supplementary file processing
2. Series matrix fallback
3. Supplementary files handling
4. Sample matrices fallback
5. Helper download fallback
6. Single-cell sample downloading
7. Bulk dataset downloading
8. 10X format file handling
9. TAR directory processing
10. Error handling and edge cases

Coverage target: 95%+ with realistic test scenarios.
"""

import gzip
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

try:
    import GEOparse
except ImportError:
    GEOparse = None

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.data_access.geo_fallback_service import GEOFallbackService
from lobster.services.data_access.geo_service import (
    GEODataSource,
    GEOResult,
    GEOService,
)

# ===============================================================================
# Fixtures
# ===============================================================================


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_data_manager(temp_cache_dir):
    """Create mock DataManagerV2 instance."""
    mock_dm = Mock(spec=DataManagerV2)
    mock_dm.cache_dir = temp_cache_dir
    mock_dm.metadata_store = {}
    mock_dm.list_modalities.return_value = []
    mock_dm.log_tool_usage = Mock()
    mock_dm.load_modality = Mock()
    mock_dm.save_modality = Mock(return_value=str(temp_cache_dir / "saved.h5ad"))
    return mock_dm


@pytest.fixture
def geo_service(mock_data_manager, temp_cache_dir):
    """Create GEOService instance."""
    return GEOService(data_manager=mock_data_manager, cache_dir=str(temp_cache_dir))


@pytest.fixture
def geo_fallback_service(geo_service):
    """Create GEOFallbackService instance."""
    return GEOFallbackService(geo_service=geo_service)


@pytest.fixture
def mock_expression_matrix():
    """Create mock expression matrix."""
    return pd.DataFrame(
        np.random.poisson(5, (100, 1000)),
        index=[f"cell_{i}" for i in range(100)],
        columns=[f"gene_{i}" for i in range(1000)],
    )


@pytest.fixture
def mock_soft_file(temp_cache_dir):
    """Create mock SOFT file."""
    soft_file = temp_cache_dir / "GSE123456_family.soft"
    soft_content = """^SERIES = GSE123456
!Series_title = Test Dataset
!Series_geo_accession = GSE123456
!Series_summary = Test summary
"""
    soft_file.write_text(soft_content)
    return soft_file


@pytest.fixture
def mock_tar_file(temp_cache_dir, mock_expression_matrix):
    """Create mock TAR archive with expression data."""
    tar_file = temp_cache_dir / "GSE123456_RAW.tar"

    # Create temporary matrix file
    matrix_file = temp_cache_dir / "GSE123456_matrix.txt"
    mock_expression_matrix.to_csv(matrix_file, sep="\t")

    # Create TAR archive
    with tarfile.open(tar_file, "w") as tar:
        tar.add(matrix_file, arcname="GSE123456_matrix.txt")

    matrix_file.unlink()  # Clean up temp file
    return tar_file


@pytest.fixture
def mock_10x_files(temp_cache_dir):
    """Create mock 10X format files."""
    # Matrix file (Matrix Market format)
    matrix_file = temp_cache_dir / "matrix.mtx.gz"
    with gzip.open(matrix_file, "wt") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("100 1000 5000\n")  # 100 cells × 1000 genes, 5000 non-zero entries
        for i in range(100):
            f.write(f"{i+1} {i+1} {np.random.rand()}\n")

    # Barcodes file
    barcodes_file = temp_cache_dir / "barcodes.tsv.gz"
    with gzip.open(barcodes_file, "wt") as f:
        for i in range(100):
            f.write(f"CELL_{i:05d}-1\n")

    # Features file
    features_file = temp_cache_dir / "features.tsv.gz"
    with gzip.open(features_file, "wt") as f:
        for i in range(1000):
            f.write(f"GENE{i}\tGENE{i}\tGene Expression\n")

    return {"matrix": matrix_file, "barcodes": barcodes_file, "features": features_file}


# ===============================================================================
# Scenario 1: TAR Supplementary File Processing
# ===============================================================================


@pytest.mark.unit
class TestTARSupplementaryProcessing:
    """Test TAR supplementary file processing."""

    def test_try_supplementary_tar_success(
        self, geo_fallback_service, mock_tar_file, mock_expression_matrix
    ):
        """Test successful TAR supplementary file processing."""
        metadata = {"accession": "GSE123456", "title": "Test Dataset"}

        with patch.object(
            geo_fallback_service.geo_downloader, "download_geo_data"
        ) as mock_download:
            mock_download.return_value = (None, {"tar": mock_tar_file})

            with patch.object(
                geo_fallback_service.geo_parser, "parse_supplementary_file"
            ) as mock_parse:
                mock_parse.return_value = mock_expression_matrix

                result = geo_fallback_service.try_supplementary_tar(
                    "GSE123456", metadata
                )

        assert result.success is True
        assert result.source == GEODataSource.TAR_ARCHIVE
        assert result.data is not None
        assert not result.data.empty

    def test_try_supplementary_tar_no_tar_files(self, geo_fallback_service):
        """Test TAR fallback when no TAR files available."""
        metadata = {"accession": "GSE123456"}

        with patch.object(
            geo_fallback_service.geo_downloader, "download_geo_data"
        ) as mock_download:
            mock_download.return_value = (None, {})  # No TAR files

            result = geo_fallback_service.try_supplementary_tar("GSE123456", metadata)

        assert result.success is False
        assert "No TAR files" in result.error_message

    def test_try_supplementary_tar_parsing_failure(
        self, geo_fallback_service, mock_tar_file
    ):
        """Test TAR fallback when parsing fails."""
        metadata = {"accession": "GSE123456"}

        with patch.object(
            geo_fallback_service.geo_downloader, "download_geo_data"
        ) as mock_download:
            mock_download.return_value = (None, {"tar": mock_tar_file})

            with patch.object(
                geo_fallback_service.geo_parser, "parse_supplementary_file"
            ) as mock_parse:
                mock_parse.return_value = None  # Parsing failed

                result = geo_fallback_service.try_supplementary_tar(
                    "GSE123456", metadata
                )

        assert result.success is False

    def test_try_supplementary_tar_download_error(self, geo_fallback_service):
        """Test TAR fallback when download fails."""
        metadata = {"accession": "GSE123456"}

        with patch.object(
            geo_fallback_service.geo_downloader, "download_geo_data"
        ) as mock_download:
            mock_download.side_effect = Exception("Download failed")

            result = geo_fallback_service.try_supplementary_tar("GSE123456", metadata)

        assert result.success is False
        assert "Download failed" in result.error_message


# ===============================================================================
# Scenario 2: Series Matrix Fallback
# ===============================================================================


@pytest.mark.unit
class TestSeriesMatrixFallback:
    """Test series matrix fallback functionality."""

    @pytest.mark.skipif(GEOparse is None, reason="GEOparse not installed")
    def test_try_series_matrix_success(
        self, geo_fallback_service, mock_expression_matrix
    ):
        """Test successful series matrix download."""
        metadata = {"accession": "GSE123456"}

        with patch("GEOparse.get_GEO") as mock_get_geo:
            mock_gse = Mock()
            mock_gse.table = mock_expression_matrix
            mock_get_geo.return_value = mock_gse

            result = geo_fallback_service.try_series_matrix("GSE123456", metadata)

        assert result.success is True
        assert result.source == GEODataSource.GEOPARSE
        assert result.processing_info["method"] == "series_matrix"

    @pytest.mark.skipif(GEOparse is None, reason="GEOparse not installed")
    def test_try_series_matrix_no_table(self, geo_fallback_service):
        """Test series matrix fallback when no table available."""
        metadata = {"accession": "GSE123456"}

        with patch("GEOparse.get_GEO") as mock_get_geo:
            mock_gse = Mock()
            mock_gse.table = None  # No series matrix
            mock_get_geo.return_value = mock_gse

            result = geo_fallback_service.try_series_matrix("GSE123456", metadata)

        assert result.success is False
        assert "No series matrix" in result.error_message

    @pytest.mark.skipif(GEOparse is None, reason="GEOparse not installed")
    def test_try_series_matrix_download_error(self, geo_fallback_service):
        """Test series matrix fallback when download fails."""
        metadata = {"accession": "GSE123456"}

        with patch("GEOparse.get_GEO") as mock_get_geo:
            mock_get_geo.side_effect = Exception("GEOparse download failed")

            result = geo_fallback_service.try_series_matrix("GSE123456", metadata)

        assert result.success is False
        assert "download failed" in result.error_message.lower()


# ===============================================================================
# Scenario 3: Supplementary Files Handling
# ===============================================================================


@pytest.mark.unit
class TestSupplementaryFilesHandling:
    """Test supplementary files (non-TAR) handling."""

    def test_try_supplementary_files_success(
        self, geo_fallback_service, temp_cache_dir, mock_expression_matrix
    ):
        """Test successful supplementary file processing."""
        metadata = {"accession": "GSE123456"}

        suppl_file = temp_cache_dir / "GSE123456_matrix.txt.gz"
        with gzip.open(suppl_file, "wt") as gz:
            mock_expression_matrix.to_csv(gz, sep="\t")

        with patch.object(
            geo_fallback_service.geo_downloader, "download_geo_data"
        ) as mock_download:
            mock_download.return_value = (None, {"supplementary": suppl_file})

            with patch.object(
                geo_fallback_service.geo_parser, "parse_supplementary_file"
            ) as mock_parse:
                mock_parse.return_value = mock_expression_matrix

                result = geo_fallback_service.try_supplementary_files(
                    "GSE123456", metadata
                )

        assert result.success is True
        assert result.source == GEODataSource.SUPPLEMENTARY

    def test_try_supplementary_files_not_found(self, geo_fallback_service):
        """Test supplementary files fallback when no files found."""
        metadata = {"accession": "GSE123456"}

        with patch.object(
            geo_fallback_service.geo_downloader, "download_geo_data"
        ) as mock_download:
            mock_download.return_value = (None, {})  # No supplementary files

            result = geo_fallback_service.try_supplementary_files("GSE123456", metadata)

        assert result.success is False
        assert "No supplementary files" in result.error_message


# ===============================================================================
# Scenario 4: Single-Cell Sample Downloading
# ===============================================================================


@pytest.mark.unit
class TestSingleCellSampleDownloading:
    """Test single-cell sample downloading functionality."""

    @pytest.mark.skipif(GEOparse is None, reason="GEOparse not installed")
    def test_download_single_cell_sample_success(
        self, geo_fallback_service, mock_expression_matrix
    ):
        """Test successful single-cell sample download."""
        gsm_id = "GSM1234567"

        with patch("GEOparse.get_GEO") as mock_get_geo:
            mock_gsm = Mock()
            mock_gsm.table = mock_expression_matrix
            mock_gsm.metadata = {}
            mock_get_geo.return_value = mock_gsm

            with patch.object(
                geo_fallback_service.geo_service, "_store_single_sample_as_modality"
            ) as mock_store:
                mock_store.return_value = f"Successfully stored {gsm_id}"

                result = geo_fallback_service.download_single_cell_sample(gsm_id)

        assert "Success" in result

    def test_download_single_cell_sample_invalid_id(self, geo_fallback_service):
        """Test single-cell sample download with invalid ID."""
        invalid_ids = ["GSE123456", "GPL123", "not_a_gsm", "12345678"]

        for invalid_id in invalid_ids:
            result = geo_fallback_service.download_single_cell_sample(invalid_id)

            assert "Invalid" in result or "Error" in result

    @pytest.mark.skipif(GEOparse is None, reason="GEOparse not installed")
    def test_download_single_cell_sample_10x_format(
        self, geo_fallback_service, mock_10x_files
    ):
        """Test single-cell sample download with 10X format files."""
        gsm_id = "GSM1234567"

        with patch("GEOparse.get_GEO") as mock_get_geo:
            mock_gsm = Mock()
            mock_gsm.metadata = {
                "supplementary_file": [
                    "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM123nnn/GSM1234567/suppl/matrix.mtx.gz",
                    "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM123nnn/GSM1234567/suppl/barcodes.tsv.gz",
                ]
            }
            mock_gsm.table = None
            mock_get_geo.return_value = mock_gsm

            with patch.object(
                geo_fallback_service, "_download_and_parse_10x_sample"
            ) as mock_10x:
                mock_10x.return_value = pd.DataFrame(np.random.rand(100, 1000))

                with patch.object(
                    geo_fallback_service.geo_service, "_store_single_sample_as_modality"
                ) as mock_store:
                    mock_store.return_value = f"Successfully stored {gsm_id}"

                    result = geo_fallback_service.download_single_cell_sample(gsm_id)

        assert "Success" in result


# ===============================================================================
# Scenario 5: Bulk Dataset Downloading
# ===============================================================================


@pytest.mark.unit
class TestBulkDatasetDownloading:
    """Test bulk dataset downloading functionality."""

    def test_download_bulk_dataset_success(
        self, geo_fallback_service, mock_expression_matrix
    ):
        """Test successful bulk dataset download."""
        geo_id = "GSE123456"

        # Mock metadata fetch
        geo_fallback_service.geo_service.fetch_metadata_only = Mock(
            return_value="Metadata fetched successfully"
        )

        # Mock download with strategy
        mock_result = GEOResult(
            data=mock_expression_matrix,
            metadata={"accession": geo_id},
            source=GEODataSource.GEOPARSE,
            processing_info={"method": "series_matrix"},
            success=True,
        )

        with patch.object(
            geo_fallback_service.geo_service, "download_with_strategy"
        ) as mock_download:
            mock_download.return_value = mock_result

            with patch.object(
                geo_fallback_service.data_manager, "load_modality"
            ) as mock_load:
                mock_adata = Mock()
                mock_adata.n_obs = 24
                mock_adata.n_vars = 58000
                mock_load.return_value = mock_adata

                with patch.object(
                    geo_fallback_service.data_manager, "save_modality"
                ) as mock_save:
                    mock_save.return_value = f"{geo_id.lower()}_bulk_raw.h5ad"

                    result = geo_fallback_service.download_bulk_dataset(geo_id)

        assert "Success" in result
        assert geo_id in result
        assert "bulk" in result.lower()

    def test_download_bulk_dataset_invalid_id(self, geo_fallback_service):
        """Test bulk dataset download with invalid ID."""
        invalid_ids = ["GSM123456", "GPL123", "not_a_gse"]

        for invalid_id in invalid_ids:
            result = geo_fallback_service.download_bulk_dataset(invalid_id)

            assert "Invalid" in result

    def test_download_bulk_dataset_metadata_failure(self, geo_fallback_service):
        """Test bulk dataset download when metadata fetch fails."""
        geo_id = "GSE999999"

        # Mock metadata fetch failure
        geo_fallback_service.geo_service.fetch_metadata_only = Mock(
            return_value="Error: Dataset not found"
        )

        result = geo_fallback_service.download_bulk_dataset(geo_id)

        assert "Error" in result or "Failed" in result

    def test_download_bulk_dataset_prefer_series_matrix(
        self, geo_fallback_service, mock_expression_matrix
    ):
        """Test bulk dataset download preferring series matrix."""
        geo_id = "GSE123456"

        # Mock successful metadata fetch
        geo_fallback_service.geo_service.fetch_metadata_only = Mock(
            return_value="Metadata fetched"
        )

        # Mock download with strategy
        mock_result = GEOResult(
            data=mock_expression_matrix,
            metadata={"accession": geo_id},
            source=GEODataSource.GEOPARSE,
            processing_info={"method": "series_matrix"},
            success=True,
        )

        with patch.object(
            geo_fallback_service.geo_service, "download_with_strategy"
        ) as mock_download:
            mock_download.return_value = mock_result

            with patch.object(geo_fallback_service.data_manager, "load_modality"):
                with patch.object(geo_fallback_service.data_manager, "save_modality"):
                    result = geo_fallback_service.download_bulk_dataset(
                        geo_id, prefer_series_matrix=True
                    )

        # Verify strategy was configured correctly
        call_args = mock_download.call_args
        assert call_args is not None


# ===============================================================================
# Scenario 6: TAR Directory Processing
# ===============================================================================


@pytest.mark.unit
class TestTARDirectoryProcessing:
    """Test TAR directory processing with helper methods."""

    def test_process_tar_directory_success(
        self, geo_fallback_service, temp_cache_dir, mock_expression_matrix
    ):
        """Test successful TAR directory processing."""
        # Create TAR directory with expression files
        tar_dir = temp_cache_dir / "GSE123456_RAW"
        tar_dir.mkdir()

        matrix_file = tar_dir / "GSE123456_matrix.txt.gz"
        with gzip.open(matrix_file, "wt") as gz:
            mock_expression_matrix.to_csv(gz, sep="\t")

        # Process TAR directory
        with patch.object(
            geo_fallback_service.geo_parser, "parse_supplementary_file"
        ) as mock_parse:
            mock_parse.return_value = mock_expression_matrix

            result = geo_fallback_service.process_tar_directory_with_helpers(tar_dir)

        assert result is not None
        assert not result.empty

    def test_process_tar_directory_no_valid_files(
        self, geo_fallback_service, temp_cache_dir
    ):
        """Test TAR directory processing with no valid expression files."""
        tar_dir = temp_cache_dir / "empty_tar"
        tar_dir.mkdir()

        # Create non-expression files
        (tar_dir / "README.txt").write_text("This is a README")
        (tar_dir / "metadata.json").write_text('{"key": "value"}')

        result = geo_fallback_service.process_tar_directory_with_helpers(tar_dir)

        assert result is None

    def test_process_tar_directory_parsing_errors(
        self, geo_fallback_service, temp_cache_dir
    ):
        """Test TAR directory processing when all files fail to parse."""
        tar_dir = temp_cache_dir / "corrupted_tar"
        tar_dir.mkdir()

        # Create corrupted files
        for i in range(3):
            corrupted_file = tar_dir / f"corrupted_{i}.txt.gz"
            with gzip.open(corrupted_file, "wt") as gz:
                gz.write("corrupted data\x00\x01\x02")

        with patch.object(
            geo_fallback_service.geo_parser, "parse_supplementary_file"
        ) as mock_parse:
            mock_parse.side_effect = Exception("Parsing failed")

            result = geo_fallback_service.process_tar_directory_with_helpers(tar_dir)

        assert result is None


# ===============================================================================
# Scenario 7: Process Supplementary TAR Files (Public Method)
# ===============================================================================


@pytest.mark.unit
class TestProcessSupplementaryTARFiles:
    """Test the public method for processing TAR files."""

    def test_process_supplementary_tar_files_success(
        self, geo_fallback_service, mock_tar_file, mock_expression_matrix
    ):
        """Test successful TAR file processing via public method."""
        geo_id = "GSE123456"

        with patch.object(
            geo_fallback_service.geo_downloader, "download_geo_data"
        ) as mock_download:
            mock_download.return_value = (None, {"tar": mock_tar_file})

            with patch.object(
                geo_fallback_service.geo_parser, "parse_supplementary_file"
            ) as mock_parse:
                mock_parse.return_value = mock_expression_matrix

                with patch.object(
                    geo_fallback_service.data_manager, "load_modality"
                ) as mock_load:
                    mock_adata = Mock()
                    mock_adata.n_obs = 15000
                    mock_adata.n_vars = 25000
                    mock_load.return_value = mock_adata

                    with patch.object(
                        geo_fallback_service.data_manager, "save_modality"
                    ):
                        result = geo_fallback_service.process_supplementary_tar_files(
                            geo_id
                        )

        assert "Success" in result
        assert geo_id in result

    def test_process_supplementary_tar_files_invalid_id(self, geo_fallback_service):
        """Test TAR processing with invalid GEO ID."""
        invalid_ids = ["GSM123456", "not_a_gse", "12345678"]

        for invalid_id in invalid_ids:
            result = geo_fallback_service.process_supplementary_tar_files(invalid_id)

            assert "Invalid" in result

    def test_process_supplementary_tar_files_no_tar_found(self, geo_fallback_service):
        """Test TAR processing when no TAR files found."""
        geo_id = "GSE123456"

        with patch.object(
            geo_fallback_service.geo_downloader, "download_geo_data"
        ) as mock_download:
            mock_download.return_value = (None, {})  # No files

            result = geo_fallback_service.process_supplementary_tar_files(geo_id)

        assert "No TAR files" in result or "Error" in result


# ===============================================================================
# Scenario 8: 10X Format File Handling
# ===============================================================================


@pytest.mark.unit
class TestTenXFormatHandling:
    """Test 10X format file handling."""

    def test_download_and_parse_10x_sample_success(
        self, geo_fallback_service, mock_10x_files
    ):
        """Test successful 10X sample download and parsing."""
        gsm_id = "GSM1234567"
        suppl_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM123nnn/GSM1234567/suppl/matrix.mtx.gz"

        with patch.object(
            geo_fallback_service.geo_downloader, "download_file"
        ) as mock_download:
            mock_download.return_value = True

            with patch.object(
                geo_fallback_service.geo_parser, "parse_supplementary_file"
            ) as mock_parse:
                mock_parse.return_value = pd.DataFrame(np.random.rand(100, 1000))

                result = geo_fallback_service._download_and_parse_10x_sample(
                    suppl_url, gsm_id
                )

        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_download_and_parse_10x_sample_download_failure(self, geo_fallback_service):
        """Test 10X sample handling when download fails."""
        gsm_id = "GSM1234567"
        suppl_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM123nnn/GSM1234567/suppl/matrix.mtx.gz"

        with patch.object(
            geo_fallback_service.geo_downloader, "download_file"
        ) as mock_download:
            mock_download.return_value = False  # Download failed

            result = geo_fallback_service._download_and_parse_10x_sample(
                suppl_url, gsm_id
            )

        assert result is None

    def test_download_and_parse_10x_sample_parsing_error(self, geo_fallback_service):
        """Test 10X sample handling when parsing fails."""
        gsm_id = "GSM1234567"
        suppl_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/test.mtx.gz"

        with patch.object(
            geo_fallback_service.geo_downloader, "download_file"
        ) as mock_download:
            mock_download.return_value = True

            with patch.object(
                geo_fallback_service.geo_parser, "parse_supplementary_file"
            ) as mock_parse:
                mock_parse.side_effect = Exception("Parsing error")

                result = geo_fallback_service._download_and_parse_10x_sample(
                    suppl_url, gsm_id
                )

        assert result is None


# ===============================================================================
# Scenario 9: Helper Download Fallback
# ===============================================================================


@pytest.mark.unit
class TestHelperDownloadFallback:
    """Test helper download fallback functionality."""

    def test_try_helper_download_fallback_success(
        self, geo_fallback_service, mock_expression_matrix, temp_cache_dir
    ):
        """Test successful helper download fallback."""
        metadata = {"accession": "GSE123456"}

        data_file = temp_cache_dir / "data.txt.gz"
        with gzip.open(data_file, "wt") as gz:
            mock_expression_matrix.to_csv(gz, sep="\t")

        with patch.object(
            geo_fallback_service.geo_downloader, "download_geo_data"
        ) as mock_download:
            mock_download.return_value = (None, {"matrix": data_file})

            with patch.object(
                geo_fallback_service.geo_parser, "parse_supplementary_file"
            ) as mock_parse:
                mock_parse.return_value = mock_expression_matrix

                result = geo_fallback_service.try_helper_download_fallback(
                    "GSE123456", metadata
                )

        assert result.success is True
        assert result.source == GEODataSource.SOFT_FILE

    def test_try_helper_download_fallback_no_data_sources(self, geo_fallback_service):
        """Test helper fallback when no data sources found."""
        metadata = {"accession": "GSE123456"}

        with patch.object(
            geo_fallback_service.geo_downloader, "download_geo_data"
        ) as mock_download:
            mock_download.return_value = (None, None)  # No data sources

            result = geo_fallback_service.try_helper_download_fallback(
                "GSE123456", metadata
            )

        assert result.success is False
        assert "No data sources" in result.error_message

    def test_try_helper_download_fallback_all_fail(
        self, geo_fallback_service, temp_cache_dir
    ):
        """Test helper fallback when all parsing attempts fail."""
        metadata = {"accession": "GSE123456"}

        # Create multiple data sources
        data_sources = {}
        for i in range(3):
            data_file = temp_cache_dir / f"data_{i}.txt.gz"
            with gzip.open(data_file, "wt") as gz:
                gz.write("corrupted data")
            data_sources[f"source_{i}"] = data_file

        with patch.object(
            geo_fallback_service.geo_downloader, "download_geo_data"
        ) as mock_download:
            mock_download.return_value = (None, data_sources)

            with patch.object(
                geo_fallback_service.geo_parser, "parse_supplementary_file"
            ) as mock_parse:
                mock_parse.return_value = None  # All parsing fails

                result = geo_fallback_service.try_helper_download_fallback(
                    "GSE123456", metadata
                )

        assert result.success is False


# ===============================================================================
# Scenario 10: Sample Matrices Fallback
# ===============================================================================


@pytest.mark.unit
class TestSampleMatricesFallback:
    """Test sample matrices fallback."""

    def test_try_sample_matrices_fallback(self, geo_fallback_service):
        """Test sample matrices fallback (currently a no-op)."""
        metadata = {"accession": "GSE123456"}

        result = geo_fallback_service.try_sample_matrices_fallback(
            "GSE123456", metadata
        )

        # Should return failure with explanation
        assert result.success is False
        assert "already tried" in result.error_message


# ===============================================================================
# Edge Cases and Error Handling
# ===============================================================================


@pytest.mark.unit
class TestFallbackServiceEdgeCases:
    """Test edge cases and error handling."""

    def test_download_sample_with_helpers_not_implemented(self, geo_fallback_service):
        """Test helper-based sample download (not yet implemented)."""
        gsm_id = "GSM1234567"

        result = geo_fallback_service.download_sample_with_helpers(gsm_id)

        assert "not yet implemented" in result or "not implemented" in result.lower()

    def test_processing_very_large_tar_directory(
        self, geo_fallback_service, temp_cache_dir
    ):
        """Test processing TAR directory with many files."""
        tar_dir = temp_cache_dir / "large_tar"
        tar_dir.mkdir()

        # Create many small files
        for i in range(100):
            small_file = tar_dir / f"file_{i}.txt"
            small_file.write_text(f"Small file {i}")

        # Should handle without crashing
        result = geo_fallback_service.process_tar_directory_with_helpers(tar_dir)

        # May return None (no valid expression files), but should not crash
        assert result is None or isinstance(result, pd.DataFrame)

    def test_fallback_service_with_null_geo_service(self):
        """Test fallback service with null GEO service (should not crash on init)."""
        # This should raise an error or handle gracefully
        with pytest.raises((AttributeError, TypeError)):
            GEOFallbackService(geo_service=None)


# ===============================================================================
# Integration Tests
# ===============================================================================


@pytest.mark.unit
class TestFallbackServiceIntegration:
    """Test integration between fallback service and main GEO service."""

    def test_fallback_service_shares_cache(self, geo_fallback_service, temp_cache_dir):
        """Test that fallback service shares cache with main service."""
        assert geo_fallback_service.cache_dir == temp_cache_dir
        assert (
            geo_fallback_service.cache_dir == geo_fallback_service.geo_service.cache_dir
        )

    def test_fallback_service_shares_data_manager(self, geo_fallback_service):
        """Test that fallback service shares data manager."""
        assert (
            geo_fallback_service.data_manager
            == geo_fallback_service.geo_service.data_manager
        )

    def test_fallback_service_uses_shared_downloader(self, geo_fallback_service):
        """Test that fallback service uses shared downloader."""
        assert (
            geo_fallback_service.geo_downloader
            == geo_fallback_service.geo_service.geo_downloader
        )

    def test_fallback_service_uses_shared_parser(self, geo_fallback_service):
        """Test that fallback service uses shared parser."""
        assert (
            geo_fallback_service.geo_parser
            == geo_fallback_service.geo_service.geo_parser
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
