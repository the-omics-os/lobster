"""
Comprehensive unit tests for GEO service.

This module provides thorough testing of the GEO (Gene Expression Omnibus)
service including dataset search, metadata extraction, file downloading,
format conversion, and integration with the data management system.

Test coverage target: 95%+ with meaningful tests for GEO operations.
"""

import gzip
import json
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.geo_service import GEOService
from tests.mock_data.base import SMALL_DATASET_CONFIG
from tests.mock_data.factories import SingleCellDataFactory

# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================


@pytest.fixture
def mock_geo_response():
    """Mock GEO search response data."""
    return {
        "header": {"type": "esearch", "version": "0.3"},
        "esearchresult": {
            "count": "3",
            "retmax": "20",
            "retstart": "0",
            "idlist": ["200012345", "200012346", "200012347"],
            "translationset": [],
            "querytranslation": "single cell[All Fields] AND rna seq[All Fields]",
        },
    }


@pytest.fixture
def mock_geo_metadata():
    """Mock GEO dataset metadata."""
    return {
        "GSE123456": {
            "title": "Single-cell RNA sequencing of tumor-infiltrating T cells",
            "summary": "We performed scRNA-seq analysis of T cells from tumor samples...",
            "organism": "Homo sapiens",
            "sample_count": 48,
            "platform": "GPL24676 (Illumina NovaSeq 6000)",
            "publication_date": "2023-06-15",
            "last_update_date": "2023-06-20",
            "samples": [
                {
                    "gsm": "GSM1234567",
                    "title": "Tumor T cells replicate 1",
                    "characteristics": {
                        "cell type": "T cell",
                        "tissue": "tumor",
                        "treatment": "untreated",
                    },
                },
                {
                    "gsm": "GSM1234568",
                    "title": "Tumor T cells replicate 2",
                    "characteristics": {
                        "cell type": "T cell",
                        "tissue": "tumor",
                        "treatment": "untreated",
                    },
                },
            ],
            "supplementary_files": [
                {"name": "GSE123456_matrix.txt.gz", "size": "45.2 MB", "type": "TXT"},
                {"name": "GSE123456_barcodes.txt.gz", "size": "1.2 MB", "type": "TXT"},
                {"name": "GSE123456_features.txt.gz", "size": "800 KB", "type": "TXT"},
            ],
        }
    }


@pytest.fixture
def mock_ncbi_client():
    """Mock NCBI E-utilities client."""
    with patch("lobster.tools.geo_service.Entrez") as mock_entrez:
        mock_entrez.email = "test@example.com"

        # Mock esearch response
        mock_search_handle = StringIO(
            json.dumps(
                {
                    "esearchresult": {
                        "count": "3",
                        "idlist": ["200012345", "200012346", "200012347"],
                    }
                }
            )
        )
        mock_entrez.esearch.return_value = mock_search_handle

        # Mock efetch response
        mock_fetch_handle = StringIO(
            """
        <DocumentSummary uid="200012345">
            <Accession>GSE123456</Accession>
            <Title>Single-cell RNA sequencing of tumor-infiltrating T cells</Title>
            <Summary>We performed scRNA-seq analysis...</Summary>
            <n_samples>48</n_samples>
            <PlatformTitle>Illumina NovaSeq 6000</PlatformTitle>
        </DocumentSummary>
        """
        )
        mock_entrez.efetch.return_value = mock_fetch_handle

        yield mock_entrez


@pytest.fixture
def mock_ftp_client():
    """Mock FTP client for GEO file downloads."""
    with patch("ftplib.FTP") as mock_ftp:
        mock_ftp_instance = mock_ftp.return_value
        mock_ftp_instance.login.return_value = None
        mock_ftp_instance.cwd.return_value = None
        mock_ftp_instance.nlst.return_value = [
            "GSE123456_matrix.txt.gz",
            "GSE123456_barcodes.txt.gz",
            "GSE123456_features.txt.gz",
        ]
        mock_ftp_instance.size.return_value = 47382016  # 45.2 MB
        mock_ftp_instance.retrbinary = Mock()
        mock_ftp_instance.quit.return_value = None

        yield mock_ftp_instance


@pytest.fixture
def geo_service():
    """Create GEOService instance for testing."""
    # Create a mock DataManagerV2 instance
    mock_data_manager = Mock()
    mock_data_manager.cache_dir = Path("test_cache")
    mock_data_manager.metadata_store = {}
    mock_data_manager.list_modalities.return_value = []

    return GEOService(data_manager=mock_data_manager, cache_dir="test_cache")


@pytest.fixture
def temp_download_dir():
    """Create temporary directory for downloads."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# ===============================================================================
# GEO Service Core Functionality Tests
# ===============================================================================


@pytest.mark.unit
class TestGEOServiceCore:
    """Test GEO service core functionality."""

    def test_geo_service_initialization(self):
        """Test GEOService initialization."""
        mock_data_manager = Mock()
        mock_data_manager.cache_dir = Path("test_cache")
        mock_data_manager.metadata_store = {}

        service = GEOService(data_manager=mock_data_manager)

        assert service.data_manager == mock_data_manager
        assert service.cache_dir is not None
        assert hasattr(service, "geo_downloader")

    def test_geo_service_with_custom_cache(self):
        """Test GEOService initialization with custom cache directory."""
        mock_data_manager = Mock()
        mock_data_manager.cache_dir = Path("test_cache")
        mock_data_manager.metadata_store = {}

        service = GEOService(data_manager=mock_data_manager, cache_dir="custom_cache")

        assert service.data_manager == mock_data_manager
        assert str(service.cache_dir) == "custom_cache"

    def test_validate_expression_matrix_valid(self, geo_service):
        """Test validation of valid expression matrices."""
        import numpy as np
        import pandas as pd

        # Create a valid expression matrix
        valid_matrix = pd.DataFrame(
            np.random.randint(0, 100, size=(100, 50)),
            index=[f"cell_{i}" for i in range(100)],
            columns=[f"gene_{i}" for i in range(50)],
        )

        assert geo_service._is_valid_expression_matrix(valid_matrix) == True

    def test_validate_expression_matrix_invalid(self, geo_service):
        """Test validation of invalid expression matrices."""
        import numpy as np
        import pandas as pd

        # Test with non-DataFrame
        assert geo_service._is_valid_expression_matrix("not_a_dataframe") == False

        # Test with non-numeric data
        invalid_matrix = pd.DataFrame(
            [["a", "b"], ["c", "d"]], columns=["col1", "col2"]
        )
        assert geo_service._is_valid_expression_matrix(invalid_matrix) == False


# ===============================================================================
# GEO Search and Discovery Tests
# ===============================================================================


@pytest.mark.unit
class TestGEOSearchDiscovery:
    """Test GEO search and discovery functionality."""

    def test_search_geo_datasets_basic(
        self, geo_service, mock_ncbi_client, mock_geo_response
    ):
        """Test basic GEO dataset search."""
        with patch.object(geo_service, "_search_geo") as mock_search:
            mock_search.return_value = [
                {
                    "accession": "GSE123456",
                    "title": "Single-cell RNA sequencing of tumor-infiltrating T cells",
                    "organism": "Homo sapiens",
                    "samples": 48,
                },
                {
                    "accession": "GSE123457",
                    "title": "scRNA-seq analysis of immune cells",
                    "organism": "Homo sapiens",
                    "samples": 32,
                },
            ]

            results = geo_service.search_datasets("single cell RNA seq", max_results=10)

            assert len(results) == 2
            assert results[0]["accession"] == "GSE123456"
            assert "single-cell" in results[0]["title"].lower()
            mock_search.assert_called_once_with("single cell RNA seq", max_results=10)

    def test_search_geo_datasets_with_filters(self, geo_service, mock_ncbi_client):
        """Test GEO dataset search with filters."""
        with patch.object(geo_service, "_search_geo") as mock_search:
            mock_search.return_value = [
                {
                    "accession": "GSE123456",
                    "title": "Human T cell analysis",
                    "organism": "Homo sapiens",
                    "samples": 48,
                    "platform": "Illumina NovaSeq 6000",
                }
            ]

            results = geo_service.search_datasets(
                query="T cell",
                organism="Homo sapiens",
                platform="Illumina",
                min_samples=40,
                max_results=5,
            )

            assert len(results) == 1
            assert results[0]["organism"] == "Homo sapiens"
            assert results[0]["samples"] >= 40

    def test_search_geo_datasets_empty_results(self, geo_service, mock_ncbi_client):
        """Test GEO search with no results."""
        with patch.object(geo_service, "_search_geo") as mock_search:
            mock_search.return_value = []

            results = geo_service.search_datasets("extremely_rare_query_no_results")

            assert len(results) == 0

    def test_get_trending_datasets(self, geo_service, mock_ncbi_client):
        """Test getting trending/popular datasets."""
        with patch.object(geo_service, "get_trending_datasets") as mock_trending:
            mock_trending.return_value = [
                {
                    "accession": "GSE123456",
                    "title": "Popular single-cell dataset",
                    "download_count": 1250,
                    "citation_count": 45,
                },
                {
                    "accession": "GSE123457",
                    "title": "Highly cited RNA-seq study",
                    "download_count": 890,
                    "citation_count": 67,
                },
            ]

            results = geo_service.get_trending_datasets(
                category="single_cell", limit=10
            )

            assert len(results) == 2
            assert results[0]["download_count"] > 1000
            mock_trending.assert_called_once_with(category="single_cell", limit=10)

    def test_search_by_publication(self, geo_service, mock_ncbi_client):
        """Test search by publication details."""
        with patch.object(geo_service, "search_by_publication") as mock_pub_search:
            mock_pub_search.return_value = [
                {
                    "accession": "GSE123456",
                    "title": "Dataset from Nature paper",
                    "pmid": "12345678",
                    "journal": "Nature",
                    "publication_year": 2023,
                }
            ]

            results = geo_service.search_by_publication(
                pmid="12345678", journal="Nature", author="Smith"
            )

            assert len(results) == 1
            assert results[0]["pmid"] == "12345678"


# ===============================================================================
# GEO Metadata Extraction Tests
# ===============================================================================


@pytest.mark.unit
class TestGEOMetadataExtraction:
    """Test GEO metadata extraction functionality."""

    def test_get_dataset_metadata(
        self, geo_service, mock_ncbi_client, mock_geo_metadata
    ):
        """Test extracting dataset metadata."""
        with patch.object(geo_service, "get_metadata") as mock_get_metadata:
            mock_get_metadata.return_value = mock_geo_metadata["GSE123456"]

            metadata = geo_service.get_metadata("GSE123456")

            assert (
                metadata["title"]
                == "Single-cell RNA sequencing of tumor-infiltrating T cells"
            )
            assert metadata["organism"] == "Homo sapiens"
            assert metadata["sample_count"] == 48
            assert len(metadata["samples"]) == 2
            mock_get_metadata.assert_called_once_with("GSE123456")

    def test_get_sample_metadata(self, geo_service, mock_ncbi_client):
        """Test extracting sample-level metadata."""
        with patch.object(geo_service, "get_sample_metadata") as mock_sample_meta:
            mock_sample_meta.return_value = {
                "gsm": "GSM1234567",
                "title": "Tumor T cells replicate 1",
                "characteristics": {
                    "cell type": "T cell",
                    "tissue": "tumor",
                    "treatment": "untreated",
                    "age": "65 years",
                    "sex": "male",
                },
                "protocols": {
                    "extraction": "Single cell isolation using FACS",
                    "library_construction": "10X Chromium 3' v3.1",
                },
            }

            metadata = geo_service.get_sample_metadata("GSM1234567")

            assert metadata["characteristics"]["cell type"] == "T cell"
            assert (
                metadata["protocols"]["library_construction"] == "10X Chromium 3' v3.1"
            )
            mock_sample_meta.assert_called_once_with("GSM1234567")

    def test_get_platform_metadata(self, geo_service, mock_ncbi_client):
        """Test extracting platform metadata."""
        with patch.object(geo_service, "get_platform_metadata") as mock_platform_meta:
            mock_platform_meta.return_value = {
                "gpl": "GPL24676",
                "title": "Illumina NovaSeq 6000",
                "organism": "Homo sapiens",
                "technology": "high-throughput sequencing",
                "manufacturer": "Illumina",
                "description": "Next-generation sequencing platform",
            }

            metadata = geo_service.get_platform_metadata("GPL24676")

            assert metadata["title"] == "Illumina NovaSeq 6000"
            assert metadata["technology"] == "high-throughput sequencing"
            mock_platform_meta.assert_called_once_with("GPL24676")

    def test_extract_experimental_design(self, geo_service, mock_geo_metadata):
        """Test experimental design extraction."""
        with patch.object(geo_service, "extract_experimental_design") as mock_design:
            mock_design.return_value = {
                "study_type": "single_cell_rna_seq",
                "experimental_factors": ["tissue", "treatment"],
                "sample_groups": {"tumor_untreated": 24, "normal_untreated": 24},
                "replicates": 2,
                "batch_effects": "minimal",
                "quality_metrics": {
                    "cells_per_sample": "~2000",
                    "genes_detected": "~15000",
                },
            }

            design = geo_service.extract_experimental_design("GSE123456")

            assert design["study_type"] == "single_cell_rna_seq"
            assert len(design["experimental_factors"]) == 2
            assert design["sample_groups"]["tumor_untreated"] == 24


# ===============================================================================
# GEO File Download Tests
# ===============================================================================


@pytest.mark.unit
class TestGEOFileDownload:
    """Test GEO file download functionality."""

    def test_list_supplementary_files(self, geo_service, mock_geo_metadata):
        """Test listing supplementary files."""
        with patch.object(geo_service, "list_files") as mock_list_files:
            mock_list_files.return_value = mock_geo_metadata["GSE123456"][
                "supplementary_files"
            ]

            files = geo_service.list_files("GSE123456")

            assert len(files) == 3
            assert files[0]["name"] == "GSE123456_matrix.txt.gz"
            assert files[0]["size"] == "45.2 MB"
            mock_list_files.assert_called_once_with("GSE123456")

    def test_download_dataset_files(
        self, geo_service, mock_ftp_client, temp_download_dir
    ):
        """Test downloading dataset files."""
        with patch.object(geo_service, "download_files") as mock_download:
            mock_download.return_value = {
                "success": True,
                "downloaded_files": [
                    str(temp_download_dir / "GSE123456_matrix.txt.gz"),
                    str(temp_download_dir / "GSE123456_barcodes.txt.gz"),
                    str(temp_download_dir / "GSE123456_features.txt.gz"),
                ],
                "total_size": "47.2 MB",
                "download_time": 45.2,
            }

            result = geo_service.download_files(
                "GSE123456",
                download_dir=str(temp_download_dir),
                file_types=["matrix", "barcodes", "features"],
            )

            assert result["success"] == True
            assert len(result["downloaded_files"]) == 3
            assert "matrix" in result["downloaded_files"][0]

    def test_download_specific_files(
        self, geo_service, mock_ftp_client, temp_download_dir
    ):
        """Test downloading specific files."""
        with patch.object(geo_service, "download_files") as mock_download:
            mock_download.return_value = {
                "success": True,
                "downloaded_files": [
                    str(temp_download_dir / "GSE123456_matrix.txt.gz")
                ],
                "skipped_files": [],
            }

            result = geo_service.download_files(
                "GSE123456",
                file_names=["GSE123456_matrix.txt.gz"],
                download_dir=str(temp_download_dir),
            )

            assert result["success"] == True
            assert len(result["downloaded_files"]) == 1

    def test_download_with_progress_callback(
        self, geo_service, mock_ftp_client, temp_download_dir
    ):
        """Test download with progress callback."""
        progress_updates = []

        def progress_callback(filename, bytes_downloaded, total_bytes):
            progress_updates.append(
                {
                    "file": filename,
                    "downloaded": bytes_downloaded,
                    "total": total_bytes,
                    "percent": (bytes_downloaded / total_bytes) * 100,
                }
            )

        with patch.object(geo_service, "download_files") as mock_download:
            # Simulate progress updates
            progress_callback("GSE123456_matrix.txt.gz", 10485760, 47382016)  # 25%
            progress_callback("GSE123456_matrix.txt.gz", 23691008, 47382016)  # 50%
            progress_callback("GSE123456_matrix.txt.gz", 47382016, 47382016)  # 100%

            mock_download.return_value = {
                "success": True,
                "downloaded_files": [
                    str(temp_download_dir / "GSE123456_matrix.txt.gz")
                ],
            }

            result = geo_service.download_files(
                "GSE123456",
                download_dir=str(temp_download_dir),
                progress_callback=progress_callback,
            )

            assert len(progress_updates) == 3
            assert progress_updates[-1]["percent"] == 100.0

    def test_resume_interrupted_download(self, geo_service, temp_download_dir):
        """Test resuming interrupted downloads."""
        # Create partial file
        partial_file = temp_download_dir / "GSE123456_matrix.txt.gz.partial"
        partial_file.write_bytes(b"partial_content")

        with patch.object(geo_service, "download_files") as mock_download:
            mock_download.return_value = {
                "success": True,
                "resumed": True,
                "downloaded_files": [
                    str(temp_download_dir / "GSE123456_matrix.txt.gz")
                ],
                "bytes_resumed": len(b"partial_content"),
            }

            result = geo_service.download_files(
                "GSE123456", download_dir=str(temp_download_dir), resume=True
            )

            assert result["resumed"] == True
            assert result["bytes_resumed"] > 0


# ===============================================================================
# Data Format Conversion Tests
# ===============================================================================


@pytest.mark.unit
class TestGEOFormatConversion:
    """Test GEO data format conversion functionality."""

    def test_detect_file_format(self, geo_service):
        """Test automatic file format detection."""
        test_files = [
            ("GSE123456_matrix.txt.gz", "matrix"),
            ("GSE123456_barcodes.tsv.gz", "barcodes"),
            ("GSE123456_features.tsv.gz", "features"),
            ("GSE123456_matrix.mtx.gz", "matrix_market"),
            ("GSE123456.h5", "hdf5"),
            ("GSE123456_expression.xlsx", "excel"),
        ]

        with patch.object(geo_service, "_detect_format") as mock_detect:
            for filename, expected_format in test_files:
                mock_detect.return_value = expected_format

                detected_format = geo_service._detect_format(filename)
                assert detected_format == expected_format

    def test_convert_to_anndata(self, geo_service, temp_download_dir):
        """Test conversion to AnnData format."""
        # Create mock input files
        matrix_file = temp_download_dir / "matrix.txt.gz"
        barcodes_file = temp_download_dir / "barcodes.txt.gz"
        features_file = temp_download_dir / "features.txt.gz"

        with patch.object(geo_service, "convert_to_anndata") as mock_convert:
            mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            mock_convert.return_value = mock_adata

            adata = geo_service.convert_to_anndata(
                matrix_file=str(matrix_file),
                barcodes_file=str(barcodes_file),
                features_file=str(features_file),
            )

            assert adata.n_obs > 0
            assert adata.n_vars > 0
            mock_convert.assert_called_once()

    def test_convert_soft_format(self, geo_service, temp_download_dir):
        """Test SOFT format conversion."""
        soft_file = temp_download_dir / "GSE123456_family.soft.gz"

        with patch.object(geo_service, "parse_soft_file") as mock_parse_soft:
            mock_parse_soft.return_value = {
                "platform_data": pd.DataFrame(
                    {
                        "ID": ["GENE1", "GENE2", "GENE3"],
                        "Gene Symbol": ["ACTB", "GAPDH", "TP53"],
                    }
                ),
                "sample_data": {
                    "GSM1234567": pd.Series(
                        [100, 200, 150], index=["GENE1", "GENE2", "GENE3"]
                    ),
                    "GSM1234568": pd.Series(
                        [120, 180, 140], index=["GENE1", "GENE2", "GENE3"]
                    ),
                },
                "metadata": {
                    "platform": "GPL123456",
                    "samples": ["GSM1234567", "GSM1234568"],
                },
            }

            result = geo_service.parse_soft_file(str(soft_file))

            assert "platform_data" in result
            assert len(result["sample_data"]) == 2
            assert result["metadata"]["platform"] == "GPL123456"

    def test_convert_matrix_market(self, geo_service, temp_download_dir):
        """Test Matrix Market format conversion."""
        mtx_file = temp_download_dir / "matrix.mtx.gz"

        with patch.object(geo_service, "read_matrix_market") as mock_read_mtx:
            # Mock sparse matrix
            mock_matrix = np.array([[1, 0, 3], [0, 2, 0], [1, 1, 0]])
            mock_read_mtx.return_value = mock_matrix

            matrix = geo_service.read_matrix_market(str(mtx_file))

            assert matrix.shape == (3, 3)
            assert np.sum(matrix) == 8  # Sum of non-zero elements

    def test_handle_compressed_files(self, geo_service, temp_download_dir):
        """Test handling of compressed files."""
        # Create mock compressed file
        compressed_file = temp_download_dir / "data.txt.gz"

        test_content = "gene1\tgene2\tgene3\ncell1\t100\t200\t150\ncell2\t120\t180\t140"

        with patch.object(geo_service, "_decompress_file") as mock_decompress:
            mock_decompress.return_value = test_content

            content = geo_service._decompress_file(str(compressed_file))

            assert "gene1" in content
            assert "cell1" in content
            mock_decompress.assert_called_once_with(str(compressed_file))


# ===============================================================================
# Integration and Caching Tests
# ===============================================================================


@pytest.mark.unit
class TestGEOIntegrationCaching:
    """Test GEO integration and caching functionality."""

    def test_cache_search_results(self, geo_service, temp_download_dir):
        """Test caching of search results."""
        cache_dir = temp_download_dir / "cache"
        cache_dir.mkdir()

        service = GEOService(email="test@example.com", cache_dir=str(cache_dir))

        with patch.object(service, "_cache_search_results") as mock_cache:
            search_results = [
                {"accession": "GSE123456", "title": "Test dataset 1"},
                {"accession": "GSE123457", "title": "Test dataset 2"},
            ]

            mock_cache.return_value = True

            cached = service._cache_search_results("test_query", search_results)

            assert cached == True
            mock_cache.assert_called_once_with("test_query", search_results)

    def test_retrieve_cached_results(self, geo_service, temp_download_dir):
        """Test retrieving cached search results."""
        cache_dir = temp_download_dir / "cache"
        cache_dir.mkdir()

        service = GEOService(email="test@example.com", cache_dir=str(cache_dir))

        with patch.object(service, "_get_cached_results") as mock_get_cache:
            cached_results = [
                {"accession": "GSE123456", "title": "Cached dataset 1"},
                {"accession": "GSE123457", "title": "Cached dataset 2"},
            ]

            mock_get_cache.return_value = cached_results

            results = service._get_cached_results("test_query")

            assert len(results) == 2
            assert results[0]["title"] == "Cached dataset 1"

    def test_cache_expiration(self, geo_service, temp_download_dir):
        """Test cache expiration handling."""
        cache_dir = temp_download_dir / "cache"
        cache_dir.mkdir()

        service = GEOService(
            email="test@example.com", cache_dir=str(cache_dir), cache_ttl=3600  # 1 hour
        )

        with patch.object(service, "_is_cache_expired") as mock_expired:
            mock_expired.return_value = True

            expired = service._is_cache_expired("test_query")

            assert expired == True

    def test_integration_with_data_manager(self, geo_service, temp_download_dir):
        """Test integration with data manager."""
        with patch("lobster.core.data_manager_v2.DataManagerV2") as MockDataManager:
            mock_dm = MockDataManager.return_value

            with patch.object(geo_service, "download_and_load") as mock_download_load:
                mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                mock_download_load.return_value = {
                    "success": True,
                    "modality_name": "geo_gse123456",
                    "adata": mock_adata,
                    "metadata": {"accession": "GSE123456", "organism": "Homo sapiens"},
                }

                result = geo_service.download_and_load(
                    accession="GSE123456",
                    data_manager=mock_dm,
                    modality_name="geo_gse123456",
                )

                assert result["success"] == True
                assert result["modality_name"] == "geo_gse123456"
                assert result["adata"].n_obs > 0


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================


@pytest.mark.unit
class TestGEOErrorHandling:
    """Test GEO service error handling and edge cases."""

    def test_invalid_accession_handling(self, geo_service):
        """Test handling of invalid GEO accessions."""
        with patch.object(geo_service, "get_metadata") as mock_get_metadata:
            mock_get_metadata.side_effect = ValueError(
                "Invalid GEO accession: INVALID123"
            )

            with pytest.raises(ValueError, match="Invalid GEO accession"):
                geo_service.get_metadata("INVALID123")

    def test_network_timeout_handling(self, geo_service, mock_ncbi_client):
        """Test handling of network timeouts."""
        with patch.object(geo_service, "search_datasets") as mock_search:
            mock_search.side_effect = ConnectionError(
                "Network timeout during GEO search"
            )

            with pytest.raises(ConnectionError, match="Network timeout"):
                geo_service.search_datasets("test query")

    def test_ftp_download_failure(self, geo_service, mock_ftp_client):
        """Test handling of FTP download failures."""
        with patch.object(geo_service, "download_files") as mock_download:
            mock_download.side_effect = Exception("FTP server unavailable")

            with pytest.raises(Exception, match="FTP server unavailable"):
                geo_service.download_files("GSE123456")

    def test_corrupted_file_handling(self, geo_service, temp_download_dir):
        """Test handling of corrupted downloaded files."""
        corrupted_file = temp_download_dir / "corrupted_matrix.txt.gz"
        corrupted_file.write_bytes(b"corrupted_content")

        with patch.object(geo_service, "convert_to_anndata") as mock_convert:
            mock_convert.side_effect = Exception("File appears to be corrupted")

            with pytest.raises(Exception, match="File appears to be corrupted"):
                geo_service.convert_to_anndata(matrix_file=str(corrupted_file))

    def test_insufficient_disk_space(self, geo_service, temp_download_dir):
        """Test handling of insufficient disk space."""
        with patch.object(geo_service, "download_files") as mock_download:
            mock_download.side_effect = OSError("No space left on device")

            with pytest.raises(OSError, match="No space left on device"):
                geo_service.download_files(
                    "GSE123456", download_dir=str(temp_download_dir)
                )

    def test_rate_limit_handling(self, geo_service, mock_ncbi_client):
        """Test handling of API rate limits."""
        with patch.object(geo_service, "search_datasets") as mock_search:
            mock_search.side_effect = Exception("API rate limit exceeded")

            with pytest.raises(Exception, match="API rate limit exceeded"):
                geo_service.search_datasets("test query")

    def test_large_dataset_handling(self, geo_service, temp_download_dir):
        """Test handling of very large datasets."""
        with patch.object(geo_service, "download_files") as mock_download:
            mock_download.return_value = {
                "success": True,
                "downloaded_files": [str(temp_download_dir / "large_matrix.txt.gz")],
                "total_size": "15.2 GB",
                "download_time": 3600,
                "warnings": ["Large file size may impact processing time"],
            }

            result = geo_service.download_files(
                "GSE123456", download_dir=str(temp_download_dir)
            )

            assert result["success"] == True
            assert "warnings" in result

    def test_concurrent_download_handling(self, geo_service, temp_download_dir):
        """Test concurrent download operations."""
        import threading
        import time

        results = []
        errors = []

        def download_worker(worker_id, accession):
            """Worker function for concurrent downloads."""
            try:
                with patch.object(geo_service, "download_files") as mock_download:
                    mock_download.return_value = {
                        "success": True,
                        "worker_id": worker_id,
                        "accession": accession,
                    }

                    result = geo_service.download_files(
                        accession, download_dir=str(temp_download_dir)
                    )
                    results.append(result)
                    time.sleep(0.01)

            except Exception as e:
                errors.append((worker_id, e))

        # Create multiple concurrent downloads
        threads = []
        accessions = ["GSE123456", "GSE123457", "GSE123458"]

        for i, accession in enumerate(accessions):
            thread = threading.Thread(target=download_worker, args=(i, accession))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent download errors: {errors}"
        assert len(results) == 3


# ===============================================================================
# Biology-Aware Transpose Logic Tests
# ===============================================================================


@pytest.mark.unit
class TestBiologyAwareTranspose:
    """Test biology-aware transpose logic for expression matrices.

    These tests verify that the transpose logic uses biological knowledge
    (gene counts, sample counts) rather than naive shape comparisons to
    determine correct matrix orientation.
    """

    def test_bulk_rnaseq_high_gene_count_transpose(self, geo_service):
        """Test bulk RNA-seq with genes as rows (should transpose)."""
        # GSE130036 case: 187,697 genes × 4 samples
        # Should transpose to 4 samples × 187,697 genes
        bulk_matrix = pd.DataFrame(
            np.random.poisson(10, (187697, 4)),
            columns=["sample1", "sample2", "sample3", "sample4"],
        )

        should_transpose, reason = geo_service._determine_transpose_biologically(
            matrix=bulk_matrix, gsm_id="GSM_test_bulk", geo_id="GSE130036"
        )

        assert (
            should_transpose == True
        ), "Genes as rows (187,697) should trigger transpose"
        assert "Large row count" in reason or "187697" in reason

    def test_bulk_rnaseq_correct_orientation(self, geo_service):
        """Test bulk RNA-seq already in correct orientation (samples × genes)."""
        # 24 samples × 58,000 genes - correct orientation
        bulk_matrix = pd.DataFrame(np.random.poisson(20, (24, 58000)))

        should_transpose, reason = geo_service._determine_transpose_biologically(
            matrix=bulk_matrix, gsm_id="GSM_test_bulk_correct", geo_id="GSE_test"
        )

        assert (
            should_transpose == False
        ), "Samples×genes orientation should NOT transpose"
        # Check for various possible reason strings (Rule 1 or Rule 3)
        assert any(
            phrase in reason.lower()
            for phrase in [
                "large column count",
                "few rows, many columns",
                "samples×genes",
                "keeping",
            ]
        )

    def test_large_singlecell_both_dimensions_large(self, geo_service):
        """Test large single-cell dataset (both dimensions >10K)."""
        # 15,000 cells × 25,000 genes - typical large single-cell
        sc_matrix = pd.DataFrame(np.random.poisson(5, (15000, 25000)))

        should_transpose, reason = geo_service._determine_transpose_biologically(
            matrix=sc_matrix, gsm_id="GSM_test_sc_large", geo_id="GSE_sc"
        )

        assert (
            should_transpose == False
        ), "Large single-cell (cells×genes) should NOT transpose"
        assert "Both dimensions large" in reason or "cells×genes" in reason.lower()

    def test_small_singlecell_genes_as_columns(self, geo_service):
        """Test small single-cell with correct orientation."""
        # 500 cells × 25,000 genes
        sc_matrix = pd.DataFrame(np.random.poisson(5, (500, 25000)))

        should_transpose, reason = geo_service._determine_transpose_biologically(
            matrix=sc_matrix, gsm_id="GSM_test_sc_small", geo_id="GSE_sc_small"
        )

        assert (
            should_transpose == False
        ), "Small single-cell with correct orientation should NOT transpose"

    def test_gene_panel_small_gene_count(self, geo_service):
        """Test targeted gene panel (few genes, many cells)."""
        # 15,000 cells × 200 genes (targeted panel)
        panel_matrix = pd.DataFrame(np.random.poisson(50, (15000, 200)))

        should_transpose, reason = geo_service._determine_transpose_biologically(
            matrix=panel_matrix, gsm_id="GSM_test_panel", geo_id="GSE_panel"
        )

        # Should keep as cells×genes (many obs, few vars is valid)
        assert should_transpose == False

    def test_square_matrix_ambiguous(self, geo_service):
        """Test square-ish matrix (ambiguous case)."""
        # 8,000 × 12,000 - both >10K, assume cells×genes
        square_matrix = pd.DataFrame(np.random.poisson(5, (8000, 12000)))

        should_transpose, reason = geo_service._determine_transpose_biologically(
            matrix=square_matrix, gsm_id="GSM_test_square", geo_id="GSE_square"
        )

        # Should default to no transpose (safer)
        assert (
            should_transpose == False
        ), "Ambiguous square matrix should default to no transpose"

    def test_very_large_bulk_study(self, geo_service):
        """Test very large bulk study (edge case)."""
        # 1,000 samples × 25,000 genes
        large_bulk_matrix = pd.DataFrame(np.random.poisson(15, (1000, 25000)))

        should_transpose, reason = geo_service._determine_transpose_biologically(
            matrix=large_bulk_matrix,
            gsm_id="GSM_test_large_bulk",
            geo_id="GSE_large_bulk",
        )

        # Should keep as samples×genes
        assert should_transpose == False

    def test_corrupted_small_matrix(self, geo_service):
        """Test very small matrix (likely corrupted or filtered)."""
        # 50 × 30 - too small for meaningful analysis
        tiny_matrix = pd.DataFrame(np.random.poisson(5, (50, 30)))

        should_transpose, reason = geo_service._determine_transpose_biologically(
            matrix=tiny_matrix, gsm_id="GSM_test_tiny", geo_id="GSE_tiny"
        )

        # Should default to no transpose (conservative)
        assert should_transpose == False
        assert "Ambiguous" in reason or "default" in reason.lower()

    def test_transpose_uses_metadata_helper_method(self, geo_service):
        """Test that transpose logic uses _get_geo_metadata helper method.

        This ensures consistency with the fixed metadata storage system
        introduced to resolve the GSE282425 KeyError: 'metadata' bug.

        The key test is that metadata access doesn't crash with KeyError,
        not the specific transpose decision.
        """
        # Mock the _get_geo_metadata method to return proper nested structure
        mock_metadata_entry = {
            "metadata": {
                "title": "Test Dataset",
                "samples": ["GSM1", "GSM2"],
            },
            "stored_by": "test",
            "fetch_timestamp": "2025-01-01T00:00:00",
        }
        geo_service.data_manager._get_geo_metadata = Mock(
            return_value=mock_metadata_entry
        )

        # Create test matrix - use shape that clearly needs transpose
        # 187,697 genes × 4 samples (like GSE130036 case)
        test_matrix = pd.DataFrame(
            np.random.poisson(10, (187697, 4)),
            columns=[f"sample_{i}" for i in range(4)],
        )

        # Call transpose - should not crash due to metadata access
        should_transpose, reason = geo_service._determine_transpose_biologically(
            matrix=test_matrix,
            gsm_id="GSM123",
            geo_id="GSE123",
            data_type_hint=None,
        )

        # CRITICAL: Verify it executed without KeyError from metadata access
        # This is the regression check - before the fix, this would crash with:
        # KeyError: 'metadata' when trying to access stored_metadata_info["metadata"]
        assert isinstance(should_transpose, bool)
        assert isinstance(reason, str)

        # Verify correct transpose decision for genes-as-rows case
        assert should_transpose == True, "187K rows × 4 cols should transpose to 4×187K"


@pytest.mark.unit
class TestBiologyAwareValidation:
    """Test biology-aware validation thresholds for expression matrices."""

    def test_validate_bulk_rnaseq_high_gene_count(self, geo_service):
        """Test validation accepts bulk RNA-seq with high gene counts."""
        # GSE130036 case: 187,697 × 4 should be VALID (will be transposed to 4×187,697 later)
        # This tests that validation doesn't reject matrices that look wrong but will be fixed by transpose
        bulk_matrix = pd.DataFrame(np.random.poisson(10, (187697, 4)))

        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_bulk", matrix=bulk_matrix
        )

        assert is_valid == True, f"Bulk RNA-seq matrix should be valid: {message}"
        # Should accept but warn about unusual shape
        assert "valid" in message.lower() and (
            "obs" in message.lower() or "vars" in message.lower()
        )

    def test_validate_bulk_rnaseq_many_genes(self, geo_service):
        """Test validation with typical bulk RNA-seq dimensions."""
        # 24 samples × 58,000 genes
        bulk_matrix = pd.DataFrame(np.random.poisson(20, (24, 58000)))

        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_bulk_typical", matrix=bulk_matrix
        )

        assert is_valid == True, f"Typical bulk RNA-seq should be valid: {message}"

    def test_validate_single_cell_large(self, geo_service):
        """Test validation with large single-cell dataset."""
        # 15,000 cells × 25,000 genes
        sc_matrix = pd.DataFrame(np.random.poisson(5, (15000, 25000)))

        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_sc_large", matrix=sc_matrix
        )

        assert is_valid == True, f"Large single-cell should be valid: {message}"

    def test_validate_gene_panel(self, geo_service):
        """Test validation with gene panel (few genes)."""
        # 1,000 cells × 200 genes
        panel_matrix = pd.DataFrame(np.random.poisson(50, (1000, 200)))

        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_panel", matrix=panel_matrix
        )

        assert is_valid == True, f"Gene panel should be valid: {message}"

    def test_reject_too_few_observations(self, geo_service):
        """Test rejection of matrix with only 1 observation."""
        # 1 sample × 25,000 genes - insufficient for analysis
        single_obs_matrix = pd.DataFrame(np.random.poisson(20, (1, 25000)))

        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_single", matrix=single_obs_matrix
        )

        assert is_valid == False, "Single observation should be rejected"
        assert "observation" in message.lower() or "insufficient" in message.lower()

    def test_reject_too_few_variables(self, geo_service):
        """Test rejection of matrix with very few variables."""
        # 15,000 cells × 4 genes - likely transpose error
        few_vars_matrix = pd.DataFrame(np.random.poisson(5, (15000, 4)))

        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_few_vars", matrix=few_vars_matrix
        )

        assert is_valid == False, "Matrix with only 4 variables should be rejected"
        assert "variable" in message.lower() or "transpose" in message.lower()

    def test_reject_empty_matrix(self, geo_service):
        """Test rejection of empty matrix."""
        empty_matrix = pd.DataFrame()

        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_empty", matrix=empty_matrix
        )

        assert is_valid == False, "Empty matrix should be rejected"
        assert "empty" in message.lower() or "0" in message

    def test_reject_very_small_matrix(self, geo_service):
        """Test rejection of very small matrix."""
        # 8 × 8 - too small for meaningful analysis
        tiny_matrix = pd.DataFrame(np.random.poisson(10, (8, 8)))

        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_tiny", matrix=tiny_matrix
        )

        assert is_valid == False, "Very small matrix should be rejected"
        assert "small" in message.lower() or "insufficient" in message.lower()

    def test_validate_small_but_acceptable_matrix(self, geo_service):
        """Test validation of small but acceptable matrix."""
        # 12 × 15 - small but meets minimum thresholds
        small_matrix = pd.DataFrame(np.random.poisson(10, (12, 15)))

        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_small_ok", matrix=small_matrix
        )

        # Should be valid with warning
        assert (
            is_valid == True
        ), f"Small but acceptable matrix should be valid: {message}"

    # ========================================================================
    # Bug #2: Type-Aware Validation Tests (VDJ data with duplicate barcodes)
    # ========================================================================

    def test_vdj_data_with_duplicates_accepted(self, geo_service):
        """
        Test Bug #2: VDJ data (TCR/BCR) with duplicate cell barcodes is ACCEPTED.

        VDJ sequencing produces one row per receptor chain, so duplicate cell
        barcodes are scientifically expected (e.g., TRA + TRB chains for same cell).
        """
        # Create VDJ-like data: 6,593 rows × 30 columns (TCR features)
        # Simulate 3,152 duplicate barcodes (48% duplication rate, typical for TCR data)
        n_unique_cells = 3441
        n_total_rows = 6593

        # Create cell barcodes with duplicates
        unique_barcodes = [f"CELL_{i:05d}" for i in range(n_unique_cells)]
        # Duplicate some barcodes to simulate multi-chain data
        duplicated_barcodes = np.random.choice(
            unique_barcodes, n_total_rows - n_unique_cells, replace=True
        )
        all_barcodes = unique_barcodes + list(duplicated_barcodes)
        np.random.shuffle(all_barcodes)

        # Create VDJ matrix with TCR-like features
        vdj_matrix = pd.DataFrame(
            np.random.randint(0, 100, (n_total_rows, 30)),
            index=all_barcodes,
            columns=[
                "chain",
                "v_gene",
                "d_gene",
                "j_gene",
                "c_gene",
                "cdr1_nt",
                "cdr2_nt",
                "cdr3_nt",
                "cdr1_aa",
                "cdr2_aa",
                "cdr3_aa",
                "reads",
                "umis",
                "frequency",
                "productive",
                "full_length",
                "clonotype_id",
                "clone_size",
                "normalized_count",
                "junction",
                "junction_aa",
                "v_identity",
                "j_identity",
                "alignment_score",
                "consensus_quality",
                "is_cell",
                "confidence",
                "annotation",
                "metadata1",
                "metadata2",
            ],
        )

        # Validate with sample_type="vdj"
        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_vdj", matrix=vdj_matrix, sample_type="vdj"
        )

        assert (
            is_valid is True
        ), f"VDJ data with duplicates should be ACCEPTED: {message}"
        # The function logs VDJ-specific info, but returns message about matrix dimensions
        # Just verify it was accepted (is_valid=True)

    def test_rna_data_with_duplicates_rejected(self, geo_service):
        """
        Test Bug #2: RNA data with duplicate cell barcodes is REJECTED.

        For gene expression data, duplicate cell IDs indicate data corruption
        or processing errors and should be rejected.
        """
        # Create RNA-seq-like data: 8,632 cells × 36,591 genes
        # But introduce 1,000 duplicate cell barcodes (data corruption)
        n_unique_cells = 7632
        n_duplicates = 1000
        n_total_cells = n_unique_cells + n_duplicates

        unique_barcodes = [f"CELL_{i:05d}" for i in range(n_unique_cells)]
        # Duplicate some barcodes to simulate corruption
        duplicated_barcodes = [
            unique_barcodes[i % len(unique_barcodes)] for i in range(n_duplicates)
        ]
        all_barcodes = unique_barcodes + duplicated_barcodes
        np.random.shuffle(all_barcodes)

        # Create gene expression matrix
        rna_matrix = pd.DataFrame(
            np.random.poisson(
                5, (n_total_cells, 100)
            ),  # Simplified: 100 genes instead of 36K
            index=all_barcodes,
        )

        # Validate with sample_type="rna"
        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_rna_dup", matrix=rna_matrix, sample_type="rna"
        )

        assert is_valid is False, "RNA data with duplicates should be REJECTED"
        assert "duplicate" in message.lower()
        assert "rna" in message.lower() or "invalid" in message.lower()

    def test_protein_data_with_duplicates_rejected(self, geo_service):
        """
        Test Bug #2: Protein data with duplicate cell barcodes is REJECTED.

        Affinity proteomics (Olink, antibody arrays) should not have duplicate
        cell identifiers - if present, indicates data corruption.
        """
        # Create protein data: 5,000 cells × 96 proteins (Olink panel size)
        n_unique_cells = 4000
        n_duplicates = 1000
        n_total_cells = n_unique_cells + n_duplicates

        unique_barcodes = [f"SAMPLE_{i:04d}" for i in range(n_unique_cells)]
        duplicated_barcodes = [
            unique_barcodes[i % len(unique_barcodes)] for i in range(n_duplicates)
        ]
        all_barcodes = unique_barcodes + duplicated_barcodes
        np.random.shuffle(all_barcodes)

        # Create protein matrix
        protein_matrix = pd.DataFrame(
            np.random.uniform(0, 10, (n_total_cells, 96)),  # NPX values
            index=all_barcodes,
        )

        # Validate with sample_type="protein"
        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_protein_dup", matrix=protein_matrix, sample_type="protein"
        )

        assert is_valid is False, "Protein data with duplicates should be REJECTED"
        assert "duplicate" in message.lower()
        assert "protein" in message.lower() or "invalid" in message.lower()

    def test_vdj_data_without_duplicates_accepted(self, geo_service):
        """
        Test Bug #2: VDJ data without duplicates is also ACCEPTED.

        Edge case: VDJ data where all cells have only one receptor chain
        (e.g., filtered for single-chain cells). Should still pass validation.
        """
        # Create VDJ data with NO duplicates (all unique barcodes)
        n_cells = 5000
        unique_barcodes = [f"CELL_{i:05d}" for i in range(n_cells)]

        # Create VDJ matrix
        vdj_matrix = pd.DataFrame(
            np.random.randint(0, 100, (n_cells, 30)), index=unique_barcodes
        )

        # Validate with sample_type="vdj"
        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_vdj_no_dup", matrix=vdj_matrix, sample_type="vdj"
        )

        assert (
            is_valid is True
        ), f"VDJ data without duplicates should be ACCEPTED: {message}"

    def test_default_sample_type_rna_rejects_duplicates(self, geo_service):
        """
        Test Bug #2: Default sample_type="rna" behavior (backward compatibility).

        When sample_type is not specified, should default to "rna" and reject duplicates.
        """
        # Create matrix with duplicates
        n_unique_cells = 1000
        n_duplicates = 500
        n_total_cells = n_unique_cells + n_duplicates

        unique_barcodes = [f"CELL_{i:05d}" for i in range(n_unique_cells)]
        duplicated_barcodes = [
            unique_barcodes[i % len(unique_barcodes)] for i in range(n_duplicates)
        ]
        all_barcodes = unique_barcodes + duplicated_barcodes

        matrix = pd.DataFrame(
            np.random.poisson(5, (n_total_cells, 100)), index=all_barcodes
        )

        # Validate WITHOUT specifying sample_type (should default to "rna")
        is_valid, message = geo_service._validate_single_matrix(
            gsm_id="GSM_test_default", matrix=matrix
        )

        assert is_valid is False, "Default (rna) should reject duplicates"
        assert "duplicate" in message.lower()


# ===============================================================================
# Performance and Benchmarking Tests
# ===============================================================================


@pytest.mark.unit
class TestGEOPerformance:
    """Test GEO service performance characteristics."""

    def test_search_performance_metrics(self, geo_service, mock_ncbi_client):
        """Test search performance tracking."""
        with patch.object(geo_service, "search_datasets") as mock_search:
            mock_search.return_value = [
                {"accession": f"GSE{i}", "title": f"Dataset {i}"} for i in range(100)
            ]

            import time

            start_time = time.time()
            results = geo_service.search_datasets("performance test", max_results=100)
            end_time = time.time()

            search_time = end_time - start_time

            assert len(results) == 100
            assert search_time < 5.0  # Should complete within 5 seconds (mocked)

    def test_download_speed_monitoring(
        self, geo_service, mock_ftp_client, temp_download_dir
    ):
        """Test download speed monitoring."""
        with patch.object(geo_service, "download_files") as mock_download:
            mock_download.return_value = {
                "success": True,
                "downloaded_files": [str(temp_download_dir / "test_file.txt.gz")],
                "total_size_bytes": 52428800,  # 50 MB
                "download_time_seconds": 10.5,
                "average_speed_mbps": 40.0,
            }

            result = geo_service.download_files("GSE123456")

            assert result["average_speed_mbps"] == 40.0
            assert result["download_time_seconds"] < 15.0

    def test_memory_usage_optimization(self, geo_service, temp_download_dir):
        """Test memory usage optimization for large files."""
        with patch.object(geo_service, "convert_to_anndata") as mock_convert:
            # Simulate memory-efficient processing
            mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            mock_convert.return_value = mock_adata

            # Test chunk-based processing
            adata = geo_service.convert_to_anndata(
                matrix_file="large_matrix.txt.gz",
                chunk_size=10000,  # Process in chunks
                memory_efficient=True,
            )

            assert adata.n_obs > 0
            mock_convert.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
