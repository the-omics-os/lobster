"""
Unit tests for SRADownloadService.

Tests cover:
- Interface compliance (IDownloadService)
- Strategy validation
- Download orchestration
- MD5 verification
- Error handling and retry
- AnnData creation
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import anndata as ad
import pandas as pd
import pytest

from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus
from lobster.services.data_access.sra_download_service import (
    FASTQLoader,
    SRADownloadManager,
    SRADownloadService,
)


@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2."""
    mock_dm = MagicMock()
    mock_dm.workspace_dir = Path("/tmp/test_workspace")
    mock_dm.list_modalities.return_value = []
    return mock_dm


@pytest.fixture
def sra_service(mock_data_manager):
    """Create SRADownloadService instance."""
    return SRADownloadService(mock_data_manager)


@pytest.fixture
def download_manager():
    """Create SRADownloadManager instance."""
    return SRADownloadManager()


@pytest.fixture
def fastq_loader():
    """Create FASTQLoader instance."""
    return FASTQLoader()


@pytest.fixture
def mock_queue_entry():
    """Create mock DownloadQueueEntry."""
    return DownloadQueueEntry(
        entry_id="test_123",
        dataset_id="SRR000001",
        database="sra",
        status=DownloadStatus.PENDING,
    )


class TestSRADownloadService:
    """Test suite for SRADownloadService."""

    def test_initialization(self, sra_service, mock_data_manager):
        """Test service initializes correctly."""
        assert sra_service.data_manager == mock_data_manager
        assert sra_service.sra_provider is not None
        assert sra_service.download_manager is not None
        assert sra_service.fastq_loader is not None

    def test_supported_databases(self, sra_service):
        """Test database support detection."""
        supported = sra_service.supported_databases()
        assert "sra" in supported
        assert "ena" in supported
        assert "ddbj" in supported
        assert len(supported) == 3

    def test_supports_database_class_method(self):
        """Test class method supports_database."""
        assert SRADownloadService.supports_database("sra")
        assert SRADownloadService.supports_database("SRA")  # Case insensitive
        assert SRADownloadService.supports_database("ena")
        assert SRADownloadService.supports_database("ddbj")
        assert not SRADownloadService.supports_database("geo")
        assert not SRADownloadService.supports_database("pride")

    def test_supported_strategies(self, sra_service):
        """Test strategy list."""
        strategies = sra_service.get_supported_strategies()
        assert "FASTQ_FIRST" in strategies
        assert len(strategies) == 1  # Phase 1 only supports FASTQ

    def test_validate_strategy_params_valid(self, sra_service):
        """Test valid strategy parameters."""
        valid, error = sra_service.validate_strategy_params({
            "verify_checksum": True,
            "layout": "PAIRED",
        })
        assert valid is True
        assert error is None

    def test_validate_strategy_params_verify_checksum_invalid_type(self, sra_service):
        """Test invalid verify_checksum type."""
        valid, error = sra_service.validate_strategy_params({
            "verify_checksum": "yes",  # Should be bool
        })
        assert valid is False
        assert "verify_checksum" in error
        assert "bool" in error

    def test_validate_strategy_params_invalid_layout(self, sra_service):
        """Test invalid layout parameter."""
        valid, error = sra_service.validate_strategy_params({
            "layout": "INVALID",
        })
        assert valid is False
        assert "layout" in error.lower()

    def test_validate_strategy_params_valid_layouts(self, sra_service):
        """Test valid layout values."""
        for layout in ["SINGLE", "PAIRED", "single", "paired"]:
            valid, error = sra_service.validate_strategy_params({"layout": layout})
            assert valid is True, f"Layout '{layout}' should be valid"

    def test_validate_strategy_params_unknown_ignored(self, sra_service):
        """Test unknown parameters are ignored (forward compatibility)."""
        valid, error = sra_service.validate_strategy_params({
            "unknown_param": "value",
            "verify_checksum": True,
        })
        assert valid is True  # Unknown params ignored

    def test_validate_strategy_params_file_type_invalid(self, sra_service):
        """Test unsupported file_type (Phase 1 only supports fastq)."""
        valid, error = sra_service.validate_strategy_params({
            "file_type": "sra",  # Not supported in Phase 1
        })
        assert valid is False
        assert "file_type" in error.lower()

    def test_check_download_size_under_threshold(self, sra_service):
        """Test no warning for downloads under 100 GB."""
        # Should not raise
        sra_service._check_download_size(50 * 1e9, "SRR001")  # 50 GB

    def test_check_download_size_over_threshold(self, sra_service):
        """Test warning for downloads over 100 GB."""
        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            sra_service._check_download_size(150 * 1e9, "SRR001")  # 150 GB

        assert "150" in str(exc_info.value)
        assert "100 GB" in str(exc_info.value)
        assert "LOBSTER_SKIP_SIZE_WARNING" in str(exc_info.value)

    def test_check_download_size_skip_warning(self, sra_service, monkeypatch):
        """Test skip warning with environment variable."""
        monkeypatch.setenv("LOBSTER_SKIP_SIZE_WARNING", "true")

        # Should not raise with env var set
        sra_service._check_download_size(150 * 1e9, "SRR001")

    @patch("lobster.services.data_access.sra_download_service.SRAProvider")
    @patch("lobster.services.data_access.sra_download_service.SRADownloadManager")
    @patch("lobster.services.data_access.sra_download_service.FASTQLoader")
    def test_download_dataset_success(
        self,
        mock_loader_class,
        mock_manager_class,
        mock_provider_class,
        sra_service,
        mock_queue_entry,
        tmp_path,
    ):
        """Test successful download workflow."""
        from lobster.core.schemas.download_urls import DownloadUrlResult, DownloadFile

        # Setup mocks
        mock_provider = mock_provider_class.return_value

        # Create typed DownloadUrlResult object
        mock_url_result = DownloadUrlResult(
            accession="SRR000001",
            database="sra",
            raw_files=[
                DownloadFile(
                    url="http://example.com/SRR000001_1.fastq.gz",
                    filename="SRR000001_1.fastq.gz",
                    size_bytes=1000000,
                    checksum="abc123",
                    checksum_type="md5",
                    file_type="raw",
                ),
                DownloadFile(
                    url="http://example.com/SRR000001_2.fastq.gz",
                    filename="SRR000001_2.fastq.gz",
                    size_bytes=1000000,
                    checksum="def456",
                    checksum_type="md5",
                    file_type="raw",
                ),
            ],
            ftp_base="ftp.sra.ebi.ac.uk",
            mirror="ena",
            layout="PAIRED",
            total_size_bytes=2000000,
            run_count=1,
            platform="ILLUMINA",
            recommended_strategy="FASTQ_FIRST",
        )

        mock_provider.get_download_urls.return_value = mock_url_result

        # Setup download manager mock
        mock_manager = mock_manager_class.return_value
        fastq_files = [tmp_path / "SRR000001_1.fastq.gz", tmp_path / "SRR000001_2.fastq.gz"]
        for f in fastq_files:
            f.write_bytes(b"mock fastq data")

        mock_manager.download_run.return_value = (
            fastq_files,
            {"mirror_used": "ena", "total_size_bytes": 2000000, "download_time_seconds": 5.0}
        )

        # Setup FASTQ loader mock
        mock_loader = mock_loader_class.return_value
        mock_adata = MagicMock(spec=ad.AnnData)
        mock_adata.n_obs = 2
        mock_adata.uns = {"fastq_files": {"paths": [str(f) for f in fastq_files]}}
        mock_loader.create_fastq_anndata.return_value = mock_adata

        # Replace service's instances with mocks
        sra_service.sra_provider = mock_provider
        sra_service.download_manager = mock_manager
        sra_service.fastq_loader = mock_loader

        # Execute download
        adata, stats, ir = sra_service.download_dataset(mock_queue_entry)

        # Assertions
        assert adata is not None
        assert adata.n_obs == 2
        assert stats["dataset_id"] == "SRR000001"
        assert stats["database"] == "sra"
        assert stats["strategy_used"] == "FASTQ_FIRST"
        assert stats["n_files"] == 2
        assert stats["layout"] == "PAIRED"
        assert stats["mirror_used"] == "ena"
        assert ir.operation == "sra_download"

    def test_download_dataset_invalid_database(self, sra_service):
        """Test error when database is not supported."""
        entry = DownloadQueueEntry(
            entry_id="test_123",
            dataset_id="GSE12345",
            database="geo",  # Wrong database
            status=DownloadStatus.PENDING,
        )

        with pytest.raises(ValueError) as exc_info:
            sra_service.download_dataset(entry)

        assert "sra/ena/ddbj" in str(exc_info.value).lower()

    def test_create_analysis_step_ir(self, sra_service, mock_queue_entry, tmp_path):
        """Test AnalysisStep IR creation."""
        fastq_paths = [tmp_path / "SRR001_1.fastq.gz", tmp_path / "SRR001_2.fastq.gz"]
        for f in fastq_paths:
            f.write_bytes(b"test")

        url_info = {
            "layout": "PAIRED",
            "platform": "ILLUMINA",
            "total_size_bytes": 2000000,
        }

        ir = sra_service._create_analysis_step_ir(
            accession="SRR000001",
            fastq_paths=fastq_paths,
            strategy_name="FASTQ_FIRST",
            verify_checksum=True,
            queue_entry_id="test_123",
            url_info=url_info,
        )

        assert ir.operation == "sra_download"
        assert ir.tool_name == "SRADownloadService.download_dataset"
        assert "SRR000001" in ir.description
        assert ir.library == "lobster"
        assert len(ir.imports) > 0
        assert "DownloadOrchestrator" in ir.code_template
        assert ir.parameters["dataset_id"] == "SRR000001"
        assert ir.parameters["database"] == "sra"
        assert ir.parameters["verify_checksum"] is True


class TestSRADownloadManager:
    """Test suite for SRADownloadManager."""

    def test_initialization(self, download_manager):
        """Test manager initializes correctly."""
        assert download_manager.CHUNK_SIZE == 8 * 1024 * 1024
        assert download_manager.MIRRORS == ["ena", "ncbi", "ddbj"]

    def test_verify_md5_success(self, download_manager, tmp_path):
        """Test MD5 verification succeeds for correct checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # Calculate actual MD5
        import hashlib
        md5 = hashlib.md5()
        md5.update(b"hello world")
        expected_md5 = md5.hexdigest()

        # Verify
        assert download_manager._verify_md5(test_file, expected_md5) is True

    def test_verify_md5_failure(self, download_manager, tmp_path):
        """Test MD5 verification fails for wrong checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        wrong_md5 = "wrongchecksum123"

        # Verify
        assert download_manager._verify_md5(test_file, wrong_md5) is False


class TestFASTQLoader:
    """Test suite for FASTQLoader."""

    def test_extract_run_id_srr(self, fastq_loader):
        """Test SRR accession extraction."""
        assert fastq_loader._extract_run_id("SRR21960766_1.fastq.gz") == "SRR21960766"
        assert fastq_loader._extract_run_id("SRR001_2.fastq.gz") == "SRR001"

    def test_extract_run_id_err(self, fastq_loader):
        """Test ERR accession extraction (ENA)."""
        assert fastq_loader._extract_run_id("ERR123456_1.fastq.gz") == "ERR123456"

    def test_extract_run_id_drr(self, fastq_loader):
        """Test DRR accession extraction (DDBJ)."""
        assert fastq_loader._extract_run_id("DRR987654_1.fastq.gz") == "DRR987654"

    def test_extract_run_id_unknown(self, fastq_loader):
        """Test fallback for unrecognized format."""
        assert fastq_loader._extract_run_id("unknown_file.fastq.gz") == "UNKNOWN"

    def test_detect_read_type_r1(self, fastq_loader):
        """Test R1 read detection."""
        assert fastq_loader._detect_read_type("SRR001_1.fastq.gz") == "R1"
        assert fastq_loader._detect_read_type("SRR001_R1.fastq.gz") == "R1"
        assert fastq_loader._detect_read_type("SRR001_R1_001.fastq.gz") == "R1"

    def test_detect_read_type_r2(self, fastq_loader):
        """Test R2 read detection."""
        assert fastq_loader._detect_read_type("SRR001_2.fastq.gz") == "R2"
        assert fastq_loader._detect_read_type("SRR001_R2.fastq.gz") == "R2"
        assert fastq_loader._detect_read_type("SRR001_R2_001.fastq.gz") == "R2"

    def test_detect_read_type_single(self, fastq_loader):
        """Test single-end read detection."""
        assert fastq_loader._detect_read_type("SRR001.fastq.gz") == "single"
        assert fastq_loader._detect_read_type("SRR001_single.fastq.gz") == "single"

    def test_create_fastq_anndata(self, fastq_loader, mock_queue_entry, tmp_path):
        """Test AnnData creation from FASTQ files."""
        # Create mock FASTQ files
        fastq_files = [
            tmp_path / "SRR000001_1.fastq.gz",
            tmp_path / "SRR000001_2.fastq.gz",
        ]
        for f in fastq_files:
            f.write_bytes(b"@read1\nACGT\n+\nIIII\n")

        metadata = {
            "layout": "PAIRED",
            "platform": "ILLUMINA",
            "mirror": "ena",
            "total_size_bytes": 1000,
        }

        # Create AnnData
        adata = fastq_loader.create_fastq_anndata(
            fastq_paths=fastq_files,
            metadata=metadata,
            queue_entry=mock_queue_entry,
        )

        # Assertions
        assert isinstance(adata, ad.AnnData)
        assert adata.n_obs == 2  # 2 FASTQ files
        assert adata.n_vars == 1  # Placeholder

        # Check .obs
        assert "run_accession" in adata.obs.columns
        assert "fastq_path" in adata.obs.columns
        assert "read_type" in adata.obs.columns
        assert all(adata.obs["run_accession"] == "SRR000001")
        assert adata.obs["read_type"].tolist() == ["R1", "R2"]

        # Check .uns
        assert adata.uns["data_type"] == "fastq_raw"
        assert "fastq_files" in adata.uns
        assert len(adata.uns["fastq_files"]["paths"]) == 2
        assert adata.uns["fastq_files"]["layout"] == "PAIRED"
        assert "processing_required" in adata.uns
        assert "alignment" in adata.uns["processing_required"]
        assert "download_provenance" in adata.uns

    def test_create_fastq_anndata_single_end(self, fastq_loader, mock_queue_entry, tmp_path):
        """Test AnnData creation for single-end FASTQ."""
        fastq_files = [tmp_path / "SRR000001.fastq.gz"]
        fastq_files[0].write_bytes(b"@read1\nACGT\n+\nIIII\n")

        metadata = {
            "layout": "SINGLE",
            "platform": "ILLUMINA",
            "mirror": "ena",
        }

        adata = fastq_loader.create_fastq_anndata(
            fastq_paths=fastq_files,
            metadata=metadata,
            queue_entry=mock_queue_entry,
        )

        assert adata.n_obs == 1
        assert adata.obs["read_type"].iloc[0] == "single"
        assert adata.uns["fastq_files"]["layout"] == "SINGLE"


class TestSRADownloadManagerErrorHandling:
    """Test suite for HTTP error handling in SRADownloadManager."""

    def test_download_with_progress_http_429_with_retry_after(
        self, download_manager, tmp_path, monkeypatch
    ):
        """Test HTTP 429 handling with Retry-After header."""
        call_count = [0]

        def mock_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First attempt: rate limit
                mock_resp = MagicMock()
                mock_resp.status_code = 429
                mock_resp.headers = {"Retry-After": "1"}
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                return mock_resp
            else:
                # Success on retry
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.headers = {"content-length": "100"}
                mock_resp.iter_content = MagicMock(return_value=[b"test" * 25])
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                return mock_resp

        with patch("requests.get", side_effect=mock_get):
            with patch("time.sleep") as mock_sleep:
                output_path = tmp_path / "test.fastq.gz"
                download_manager._download_with_progress(
                    url="http://example.com/test.fastq.gz",
                    output_path=output_path,
                    max_retries=3,
                )

                # Should have slept once for retry
                assert mock_sleep.called
                assert call_count[0] == 2  # Two requests (429 then 200)

    def test_download_with_progress_http_500_retry(
        self, download_manager, tmp_path
    ):
        """Test HTTP 500 retry with exponential backoff."""
        call_count = [0]

        def mock_get(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()

            if call_count[0] <= 2:
                # First two attempts: server error
                mock_resp.status_code = 500
            else:
                # Success on third attempt
                mock_resp.status_code = 200
                mock_resp.headers = {"content-length": "100"}
                mock_resp.iter_content = MagicMock(return_value=[b"test" * 25])

            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("requests.get", side_effect=mock_get):
            with patch("time.sleep") as mock_sleep:
                output_path = tmp_path / "test.fastq.gz"
                download_manager._download_with_progress(
                    url="http://example.com/test.fastq.gz",
                    output_path=output_path,
                    max_retries=3,
                )

                # Should have retried twice (500, 500, then 200)
                assert mock_sleep.call_count == 2
                assert call_count[0] == 3

    def test_download_with_progress_http_204_permission_error(
        self, download_manager, tmp_path
    ):
        """Test HTTP 204 (no content) raises PermissionError."""
        def mock_get(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 204
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("requests.get", side_effect=mock_get):
            output_path = tmp_path / "test.fastq.gz"

            with pytest.raises(PermissionError) as exc_info:
                download_manager._download_with_progress(
                    url="http://example.com/test.fastq.gz",
                    output_path=output_path,
                )

            assert "204" in str(exc_info.value)
            assert "permission" in str(exc_info.value).lower()

    def test_download_with_progress_timeout_retry(
        self, download_manager, tmp_path
    ):
        """Test timeout handling with retry."""
        import requests as requests_module

        call_count = [0]

        def mock_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise requests_module.exceptions.Timeout("Connection timed out")
            # Success on second attempt
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-length": "100"}
            mock_resp.iter_content = MagicMock(return_value=[b"test" * 25])
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("requests.get", side_effect=mock_get):
            with patch("time.sleep"):
                output_path = tmp_path / "test.fastq.gz"
                download_manager._download_with_progress(
                    url="http://example.com/test.fastq.gz",
                    output_path=output_path,
                    max_retries=3,
                )

                assert call_count[0] == 2  # Timeout then success


@pytest.mark.integration
class TestSRADownloadServiceIntegration:
    """Integration tests requiring network access."""

    @pytest.mark.skip(reason="Requires network access and takes time")
    def test_download_real_small_dataset(self, sra_service, tmp_path):
        """Test download of real (tiny) SRA dataset."""
        # SRR000001 is a very small public dataset
        entry = DownloadQueueEntry(
            entry_id="integration_test",
            dataset_id="SRR000001",
            database="sra",
            status=DownloadStatus.PENDING,
        )

        adata, stats, ir = sra_service.download_dataset(entry)

        assert adata is not None
        assert stats["dataset_id"] == "SRR000001"
        assert stats["database"] == "sra"
        assert stats["total_size_mb"] < 50  # Should be small
        assert Path(adata.uns["fastq_files"]["paths"][0]).exists()
