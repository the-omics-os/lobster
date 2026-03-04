"""
Real API integration tests for SRADownloadService.

These tests require network access and hit real SRA endpoints.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus
from lobster.services.data_access.sra_download_service import SRADownloadService

pytestmark = [pytest.mark.integration]


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
