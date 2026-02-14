"""
Unit tests for database-specific queue preparers.

Tests IQueuePreparer implementations: GEO, PRIDE, SRA, MassIVE.
All provider calls are mocked — no network access.
"""

from unittest.mock import MagicMock, patch

import pytest

from lobster.core.interfaces.queue_preparer import (
    IQueuePreparer,
    QueuePreparationResult,
)
from lobster.core.schemas.download_queue import DownloadQueueEntry, StrategyConfig
from lobster.core.schemas.download_urls import DownloadFile, DownloadUrlResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2."""
    dm = MagicMock()
    dm.metadata_store = {}
    dm.download_queue.list_entries.return_value = []
    return dm


def _make_url_result(
    accession: str,
    database: str,
    primary_files=None,
    raw_files=None,
    processed_files=None,
    search_files=None,
    supplementary_files=None,
    ftp_base=None,
    layout=None,
    platform=None,
    run_count=None,
    mirror=None,
    error=None,
) -> DownloadUrlResult:
    """Helper to build DownloadUrlResult for tests."""
    return DownloadUrlResult(
        accession=accession,
        database=database,
        primary_files=primary_files or [],
        raw_files=raw_files or [],
        processed_files=processed_files or [],
        search_files=search_files or [],
        supplementary_files=supplementary_files or [],
        ftp_base=ftp_base,
        layout=layout,
        platform=platform,
        run_count=run_count,
        mirror=mirror,
        error=error,
    )


# ===========================================================================
# GEOQueuePreparer
# ===========================================================================


class TestGEOQueuePreparer:
    """Tests for GEOQueuePreparer."""

    def test_supported_databases(self, mock_data_manager):
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

        preparer = GEOQueuePreparer(mock_data_manager)
        assert preparer.supported_databases() == ["geo"]

    @patch(
        "lobster.services.data_access.geo_queue_preparer.GEOQueuePreparer._get_geo_service"
    )
    def test_fetch_metadata_fresh(self, mock_get_service, mock_data_manager):
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

        mock_service = MagicMock()
        mock_service.fetch_metadata_only.return_value = (
            {"title": "Test", "n_samples": 5},
            {"is_valid": True},
        )
        mock_get_service.return_value = mock_service

        preparer = GEOQueuePreparer(mock_data_manager)
        metadata, validation = preparer.fetch_metadata("GSE180759")

        assert metadata["title"] == "Test"
        assert validation["is_valid"] is True
        mock_service.fetch_metadata_only.assert_called_once_with("GSE180759")

    def test_fetch_metadata_cached(self, mock_data_manager):
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

        mock_data_manager.metadata_store = {
            "GSE180759": {
                "metadata": {"title": "Cached Dataset", "n_samples": 10},
                "validation_result": {"is_valid": True},
            }
        }

        preparer = GEOQueuePreparer(mock_data_manager)
        metadata, validation = preparer.fetch_metadata("GSE180759")

        assert metadata["title"] == "Cached Dataset"

    @patch(
        "lobster.services.data_access.geo_queue_preparer.GEOQueuePreparer._get_geo_provider"
    )
    def test_extract_download_urls(self, mock_get_provider, mock_data_manager):
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

        url_result = _make_url_result(
            "GSE180759",
            "geo",
            primary_files=[
                DownloadFile(url="ftp://test/matrix.txt.gz", filename="matrix.txt.gz")
            ],
        )
        mock_provider = MagicMock()
        mock_provider.get_download_urls.return_value = url_result
        mock_get_provider.return_value = mock_provider

        preparer = GEOQueuePreparer(mock_data_manager)
        result = preparer.extract_download_urls("GSE180759")

        assert result.accession == "GSE180759"
        assert len(result.primary_files) == 1

    def test_recommend_strategy_fallback(self, mock_data_manager):
        """Test that recommend_strategy falls back to heuristics on error."""
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

        preparer = GEOQueuePreparer(mock_data_manager)

        url_data = _make_url_result(
            "GSE180759",
            "geo",
            primary_files=[
                DownloadFile(url="ftp://test/data.h5ad", filename="data.h5ad"),
            ],
        )
        metadata = {"title": "Test scRNA-seq", "n_samples": 5}

        # LLM will fail (no model configured), should fallback
        strategy = preparer.recommend_strategy(metadata, url_data, "GSE180759")

        assert isinstance(strategy, StrategyConfig)
        assert strategy.strategy_name == "H5_FIRST"
        assert strategy.confidence >= 0.85


# ===========================================================================
# PRIDEQueuePreparer
# ===========================================================================


class TestPRIDEQueuePreparer:
    """Tests for PRIDEQueuePreparer."""

    def test_supported_databases(self, mock_data_manager):
        from lobster.services.data_access.pride_queue_preparer import PRIDEQueuePreparer

        preparer = PRIDEQueuePreparer(mock_data_manager)
        dbs = preparer.supported_databases()
        assert "pride" in dbs
        assert "proteomexchange" in dbs

    def test_recommend_strategy_processed_files(self, mock_data_manager):
        from lobster.services.data_access.pride_queue_preparer import PRIDEQueuePreparer

        preparer = PRIDEQueuePreparer(mock_data_manager)

        url_data = _make_url_result(
            "PXD063610",
            "pride",
            processed_files=[
                DownloadFile(url="https://test/results.txt", filename="results.txt"),
            ],
        )

        strategy = preparer.recommend_strategy({}, url_data, "PXD063610")

        assert strategy.strategy_name == "RESULT_FIRST"
        assert strategy.confidence == 0.90

    def test_recommend_strategy_mzml(self, mock_data_manager):
        from lobster.services.data_access.pride_queue_preparer import PRIDEQueuePreparer

        preparer = PRIDEQueuePreparer(mock_data_manager)

        url_data = _make_url_result(
            "PXD063610",
            "pride",
            raw_files=[
                DownloadFile(url="https://test/data.mzML", filename="data.mzML"),
            ],
        )

        strategy = preparer.recommend_strategy({}, url_data, "PXD063610")

        assert strategy.strategy_name == "MZML_FIRST"
        assert strategy.confidence == 0.80

    def test_recommend_strategy_search_files(self, mock_data_manager):
        from lobster.services.data_access.pride_queue_preparer import PRIDEQueuePreparer

        preparer = PRIDEQueuePreparer(mock_data_manager)

        url_data = _make_url_result(
            "PXD063610",
            "pride",
            search_files=[
                DownloadFile(url="https://test/output.mzid", filename="output.mzid"),
            ],
        )

        strategy = preparer.recommend_strategy({}, url_data, "PXD063610")

        assert strategy.strategy_name == "SEARCH_FIRST"
        assert strategy.confidence == 0.75

    def test_recommend_strategy_raw_only(self, mock_data_manager):
        from lobster.services.data_access.pride_queue_preparer import PRIDEQueuePreparer

        preparer = PRIDEQueuePreparer(mock_data_manager)

        url_data = _make_url_result(
            "PXD063610",
            "pride",
            raw_files=[
                DownloadFile(url="https://test/data.raw", filename="data.raw"),
            ],
        )

        strategy = preparer.recommend_strategy({}, url_data, "PXD063610")

        assert strategy.strategy_name == "RAW_FIRST"
        assert strategy.confidence == 0.60

    def test_recommend_strategy_no_files(self, mock_data_manager):
        from lobster.services.data_access.pride_queue_preparer import PRIDEQueuePreparer

        preparer = PRIDEQueuePreparer(mock_data_manager)

        url_data = _make_url_result("PXD063610", "pride")

        strategy = preparer.recommend_strategy({}, url_data, "PXD063610")

        assert strategy.strategy_name == "AUTO"
        assert strategy.confidence == 0.40


# ===========================================================================
# SRAQueuePreparer
# ===========================================================================


class TestSRAQueuePreparer:
    """Tests for SRAQueuePreparer."""

    def test_supported_databases(self, mock_data_manager):
        from lobster.services.data_access.sra_queue_preparer import SRAQueuePreparer

        preparer = SRAQueuePreparer(mock_data_manager)
        dbs = preparer.supported_databases()
        assert "sra" in dbs
        assert "ena" in dbs

    def test_recommend_strategy_with_files(self, mock_data_manager):
        from lobster.services.data_access.sra_queue_preparer import SRAQueuePreparer

        preparer = SRAQueuePreparer(mock_data_manager)

        url_data = _make_url_result(
            "SRP123456",
            "sra",
            raw_files=[
                DownloadFile(
                    url="ftp://test/SRR001.fastq.gz", filename="SRR001.fastq.gz"
                ),
                DownloadFile(
                    url="ftp://test/SRR002.fastq.gz", filename="SRR002.fastq.gz"
                ),
            ],
            layout="PAIRED",
            platform="ILLUMINA",
            run_count=2,
            mirror="ena",
        )

        strategy = preparer.recommend_strategy({}, url_data, "SRP123456")

        assert strategy.strategy_name == "FASTQ_FIRST"
        assert strategy.confidence == 0.90
        assert strategy.strategy_params["layout"] == "PAIRED"
        assert strategy.strategy_params["platform"] == "ILLUMINA"
        assert strategy.strategy_params["run_count"] == 2

    def test_recommend_strategy_no_files(self, mock_data_manager):
        from lobster.services.data_access.sra_queue_preparer import SRAQueuePreparer

        preparer = SRAQueuePreparer(mock_data_manager)

        url_data = _make_url_result("SRP123456", "sra", layout="SINGLE")

        strategy = preparer.recommend_strategy({}, url_data, "SRP123456")

        assert strategy.strategy_name == "FASTQ_FIRST"
        assert strategy.confidence == 0.60

    def test_timeout_scales_by_run_count(self, mock_data_manager):
        from lobster.services.data_access.sra_queue_preparer import SRAQueuePreparer

        preparer = SRAQueuePreparer(mock_data_manager)

        # Large run count should get larger timeout
        url_data = _make_url_result("SRP123456", "sra", run_count=100)
        strategy = preparer.recommend_strategy({}, url_data, "SRP123456")
        assert strategy.execution_params["timeout"] == 7200

        # Small run count
        url_data_small = _make_url_result("SRP123456", "sra", run_count=3)
        strategy_small = preparer.recommend_strategy({}, url_data_small, "SRP123456")
        assert strategy_small.execution_params["timeout"] == 1800


# ===========================================================================
# MassIVEQueuePreparer
# ===========================================================================


class TestMassIVEQueuePreparer:
    """Tests for MassIVEQueuePreparer."""

    def test_supported_databases(self, mock_data_manager):
        from lobster.services.data_access.massive_queue_preparer import (
            MassIVEQueuePreparer,
        )

        preparer = MassIVEQueuePreparer(mock_data_manager)
        assert preparer.supported_databases() == ["massive"]

    def test_recommend_strategy_with_processed(self, mock_data_manager):
        from lobster.services.data_access.massive_queue_preparer import (
            MassIVEQueuePreparer,
        )

        preparer = MassIVEQueuePreparer(mock_data_manager)

        url_data = _make_url_result(
            "MSV000012345",
            "massive",
            processed_files=[
                DownloadFile(url="ftp://test/result.txt", filename="result.txt"),
            ],
            ftp_base="ftp://massive.ucsd.edu/MSV000012345/",
        )

        strategy = preparer.recommend_strategy({}, url_data, "MSV000012345")

        assert strategy.strategy_name == "RESULT_FIRST"
        assert strategy.confidence == 0.80

    def test_recommend_strategy_ftp_only(self, mock_data_manager):
        from lobster.services.data_access.massive_queue_preparer import (
            MassIVEQueuePreparer,
        )

        preparer = MassIVEQueuePreparer(mock_data_manager)

        url_data = _make_url_result(
            "MSV000012345",
            "massive",
            ftp_base="ftp://massive.ucsd.edu/MSV000012345/",
        )

        strategy = preparer.recommend_strategy({}, url_data, "MSV000012345")

        assert strategy.strategy_name == "RAW_FIRST"
        assert strategy.confidence == 0.65
        assert (
            strategy.strategy_params["ftp_base"]
            == "ftp://massive.ucsd.edu/MSV000012345/"
        )

    def test_recommend_strategy_no_info(self, mock_data_manager):
        from lobster.services.data_access.massive_queue_preparer import (
            MassIVEQueuePreparer,
        )

        preparer = MassIVEQueuePreparer(mock_data_manager)

        url_data = _make_url_result("MSV000012345", "massive")

        strategy = preparer.recommend_strategy({}, url_data, "MSV000012345")

        assert strategy.strategy_name == "RAW_FIRST"
        assert strategy.confidence == 0.40


# ===========================================================================
# GEO Fallback Strategy Helpers
# ===========================================================================


class TestGEOFallbackHelpers:
    """Test the extracted GEO helper functions."""

    def test_is_single_cell_dataset_positive(self):
        from lobster.services.data_access.geo_queue_preparer import (
            _is_single_cell_dataset,
        )

        assert _is_single_cell_dataset({"title": "scRNA-seq of human brain"})
        assert _is_single_cell_dataset({"summary": "10x Chromium single cell"})
        assert _is_single_cell_dataset({"platform": "GPL24676 Chromium"})
        assert _is_single_cell_dataset({"library_strategy": "10x scRNA"})

    def test_is_single_cell_dataset_negative(self):
        from lobster.services.data_access.geo_queue_preparer import (
            _is_single_cell_dataset,
        )

        assert not _is_single_cell_dataset({"title": "Bulk RNA-seq of liver"})
        assert not _is_single_cell_dataset({})

    def test_create_fallback_strategy_h5(self):
        from lobster.services.data_access.geo_queue_preparer import (
            _create_fallback_strategy,
        )

        url_data = _make_url_result(
            "GSE12345",
            "geo",
            primary_files=[
                DownloadFile(url="ftp://test/data.h5ad", filename="data.h5ad"),
            ],
        )

        strategy = _create_fallback_strategy(url_data, {"title": "Test"})
        assert strategy.strategy_name == "H5_FIRST"
        assert strategy.confidence == 0.90

    def test_create_fallback_strategy_matrix(self):
        from lobster.services.data_access.geo_queue_preparer import (
            _create_fallback_strategy,
        )

        url_data = _make_url_result(
            "GSE12345",
            "geo",
            primary_files=[
                DownloadFile(
                    url="ftp://test/matrix.txt.gz",
                    filename="matrix.txt.gz",
                    file_type="matrix",
                ),
            ],
        )

        strategy = _create_fallback_strategy(url_data, {"title": "Bulk RNA-seq"})
        assert strategy.strategy_name == "MATRIX_FIRST"

    def test_create_recommended_strategy_h5ad(self):
        from lobster.services.data_access.geo_queue_preparer import (
            _create_recommended_strategy,
        )

        url_data = _make_url_result("GSE12345", "geo")
        analysis = {"has_h5ad": True}
        metadata = {"n_samples": 5}

        strategy = _create_recommended_strategy(
            MagicMock(), analysis, metadata, url_data
        )
        assert strategy.strategy_name == "H5_FIRST"
        assert strategy.confidence == 0.95
        assert strategy.concatenation_strategy == "auto"

    def test_create_recommended_strategy_large_samples(self):
        from lobster.services.data_access.geo_queue_preparer import (
            _create_recommended_strategy,
        )

        url_data = _make_url_result("GSE12345", "geo")
        analysis = {"has_processed_matrix": True}
        metadata = {"n_samples": 50, "platform": "GPL570"}

        strategy = _create_recommended_strategy(
            MagicMock(), analysis, metadata, url_data
        )
        assert strategy.concatenation_strategy == "intersection"
        assert strategy.strategy_params["use_intersecting_genes_only"] is True


# ===========================================================================
# IQueuePreparer.prepare_queue_entry (template method)
# ===========================================================================


class TestPrepareQueueEntry:
    """Test the template method on IQueuePreparer."""

    @patch(
        "lobster.services.data_access.pride_queue_preparer.PRIDEQueuePreparer._get_pride_provider"
    )
    def test_prepare_queue_entry_pride(self, mock_get_provider, mock_data_manager):
        from lobster.services.data_access.pride_queue_preparer import PRIDEQueuePreparer

        mock_provider = MagicMock()
        mock_provider.get_project_metadata.return_value = {
            "title": "Test PRIDE Project",
            "organisms": [],
        }
        mock_provider.get_download_urls.return_value = _make_url_result(
            "PXD063610",
            "pride",
            processed_files=[
                DownloadFile(url="https://test/results.txt", filename="results.txt"),
            ],
        )
        mock_get_provider.return_value = mock_provider

        preparer = PRIDEQueuePreparer(mock_data_manager)
        result = preparer.prepare_queue_entry("PXD063610", priority=3)

        assert isinstance(result, QueuePreparationResult)
        assert isinstance(result.queue_entry, DownloadQueueEntry)
        assert result.queue_entry.dataset_id == "PXD063610"
        assert result.queue_entry.database == "pride"
        assert result.queue_entry.priority == 3
        assert result.queue_entry.entry_id.startswith("queue_PXD063610_")
        assert result.queue_entry.recommended_strategy.strategy_name == "RESULT_FIRST"
        assert result.url_data is not None
        assert "PRIDE" in result.validation_summary
