"""
Unit tests for QueuePreparationService (router).

Tests database detection, preparer routing, and error handling.
All provider calls are mocked — no network access.
"""

from unittest.mock import MagicMock, patch

import pytest

from lobster.core.interfaces.queue_preparer import (
    IQueuePreparer,
    QueuePreparationResult,
)
from lobster.core.schemas.download_queue import (
    DownloadQueueEntry,
    DownloadStatus,
    StrategyConfig,
)
from lobster.core.schemas.download_urls import DownloadFile, DownloadUrlResult
from lobster.services.data_access.queue_preparation_service import (
    PreparerNotFoundError,
    QueuePreparationService,
)

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


class FakePreparer(IQueuePreparer):
    """Fake preparer for testing registration and routing."""

    def __init__(self, data_manager, databases):
        super().__init__(data_manager)
        self._databases = databases

    def supported_databases(self):
        return self._databases

    def fetch_metadata(self, accession):
        return {"title": f"Fake {accession}"}, None

    def extract_download_urls(self, accession):
        return DownloadUrlResult(accession=accession, database=self._databases[0])

    def recommend_strategy(self, metadata, url_data, accession):
        return StrategyConfig(
            strategy_name="FAKE_STRATEGY",
            concatenation_strategy="auto",
            confidence=0.99,
            rationale="Fake strategy for testing",
        )


# ===========================================================================
# Database Detection
# ===========================================================================


class TestDatabaseDetection:
    """Test accession → database detection via AccessionResolver."""

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_detect_geo(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        assert service.detect_database("GSE180759") == "geo"

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_detect_pride(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        assert service.detect_database("PXD063610") == "pride"

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_detect_sra_study(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        assert service.detect_database("SRP123456") == "sra"

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_detect_sra_run(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        assert service.detect_database("SRR001234") == "sra"

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_detect_massive(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        assert service.detect_database("MSV000012345") == "massive"

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_detect_unknown(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        assert service.detect_database("UNKNOWN123") is None


# ===========================================================================
# Registration and Routing
# ===========================================================================


class TestRegistrationAndRouting:
    """Test preparer registration and routing logic."""

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_register_preparer(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        fake = FakePreparer(mock_data_manager, ["test_db"])
        service.register_preparer(fake)

        assert "test_db" in service.list_supported_databases()
        assert service.get_preparer_for_database("test_db") is fake

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_register_multi_db(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        fake = FakePreparer(mock_data_manager, ["db_a", "db_b"])
        service.register_preparer(fake)

        assert "db_a" in service.list_supported_databases()
        assert "db_b" in service.list_supported_databases()
        assert service.get_preparer_for_database("db_a") is fake
        assert service.get_preparer_for_database("db_b") is fake

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_register_empty_raises(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        fake = FakePreparer(mock_data_manager, [])

        with pytest.raises(ValueError, match="does not declare"):
            service.register_preparer(fake)

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_list_supported_databases_sorted(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        service.register_preparer(FakePreparer(mock_data_manager, ["zeta"]))
        service.register_preparer(FakePreparer(mock_data_manager, ["alpha"]))

        dbs = service.list_supported_databases()
        assert dbs == ["alpha", "zeta"]


# ===========================================================================
# Prepare (end-to-end routing)
# ===========================================================================


class TestPrepare:
    """Test the prepare() method with routing."""

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_prepare_with_explicit_database(self, mock_register, mock_data_manager):
        # Use a real database name that passes DownloadQueueEntry validation
        service = QueuePreparationService(mock_data_manager)
        fake = FakePreparer(mock_data_manager, ["metabolights"])
        service.register_preparer(fake)

        result = service.prepare("MTBLS123", database="metabolights", priority=2)

        assert isinstance(result, QueuePreparationResult)
        assert result.queue_entry.dataset_id == "MTBLS123"
        assert result.queue_entry.database == "metabolights"
        assert result.queue_entry.priority == 2
        assert result.queue_entry.recommended_strategy.strategy_name == "FAKE_STRATEGY"

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_prepare_with_auto_detection(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        fake = FakePreparer(mock_data_manager, ["pride"])
        service.register_preparer(fake)

        result = service.prepare("PXD063610")

        assert result.queue_entry.database == "pride"

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_prepare_unknown_accession_raises(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)

        with pytest.raises(ValueError, match="Cannot detect database"):
            service.prepare("UNKNOWN_ACC_999")

    @patch(
        "lobster.services.data_access.queue_preparation_service."
        "QueuePreparationService._register_default_preparers"
    )
    def test_prepare_no_preparer_raises(self, mock_register, mock_data_manager):
        service = QueuePreparationService(mock_data_manager)
        # Don't register any preparers

        with pytest.raises(PreparerNotFoundError, match="No queue preparer"):
            service.prepare("PXD063610", database="pride")


# ===========================================================================
# Default Registration
# ===========================================================================


class TestDefaultRegistration:
    """Test that default preparers register successfully."""

    def test_default_preparers_register(self, mock_data_manager):
        """All 5 default preparers should register without error."""
        service = QueuePreparationService(mock_data_manager)
        dbs = service.list_supported_databases()

        assert "geo" in dbs
        assert "pride" in dbs
        assert "sra" in dbs
        assert "massive" in dbs
        assert "metabolights" in dbs

    def test_default_preparers_count(self, mock_data_manager):
        """Should have at least 5 databases registered."""
        service = QueuePreparationService(mock_data_manager)
        assert len(service.list_supported_databases()) >= 5


# ===========================================================================
# Entry-Point Discovery (PLUG-01)
# ===========================================================================


class TestEntryPointDiscovery:
    """
    Validate that QueuePreparationService discovers preparers via entry points.

    These tests mock component_registry.list_queue_preparers to simulate
    what happens after pyproject.toml entry-point declarations are in place.
    Tests are GREEN because the mock substitutes for the missing declarations.
    Plan 02 adds the real declarations; Plan 03 gates the hardcoded fallback.
    """

    def test_geo_discovered_via_entry_point(self, mock_data_manager):
        """Patching list_queue_preparers returns geo in list_supported_databases."""
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

        with patch(
            "lobster.core.component_registry.component_registry.list_queue_preparers",
            return_value={"geo": GEOQueuePreparer},
        ):
            service = QueuePreparationService(mock_data_manager)
            assert "geo" in service.list_supported_databases()

    def test_all_5_databases_registered_via_entry_points(self, mock_data_manager):
        """Patching list_queue_preparers with all 5 classes populates service registry."""
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer
        from lobster.services.data_access.massive_queue_preparer import (
            MassIVEQueuePreparer,
        )
        from lobster.services.data_access.metabolights_queue_preparer import (
            MetaboLightsQueuePreparer,
        )
        from lobster.services.data_access.pride_queue_preparer import PRIDEQueuePreparer
        from lobster.services.data_access.sra_queue_preparer import SRAQueuePreparer

        all_5 = {
            "geo": GEOQueuePreparer,
            "sra": SRAQueuePreparer,
            "pride": PRIDEQueuePreparer,
            "massive": MassIVEQueuePreparer,
            "metabolights": MetaboLightsQueuePreparer,
        }

        with patch(
            "lobster.core.component_registry.component_registry.list_queue_preparers",
            return_value=all_5,
        ):
            service = QueuePreparationService(mock_data_manager)
            supported = service.list_supported_databases()
            for db in all_5:
                assert db in supported, (
                    f"Expected '{db}' in list_supported_databases() after patching "
                    "list_queue_preparers with all 5 classes."
                )


# ===========================================================================
# Fallback Gating (PLUG-06)
# ===========================================================================


class TestFallbackGating:
    """
    Validate the hardcoded fallback gate for QueuePreparationService.

    Requirements: PLUG-06
    """

    def test_fallback_flag_is_false_by_default(self):
        """_ALLOW_HARDCODED_FALLBACK must exist and default to False."""
        import lobster.services.data_access.queue_preparation_service as qps_module

        assert hasattr(qps_module, "_ALLOW_HARDCODED_FALLBACK"), (
            "queue_preparation_service module is missing _ALLOW_HARDCODED_FALLBACK. "
            "Add: _ALLOW_HARDCODED_FALLBACK = False at module level."
        )
        assert qps_module._ALLOW_HARDCODED_FALLBACK is False, (
            "_ALLOW_HARDCODED_FALLBACK must default to False to disable hardcoded fallback. "
            "Set to True only for debugging/emergency recovery."
        )

    def test_fallback_skipped_when_flag_false(self, mock_data_manager):
        """
        When _ALLOW_HARDCODED_FALLBACK=False and entry-point discovery returns empty,
        no hardcoded preparer classes are instantiated.

        Verifies that the gate prevents silent fallback to hardcoded imports.
        """
        import lobster.services.data_access.queue_preparation_service as qps_module

        if not hasattr(qps_module, "_ALLOW_HARDCODED_FALLBACK"):
            pytest.skip("_ALLOW_HARDCODED_FALLBACK not yet added")

        with (
            patch(
                "lobster.core.component_registry.component_registry.list_queue_preparers",
                return_value={},
            ),
            patch(
                "lobster.services.data_access.queue_preparation_service"
                "._ALLOW_HARDCODED_FALLBACK",
                False,
            ),
            patch(
                "lobster.services.data_access.geo_queue_preparer.GEOQueuePreparer"
            ) as mock_geo,
        ):
            service = QueuePreparationService(mock_data_manager)
            # With flag=False and no entry-point discovery, no databases registered
            supported = service.list_supported_databases()
            # The hardcoded GEOQueuePreparer should NOT have been instantiated
            mock_geo.assert_not_called()
            # And the database list should be empty (no fallback, no entry points)
            assert len(supported) == 0, (
                f"Expected empty database list when _ALLOW_HARDCODED_FALLBACK=False "
                f"and no entry points discovered, got: {supported}"
            )
