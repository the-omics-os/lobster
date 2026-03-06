"""
Unit tests for IDownloadService implementations (GEO, PRIDE, MassIVE).

Tests cover:
- Service instantiation and configuration
- supported_databases() method
- get_supported_strategies() method
- validate_strategy_params() method
- validate_strategy() wrapper method
- Interface compliance
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from lobster.core.interfaces.download_service import IDownloadService
from lobster.core.schemas.download_queue import StrategyConfig


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManagerV2 with proper attributes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        cache_dir = workspace / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        dm = Mock()
        dm.workspace_path = workspace
        dm.cache_dir = cache_dir
        dm.modalities = {}
        dm.download_queue = Mock()
        dm.log_tool_usage = Mock()

        yield dm


class TestGEODownloadService:
    """Tests for GEODownloadService."""

    def test_instantiation(self, mock_data_manager):
        """Test service can be instantiated."""
        from lobster.services.data_access.geo_download_service import (
            GEODownloadService,
        )

        service = GEODownloadService(mock_data_manager)
        assert service is not None
        assert isinstance(service, IDownloadService)

    def test_supported_databases(self, mock_data_manager):
        """Test supported_databases returns correct list."""
        from lobster.services.data_access.geo_download_service import (
            GEODownloadService,
        )

        service = GEODownloadService(mock_data_manager)
        databases = service.supported_databases()

        assert isinstance(databases, list)
        assert "geo" in databases

    def test_get_supported_strategies(self, mock_data_manager):
        """Test get_supported_strategies returns expected strategies."""
        from lobster.services.data_access.geo_download_service import (
            GEODownloadService,
        )

        service = GEODownloadService(mock_data_manager)
        strategies = service.get_supported_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) > 0
        # GEO should support at least H5_FIRST and MATRIX_FIRST
        assert "H5_FIRST" in strategies or "MATRIX_FIRST" in strategies

    def test_validate_strategy_params_valid(self, mock_data_manager):
        """Test validate_strategy_params with valid parameters."""
        from lobster.services.data_access.geo_download_service import (
            GEODownloadService,
        )

        service = GEODownloadService(mock_data_manager)
        # Empty params should be valid
        is_valid, error = service.validate_strategy_params({})
        assert is_valid is True
        assert error is None

    def test_validate_strategy_wrapper(self, mock_data_manager):
        """Test validate_strategy wrapper method."""
        from lobster.services.data_access.geo_download_service import (
            GEODownloadService,
        )

        service = GEODownloadService(mock_data_manager)
        # Should not raise for valid params
        service.validate_strategy({})

    def test_validate_strategy_wrapper_accepts_auto(self, mock_data_manager):
        """AUTO should mean auto-detect, not invalid manual override."""
        from lobster.services.data_access.geo_download_service import (
            GEODownloadService,
        )

        service = GEODownloadService(mock_data_manager)
        service.validate_strategy({"strategy_name": "AUTO"})

    def test_download_dataset_normalizes_auto_strategy(self, mock_data_manager):
        """AUTO should not be forwarded as a manual GEO pipeline override."""
        from lobster.services.data_access.geo_download_service import (
            GEODownloadService,
        )

        service = GEODownloadService(mock_data_manager)
        queue_entry = Mock(
            database="geo",
            dataset_id="GSE12345",
            entry_id="queue_GSE12345_test",
            recommended_strategy=StrategyConfig(
                strategy_name="AUTO",
                concatenation_strategy="auto",
                confidence=0.5,
                rationale="Use auto-detection",
            ),
        )
        adata = Mock(n_obs=10, n_vars=20)
        captured_kwargs: Dict[str, Any] = {}

        def _capture_download(**kwargs):
            captured_kwargs.update(kwargs)
            return "Downloaded successfully"

        service.geo_service.download_dataset = Mock(side_effect=_capture_download)
        service._find_stored_modality = Mock(return_value="geo_gse12345_test")
        mock_data_manager.get_modality.return_value = adata

        _adata, stats, _ir = service.download_dataset(
            queue_entry,
            strategy_override={
                "strategy_name": "AUTO",
                "strategy_params": {"use_intersecting_genes_only": True},
            },
        )

        assert captured_kwargs["manual_strategy_override"] is None
        assert captured_kwargs["use_intersecting_genes_only"] is True
        assert stats["strategy_used"] == "auto"

    def test_get_service_info(self, mock_data_manager):
        """Test get_service_info returns expected structure."""
        from lobster.services.data_access.geo_download_service import (
            GEODownloadService,
        )

        service = GEODownloadService(mock_data_manager)
        info = service.get_service_info()

        assert "service_name" in info
        assert "supported_databases" in info
        assert "supported_strategies" in info
        assert info["service_name"] == "GEODownloadService"


class TestPRIDEDownloadService:
    """Tests for PRIDEDownloadService."""

    def test_instantiation(self, mock_data_manager):
        """Test service can be instantiated."""
        from lobster.services.data_access.pride_download_service import (
            PRIDEDownloadService,
        )

        service = PRIDEDownloadService(mock_data_manager)
        assert service is not None
        assert isinstance(service, IDownloadService)

    def test_supported_databases(self, mock_data_manager):
        """Test supported_databases returns correct list."""
        from lobster.services.data_access.pride_download_service import (
            PRIDEDownloadService,
        )

        service = PRIDEDownloadService(mock_data_manager)
        databases = service.supported_databases()

        assert isinstance(databases, list)
        assert "pride" in databases
        assert "pxd" in databases

    def test_get_supported_strategies(self, mock_data_manager):
        """Test get_supported_strategies returns expected strategies."""
        from lobster.services.data_access.pride_download_service import (
            PRIDEDownloadService,
        )

        service = PRIDEDownloadService(mock_data_manager)
        strategies = service.get_supported_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) > 0
        # PRIDE should support RESULT_FIRST and MZML_FIRST
        expected = ["RESULT_FIRST", "MZML_FIRST", "SEARCH_FIRST", "RAW_FIRST"]
        for strategy in expected:
            assert strategy in strategies

    def test_validate_strategy_params_valid(self, mock_data_manager):
        """Test validate_strategy_params with valid parameters."""
        from lobster.services.data_access.pride_download_service import (
            PRIDEDownloadService,
        )

        service = PRIDEDownloadService(mock_data_manager)
        # Empty params should be valid
        is_valid, error = service.validate_strategy_params({})
        assert is_valid is True
        assert error is None

    def test_validate_strategy_wrapper(self, mock_data_manager):
        """Test validate_strategy wrapper method."""
        from lobster.services.data_access.pride_download_service import (
            PRIDEDownloadService,
        )

        service = PRIDEDownloadService(mock_data_manager)
        # Should not raise for valid params
        service.validate_strategy({})

    def test_get_service_info(self, mock_data_manager):
        """Test get_service_info returns expected structure."""
        from lobster.services.data_access.pride_download_service import (
            PRIDEDownloadService,
        )

        service = PRIDEDownloadService(mock_data_manager)
        info = service.get_service_info()

        assert "service_name" in info
        assert "supported_databases" in info
        assert "supported_strategies" in info
        assert info["service_name"] == "PRIDEDownloadService"
        assert "pride" in info["supported_databases"]


class TestMassIVEDownloadService:
    """Tests for MassIVEDownloadService."""

    def test_instantiation(self, mock_data_manager):
        """Test service can be instantiated."""
        from lobster.services.data_access.massive_download_service import (
            MassIVEDownloadService,
        )

        service = MassIVEDownloadService(mock_data_manager)
        assert service is not None
        assert isinstance(service, IDownloadService)

    def test_supported_databases(self, mock_data_manager):
        """Test supported_databases returns correct list."""
        from lobster.services.data_access.massive_download_service import (
            MassIVEDownloadService,
        )

        service = MassIVEDownloadService(mock_data_manager)
        databases = service.supported_databases()

        assert isinstance(databases, list)
        assert "massive" in databases
        assert "msv" in databases

    def test_get_supported_strategies(self, mock_data_manager):
        """Test get_supported_strategies returns expected strategies."""
        from lobster.services.data_access.massive_download_service import (
            MassIVEDownloadService,
        )

        service = MassIVEDownloadService(mock_data_manager)
        strategies = service.get_supported_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) > 0
        # MassIVE should support RESULT_FIRST and MZML_FIRST
        expected = ["RESULT_FIRST", "MZML_FIRST", "SEARCH_FIRST", "RAW_FIRST"]
        for strategy in expected:
            assert strategy in strategies

    def test_validate_strategy_params_valid(self, mock_data_manager):
        """Test validate_strategy_params with valid parameters."""
        from lobster.services.data_access.massive_download_service import (
            MassIVEDownloadService,
        )

        service = MassIVEDownloadService(mock_data_manager)
        # Empty params should be valid
        is_valid, error = service.validate_strategy_params({})
        assert is_valid is True
        assert error is None

    def test_validate_strategy_wrapper(self, mock_data_manager):
        """Test validate_strategy wrapper method."""
        from lobster.services.data_access.massive_download_service import (
            MassIVEDownloadService,
        )

        service = MassIVEDownloadService(mock_data_manager)
        # Should not raise for valid params
        service.validate_strategy({})

    def test_get_service_info(self, mock_data_manager):
        """Test get_service_info returns expected structure."""
        from lobster.services.data_access.massive_download_service import (
            MassIVEDownloadService,
        )

        service = MassIVEDownloadService(mock_data_manager)
        info = service.get_service_info()

        assert "service_name" in info
        assert "supported_databases" in info
        assert "supported_strategies" in info
        assert info["service_name"] == "MassIVEDownloadService"
        assert "massive" in info["supported_databases"]


class TestDownloadOrchestrator:
    """Tests for DownloadOrchestrator service registration."""

    def test_auto_registration(self, mock_data_manager):
        """Test that orchestrator auto-registers all available services."""
        from lobster.tools.download_orchestrator import DownloadOrchestrator

        orchestrator = DownloadOrchestrator(mock_data_manager)
        databases = orchestrator.list_supported_databases()

        # Should have all 5 databases: GEO, SRA, PRIDE, MassIVE, MetaboLights
        assert "geo" in databases
        assert "pride" in databases
        assert "massive" in databases
        assert "sra" in databases
        assert "metabolights" in databases

    def test_service_lookup_geo(self, mock_data_manager):
        """Test service lookup for GEO."""
        from lobster.services.data_access.geo_download_service import (
            GEODownloadService,
        )
        from lobster.tools.download_orchestrator import DownloadOrchestrator

        orchestrator = DownloadOrchestrator(mock_data_manager)
        service = orchestrator.get_service_for_database("geo")

        assert service is not None
        assert isinstance(service, GEODownloadService)

    def test_service_lookup_pride(self, mock_data_manager):
        """Test service lookup for PRIDE."""
        from lobster.services.data_access.pride_download_service import (
            PRIDEDownloadService,
        )
        from lobster.tools.download_orchestrator import DownloadOrchestrator

        orchestrator = DownloadOrchestrator(mock_data_manager)
        service = orchestrator.get_service_for_database("pride")

        assert service is not None
        assert isinstance(service, PRIDEDownloadService)

    def test_service_lookup_massive(self, mock_data_manager):
        """Test service lookup for MassIVE."""
        from lobster.services.data_access.massive_download_service import (
            MassIVEDownloadService,
        )
        from lobster.tools.download_orchestrator import DownloadOrchestrator

        orchestrator = DownloadOrchestrator(mock_data_manager)
        service = orchestrator.get_service_for_database("massive")

        assert service is not None
        assert isinstance(service, MassIVEDownloadService)

    def test_service_lookup_case_insensitive(self, mock_data_manager):
        """Test that service lookup is case-insensitive."""
        from lobster.tools.download_orchestrator import DownloadOrchestrator

        orchestrator = DownloadOrchestrator(mock_data_manager)

        # All these should find services
        assert orchestrator.get_service_for_database("GEO") is not None
        assert orchestrator.get_service_for_database("Geo") is not None
        assert orchestrator.get_service_for_database("PRIDE") is not None
        assert orchestrator.get_service_for_database("Pride") is not None

    def test_service_lookup_unknown_returns_none(self, mock_data_manager):
        """Test that unknown database returns None."""
        from lobster.tools.download_orchestrator import DownloadOrchestrator

        orchestrator = DownloadOrchestrator(mock_data_manager)
        service = orchestrator.get_service_for_database("unknown_database")

        assert service is None


class TestInterfaceCompliance:
    """Test that all services properly implement IDownloadService interface."""

    @pytest.mark.parametrize(
        "service_class_path",
        [
            "lobster.services.data_access.geo_download_service.GEODownloadService",
            "lobster.services.data_access.pride_download_service.PRIDEDownloadService",
            "lobster.services.data_access.massive_download_service.MassIVEDownloadService",
        ],
    )
    def test_implements_interface(self, mock_data_manager, service_class_path):
        """Test that service implements all required interface methods."""
        # Import the service class dynamically
        module_path, class_name = service_class_path.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        service_class = getattr(module, class_name)

        service = service_class(mock_data_manager)

        # Check all required methods exist and are callable
        assert hasattr(service, "supported_databases")
        assert callable(service.supported_databases)

        assert hasattr(service, "download_dataset")
        assert callable(service.download_dataset)

        assert hasattr(service, "validate_strategy_params")
        assert callable(service.validate_strategy_params)

        assert hasattr(service, "validate_strategy")
        assert callable(service.validate_strategy)

        assert hasattr(service, "get_supported_strategies")
        assert callable(service.get_supported_strategies)

        # Check return types
        databases = service.supported_databases()
        assert isinstance(databases, list)
        assert all(isinstance(db, str) for db in databases)

        strategies = service.get_supported_strategies()
        assert isinstance(strategies, list)
        assert all(isinstance(s, str) for s in strategies)

        valid, error = service.validate_strategy_params({})
        assert isinstance(valid, bool)
        assert error is None or isinstance(error, str)


# ===========================================================================
# Entry-Point Discovery (PLUG-02)
# ===========================================================================


class TestEntryPointDiscovery:
    """
    Validate that DownloadOrchestrator discovers services via entry points.

    These tests mock component_registry.list_download_services to simulate
    what happens after pyproject.toml entry-point declarations are in place.
    Tests are GREEN because the mock substitutes for the missing declarations.
    Plan 02 adds the real declarations; Plan 03 gates the hardcoded fallback.
    """

    def test_geo_discovered_via_entry_point(self, mock_data_manager):
        """Patching list_download_services returns geo in list_supported_databases."""
        from lobster.services.data_access.geo_download_service import GEODownloadService
        from lobster.tools.download_orchestrator import DownloadOrchestrator

        with patch(
            "lobster.core.component_registry.component_registry.list_download_services",
            return_value={"geo": GEODownloadService},
        ):
            orchestrator = DownloadOrchestrator(mock_data_manager)
            assert "geo" in orchestrator.list_supported_databases()

    def test_all_5_databases_registered_via_entry_points(self, mock_data_manager):
        """Patching list_download_services with all 5 classes populates orchestrator."""
        from lobster.services.data_access.geo_download_service import GEODownloadService
        from lobster.services.data_access.massive_download_service import (
            MassIVEDownloadService,
        )
        from lobster.services.data_access.metabolights_download_service import (
            MetaboLightsDownloadService,
        )
        from lobster.services.data_access.pride_download_service import (
            PRIDEDownloadService,
        )
        from lobster.services.data_access.sra_download_service import SRADownloadService
        from lobster.tools.download_orchestrator import DownloadOrchestrator

        all_5 = {
            "geo": GEODownloadService,
            "sra": SRADownloadService,
            "pride": PRIDEDownloadService,
            "massive": MassIVEDownloadService,
            "metabolights": MetaboLightsDownloadService,
        }

        with patch(
            "lobster.core.component_registry.component_registry.list_download_services",
            return_value=all_5,
        ):
            orchestrator = DownloadOrchestrator(mock_data_manager)
            supported = orchestrator.list_supported_databases()
            for db in all_5:
                assert db in supported, (
                    f"Expected '{db}' in list_supported_databases() after patching "
                    "list_download_services with all 5 classes."
                )


# ===========================================================================
# Fallback Gating for DownloadOrchestrator (PLUG-06)
# ===========================================================================


class TestFallbackGating:
    """
    Validate the hardcoded fallback gate for DownloadOrchestrator.

    Requirements: PLUG-06
    """

    def test_fallback_flag_is_false_by_default(self):
        """_ALLOW_HARDCODED_FALLBACK must exist and default to False."""
        import lobster.tools.download_orchestrator as do_module

        assert hasattr(do_module, "_ALLOW_HARDCODED_FALLBACK"), (
            "download_orchestrator module is missing _ALLOW_HARDCODED_FALLBACK. "
            "Add: _ALLOW_HARDCODED_FALLBACK = False at module level."
        )
        assert do_module._ALLOW_HARDCODED_FALLBACK is False, (
            "_ALLOW_HARDCODED_FALLBACK must default to False to disable hardcoded fallback. "
            "Set to True only for debugging/emergency recovery."
        )

    def test_fallback_skipped_when_flag_false(self, mock_data_manager):
        """
        When _ALLOW_HARDCODED_FALLBACK=False and entry-point discovery returns empty,
        no hardcoded service classes are instantiated.

        Verifies that the gate prevents silent fallback to hardcoded imports.
        """
        import lobster.tools.download_orchestrator as do_module
        from lobster.tools.download_orchestrator import DownloadOrchestrator

        if not hasattr(do_module, "_ALLOW_HARDCODED_FALLBACK"):
            pytest.skip("_ALLOW_HARDCODED_FALLBACK not yet added")

        with (
            patch(
                "lobster.core.component_registry.component_registry.list_download_services",
                return_value={},
            ),
            patch(
                "lobster.tools.download_orchestrator._ALLOW_HARDCODED_FALLBACK",
                False,
            ),
            patch(
                "lobster.services.data_access.geo_download_service.GEODownloadService"
            ) as mock_geo,
        ):
            orchestrator = DownloadOrchestrator(mock_data_manager)
            # With flag=False and no entry-point discovery, no databases registered
            supported = orchestrator.list_supported_databases()
            # The hardcoded GEODownloadService should NOT have been instantiated
            mock_geo.assert_not_called()
            # And the database list should be empty (no fallback, no entry points)
            assert len(supported) == 0, (
                f"Expected empty database list when _ALLOW_HARDCODED_FALLBACK=False "
                f"and no entry points discovered, got: {supported}"
            )
