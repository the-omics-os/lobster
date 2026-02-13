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
from unittest.mock import Mock

import pytest

from lobster.core.interfaces.download_service import IDownloadService


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

        # Should have GEO, PRIDE, and MassIVE registered
        assert "geo" in databases
        assert "pride" in databases
        assert "massive" in databases

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
