"""
GEO data service facade -- thin orchestration layer.

This module provides the GEOService class as a composition of 5 focused domain
modules, each handling a specific responsibility:

- MetadataFetcher: metadata fetching, extraction, validation (~12 methods)
- DownloadExecutor: download coordination and pipeline steps (~10 methods)
- ArchiveProcessor: TAR extraction, nested archives, 10X handling (~8 methods)
- MatrixParser: matrix validation, file classification, sample downloads (~18 methods)
- SampleConcatenator: sample storage and concatenation (~5 methods)

The facade preserves the exact public API (fetch_metadata_only, download_dataset,
download_with_strategy) and uses __getattr__ to forward private method calls to
the appropriate domain module for full backward compatibility.

Phase 4 Plan 02: GEO Service Decomposition.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anndata
import pandas as pd

try:
    import GEOparse
except ImportError:
    GEOparse = None

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.exceptions import (
    UnsupportedPlatformError,
)

# Re-export constants for backward compatibility -- consumers import from here
from lobster.services.data_access.geo.constants import (
    DownloadStrategy,
    GEODataSource,
    GEOResult,
    GEOServiceError,
)
from lobster.services.data_access.geo.downloader import (
    GEODownloadManager,
)
from lobster.services.data_access.geo.loaders.tenx import TenXGenomicsLoader
from lobster.services.data_access.geo.parser import GEOParser
from lobster.services.data_access.geo.strategy import (
    PipelineStrategyEngine,
    PipelineType,
)

# Re-exports from helpers.py (added in Plan 01 Task 2) -- keep for backward compat
from lobster.services.data_access.geo.helpers import (
    ARCHIVE_EXTENSIONS,
    RetryOutcome,
    RetryResult,
    _is_archive_url,
    _is_data_valid as _is_data_valid_fn,
    _retry_with_backoff as _retry_with_backoff_fn,
    _score_expression_file,
)

# Domain module imports
from lobster.services.data_access.geo.metadata_fetch import MetadataFetcher
from lobster.services.data_access.geo.download_execution import DownloadExecutor
from lobster.services.data_access.geo.archive_processing import ArchiveProcessor
from lobster.services.data_access.geo.matrix_parsing import MatrixParser
from lobster.services.data_access.geo.concatenation import SampleConcatenator

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress GEOparse DEBUG logging to avoid noise in non-debug mode
geoparse_logger = logging.getLogger("GEOparse")
geoparse_logger.setLevel(logging.WARNING)


class GEOService:
    """
    Professional service for accessing and processing GEO data.

    This class is a thin facade that composes 5 domain modules for focused
    responsibility while preserving the original API surface. All public
    methods delegate to the appropriate module; private methods are forwarded
    via __getattr__ for full backward compatibility with existing test mocks
    and external callers.
    """

    def __init__(
        self,
        data_manager: DataManagerV2,
        cache_dir: Optional[str] = None,
        console=None,
        email: Optional[str] = None,
    ):
        """
        Initialize the GEO service with modular architecture.

        Args:
            data_manager: DataManagerV2 instance for storing processed data as modalities
            cache_dir: Directory to cache downloaded files
            console: Rich console instance for display (creates new if None)
            email: Optional email for NCBI Entrez (for backward compatibility, not currently used)
        """
        if GEOparse is None:
            from lobster.core.component_registry import get_install_command

            cmd = get_install_command("GEOparse")
            raise ImportError(
                f"GEOparse is required but not installed. Install with: {cmd}"
            )

        self.data_manager = data_manager
        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.data_manager.cache_dir / "geo"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.console = console

        # Initialize helper services for fallback functionality
        self.geo_downloader = GEODownloadManager(
            cache_dir=self.cache_dir, console=self.console
        )
        self.geo_parser = GEOParser()

        # Initialize the pipeline strategy engine
        self.pipeline_engine = PipelineStrategyEngine()

        # Initialize the 10X Genomics loader (adaptive format handling)
        self.tenx_loader = TenXGenomicsLoader(
            geo_downloader=self.geo_downloader, cache_dir=self.cache_dir
        )

        # Default download strategy
        self.download_strategy = DownloadStrategy()

        # Compose domain modules
        self._metadata_fetcher = MetadataFetcher(self)
        self._download_executor = DownloadExecutor(self)
        self._archive_processor = ArchiveProcessor(self)
        self._matrix_parser = MatrixParser(self)
        self._concatenator = SampleConcatenator(self)

        logger.info(
            "GEOService initialized with modular architecture: GEOparse + dynamic pipeline strategy + 10X adaptive loader"
        )

    # ------------------------------------------------------------------
    # Helper wrappers (delegated to geo.helpers -- Plan 01 backward compat)
    # ------------------------------------------------------------------

    def _is_data_valid(
        self, data: Optional[Union[pd.DataFrame, anndata.AnnData]]
    ) -> bool:
        """Check if data is valid (non-None and non-empty). Delegates to helpers."""
        return _is_data_valid_fn(data)

    def _retry_with_backoff(
        self,
        operation: Callable[[], Any],
        operation_name: str,
        max_retries: int = 5,
        base_delay: float = 1.0,
        is_ftp: bool = False,
    ) -> RetryResult:
        """Retry with exponential backoff. Delegates to helpers."""
        return _retry_with_backoff_fn(
            operation=operation,
            operation_name=operation_name,
            max_retries=max_retries,
            base_delay=base_delay,
            is_ftp=is_ftp,
            console=getattr(self, "console", None),
        )

    # ------------------------------------------------------------------
    # Public API -- explicit delegation for discoverability
    # ------------------------------------------------------------------

    def fetch_metadata_only(self, geo_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Fetch and validate GEO metadata with fallback mechanisms."""
        return self._metadata_fetcher.fetch_metadata_only(geo_id)

    def download_dataset(self, geo_id: str, adapter: str = None, **kwargs) -> str:
        """Download and process a dataset using modular strategy with fallbacks."""
        return self._download_executor.download_dataset(geo_id, adapter, **kwargs)

    def download_with_strategy(
        self,
        geo_id: str,
        manual_strategy_override: PipelineType = None,
        use_intersecting_genes_only: bool = None,
    ) -> GEOResult:
        """Master function implementing layered download approach."""
        return self._download_executor.download_with_strategy(
            geo_id, manual_strategy_override, use_intersecting_genes_only
        )

    # ------------------------------------------------------------------
    # __getattr__ -- forward private method calls to domain modules
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        """Forward private method calls to domain modules for backward compatibility.

        This ensures that existing test mocks (e.g., mock.patch.object(service, '_process_tar_file'))
        and cross-module calls (e.g., self.service._process_supplementary_files()) continue to work
        transparently after decomposition.

        Uses self.__dict__ to access module instances safely, avoiding infinite
        recursion when __getattr__ is called before __init__ completes.

        If modules haven't been initialized (e.g., when tests patch __init__),
        lazily creates them on first attribute access.
        """
        # Lazily initialize domain modules if __init__ was bypassed (e.g., test mocks)
        module_names = [
            "_metadata_fetcher",
            "_download_executor",
            "_archive_processor",
            "_matrix_parser",
            "_concatenator",
        ]
        if not any(self.__dict__.get(m) for m in module_names):
            # Modules haven't been created -- initialize them now
            self.__dict__["_metadata_fetcher"] = MetadataFetcher(self)
            self.__dict__["_download_executor"] = DownloadExecutor(self)
            self.__dict__["_archive_processor"] = ArchiveProcessor(self)
            self.__dict__["_matrix_parser"] = MatrixParser(self)
            self.__dict__["_concatenator"] = SampleConcatenator(self)

        # Access modules from __dict__ to avoid recursion
        modules = [
            self.__dict__.get(m) for m in module_names
        ]
        for module in modules:
            if module is None:
                continue
            try:
                return getattr(module, name)
            except AttributeError:
                continue
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
