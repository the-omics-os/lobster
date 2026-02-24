"""
GEO Queue Preparer — prepares download queue entries for GEO datasets.

Extracts strategy analysis logic previously inlined in research_agent.py,
providing the GEO-specific implementation of IQueuePreparer.

GEO is unique among preparers in using an LLM (DataExpertAssistant) for
strategy recommendation. Falls back to URL-based heuristics on failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from lobster.core.interfaces.queue_preparer import IQueuePreparer
from lobster.utils.logger import get_logger

if TYPE_CHECKING:
    from lobster.core.schemas.download_queue import StrategyConfig
    from lobster.core.schemas.download_urls import DownloadUrlResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers (extracted from research_agent.py)
# ---------------------------------------------------------------------------


def _is_single_cell_dataset(metadata: dict) -> bool:
    """Detect if dataset is single-cell based on metadata keywords.

    Delegates to unified DataTypeDetector from omics_registry.
    """
    try:
        from lobster.core.omics_registry import DataTypeDetector

        return DataTypeDetector().is_single_cell(metadata)
    except ImportError:
        # Fallback: inline detection if omics_registry not available
        single_cell_keywords = [
            "single-cell",
            "single cell",
            "scRNA-seq",
            "10x",
            "10X",
            "droplet",
            "Drop-seq",
            "Smart-seq",
            "CEL-seq",
            "inDrop",
            "single nuclei",
            "snRNA-seq",
            "scATAC-seq",
            "Chromium",
        ]
        text_fields = [
            metadata.get("title", ""),
            metadata.get("summary", ""),
            metadata.get("overall_design", ""),
            metadata.get("type", ""),
            metadata.get("description", ""),
        ]
        for field in text_fields:
            if any(kw.lower() in field.lower() for kw in single_cell_keywords):
                return True
        return False


def _is_proteomics_dataset(metadata: dict) -> bool:
    """Detect if dataset is proteomics based on metadata keywords and platform.

    Delegates to unified DataTypeDetector from omics_registry.
    """
    try:
        from lobster.core.omics_registry import DataTypeDetector

        return DataTypeDetector().is_proteomics(metadata)
    except ImportError:
        # Fallback: check main keywords only
        proteomics_keywords = [
            "proteomics",
            "proteome",
            "mass spectrometry",
            "mass spec",
            "ms/ms",
            "lc-ms",
            "orbitrap",
            "tmt",
            "itraq",
            "silac",
            "olink",
            "somascan",
        ]
        text_fields = [
            metadata.get("title", ""),
            metadata.get("summary", ""),
            metadata.get("overall_design", ""),
        ]
        for field in text_fields:
            if any(kw.lower() in field.lower() for kw in proteomics_keywords):
                return True
        return False


def _create_recommended_strategy(
    strategy_config,
    analysis: dict,
    metadata: dict,
    url_data: "DownloadUrlResult",
) -> "StrategyConfig":
    """
    Convert DataExpertAssistant analysis to download_queue.StrategyConfig.

    Args:
        strategy_config: File-level strategy from extract_strategy_config()
        analysis: Analysis dict from analyze_download_strategy()
        metadata: GEO metadata dictionary
        url_data: DownloadUrlResult from GEOProvider.get_download_urls()
    """
    from lobster.core.schemas.download_queue import StrategyConfig

    # Primary strategy based on file availability
    if analysis.get("has_h5ad", False):
        strategy_name = "H5_FIRST"
        confidence = 0.95
        rationale = (
            f"H5AD file available with optimal single-file structure "
            f"({url_data.file_count} total files)"
        )
    elif analysis.get("has_processed_matrix", False):
        strategy_name = "MATRIX_FIRST"
        confidence = 0.85
        rationale = (
            f"Processed matrix available "
            f"({getattr(strategy_config, 'processed_matrix_name', 'unknown')})"
        )
    elif analysis.get("has_raw_matrix", False) or analysis.get(
        "raw_data_available", False
    ):
        strategy_name = "SAMPLES_FIRST"
        confidence = 0.75
        rationale = "Raw data available for full preprocessing control"
    else:
        strategy_name = "AUTO"
        confidence = 0.50
        rationale = "No clear optimal strategy detected, using auto-detection"

    # Concatenation strategy based on sample count
    n_samples = metadata.get("n_samples", metadata.get("sample_count", 0))
    platform = metadata.get("platform", "")

    if n_samples < 20 and platform:
        concatenation_strategy = "union"
        use_intersecting_genes_only = False
    elif n_samples >= 20:
        concatenation_strategy = "intersection"
        use_intersecting_genes_only = True
    else:
        concatenation_strategy = "auto"
        use_intersecting_genes_only = None

    # Execution parameters scaled by file count
    file_count = url_data.file_count
    if file_count > 100:
        timeout = 7200
        max_retries = 5
    elif file_count > 20:
        timeout = 3600
        max_retries = 3
    else:
        timeout = 1800
        max_retries = 3

    return StrategyConfig(
        strategy_name=strategy_name,
        concatenation_strategy=concatenation_strategy,
        confidence=confidence,
        rationale=rationale,
        strategy_params={"use_intersecting_genes_only": use_intersecting_genes_only},
        execution_params={
            "timeout": timeout,
            "max_retries": max_retries,
            "verify_checksum": True,
            "resume_enabled": False,
        },
    )


def _create_fallback_strategy(
    url_data: "DownloadUrlResult",
    metadata: dict,
) -> "StrategyConfig":
    """
    Create fallback strategy when LLM extraction fails.

    Uses data-type aware URL-based heuristics for strategy recommendation.
    """
    from lobster.core.schemas.download_queue import StrategyConfig

    is_single_cell = _is_single_cell_dataset(metadata)

    if url_data.h5_url:
        strategy_name = "H5_FIRST"
        confidence = 0.90
        rationale = "H5AD file URL found (single-cell optimized format)"
    elif is_single_cell and url_data.raw_files and len(url_data.raw_files) > 0:
        raw_urls = url_data.get_raw_urls_as_strings()
        has_mtx = any(
            "mtx" in url.lower() or "matrix" in url.lower() for url in raw_urls
        )
        if has_mtx:
            strategy_name = "RAW_FIRST"
            confidence = 0.80
            rationale = (
                f"Single-cell dataset with MTX files detected "
                f"({len(url_data.raw_files)} raw files)"
            )
        else:
            strategy_name = "SAMPLES_FIRST"
            confidence = 0.70
            rationale = (
                f"Single-cell dataset with raw data files "
                f"({len(url_data.raw_files)} files)"
            )
    elif url_data.matrix_url:
        if is_single_cell:
            strategy_name = "MATRIX_FIRST"
            confidence = 0.70
            rationale = "Single-cell dataset with matrix file (may be processed data)"
        else:
            strategy_name = "MATRIX_FIRST"
            confidence = 0.75
            rationale = "Matrix file URL found (bulk RNA-seq or processed data)"
    elif url_data.raw_files and len(url_data.raw_files) > 0:
        strategy_name = "SAMPLES_FIRST"
        confidence = 0.65
        rationale = f"Raw data URLs found ({len(url_data.raw_files)} files, bulk RNA-seq likely)"
    else:
        strategy_name = "AUTO"
        confidence = 0.50
        rationale = "No clear file pattern detected, using auto-detection"

    data_type_info = (
        " (single-cell dataset)" if is_single_cell else " (bulk/unknown dataset)"
    )
    rationale += data_type_info

    n_samples = metadata.get("n_samples", metadata.get("sample_count", 0))
    concatenation_strategy = "intersection" if n_samples >= 20 else "auto"

    return StrategyConfig(
        strategy_name=strategy_name,
        concatenation_strategy=concatenation_strategy,
        confidence=confidence,
        rationale=rationale,
        strategy_params={"use_intersecting_genes_only": None},
        execution_params={
            "timeout": 3600,
            "max_retries": 3,
            "verify_checksum": True,
            "resume_enabled": False,
        },
    )


# ---------------------------------------------------------------------------
# GEOQueuePreparer
# ---------------------------------------------------------------------------


class GEOQueuePreparer(IQueuePreparer):
    """
    Prepares download queue entries for GEO (Gene Expression Omnibus) datasets.

    Uses LLM-based strategy analysis via DataExpertAssistant with fallback
    to URL-based heuristics. GEO is the most complex preparer because it
    leverages LLM analysis for optimal download strategy.
    """

    # Lazy-loaded instances
    _geo_service = None
    _geo_provider = None
    _data_expert = None

    def supported_databases(self) -> List[str]:
        return ["geo"]

    def _get_geo_service(self):
        if self._geo_service is None:
            from lobster.services.data_access.geo_service import GEOService

            self._geo_service = GEOService(data_manager=self.data_manager)
            logger.debug("Lazy-loaded GEOService for GEOQueuePreparer")
        return self._geo_service

    def _get_geo_provider(self):
        if self._geo_provider is None:
            from lobster.tools.providers.geo_provider import GEOProvider

            self._geo_provider = GEOProvider(self.data_manager)
            logger.debug("Lazy-loaded GEOProvider for GEOQueuePreparer")
        return self._geo_provider

    def _get_data_expert(self):
        if self._data_expert is None:
            from lobster.agents.data_expert.assistant import DataExpertAssistant

            self._data_expert = DataExpertAssistant()
            logger.debug("Lazy-loaded DataExpertAssistant for GEOQueuePreparer")
        return self._data_expert

    def fetch_metadata(
        self, accession: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Fetch GEO metadata via GEOService, with metadata_store cache check."""
        # Check metadata_store cache first
        cached = self.data_manager.metadata_store.get(accession)
        if cached and cached.get("metadata"):
            logger.info(f"Using cached metadata for {accession}")
            return cached["metadata"], cached.get("validation_result")

        # Fetch fresh metadata
        geo_service = self._get_geo_service()
        metadata, validation_result = geo_service.fetch_metadata_only(accession)
        return metadata, validation_result

    def extract_download_urls(self, accession: str) -> "DownloadUrlResult":
        """Extract download URLs via GEOProvider."""
        geo_provider = self._get_geo_provider()
        url_data = geo_provider.get_download_urls(accession)

        if url_data.error:
            logger.warning(f"URL extraction warning for {accession}: {url_data.error}")

        return url_data

    @staticmethod
    def _derive_analysis(strategy_config, url_data: "DownloadUrlResult") -> dict:
        """
        Derive analysis dict from StrategyConfig fields and URL data.

        The DataExpertAssistant's StrategyConfig has fields like
        processed_matrix_name, raw_UMI_like_matrix_name, raw_data_available.
        We translate those into the analysis keys that _create_recommended_strategy
        expects (has_h5ad, has_processed_matrix, has_raw_matrix, raw_data_available).
        """
        # Support both Pydantic model and dict
        if isinstance(strategy_config, dict):
            get = strategy_config.get
        else:

            def get(k, d=None):
                return getattr(strategy_config, k, d)

        has_h5ad = bool(url_data.h5_url) or any(
            f.filename.endswith((".h5ad", ".h5")) for f in url_data.primary_files
        )
        has_processed_matrix = bool(get("processed_matrix_name", ""))
        has_raw_matrix = bool(get("raw_UMI_like_matrix_name", ""))
        raw_data_available = bool(get("raw_data_available", False))

        return {
            "has_h5ad": has_h5ad,
            "has_processed_matrix": has_processed_matrix,
            "has_raw_matrix": has_raw_matrix,
            "raw_data_available": raw_data_available,
        }

    def recommend_strategy(
        self,
        metadata: Dict[str, Any],
        url_data: "DownloadUrlResult",
        accession: str,
    ) -> "StrategyConfig":
        """
        Recommend strategy using LLM analysis with fallback to heuristics.

        Attempts DataExpertAssistant LLM extraction first. On failure,
        falls back to URL-based heuristics.
        """
        try:
            assistant = self._get_data_expert()

            # Check if strategy_config is already cached
            cached = self.data_manager.metadata_store.get(accession, {})
            cached_strategy_config = (
                cached.get("strategy_config") if isinstance(cached, dict) else None
            )

            if not cached_strategy_config:
                # Extract via LLM
                logger.info(f"Extracting download strategy for {accession} via LLM")
                strategy_config = assistant.extract_strategy_config(metadata, accession)

                if strategy_config:
                    # Persist to metadata_store
                    self.data_manager._store_geo_metadata(
                        geo_id=accession,
                        metadata=metadata,
                        stored_by="geo_queue_preparer",
                        strategy_config=(
                            strategy_config.model_dump()
                            if hasattr(strategy_config, "model_dump")
                            else strategy_config
                        ),
                    )

                    analysis = self._derive_analysis(strategy_config, url_data)
                    return _create_recommended_strategy(
                        strategy_config, analysis, metadata, url_data
                    )
                else:
                    return _create_fallback_strategy(url_data, metadata)
            else:
                # Use existing cached strategy config
                analysis = self._derive_analysis(cached_strategy_config, url_data)
                return _create_recommended_strategy(
                    cached_strategy_config, analysis, metadata, url_data
                )

        except Exception as e:
            logger.warning(
                f"LLM strategy extraction failed for {accession}: {e}, "
                "using URL-based fallback"
            )
            return _create_fallback_strategy(url_data, metadata)
