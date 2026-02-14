"""
MassIVE Queue Preparer — prepares download queue entries for MassIVE (MSV*) datasets.

Uses heuristic strategy recommendation (no LLM needed). Note that MassIVE's
PROXI v0.1 API doesn't provide file listings — actual file discovery happens
during FTP directory scan at download time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from lobster.core.interfaces.queue_preparer import IQueuePreparer
from lobster.utils.logger import get_logger

if TYPE_CHECKING:
    from lobster.core.data_manager_v2 import DataManagerV2
    from lobster.core.schemas.download_queue import StrategyConfig
    from lobster.core.schemas.download_urls import DownloadUrlResult

logger = get_logger(__name__)


class MassIVEQueuePreparer(IQueuePreparer):
    """
    Prepares download queue entries for MassIVE mass spectrometry datasets.

    MassIVE has limited API support (PROXI v0.1). File listing happens at
    download time via FTP scan. Strategy is RESULT_FIRST if processed files
    exist, else RAW_FIRST with ftp_base for directory traversal.
    """

    _massive_provider = None

    def supported_databases(self) -> List[str]:
        return ["massive"]

    def _get_massive_provider(self):
        if self._massive_provider is None:
            from lobster.tools.providers.massive_provider import MassIVEProvider

            self._massive_provider = MassIVEProvider(self.data_manager)
            logger.debug("Lazy-loaded MassIVEProvider for MassIVEQueuePreparer")
        return self._massive_provider

    def fetch_metadata(
        self, accession: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Fetch MassIVE dataset metadata via PROXI API."""
        provider = self._get_massive_provider()
        metadata = provider.get_dataset_metadata(accession)
        return metadata, None

    def extract_download_urls(self, accession: str) -> "DownloadUrlResult":
        """
        Extract download URLs via MassIVEProvider.

        Note: MassIVE PROXI v0.1 typically returns only ftp_base.
        File listing happens during actual download (FTP scan).
        """
        provider = self._get_massive_provider()
        return provider.get_download_urls(accession)

    def recommend_strategy(
        self,
        metadata: Dict[str, Any],
        url_data: "DownloadUrlResult",
        accession: str,
    ) -> "StrategyConfig":
        """
        Recommend MassIVE download strategy.

        RESULT_FIRST if processed files detected, else RAW_FIRST with
        ftp_base for directory traversal at download time.
        """
        from lobster.core.schemas.download_queue import StrategyConfig

        if url_data.processed_files:
            strategy_name = "RESULT_FIRST"
            confidence = 0.80
            rationale = (
                f"Processed result files available "
                f"({len(url_data.processed_files)} files)"
            )
        elif url_data.ftp_base:
            strategy_name = "RAW_FIRST"
            confidence = 0.65
            rationale = f"FTP base available for directory scan ({url_data.ftp_base})"
        else:
            # Use RAW_FIRST (not AUTO) since MassIVEDownloadService
            # doesn't implement AUTO strategy
            strategy_name = "RAW_FIRST"
            confidence = 0.40
            rationale = (
                "No file listing or FTP base available; will attempt raw download"
            )

        return StrategyConfig(
            strategy_name=strategy_name,
            concatenation_strategy="auto",
            confidence=confidence,
            rationale=rationale,
            strategy_params={
                "ftp_base": url_data.ftp_base,
            },
            execution_params={
                "timeout": 3600,
                "max_retries": 3,
                "verify_checksum": True,
                "resume_enabled": False,
            },
        )
