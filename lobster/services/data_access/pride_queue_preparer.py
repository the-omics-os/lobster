"""
PRIDE Queue Preparer — prepares download queue entries for PRIDE (PXD*) datasets.

Uses heuristic strategy recommendation (no LLM needed). Strategy priority:
processed_files → mzML → search_files → raw_files.
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


class PRIDEQueuePreparer(IQueuePreparer):
    """
    Prepares download queue entries for PRIDE proteomics datasets.

    PRIDE datasets have a distinct file hierarchy: processed results,
    search engine outputs, mzML peak lists, and raw instrument files.
    Strategy is determined by which file types are available.
    """

    _pride_provider = None

    def supported_databases(self) -> List[str]:
        return ["pride", "proteomexchange"]

    def _get_pride_provider(self):
        if self._pride_provider is None:
            from lobster.tools.providers.pride_provider import PRIDEProvider

            self._pride_provider = PRIDEProvider(self.data_manager)
            logger.debug("Lazy-loaded PRIDEProvider for PRIDEQueuePreparer")
        return self._pride_provider

    def fetch_metadata(
        self, accession: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Fetch PRIDE project metadata via PRIDEProvider."""
        provider = self._get_pride_provider()
        metadata = provider.get_project_metadata(accession)

        # Extract organism info for summary
        try:
            from lobster.tools.providers.pride_normalizer import PRIDENormalizer

            organisms = metadata.get("organisms", [])
            organism_names = [
                PRIDENormalizer.safe_get_organism_name(org) for org in organisms
            ]
            metadata["_organism_names"] = organism_names
        except Exception:
            pass

        return metadata, None

    def extract_download_urls(self, accession: str) -> "DownloadUrlResult":
        """Extract download URLs via PRIDEProvider."""
        provider = self._get_pride_provider()
        return provider.get_download_urls(accession)

    def recommend_strategy(
        self,
        metadata: Dict[str, Any],
        url_data: "DownloadUrlResult",
        accession: str,
    ) -> "StrategyConfig":
        """
        Recommend PRIDE download strategy based on available file types.

        Priority: RESULT_FIRST > MZML_FIRST > SEARCH_FIRST > RAW_FIRST.
        """
        from lobster.core.schemas.download_queue import StrategyConfig

        # Check for processed result files (best case)
        if url_data.processed_files:
            strategy_name = "RESULT_FIRST"
            confidence = 0.90
            rationale = (
                f"Processed result files available "
                f"({len(url_data.processed_files)} files)"
            )
        # Check for mzML files in raw_files (converted peak lists)
        elif url_data.raw_files and any(
            f.filename.lower().endswith(".mzml")
            or f.filename.lower().endswith(".mzml.gz")
            for f in url_data.raw_files
        ):
            strategy_name = "MZML_FIRST"
            confidence = 0.80
            mzml_count = sum(
                1
                for f in url_data.raw_files
                if f.filename.lower().endswith((".mzml", ".mzml.gz"))
            )
            rationale = f"mzML peak list files available ({mzml_count} files)"
        # Check for search engine output files
        elif url_data.search_files:
            strategy_name = "SEARCH_FIRST"
            confidence = 0.75
            rationale = (
                f"Search engine output files available "
                f"({len(url_data.search_files)} files)"
            )
        # Only raw instrument files
        elif url_data.raw_files:
            strategy_name = "RAW_FIRST"
            confidence = 0.60
            rationale = (
                f"Only raw instrument files available "
                f"({len(url_data.raw_files)} files)"
            )
        else:
            strategy_name = "AUTO"
            confidence = 0.40
            rationale = "No downloadable files found, using auto-detection"

        # Estimate timeout based on file count and size
        total_files = url_data.file_count
        if total_files > 50:
            timeout = 7200
            max_retries = 5
        elif total_files > 10:
            timeout = 3600
            max_retries = 3
        else:
            timeout = 1800
            max_retries = 3

        return StrategyConfig(
            strategy_name=strategy_name,
            concatenation_strategy="auto",
            confidence=confidence,
            rationale=rationale,
            strategy_params={
                "file_type_priority": strategy_name.replace("_FIRST", "").lower(),
            },
            execution_params={
                "timeout": timeout,
                "max_retries": max_retries,
                "verify_checksum": True,
                "resume_enabled": False,
            },
        )
