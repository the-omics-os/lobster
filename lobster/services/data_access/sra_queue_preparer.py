"""
SRA Queue Preparer — prepares download queue entries for SRA (SRR*/SRP*) datasets.

Uses heuristic strategy recommendation (no LLM needed). Strategy is always
FASTQ_FIRST with layout/platform metadata from the ENA filereport API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from lobster.core.interfaces.queue_preparer import IQueuePreparer
from lobster.utils.logger import get_logger

if TYPE_CHECKING:
    from lobster.core.schemas.download_queue import StrategyConfig
    from lobster.core.schemas.download_urls import DownloadUrlResult

logger = get_logger(__name__)


class SRAQueuePreparer(IQueuePreparer):
    """
    Prepares download queue entries for SRA sequencing datasets.

    SRA datasets are always FASTQ-based. Strategy is straightforward:
    FASTQ_FIRST with paired/single layout info and timeout scaled by run count.
    """

    _sra_provider = None

    def supported_databases(self) -> List[str]:
        return ["sra", "ena"]

    def _get_sra_provider(self):
        if self._sra_provider is None:
            from lobster.tools.providers.sra_provider import SRAProvider

            self._sra_provider = SRAProvider(self.data_manager)
            logger.debug("Lazy-loaded SRAProvider for SRAQueuePreparer")
        return self._sra_provider

    def fetch_metadata(
        self, accession: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Fetch SRA metadata via SRAProvider."""
        provider = self._get_sra_provider()

        try:
            pub_metadata = provider.extract_publication_metadata(accession)
            # Convert PublicationMetadata to dict
            metadata = {
                "title": pub_metadata.title,
                "abstract": pub_metadata.abstract,
                "journal": pub_metadata.journal,
                "published": pub_metadata.published,
                "keywords": pub_metadata.keywords,
                "uid": pub_metadata.uid,
            }
        except Exception as e:
            logger.warning(f"SRA metadata extraction failed for {accession}: {e}")
            metadata = {"accession": accession, "error": str(e)}

        return metadata, None

    def extract_download_urls(self, accession: str) -> "DownloadUrlResult":
        """Extract download URLs via SRAProvider (ENA filereport API)."""
        provider = self._get_sra_provider()
        return provider.get_download_urls(accession)

    def recommend_strategy(
        self,
        metadata: Dict[str, Any],
        url_data: "DownloadUrlResult",
        accession: str,
    ) -> "StrategyConfig":
        """
        Recommend SRA download strategy.

        SRA is always FASTQ_FIRST. Layout (PAIRED/SINGLE) and platform
        influence execution parameters.
        """
        from lobster.core.schemas.download_queue import StrategyConfig

        layout = url_data.layout or "UNKNOWN"
        platform = url_data.platform or "UNKNOWN"
        run_count = url_data.run_count or 1

        # Confidence based on how much info we have
        if url_data.raw_files:
            confidence = 0.90
            rationale = (
                f"FASTQ files available ({len(url_data.raw_files)} files, "
                f"layout={layout}, platform={platform})"
            )
        else:
            confidence = 0.60
            rationale = (
                f"No direct FASTQ URLs found, will attempt download "
                f"(layout={layout}, platform={platform})"
            )

        # Scale timeout by run count
        if run_count > 50:
            timeout = 7200
            max_retries = 5
        elif run_count > 10:
            timeout = 3600
            max_retries = 3
        else:
            timeout = 1800
            max_retries = 3

        return StrategyConfig(
            strategy_name="FASTQ_FIRST",
            concatenation_strategy="auto",
            confidence=confidence,
            rationale=rationale,
            strategy_params={
                "layout": layout,
                "platform": platform,
                "run_count": run_count,
                "mirror": url_data.mirror or "ena",
            },
            execution_params={
                "timeout": timeout,
                "max_retries": max_retries,
                "verify_checksum": True,
                "resume_enabled": False,
            },
        )
