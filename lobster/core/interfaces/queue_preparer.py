"""
Queue Preparer Interface for symmetric download queue preparation.

This module defines the abstract interface for database-specific queue preparation
implementations, enabling a clean separation between queue preparation and download
execution. Mirrors the IDownloadService pattern.

The QueuePreparationService uses this interface to:
1. Detect which preparer handles a given database type
2. Fetch metadata from the database
3. Extract download URLs
4. Recommend a download strategy
5. Create a complete DownloadQueueEntry

Usage Example:
    >>> from lobster.core.interfaces.queue_preparer import IQueuePreparer
    >>> from lobster.core.data_manager_v2 import DataManagerV2
    >>>
    >>> class PRIDEQueuePreparer(IQueuePreparer):
    ...     def supported_databases(self) -> List[str]:
    ...         return ["pride"]
    ...     # ... implement abstract methods
    >>>
    >>> dm = DataManagerV2(workspace_dir="./workspace")
    >>> preparer = PRIDEQueuePreparer(dm)
    >>> result = preparer.prepare_queue_entry("PXD063610")
    >>> print(result.queue_entry.entry_id)
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from lobster.core.data_manager_v2 import DataManagerV2
    from lobster.core.schemas.download_queue import (
        DownloadQueueEntry,
        StrategyConfig,
    )
    from lobster.core.schemas.download_urls import DownloadUrlResult


@dataclass
class QueuePreparationResult:
    """
    Result of preparing a download queue entry.

    Bundles the queue entry with supporting data for logging and debugging.

    Attributes:
        queue_entry: The prepared DownloadQueueEntry ready to add to queue
        url_data: DownloadUrlResult from the provider (for inspection/logging)
        metadata: Raw metadata dict from the database
        validation_summary: Human-readable summary of preparation
    """

    queue_entry: "DownloadQueueEntry"
    url_data: Optional["DownloadUrlResult"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_summary: str = ""


class IQueuePreparer(ABC):
    """
    Abstract base class for database-specific queue preparation.

    This interface defines the contract that all queue preparer implementations
    must follow to integrate with the QueuePreparationService. Each implementation
    handles preparing download queue entries for a specific bioinformatics database
    (GEO, SRA, PRIDE, MassIVE, etc.).

    The template method `prepare_queue_entry()` calls three abstract methods
    in sequence: fetch_metadata → extract_download_urls → recommend_strategy,
    then assembles the DownloadQueueEntry. Subclasses only implement these
    three methods.

    Attributes:
        data_manager (DataManagerV2): Data manager for metadata caching and queue access.
    """

    def __init__(self, data_manager: "DataManagerV2"):
        """
        Initialize the queue preparer with a data manager.

        Args:
            data_manager: DataManagerV2 instance for metadata caching and queue access
        """
        self.data_manager = data_manager

    @abstractmethod
    def supported_databases(self) -> List[str]:
        """
        Get list of database identifiers this preparer handles.

        Returns:
            List[str]: Database identifiers (lowercase). Examples:
                - ["geo"]
                - ["pride"]
                - ["sra"]
                - ["massive"]
        """
        ...

    @abstractmethod
    def fetch_metadata(
        self, accession: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Fetch metadata for a dataset from the database.

        Args:
            accession: Dataset accession (e.g., "GSE180759", "PXD063610")

        Returns:
            Tuple of:
                - metadata: Dict with database-specific metadata
                - validation_info: Optional dict with validation details (can be None)
        """
        ...

    @abstractmethod
    def extract_download_urls(self, accession: str) -> "DownloadUrlResult":
        """
        Extract download URLs for a dataset.

        Args:
            accession: Dataset accession

        Returns:
            DownloadUrlResult with categorized download URLs
        """
        ...

    @abstractmethod
    def recommend_strategy(
        self,
        metadata: Dict[str, Any],
        url_data: "DownloadUrlResult",
        accession: str,
    ) -> "StrategyConfig":
        """
        Recommend a download strategy based on metadata and available URLs.

        Args:
            metadata: Dataset metadata from fetch_metadata()
            url_data: Download URLs from extract_download_urls()
            accession: Dataset accession for context

        Returns:
            StrategyConfig with recommended strategy, confidence, and rationale
        """
        ...

    def prepare_queue_entry(
        self, accession: str, priority: int = 5
    ) -> QueuePreparationResult:
        """
        Prepare a complete download queue entry (template method).

        Calls fetch_metadata → extract_download_urls → recommend_strategy
        in sequence, then assembles the DownloadQueueEntry.

        Args:
            accession: Dataset accession (e.g., "PXD063610", "GSE180759")
            priority: Download priority (1=highest, 10=lowest)

        Returns:
            QueuePreparationResult with queue_entry, url_data, metadata, and summary
        """
        from lobster.core.schemas.download_queue import (
            DownloadQueueEntry,
            DownloadStatus,
        )

        # Step 1: Fetch metadata
        metadata, validation_info = self.fetch_metadata(accession)

        # Step 2: Extract download URLs
        url_data = self.extract_download_urls(accession)

        # Step 3: Recommend strategy
        recommended_strategy = self.recommend_strategy(metadata, url_data, accession)

        # Step 4: Assemble queue entry
        database = self.supported_databases()[0]
        entry_id = f"queue_{accession}_{uuid.uuid4().hex[:8]}"

        # Bridge URL data to queue entry fields
        queue_fields = url_data.to_queue_entry_fields()

        queue_entry = DownloadQueueEntry(
            entry_id=entry_id,
            dataset_id=accession,
            database=database,
            priority=priority,
            status=DownloadStatus.PENDING,
            metadata=metadata,
            validation_result=validation_info,
            recommended_strategy=recommended_strategy,
            matrix_url=queue_fields.get("matrix_url"),
            raw_urls=queue_fields.get("raw_urls", []),
            supplementary_urls=queue_fields.get("supplementary_urls", []),
            h5_url=queue_fields.get("h5_url"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Build summary
        summary_parts = [
            f"Database: {database.upper()}",
            f"Accession: {accession}",
            f"Strategy: {recommended_strategy.strategy_name} "
            f"(confidence: {recommended_strategy.confidence:.2f})",
            f"Files: {url_data.file_count}",
        ]
        if recommended_strategy.rationale:
            summary_parts.append(f"Rationale: {recommended_strategy.rationale}")

        return QueuePreparationResult(
            queue_entry=queue_entry,
            url_data=url_data,
            metadata=metadata,
            validation_summary=" | ".join(summary_parts),
        )
