"""
Queue Preparation Service — central routing for database-specific queue preparers.

Mirrors the DownloadOrchestrator pattern for the preparation side. Uses
AccessionResolver to detect database type from accession, then routes
to the appropriate IQueuePreparer implementation.

Example usage:
    >>> from lobster.services.data_access.queue_preparation_service import (
    ...     QueuePreparationService,
    ... )
    >>> from lobster.core.data_manager_v2 import DataManagerV2
    >>>
    >>> dm = DataManagerV2(workspace_dir="./workspace")
    >>> service = QueuePreparationService(dm)
    >>> result = service.prepare("PXD063610")
    >>> print(result.queue_entry.entry_id)
    >>> print(result.validation_summary)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from lobster.core.interfaces.queue_preparer import (
    IQueuePreparer,
    QueuePreparationResult,
)
from lobster.utils.logger import get_logger

if TYPE_CHECKING:
    from lobster.core.data_manager_v2 import DataManagerV2

logger = get_logger(__name__)


class PreparerNotFoundError(Exception):
    """Raised when no queue preparer is registered for a database type."""

    def __init__(self, database: str, available_databases: List[str]):
        self.database = database
        self.available_databases = available_databases
        message = (
            f"No queue preparer registered for database '{database}'. "
            f"Available databases: {', '.join(available_databases) or 'none'}"
        )
        super().__init__(message)


# Mapping from AccessionResolver field names to database keys used by preparers.
# AccessionResolver.detect_field() returns field names like "geo_accession",
# "pride_accession", etc. We map those to the database keys our preparers use.
# Only map accession types that providers actually support.
# GSM (sample) and GPL (platform) are excluded — GEOProvider only handles GSE/GDS.
# SRA sample accessions (SRS*) excluded — SRAProvider needs study/experiment/run level.
_FIELD_TO_DATABASE = {
    "geo_accession": "geo",  # GSE*
    "geo_dataset_accession": "geo",  # GDS*
    "pride_accession": "pride",  # PXD*
    "massive_accession": "massive",  # MSV*
    "sra_study_accession": "sra",  # SRP*
    "sra_experiment_accession": "sra",  # SRX*
    "sra_run_accession": "sra",  # SRR*
    "ena_study_accession": "sra",  # ERP* → routes through SRA preparer
    "ena_experiment_accession": "sra",  # ERX*
    "ena_run_accession": "sra",  # ERR*
    "metabolights_accession": "metabolights",  # MTBLS*
    "metabolomics_workbench_accession": "metabolomics_workbench",  # ST*
}


class QueuePreparationService:
    """
    Central router for database-specific queue preparers.

    Routes queue preparation requests to appropriate IQueuePreparer
    implementations based on accession pattern detection.

    Mirrors the DownloadOrchestrator pattern on the preparation side.

    Attributes:
        data_manager: DataManagerV2 instance
        _preparers: Registry mapping database names to preparers
    """

    def __init__(
        self,
        data_manager: "DataManagerV2",
        workspace_path: Optional[str] = None,
    ):
        """
        Initialize QueuePreparationService with data manager.

        Args:
            data_manager: DataManagerV2 instance
            workspace_path: Optional workspace path (unused, for API symmetry)
        """
        self.data_manager = data_manager
        self._preparers: Dict[str, IQueuePreparer] = {}
        self._register_default_preparers()
        logger.info("QueuePreparationService initialized")

    def _register_default_preparers(self) -> None:
        """
        Register default queue preparers for all supported databases.

        Phase 1: Entry-point discovery (external/plugin queue preparers)
        Phase 2: Hardcoded defaults (GEO, PRIDE, SRA, MassIVE)

        Uses lazy imports with try/except to handle missing dependencies.
        """
        # Phase 1: Discover queue preparers from entry points
        try:
            from lobster.core.component_registry import component_registry

            for name, preparer_cls in component_registry.list_queue_preparers().items():
                try:
                    preparer = preparer_cls(self.data_manager)
                    self.register_preparer(preparer)
                    logger.debug(f"Registered queue preparer '{name}' from entry point")
                except Exception as e:
                    logger.warning(
                        f"Failed to load queue preparer '{name}' "
                        f"from entry point: {e}"
                    )
        except Exception as e:
            logger.debug(f"Entry point queue preparer discovery skipped: {e}")

        # Phase 2: Hardcoded fallback preparers

        # GEO
        try:
            from lobster.services.data_access.geo_queue_preparer import (
                GEOQueuePreparer,
            )

            self.register_preparer(GEOQueuePreparer(self.data_manager))
            logger.info("Auto-registered GEOQueuePreparer")
        except ImportError as e:
            logger.warning(f"GEOQueuePreparer not available: {e}")

        # PRIDE
        try:
            from lobster.services.data_access.pride_queue_preparer import (
                PRIDEQueuePreparer,
            )

            self.register_preparer(PRIDEQueuePreparer(self.data_manager))
            logger.info("Auto-registered PRIDEQueuePreparer")
        except ImportError as e:
            logger.warning(f"PRIDEQueuePreparer not available: {e}")

        # SRA
        try:
            from lobster.services.data_access.sra_queue_preparer import (
                SRAQueuePreparer,
            )

            self.register_preparer(SRAQueuePreparer(self.data_manager))
            logger.info("Auto-registered SRAQueuePreparer")
        except ImportError as e:
            logger.warning(f"SRAQueuePreparer not available: {e}")

        # MassIVE
        try:
            from lobster.services.data_access.massive_queue_preparer import (
                MassIVEQueuePreparer,
            )

            self.register_preparer(MassIVEQueuePreparer(self.data_manager))
            logger.info("Auto-registered MassIVEQueuePreparer")
        except ImportError as e:
            logger.warning(f"MassIVEQueuePreparer not available: {e}")

    def register_preparer(self, preparer: IQueuePreparer) -> None:
        """
        Register a queue preparer for one or more databases.

        Args:
            preparer: IQueuePreparer implementation to register

        Raises:
            ValueError: If preparer does not support any databases
        """
        supported_dbs = preparer.supported_databases()

        if not supported_dbs:
            raise ValueError(
                f"Preparer {preparer.__class__.__name__} does not declare "
                "any supported databases"
            )

        for db in supported_dbs:
            db_lower = db.lower()
            if db_lower in self._preparers:
                logger.warning(
                    f"Overwriting existing preparer for database '{db}' "
                    f"(old: {self._preparers[db_lower].__class__.__name__}, "
                    f"new: {preparer.__class__.__name__})"
                )
            self._preparers[db_lower] = preparer
            logger.info(f"Registered {preparer.__class__.__name__} for database '{db}'")

    def detect_database(self, accession: str) -> Optional[str]:
        """
        Detect which database an accession belongs to.

        Uses AccessionResolver to identify the accession type, then maps
        to the database key used by our preparers.

        Args:
            accession: Dataset accession (e.g., "PXD063610", "GSE180759")

        Returns:
            Database key (e.g., "pride", "geo", "sra") or None if unknown
        """
        from lobster.core.identifiers import get_accession_resolver

        resolver = get_accession_resolver()
        field = resolver.detect_field(accession)

        if field is None:
            logger.warning(f"AccessionResolver could not identify: {accession}")
            return None

        database = _FIELD_TO_DATABASE.get(field)
        if database is None:
            logger.warning(
                f"No preparer mapping for field '{field}' (accession: {accession})"
            )
            return None

        return database

    def get_preparer_for_database(self, database: str) -> Optional[IQueuePreparer]:
        """
        Get registered preparer for database type.

        Args:
            database: Database identifier (case-insensitive)

        Returns:
            Registered IQueuePreparer instance, or None if not found
        """
        return self._preparers.get(database.lower())

    def list_supported_databases(self) -> List[str]:
        """
        List all databases supported by registered preparers.

        Returns:
            Sorted list of database identifiers (lowercase)
        """
        return sorted(self._preparers.keys())

    def prepare(
        self,
        accession: str,
        database: Optional[str] = None,
        priority: int = 5,
    ) -> QueuePreparationResult:
        """
        Prepare a download queue entry for any supported database.

        Detects database from accession pattern (or uses explicit database),
        routes to appropriate preparer, and returns a complete queue entry.

        Args:
            accession: Dataset accession (e.g., "PXD063610", "GSE180759")
            database: Optional explicit database override (skips detection)
            priority: Download priority (1=highest, 10=lowest)

        Returns:
            QueuePreparationResult with queue_entry ready to add to queue

        Raises:
            PreparerNotFoundError: If no preparer registered for the database
            ValueError: If accession cannot be identified
        """
        # Detect database if not provided
        if database is None:
            database = self.detect_database(accession)
            if database is None:
                raise ValueError(
                    f"Cannot detect database for accession '{accession}'. "
                    f"Supported databases: {', '.join(self.list_supported_databases())}"
                )

        database = database.lower()
        logger.info(f"Preparing queue entry for {accession} (database={database})")

        # Find preparer
        preparer = self.get_preparer_for_database(database)
        if preparer is None:
            raise PreparerNotFoundError(database, self.list_supported_databases())

        # Delegate to preparer
        result = preparer.prepare_queue_entry(accession, priority=priority)

        logger.info(
            f"Queue entry prepared: {result.queue_entry.entry_id} "
            f"| {result.validation_summary}"
        )

        return result
