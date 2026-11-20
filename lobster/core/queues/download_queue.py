"""
Download queue management with JSON Lines persistence.

This module provides the DownloadQueue class for managing dataset download
requests with atomic operations, automatic backups, and thread-safe access.
"""

import json
import logging
import os
import shutil
import socket
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import psutil

from lobster.core.figures.errors import ManagedPathError
from lobster.core.figures.export_dirs import assert_dirs_allowed
from lobster.core.figures.export_guard import (
    MANAGED_EXPORT_REASON,
    assert_export_allowed,
)
from lobster.core.queues.queue_storage import (
    atomic_write_jsonl,
    backups_enabled,
    queue_file_lock,
)
from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus

logger = logging.getLogger(__name__)

# A claim that recorded no process (written before claims did) counts as left
# behind once it is this old.
UNOWNED_CLAIM_STALE_AFTER = timedelta(hours=1)


def _standard_workspace(queue_file: Path) -> Optional[Path]:
    """The workspace of a queue at the data manager's fixed location
    ``<workspace>/.lobster/queues/<file>`` (lexical; None for any other location).

    A relative ``queue_file`` is first joined to the current directory (no link is
    resolved), so a shortened spelling of that location is recognised too."""
    spelled = queue_file if queue_file.is_absolute() else Path.cwd() / queue_file
    folder = spelled.parent
    if folder.name == "queues" and folder.parent.name == ".lobster":
        return folder.parent.parent
    return None


def _copy_to_new_file(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target`` like ``shutil.copy2``, creating ``target`` as a NEW
    file: an existing entry of any kind (a link, dangling included, is never followed)
    is a typed refusal (fixed reason), so a backup never writes through a link."""
    with open(source, "rb") as src:  # the source first: nothing created if unreadable
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            raise ManagedPathError(MANAGED_EXPORT_REASON) from None
        try:
            try:
                dst = os.fdopen(fd, "wb")
            except BaseException:
                os.close(fd)  # not handed over: close it here
                raise
            with dst:
                shutil.copyfileobj(src, dst)
            shutil.copystat(source, target)
        except BaseException:
            _remove_new_file(target)  # no partial backup is left behind
            raise


def _remove_new_file(path: Path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class DownloadQueueError(Exception):
    """Base exception for download queue operations."""

    pass


class EntryNotFoundError(DownloadQueueError):
    """Raised when a queue entry is not found."""

    pass


class DownloadQueue:
    """
    Thread-safe download queue with JSON Lines persistence.

    This class manages a queue of dataset download requests with automatic
    persistence to disk, atomic operations, backup functionality, and
    thread-safe access for concurrent operations.

    Features:
        - JSON Lines (.jsonl) format for append-only durability
        - Automatic backups before modifications
        - Schema validation on read/write
        - Thread-safe operations with file locking
        - Efficient filtering by status
        - Atomic writes (temp file + rename)

    Attributes:
        queue_file: Path to JSON Lines queue file
        backup_dir: Directory for backup files
    """

    def __init__(self, queue_file: Path, *, guard_workspace: Optional[Path] = None):
        """
        Initialize download queue with JSON Lines file.

        Args:
            queue_file: Path to queue file (will be created if doesn't exist)
            guard_workspace: Workspace whose managed figure folder the queue never
                writes into (default: the workspace of a queue at the data manager's
                location ``<workspace>/.lobster/queues/``; elsewhere: not checked)

        Raises:
            ManagedPathError: The queue, its lock or one of its folders is inside
                the managed figure folder (nothing is created)
        """
        self.queue_file = Path(queue_file)
        # Use shared backup directory in queues folder
        self.backup_dir = self.queue_file.parent / "backups"
        self._lock = threading.Lock()
        self._file_lock_path = self.queue_file.with_suffix(".lock")
        self.guard_workspace = (
            Path(guard_workspace)
            if guard_workspace is not None
            else _standard_workspace(self.queue_file)
        )
        self._assert_writable()

        # Create directories if they don't exist
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize empty queue file if it doesn't exist
        if not self.queue_file.exists():
            self.queue_file.touch()

        logger.debug(f"Initialized DownloadQueue at {self.queue_file}")

    def _assert_writable(self) -> None:
        """Raise ManagedPathError (fixed reason) before a queue write when the queue
        file, its lock, its folder or its backup folder is inside the guarded
        workspace's managed figure folder."""
        if self.guard_workspace is None:
            return
        for leaf in (self._file_lock_path, self.queue_file):
            assert_export_allowed(self.guard_workspace, leaf)
        for folder in (self.queue_file.parent, self.backup_dir):
            assert_dirs_allowed(self.guard_workspace, folder)

    @contextmanager
    def _locked(self):
        # Every queue write (lock file, backup, temp file, replace, clear) is made
        # under this lock: checked first.
        self._assert_writable()
        with queue_file_lock(self._lock, self._file_lock_path):
            yield

    def add_entry(self, entry: DownloadQueueEntry) -> str:
        """
        Add entry to queue with atomic write.

        Args:
            entry: DownloadQueueEntry to add

        Returns:
            str: Entry ID of added entry

        Raises:
            DownloadQueueError: If entry already exists or write fails
        """
        with self._locked():
            # Check if entry already exists
            existing_entries = self._load_entries()
            if any(e.entry_id == entry.entry_id for e in existing_entries):
                raise DownloadQueueError(
                    f"Entry with ID '{entry.entry_id}' already exists"
                )

            # Backup before modification
            self._backup_queue()

            # Atomic rewrite ensures consistent file state across processes
            try:
                existing_entries.append(entry)
                self._write_entries_atomic(existing_entries)
                logger.debug(f"Added entry {entry.entry_id} to queue")
                return entry.entry_id

            except Exception as e:
                logger.error(f"Failed to add entry to queue: {e}")
                raise DownloadQueueError(f"Failed to add entry: {e}") from e

    def get_entry(self, entry_id: str) -> DownloadQueueEntry:
        """
        Retrieve specific entry by ID.

        Args:
            entry_id: Unique entry identifier

        Returns:
            DownloadQueueEntry: Retrieved entry

        Raises:
            EntryNotFoundError: If entry not found
        """
        with self._lock:
            entries = self._load_entries()
            for entry in entries:
                if entry.entry_id == entry_id:
                    return entry

            raise EntryNotFoundError(f"Entry '{entry_id}' not found in queue")

    def update_status(
        self,
        entry_id: str,
        status: DownloadStatus,
        expected_current_status: Optional[DownloadStatus] = None,
        modality_name: Optional[str] = None,
        error: Optional[str] = None,
        downloaded_by: Optional[str] = None,
    ) -> Optional[DownloadQueueEntry]:
        """
        Update entry status and optional fields with optional CAS check.

        When ``expected_current_status`` is provided, the update only proceeds
        if the entry's current status matches. If it does not match, the method
        returns ``None`` (compare-and-swap failure) without modifying the entry.
        This prevents duplicate workers from claiming the same queue entry.

        When ``expected_current_status`` is ``None`` (default), the method
        behaves as before -- any status transition is allowed (backward compat).

        Args:
            entry_id: Entry to update
            status: New status
            expected_current_status: If provided, CAS precondition -- current
                status must match for the update to proceed
            modality_name: Optional modality name
            error: Optional error message
            downloaded_by: Optional agent/user identifier

        Returns:
            Updated DownloadQueueEntry on success, or None if CAS
            precondition failed

        Raises:
            EntryNotFoundError: If entry not found
            DownloadQueueError: If update fails
        """
        with self._locked():
            # Load all entries
            entries = self._load_entries()
            entry_found = False
            updated_entry = None

            # Find and update entry
            for entry in entries:
                if entry.entry_id == entry_id:
                    entry_found = True

                    # CAS check
                    if expected_current_status is not None:
                        # Handle both enum and string comparison
                        current_val = (
                            entry.status.value
                            if hasattr(entry.status, "value")
                            else entry.status
                        )
                        expected_val = (
                            expected_current_status.value
                            if hasattr(expected_current_status, "value")
                            else expected_current_status
                        )
                        if current_val != expected_val:
                            logger.info(
                                f"CAS failed for entry '{entry_id}': "
                                f"expected {expected_val}, actual {current_val}"
                            )
                            return None

                    entry.update_status(
                        status=status,
                        modality_name=modality_name,
                        error=error,
                        downloaded_by=downloaded_by,
                    )
                    if getattr(status, "value", status) == "in_progress":
                        # The claimer: reclaim_stale tells a dead claim from a live one.
                        entry.claimed_pid = os.getpid()
                        entry.claimed_host = socket.gethostname()
                    updated_entry = entry
                    break

            if not entry_found:
                raise EntryNotFoundError(f"Entry '{entry_id}' not found in queue")

            # Backup before modification
            self._backup_queue()

            # Write all entries atomically
            self._write_entries_atomic(entries)

            logger.debug(f"Updated entry {entry_id} status to {status}")
            return updated_entry

    def reclaim_stale(self, entry_id: str) -> Optional[str]:
        """
        Take back an in_progress claim left by a run that is gone.

        A claim is stale when its process is no longer running on this host,
        or when it recorded no process and is older than
        UNOWNED_CLAIM_STALE_AFTER. A claim from another host is never taken.
        The entry goes to FAILED with a note, so it can be downloaded again.

        Returns:
            The line to show for a claim taken back, else None.
        """
        with self._locked():
            entries = self._load_entries()
            entry = next((e for e in entries if e.entry_id == entry_id), None)
            if entry is None or getattr(entry.status, "value", entry.status) != (
                "in_progress"
            ):
                return None
            since = entry.updated_at.isoformat(sep=" ", timespec="seconds")
            if entry.claimed_pid is None:
                if datetime.now() - entry.updated_at < UNOWNED_CLAIM_STALE_AFTER:
                    return None
                why = f"was left in progress (claimed {since}; no process was recorded)"
            elif entry.claimed_host == socket.gethostname() and not (
                # shortcut: a reused pid makes a dead claim look alive; the entry
                # then stays in progress as before this check.
                psutil.pid_exists(entry.claimed_pid)
            ):
                why = (
                    f"was left in progress by process {entry.claimed_pid}, which is "
                    f"no longer running (claimed {since})"
                )
            else:
                return None
            entry.update_status(DownloadStatus.FAILED, error=f"Claim taken back: {why}")
            self._backup_queue()
            self._write_entries_atomic(entries)

        note = f"Entry '{entry_id}' {why}; starting the download again."
        logger.warning(note)
        return note

    def list_entries(
        self, status: Optional[DownloadStatus] = None
    ) -> List[DownloadQueueEntry]:
        """
        List all entries, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List[DownloadQueueEntry]: List of entries
        """
        with self._lock:
            entries = self._load_entries()

            if status is not None:
                entries = [e for e in entries if e.status == status]

            return entries

    def remove_entry(self, entry_id: str) -> None:
        """
        Remove entry from queue.

        Args:
            entry_id: Entry to remove

        Raises:
            EntryNotFoundError: If entry not found
            DownloadQueueError: If removal fails
        """
        with self._locked():
            entries = self._load_entries()
            original_count = len(entries)

            # Filter out entry
            entries = [e for e in entries if e.entry_id != entry_id]

            if len(entries) == original_count:
                raise EntryNotFoundError(f"Entry '{entry_id}' not found in queue")

            # Backup before modification
            self._backup_queue()

            # Write remaining entries atomically
            self._write_entries_atomic(entries)

            logger.debug(f"Removed entry {entry_id} from queue")

    def clear_queue(self) -> int:
        """
        Clear all entries from queue.

        Returns:
            int: Number of entries cleared

        Raises:
            DownloadQueueError: If clear operation fails
        """
        with self._locked():
            entries = self._load_entries()
            entry_count = len(entries)

            if entry_count > 0:
                # Backup before modification
                self._backup_queue()

                # Clear queue file
                try:
                    self.queue_file.write_text("", encoding="utf-8")
                    logger.debug(f"Cleared {entry_count} entries from queue")
                except Exception as e:
                    logger.error(f"Failed to clear queue: {e}")
                    raise DownloadQueueError(f"Failed to clear queue: {e}") from e

            return entry_count

    def _load_entries(self) -> List[DownloadQueueEntry]:
        """
        Load all entries from queue file.

        Returns:
            List[DownloadQueueEntry]: List of entries

        Raises:
            DownloadQueueError: If loading fails
        """
        entries = []

        if not self.queue_file.exists():
            return entries

        try:
            with open(self.queue_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        entry = DownloadQueueEntry.from_dict(data)
                        entries.append(entry)
                    except Exception as e:
                        logger.warning(
                            f"Skipping invalid entry at line {line_num}: {e}"
                        )
                        continue

            return entries

        except Exception as e:
            logger.error(f"Failed to load queue: {e}")
            raise DownloadQueueError(f"Failed to load queue: {e}") from e

    def _write_entries_atomic(self, entries: List[DownloadQueueEntry]) -> None:
        """
        Write entries atomically using temp file + rename.

        Args:
            entries: List of entries to write

        Raises:
            DownloadQueueError: If write fails
        """
        try:
            atomic_write_jsonl(self.queue_file, entries, lambda e: e.to_dict())
        except Exception as e:
            logger.error(f"Failed to write queue atomically: {e}")
            raise DownloadQueueError(f"Failed to write queue: {e}") from e

    def _backup_queue(self) -> Path:
        """
        Create timestamped backup of queue file.

        Returns:
            Path: Path to backup file

        Raises:
            DownloadQueueError: If backup fails
        """
        if not backups_enabled():
            return None

        if not self.queue_file.exists() or self.queue_file.stat().st_size == 0:
            # No need to backup empty file
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_file = self.backup_dir / f"queue_backup_{timestamp}.jsonl"

        try:
            # A new file only: never written through an existing entry or link.
            _copy_to_new_file(self.queue_file, backup_file)
            logger.debug("Created queue backup")
            return backup_file

        except ManagedPathError:
            raise  # the update is refused before it writes (fixed reason)
        except Exception as e:
            # Fixed text (the error can name the backup path): the kind only.
            logger.warning("Failed to create backup (%s)", type(e).__name__)
            # Don't raise - backup failure shouldn't stop operations
            return None

    def get_statistics(self) -> dict:
        """
        Get queue statistics.

        Returns:
            dict: Statistics including counts by status
        """
        with self._lock:
            entries = self._load_entries()

            # Initialize with ALL DownloadStatus values (ensures consistency)
            stats = {
                "total_entries": len(entries),
                "by_status": {status.value: 0 for status in DownloadStatus},
                "by_database": {},
                "by_priority": {},
            }

            for entry in entries:
                # Count by status
                stats["by_status"][entry.status] += 1

                # Count by database
                if entry.database not in stats["by_database"]:
                    stats["by_database"][entry.database] = 0
                stats["by_database"][entry.database] += 1

                # Count by priority
                priority_str = str(entry.priority)
                if priority_str not in stats["by_priority"]:
                    stats["by_priority"][priority_str] = 0
                stats["by_priority"][priority_str] += 1

            return stats
