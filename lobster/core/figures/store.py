"""FigureStore protocol and the local, filesystem-backed authority (contract §1, §2, §6, §9).

Local guarantees: atomic visibility + no-replace through this store (``os.mkdir`` / ``os.link``,
``revision.json`` written last), detection of byte changes on read. Qualified for POSIX local
filesystems. Not a crash-proof journal and no protection against other local code.

Writer lock: every store WRITE (``commit_execution``, ``commit``,
``add_rendition`` and deletion) runs under ONE in-process lock per resolved local store root,
shared by every ``LocalFigureStore`` instance of that root in this process. It protects only
that local copy within ONE process. Local qualification is ONE process per local workspace
directory: there is no cross-process lock (no flock) and no S3, Cloud or other-process claim.
Documented exception: a subprocess worker sharing the same local directory is outside this
qualification (reported OFF); enabling it needs a new decision.
"""

from __future__ import annotations

import datetime as _dt
import errno
import hashlib
import json
import os
import secrets
import shutil
import stat as _stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from lobster.core.figures import paths as P
from lobster.core.figures.errors import (
    AttemptConflictError,
    DurabilityError,
    FigureDeletedError,
    FigureInvalidError,
    FigureNotFoundError,
    GenerationFencedError,
    IntegrityError,
    ParentDeletedError,
    StorageError,
    StoreUnsupportedError,
    safe_reason,
)
from lobster.core.figures.ids import (
    PREFIX_EXECUTION,
    PREFIX_FIGURE,
    PREFIX_RENDITION,
    PREFIX_REVISION,
    FigureRef,
    is_valid_id,
    new_id,
)
from lobster.core.figures.payload import sha256_hex, strict_json_loads
from lobster.core.figures.records import (
    EXECUTION_REQUEST_SCHEMA,
    EXECUTION_SCHEMA,
    REVISION_SCHEMA,
    STATUS_AVAILABLE,
    STATUS_PENDING,
    Candidate,
    CommitResult,
    attempt_digest,
    execution_plan_digest,
    merge_inherited_inputs,
)

#: Caps for execution blobs (contract §4/§9).
MAX_EXECUTION_REQUEST_BYTES = 1 * 1024 * 1024
MAX_CODE_BYTES = 4 * 1024 * 1024
MAX_ACTIVITY_BYTES = 16 * 1024 * 1024


#: Deletion fence records in the RETAINED ``_attempts/``. Fixed names,
#: deterministic bytes, written no-replace: a re-run writes identical bytes and confirms them
#: again. They are never withdrawn.
TOMBSTONE_SCHEMA = "lobster.figure_tombstone/1"
PURGE_SCHEMA = "lobster.figure_purge/1"
_FENCE_REF = "deleted-ref-"
_FENCE_EXEC = "deleted-exec-"
_FENCE_INSTANCE = "deleted-inst-"
_FENCE_PURGE = "purge-"
_UNREADABLE = object()  # a fenced revision whose record could not be read


@dataclass(frozen=True)
class FenceReport:
    """What one fencing call KNOWS (per-entry, honest partial outcome; no rollback).

    ``confirmed``: entries that exist with durability confirmed (incl. ones an earlier run wrote).
    ``unconfirmed``: entries that exist but whose durability is unconfirmed (PENDING: the
    fenced object is already invisible; nothing may be physically removed for it yet).
    ``stopped``: a write failed and the remaining entries were not attempted.
    ``failed_entry_may_exist``: the failed entry may exist (its state is unknown).
    """

    planned: int
    confirmed: int
    unconfirmed: int
    stopped: bool = False
    failed_entry_may_exist: bool = False
    inventory_unreadable: bool = (
        False  # the entries to write could not all be determined
    )

    @property
    def complete(self) -> bool:
        return (
            not self.stopped and not self.inventory_unreadable and self.unconfirmed == 0
        )

    @property
    def not_attempted(self) -> Optional[int]:
        """Entries not attempted; None when the inventory itself was unreadable."""
        if self.inventory_unreadable:
            return None
        failed = 1 if self.stopped else 0
        return max(0, self.planned - self.confirmed - self.unconfirmed - failed)


@dataclass(frozen=True)
class CleanupReport:
    """Payload cleanup after the last live reference (claims and fences are retained)."""

    executions: int = 0  # execution records (+ activity) removed, each fenced first
    code: int = 0  # ``_code`` blobs removed
    sources: int = 0  # ``_sources`` blobs removed
    failed: int = 0  # a removal or a check failed: the payload stays (safe)


@dataclass(frozen=True)
class RemoveReport:
    """Physical removal of already-fenced revisions (invisible either way)."""

    removed: int
    failed: int
    cleanup: Optional[CleanupReport] = None
    kept: Tuple[str, ...] = ()  # fenced refs still on disk: pass them to a re-run

    @property
    def cleanup_pending(self) -> bool:
        return self.failed > 0 or bool(self.cleanup and self.cleanup.failed)


@dataclass(frozen=True)
class PurgeReport:
    """Session deletion in ONE writer-lock hold: fence, re-verify, remove."""

    record: FenceReport  # this call's purge-action record
    tombstones: Optional[FenceReport]  # None: the purge record did not complete
    fence_verified: bool = False  # this call's complete fence re-read and re-confirmed
    removal: Optional[RemoveReport] = None  # None: nothing was removed

    @property
    def complete(self) -> bool:
        return (
            self.record.complete
            and self.tombstones is not None
            and self.tombstones.complete
            and self.fence_verified
            and self.removal is not None
            and not self.removal.cleanup_pending
        )


@dataclass(frozen=True)
class AttemptFence:
    """The born set of one attempt (the attempt fence): the session-purge records present in
    ``_attempts/`` when the attempt began, captured under the writer lock BEFORE any input
    read. In-process only: never serialized and never sent to a child.

    ``purges is None``: the born set could not be read. Every commit under such a fence
    refuses (typed ``storage_failed``), so an unreadable start never becomes a free pass.
    """

    root_key: str  # the writer-lock key of the store root (realpath)
    purges: Optional[frozenset]


@dataclass
class Page:
    items: List[Dict[str, Any]]
    next_cursor: Optional[str]
    catalog_generation: Optional[str]


@runtime_checkable
class FigureStore(Protocol):
    """Engine ↔ store seam (contract §9). Local = :class:`LocalFigureStore`; Cloud = catalog-backed.

    Byte ABI: execution blobs travel as exact bytes; the store allocates IDs, verifies every sha,
    and derives scope from its own authenticated context. ``read_record``/``list`` return
    metadata only (payload integrity NOT verified); ``read`` verifies payload shas.
    """

    def commit_execution(
        self,
        record_bytes: bytes,
        code_bytes: Optional[bytes],
        activity_bytes: Optional[bytes],
    ) -> str: ...

    def commit(
        self, candidate: Candidate, *, execution_id: str, digest: str
    ) -> CommitResult: ...

    def read(self, ref: "str | FigureRef") -> Tuple[bytes, bytes, bytes]: ...

    def read_record(self, ref: "str | FigureRef") -> Dict[str, Any]: ...

    def read_execution(self, execution_id: str) -> Dict[str, Any]: ...

    def read_source(self, source_id: str) -> bytes: ...

    def list(self, cursor: Optional[str] = None, limit: int = 500) -> Page: ...

    def dependents(self, instance_id: str) -> List[Tuple[str, bool]]: ...

    def add_rendition(
        self,
        ref: "str | FigureRef",
        data: bytes,
        settings: Dict[str, Any],
        renderer_receipt: Dict[str, Any],
    ) -> Dict[str, Any]: ...

    def find_rendition(
        self,
        ref: "str | FigureRef",
        settings: Dict[str, Any],
        renderer_receipt: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]: ...


_ROOT_LOCKS: Dict[str, "threading.RLock"] = {}
_ROOT_LOCKS_GUARD = threading.Lock()


def _root_lock(root: Path) -> "threading.RLock":
    """The ONE in-process writer lock for a resolved local store root (see module docstring).

    Keyed by ``os.path.realpath`` at store construction, so every instance for the same
    directory in this process shares it. Re-entrant: a locked write may call other locked
    writes and reads. In-process only: it says nothing about other processes.
    """
    key = os.path.realpath(str(root))
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(key)
        if lock is None:
            lock = _ROOT_LOCKS[key] = threading.RLock()
        return lock


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _fsync_dir(path: Path) -> None:
    """fsync a directory so a new entry in it is durable. Failures RAISE (never swallowed)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError as exc:
        raise DurabilityError(
            "directory open for fsync failed",
            details={"reason": "dir_open_failed", "errno": exc.errno},
        ) from None
    try:
        os.fsync(fd)
    except OSError as exc:
        reason = (
            "dir_fsync_unsupported"
            if exc.errno in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP)
            else "dir_fsync_failed"
        )
        raise DurabilityError(
            "directory fsync failed", details={"reason": reason, "errno": exc.errno}
        ) from None
    finally:
        os.close(fd)


def _marker_present(path: Path) -> bool:
    """Is ``path`` a regular marker file (lstat: a final link is not followed)? Absent ->
    False. Any other I/O failure, or an entry that is not a regular file (a link included),
    raises a TYPED StorageError: never a raw OSError and never a false absence."""
    try:
        st = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        raise StorageError("marker stat failed") from None
    if not _stat.S_ISREG(st.st_mode):
        raise StorageError("marker is not a regular file")
    return True


_HEX = frozenset("0123456789abcdef")

#: Fields the store adds to an execution request when it writes the record (the rest is the
#: request; ``schema`` and the delivery fields are replaced inside execution_plan_digest).
_EXECUTION_STORE_FIELDS = (
    "execution_id",
    "scope",
    "plan_digest",
    "request_sha256",
    "created_at",
)


def _is_hex(text: Any, length: int) -> bool:
    return (
        isinstance(text, str) and len(text) == length and all(c in _HEX for c in text)
    )


def _valid_attempt_base(base: str) -> bool:
    """The attempt-base grammar the execution request accepts (``execution_attempt_base``)."""
    return isinstance(base, str) and base.count(":") == 1 and all(base.split(":"))


def _plan_shape_ok(planned: Any, count: Any) -> bool:
    """The planned-candidate shape the execution request verifier accepts (one rule, used
    by the verifier and by release decisions): a non-empty list of objects whose int
    ``index`` values are unique, sorted and in ``0..candidate_count-1``."""
    if not (isinstance(planned, list) and planned and type(count) is int):
        return False
    indices = [p.get("index") if isinstance(p, dict) else None for p in planned]
    if any(type(i) is not int or not 0 <= i < count for i in indices):
        return False
    return len(set(indices)) == len(indices) and indices == sorted(indices)


def _read_regular(path: Path) -> Optional[bytes]:
    """Read a store file without following a final symlink. Absent -> None; a symlink, a
    non-regular entry or any other I/O failure raises a TYPED StorageError (never a false
    absence, never a read through a link)."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(str(path), flags)
    except FileNotFoundError:
        return None
    except OSError:  # ELOOP (a symlink), ENOTDIR, EACCES, ...
        raise StorageError("unexpected entry at a store path") from None
    try:
        if not _stat.S_ISREG(os.fstat(fd).st_mode):
            raise StorageError("unexpected entry type at a store path")
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    except OSError:
        raise StorageError("read failed") from None
    finally:
        os.close(fd)


def _parse_marker(data: bytes, expected: Dict[str, str], hashes: Sequence[Any]) -> Dict:
    """Parse a registered marker; corrupt or misplaced -> IntegrityError (never skipped).

    ``expected``: fields that must equal the marker's location (ids, file name).
    ``hashes``: where a sha256 string must be: a key (nested ``{"sha256": ...}``) or ``None``
    for a top-level ``sha256``.
    """
    try:
        rec = json.loads(data)
    except (ValueError, RecursionError):
        raise IntegrityError(
            "registered record is not valid JSON", details={"reason": "invalid_json"}
        ) from None

    def _has_sha(key) -> bool:
        node = rec if key is None else rec.get(key)
        return isinstance(node, dict) and isinstance(node.get("sha256"), str)

    if not (
        isinstance(rec, dict)
        and all(rec.get(k) == v for k, v in expected.items())
        and all(_has_sha(key) for key in hashes)
    ):
        raise IntegrityError(
            "registered record does not match its location", details={"reason": "shape"}
        )
    return rec


def _marker_identity(path: Path) -> Tuple[str, int, int, int, int]:
    """Identity of one immutable marker file for the durability memo (lstat only)."""
    st = os.lstat(path)
    return (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _ensure_dir(path: Path) -> None:
    """Create ``path`` (and missing parents) durably: each new entry's parent is fsynced.

    An EXISTING directory's entry is confirmed again (its parent is fsynced): an earlier call
    may have created it and then failed its parent fsync, and a retry must not report success
    on top of an unconfirmed entry (durability follow-up item 1).
    """
    if not path.is_dir():
        _ensure_dir(path.parent)
        try:
            os.mkdir(path)
        except FileExistsError:
            if not path.is_dir():
                raise
    _fsync_dir(path.parent)


def _write_no_replace(target: Path, data: bytes) -> bool:
    """Write ``data`` to ``target`` without ever replacing an existing file.

    Returns True if written, False if ``target`` already existed with identical bytes.
    Raises IntegrityError if it existed with different bytes, DurabilityError if a file or
    directory fsync failed (the entry may then exist, but durability is unconfirmed).
    An identical existing entry (earlier call, or a concurrent identical writer) is confirmed
    again before False is returned, so a retry never skips an unconfirmed entry.
    """
    written, unconfirmed = _link_no_replace(target, data)
    if unconfirmed is not None:
        raise unconfirmed
    return written


def _link_no_replace(
    target: Path, data: bytes
) -> Tuple[bool, Optional[DurabilityError]]:
    """:func:`_write_no_replace` that RETURNS (not raises) a confirmation failure that
    happened once ``target`` exists with exactly ``data``: ``(written, error_or_None)``.

    Callers that must report an uncertain acknowledgement as PENDING use this instead of
    re-reading the file after a failure (that read can fail too). Every failure
    before the entry exists still raises.

    An existing ``target`` is read without following a final link: a link or any other
    non-regular entry is never reused as the identical entry (typed StorageError).
    """
    existing = _read_regular(target)
    if existing is not None:
        if existing != data:
            raise IntegrityError(
                "existing file differs", details={"reason": "existing_file_differs"}
            )
        try:
            _ensure_dir(target.parent)
            _fsync_dir(target.parent)
        except DurabilityError as exc:
            return False, exc
        return False, None
    _ensure_dir(target.parent)
    written = True
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError as exc:
                raise DurabilityError(
                    "file fsync failed",
                    details={"reason": "file_fsync_failed", "errno": exc.errno},
                ) from None
        try:
            os.link(tmp, str(target))
        except FileExistsError:
            if _read_regular(target) != data:
                raise IntegrityError(
                    "concurrent writer produced different bytes",
                    details={"reason": "concurrent_writer_differs"},
                )
            written = (
                False  # identical concurrent writer: still confirm the entry below
            )
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    try:
        _fsync_dir(target.parent)  # the entry exists with ``data`` from here on
    except DurabilityError as exc:
        return written, exc
    return written, None


def _json_bytes(obj: Any) -> bytes:
    return (
        json.dumps(obj, sort_keys=True, indent=1, allow_nan=False, default=str).encode(
            "utf-8"
        )
        + b"\n"
    )


class LocalFigureStore:
    """Filesystem authority for one session workspace (``<workspace>/plots``)."""

    def __init__(
        self,
        workspace_path: "str | Path",
        *,
        session_id: str = "local",
        tenant: Optional[str] = None,
    ):
        if os.name != "posix" or not hasattr(os, "link"):
            raise StoreUnsupportedError(
                "LocalFigureStore is qualified for POSIX local filesystems only",
                details={"reason": "platform_not_posix"},
            )
        self.workspace = Path(workspace_path)
        self.root = self.workspace / P.PLOTS_DIR
        self.session_id = session_id
        self.tenant = tenant
        # Local durability receipts ONLY: one entry per immutable
        # marker FILE identity (path, dev, inode, size, mtime_ns) this instance confirmed by
        # its own successful write or a successful re-confirmation. A new or re-created
        # marker never inherits a receipt. Never caches auth/lifecycle; never skips the
        # read() hash checks.
        self._confirmed: set = set()
        self._lock = _root_lock(self.root)
        self._root_key = os.path.realpath(str(self.root))

    # ------------------------------------------------------------------ paths
    def _rev_dir(self, ref: FigureRef) -> Path:
        return self.root / ref.figure_id / ref.revision_id

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix()

    # ------------------------------------------------ the filesystem choke point
    # Every access of this store to an existing or new artifact below the workspace (read,
    # presence, identity, enumeration, fsync, write, link, mkdir, unlink, removal) goes
    # through the methods of this section, and each of them checks :meth:`_guard` first.
    # No other method touches the filesystem (pinned by a source-contract test). Fail-closed:
    # a symbolic link, or any non-directory entry where a directory is expected, raises a
    # typed StorageError and is never followed or reused. With every component below the
    # workspace a real directory by lstat, the real path equals the normalized lexical path
    # below the workspace. The workspace path and its ancestors are trusted as given. Scope:
    # links that exist when the store looks; one created concurrently after the check is
    # not covered (no descriptor-relative traversal).

    def _guard(self, directory: Path) -> bool:
        """lstat every component from the workspace down to ``directory`` (inclusive).

        True: every component exists and is a real directory. False: one is absent (the ones
        below it are not looked at). A symbolic link or any other entry type, an lstat
        failure, a ``.``/``..`` component or a path outside the workspace raises typed.
        """
        try:
            parts = directory.relative_to(self.workspace).parts
        except ValueError:
            raise StorageError("path outside the store") from None
        cur = self.workspace
        for part in parts:
            if part in (".", ".."):
                raise StorageError("unexpected store path")
            cur = cur / part
            try:
                st = os.lstat(cur)
            except FileNotFoundError:
                return False
            except OSError:
                raise StorageError("lstat failed") from None
            if not _stat.S_ISDIR(st.st_mode):
                raise StorageError("unexpected entry type at a store path")
        return True

    def _linked(self, path: Path) -> bool:
        """Is any EXISTING component from the workspace down to ``path`` (inclusive) a
        symbolic link (lstat; nothing followed)? Other lstat failures raise typed."""
        try:
            parts = path.relative_to(self.workspace).parts
        except ValueError:
            raise StorageError("path outside the store") from None
        cur = self.workspace
        for part in parts:
            cur = cur / part
            try:
                st = os.lstat(cur)
            except FileNotFoundError:
                return False
            except OSError:
                raise StorageError("lstat failed") from None
            if _stat.S_ISLNK(st.st_mode):
                return True
            if not _stat.S_ISDIR(st.st_mode):
                return False
        return False

    def _require_real_dir(self, path: Path) -> bool:
        """lstat (whole chain, :meth:`_guard`): False if absent; True for a real directory;
        anything else raises typed."""
        return self._guard(path)

    def _read(self, path: Path) -> Optional[bytes]:
        """A regular file below checked real directories, never through a final link.
        Absent -> None; a link, another entry type or an I/O failure raises typed."""
        if not self._guard(path.parent):
            return None
        return _read_regular(path)

    def _read_or(self, path: Path, missing: Exception) -> bytes:
        """:meth:`_read`, raising ``missing`` when the file is absent."""
        data = self._read(path)
        if data is None:
            raise missing
        return data

    def _present(self, path: Path) -> bool:
        """Is ``path`` a regular file (lstat) below checked real directories? Absent ->
        False; a link or another entry type raises typed (never a false absence)."""
        if not self._guard(path.parent):
            return False
        return _marker_present(path)

    def _identity(self, path: Path) -> Tuple[str, int, int, int, int]:
        """The durability-memo identity of one marker file (lstat, checked chain)."""
        if not self._guard(path.parent):
            raise StorageError("marker vanished")
        return _marker_identity(path)

    def _fsync(self, directory: Path) -> None:
        """fsync a checked real directory (:func:`_fsync_dir`; absent -> DurabilityError)."""
        self._guard(directory)
        _fsync_dir(directory)

    def _listdir_real(self, path: Path) -> List[str]:
        """Sorted names in a REAL directory (checked chain). Absent -> []; a symlink or any
        other entry type, or an enumeration failure, raises typed."""
        if not self._guard(path):
            return []
        try:
            return sorted(os.listdir(path))
        except OSError:
            raise StorageError("enumeration failed") from None

    def _lstat_or_none(self, path: Path) -> Optional[os.stat_result]:
        """lstat of an entry below checked real directories. Absent (the entry or a
        directory above it) -> None; any other failure raises typed."""
        if not self._guard(path.parent):
            return None
        try:
            return os.lstat(path)
        except FileNotFoundError:
            return None
        except OSError:
            raise StorageError("lstat failed") from None

    def _lstat(self, path: Path) -> os.stat_result:
        """:meth:`_lstat_or_none` where absence also raises typed (incl. vanished during an
        inventory: never skipped)."""
        st = self._lstat_or_none(path)
        if st is None:
            raise StorageError("lstat failed")
        return st

    def _lstat_mode(self, path: Path) -> int:
        return self._lstat(path).st_mode

    def _unlink(self, path: Path) -> None:
        """Unlink one entry below checked real directories (OSError propagates)."""
        self._guard(path.parent)
        os.unlink(path)

    def _rmtree(self, directory: Path) -> None:
        """Remove a REAL directory tree (checked chain; nested links are unlinked, never
        followed). Absent: nothing to do."""
        if self._guard(directory):
            shutil.rmtree(directory)

    def _rmdir_quiet(self, path: Path) -> None:
        """Best effort: remove an empty real directory; failures are ignored."""
        try:
            if self._guard(path):
                os.rmdir(path)
        except (OSError, StorageError):
            pass

    def _mkdir_new(self, path: Path) -> None:
        """Create one new directory under a checked real parent (FileExistsError if taken)."""
        if not self._guard(path.parent):
            raise StorageError("store directory missing")
        os.mkdir(path)

    def _mkdirs(self, path: Path) -> None:
        """:func:`_ensure_dir` below checked real directories only."""
        self._guard(path)
        _ensure_dir(path)

    def _write(self, target: Path, data: bytes) -> bool:
        """:func:`_write_no_replace` into a checked real directory chain."""
        self._guard(target.parent)
        return _write_no_replace(target, data)

    def _link(
        self, target: Path, data: bytes
    ) -> Tuple[bool, Optional[DurabilityError]]:
        """:func:`_link_no_replace` into a checked real directory chain."""
        self._guard(target.parent)
        return _link_no_replace(target, data)

    def figure_path(self, ref: "str | FigureRef") -> str:
        """The workspace-relative path of a revision's figure file, as a commit reports it.

        Pure: validates ``ref`` with :meth:`FigureRef.parse` (``ValueError`` otherwise) and
        applies the same formula every :class:`CommitResult` path uses. No I/O, no lock: it
        says nothing about whether that revision exists, is available or was deleted.
        """
        return self._rel(self._rev_dir(FigureRef.parse(ref)) / P.FIGURE_FILE)

    @property
    def writer_lock(self) -> "threading.RLock":
        """This root's in-process writer lock (re-entrant; see the module docstring)."""
        return self._lock

    # ----------------------------------------------------------- attempt fence
    def attempt_fence(self) -> AttemptFence:
        """Capture this attempt's born set under the writer lock. Call it before the
        attempt reads any input. Never raises for an unreadable ``_attempts/``: the fence
        then records ``purges=None`` and every commit under it refuses."""
        with self._lock:
            try:
                purges: Optional[frozenset] = self._purge_names()
            except StorageError:
                purges = None
            return AttemptFence(self._root_key, purges)

    def _purge_names(self) -> frozenset:
        """Names of the session-purge records in ``_attempts/`` (any name with the purge
        prefix counts, so a malformed one also fences). Absent root/dir -> empty; a symlink,
        another entry type, or an enumeration failure raises typed."""
        if not self._require_real_dir(self.root):
            return frozenset()
        return frozenset(
            name
            for name in self._listdir_real(self.root / P.ATTEMPTS_DIR)
            if name.startswith(_FENCE_PURGE)
        )

    def _check_attempt_fence(self, fence: Optional[AttemptFence]) -> None:
        """Under the writer lock, before the idempotency lookup. ``None`` = no born-set check
        (tests and direct non-graph calls; the production writer sites always pass one, see
        the source contract test)."""
        if fence is None:
            return
        if not isinstance(fence, AttemptFence) or fence.root_key != self._root_key:
            raise ValueError(
                "attempt fence from another store root"
            )  # programming error
        if fence.purges is None:
            raise StorageError("the purge state at attempt start was unreadable")
        if not self._purge_names() <= fence.purges:
            raise GenerationFencedError("a session purge happened during this attempt")

    # ----------------------------------------------------------------- fences
    def _fence_path(self, prefix: str, value: str) -> Path:
        return self.root / P.ATTEMPTS_DIR / f"{prefix}{_key_hash(value)}.json"

    def _ref_deleted(self, ref: FigureRef) -> bool:
        """A revision tombstone exists (typed StorageError on I/O, never a false absence)."""
        return self._present(self._fence_path(_FENCE_REF, str(ref)))

    def _execution_deleted(self, execution_id: str) -> bool:
        return self._present(self._fence_path(_FENCE_EXEC, str(execution_id)))

    def _instance_deleted(self, instance_id: str) -> bool:
        return self._present(self._fence_path(_FENCE_INSTANCE, str(instance_id)))

    def _assert_locked(self) -> None:
        """Destructive helpers run only inside a writer-lock hold."""
        if not self._lock._is_owned():
            raise StorageError("writer lock not held")

    def _verify_fence(self, path: Path, data: bytes) -> bool:
        """Is this fence AUTHORITY now? Presence alone never is.

        True only for a regular file (final entry not a symlink; root and ``_attempts``
        real directories by lstat) holding exactly ``data``, whose entry is confirmed durable
        again now. Absent -> False. Anything else raises typed: different bytes
        (IntegrityError), a wrong type or I/O failure (StorageError), or an unconfirmed entry
        (DurabilityError).
        """
        if not (
            self._require_real_dir(self.root) and self._require_real_dir(path.parent)
        ):
            return False
        raw = self._read(path)
        if raw is None:
            return False
        if raw != data:
            raise IntegrityError(
                "fence record differs", details={"reason": "existing_file_differs"}
            )
        self._fsync(path.parent)
        return True

    def _write_fences(self, entries: Sequence[Tuple[Path, bytes]]) -> FenceReport:
        """Write fence entries in order; stop at the first failure (never rolled back)."""
        self._assert_locked()
        confirmed = unconfirmed = 0
        for path, data in entries:
            try:
                _, unconfirmed_error = self._link(path, data)
            except Exception:  # the failed entry's state is not proven: unknown
                return FenceReport(
                    len(entries),
                    confirmed,
                    unconfirmed,
                    True,
                    failed_entry_may_exist=True,
                )
            if unconfirmed_error is None:
                confirmed += 1
            else:
                unconfirmed += 1
        return FenceReport(len(entries), confirmed, unconfirmed)

    def _tombstone(self, prefix: str, field: str, value: str) -> Tuple[Path, bytes]:
        return (
            self._fence_path(prefix, value),
            _json_bytes({"schema": TOMBSTONE_SCHEMA, field: value}),
        )

    def fence_deletion(
        self, refs: Sequence[str], *, instance_ids: Sequence[str] = ()
    ) -> FenceReport:
        """Make ``refs`` invisible and fence late commits of ``instance_ids`` (dataset deletion).

        Under the writer lock. Deleted-instance records first, then one tombstone per ref; each
        is durable before the next. Every ref must currently be registered or already fenced
        (checked before anything is written). Nothing is physically removed here: call
        :meth:`remove_fenced` only after a COMPLETE report.
        """
        with self._lock:
            parsed = [FigureRef.parse(r) for r in refs]
            ids = [str(i) for i in instance_ids]
            if any(not i for i in ids):
                raise ValueError("fence_deletion() needs non-empty instance ids")
            for ref in parsed:
                if not (self._ref_deleted(ref) or self._fenceable(ref)):
                    raise FigureNotFoundError(
                        "figure not found", details={"reason": "revision_not_found"}
                    )
            entries = [self._tombstone(_FENCE_INSTANCE, "instance_id", i) for i in ids]
            entries += [self._tombstone(_FENCE_REF, "ref", str(r)) for r in parsed]
            return self._write_fences(entries)

    def fence_purge(self) -> Tuple[FenceReport, Optional[FenceReport]]:
        """Session deletion, FENCE ONLY: this never removes anything.

        Under the writer lock: (1) one new purge-action record, durable, even with zero claims;
        (2) only if (1) is complete: a tombstone for every managed object of a strict inventory
        (:meth:`_purge_inventory`). Returns ``(purge_record, tombstones)``; ``tombstones`` is
        None when (1) did not complete. Physical removal happens only inside :meth:`purge`.
        """
        with self._lock:
            _, record, tombs = self._fence_purge_locked()
            return record, tombs

    def purge(self) -> PurgeReport:
        """Session deletion: fence, re-verify, remove, in ONE writer-lock hold.

        Removal is authorized only by the fence THIS call wrote and then re-verified: this
        call's purge record and every tombstone of a fresh strict inventory must exist as
        regular files holding exactly the expected bytes, confirmed durable again
        (:meth:`_verify_fence`). No earlier purge record or report authorizes removal; a
        re-run starts a new action. Anything less removes nothing.
        """
        with self._lock:
            purge_entry, record, tombs = self._fence_purge_locked()
            if tombs is None or not (record.complete and tombs.complete):
                return PurgeReport(record, tombs)
            try:
                entries, _ = self._purge_inventory()
                verified = all(
                    self._verify_fence(path, data)
                    for path, data in [purge_entry, *entries]
                )
            except Exception:  # unknown, unreadable or unconfirmed: never authority
                verified = False
            if not verified:
                return PurgeReport(record, tombs)
            return PurgeReport(record, tombs, True, self._remove_purged_locked())

    def _fence_purge_locked(
        self,
    ) -> Tuple[Tuple[Path, bytes], FenceReport, Optional[FenceReport]]:
        self._require_real_dir(self.root)
        attempts = self.root / P.ATTEMPTS_DIR
        self._require_real_dir(attempts)
        purge_id = secrets.token_hex(16)  # a record name, not a figure ID
        purge_entry = (
            attempts / f"{_FENCE_PURGE}{purge_id}.json",
            _json_bytes({"schema": PURGE_SCHEMA, "purge_id": purge_id}),
        )
        record = self._write_fences([purge_entry])
        if not record.complete:
            return purge_entry, record, None
        try:
            entries, _ = self._purge_inventory()
        except Exception:  # an unknown/unreadable inventory never authorizes removal
            return purge_entry, record, FenceReport(0, 0, 0, inventory_unreadable=True)
        return purge_entry, record, self._write_fences(entries)

    def purge_targets(self) -> Tuple[Tuple[str, ...], Tuple[Tuple[Any, ...], ...]]:
        """What a session purge would fence and remove, by identity (deletion-plan fingerprint):
        the strict inventory's tombstone file names, and each legacy flat file as
        ``(name, st_dev, st_ino, st_size, st_mtime_ns)`` by lstat, both sorted. Under the
        writer lock; anything unknown or unreadable raises typed (never a partial list).
        """
        with self._lock:
            legacy: List[Tuple[Any, ...]] = []
            entries, _ = self._purge_inventory(legacy_out=legacy)
            return tuple(p.name for p, _ in entries), tuple(sorted(legacy))

    def _purge_inventory(
        self, legacy_out: Optional[List[Tuple[Any, ...]]] = None
    ) -> Tuple[List[Tuple[Path, bytes]], int]:
        """Every managed object under ``plots/``, read STRICTLY.

        Returns ``(tombstone entries, legacy flat files)``: one tombstone per revision claim,
        revision directory and execution (claim or record); accepted legacy flat files
        (regular files directly in ``plots/``, not store objects) are only counted. Anything
        unknown, of the wrong type (a symlink included), malformed or unreadable RAISES typed:
        it is never skipped and never becomes complete purge authority.
        """
        entries: Dict[Path, bytes] = {}

        def add(prefix: str, field: str, value: str) -> None:
            path, data = self._tombstone(prefix, field, value)
            entries[path] = data

        attempts = self.root / P.ATTEMPTS_DIR
        for name in self._listdir_real(attempts):
            path = attempts / name
            stem = name[: -len(".json")] if name.endswith(".json") else None
            if name.startswith(".tmp-"):  # write debris of an interrupted write
                self._lstat_regular(path)
            elif stem is not None and _is_hex(stem, 64):
                claim = self._parse_claim(path, stem)
                if "ref" in claim:
                    add(_FENCE_REF, "ref", claim["ref"])
                else:
                    add(_FENCE_EXEC, "execution_id", claim["execution_id"])
            elif stem is not None and (
                any(
                    stem.startswith(p) and _is_hex(stem[len(p) :], 64)
                    for p in (_FENCE_REF, _FENCE_EXEC, _FENCE_INSTANCE)
                )
                or (
                    stem.startswith(_FENCE_PURGE)
                    and _is_hex(stem[len(_FENCE_PURGE) :], 32)
                )
            ):
                self._lstat_regular(path)
            else:
                raise StorageError("unknown entry in the attempts directory")
        executions = self.root / P.EXECUTIONS_DIR
        for name in self._listdir_real(executions):
            path = executions / name
            if name.startswith(".tmp-"):
                self._lstat_regular(path)
                continue
            for suffix in (".activity.json", ".json"):
                if name.endswith(suffix):
                    execution_id = name[: -len(suffix)]
                    break
            else:
                execution_id = ""
            if not is_valid_id(execution_id):
                raise StorageError("unknown entry in the executions directory")
            self._lstat_regular(path)
            add(_FENCE_EXEC, "execution_id", execution_id)
        legacy = 0
        for name in self._listdir_real(self.root):
            path = self.root / name
            mode = self._lstat_mode(path)
            if name in P.RESERVED_DIRS:
                if not _stat.S_ISDIR(mode):
                    raise StorageError("unexpected entry type at a store path")
            elif _stat.S_ISDIR(mode) and is_valid_id(name):
                for revision_id in self._listdir_real(path):
                    if not (
                        is_valid_id(revision_id)
                        and _stat.S_ISDIR(self._lstat_mode(path / revision_id))
                    ):
                        raise StorageError("unknown entry in a figure directory")
                    add(_FENCE_REF, "ref", str(FigureRef(name, revision_id)))
            elif _stat.S_ISREG(mode):
                legacy += 1  # an accepted legacy flat file (removed by a purge only)
                if legacy_out is not None:
                    st = self._lstat(path)
                    if not _stat.S_ISREG(st.st_mode):
                        raise StorageError("unexpected entry type at a store path")
                    legacy_out.append(
                        (name, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                    )
            else:
                raise StorageError("unknown entry in the store root")
        return sorted(entries.items()), legacy

    def _parse_claim(self, path: Path, stem: str) -> Dict[str, str]:
        """A claim exactly as the store writes it: canonical bytes, the exact key set, string
        values, valid IDs, the attempt base/key grammar the store accepts (an opaque
        ``<a>:<b>`` base; a key ``<base>:<index>``), 64-hex sha256 digests, and bound to its
        file name. Anything else raises typed. IDs stay opaque."""
        raw = self._read(path)
        if raw is None:
            raise StorageError("claim vanished")
        try:
            claim = json.loads(raw)
        except (ValueError, RecursionError):
            raise StorageError("unrecognized claim") from None
        valid = (
            isinstance(claim, dict)
            and all(isinstance(v, str) for v in claim.values())
            and _json_bytes(claim) == raw
            and is_valid_id(claim.get("execution_id"))
        )
        if valid and set(claim) == {
            "attempt_key",
            "attempt_digest",
            "ref",
            "execution_id",
        }:
            try:
                FigureRef.parse(claim["ref"])
            except (TypeError, ValueError):
                valid = False
            base, _, index = claim["attempt_key"].rpartition(":")
            valid = (
                valid
                and _valid_attempt_base(base)
                and index.isascii()
                and index.isdigit()
                and (index == "0" or not index.startswith("0"))  # canonical; no int()
                and _is_hex(claim["attempt_digest"], 64)
                and _key_hash(claim["attempt_key"]) == stem
            )
        elif valid and set(claim) == {"attempt_base", "execution_id", "plan_digest"}:
            valid = (
                _valid_attempt_base(claim["attempt_base"])
                and _is_hex(claim["plan_digest"], 64)
                and _key_hash("exec:" + claim["attempt_base"]) == stem
            )
        else:
            valid = False
        if not valid:
            raise StorageError("unrecognized claim")
        return claim

    def _lstat_regular(self, path: Path) -> None:
        if not _stat.S_ISREG(self._lstat_mode(path)):
            raise StorageError("unexpected entry type at a store path")

    def remove_fenced(self, refs: Sequence[str]) -> RemoveReport:
        """Physically remove already-fenced revisions (dataset deletion, after the fence).

        Under the writer lock. Every ref's tombstone must be AUTHORITY first
        (:meth:`_verify_fence`: exact bytes, regular, confirmed durable again), checked for all
        refs before anything is removed: an absent tombstone refuses (typed), a pending, corrupt
        or symlinked one raises typed. lstat at the delete boundary: the root, figure and
        revision directories must be real directories (no symlinks).

        Order: a revision's record is the retry evidence for its payload,
        so every record is read first, the payload is cleaned, and a revision directory is
        removed only once the payload cleanup fully succeeded. An unreadable record, or any
        cleanup failure, KEEPS the (invisible) directory and reports it in ``kept``: a re-run
        with those refs re-reads the records and completes. Never reported clean on leftovers.
        """
        with self._lock:
            parsed = list(dict.fromkeys(FigureRef.parse(r) for r in refs))
            for ref in parsed:
                if not self._verify_fence(
                    *self._tombstone(_FENCE_REF, "ref", str(ref))
                ):
                    raise StorageError("refusing to remove an unfenced revision")
            records: Dict[FigureRef, Any] = {}
            for ref in parsed:
                try:
                    records[ref] = self._fenced_record(ref)
                except Exception:  # unreadable or wrong type: its payload is unknown
                    records[ref] = _UNREADABLE
            cleanup = self._cleanup_released(
                [r for r in records.values() if isinstance(r, dict)]
            )
            removed = 0
            kept: List[str] = []
            for ref, record in records.items():
                if record is _UNREADABLE or (
                    isinstance(record, dict) and cleanup.failed
                ):
                    kept.append(str(ref))  # invisible; a re-run re-reads its record
                    continue
                try:
                    self._remove_revision_dir(ref)
                    removed += 1
                except Exception:
                    kept.append(str(ref))
            return RemoveReport(removed, len(kept), cleanup, tuple(kept))

    def _fenced_record(self, ref: FigureRef) -> Optional[Dict[str, Any]]:
        """The record of a fenced revision still on disk; None only when its directory is
        already gone. A directory WITHOUT its record (a partial first commit, or lost retry
        evidence) is unknown and raises typed, so it is kept, never cleaned.
        A symlink or wrong-type entry, or an unreadable record, raises typed.
        """
        if not self._require_real_dir(self.root):
            return None
        fig_dir = self.root / ref.figure_id
        if not self._require_real_dir(fig_dir):
            return None
        rev_dir = fig_dir / ref.revision_id
        if not self._require_real_dir(rev_dir):
            return None
        raw = self._read(rev_dir / P.REVISION_FILE)
        if raw is None:
            raise StorageError("a fenced revision directory has no record")
        record = _parse_marker(
            raw,
            {"figure_id": ref.figure_id, "revision_id": ref.revision_id},
            ("figure", "config"),
        )
        self._payload_ids(record)  # a malformed payload reference is unknown: kept
        return record

    @staticmethod
    def _payload_ids(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """(source_id, execution_id) of a revision record, exactly as the store writes them
        (``_write_revision``): the revision schema; a ``source`` key holding None or
        {source_id: sha256-<64 hex>, ...}; an ``execution`` key holding {execution_id: <valid
        id>, ...} (never null). A missing key, a null execution, another schema, or any other
        shape raises typed: unknown references never free payload."""
        if not (
            isinstance(record, dict)
            and record.get("schema") == REVISION_SCHEMA
            and "source" in record
            and "execution" in record
        ):
            raise StorageError("unknown revision record shape")
        source, execution = record["source"], record["execution"]
        source_id = None
        if source is not None:
            source_id = source.get("source_id") if isinstance(source, dict) else None
            if not (
                isinstance(source_id, str)
                and source_id.startswith("sha256-")
                and _is_hex(source_id[len("sha256-") :], 64)
            ):
                raise StorageError("malformed source reference")
        execution_id = (  # the writer always records a non-null execution
            execution.get("execution_id") if isinstance(execution, dict) else None
        )
        if not is_valid_id(execution_id):
            raise StorageError("malformed execution reference")
        return source_id, execution_id

    def _gc_survivors(self) -> List[Dict[str, Any]]:
        """Revision records that stay LIVE for payload cleanup (GC authority).

        Distinct from visibility (:meth:`_iter_records`, unchanged): a revision is not live only
        when its tombstone is AUTHORITY (:meth:`_verify_fence`); mere presence never is, so a
        pending, corrupt or symlinked tombstone raises. lstat throughout, nothing followed: the
        root, figure and revision directories must be real directories with valid IDs, the
        record a regular file read without following a link (:func:`_read_regular`). Regular
        files in the root are legacy flat files (no store reference). A revision directory
        without its record, or any other entry, raises: its references are unknown, so the
        caller keeps every payload. Reuses the existing helpers; no second parser.
        """
        if not self._require_real_dir(self.root):
            return []
        out: List[Dict[str, Any]] = []
        for name in self._listdir_real(self.root):
            if name in P.RESERVED_DIRS:
                continue
            fig_dir = self.root / name
            mode = self._lstat_mode(fig_dir)
            if _stat.S_ISREG(mode):
                continue  # a legacy flat file
            if not (_stat.S_ISDIR(mode) and is_valid_id(name)):
                raise StorageError("unknown entry in the store root")
            for rev_name in self._listdir_real(fig_dir):
                rev_dir = fig_dir / rev_name
                if not (
                    is_valid_id(rev_name) and _stat.S_ISDIR(self._lstat_mode(rev_dir))
                ):
                    raise StorageError("unknown entry in a figure directory")
                ref = FigureRef(name, rev_name)
                if self._verify_fence(*self._tombstone(_FENCE_REF, "ref", str(ref))):
                    continue  # deleted with authority: not live
                raw = self._read(rev_dir / P.REVISION_FILE)
                if raw is None:
                    raise StorageError(
                        "unfinished revision: its references are unknown"
                    )
                record = _parse_marker(
                    raw,
                    {"figure_id": name, "revision_id": rev_name},
                    ("figure", "config"),
                )
                self._payload_ids(record)
                out.append(record)
        return out

    def _parse_execution(self, raw: bytes, execution_id: str) -> Dict[str, Any]:
        """An execution record exactly as the store writes it, bound to its name and plan:
        a JSON object, ``execution_id`` equal to its file's, a
        ``code_id`` field (None or sha256-<64 hex>), and a 64-hex ``plan_digest`` that the
        record's own plan recomputes to (:func:`execution_plan_digest` over the record minus
        the fields the store adds). Anything else raises typed: malformed means unknown, so
        payload is kept, never "not referenced"."""
        record = strict_json_loads(raw, what="execution record")
        code_id = record.get("code_id") if isinstance(record, dict) else None
        plan = (
            {k: v for k, v in record.items() if k not in _EXECUTION_STORE_FIELDS}
            if isinstance(record, dict)
            else {}
        )
        if not (
            isinstance(record, dict)
            and record.get("schema") == EXECUTION_SCHEMA  # the digest ignores schema
            and record.get("execution_id") == execution_id
            and "code_id" in record
            and (
                code_id is None
                or (
                    isinstance(code_id, str)
                    and code_id.startswith("sha256-")
                    and _is_hex(code_id[len("sha256-") :], 64)
                )
            )
            and _is_hex(record.get("plan_digest"), 64)
            and execution_plan_digest(plan) == record["plan_digest"]
        ):
            raise StorageError("malformed execution record")
        return record

    def _remove_revision_dir(self, ref: FigureRef) -> None:
        """Remove one fenced revision directory (its payload was cleaned before)."""
        self._assert_locked()
        if not self._require_real_dir(self.root):
            return
        fig_dir = self.root / ref.figure_id
        if not self._require_real_dir(fig_dir):
            return
        rev_dir = fig_dir / ref.revision_id
        if self._require_real_dir(rev_dir):
            try:
                self._unlink(rev_dir / P.REVISION_FILE)  # the marker goes first
            except FileNotFoundError:
                pass
            self._rmtree(rev_dir)
        self._rmdir_quiet(fig_dir)  # only when empty

    # ------------------------------------------------------- payload cleanup
    def _cleanup_released(self, released: Sequence[Dict[str, Any]]) -> CleanupReport:
        """Remove payload the released revisions referenced once nothing live does.

        Claims and fences are retained. An execution record (and activity) is removed only when
        no surviving revision references it AND every planned candidate's revision claim is
        fenced, so no live or in-flight commit can need it; an execution tombstone is made
        durable first (a retry never re-creates the payload). Within one execution the payload
        goes before the record (code blob, then activity, then the record LAST): the record is
        the retry evidence, so a partial failure is retried, never silently lost. A
        ``_code`` blob goes when no execution record that stays references it; a
        ``_sources`` blob when no surviving revision does. Any failed or unknown check keeps
        the payload (cleanup pending, never unsafe).
        """
        self._assert_locked()
        if not released:
            return CleanupReport()
        try:
            live = [self._payload_ids(r) for r in self._gc_survivors()]
            gone = [self._payload_ids(r) for r in released]
        except Exception:  # unknown survivors or references: keep every payload
            return CleanupReport(failed=1)
        sources = {s for s, _ in gone} - {s for s, _ in live}
        execs = {e for _, e in gone} - {e for _, e in live}
        exec_dir = self.root / P.EXECUTIONS_DIR
        n_exec = n_code = n_src = failed = 0
        # lstat the parents first: a symlinked or wrong-type store root or
        # executions directory is never read through; nothing in it is touched
        try:
            exec_dir_ok = self._require_real_dir(self.root)
            self._require_real_dir(exec_dir)  # absent: nothing to read; a link raises
        except StorageError:
            exec_dir_ok = False
            failed += 1
        # pass 1: which executions are releasable (read only, never through a link)
        releasable: Dict[str, Dict[str, Any]] = {}
        for eid in sorted(e for e in execs if is_valid_id(e) and exec_dir_ok):
            try:
                raw = self._read(exec_dir / f"{eid}.json")
                if raw is None:
                    continue  # already cleaned by an earlier run
                record = self._parse_execution(raw, eid)
                if self._all_candidates_fenced(eid, record):
                    releasable[eid] = record
                # else: still needed (live or in flight)
            except Exception:
                failed += 1
        # code still referenced by an execution record that STAYS (unknown -> keep all)
        try:
            staying = self._referenced_code_ids(exec_dir, exclude=set(releasable))
        except (
            Exception
        ):  # unknown references (incl. a wrong-type entry): keep all code
            staying = None
        # pass 2: fence, then payload, then the record last
        freed: set = set()
        for eid, record in releasable.items():
            try:
                if not self._write_fences(
                    [self._tombstone(_FENCE_EXEC, "execution_id", eid)]
                ).complete:
                    failed += 1
                    continue
                code_id = record.get("code_id")
                if code_id and code_id not in freed:
                    if staying is None:
                        failed += 1  # unknown references: keep the code and the record
                        continue
                    if code_id not in staying:
                        outcome = self._unlink_blob(
                            self.root / P.CODE_DIR, f"{code_id}.py", code_id
                        )
                        if outcome is False:
                            failed += 1
                            continue
                        freed.add(code_id)
                        n_code += 1 if outcome else 0
                for path in (
                    exec_dir / f"{eid}.activity.json",
                    exec_dir / f"{eid}.json",  # the record LAST
                ):
                    try:
                        self._unlink(path)
                    except FileNotFoundError:
                        pass
                n_exec += 1
            except Exception:
                failed += 1
        for source_id in sorted(s for s in sources if isinstance(s, str)):
            outcome = self._unlink_blob(
                self.root / P.SOURCES_DIR, f"{source_id}.json", source_id
            )
            if outcome is False:
                failed += 1
            elif outcome:
                n_src += 1
        return CleanupReport(n_exec, n_code, n_src, failed)

    def _all_candidates_fenced(
        self, execution_id: str, execution: Dict[str, Any]
    ) -> bool:
        """Is every planned candidate of this (parsed, plan-bound) execution deleted with
        authority? False when a candidate has no revision claim yet (it may be in flight) or
        its ref tombstone is absent. Raises typed when anything is malformed or bound elsewhere:
        the attempt base grammar, the planned (index, digest) shape,
        the execution claim (``exec:<base>`` -> this execution_id and plan_digest), and each
        revision claim's execution_id and attempt_digest against this execution's plan.
        """
        base = execution.get("attempt_base")
        planned = execution.get("planned_candidates")
        if not (
            _valid_attempt_base(base)
            and _plan_shape_ok(planned, execution.get("candidate_count"))
        ):
            raise StorageError("malformed execution plan")
        attempts = self.root / P.ATTEMPTS_DIR
        if not self._require_real_dir(attempts):
            raise StorageError("execution without its claim")
        stem = _key_hash("exec:" + base)
        if self._read(attempts / f"{stem}.json") is None:
            raise StorageError("execution without its claim")
        claim = self._parse_claim(attempts / f"{stem}.json", stem)
        if (
            "plan_digest" not in claim
            or claim["execution_id"] != execution_id
            or claim["plan_digest"] != execution["plan_digest"]
        ):
            raise StorageError("execution claim bound elsewhere")
        for p in planned:
            index = p.get("index") if isinstance(p, dict) else None
            digest = p.get("attempt_digest") if isinstance(p, dict) else None
            if type(index) is not int or index < 0 or not _is_hex(digest, 64):
                raise StorageError("malformed execution plan")
            stem = _key_hash(f"{base}:{index}")
            if self._read(attempts / f"{stem}.json") is None:
                return False  # no revision claim: the candidate may still be in flight
            claim = self._parse_claim(attempts / f"{stem}.json", stem)
            if (
                "ref" not in claim
                or claim["execution_id"] != execution_id
                or claim["attempt_digest"] != digest
            ):
                raise StorageError("revision claim bound elsewhere")
            self._fsync(
                attempts
            )  # a claim relied on is confirmed again (as _read_claim)
            if not self._verify_fence(
                *self._tombstone(_FENCE_REF, "ref", str(FigureRef.parse(claim["ref"])))
            ):
                return False
        return True

    def _referenced_code_ids(
        self, exec_dir: Path, exclude: "set | frozenset" = frozenset()
    ) -> set:
        """Code ids referenced by the execution records in ``exec_dir`` minus ``exclude``.

        lstat: a real directory; each record a regular file read without following a link;
        activity records and write debris are skipped; each record is parsed strictly
        (:meth:`_parse_execution`); a malformed record or any other entry raises typed (unknown
        references never free code)."""
        out = set()
        for name in self._listdir_real(exec_dir):
            if name.startswith(".tmp-") or name.endswith(".activity.json"):
                continue
            execution_id = name[: -len(".json")] if name.endswith(".json") else ""
            if not is_valid_id(execution_id):
                raise StorageError("unknown entry in the executions directory")
            if execution_id in exclude:
                continue
            raw = self._read(exec_dir / name)
            if raw is None:
                continue  # removed meanwhile by this same lock hold: not referenced
            code_id = self._parse_execution(raw, execution_id)["code_id"]
            if code_id:
                out.add(code_id)
        return out

    def _unlink_blob(self, directory: Path, name: str, blob_id: str) -> Optional[bool]:
        """Unlink one content-addressed blob (``sha256-<hex>`` ids only).

        lstat both levels: ``directory`` must be a real directory and the
        blob a regular file; a symlink at either level is refused and its target never
        touched. True: removed now. None: already absent. False: refused or failed (kept).
        """
        hexpart = blob_id[len("sha256-") :] if blob_id.startswith("sha256-") else ""
        if not _is_hex(hexpart, 64):
            return False
        try:
            if not self._guard(directory):
                return None
        except StorageError:
            return False
        path = directory / name
        try:
            st = self._lstat_or_none(path)
        except StorageError:
            return False
        if st is None:
            return None
        if not _stat.S_ISREG(st.st_mode):
            return False
        try:
            self._unlink(path)
        except FileNotFoundError:
            return None
        except (OSError, StorageError):
            return False
        return True

    def _remove_purged_locked(self) -> RemoveReport:
        """:meth:`purge` stage 3 only: remove every ``plots/`` entry EXCEPT ``_attempts/``.

        Called only inside :meth:`purge`'s lock hold, after its fence was re-verified. lstat per
        entry: real directories are removed as trees, regular files are unlinked; any other entry
        type is never touched (counted as failed; the strict inventory refused it already).
        """
        self._assert_locked()
        if not self._require_real_dir(self.root):
            return RemoveReport(0, 0)
        entries = [self.root / name for name in self._listdir_real(self.root)]
        removed = failed = 0
        for entry in entries:
            if entry.name == P.ATTEMPTS_DIR:
                continue
            try:
                st = self._lstat_or_none(entry)
            except StorageError:
                failed += 1
                continue
            if st is None:
                continue
            try:
                if _stat.S_ISDIR(st.st_mode):  # a real directory (lstat)
                    self._rmtree(entry)
                elif _stat.S_ISREG(st.st_mode):
                    self._unlink(entry)
                else:
                    failed += 1
                    continue
                removed += 1
            except (OSError, StorageError):
                failed += 1
        return RemoveReport(removed, failed)

    # -------------------------------------------------------------- execution
    def commit_execution(
        self,
        record_bytes: bytes,
        code_bytes: Optional[bytes],
        activity_bytes: Optional[bytes],
        *,
        fence: Optional[AttemptFence] = None,
    ) -> str:
        """Durably record one attempt's execution before any of its revisions (contract §6.3, §9).

        ``record_bytes`` = canonical ``lobster.figure_execution_request/2`` JSON (no ID, path,
        scope or timestamp). The blobs are verified against it. Claim-first: the attempt claim
        binds ``attempt_base`` to (execution_id, plan_digest); a retry with the same plan reuses
        the original execution (and its original activity link); a changed plan (code, tool,
        parameters, persist, versions, planned (index, digest) set) is an ``attempt_conflict``.
        Runs under the root's writer lock. ``fence`` (attempt fence): a session purge recorded after
        the attempt began refuses (``generation_fenced``) before anything else.
        """
        with self._lock:
            self._check_attempt_fence(fence)
            return self._commit_execution(record_bytes, code_bytes, activity_bytes)

    def _commit_execution(
        self,
        record_bytes: bytes,
        code_bytes: Optional[bytes],
        activity_bytes: Optional[bytes],
    ) -> str:
        request = self._verify_execution_request(
            record_bytes, code_bytes, activity_bytes
        )
        plan_digest = execution_plan_digest(request)
        attempt_base = request["attempt_base"]
        claim_path = (
            self.root / P.ATTEMPTS_DIR / f"{_key_hash('exec:' + attempt_base)}.json"
        )
        claim = self._read_claim(claim_path)
        if claim is None:
            execution_id = new_id(PREFIX_EXECUTION)
            claim_bytes = _json_bytes(
                {
                    "attempt_base": attempt_base,
                    "execution_id": execution_id,
                    "plan_digest": plan_digest,
                }
            )
            try:
                self._write(claim_path, claim_bytes)
            except IntegrityError:
                claim = self._read_claim(
                    claim_path
                )  # lost the race: defer to the winner
            else:
                claim = {"execution_id": execution_id, "plan_digest": plan_digest}
        if self._execution_deleted(str(claim.get("execution_id"))):
            # An execution tombstone fences the ORIGINAL attempt. It may exist with no
            # revision claim (a figure never committed), so it stays generation_fenced and
            # is never named figure_deleted (reasons/3)
            raise GenerationFencedError("this attempt was deleted")
        if claim.get("plan_digest") != plan_digest:
            raise AttemptConflictError(
                "same attempt, different execution plan",
                details={"reason": "execution_plan_changed"},
            )
        execution_id = claim["execution_id"]
        record_path = self.root / P.EXECUTIONS_DIR / f"{execution_id}.json"
        if self._read(record_path) is not None:  # a link is refused, never reused
            self._verify_execution_record(execution_id, plan_digest)
            self._fsync(
                record_path.parent
            )  # a reused record is durable before reuse is reported
            return execution_id
        # First delivery, or completing an interrupted one with the claimed ID.
        if code_bytes is not None:
            self._write(self.root / P.CODE_DIR / f"{request['code_id']}.py", code_bytes)
        activity_ref = None
        if activity_bytes is not None:
            activity_path = (
                self.root / P.EXECUTIONS_DIR / f"{execution_id}.activity.json"
            )
            original = self._read(activity_path)  # a link is refused, never reused
            if original is not None:
                activity_bytes = original  # the original observation wins
            else:
                self._write(activity_path, activity_bytes)
            activity_ref = {
                "path": self._rel(activity_path),
                "sha256": sha256_hex(activity_bytes),
                "bytes": len(activity_bytes),
            }
            activity_id = json.loads(activity_bytes).get("id")
        else:
            activity_id = None
        record = dict(request)
        record.update(
            {
                "schema": EXECUTION_SCHEMA,
                "execution_id": execution_id,
                "scope": {"session": self.session_id, "tenant": self.tenant},
                "plan_digest": plan_digest,
                "request_sha256": sha256_hex(record_bytes),
                "activity_id": None if activity_id is None else str(activity_id),
                "activity_record": activity_ref,
                "created_at": _utcnow(),
            }
        )
        self._write(record_path, _json_bytes(record))
        return execution_id

    @staticmethod
    def _verify_execution_request(
        record_bytes: bytes,
        code_bytes: Optional[bytes],
        activity_bytes: Optional[bytes],
    ) -> Dict[str, Any]:
        def bad(reason: str) -> FigureInvalidError:
            return FigureInvalidError(
                "invalid execution request", details={"reason": reason}
            )

        if (
            not isinstance(record_bytes, bytes)
            or len(record_bytes) > MAX_EXECUTION_REQUEST_BYTES
        ):
            raise bad("execution_request_too_large")
        request = strict_json_loads(record_bytes, what="execution request")
        if (
            not isinstance(request, dict)
            or request.get("schema") != EXECUTION_REQUEST_SCHEMA
        ):
            raise bad("execution_request_schema")
        base = request.get("attempt_base")
        if (
            not isinstance(base, str)
            or base.count(":") != 1
            or not all(base.split(":"))
        ):
            raise bad("execution_attempt_base")
        code_id = request.get("code_id")
        if code_id is None:
            if code_bytes is not None or request.get("code") is not None:
                raise bad("code_blob_unexpected")
        else:
            if code_bytes is None:
                raise bad("code_blob_required")
            if len(code_bytes) > MAX_CODE_BYTES:
                raise bad("code_too_large")
            fp = request.get("code") or {}
            if (
                code_id != f"sha256-{sha256_hex(code_bytes)}"
                or fp.get("sha256") != sha256_hex(code_bytes)
                or fp.get("bytes") != len(code_bytes)
            ):
                raise IntegrityError(
                    "code blob does not match request",
                    details={"reason": "code_blob_mismatch"},
                )
        if request.get("activity_record") is None:
            if (
                activity_bytes is not None
                or request.get("activity_id") is not None
                or request.get("activity_unavailable") != "provenance_disabled"
            ):
                raise bad("activity_fields_inconsistent")
        else:
            if activity_bytes is None:
                raise bad("activity_blob_required")
            if len(activity_bytes) > MAX_ACTIVITY_BYTES:
                raise bad("activity_too_large")
            rec = request["activity_record"]
            if rec.get("sha256") != sha256_hex(activity_bytes) or rec.get(
                "bytes"
            ) != len(activity_bytes):
                raise IntegrityError(
                    "activity blob does not match request",
                    details={"reason": "activity_blob_mismatch"},
                )
            activity = strict_json_loads(activity_bytes, what="activity")
            activity_id = request.get("activity_id")
            if (
                not isinstance(activity_id, str)
                or not activity_id
                or not isinstance(activity, dict)
                or str(activity.get("id")) != activity_id
                or request.get("activity_unavailable") is not None
            ):
                raise bad("activity_fields_inconsistent")
        if not _plan_shape_ok(
            request.get("planned_candidates"), request.get("candidate_count")
        ):
            raise bad("planned_candidates")
        return request

    def _verify_execution_record(
        self, execution_id: str, plan_digest: str
    ) -> Dict[str, Any]:
        record = self.read_execution(execution_id)
        if record.get("plan_digest") != plan_digest:
            raise IntegrityError(
                "execution record does not match its claim",
                details={"reason": "execution_record_mismatch"},
            )
        if record.get("code_id"):
            code_path = self.root / P.CODE_DIR / f"{record['code_id']}.py"
            code = self._read(code_path)  # a link is refused, never reused
            if code is None or f"sha256-{sha256_hex(code)}" != record["code_id"]:
                raise IntegrityError(
                    "execution code changed after save",
                    details={"reason": "code_blob_changed"},
                )
        activity = record.get("activity_record")
        if activity:
            path = self.workspace / activity["path"]
            observed = self._read(path)
            if observed is None or sha256_hex(observed) != activity["sha256"]:
                raise IntegrityError(
                    "execution activity record changed after save",
                    details={"reason": "activity_blob_changed"},
                )
        return record

    def read_execution(self, execution_id: str) -> Dict[str, Any]:
        path = self.root / P.EXECUTIONS_DIR / f"{execution_id}.json"
        if self._execution_deleted(execution_id):
            raise FigureNotFoundError(
                "unknown execution", details={"reason": "execution_not_found"}
            )
        raw = self._read(path)  # real directories only, no final link
        if raw is None:
            raise FigureNotFoundError(
                "unknown execution", details={"reason": "execution_not_found"}
            )
        return json.loads(raw)

    def history(self, figure_id):
        """Linked timeline over immutable revision ancestry, migrated on first access."""
        from .history import Timeline

        with self._lock:
            return Timeline(self, figure_id).view()

    def delete_revision(self, ref, generation):
        from .history import Timeline

        with self._lock:
            return Timeline(self, FigureRef.parse(ref).figure_id).delete(
                ref, generation
            )

    # ----------------------------------------------------------------- commit
    def commit(
        self,
        candidate: Candidate,
        *,
        execution_id: str,
        digest: str,
        fence: Optional[AttemptFence] = None,
    ) -> CommitResult:
        """Commit one candidate of a recorded execution (contract §6.2/§6.3).

        The store re-derives the digest from the candidate, the parent record and the execution
        record, and requires (index, digest) to be in the execution's immutable plan: a candidate
        can never attach to an execution whose recipe it does not belong to. Runs under the
        root's writer lock; ``fence`` as in :meth:`commit_execution`, checked first.
        """
        with self._lock:
            self._check_attempt_fence(fence)
            return self._commit(candidate, execution_id=execution_id, digest=digest)

    def _commit(
        self, candidate: Candidate, *, execution_id: str, digest: str
    ) -> CommitResult:
        # Deletion fences run BEFORE the idempotency lookup (contract §6.2). A VERIFIED edge to a
        # deleted dataset instance is fenced (it would have been a verified dependent); an
        # unverified one is not (such figures are kept as "may depend"), so editing a kept
        # figure keeps working. Nothing is matched by name.
        if self._execution_deleted(execution_id):
            raise GenerationFencedError("this attempt was deleted")
        if any(
            e.verified and e.instance_id and self._instance_deleted(e.instance_id)
            for e in candidate.inputs
        ):
            raise GenerationFencedError("an input of this attempt was deleted")
        execution = self.read_execution(execution_id)
        expected_key = f"{execution['attempt_base']}:{candidate.candidate_index}"
        if candidate.attempt_key != expected_key:
            raise AttemptConflictError(
                "attempt key does not belong to this execution",
                details={"reason": "attempt_key_mismatch"},
            )
        planned = {
            (p["index"], p["attempt_digest"])
            for p in execution.get("planned_candidates", [])
        }
        if (candidate.candidate_index, digest) not in planned:
            raise AttemptConflictError(
                "candidate is not part of this execution's plan",
                details={"reason": "candidate_not_in_plan"},
            )

        parent_record = None
        if candidate.parent_ref:
            parent = FigureRef.parse(candidate.parent_ref)
            if not self._has_marker(parent):
                raise ParentDeletedError(
                    "parent is not available", details={"reason": "parent_unavailable"}
                )
            # a parent whose save confirmation is pending raises DurabilityError
            # (durability_failed), never parent_deleted
            parent_record = self.read_record(parent)
        merged = merge_inherited_inputs(
            candidate.inputs, (parent_record or {}).get("inputs", [])
        )
        if merged != tuple(candidate.inputs):
            raise FigureInvalidError(
                "candidate inputs are not normalized against the parent",
                details={"reason": "inputs_not_normalized"},
            )
        recomputed = attempt_digest(
            candidate,
            parent_figure_sha256=(parent_record or {}).get("figure", {}).get("sha256"),
            parent_config_sha256=(parent_record or {}).get("config", {}).get("sha256"),
            code_fingerprint=execution.get("code"),
            tool=execution.get("tool"),
            persist=execution.get("persist"),
            engine_version=execution.get("engine_version"),
            plotly_version=execution.get("plotly_version"),
        )
        if recomputed != digest:
            raise FigureInvalidError(
                "attempt digest does not match the candidate",
                details={"reason": "digest_mismatch"},
            )

        claim_path = (
            self.root / P.ATTEMPTS_DIR / f"{_key_hash(candidate.attempt_key)}.json"
        )
        claim = self._read_claim(claim_path)
        if claim is not None:
            return self._commit_claimed(candidate, claim, execution_id, digest)

        figure_id = (
            FigureRef.parse(candidate.parent_ref).figure_id
            if candidate.parent_ref
            else self._new_figure_dir()
        )
        ref = FigureRef(figure_id, new_id(PREFIX_REVISION))
        claim_bytes = _json_bytes(
            {
                "attempt_key": candidate.attempt_key,
                "attempt_digest": digest,
                "ref": str(ref),
                "execution_id": execution_id,
            }
        )
        try:
            self._write(claim_path, claim_bytes)
        except IntegrityError:
            # Lost a race on the same attempt key: defer to the winner.
            if not candidate.parent_ref:
                self._rmdir_quiet(self.root / figure_id)
            return self._commit_claimed(
                candidate, self._read_claim(claim_path), execution_id, digest
            )
        return self._write_revision(
            candidate, ref, execution_id=execution_id, digest=digest
        )

    def _commit_claimed(
        self,
        candidate: Candidate,
        claim: Dict[str, Any],
        execution_id: str,
        digest: str,
    ) -> CommitResult:
        if self._ref_deleted(FigureRef.parse(claim["ref"])):
            # Never complete or republish a deleted revision (contract §6.2, reasons/3)
            raise FigureDeletedError("this revision was deleted")
        if (
            claim["attempt_digest"] != digest
            or claim.get("execution_id") != execution_id
        ):
            raise AttemptConflictError(
                "same attempt key, different content",
                details={"reason": "attempt_digest_changed"},
            )
        ref = FigureRef.parse(claim["ref"])
        if self._has_marker(ref):
            # Reuse only after verifying the committed artifact (not marker presence alone)
            # and re-confirming its durability; an uncertain confirmation is PENDING.
            try:
                record_bytes, figure_bytes, config_bytes = self.read(ref)
            except StorageError as exc:  # incl. DurabilityError; integrity errors raise
                return self._pending(candidate, exc)
            record = json.loads(record_bytes)
            if (
                record.get("execution", {}).get("attempt_digest") != digest
                or figure_bytes != candidate.figure_bytes
                or config_bytes != candidate.config_bytes
            ):
                raise IntegrityError(
                    "committed revision does not match its claim",
                    details={"reason": "claimed_revision_mismatch"},
                )
            return self._result(candidate, ref)
        # Interrupted earlier commit: complete it idempotently with the claimed IDs.
        return self._write_revision(
            candidate, ref, execution_id=execution_id, digest=digest
        )

    def _write_revision(
        self,
        candidate: Candidate,
        ref: FigureRef,
        *,
        execution_id: str,
        digest: str,
    ) -> CommitResult:
        if self._ref_deleted(
            ref
        ):  # defence in depth: never re-create a deleted revision
            raise FigureDeletedError("this revision was deleted")
        rev_dir = self._rev_dir(ref)
        self._mkdirs(self.root / ref.figure_id)
        self._mkdirs(rev_dir)  # no-op when completing a claimed, interrupted commit
        self._write(rev_dir / P.FIGURE_FILE, candidate.figure_bytes)
        self._write(rev_dir / P.CONFIG_FILE, candidate.config_bytes)
        source_entry = None
        if candidate.source is not None:
            source_id = f"sha256-{sha256_hex(candidate.source.payload_bytes)}"
            self._write(
                self.root / P.SOURCES_DIR / f"{source_id}.json",
                candidate.source.payload_bytes,
            )
            source_entry = {
                "source_id": source_id,
                "declared": candidate.source.declared,
            }
        # Inputs were normalized against the parent and checked in commit(), so the record
        # and the immutable descriptor agree on authoritative inputs.
        inputs = [edge.to_dict() for edge in candidate.inputs]
        from .history import change_details

        details = None
        if candidate.parent_ref:
            parent_bytes, before, before_config = self.read(candidate.parent_ref)
            parent_metadata = json.loads(parent_bytes)
            details = change_details(
                json.loads(before), json.loads(candidate.figure_bytes)
            )
            old_attachments = (
                (parent_metadata.get("source") or {}).get("declared") or {}
            ).get("attachments", [])
            new_attachments = (
                candidate.source.declared.get("attachments", [])
                if candidate.source
                else []
            )
            attachment_diff = change_details(
                {"attachments": old_attachments}, {"attachments": new_attachments}
            )
            details["attachments"] = attachment_diff["properties"]
            details["config"] = change_details(
                json.loads(before_config), json.loads(candidate.config_bytes)
            )["properties"]
            if attachment_diff["property_count"]:
                details["summary"] += "; updated data attachments"
            if details["config"]:
                details["summary"] += "; changed plot configuration"
        execution_record = self.read_execution(execution_id)
        timeline_after = (execution_record.get("parameters") or {}).get(
            "timeline_insert_after"
        )
        record = {
            "derived_from": candidate.parent_ref,
            "timeline_insert_after": timeline_after,
            "schema": REVISION_SCHEMA,
            "figure_id": ref.figure_id,
            "revision_id": ref.revision_id,
            "parents": (
                [FigureRef.parse(candidate.parent_ref).revision_id]
                if candidate.parent_ref
                else []
            ),
            "scope": {"session": self.session_id, "tenant": self.tenant},
            "label": candidate.label,
            "change": {
                "kind": candidate.kind,
                "actor": candidate.actor,
                "request": candidate.request,
                "resolved_intent": candidate.resolved_intent,
                "effect": None,
                "config_effect": None,
                "unbound_traces": [],
                "touches_data": details["touches_data"] if details else "unknown",
                "details": details,
                "summary": details["summary"] if details else "Original plot",
            },
            "inputs": inputs,
            "source": source_entry,
            "execution": {
                "execution_id": execution_id,
                "attempt_key": candidate.attempt_key,
                "attempt_digest": digest,
                "candidate_index": candidate.candidate_index,
                "template": candidate.template,
            },
            "figure": {
                "path": P.FIGURE_FILE,
                "sha256": sha256_hex(candidate.figure_bytes),
                "bytes": len(candidate.figure_bytes),
            },
            "config": {
                "path": P.CONFIG_FILE,
                "sha256": sha256_hex(candidate.config_bytes),
            },
            "created_at": _utcnow(),
        }
        rev_path = rev_dir / P.REVISION_FILE
        if self._read(rev_path) is None:  # a link at the marker raises, never reused
            # Last write = availability marker. If its directory fsync fails AFTER this call
            # linked it, the revision exists but this save's confirmation is uncertain:
            # PENDING (never rolled back; a retry or reader re-confirms it). Earlier failures
            # (nothing linked yet) still raise.
            _, unconfirmed = self._link(rev_path, _json_bytes(record))
            if unconfirmed is not None:
                return self._pending(candidate, unconfirmed)
            self._remember(rev_path)
        else:  # linked by a concurrent writer: never inherit, re-confirm it here
            try:
                self._require_confirmed(ref)
            except StorageError as exc:  # incl. DurabilityError
                return self._pending(candidate, exc)
        return self._result(candidate, ref)

    def _remember(self, marker: Path) -> None:
        """Record a durability receipt. Bookkeeping only: a failure here never undoes a
        successful acknowledgement; the next read simply re-confirms. That includes a typed
        StorageError from the chain check (e.g. an I/O fault on a directory lstat)."""
        try:
            self._confirmed.add(self._identity(marker))
        except (OSError, StorageError):
            pass

    def _pending(self, candidate: Candidate, exc: StorageError) -> CommitResult:
        """Uncertain save acknowledgement: no public ref/path (private claim locator only)."""
        return CommitResult(
            candidate_index=candidate.candidate_index,
            status=STATUS_PENDING,
            reason=safe_reason(exc),
            label=candidate.label,
        )

    def _result(self, candidate: Candidate, ref: FigureRef) -> CommitResult:
        try:
            self.history(ref.figure_id)
        except (StorageError, OSError) as exc:
            return self._pending(
                candidate,
                (
                    exc
                    if isinstance(exc, StorageError)
                    else StorageError("History publication failed")
                ),
            )
        return CommitResult(
            candidate_index=candidate.candidate_index,
            status=STATUS_AVAILABLE,
            ref=str(ref),
            path=self.figure_path(ref),
            label=candidate.label,
        )

    def _new_figure_dir(self) -> str:
        self._mkdirs(
            self.root
        )  # the root checked by lstat; the new id is a fresh mkdir
        for _ in range(16):
            figure_id = new_id(PREFIX_FIGURE)
            try:
                self._mkdir_new(self.root / figure_id)
            except FileExistsError:
                continue
            self._fsync(self.root)
            return figure_id
        raise RuntimeError("could not allocate a unique figure id")

    def _read_claim(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read an existing claim; its entry is confirmed again before it is relied on.
        Real directories only (lstat) and no final link: a linked claim is never reused.
        """
        raw = self._read(path)
        if raw is None:
            return None
        claim = json.loads(raw)
        self._fsync(path.parent)
        return claim

    # ------------------------------------------------------------------ reads
    def _refuse_deleted(self, ref: FigureRef) -> None:
        """A tombstoned ref is DELETED (reasons/3 ``figure_deleted``), distinguishable from
        one that never existed: the tombstone is retained (typed StorageError on I/O).
        """
        if self._ref_deleted(ref):
            raise FigureDeletedError("this revision was deleted")

    def _fenceable(self, ref: FigureRef) -> bool:
        """May ``ref`` be HIDDEN (tombstoned)? Registered, or its revision entry sits behind
        a link: hiding never follows or reuses it, and removal refuses a link (kept)."""
        marker = self._rev_dir(ref) / P.REVISION_FILE
        try:
            return self._has_marker(ref)
        except StorageError:
            if self._linked(marker):
                return True
            raise

    def _has_marker(self, ref: FigureRef) -> bool:
        """Registered and NOT deleted (a tombstone hides it; typed on I/O error)."""
        if self._ref_deleted(ref):
            return False
        return self._present(self._rev_dir(ref) / P.REVISION_FILE)

    def _load_record(self, path: Path, figure_id: str, revision_id: str) -> Dict:
        data = self._read_or(path, StorageError("revision marker vanished"))
        return _parse_marker(
            data,
            {"figure_id": figure_id, "revision_id": revision_id},
            ("figure", "config"),
        )

    def _confirm_entries(
        self, marker: Path, dirs: Sequence[Path], seen: Optional[set] = None
    ) -> None:
        """Re-confirm a marker's entries (fsync each dir) unless THIS marker file already
        has a receipt; raises DurabilityError on failure. ``seen`` shares the fsyncs of one
        listing call (a dir fsync after the scan covers every entry the scan returned).
        """
        try:
            key = self._identity(marker)
        except OSError:  # explicit typed error, never a false absence
            raise StorageError("marker stat failed") from None
        if key in self._confirmed:
            return
        for d in dirs:
            if seen is not None and d in seen:
                continue
            self._fsync(d)
            if seen is not None:
                seen.add(d)
        self._confirmed.add(key)

    def _require_confirmed(self, ref: FigureRef, seen: Optional[set] = None) -> None:
        rev_dir = self._rev_dir(ref)
        self._confirm_entries(
            rev_dir / P.REVISION_FILE, (rev_dir, rev_dir.parent, self.root), seen
        )

    def _is_available(self, ref: FigureRef) -> bool:
        """Marker present AND durability confirmed (re-confirmed if needed)."""
        if not self._has_marker(ref):
            return False
        try:
            self._require_confirmed(ref)
        except StorageError:  # incl. DurabilityError
            return False
        return True

    def _project(self, rec: Dict[str, Any], seen: set) -> Dict[str, Any]:
        """List-time availability PROJECTION (metadata only; revision.json never rewritten)."""
        try:
            self._require_confirmed(
                FigureRef(rec["figure_id"], rec["revision_id"]), seen
            )
            rec["availability"] = STATUS_AVAILABLE
        except StorageError as exc:  # incl. DurabilityError: listed, explicitly pending
            rec["availability"] = STATUS_PENDING
            rec["availability_reason"] = safe_reason(exc)
        return rec

    def read_record(self, ref: "str | FigureRef") -> Dict[str, Any]:
        """Revision METADATA only. Payload integrity is NOT verified here (use :meth:`read`)."""
        ref = FigureRef.parse(ref)
        path = self._rev_dir(ref) / P.REVISION_FILE
        self._refuse_deleted(ref)
        if not self._present(path):  # typed StorageError on I/O, never a false absence
            raise FigureNotFoundError(
                "figure not found", details={"reason": "revision_not_found"}
            )
        self._require_confirmed(ref)  # pending -> DurabilityError, never "not found"
        return self._load_record(path, ref.figure_id, ref.revision_id)

    def read(self, ref: "str | FigureRef") -> Tuple[bytes, bytes, bytes]:
        """Return (revision, figure, config) bytes, verified against recorded shas."""
        ref = FigureRef.parse(ref)
        rev_dir = self._rev_dir(ref)
        record_bytes_path = rev_dir / P.REVISION_FILE
        self._refuse_deleted(ref)
        if not self._present(record_bytes_path):
            raise FigureNotFoundError(
                "figure not found", details={"reason": "revision_not_found"}
            )
        self._require_confirmed(ref)  # the memo never skips the hash checks below
        record_bytes = self._read_or(
            record_bytes_path, StorageError("revision marker vanished")
        )
        record = _parse_marker(
            record_bytes,
            {"figure_id": ref.figure_id, "revision_id": ref.revision_id},
            ("figure", "config"),
        )
        figure_bytes = self._read_or(
            rev_dir / P.FIGURE_FILE,
            IntegrityError(
                "figure.json missing after save",
                details={"reason": "figure_changed", "file": P.FIGURE_FILE},
            ),
        )
        config_bytes = self._read_or(
            rev_dir / P.CONFIG_FILE,
            IntegrityError(
                "config.json missing after save",
                details={"reason": "config_changed", "file": P.CONFIG_FILE},
            ),
        )
        if sha256_hex(figure_bytes) != record["figure"]["sha256"]:
            raise IntegrityError(
                "figure.json changed after save",
                details={"reason": "figure_changed", "file": P.FIGURE_FILE},
            )
        if sha256_hex(config_bytes) != record["config"]["sha256"]:
            raise IntegrityError(
                "config.json changed after save",
                details={"reason": "config_changed", "file": P.CONFIG_FILE},
            )
        return record_bytes, figure_bytes, config_bytes

    def read_source(self, source_id: str) -> bytes:
        if not source_id.startswith("sha256-"):
            raise FigureNotFoundError(
                "unknown source", details={"reason": "source_not_found"}
            )
        path = self.root / P.SOURCES_DIR / f"{source_id}.json"
        data = self._read(path)  # checked chain, never through a link
        if data is None:
            raise FigureNotFoundError(
                "source not found", details={"reason": "source_not_found"}
            )
        if f"sha256-{sha256_hex(data)}" != source_id:
            raise IntegrityError(
                "source changed after save",
                details={"reason": "source_changed"},
            )
        return data

    def _iter_records(self, figure_id=None) -> Iterable[Dict[str, Any]]:
        """Every registered revision record. Enumeration/read failures raise TYPED errors and
        a corrupt registered record raises IntegrityError: never skipped, never an empty or
        partial catalog presented as complete, never a fabricated pending item.
        A revision directory WITHOUT a marker is an unfinished commit and is not listed.
        """
        # Only REAL directories (lstat) are enumerated; a link in the catalog is never
        # followed or listed (a link is not an entry this store created).
        try:
            if figure_id is not None:
                FigureRef.parse(figure_id + "@r_aaaaaaaaaa")
                directory = self.root / figure_id
                fig_dirs = [directory] if self._require_real_dir(directory) else []
            else:
                fig_dirs = [
                    self.root / name
                    for name in self._listdir_real(self.root)
                    if name not in P.RESERVED_DIRS
                    and _stat.S_ISDIR(self._lstat_mode(self.root / name))
                ]
        except StorageError:
            raise StorageError("catalog enumeration failed") from None
        out: List[Dict[str, Any]] = []
        for fig_dir in fig_dirs:
            try:
                rev_dirs = [
                    fig_dir / name
                    for name in self._listdir_real(fig_dir)
                    if _stat.S_ISDIR(self._lstat_mode(fig_dir / name))
                ]
            except StorageError:
                raise StorageError("catalog enumeration failed") from None
            for rev_dir in rev_dirs:
                rec_path = rev_dir / P.REVISION_FILE
                if (
                    is_valid_id(fig_dir.name)
                    and is_valid_id(rev_dir.name)
                    and self._ref_deleted(FigureRef(fig_dir.name, rev_dir.name))
                ):
                    continue  # deleted (cleanup may be pending): never listed
                if self._present(rec_path):
                    out.append(self._load_record(rec_path, fig_dir.name, rev_dir.name))
        out.sort(
            key=lambda r: (
                r.get("created_at", ""),
                r.get("figure_id", ""),
                r.get("revision_id", ""),
            )
        )
        return out

    def list(self, cursor: Optional[str] = None, limit: int = 500) -> Page:
        """Revision metadata (historical included). Payload integrity is NOT verified."""
        records = list(self._iter_records())
        start = int(cursor) if cursor else 0
        seen: set = set()
        items = [self._project(r, seen) for r in records[start : start + limit]]
        nxt = str(start + limit) if start + limit < len(records) else None
        return Page(items=items, next_cursor=nxt, catalog_generation=str(len(records)))

    def dependents(self, instance_id: str) -> List[Tuple[str, bool]]:
        """``(ref, verified)`` for every revision with an input edge of exactly ``instance_id``.

        One tuple per ref, in catalog order; ``verified`` is True iff ANY matching edge of that
        revision is verified (an unverified alias never hides a verified one). Only a stored
        JSON ``true`` grants verified authority: ``1``, ``"true"``, ``"false"`` or any other
        non-bool stays unverified ("may depend"), never promoted. The ID must be a
        non-empty ``str``: an unknown identity is refused before the catalog is read (it would
        otherwise match null edges); nothing is ever matched by name.
        """
        if not isinstance(instance_id, str):
            raise TypeError("dependents() needs an instance_id string")
        if not instance_id:
            raise ValueError("dependents() needs a non-empty instance_id")
        found: List[Tuple[str, bool]] = []
        for rec in self._iter_records():
            matches = [
                e for e in rec.get("inputs", []) if e.get("instance_id") == instance_id
            ]
            if matches:
                found.append(
                    (
                        f"{rec['figure_id']}@{rec['revision_id']}",
                        any(e.get("verified") is True for e in matches),
                    )
                )
        return found

    # ------------------------------------------------------------- renditions
    def _renditions_dir(self, ref: FigureRef) -> Path:
        return self._rev_dir(ref) / P.RENDITIONS_DIR

    def add_rendition(
        self,
        ref: "str | FigureRef",
        data: bytes,
        settings: Dict[str, Any],
        renderer_receipt: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Append a rendition of an available revision (contract §9; local = detection only).

        Never reachable from the child. The receipt's input shas must equal the revision's
        recorded figure/config shas. Bytes first, record last (the record is the marker).
        Runs under the root's writer lock.
        """
        with self._lock:
            return self._add_rendition(ref, data, settings, renderer_receipt)

    def _add_rendition(
        self,
        ref: "str | FigureRef",
        data: bytes,
        settings: Dict[str, Any],
        renderer_receipt: Dict[str, Any],
    ) -> Dict[str, Any]:
        ref = FigureRef.parse(ref)
        record = self.read_record(ref)
        inputs = (renderer_receipt or {}).get("inputs") or {}
        if (
            inputs.get("figure_sha256") != record["figure"]["sha256"]
            or inputs.get("config_sha256") != record["config"]["sha256"]
        ):
            raise IntegrityError(
                "rendition inputs do not match the revision",
                details={"reason": "rendition_inputs_mismatch"},
            )
        fmt = settings.get("format")
        if fmt not in ("html", "png", "svg"):
            raise FigureInvalidError(
                "unsupported rendition format", details={"reason": "rendition_format"}
            )
        rendition_id = new_id(PREFIX_RENDITION)
        rdir = self._renditions_dir(ref)
        data_file = f"{rendition_id}.{fmt}"
        self._write(rdir / data_file, data)
        rendition = {
            "schema": "lobster.figure_rendition/1",
            "rendition_id": rendition_id,
            "figure_id": ref.figure_id,
            "revision_id": ref.revision_id,
            "file": data_file,
            "path": self._rel(rdir / data_file),
            "sha256": sha256_hex(data),
            "bytes": len(data),
            "settings": settings,
            "renderer": renderer_receipt.get("renderer"),
            "inputs": inputs,
            "trust": "local_renderer_detection_only",
            "created_at": _utcnow(),
        }
        marker_path = rdir / f"{rendition_id}.json"
        _, unconfirmed = self._link(marker_path, _json_bytes(rendition))
        if unconfirmed is not None:  # linked, confirmation uncertain: never rolled back
            return dict(
                rendition,
                availability=STATUS_PENDING,
                availability_reason=safe_reason(unconfirmed),
            )
        self._remember(marker_path)
        return dict(rendition, availability=STATUS_AVAILABLE)

    def list_renditions(self, ref: "str | FigureRef") -> List[Dict[str, Any]]:
        """Rendition records of a revision (metadata; bytes not verified here)."""
        ref = FigureRef.parse(ref)
        self._refuse_deleted(ref)
        rdir = self._renditions_dir(ref)
        try:  # checked chain: a linked revision or renditions directory raises typed
            paths = [
                rdir / name
                for name in self._listdir_real(rdir)
                if name.startswith("p_") and name.endswith(".json")
            ]
        except StorageError:
            raise StorageError("rendition enumeration failed") from None
        out = []
        seen: set = set()
        for path in paths:
            if not self._present(path):  # a linked marker raises typed, never reused
                continue
            rec = _parse_marker(  # corrupt -> IntegrityError, never skipped
                self._read_or(path, StorageError("rendition marker vanished")),
                {
                    "rendition_id": path.stem,
                    "figure_id": ref.figure_id,
                    "revision_id": ref.revision_id,
                },
                (None,),
            )
            try:  # availability projection (metadata only; record never rewritten)
                self._confirm_entries(path, (rdir, rdir.parent), seen)
                rec["availability"] = STATUS_AVAILABLE
            except StorageError as exc:  # incl. DurabilityError
                rec["availability"] = STATUS_PENDING
                rec["availability_reason"] = safe_reason(exc)
            out.append(rec)
        return out

    def read_rendition(self, ref: "str | FigureRef", rendition_id: str) -> bytes:
        ref = FigureRef.parse(ref)
        for rec in self.list_renditions(ref):
            if rec.get("rendition_id") == rendition_id:
                if rec.get("availability") == STATUS_PENDING:
                    raise DurabilityError(
                        "rendition save confirmation pending",
                        details={"reason": "dir_fsync_failed"},
                    )
                changed = IntegrityError(
                    "rendition changed after save",
                    details={"reason": "rendition_changed"},
                )
                path = self._renditions_dir(ref) / str(rec.get("file"))
                data = self._read_or(
                    path, changed
                )  # missing data = changed; a link raises
                if sha256_hex(data) != rec["sha256"]:
                    raise IntegrityError(
                        "rendition changed after save",
                        details={"reason": "rendition_changed"},
                    )
                return data
        raise FigureNotFoundError(
            "rendition not found", details={"reason": "rendition_not_found"}
        )

    def find_rendition(
        self,
        ref: "str | FigureRef",
        settings: Dict[str, Any],
        renderer_receipt: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """An existing, byte-verified rendition with identical settings, renderer and inputs."""
        ref = FigureRef.parse(ref)
        for rec in self.list_renditions(ref):
            if (
                rec.get("settings") == settings
                and rec.get("renderer") == renderer_receipt.get("renderer")
                and rec.get("inputs") == renderer_receipt.get("inputs")
            ):
                if rec.get("availability") == STATUS_PENDING:
                    return rec  # never append a duplicate while confirmation is pending
                try:
                    self.read_rendition(ref, rec["rendition_id"])
                except IntegrityError:
                    continue  # damaged: a new rendition is appended, the old one is kept
                return rec
        return None

    def execution_outcomes(self, execution_id: str) -> List[Dict[str, Any]]:
        """Reconcile planned candidates against available revisions (``not_committed`` gaps)."""
        record = self.read_execution(execution_id)
        by_index = {}
        for rec in self._iter_records():
            ex = rec.get("execution") or {}
            if ex.get("execution_id") == execution_id:
                by_index[ex.get("candidate_index")] = FigureRef(
                    rec["figure_id"], rec["revision_id"]
                )
        outcomes = []
        for planned in record.get("planned_candidates", []):
            idx = planned["index"]
            ref = by_index.get(idx)
            if ref is None:
                status = "not_committed"
            else:
                status = STATUS_AVAILABLE if self._is_available(ref) else STATUS_PENDING
            outcomes.append(
                {
                    "index": idx,
                    "status": status,
                    "ref": str(ref) if status == STATUS_AVAILABLE else None,
                }
            )
        return outcomes
