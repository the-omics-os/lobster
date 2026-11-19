"""Run a catalog engine on one dedicated worker thread, failing open.

One thread: tool bodies, and so ``save_modality``, run concurrently on LangGraph
executor threads and on the background event-loop thread, while an ORM engine keeps
one connection per thread and refuses to work on a thread with a running event loop.
One worker serialises every engine call and owns the only connection.

One lock file per workspace: two Lobster processes on the same workspace must not
write the catalog at the same time. Without it, saves of the same key fail at once
with "database is locked" or split one version history into two.

Fail-open: the catalog is a record, never a reason for an analysis to fail. A call
that raises returns nothing and is reported once; a call that does not finish within
the timeout marks the catalog degraded, and every later call is skipped. A call that
timed out is abandoned: if it has not reached the engine yet (queued, or waiting for
the lock), it never does.

A recorded save (``save_section``) holds the lock from ``keep_path`` until its file
is registered, so two processes saving the same file cannot record each other's
bytes. It waits at most ``KEEP_TIMEOUT_S`` to start (of which at most
``LOCK_WAIT_S`` for another process's lock; otherwise the save is not recorded,
and the catalog is not degraded), then ``REGISTER_WAIT_S`` for the registration,
which goes on in the background when it takes longer.

Start-up has its own budget: building the engine (imports, database connection) is
not charged to the first save. A call that needs an answer waits a few seconds for
the engine; if it is still starting, writes are queued behind the build and reads
answer the default. Only a build that exceeds its own budget degrades the catalog.

The worker is a daemon thread, so a stuck engine never blocks interpreter exit; the
open run is closed by an ``atexit`` handler with a shorter timeout. If the engine is
still building then, the handler waits for it, within that same timeout, so a
short command's run is recorded. A forked child
inherits no worker: its copies of the catalog are degraded at once and skip the
exit handler (the parent owns the run).
"""

from __future__ import annotations

import atexit
import dataclasses
import errno
import logging
import os
import queue
import threading
import time
import weakref
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

from lobster.core.catalog.port import (
    ArtifactRef,
    Catalog,
    CatalogStatus,
    RecordedSave,
    RunStatus,
    SessionInfo,
    TransformRef,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0
AT_EXIT_TIMEOUT_S = 10.0
BUILD_TIMEOUT_S = 120.0
STARTUP_WAIT_S = 5.0
KEEP_TIMEOUT_S = 5.0
REGISTER_WAIT_S = 5.0
LOCK_WAIT_S = 3.0
LOCK_POLL_S = 0.05

_instances: "weakref.WeakSet[ThreadedCatalog]" = weakref.WeakSet()


def _after_fork_in_child() -> None:
    for catalog in list(_instances):
        catalog._forked_in_child()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


class CatalogEngineExit(RuntimeError):
    """An engine raised ``SystemExit`` or another non-``Exception`` error."""


class _EngineUnavailable(RuntimeError):
    """The engine failed to start, or the call was abandoned; nothing to report."""


class _LockBusy(RuntimeError):
    """Another process held the workspace catalog lock for too long."""


_BUSY = {errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, getattr(errno, "EDEADLK", -1)}


def _try_lock(handle: Any) -> bool:
    """Take the lock if it is free; other errors (no locking here) are raised."""
    try:
        if os.name == "nt":  # pragma: no cover - Windows specific
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in _BUSY:
            return False
        raise
    return True


def _unlock(handle: Any) -> None:
    if os.name == "nt":  # pragma: no cover - Windows specific
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _acquire(path: Path, wait: Optional[float], abandoned: threading.Event) -> Any:
    """An open file holding the workspace catalog lock (see ``workspace_lock``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        deadline = None if wait is None else time.monotonic() + wait
        while not _try_lock(handle):
            if abandoned.is_set():
                raise _EngineUnavailable()
            if deadline is not None and time.monotonic() >= deadline:
                raise _LockBusy(
                    f"another process held the catalog lock for more than {wait:g} s"
                )
            abandoned.wait(LOCK_POLL_S)
    except BaseException:
        handle.close()
        raise
    return handle


def _release(handle: Any) -> None:
    try:
        _unlock(handle)
    finally:
        handle.close()


@contextmanager
def workspace_lock(
    path: Path,
    wait: Optional[float] = None,
    abandoned: Optional[threading.Event] = None,
) -> Iterator[None]:
    """Hold the workspace catalog lock (the same lock as ``InterProcessFileLock``).

    Waits at most ``wait`` seconds (None: until it is free), and gives up as soon as
    ``abandoned`` is set, so a call its caller stopped waiting for never runs late.
    """
    handle = _acquire(path, wait, abandoned or threading.Event())
    try:
        yield
    finally:
        _release(handle)


def _step_log(session: SessionInfo) -> Optional[Path]:
    """The session's step log, if it has one yet."""
    if session.session_dir is None:
        return None
    candidate = Path(session.session_dir) / "provenance.jsonl"
    return candidate if candidate.is_file() else None


def _file_state(path: Path) -> Optional[tuple]:
    """What tells one write of ``path`` from the next (inode, size, mtime)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


class _Worker:
    """One daemon thread running submitted callables in order."""

    def __init__(self, name: str, on_exit: Optional[Callable[[], None]] = None):
        self._queue: "queue.SimpleQueue[Optional[tuple]]" = queue.SimpleQueue()
        self._stopped = False
        self._on_exit = on_exit
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    @property
    def ident(self) -> Optional[int]:
        return self._thread.ident

    def submit(self, fn: Callable[[], Any]) -> Future:
        if self._stopped:
            raise RuntimeError("catalog worker stopped")
        future: Future = Future()
        self._queue.put((future, fn))
        return future

    def stop(self) -> None:
        self._stopped = True
        self._queue.put(None)

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                if self._on_exit is not None:
                    self._on_exit()
                return
            future, fn = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except Exception as exc:
                future.set_exception(exc)
            except (
                BaseException
            ) as exc:  # SystemExit from an engine must not end the worker
                future.set_exception(CatalogEngineExit(f"{type(exc).__name__}: {exc}"))


class ThreadedCatalog:
    """A ``Catalog`` that runs ``engine_factory()`` and every engine call on one worker.

    ``engine_factory`` runs on the worker too, so the engine is built (and its
    database connected) on the thread that will use it. Writes run under an
    inter-process lock on ``lock_path``; ``status`` is a read. ``keep_path`` only
    names a place, and the saving thread keeps the bytes: while the engine is
    starting, or when it answers late, the save goes ahead unprepared and nothing
    is moved. A recorded save holds the lock from ``keep_path`` to its registration
    (``save_section``), which excludes other processes completely. Within one
    process the hold is shared: two threads saving the same key at the same time
    are not kept apart by it.
    """

    def __init__(
        self,
        engine_factory: Callable[[], Catalog],
        *,
        lock_path: Path,
        engine: str,
        mode: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        build_timeout: float = BUILD_TIMEOUT_S,
        startup_wait: float = STARTUP_WAIT_S,
        key_root: Optional[Path] = None,
    ):
        self._lock_path = Path(lock_path)
        self._key_root = None if key_root is None else Path(key_root)
        self._label = (engine, mode)
        self._timeout = timeout
        self._build_timeout = build_timeout
        self._startup_wait = startup_wait
        self._ready = threading.Event()
        self._forked = False
        # Reentrant: a done-callback of a queued call may run on the submitting
        # thread, inside ``begin_run``'s hold of this lock.
        self._state_lock = threading.RLock()
        self._degraded_reason: Optional[str] = None
        self._reported: set = set()
        self._session: Optional[SessionInfo] = None
        self._run_open = False
        # How the open run ended, as its command said; None leaves it open (an
        # unfinished run is never recorded as completed).
        self._end_status: Optional[RunStatus] = None
        self._end_reason: Optional[str] = None
        self._turn_open = False  # a session's turn started and has not ended
        self._engine: Optional[Catalog] = None
        # The lock held for a recorded save, from keep_path to its registration;
        # used on the worker only.
        self._hold_handle: Any = None
        self._hold_count = 0
        self._worker = _Worker(name="lobster-catalog", on_exit=self._drop_hold)
        self._worker.submit(lambda: self._build(engine_factory))
        self._watchdog = threading.Timer(build_timeout, self._check_started)
        self._watchdog.daemon = True
        self._watchdog.start()
        _instances.add(self)
        atexit.register(self._at_exit)

    @property
    def key_root(self) -> Optional[Path]:
        """The folder a saved file's key is relative to (None: its workspace)."""
        return self._key_root

    # -- Catalog --------------------------------------------------------------

    def begin_run(self, session: SessionInfo) -> None:
        """Fire and forget: the next call waits behind it on the worker."""
        with self._state_lock:
            previous = self._session if self._run_open else None
            previous_status, self._end_status = self._end_status, None
            previous_reason, self._end_reason = self._end_reason, None
            self._turn_open = False
            self._session = session
            self._run_open = True
            # Both calls are queued while the lock is held (``wait=False`` returns
            # at once), so two switches cannot interleave their end and begin.
            # Known, deferred: a save whose section is open during a switch is
            # registered under the run the switch begins.
            if previous is not None and previous_status is not None:
                # A session switch: the run being left ends with its own step log.
                self._call(
                    "end_run",
                    previous_status,
                    _step_log(previous),
                    None,
                    previous_reason,
                    write=True,
                    wait=False,
                )
            self._call("begin_run", session, write=True, wait=False)

    @contextmanager
    def save_section(self, path: Path) -> Iterator[RecordedSave]:
        """One recorded save of ``path``, under the workspace lock throughout.

        The lock is taken (waiting at most ``LOCK_WAIT_S``) before ``keep_path``
        and released after the registration, so another process cannot write the
        same file in between. When it is not free in time, or the engine does not
        answer within ``KEEP_TIMEOUT_S``, the save is not recorded (fail-open).
        While the engine is starting, the save is registered later, unprepared.
        The registration is skipped when the file changed after the save wrote it.
        """
        path = Path(path)
        if self._degraded_reason is not None:
            yield RecordedSave(recorded=False)
            return
        if not self._ready.is_set() and not self._ready.wait(self._startup_wait):
            self._note_starting("save")
            yield RecordedSave(
                register=lambda **fields: self.register_artifact(path, **fields)
            )
            return
        begun = self._begin_section(path)
        if begun is None:
            yield RecordedSave(recorded=False)
            # Once the lock is free, no record may point at bytes this save replaced.
            self.note_unrecorded(path)
            return
        ended = []

        def register(**fields: Any) -> Optional[ArtifactRef]:
            ended.append(True)
            return self._end_section(path, fields, _file_state(path))

        try:
            yield RecordedSave(keep=begun[0], register=register)
        finally:
            if not ended:
                self._end_section(path, None, None)

    def _begin_section(self, path: Path) -> Optional[tuple]:
        abandoned = threading.Event()

        def begin() -> tuple:
            engine = self._engine
            if (
                engine is None
                or self._degraded_reason is not None
                or abandoned.is_set()
            ):
                raise _EngineUnavailable()
            self._hold(LOCK_WAIT_S, abandoned)
            try:
                if abandoned.is_set():
                    raise _EngineUnavailable()
                try:
                    keep, error = engine.keep_path(path), None
                except Exception as exc:  # no keep, but the save is still recorded
                    keep, error = None, exc
                if abandoned.is_set():
                    raise _EngineUnavailable()
            except BaseException:
                self._unhold()
                raise
            return keep, error

        try:
            future = self._worker.submit(begin)
        except RuntimeError:
            return None
        try:
            keep, error = future.result(timeout=KEEP_TIMEOUT_S)
        except FutureTimeout:
            abandoned.set()
            future.cancel()
            future.add_done_callback(self._unhold_if_begun)
            self._report(
                "save",
                f"the catalog did not answer within {KEEP_TIMEOUT_S:g} s; "
                "saves are not recorded while it is busy",
            )
            return None
        except _EngineUnavailable:
            return None
        except _LockBusy as exc:
            self._report("save", f"{exc}; saves are not recorded while it is busy")
            return None
        except Exception as exc:
            self._report("save", f"{type(exc).__name__}: {exc}")
            return None
        if error is not None:
            self._report("keep_path", f"{type(error).__name__}: {error}")
        return (self._usable_keep(path, keep),)

    def _end_section(
        self, path: Path, fields: Optional[Dict[str, Any]], written: Optional[tuple]
    ) -> Optional[ArtifactRef]:
        def end() -> Optional[ArtifactRef]:
            try:
                engine = self._engine
                if fields is None or engine is None or self._degraded_reason:
                    return None
                if written is None or _file_state(path) != written:
                    self._report(
                        "register_artifact",
                        "the file was replaced by another writer before it was "
                        "recorded; that save is not recorded",
                    )
                    return None
                ref = engine.register_artifact(path, **fields)
                if _file_state(path) != written:
                    # Changed while LaminDB read it: no record may claim the
                    # bytes now at the name.
                    engine.note_unrecorded(path)
                    self._report(
                        "register_artifact",
                        "the file was replaced by another writer while it was "
                        "recorded; its record is marked",
                    )
                return ref
            finally:
                self._unhold()

        try:
            future = self._worker.submit(end)
        except RuntimeError:
            return None  # stopped: the worker let the lock go when it stopped
        try:
            return future.result(timeout=min(self._timeout, REGISTER_WAIT_S))
        except FutureTimeout:  # it finishes in the background, then lets go
            future.add_done_callback(
                lambda done: self._observe("register_artifact", done)
            )
        except Exception as exc:
            self._report("register_artifact", f"{type(exc).__name__}: {exc}")
        return None

    def _hold(self, wait: Optional[float], abandoned: threading.Event) -> None:
        if self._hold_count == 0:
            self._hold_handle = _acquire(self._lock_path, wait, abandoned)
        self._hold_count += 1

    def _unhold(self) -> None:
        if self._hold_count == 0:
            return
        self._hold_count -= 1
        if self._hold_count == 0:
            self._drop_hold()

    def _drop_hold(self) -> None:
        handle, self._hold_handle, self._hold_count = self._hold_handle, None, 0
        if handle is not None:
            try:
                _release(handle)
            except OSError:
                logger.debug("Catalog: could not release the lock", exc_info=True)

    def _unhold_if_begun(self, future: Future) -> None:
        # A section the caller gave up on, but whose start won the race: let go.
        if future.cancelled() or future.exception() is not None:
            return
        try:
            self._worker.submit(self._unhold)
        except RuntimeError:
            pass

    def _usable_keep(self, path: Path, target: Any) -> Optional[Path]:
        if target is None:
            return None
        try:
            target = Path(target)
        except TypeError:
            target = None
        if target is None or target == path or target.exists():
            self._report("keep_path", "the engine answered an unusable path")
            return None
        return target

    def keep_path(self, path: Path) -> Optional[Path]:
        path = Path(path)
        target = self._call(
            "keep_path",
            path,
            write=True,
            late=False,
            on_timeout="report",
            timeout=KEEP_TIMEOUT_S,
            lock_wait=LOCK_WAIT_S,
        )
        return self._usable_keep(path, target)

    def register_artifact(
        self,
        path: Path,
        *,
        key: str,
        kind: str,
        otype: Optional[str],
        description: str,
        meta: Dict[str, Any],
    ) -> Optional[ArtifactRef]:
        return self._call(
            "register_artifact",
            Path(path),
            write=True,
            on_timeout="continue",
            timeout=min(self._timeout, REGISTER_WAIT_S),
            key=key,
            kind=kind,
            otype=otype,
            description=description,
            meta=meta,
        )

    def note_unrecorded(self, path: Path) -> None:
        """Fire and forget, waiting at most ``LOCK_WAIT_S`` for the lock.

        When the lock stays busy the stale records are marked by the next recorded
        save of the same file instead.
        """
        self._call(
            "note_unrecorded", Path(path), write=True, wait=False, lock_wait=LOCK_WAIT_S
        )

    def link_input(self, path: Path) -> Optional[ArtifactRef]:
        """Fire and forget: hashing a large input must not hold up the load."""
        return self._call("link_input", Path(path), write=True, wait=False)

    def register_transform(
        self, *, key: str, kind: str, source: str, description: str
    ) -> Optional[TransformRef]:
        return self._call(
            "register_transform",
            write=True,
            key=key,
            kind=kind,
            source=source,
            description=description,
        )

    def end_run(
        self,
        status: RunStatus,
        report_path: Optional[Path] = None,
        environment_path: Optional[Path] = None,
        reason: Optional[str] = None,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        with self._state_lock:
            self._run_open = False
        self._call(
            "end_run",
            status,
            report_path,
            environment_path,
            reason,
            write=True,
            timeout=timeout,
        )

    def status(self) -> CatalogStatus:
        engine, mode = self._label
        fallback = CatalogStatus(engine=engine, mode=mode)  # type: ignore[arg-type]
        current = self._call("status", write=False, default=fallback)
        if not isinstance(current, CatalogStatus):
            current = fallback
        reason = self._degraded_reason
        if reason is not None:
            return dataclasses.replace(current, degraded=True, reason=reason)
        return current

    # -- lifecycle ------------------------------------------------------------

    @property
    def degraded(self) -> bool:
        return self._degraded_reason is not None

    @property
    def worker_ident(self) -> Optional[int]:
        return self._worker.ident

    def set_end_status(
        self,
        status: Optional[RunStatus],
        session_id: Optional[str] = None,
        reason: Optional[str] = None,
        *,
        if_unset: bool = False,
        if_unfinished: bool = False,
    ) -> None:
        """How the open run ends when the process exits or the session switches.

        With ``session_id``, only that session's run: a server running sessions
        side by side must not end one with another's outcome. ``reason`` says why a
        run did not complete (an exit code, an error class). None: a turn started,
        so the run has no end status until the turn ends.

        ``if_unset`` sets it only when nothing is set and no turn is running (a
        chat that ends normally ends as its last turn did, never completed over
        a turn cut short). ``if_unfinished`` sets it unless a finished turn's
        outcome stands (SIGTERM aborts a running turn, not one already over).
        """
        with self._state_lock:
            current = None if self._session is None else self._session.session_id
            if not self._run_open or session_id not in (None, current):
                return
            if status is None:
                self._end_status, self._end_reason = None, None
                self._turn_open = True
                return
            if if_unset and (self._end_status is not None or self._turn_open):
                return
            if if_unfinished and self._end_status is not None:
                return
            self._end_status, self._end_reason = status, reason
            self._turn_open = False

    def close(self, status: RunStatus = "completed") -> None:
        """End the open run and stop the worker (tests, explicit shutdown)."""
        atexit.unregister(self._at_exit)
        self._finish(status, timeout=self._timeout)

    def _at_exit(self) -> None:
        if self._forked:
            return
        self._finish(
            self._end_status,
            reason=self._end_reason,
            timeout=min(self._timeout, AT_EXIT_TIMEOUT_S),
            wait_for_build=True,
        )

    def _forked_in_child(self) -> None:
        # Locks may have been held by threads that do not exist in the child.
        self._state_lock = threading.RLock()
        self._forked = True
        self._run_open = False
        self._degraded_reason = "the process was forked; the parent keeps the catalog"

    def _finish(
        self,
        status: Optional[RunStatus],
        *,
        timeout: float,
        reason: Optional[str] = None,
        wait_for_build: bool = False,
    ) -> None:
        # A degraded catalog ends nothing: if it degraded after begin_run (a call
        # timed out), that run stays open in the catalog, with no finish time.
        self._watchdog.cancel()
        deadline = time.monotonic() + timeout
        with self._state_lock:  # a switch at exit must not be read half-way
            # No status: the command never said how it ended; the run stays open.
            is_open = (
                self._run_open and self._degraded_reason is None and status is not None
            )
            session = self._session
        if is_open:
            starting = not self._ready.is_set()
            report = None if session is None else _step_log(session)
            self.end_run(status, report, None, reason, timeout=timeout)
            if starting and wait_for_build:
                self._wait_for_worker(deadline)
        self._worker.stop()

    def _wait_for_worker(self, deadline: float) -> None:
        """Wait until the calls queued behind a still-running build have run.

        A one-shot command can exit before the engine is built; the worker is a
        daemon, so the run's begin and end, queued behind the build, would die with
        the process. Waits within the exit budget, never beyond it.
        """
        try:
            drained = self._worker.submit(lambda: None)
        except RuntimeError:
            return
        try:
            drained.result(timeout=max(0.0, deadline - time.monotonic()))
        except FutureTimeout:
            logger.info("Catalog: the engine did not finish starting before exit")

    # -- internals --------------------------------------------------------------

    def _build(self, engine_factory: Callable[[], Catalog]) -> None:
        try:
            engine = engine_factory()
        except BaseException as exc:  # SystemExit included: an engine that cannot start
            detail = f"{type(exc).__name__}: {exc}"[:200]
            self._degrade(f"the catalog could not start ({detail})")
            logger.debug("Catalog engine start failed", exc_info=True)
            self._ready.set()
            return
        # Publishing the engine and the watchdog's verdict are decided under one
        # lock: either the watchdog latched first and the engine is never used, or
        # the engine is ready and the watchdog stands down.
        with self._state_lock:
            if self._degraded_reason is None:
                self._engine = engine
            self._ready.set()

    def _check_started(self) -> None:
        with self._state_lock:
            if self._ready.is_set():
                return
            reason = f"the catalog did not start within {self._build_timeout:g} s"
            latched = self._latch(reason)
        if latched:
            self._warn_degraded(reason)

    def _call(
        self,
        op: str,
        *args: Any,
        write: bool,
        wait: bool = True,
        late: bool = True,
        default: Any = None,
        timeout: Optional[float] = None,
        on_timeout: str = "degrade",
        lock_wait: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        if self._degraded_reason is not None:
            return default
        abandoned = threading.Event()

        def body() -> Any:
            engine = self._engine
            # Calls queued behind a build that ran over budget must not run when it
            # finishes: half a run would be recorded and never ended. Nor may a call
            # its caller stopped waiting for.
            if (
                engine is None
                or self._degraded_reason is not None
                or abandoned.is_set()
            ):
                raise _EngineUnavailable()
            method = getattr(engine, op)
            if not write or self._hold_count:
                # A recorded save holds the lock for this process; the worker
                # serialises the calls within it.
                return method(*args, **kwargs)
            with workspace_lock(self._lock_path, lock_wait, abandoned):
                if abandoned.is_set():
                    raise _EngineUnavailable()
                return method(*args, **kwargs)

        if wait and not self._ready.is_set():
            # Start-up is not charged to this call: wait briefly, then queue writes
            # behind the build and answer reads with the default.
            if not self._ready.wait(self._startup_wait):
                self._note_starting(op)
                if not write or not late:
                    return default
                wait = False
            elif self._degraded_reason is not None:
                return default

        try:
            future = self._worker.submit(body)
        except RuntimeError:
            return default
        if not wait:
            future.add_done_callback(lambda done: self._observe(op, done))
            return default
        limit = self._timeout if timeout is None else timeout
        try:
            return future.result(timeout=limit)
        except FutureTimeout:
            if on_timeout == "continue":  # it finishes in the background
                future.add_done_callback(lambda done: self._observe(op, done))
                return default
            abandoned.set()  # not at the engine yet: it never will be
            future.cancel()
            reason = f"{op} did not finish within {limit:g} s"
            if on_timeout == "degrade":
                self._degrade(reason)
            else:
                self._report(op, reason)
            return default
        except _EngineUnavailable:
            return default
        except Exception as exc:
            self._report(op, f"{type(exc).__name__}: {exc}")
            return default

    def _observe(self, op: str, future: Future) -> None:
        exc = future.exception()
        if exc is not None and not isinstance(exc, _EngineUnavailable):
            self._report(op, f"{type(exc).__name__}: {exc}")

    def _note_starting(self, op: str) -> None:
        with self._state_lock:
            if "__starting__" in self._reported:
                return
            self._reported.add("__starting__")
        logger.info(
            "Catalog: still starting; %s did not wait for it (recorded later if possible)",
            op,
        )

    def _report(self, op: str, detail: str) -> None:
        # Once per operation and kind of problem.
        key = (op, detail[:60])
        with self._state_lock:
            if key in self._reported:
                return
            self._reported.add(key)
        logger.warning(
            "Catalog: %s failed and was skipped; the analysis is not affected (%s)",
            op,
            detail[:300],
        )

    def _degrade(self, reason: str) -> None:
        with self._state_lock:
            latched = self._latch(reason)
        if latched:
            self._warn_degraded(reason)

    def _latch(self, reason: str) -> bool:
        """Record ``reason`` unless already degraded; the caller holds the state lock."""
        if self._degraded_reason is not None:
            return False
        self._degraded_reason = reason
        return True

    def _warn_degraded(self, reason: str) -> None:
        logger.warning(
            "Catalog: %s. Nothing more is recorded in this session; the analysis is not affected.",
            reason,
        )
