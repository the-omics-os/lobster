"""The claim of one Lobster process on an ``output_dir``'s session folder.

Two servers (two hosts, or a host that restarted while the old process was
still stopping) must not write into one session folder: the later
``result.json`` would silently replace the earlier. The claim is a lock file in
the hidden workspace, created with ``O_CREAT | O_EXCL``, that names the process
(its pid). It is released when the session is closed or the process exits; a
lock whose process is gone is taken over.

Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Set, Tuple

from lobster.cli_internal import mcp_turns

WORKSPACE_DIR = ".lobster_workspace"  # the engine's own default name
LOCK_NAME = "mcp.lock"
# A lock file that exists but holds no pid yet is being written; older than
# this it was left by a process that died between creating and writing it.
WRITE_GRACE_S = 5.0
# A takeover guard older than this was left by a process that died during it.
GUARD_STALE_S = 30.0

# The session folders this process holds, by file identity: two spellings of
# one folder (letter case on macOS, a link) are one claim.
_held: Set[Tuple[int, int]] = set()
# Taking, judging and releasing a claim is one step for the threads of this
# process: between a release's check and its unlink, another thread must not
# judge the same lock stale.
_held_lock = threading.RLock()


class InUse(Exception):
    """The folder is held; ``pid`` is the holder's when it is known."""

    def __init__(self, pid: Optional[int] = None):
        super().__init__(f"held by process {pid}" if pid else "held")
        self.pid = pid


class FolderClaim:
    """A held lock. ``kept`` is set by whoever keeps the session the claim
    belongs to (it then releases the claim); a run that ends without a session
    releases it."""

    def __init__(self, lock: Path, key: Tuple[int, int]):
        self.lock = lock
        self.key = key
        self.kept = False
        self.made = False  # claim_new made the session folder in this call
        self.moved_to: Optional[Path] = None  # where an earlier one was moved

    def discard(self) -> None:
        """Remove the session folder this claim made, for a run that never
        started, and release the claim. The only removal of a folder by the
        server: one that existed before the call is never removed."""
        with _held_lock:
            if self.made and self.key in _held:
                shutil.rmtree(self.lock.parent, ignore_errors=True)
            self.release()

    def release(self) -> None:
        """Remove the lock file if it is still this process's; safe twice."""
        with _held_lock:
            if self.key not in _held:
                return
            _held.discard(self.key)
            if _read_pid(self.lock) == os.getpid():
                self.lock.unlink(missing_ok=True)


def claim_new(
    output_dir: Path, leftover: Callable[[], bool] = lambda: False
) -> FolderClaim:
    """Make *output_dir* and its session folder, mark it as the server's (the
    turn record, before anything else) and claim it. ``mkdir`` of the session
    folder is the atomic step: of two callers one gets ``InUse``. Other
    ``OSError`` (a folder that cannot be made) is raised as it is.

    A session folder that already exists is never emptied. When it carries the
    server's record and *leftover* says it is only what a run that never
    finished left behind, it is moved aside whole (``moved_to``) and a fresh
    one is made; otherwise, or when the move fails, ``InUse``."""
    workspace = output_dir / WORKSPACE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    moved_to = None
    # One step for the threads of this process: none may move a folder another
    # has just made and not yet locked.
    with _held_lock:
        try:
            workspace.mkdir()
        except FileExistsError:
            moved_to = _move_aside(workspace, leftover)
            try:
                workspace.mkdir()
            except FileExistsError:  # another process made one in between
                raise InUse(_holder(workspace / LOCK_NAME)) from None
        try:
            mcp_turns.mark(workspace)
            claim = _take(workspace / LOCK_NAME)
        except InUse:  # a caller that took the folder over first owns it now
            raise
        except BaseException:
            (workspace / mcp_turns.STATE_NAME).unlink(missing_ok=True)
            workspace.rmdir()
            raise
    claim.made, claim.moved_to = True, moved_to
    return claim


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _move_aside(workspace: Path, leftover: Callable[[], bool]) -> Path:
    """Move an existing session folder to a new sibling name, whole, under its
    claim, and return where it went. The lock file is the atomic step, so of
    two callers one gets ``InUse``; *leftover* is asked again once the claim is
    held. Only a folder the server made (it holds the turn record) is moved; a
    link is not, and neither is a workspace of ``lobster chat`` or ``lobster
    query``. Nothing in the folder is changed: what the user may have added
    since moves with it."""
    # Asked before the lock is touched: a caller that will not move the folder
    # must not hold up the one that made it, which is about to write its lock.
    try:
        is_dir = stat.S_ISDIR(os.lstat(workspace).st_mode)  # a link is not
    except OSError:
        is_dir = False
    if not is_dir or not mcp_turns.is_marked(workspace) or not leftover():
        raise InUse(_holder(workspace / LOCK_NAME))
    # shortcut: a run that is between making the folder and writing its lock
    # looks like a leftover for that instant; upgrade if two servers share a host.
    try:
        claim = _take(workspace / LOCK_NAME)
    except FileNotFoundError:  # another process moved it first
        raise InUse() from None
    with _held_lock:
        try:
            if not mcp_turns.is_marked(workspace) or not leftover():
                raise InUse()
            base = f"{WORKSPACE_DIR}.unfinished-{_stamp()}"
            target, number = workspace.with_name(base), 1
            while os.path.lexists(target):
                number += 1
                target = workspace.with_name(f"{base}-{number}")
            try:
                os.rename(workspace, target)
            except OSError:
                raise InUse() from None
        except BaseException:
            claim.release()
            raise
        # The claim's lock moved with the folder: it is this process's own file.
        _held.discard(claim.key)
        (target / LOCK_NAME).unlink(missing_ok=True)
    return target


def claim_existing(output_dir: Path) -> FolderClaim:
    """Claim the session folder an earlier run left (``FileNotFoundError`` when
    there is none); ``InUse`` while another process holds it."""
    return _take(output_dir / WORKSPACE_DIR / LOCK_NAME)


def holder(output_dir: Path) -> Optional[int]:
    """The pid of another live process holding *output_dir*, else None."""
    return _holder(output_dir / WORKSPACE_DIR / LOCK_NAME)


def _key(lock: Path) -> Tuple[int, int]:
    """The identity of the session folder *lock* is in (``OSError`` when it is
    not there)."""
    folder = os.stat(lock.parent)
    return folder.st_dev, folder.st_ino


def _take(lock: Path) -> FolderClaim:
    with _held_lock:
        for _ in range(3):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                pid = _read_pid(lock)
                if _live(lock, pid):
                    raise InUse(pid) from None
                _remove_stale(lock, pid)
                continue
            try:
                key = _key(lock)
            except BaseException:
                os.close(fd)
                lock.unlink(missing_ok=True)
                raise
            # Held before the pid is written: whoever reads the pid in between
            # must find the claim live, not a stale lock of an earlier process.
            _held.add(key)
            with os.fdopen(fd, "w") as handle:
                handle.write(f"{os.getpid()}\n")
            return FolderClaim(lock, key)
        raise InUse()  # lost the race for the same stale lock three times


def _holder(lock: Path) -> Optional[int]:
    pid = _read_pid(lock)
    return pid if pid and pid != os.getpid() and _live(lock, pid) else None


def _read_pid(lock: Path) -> Optional[int]:
    try:
        return int(lock.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _live(lock: Path, pid: Optional[int]) -> bool:
    if pid is None:
        try:
            return time.time() - lock.stat().st_mtime < WRITE_GRACE_S
        except OSError:
            return False
    if pid == os.getpid():
        # Only a claim made in this process counts; the same number in a lock
        # left by an earlier process is stale.
        with _held_lock:
            try:
                return _key(lock) in _held
            except OSError:
                return False
    return _alive(pid)


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        # shortcut: os.kill(pid, 0) would end the process on Windows and there
        # is no probe here; a lock left by a crash is removed by hand.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists, owned by someone else
        return True
    except OSError:
        return False
    return True


def _remove_stale(lock: Path, judged: Optional[int]) -> None:
    """Remove the lock judged stale, under a guard file only one process can hold
    (a lock another process has just made in its place is then never removed).
    A caller that does not get the guard leaves the lock; its next try sees the
    new holder."""
    guard = lock.with_name(lock.name + ".takeover")
    try:
        os.close(os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
    except FileExistsError:
        try:
            if time.time() - guard.stat().st_mtime > GUARD_STALE_S:
                guard.unlink(missing_ok=True)  # its holder died
        except OSError:
            pass
        return
    try:
        if _read_pid(lock) == judged and not _live(lock, judged):
            lock.unlink(missing_ok=True)
    finally:
        guard.unlink(missing_ok=True)
