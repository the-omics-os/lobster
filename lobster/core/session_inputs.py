"""The files a session loaded or imported, so an export never overwrites one.

Each data manager is tracked here (weakly) from the moment it is made. A file it
loads or imports is noted by its file identity (device and inode), for this process,
so another spelling of it (``..``, a link, a hard link, another case) is the same
file. A load also keeps the file's size and SHA-256 on the session record
(``loaded_file``), which a restored session reads back: so a later process still
knows the file, without its path.

:func:`is_input` answers for an existing target, for every tracked session whose
workspace lies in the folder the writer works in. A folder that was imported counts
as an input with every file under it (a hard link to one of them is not seen).
"""

import hashlib
import os
import threading
import weakref
from pathlib import Path
from typing import Any, Iterator, Set, Tuple, Union

_lock = threading.Lock()
#: data manager -> (device, inode) of the files and folders it read (this process only).
_noted: "weakref.WeakKeyDictionary[Any, Set[Tuple[int, int]]]" = (
    weakref.WeakKeyDictionary()
)


def _real(path: Union[str, Path]) -> str:
    return os.path.realpath(os.fspath(path))


def _identity(path: Union[str, Path]) -> Tuple[int, int]:
    st = os.stat(path)
    return st.st_dev, st.st_ino


def track(data_manager: Any) -> None:
    """Start answering for ``data_manager`` (its recorded loads included)."""
    with _lock:
        _noted.setdefault(data_manager, set())


def note_input(data_manager: Any, source: Any) -> None:
    """Remember that ``data_manager`` read the file or folder at ``source``."""
    if not isinstance(source, (str, Path)):
        return
    try:
        identity = _identity(source)
    except (OSError, ValueError):  # not a file on this disk (a URL, an accession)
        return
    with _lock:
        _noted.setdefault(data_manager, set()).add(identity)


def _sessions_in(root: str) -> Iterator[Tuple[Any, Set[Tuple[int, int]]]]:
    prefix = root.rstrip(os.sep) + os.sep
    with _lock:
        held = [(dm, set(paths)) for dm, paths in _noted.items()]
    for dm, paths in held:
        workspace = getattr(dm, "workspace_path", None)
        if workspace is None:
            continue
        ws = _real(workspace)
        if ws == root or ws.startswith(prefix):
            yield dm, paths


def _recorded_loads(data_manager: Any) -> Iterator[Tuple[int, str]]:
    """(size, sha256) of every file the session record says a load read."""
    provenance = getattr(data_manager, "provenance", None)
    for entity in list(getattr(provenance, "entities", {}).values()):
        loaded = ((entity or {}).get("metadata") or {}).get("loaded_file")
        if isinstance(loaded, dict) and isinstance(loaded.get("sha256"), str):
            try:
                yield int(loaded.get("size")), loaded["sha256"]
            except (TypeError, ValueError):
                continue


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_input(root: Union[str, Path], target: Union[str, Path]) -> bool:
    """True when ``target`` is an existing file a session in ``root`` loaded or imported.

    By file identity for this process (the file, or a folder above it); by size, then
    SHA-256 (read only on a size match) against the session record. Errors
    propagate: the caller refuses on them.
    """
    real = _real(target)
    if not os.path.isfile(real):
        return False
    sessions = list(_sessions_in(_real(root)))
    noted = set().union(*(ids for _, ids in sessions))
    if noted:
        here = Path(real)
        if any(_identity(p) in noted for p in (here, *here.parents)):
            return True
    size = os.stat(real).st_size
    digests = {sha for dm, _ in sessions for n, sha in _recorded_loads(dm) if n == size}
    return bool(digests) and _sha256(real) in digests
