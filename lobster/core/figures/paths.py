"""Managed-path classifier and local layout (contract §2).

Rule: a workspace-relative path is *managed* iff it lies under a **subdirectory** of
``plots/`` (any depth, any name). Only regular files directly inside ``plots/`` are legacy.
This is a path rule, not a security classification.
"""

from __future__ import annotations

import os
import stat as _stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NamedTuple, Optional, Tuple, Union

PLOTS_DIR = "plots"
SOURCES_DIR = "_sources"
CODE_DIR = "_code"
EXECUTIONS_DIR = "_executions"
ATTEMPTS_DIR = "_attempts"
STAGING_DIR = ".staging"
RENDITIONS_DIR = "renditions"

REVISION_FILE = "revision.json"
FIGURE_FILE = "figure.json"
CONFIG_FILE = "config.json"

#: Directory names that never hold figures (skipped by listings).
RESERVED_DIRS = frozenset(
    {
        SOURCES_DIR,
        CODE_DIR,
        EXECUTIONS_DIR,
        ATTEMPTS_DIR,
        STAGING_DIR,
        "_history",
        "_attachments",
    }
)
_RESERVED_BY_FOLD = {name.casefold(): name for name in RESERVED_DIRS}


def _parts(rel: Union[str, Path]) -> tuple[str, ...]:
    text = str(rel)
    # Normalise both separators; reject nothing here (callers resolve symlinks/roots first).
    if "\\" in text:
        parts = PureWindowsPath(text).parts
    else:
        parts = PurePosixPath(text).parts
    cleaned: list[str] = []
    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            if cleaned:
                cleaned.pop()
            continue
        cleaned.append(part)
    if cleaned and cleaned[0] in ("/", "\\"):
        cleaned = cleaned[1:]
    return tuple(cleaned)


def is_managed_figure_path(rel: Union[str, Path]) -> bool:
    """True iff ``rel`` (workspace-relative) is inside a subdirectory of ``plots/``."""
    parts = _parts(rel)
    return len(parts) >= 3 and parts[0] == PLOTS_DIR


def is_legacy_plot_file(rel: Union[str, Path]) -> bool:
    """True iff ``rel`` names a file directly inside ``plots/`` (legacy flat layout)."""
    parts = _parts(rel)
    return len(parts) == 2 and parts[0] == PLOTS_DIR


#: Classes of a target against a workspace. ``plots_child`` is a direct child of
#: ``plots/`` that is not reserved: a legacy flat file when it lstats as a regular file.
ANCESTOR = "ancestor"
OUTSIDE = "outside"
WORKSPACE = "workspace"
ROOT = "root"
RESERVED = "reserved"
PLOTS_CHILD = "plots_child"
INSIDE = "inside"
_WRITER_REFUSED = frozenset({ANCESTOR, ROOT, RESERVED, INSIDE})
_UNDER_PLOTS = frozenset({ROOT, RESERVED, PLOTS_CHILD, INSIDE})


class PathClass(NamedTuple):
    """Both classes of one target: ``lexical`` (abspath + normpath, nothing resolved) and
    ``resolved`` (real parents, plus the final component when ``follow_final``)."""

    lexical: str
    resolved: str


def _class_of(ws: str, target: str) -> str:
    """Class of an absolute normalized ``target`` against the absolute normalized ``ws``.

    No I/O. The workspace itself and every ancestor of it are checked FIRST (``..``, the
    parent, ``/``), before any relative-path logic."""
    common = os.path.commonpath([ws, target])
    if common == target:
        return ANCESTOR
    if common != ws:
        return OUTSIDE
    rel = os.path.relpath(target, ws).split(os.sep)
    if rel[0] != PLOTS_DIR:
        return WORKSPACE
    if len(rel) == 1:
        return ROOT
    if rel[1] in RESERVED_DIRS:
        return RESERVED
    return PLOTS_CHILD if len(rel) == 2 else INSIDE


# ------------------------------------------- canonical identity at the managed root
def _lstat_id(path: str) -> Optional[Tuple[int, int]]:
    """``(st_dev, st_ino)`` by lstat; None when absent. Other failures raise."""
    try:
        st = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return (st.st_dev, st.st_ino)


def _swap_answer(parent: str, name: str) -> Optional[bool]:
    """Looking up ``name`` with its ASCII case swapped in ``parent``: True when that finds
    a different entry or none (case-sensitive), False when it finds the same entry, None
    when ``name`` has no ASCII letter to swap."""
    if not name.isascii() or name.swapcase() == name:
        return None
    own = os.lstat(os.path.join(parent, name))
    try:
        other = os.lstat(os.path.join(parent, name.swapcase()))
    except FileNotFoundError:
        return True
    return (other.st_dev, other.st_ino) != (own.st_dev, own.st_ino)


def _case_sensitive(directory: str) -> Optional[bool]:
    """Is ``directory`` PROVEN case-sensitive (True) or case-insensitive (False)?

    Read-only probe, nothing is created: a swapped-case lookup of an existing entry INSIDE
    ``directory`` (first 64 entries). Only a witness in the lookup directory itself counts:
    an ancestor, a sibling or the same device proves nothing, since case behaviour can be
    set per directory (casefold attributes). None when nothing inside can tell (empty, no
    ASCII-lettered name in the first 64 entries, a lookup failure); callers then reserve.
    """
    try:
        with os.scandir(directory) as entries:
            for index, entry in enumerate(entries):
                if index >= 64:
                    break
                answer = _swap_answer(directory, entry.name)
                if answer is not None:
                    return answer
    except OSError:
        return None
    return None


def _same_entry(parent: str, name: str, canonical: str) -> bool:
    """Does ``name`` in directory ``parent`` address the entry ``canonical``?

    Exact -> True; a name that does not casefold to ``canonical`` -> False (no I/O).
    Otherwise by filesystem identity (lstat): both present -> same (dev, inode); one present
    -> False (a case-insensitive lookup would find both); neither present -> True unless
    ``parent`` is PROVEN case-sensitive (creating ``name`` could create the entry later
    used as ``canonical``). A lookup failure answers True, so writers refuse.
    """
    if name == canonical:
        return True
    if name.casefold() != canonical.casefold():
        return False
    try:
        mine = _lstat_id(os.path.join(parent, name))
        theirs = _lstat_id(os.path.join(parent, canonical))
    except OSError:
        return True
    if mine is not None or theirs is not None:
        return mine == theirs
    return _case_sensitive(parent) is not True


def _canonical(ws: str, target: str) -> str:
    """``target`` (absolute, normalized) with its managed-root names spelled canonically.

    The first component below ``ws`` that addresses ``plots/`` (:func:`_same_entry`) is
    spelled ``plots``, and a component directly below it that addresses a reserved
    directory is spelled with the reserved name. Every other path is returned as given.
    """
    if target == ws or os.path.commonpath([ws, target]) != ws:
        return target
    rel = os.path.relpath(target, ws).split(os.sep)
    if not _same_entry(ws, rel[0], PLOTS_DIR):
        return target
    rel[0] = PLOTS_DIR
    if len(rel) > 1:
        reserved = _RESERVED_BY_FOLD.get(rel[1].casefold())
        if reserved is not None and _same_entry(
            os.path.join(ws, PLOTS_DIR), rel[1], reserved
        ):
            rel[1] = reserved
    return os.path.join(ws, *rel)


def _rebase(ws_real: str, target: str) -> str:
    """``target`` re-expressed under ``ws_real`` when it, or an existing ancestor of it, IS
    the workspace directory by identity (lstat) under another spelling; else unchanged.
    """
    if os.path.commonpath([ws_real, target]) == ws_real:
        return target
    try:
        ws_id = _lstat_id(ws_real)
    except OSError:
        return target
    cur, rest = target, []
    while ws_id is not None:
        try:
            same = _lstat_id(cur) == ws_id
        except OSError:
            same = False
        if same:
            return os.path.join(ws_real, *reversed(rest))
        parent, name = os.path.split(cur)
        if not name or parent == cur:
            break
        rest.append(name)
        cur = parent
    return target


def classify(
    workspace: Union[str, Path], target: Union[str, Path], *, follow_final: bool
) -> PathClass:
    """Classify ``target`` lexically AND resolved.

    Lexical: ``abspath`` + ``normpath`` of both paths; no symlink is resolved. Resolved: the
    real workspace, and the target's real parent plus its final name, or the target's
    realpath when ``follow_final`` (writers: a write follows the final symlink). A path rule,
    not a security classification; the caller still lstats at the write/delete boundary.

    Both halves use the canonical identity of the managed root: a spelling the filesystem
    treats as ``plots/`` (or as a reserved directory below it), such as ``PLOTS`` on a
    case-insensitive filesystem, classifies as that directory (:func:`_same_entry`), and
    the resolved half is re-based onto the real workspace when an ancestor IS the
    workspace under another spelling (:func:`_rebase`).
    """
    ws_lex = os.path.normpath(os.path.abspath(workspace))
    t_lex = os.path.normpath(os.path.abspath(target))
    ws_real = os.path.realpath(workspace)
    # The resolved half starts from the ORIGINAL spelling: normpath would collapse
    # ``symlink/..`` lexically before the symlink is resolved.
    original = os.fspath(target)
    if not os.path.isabs(original):
        original = os.path.join(os.getcwd(), original)
    name = os.path.basename(original)
    if follow_final or name in ("", ".", ".."):  # no final entry of its own to keep
        t_res = os.path.realpath(original)
    else:
        t_res = os.path.join(os.path.realpath(os.path.dirname(original)), name)
    return PathClass(
        _class_of(ws_lex, _canonical(ws_lex, t_lex)),
        _class_of(ws_real, _canonical(ws_real, _rebase(ws_real, t_res))),
    )


def _lstat_mode(path: str) -> Optional[int]:
    """lstat mode; None when absent. Any other failure raises (callers refuse)."""
    try:
        return os.lstat(path).st_mode
    except FileNotFoundError:
        return None


def _writer_refuses(workspace: Union[str, Path], target: Union[str, Path]) -> bool:
    """Does a generic WRITER refuse ``target``? (follow_final=True)

    Refused when EITHER class is ``ancestor`` / ``root`` / ``reserved`` / ``inside``. A
    ``plots_child`` is allowed only as a legacy flat file reached WITHOUT an alias: both
    classes ``plots_child``, ``plots/`` itself a real directory by lstat (checked first),
    and the final entry absent or a regular file by lstat (a final symlink there refuses,
    dangling included). Outside and workspace paths are unchanged unless they RESOLVE into
    ``plots/`` (an alias). An unexpected lstat failure under ``plots/`` refuses.
    """
    lexical, resolved = classify(workspace, target, follow_final=True)
    if lexical in _WRITER_REFUSED or resolved in _WRITER_REFUSED:
        return True
    if PLOTS_CHILD not in (lexical, resolved):
        return False  # outside / workspace on both sides: unchanged
    if lexical != resolved:
        return True  # reached through an alias, or a legacy name pointing elsewhere
    ws_lex = os.path.normpath(os.path.abspath(workspace))
    t_lex = os.path.normpath(os.path.abspath(target))
    try:
        root_mode = _lstat_mode(os.path.join(ws_lex, PLOTS_DIR))
        final_mode = _lstat_mode(t_lex)
    except OSError:
        return True
    if root_mode is not None and not _stat.S_ISDIR(root_mode):
        return True  # plots/ is a symlink or not a directory
    return final_mode is not None and not _stat.S_ISREG(final_mode)


def _shown_path(workspace: Union[str, Path], target: Union[str, Path]) -> str:
    """The workspace-relative spelling shown in a refusal (never an outside path)."""
    for ws, t in (
        (os.path.normpath(os.path.abspath(workspace)), os.path.abspath(target)),
        (os.path.realpath(workspace), os.path.realpath(target)),
    ):
        t = os.path.normpath(t)
        if os.path.commonpath([ws, t]) == ws:
            return os.path.relpath(t, ws)
    return "."


def assert_not_managed(workspace: Path, target: Path) -> None:
    """Raise :class:`ManagedPathError` if a generic writer must not write ``target``.

    For generic writers (filesystem tools, workspace writes, imports). Classifies the
    lexical AND the resolved target (:func:`classify`, ``follow_final=True``); see
    :func:`_writer_refuses` for the rule. Pass the target as the caller spelled it (not
    pre-resolved), or the lexical half cannot see a symlink.
    """
    from lobster.core.figures.errors import ManagedPathError

    if _writer_refuses(workspace, target):
        rel = _shown_path(workspace, target)
        raise ManagedPathError(
            f"'{rel}' is inside the managed figure namespace; figures are saved only through the trusted save step",
            details={"path": rel},
        )
