"""Keep tool exports inside the session workspace (export write policy, 1.2.1).

Model-facing export tools (DE tables, annotation mappings, ML matrices, pathway
CSVs) build a target from a model-supplied path or a modality name. A target is
allowed only when it lies inside the session workspace, or inside a folder the
person listed in ``LOBSTER_EXPORT_ROOTS`` before Lobster started. The setting is
read once, when this module is loaded. No caller, tool argument or model can widen
it, and the functions here take no roots argument.

A target is checked where the operating system would write it: the joined
spelling, with each symlink and ``..`` resolved in the order the OS follows them,
must be under the resolved workspace or a resolved root. So must every missing
folder of the spelling, because a recursive creator makes each one where its own
resolved spelling points (a spelling may leave the workspace, create a folder, and
come back in). A link inside the workspace that leads out is refused; a spelling
through an alias of the workspace (for example ``/tmp`` for ``/private/tmp``) is
accepted. A link loop, a dangling link before the last part, or any other error
while classifying a target refuses the write (fail closed).

A refusal carries one fixed reason. It names the setting, never the path, and is
raised without a displayed chain. Limits: a check-then-write race with another
process of the same user (swapping a folder for a link) is not closed, and a
case-variant spelling of the workspace on a case-insensitive file system is refused
rather than matched.

The same module holds the modality-name check: a modality name becomes part of
file names, so it must be one plain name.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

#: The environment variable a person sets, before start, to allow extra folders.
EXPORT_ROOTS_ENV = "LOBSTER_EXPORT_ROOTS"

#: The fixed reason carried by every workspace refusal (no path is echoed).
WORKSPACE_EXPORT_REASON = (
    "this export path is outside the session workspace. Use a file name under "
    "exports/ (for example de_results.csv), or list the folder in "
    "LOBSTER_EXPORT_ROOTS before starting Lobster."
)

#: The fixed user-facing refusal returned by export tools.
WORKSPACE_EXPORT_MESSAGE = "Error: " + WORKSPACE_EXPORT_REASON

#: Longest modality name accepted.
MAX_MODALITY_NAME_LENGTH = 200

#: The fixed reason carried by every modality-name refusal (no name is echoed).
MODALITY_NAME_REASON = (
    "a modality name must be one plain name: no '/' or '\\', no control characters, "
    "not '.' or '..', at most 200 characters."
)

PathLike = Union[str, "os.PathLike[str]"]


class WorkspaceExportError(ValueError):
    """An export target outside the workspace and the listed roots (fixed reason)."""


class InvalidModalityNameError(ValueError):
    """A modality name that is not one plain name (fixed reason)."""


def parse_export_roots(text: Optional[str]) -> Tuple[Path, ...]:
    """The usable folders in a ``LOBSTER_EXPORT_ROOTS`` value, in order.

    Entries are separated by ``os.pathsep``; ``~`` is expanded. An entry is kept
    only when it is an absolute path to an existing directory; empty, relative,
    missing, non-directory and unreadable entries are dropped (counted in a
    warning, without their text).
    """
    if not isinstance(text, str) or not text:
        return ()
    roots: List[Path] = []
    dropped = 0
    for entry in text.split(os.pathsep):
        if not entry:
            continue
        try:
            candidate = Path(entry).expanduser()
            usable = candidate.is_absolute() and candidate.is_dir()
        except (OSError, ValueError, RuntimeError):
            usable = False
        if usable:
            if candidate not in roots:
                roots.append(candidate)
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "%s: %d entr%s ignored (each must be an existing absolute folder)",
            EXPORT_ROOTS_ENV,
            dropped,
            "y" if dropped == 1 else "ies",
        )
    return tuple(roots)


def _read_export_roots() -> Tuple[Path, ...]:
    return parse_export_roots(os.environ.get(EXPORT_ROOTS_ENV))


#: Read once at load. Later changes to the environment have no effect.
_EXPORT_ROOTS: Tuple[Path, ...] = _read_export_roots()


def export_roots() -> Tuple[Path, ...]:
    """The extra export folders the person listed before start (may be empty)."""
    return _EXPORT_ROOTS


def _under(child: str, parent: str) -> bool:
    return os.path.commonpath([child, parent]) == parent


def _bases(workspace: PathLike) -> List[str]:
    """The resolved workspace, then each resolved root."""
    spelled = os.fspath(workspace)
    if not isinstance(spelled, str) or not os.path.isabs(spelled):
        raise ValueError("workspace must be an absolute path")
    return [
        os.path.realpath(base) for base in (spelled, *map(os.fspath, _EXPORT_ROOTS))
    ]


def _prefixes(joined: str) -> List[Tuple[str, str]]:
    """Each lexical prefix of ``joined`` with its last part, from the root down."""
    found = []
    path = joined
    while True:
        head, tail = os.path.split(path)
        if tail:
            found.append((path, tail))
        if head == path or not head:
            break
        path = head
    return found[::-1]


def _inside(bases: List[str], workspace: str, target: PathLike) -> bool:
    spelled = os.fspath(target)
    if not isinstance(spelled, str):
        raise TypeError("target must be a text path")
    joined = os.path.join(workspace, spelled)  # an absolute target replaces the base
    resolved = os.path.realpath(joined)  # each link, then '..', as the OS follows them
    if not any(_under(resolved, base) for base in bases):
        return False
    prefixes = _prefixes(joined)
    for index, (prefix, part) in enumerate(prefixes):
        if part in (".", ".."):
            continue  # never created
        if os.path.lexists(prefix):
            try:
                os.stat(prefix)  # a loop or an unreadable link raises
            except FileNotFoundError:
                if index != len(prefixes) - 1:
                    raise  # a dangling link before the last part
            continue
        # A missing folder (or the file) is created where its spelling resolves.
        created = os.path.realpath(prefix)
        if not any(_under(created, base) for base in bases):
            return False
    return True


def outside_workspace(workspace: PathLike, *targets: PathLike) -> bool:
    """True when ANY of ``targets`` is outside the workspace and the listed roots.

    Pass each target as the export spelled it (joined to its base directory, not
    resolved). A relative target is taken against the workspace. Any error while
    classifying (a bad type, a NUL byte, a resolution failure, a workspace that is
    not an absolute path) counts as outside: the write is refused, not allowed.
    """
    try:
        bases = _bases(workspace)
        base = os.fspath(workspace)
        return not all(_inside(bases, base, target) for target in targets)
    except Exception:  # noqa: BLE001 - fail closed on any classification error
        return True


def assert_inside_workspace(workspace: PathLike, target: PathLike) -> None:
    """Raise :class:`WorkspaceExportError` unless ``target`` may be written."""
    if outside_workspace(workspace, target):
        raise WorkspaceExportError(WORKSPACE_EXPORT_REASON) from None


def plain_name_part(label: str) -> str:
    """``label`` (a data value such as a group label) as part of a plain modality
    name: ``%``, ``/``, ``\\`` and control characters are percent-encoded, so two
    different labels never give the same part. Everything else is kept."""
    text = str(label)
    return "".join(
        f"%{ord(ch):02X}" if ch in "%/\\" or ord(ch) < 0x20 or ord(ch) == 0x7F else ch
        for ch in text
    )


def check_modality_name(name: str) -> str:
    """Return ``name`` when it is one plain name, else raise
    :class:`InvalidModalityNameError`.

    Refused: a non-text value, empty, longer than 200 characters, ``.`` or ``..``,
    or containing ``/``, ``\\``, NUL or another control character. Spaces, dots
    inside the name and non-ASCII letters are allowed.
    """
    if (
        not isinstance(name, str)
        or not name
        or len(name) > MAX_MODALITY_NAME_LENGTH
        or name in (".", "..")
        or any(ch in "/\\" or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in name)
    ):
        raise InvalidModalityNameError(MODALITY_NAME_REASON) from None
    return name
