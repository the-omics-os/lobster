"""The server's own record of how each turn of a session ended.

One small JSON file in the hidden workspace. It is written first when the
server makes the folder, and it is what marks the folder as the server's: a
``.lobster_workspace`` without it (one of ``lobster chat``, say) is never
moved. It is written again whenever a turn ends in any state: the turn number, the outcome (``completed``, ``cancelled`` or
``failed``) and the names of the modalities the session held at that moment.
Two things read it: a folder is moved aside for a new run only when it shows no
completed turn, and a restored session compares it with what came back.

It is the server's, not the engine's: nothing the model or the host writes
reaches it, and a file that cannot be read is reported, never guessed at.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

STATE_NAME = "mcp_turns.json"
VERSION = 1
# A session of more turns than this loses the oldest entries; the result files
# still show the finished ones.
MAX_TURNS_KEPT = 200
MAX_NAMES_KEPT = 1000
OUTCOMES = ("completed", "cancelled", "failed")

_write_lock = threading.Lock()

# Why a session may not be continued. The record keeps only the code; the host
# reads the server's own sentence for it, never text from the file (a folder can
# come from someone else).
FOREIGN_DATA = "foreign_data"
BLOCKED_REASONS = {
    FOREIGN_DATA: (
        "an earlier turn read another file and that data could not be removed "
        "from the session, so a further turn could report it as the result"
    ),
}
BLOCKED_OTHER = "the session folder marks this session as not continuable"


def load(workspace: Path) -> Optional[Dict[str, Any]]:
    """The record in *workspace*; None when there is none. Raises ``ValueError``
    when the file is there and cannot be read as a record (a link, text that is
    not JSON, a wrong shape)."""
    path = workspace / STATE_NAME
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:  # a link (ELOOP), a permission or a device problem
        raise ValueError(f"{STATE_NAME} cannot be opened ({exc.strerror})") from None
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{STATE_NAME} cannot be read ({exc})") from None
    turns = state.get("turns") if isinstance(state, dict) else None
    if not isinstance(turns, list) or state.get("v") != VERSION:
        raise ValueError(f"{STATE_NAME} is not a turn record")
    for entry in turns:
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("turn"), int)
            and entry.get("outcome") in OUTCOMES
            and isinstance(entry.get("modalities"), list)
        ):
            raise ValueError(f"{STATE_NAME} holds an entry that is not a turn")
    if "blocked" in state and not isinstance(state["blocked"], str):
        raise ValueError(f"{STATE_NAME} is not a turn record")
    return state


def mark(workspace: Path) -> None:
    """Write an empty record into *workspace*, the folder the server has just
    made (or cleared); ``OSError`` when it cannot be written."""
    _write(workspace, {"v": VERSION, "turns": []})


def is_marked(workspace: Path) -> bool:
    """True when *workspace* holds the server's record as a file of its own (a
    link at that name does not count)."""
    path = workspace / STATE_NAME
    return path.is_file() and not path.is_symlink()


def shows_completed_turn(state: Dict[str, Any]) -> bool:
    return any(entry["outcome"] == "completed" for entry in state["turns"])


def last_modalities(state: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """The modality names at the end of the newest turn, any outcome; None
    when the record has no turn."""
    if not state or not state["turns"]:
        return None
    return list(state["turns"][-1]["modalities"])


def blocked_reason(workspace: Path) -> Optional[str]:
    """Why the session in *workspace* may not be continued, if it was blocked:
    the server's sentence for the recorded code."""
    try:
        state = load(workspace)
    except ValueError:
        return None
    code = state.get("blocked") if state else None
    if not code:
        return None
    return BLOCKED_REASONS.get(code, BLOCKED_OTHER)


def record(workspace: Path, turn: int, outcome: str, modalities: Sequence[str]) -> bool:
    """Add the end of *turn* to the record in *workspace*; False (and a line on
    stderr) when it could not be written. A workspace that is not a folder of
    ours (a link, or gone) is left alone."""
    return _update(
        workspace,
        lambda state: state["turns"].append(
            {
                "turn": turn,
                "outcome": outcome,
                "modalities": list(modalities)[:MAX_NAMES_KEPT],
            }
        ),
    )


def block(workspace: Path, code: str) -> bool:
    """Mark the session in *workspace* as not continuable, for the reason
    *code* (a key of ``BLOCKED_REASONS``)."""
    return _update(workspace, lambda state: state.update(blocked=code))


def _update(workspace: Path, change: Any) -> bool:
    if workspace.is_symlink() or not workspace.is_dir():
        return False
    with _write_lock:
        try:
            state = load(workspace) or {"v": VERSION, "turns": []}
        except ValueError:
            state = {"v": VERSION, "turns": []}  # an unreadable record is replaced
        change(state)
        state["turns"] = state["turns"][-MAX_TURNS_KEPT:]
        try:
            _write(workspace, state)
        except OSError as exc:
            print(f"lobster mcp: {STATE_NAME} not written: {exc!r}", file=sys.stderr)
            return False
    return True


def _write(workspace: Path, state: Dict[str, Any]) -> None:
    # A new file renamed over the old one: a reader never sees half a record, and
    # a link at the final name is replaced, never written through.
    fd, tmp = tempfile.mkstemp(dir=workspace, prefix=STATE_NAME + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(tmp, workspace / STATE_NAME)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
