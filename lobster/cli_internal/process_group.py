"""Start and signal the Go TUI child in its own process group.

Stdlib only: ``go_tui_launcher`` must stay free of heavy imports.

POSIX puts the child in a new session (``os.setsid``) and signals the whole
group with ``os.killpg``. Windows has neither; it starts the child with
``CREATE_NEW_PROCESS_GROUP`` and sends ``CTRL_BREAK_EVENT`` to that group,
with ``proc.kill()`` (TerminateProcess) as the hard stop.
"""

from __future__ import annotations

import os
import signal
import subprocess  # nosec B404 # only types and constants are used here
import sys
from typing import Any, Dict


def _is_windows() -> bool:
    return sys.platform == "win32"


def new_group_popen_kwargs() -> Dict[str, Any]:
    """Popen kwargs that start the child in a new process group."""
    if _is_windows():
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"preexec_fn": os.setsid}


def signal_group(proc: subprocess.Popen, *, force: bool = False) -> None:
    """Ask the child's process group to stop; ``force`` kills it.

    Raises like the underlying call so callers keep their fallback to
    ``proc.terminate()`` / ``proc.kill()``.
    """
    if _is_windows():
        if force:
            proc.kill()
        else:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL if force else signal.SIGTERM)
