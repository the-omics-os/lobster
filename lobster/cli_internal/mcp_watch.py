"""Thread stacks on stderr when the MCP server cannot answer a ``lobster_wait``.

A wait that gets no reply leaves the host nothing to go on. Every thread's
stack then goes to stderr (the host's MCP log), never to stdout, which carries
the protocol. A native call that holds the GIL stops every Python thread, so
faulthandler's own watchdog thread is the backstop: it writes without Python.
"""

from __future__ import annotations

import faulthandler
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from typing import IO, Dict, Iterator, Optional

LATE_LINE = "lobster_wait did not reply in time; thread stacks follow"
# A wait counts as late this long after its own timeout_s.
GRACE_S = 30.0
# faulthandler's watchdog writes this long after a late wait that Python could
# not report: no Python thread ran, so the plain line could not be written.
BACKSTOP_S = 10.0


def _stderr() -> Optional[IO[str]]:
    """The process's stderr when it is a real file (faulthandler writes to a
    file descriptor), else None: then nothing is written."""
    stream = sys.__stderr__
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    return stream


def stacks_on_signal() -> None:
    """``kill -USR1 <pid>`` writes every thread's stack to stderr. Windows has
    no SIGUSR1: nothing is registered there."""
    stream = _stderr()
    if stream is not None and hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=stream, all_threads=True)


class WaitWatch:
    """Watches the waits that have not replied yet.

    faulthandler has one watchdog timer per process, so it is set for the
    earliest open wait and cancelled only when no wait is open: a wait that
    ends never disarms another.
    """

    def __init__(self, stream: Optional[IO[str]] = None) -> None:
        self._stream = stream if stream is not None else _stderr()
        self._lock = threading.Lock()
        self._due: Dict[object, float] = {}  # open waits not yet reported

    @contextmanager
    def watching(self, after_s: float) -> Iterator[None]:
        """Report the stacks if the block has not ended *after_s* from now."""
        if self._stream is None:
            yield
            return
        token = object()
        timer = threading.Timer(after_s, self._late, (token,))
        timer.daemon = True
        with self._lock:
            self._due[token] = time.monotonic() + after_s
            self._arm()
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
            with self._lock:
                self._due.pop(token, None)
                self._arm()

    def _late(self, token: object) -> None:
        with self._lock:
            if self._due.pop(token, None) is None:
                return
            self._arm()  # reported here: the backstop stands down for it
        self._stream.write(LATE_LINE + "\n" + _stacks())
        self._stream.flush()

    def _arm(self) -> None:
        if not self._due:
            faulthandler.cancel_dump_traceback_later()
            return
        left = max(0.0, min(self._due.values()) - time.monotonic())
        # shortcut: the backstop writes with the GIL held and cancelling it waits
        # for that write, so a host that never reads a full stderr pipe would
        # stall the server here; such a host already stalls every log line.
        faulthandler.dump_traceback_later(left + BACKSTOP_S, file=self._stream)


def _stacks() -> str:
    """Every thread's stack, most recent call last, the way faulthandler names
    threads. Written by Python, so the write lets other threads run."""
    names = {t.ident: t.name for t in threading.enumerate()}
    parts = []
    for ident, frame in sys._current_frames().items():
        parts.append(f"Thread 0x{ident:016x} ({names.get(ident, '?')}):\n")
        parts.extend(traceback.format_stack(frame))
    return "".join(parts)
