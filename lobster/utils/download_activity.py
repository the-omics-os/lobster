"""The file download in progress, for a host that polls ``lobster_status``, and the
stop signal of the run that started it.

One run is active at a time, so one slot of each is enough. The downloader reports
progress into it and checks ``stopped()``; the MCP server reads the progress and sets
the stop.
"""

import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()
_current: Optional[Dict[str, Any]] = None
_stop: Optional[threading.Event] = None


def report(file: str, done: float, total: float, bytes_per_s: float) -> None:
    """Record a file's progress (sizes in bytes; rate in bytes per second)."""
    global _current
    state: Dict[str, Any] = {
        "file": file,
        "mb_done": round(done / 1e6, 1),
        "mb_total": round(total / 1e6, 1),
        "kb_per_s": round(bytes_per_s / 1e3),
    }
    if total > done and bytes_per_s > 0:
        state["eta_min"] = round((total - done) / bytes_per_s / 60)
    with _lock:
        _current = state


def clear() -> None:
    global _current
    with _lock:
        _current = None


def snapshot() -> Optional[Dict[str, Any]]:
    with _lock:
        return dict(_current) if _current else None


def set_stop(event: Optional[threading.Event]) -> None:
    global _stop
    _stop = event


def stopped() -> bool:
    """True once the run that is downloading was cancelled."""
    return _stop is not None and _stop.is_set()
