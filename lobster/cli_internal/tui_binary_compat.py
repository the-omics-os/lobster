"""Version gate for lobster-tui binaries found in the user cache or on PATH.

Stdlib only: the chat fast path imports this before anything heavy loads.
"""

from __future__ import annotations

import logging
import re
import subprocess  # nosec B404 # runs a local binary with an argv list, no shell
import sys
from typing import Dict, Optional

from lobster.version import __version__

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"lobster-tui (\d+)\.(\d+)")
_PROBE_TIMEOUT = 3  # seconds; a binary that cannot answer by then is skipped
_cache: Dict[str, Optional[str]] = {}  # path -> reported "major.minor" (None: unknown)
_rejected: Dict[str, Optional[str]] = {}
_hinted = False


def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def is_compatible(path: str) -> bool:
    """True when ``path --version`` reports the installed lobster-ai major.minor."""
    if path not in _cache:
        try:
            out = subprocess.run(  # nosec B603
                [path, "--version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT,
                check=False,
            ).stdout
            match = _VERSION_RE.search(out)
            _cache[path] = f"{match[1]}.{match[2]}" if match else None
        except (OSError, subprocess.SubprocessError):
            _cache[path] = None
    found = _cache[path]
    ok = found == _major_minor(__version__)
    if not ok:
        logger.debug("lobster-tui at %s skipped (version %s)", path, found)
        _rejected[path] = found
    return ok


def hint_if_rejected() -> None:
    """Once per process, say why no binary was used. Call only when none was found."""
    global _hinted
    if _rejected and not _hinted:
        _hinted = True
        path, found = next(iter(_rejected.items()))
        print(
            f"Ignoring lobster-tui at {path} (version {found or 'unknown'}, "
            f"need {_major_minor(__version__)}.x); using the built-in interface.",
            file=sys.stderr,
        )
