"""Locate the lobster-tui Go binary across all installation methods."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Binary name on all platforms (Windows gets .exe appended automatically via PATH lookup)
_BINARY_NAME = "lobster-tui"


def find_tui_binary() -> Optional[str]:
    """Find the lobster-tui binary. Returns absolute path string or None.

    Search order:
      1. Platform wheel: try ``import lobster_ai_tui`` and check its data scripts dir
      2. User cache: ~/.cache/lobster/bin/lobster-tui
      3. PATH: shutil.which("lobster-tui")
      4. Development build: lobster-tui/lobster-tui relative to the repo root
         (works when running directly from the source checkout)

    Returns None when nothing is found — callers must fall back gracefully.
    """
    # ------------------------------------------------------------------ #
    # 1. Platform wheel (lobster-ai-tui PyPI package)                     #
    # ------------------------------------------------------------------ #
    try:
        import lobster_ai_tui  # type: ignore[import]

        wheel_dir = Path(lobster_ai_tui.__file__).parent
        # Wheels typically place binaries in a `bin/` subdir of the package
        for candidate in (
            wheel_dir / "bin" / _BINARY_NAME,
            wheel_dir / _BINARY_NAME,
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                logger.debug("lobster-tui found via wheel: %s", candidate)
                return str(candidate)
    except ImportError:
        pass  # Package not installed — not an error

    # ------------------------------------------------------------------ #
    # 2. User cache: ~/.cache/lobster/bin/lobster-tui                     #
    # ------------------------------------------------------------------ #
    cache_bin = Path.home() / ".cache" / "lobster" / "bin" / _BINARY_NAME
    if cache_bin.is_file() and os.access(cache_bin, os.X_OK):
        logger.debug("lobster-tui found in cache: %s", cache_bin)
        return str(cache_bin)

    # ------------------------------------------------------------------ #
    # 3. PATH lookup                                                       #
    # ------------------------------------------------------------------ #
    which = shutil.which(_BINARY_NAME)
    if which:
        logger.debug("lobster-tui found on PATH: %s", which)
        return which

    # ------------------------------------------------------------------ #
    # 4. Development build — lobster-tui/lobster-tui relative to repo     #
    # ------------------------------------------------------------------ #
    # Walk up from this file to find the repo root (contains lobster-tui/ dir)
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        dev_bin = ancestor / "lobster-tui" / _BINARY_NAME
        if dev_bin.is_file() and os.access(dev_bin, os.X_OK):
            logger.debug("lobster-tui found in dev build: %s", dev_bin)
            return str(dev_bin)
        # Stop searching once we leave plausible repo roots (e.g. at /Users or /)
        if ancestor == ancestor.parent:
            break

    logger.debug("lobster-tui binary not found")
    return None
