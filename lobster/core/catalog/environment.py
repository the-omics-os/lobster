"""The software environment of a catalog run, as text.

Names and versions of the installed distributions, plus the Python version, platform
and Lobster version. Versions only: an editable install shows the version it was
installed at, not later edits to its files, and no path or user name is recorded.
The text is the same for the same environment, so one run after another records
the same file.
"""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import re
import sys


def environment_text(lobster_version: str) -> str:
    packages = {}
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"]
            if name:
                # The first listing of a name is the one the import system uses.
                packages.setdefault(re.sub(r"[-_.]+", "-", name).lower(), dist.version)
        except Exception:  # a half-installed distribution costs only its own line
            continue
    header = [
        f"# python {platform.python_version()} ({sys.implementation.name})",
        f"# platform {sys.platform}/{platform.machine()}",
        f"# lobster {lobster_version}",
    ]
    return "\n".join(header + [f"{n}=={v}" for n, v in sorted(packages.items())]) + "\n"
