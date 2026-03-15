"""Backward-compat shim. Use lobster.core.notebooks.exporter instead."""

import warnings as _w

_w.warn(
    "Import from 'lobster.core.notebooks.exporter' instead of "
    "'lobster.core.notebook_exporter'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.notebooks.exporter import *  # noqa: F401,F403
