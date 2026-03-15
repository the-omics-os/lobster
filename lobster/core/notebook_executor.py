"""Backward-compat shim. Use lobster.core.notebooks.executor instead."""

import warnings as _w

_w.warn(
    "Import from 'lobster.core.notebooks.executor' instead of "
    "'lobster.core.notebook_executor'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.notebooks.executor import *  # noqa: F401,F403
