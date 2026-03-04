"""Backward-compat shim. Use lobster.core.runtime.workspace instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.core.runtime.workspace' instead of "
    "'lobster.core.workspace'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.runtime.workspace import *  # noqa: F401,F403
