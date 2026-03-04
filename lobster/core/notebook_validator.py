"""Backward-compat shim. Use lobster.core.notebooks.validator instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.core.notebooks.validator' instead of "
    "'lobster.core.notebook_validator'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.notebooks.validator import *  # noqa: F401,F403
