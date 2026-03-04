"""Backward-compat shim. Use lobster.core.governance.license_manager instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.core.governance.license_manager' instead of "
    "'lobster.core.license_manager'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.governance.license_manager import *  # noqa: F401,F403
