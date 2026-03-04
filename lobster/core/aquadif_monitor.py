"""Backward-compat shim. Use lobster.core.governance.aquadif_monitor instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.core.governance.aquadif_monitor' instead of "
    "'lobster.core.aquadif_monitor'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.governance.aquadif_monitor import *  # noqa: F401,F403
