"""Backward-compat shim. Use lobster.core.provenance.ir_coverage instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.core.provenance.ir_coverage' instead of "
    "'lobster.core.ir_coverage'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.provenance.ir_coverage import *  # noqa: F401,F403
