"""Backward-compat shim. Use lobster.core.provenance.analysis_ir instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.core.provenance.analysis_ir' instead of "
    "'lobster.core.analysis_ir'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.provenance.analysis_ir import *  # noqa: F401,F403
