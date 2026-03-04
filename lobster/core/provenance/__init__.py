"""Provenance tracking: analysis IR, lineage, coverage analysis."""


def __getattr__(name):
    """Backward-compat shim for old `from lobster.core.provenance import X` usage."""
    import importlib
    import warnings

    _provenance_mod = importlib.import_module("lobster.core.provenance.provenance")
    if hasattr(_provenance_mod, name):
        warnings.warn(
            f"Import '{name}' from 'lobster.core.provenance.provenance' instead of "
            "'lobster.core.provenance'. Shim will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_provenance_mod, name)
    raise AttributeError(f"module 'lobster.core.provenance' has no attribute {name!r}")
