"""Provenance tracking: analysis IR, lineage, coverage analysis."""

_SUBMODULES = (
    "lobster.core.provenance.provenance",
    "lobster.core.provenance.analysis_ir",
    "lobster.core.provenance.lineage",
    "lobster.core.provenance.ir_coverage",
)


def __getattr__(name):
    """Backward-compat shim for old `from lobster.core.provenance import X` usage."""
    import importlib
    import warnings

    for mod_path in _SUBMODULES:
        mod = importlib.import_module(mod_path)
        if hasattr(mod, name):
            warnings.warn(
                f"Import '{name}' from '{mod_path}' instead of "
                "'lobster.core.provenance'. Shim will be removed in v2.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(mod, name)
    raise AttributeError(f"module 'lobster.core.provenance' has no attribute {name!r}")
