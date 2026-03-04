# Phase 10: Fix Provenance `__getattr__` Shim - Research

**Researched:** 2026-03-04
**Domain:** Python `__getattr__` package shim, backward-compatible imports
**Confidence:** HIGH

## Summary

The `provenance/__init__.py` `__getattr__` shim created in Phase 06 only delegates to `provenance.py` (which exports `ProvenanceTracker`). It does not search `analysis_ir.py`, `lineage.py`, or `ir_coverage.py`. This means `from lobster.core.provenance import AnalysisStep` raises `AttributeError` (surfaces as `ImportError` in `from ... import X` syntax) at 6 runtime call sites in `de_analysis_expert.py`, plus 2 TYPE_CHECKING-only sites in `core/interfaces/download_service.py` and `core/protocols.py`.

The fix is straightforward: extend the `__getattr__` function in `provenance/__init__.py` to search all 4 submodules (provenance.py, analysis_ir.py, lineage.py, ir_coverage.py) instead of only `provenance.py`. This is a ~15-line change to one file, plus tests.

**Primary recommendation:** Extend `provenance/__init__.py.__getattr__` to search all 4 submodules in order. Do NOT update the 6+2 call sites in de_analysis_expert.py/interfaces/protocols -- the shim should make them work transparently, matching the existing pattern for `ProvenanceTracker`.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CORE-04 | Import-linter config updated for new subpackage paths | Provenance `__getattr__` shim must cover all submodule exports so that `from lobster.core.provenance import X` works for any public name, completing the backward-compat story started in Phase 06 |
</phase_requirements>

## Standard Stack

### Core

No new libraries needed. This is a pure Python `__getattr__` module-level shim pattern.

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python `importlib` | stdlib | Dynamic module import in `__getattr__` | Already used in existing shim |
| Python `warnings` | stdlib | DeprecationWarning emission | Already used in existing shim |

## Architecture Patterns

### Current Broken Pattern

```python
# provenance/__init__.py (CURRENT - only searches provenance.py)
def __getattr__(name):
    import importlib, warnings
    _provenance_mod = importlib.import_module("lobster.core.provenance.provenance")
    if hasattr(_provenance_mod, name):
        warnings.warn(...)
        return getattr(_provenance_mod, name)
    raise AttributeError(...)
```

### Fixed Pattern: Multi-Module `__getattr__` Search

```python
# provenance/__init__.py (FIXED - searches all 4 submodules)
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
```

### Key Design Decisions

1. **Search order matters:** `provenance.py` first (most common existing usage -- `ProvenanceTracker`), then `analysis_ir.py` (second most common -- `AnalysisStep`), then `lineage.py`, then `ir_coverage.py`.

2. **Do NOT update call sites:** The 6 de_analysis_expert.py call sites and 2 TYPE_CHECKING sites use `from lobster.core.provenance import AnalysisStep` -- this is the "old-style" import that the shim is designed to support. Changing them would be scope creep (that belongs to Phase 11's broader migration or future SHIM-01 retirement).

3. **Deprecation message is specific:** Points to the exact submodule (`lobster.core.provenance.analysis_ir`) not just the package. This helps developers migrate to the canonical path.

4. **`_SUBMODULES` tuple at module level:** Avoids re-defining the list on every `__getattr__` call. Tuple is immutable and cheap.

### Anti-Patterns to Avoid

- **Do NOT eagerly import submodules in `__init__.py`:** This would defeat lazy loading and potentially cause circular imports. The `__getattr__` shim fires only on attribute access.
- **Do NOT add `__all__` to `__init__.py`:** Wildcard imports (`from lobster.core.provenance import *`) are not a supported pattern and adding `__all__` would change semantics.
- **Do NOT change the existing shim behavior for `ProvenanceTracker`:** The warning message and stacklevel must remain identical for existing test coverage.

## Affected Code Inventory

### Files That Import `from lobster.core.provenance import X` (runtime, not TYPE_CHECKING)

| File | Name Imported | Line(s) |
|------|---------------|---------|
| `de_analysis_expert.py` | `AnalysisStep` | 640, 1123, 2096, 2294 |
| `de_analysis_expert.py` | `AnalysisStep as ProvAnalysisStep` | 2467, 2937 |
| `pseudobulk_service.py` | `ProvenanceTracker` | 25 |
| 7 test files | `ProvenanceTracker` | various |

### Files That Import `from lobster.core.provenance import X` (TYPE_CHECKING only)

| File | Name Imported |
|------|---------------|
| `core/interfaces/download_service.py` | `AnalysisStep` |
| `core/protocols.py` | `AnalysisStep` |

### Public Names Per Submodule

| Submodule | Key Public Names |
|-----------|-----------------|
| `provenance.py` | `ProvenanceTracker` |
| `analysis_ir.py` | `ParameterSpec`, `AnalysisStep`, `validate_ir_list`, `extract_unique_imports`, `extract_unique_helper_code`, `create_minimal_ir`, `create_data_loading_ir`, `create_data_saving_ir` |
| `lineage.py` | `LINEAGE_KEY`, `SUFFIX_PATTERNS`, `LineageMetadata`, `extract_base_name`, `infer_processing_step`, `create_lineage_metadata`, `attach_lineage`, `get_lineage`, `has_lineage`, `ensure_lineage`, `get_lineage_dict` |
| `ir_coverage.py` | `ServiceCoverage`, `CoverageReport`, `IRCoverageAnalyzer`, `main` |

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Multi-module attr lookup | Custom import hooks or `sys.meta_path` | Simple `__getattr__` loop over submodule list | `__getattr__` is the standard Python mechanism (PEP 562) |

## Common Pitfalls

### Pitfall 1: Name Collision Across Submodules
**What goes wrong:** If two submodules export the same name, the first one in `_SUBMODULES` wins silently.
**Why it happens:** Linear search stops at first match.
**How to avoid:** Verify no name collisions exist across the 4 submodules. Current state: `get_lineage` exists in both `provenance.py` and `lineage.py`; `to_dict`/`from_dict` exist as methods in multiple classes but these are not module-level public names accessed via `from module import name`.
**Warning signs:** Unexpected object type after import.
**Resolution:** `get_lineage` in `provenance.py` is a method on `ProvenanceTracker`, not a module-level function. Module-level `get_lineage` is only in `lineage.py`. No actual collision exists for `__getattr__` since `hasattr(provenance_mod, "get_lineage")` will be False (it is a bound method, not a module attribute). Verified: no name collision risk.

### Pitfall 2: Warning Stacklevel
**What goes wrong:** DeprecationWarning points to the wrong call site (inside the shim instead of the caller).
**Why it happens:** `stacklevel=2` is correct for direct `from X import Y`, but if imported transitively, the warning points to the intermediate module.
**How to avoid:** Keep `stacklevel=2` -- matches existing behavior. The existing shim already uses this.

### Pitfall 3: Breaking Existing `ProvenanceTracker` Test
**What goes wrong:** The existing test `test_core_subpackage_shims.py` checks that `from lobster.core.provenance import ProvenanceTracker` emits a DeprecationWarning mentioning `lobster.core.provenance.provenance`.
**How to avoid:** Ensure the warning message format stays compatible. The existing test checks `new_path.rsplit(".", 1)[0] in str(w.message)` which resolves to `"lobster.core.provenance"` being in the message -- this will still match with the new multi-module shim.

## Code Examples

### The Fix (provenance/__init__.py)

```python
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
```

### Test: AnalysisStep Accessible via Shim

```python
import importlib
import sys
import warnings

def test_provenance_shim_resolves_analysis_step():
    """AnalysisStep from analysis_ir.py reachable via provenance package shim."""
    sys.modules.pop("lobster.core.provenance", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from lobster.core.provenance import AnalysisStep
    assert AnalysisStep is not None
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("analysis_ir" in str(w.message) for w in dep)
```

### Test: All Public Names Accessible

```python
import importlib
import warnings

# Names that must be resolvable via the shim
MUST_RESOLVE = [
    ("AnalysisStep", "analysis_ir"),
    ("ParameterSpec", "analysis_ir"),
    ("ProvenanceTracker", "provenance"),
    ("LineageMetadata", "lineage"),
    ("IRCoverageAnalyzer", "ir_coverage"),
]

def test_all_key_names_resolvable():
    for name, expected_submod in MUST_RESOLVE:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            mod = importlib.import_module("lobster.core.provenance")
            obj = getattr(mod, name)
            assert obj is not None
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` [tool.pytest] |
| Quick run command | `python -m pytest tests/unit/core/test_core_subpackage_shims.py -x -q` |
| Full suite command | `python -m pytest tests/unit/core/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CORE-04 | `from lobster.core.provenance import AnalysisStep` resolves | unit | `python -m pytest tests/unit/core/test_core_subpackage_shims.py -x -q` | Partial -- existing file tests ProvenanceTracker but not AnalysisStep specifically |
| CORE-04 | All public names in analysis_ir/lineage/ir_coverage reachable | unit | `python -m pytest tests/unit/core/test_provenance_shim_coverage.py -x -q` | No -- Wave 0 |
| CORE-04 | de_analysis_expert call sites execute without ImportError | integration | `python -c "from lobster.core.provenance import AnalysisStep; print('OK')"` | No -- smoke test |
| CORE-04 | Existing ProvenanceTracker shim preserved | unit | `python -m pytest tests/unit/core/test_core_subpackage_shims.py::TestShimReexportsAndWarns::test_shim_reexports_and_warns[provenance] -x -q` | Yes |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/unit/core/test_core_subpackage_shims.py -x -q`
- **Per wave merge:** `python -m pytest tests/unit/core/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] Add `AnalysisStep`, `ParameterSpec`, `LineageMetadata`, `IRCoverageAnalyzer` to shim test coverage in `test_core_subpackage_shims.py` or a new test file
- [ ] Smoke test: `python -c "from lobster.core.provenance import AnalysisStep"` passes

## Scope Boundary

### In Scope
- Extend `provenance/__init__.py.__getattr__` to search all 4 submodules
- Add/extend tests for the multi-module shim
- Verify existing tests still pass

### Out of Scope
- Migrating the 6 de_analysis_expert.py call sites to canonical paths (Phase 11 / SHIM-01)
- Migrating the 2 TYPE_CHECKING call sites to canonical paths
- Any changes to `analysis_ir.py`, `lineage.py`, `ir_coverage.py`, or `provenance.py` themselves
- Import-linter rule changes (those were already handled in Phase 06)

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `lobster/core/provenance/__init__.py` (current broken shim)
- Direct code inspection: `lobster/core/provenance/analysis_ir.py` (AnalysisStep location)
- Direct code inspection: `de_analysis_expert.py` lines 640, 1123, 2096, 2294, 2467, 2937
- Direct code inspection: `tests/unit/core/test_core_subpackage_shims.py` (existing test coverage)
- `.planning/v1.0-MILESTONE-AUDIT.md` (gap identification)
- Phase 06 VERIFICATION.md (confirms shim only covers provenance.py)
- PEP 562 -- Module `__getattr__` (Python standard, well-established since 3.7)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no external dependencies, pure Python stdlib
- Architecture: HIGH - extending existing pattern with one additional loop
- Pitfalls: HIGH - verified no name collisions, checked existing test compatibility

**Research date:** 2026-03-04
**Valid until:** Indefinite (Python `__getattr__` semantics are stable)
