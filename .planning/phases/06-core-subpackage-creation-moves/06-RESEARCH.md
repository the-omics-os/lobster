# Phase 6: Core Subpackage Creation + Moves - Research

**Researched:** 2026-03-04
**Domain:** Python package restructuring, backward-compatible module moves, import shims, import-linter
**Confidence:** HIGH

## Summary

Phase 6 reorganizes 13 files from the flat `lobster/core/` directory into 5 domain subpackages (`runtime/`, `queues/`, `notebooks/`, `provenance/`, `governance/`). Each moved file leaves behind a backward-compatible shim at the old path that re-exports everything and emits a `DeprecationWarning`. This is a pure mechanical restructuring with zero behavior changes.

The codebase already has a proven shim pattern in `lobster/core/vector/__init__.py` that re-exports from `lobster.services.vector` with a deprecation warning. The same pattern applies here. The primary risk is the high import count on `analysis_ir.py` (125 importers) and `provenance.py` (23 importers), but shims make this safe. Cross-subpackage dependencies are unidirectional (notebooks -> provenance, queues internal only) with no cycles.

**Primary recommendation:** Create all 5 subpackage `__init__.py` files first, move files one subpackage at a time starting with the lowest-blast-radius group (governance: 2 files, ~14 importers), and verify `pytest` passes after each subpackage move.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CORE-01 | Core subpackages created: runtime/, queues/, notebooks/, provenance/, governance/ | Move Map section defines exact subpackage structure and file assignments |
| CORE-02 | 13 files moved to domain subpackages with backward-compatible shims at old paths | Shim Pattern section provides verified template from existing vector/ shims |
| CORE-03 | Shims emit DeprecationWarning with removal version | Shim Pattern section specifies exact warning format with v2.0.0 removal target |
| CORE-04 | Import-linter config updated for new subpackage paths | Import-Linter section documents current config, pre-existing violations, and required updates |
| CORE-05 | No import cycles introduced | Dependency Graph section proves all cross-subpackage deps are unidirectional |
</phase_requirements>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| import-linter | >=2.1 | Enforce import boundaries between subpackages | Already declared in pyproject.toml dev deps |
| Python warnings module | stdlib | Emit DeprecationWarning from shims | Standard Python deprecation mechanism |
| pytest | existing | Validate all imports still work post-move | Already the project test framework |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| N/A | - | No new libraries needed | This is pure file reorganization |

**Installation:** No new packages required.

## Architecture Patterns

### Target Directory Structure
```
lobster/core/
├── __init__.py                 # Existing (unchanged)
├── runtime/
│   ├── __init__.py             # NEW - empty or minimal
│   └── workspace.py            # MOVED from core/workspace.py
├── queues/
│   ├── __init__.py             # NEW - empty or minimal
│   ├── download_queue.py       # MOVED from core/download_queue.py
│   ├── publication_queue.py    # MOVED from core/publication_queue.py
│   └── queue_storage.py        # MOVED from core/queue_storage.py
├── notebooks/
│   ├── __init__.py             # NEW - empty or minimal
│   ├── executor.py             # MOVED from core/notebook_executor.py (RENAMED)
│   ├── exporter.py             # MOVED from core/notebook_exporter.py (RENAMED)
│   └── validator.py            # MOVED from core/notebook_validator.py (RENAMED)
├── provenance/
│   ├── __init__.py             # NEW - empty or minimal
│   ├── analysis_ir.py          # MOVED from core/analysis_ir.py
│   ├── provenance.py           # MOVED from core/provenance.py
│   ├── lineage.py              # MOVED from core/lineage.py
│   └── ir_coverage.py          # MOVED from core/ir_coverage.py
├── governance/
│   ├── __init__.py             # NEW - empty or minimal
│   ├── license_manager.py      # MOVED from core/license_manager.py
│   └── aquadif_monitor.py      # MOVED from core/aquadif_monitor.py
├── adapters/                   # UNCHANGED
├── backends/                   # UNCHANGED
├── identifiers/                # UNCHANGED
├── interfaces/                 # UNCHANGED
├── schemas/                    # UNCHANGED
├── utils/                      # UNCHANGED
├── vector/                     # UNCHANGED (already has shims)
├── client.py                   # STAYS (deferred to future milestone)
├── component_registry.py       # STAYS
├── config_resolver.py          # STAYS
├── data_manager_v2.py          # STAYS (Phase 7 owns this move)
├── exceptions.py               # STAYS (shared infrastructure)
├── extraction_cache.py         # STAYS
├── omics_registry.py           # STAYS
├── plot_manager.py             # STAYS
├── plugin_loader.py            # STAYS
├── protocols.py                # STAYS
├── ris_parser.py               # STAYS
├── sparse_utils.py             # STAYS
└── uv_tool_env.py              # STAYS
```

### Files NOT moved in this phase
- `data_manager_v2.py` -- Phase 7 (PR-7) due to 197 importers blast radius
- `client.py` -- Deferred to future milestone (2,867 LOC)
- `component_registry.py`, `omics_registry.py`, `plugin_loader.py` -- The unified cleanup plan mentions `registry/` as optional; the requirements (CORE-01) list exactly 5 subpackages. Do NOT create a registry/ subpackage.

### Pattern: Backward-Compatible Shim

Proven pattern from `lobster/core/vector/__init__.py`:

```python
# lobster/core/download_queue.py (SHIM -- replaces original file)
"""Backward-compat shim. Use lobster.core.queues.download_queue instead."""

import warnings as _w

_w.warn(
    "Import from 'lobster.core.queues.download_queue' instead of "
    "'lobster.core.download_queue'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.queues.download_queue import *  # noqa: F401,F403
```

**Critical details:**
- `stacklevel=2` ensures the warning points to the caller, not the shim
- `import *` re-exports the exact same class objects -- `isinstance` checks work because they are the same objects, not copies
- `# noqa: F401,F403` suppresses flake8 warnings about wildcard imports and unused imports
- The warning fires once per module-level import (Python caches module objects)

### Pattern: Notebook File Renaming

Three notebook files get shorter names when moved:
- `notebook_executor.py` -> `notebooks/executor.py`
- `notebook_exporter.py` -> `notebooks/exporter.py`
- `notebook_validator.py` -> `notebooks/validator.py`

The shim for these uses the new name:
```python
# lobster/core/notebook_executor.py (SHIM)
"""Backward-compat shim. Use lobster.core.notebooks.executor instead."""

import warnings as _w

_w.warn(
    "Import from 'lobster.core.notebooks.executor' instead of "
    "'lobster.core.notebook_executor'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.notebooks.executor import *  # noqa: F401,F403
```

### Pattern: Subpackage `__init__.py`

Keep `__init__.py` files minimal. Do NOT re-export everything from the subpackage -- let consumers import from specific modules:

```python
# lobster/core/queues/__init__.py
"""Queue infrastructure for download and publication pipelines."""
```

This avoids circular imports and keeps discovery fast.

### Anti-Patterns to Avoid
- **Re-exporting from subpackage `__init__.py`:** Do NOT add `from .download_queue import *` to `core/queues/__init__.py`. This creates import coupling and can trigger circular imports. Consumers should import from the specific module.
- **Updating all 125 importers of analysis_ir.py:** Do NOT change existing import paths in this phase. That is Phase 10 (Shim Retirement). Shims exist specifically to avoid this.
- **Moving data_manager_v2.py:** This is explicitly Phase 7. Do not touch it.
- **Creating registry/ subpackage:** The requirements specify exactly 5 subpackages. The unified plan marked registry/ as "optional -- evaluate". Do not create it.

## Move Map (13 files)

### Blast Radius by Subpackage

| Subpackage | Files | Total Importers | Risk |
|------------|-------|-----------------|------|
| governance/ | license_manager.py, aquadif_monitor.py | ~14 | LOW |
| queues/ | download_queue.py, publication_queue.py, queue_storage.py | ~18 | LOW |
| notebooks/ | executor.py, exporter.py, validator.py | ~11 | LOW |
| runtime/ | workspace.py | ~10 | LOW |
| provenance/ | analysis_ir.py, provenance.py, lineage.py, ir_coverage.py | ~151 | MEDIUM |

**Recommended move order:** governance -> queues -> notebooks -> runtime -> provenance (lowest to highest blast radius).

### Cross-Subpackage Dependencies

```
notebooks/exporter.py  --->  provenance/analysis_ir.py   (import AnalysisStep, ParameterSpec, etc.)
notebooks/exporter.py  --->  provenance/provenance.py    (import ProvenanceTracker)
notebooks/executor.py  --->  core/data_manager_v2.py     (stays in place, not a cross-subpackage dep)
notebooks/exporter.py  --->  core/data_manager_v2.py     (stays in place)
download_queue.py      --->  queue_storage.py             (SAME subpackage: queues/)
publication_queue.py   --->  queue_storage.py             (SAME subpackage: queues/)
download_queue.py      --->  core/schemas/download_queue  (schemas/ stays, not affected)
publication_queue.py   --->  core/schemas/publication_queue (schemas/ stays, not affected)
provenance/provenance.py --> core/utils/h5ad_utils.py     (utils/ stays, not affected)
```

**Cycle analysis:** All dependencies flow FROM notebooks TO provenance, and FROM queues internally. No reverse dependencies exist. No cycles possible between the 5 new subpackages.

### Internal Import Updates Required

When a file moves, its internal imports from `lobster.core.X` to siblings in the same subpackage should use the NEW canonical path (not rely on shims). Specifically:

1. **download_queue.py** imports `from lobster.core.queue_storage import ...` -- update to `from lobster.core.queues.queue_storage import ...`
2. **publication_queue.py** imports `from lobster.core.queue_storage import ...` -- update to `from lobster.core.queues.queue_storage import ...`
3. **notebook_exporter.py** imports `from lobster.core.analysis_ir import ...` -- update to `from lobster.core.provenance.analysis_ir import ...`
4. **notebook_exporter.py** imports `from lobster.core.provenance import ProvenanceTracker` -- update to `from lobster.core.provenance.provenance import ProvenanceTracker`
5. **notebook_exporter.py** imports `from lobster.core.data_manager_v2 import DataManagerV2` -- leave as-is (data_manager_v2 moves in Phase 7)
6. **notebook_executor.py** imports `from lobster.core.data_manager_v2 import DataManagerV2` -- leave as-is (Phase 7)

**Key rule:** Files within the SAME NEW subpackage should import each other via the new canonical path. Files importing from modules that are NOT being moved in this phase should keep current paths.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Deprecation warnings | Custom warning infrastructure | `warnings.warn(msg, DeprecationWarning, stacklevel=2)` | stdlib, proven in vector/ shims |
| Import re-export | Manual class-by-class re-export | `from new_module import *` | Preserves isinstance identity, proven pattern |
| Cycle detection | Manual import tracing | `lint-imports` (import-linter) | Already configured in `.importlinter` |
| Finding all importers | Manual grep | `grep -r "from lobster.core.X import" --include="*.py"` | Reliable for this codebase structure |

## Common Pitfalls

### Pitfall 1: Circular imports from subpackage __init__.py
**What goes wrong:** Adding `from .download_queue import DownloadQueue` to `core/queues/__init__.py` can create circular imports if `download_queue.py` imports from another subpackage whose `__init__.py` also eagerly imports.
**Why it happens:** Python executes `__init__.py` during package import, before any submodule is fully loaded.
**How to avoid:** Keep all subpackage `__init__.py` files as docstring-only or with minimal constants. Never import from submodules in `__init__.py`.
**Warning signs:** `ImportError: cannot import name 'X' from partially initialized module`

### Pitfall 2: Shim fires on wrong stacklevel
**What goes wrong:** Warning shows the shim file as the source instead of the actual caller.
**Why it happens:** `stacklevel=1` (default) points to the `warnings.warn()` call itself.
**How to avoid:** Always use `stacklevel=2` in shims. This makes the warning point to the file that did `from lobster.core.download_queue import ...`.
**Warning signs:** Warning traceback points to the shim file, not the importing file.

### Pitfall 3: Missing explicit re-exports for __all__-less modules
**What goes wrong:** `from module import *` only exports names not starting with `_`. If the moved module has important underscore-prefixed names used externally, they get lost.
**Why it happens:** Python's `import *` behavior respects `__all__` if defined, otherwise excludes `_`-prefixed names.
**How to avoid:** Check each file: only `queue_storage.py` defines `__all__`. For files without `__all__`, verify no underscore-prefixed public API is imported externally. Based on analysis, none of the 13 files have externally-used underscore names beyond the classes/functions already visible.
**Warning signs:** `ImportError: cannot import name '_some_helper'` from external code.

### Pitfall 4: Tests importing via old paths get silent deprecation warnings
**What goes wrong:** Test suite produces hundreds of DeprecationWarning lines cluttering output.
**Why it happens:** All existing test files import from old paths (which are now shims).
**How to avoid:** This is expected and acceptable for Phase 6. Tests are NOT updated to new paths in this phase -- that happens in Phase 10 (Shim Retirement). The warnings are informational. Consider `filterwarnings` in pytest config if noise is excessive, but do NOT suppress them in production code.
**Warning signs:** Console flood of deprecation warnings during test runs.

### Pitfall 5: Notebook file renaming breaks test patch paths
**What goes wrong:** Tests that `mock.patch("lobster.core.notebook_executor.SomeClass")` may need updating since the shim re-exports from a different module path.
**Why it happens:** `mock.patch` patches the name at the specified module path. If the shim does `from X import *`, the objects are re-bound in the shim module, so patching the shim path still works. But if tests patch internal implementation details, those paths change.
**How to avoid:** The shim `import *` creates name bindings in the shim module, so `mock.patch("lobster.core.notebook_executor.NotebookExecutor")` will still work because the name exists in the shim module. Only patches targeting private internals of the moved file would break. Verify test patch paths after each subpackage move.
**Warning signs:** `AttributeError: <module 'lobster.core.notebook_executor'> does not have attribute 'X'`

### Pitfall 6: Import-linter pre-existing violations mask new violations
**What goes wrong:** `lint-imports` already reports 3 broken contracts (layers, domain deps, independence). New violations from the restructuring could be missed.
**Why it happens:** The independence contract references `lobster.core.download_queue` which will become a shim.
**How to avoid:** Update the `.importlinter` config to reference new canonical paths. Run `lint-imports` before and after to diff the violations. The independence contract must be updated to reference `lobster.core.queues.download_queue` instead of `lobster.core.download_queue`.

## Import-Linter Configuration

### Current State
The `.importlinter` file has 3 contracts, all currently BROKEN due to pre-existing violations:
1. **core-agents-layers** -- services import agents (6 violations)
2. **no-domain-deps-in-core** -- pre-existing domain dep violations
3. **core-independence** -- data_manager_v2 imports download_queue and component_registry (via adapters)

### Required Updates for Phase 6

The `core-independence` contract currently references flat paths:
```ini
modules =
    lobster.core.data_manager_v2
    lobster.core.component_registry
    lobster.core.download_queue
```

After Phase 6, `download_queue` lives at `lobster.core.queues.download_queue`. Update to:
```ini
modules =
    lobster.core.data_manager_v2
    lobster.core.component_registry
    lobster.core.queues.download_queue
```

**Optional new contract** -- enforce no cycles between new subpackages:
```ini
[importlinter:contract:core-subpackage-independence]
name = Core subpackages are loosely coupled
type = independence
modules =
    lobster.core.runtime
    lobster.core.queues
    lobster.core.notebooks
    lobster.core.provenance
    lobster.core.governance
```

Note: This will report `notebooks -> provenance` as a violation because `notebook_exporter.py` imports from `analysis_ir.py` and `provenance.py`. This is an EXPECTED dependency (notebooks need provenance data to export). Options:
1. Use `ignore_imports` to allow this specific dependency
2. Use a `layers` contract instead (provenance below notebooks)
3. Accept the violation as documented and address in future

**Recommendation:** Use a layers contract with provenance below notebooks:
```ini
[importlinter:contract:core-subpackage-layers]
name = Core subpackages respect layering
type = layers
layers =
    lobster.core.notebooks
    lobster.core.provenance
    lobster.core.governance
    lobster.core.queues
    lobster.core.runtime
```

This allows notebooks -> provenance (higher layer can import lower) while preventing reverse dependencies.

## Code Examples

### Complete Shim File (verified pattern from core/vector/)
```python
# lobster/core/analysis_ir.py (SHIM after move)
"""Backward-compat shim. Use lobster.core.provenance.analysis_ir instead."""

import warnings as _w

_w.warn(
    "Import from 'lobster.core.provenance.analysis_ir' instead of "
    "'lobster.core.analysis_ir'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.provenance.analysis_ir import *  # noqa: F401,F403
```

### Minimal Subpackage __init__.py
```python
# lobster/core/provenance/__init__.py
"""Provenance tracking: analysis IR, lineage, coverage analysis."""
```

### Updated Internal Import (moved file)
```python
# lobster/core/queues/download_queue.py (MOVED -- update internal import)
# OLD: from lobster.core.queue_storage import (...)
# NEW:
from lobster.core.queues.queue_storage import (
    InterProcessFileLock,
    atomic_write_json,
    atomic_write_jsonl,
    backups_enabled,
    queue_file_lock,
)
```

### Smoke Test for Shim Validation
```python
# tests/unit/core/test_core_subpackage_shims.py
"""Verify all 13 shims re-export correctly and emit DeprecationWarning."""
import warnings
import pytest


SHIM_PAIRS = [
    ("lobster.core.download_queue", "lobster.core.queues.download_queue", "DownloadQueue"),
    ("lobster.core.publication_queue", "lobster.core.queues.publication_queue", "PublicationQueue"),
    ("lobster.core.queue_storage", "lobster.core.queues.queue_storage", "InterProcessFileLock"),
    ("lobster.core.notebook_executor", "lobster.core.notebooks.executor", "NotebookExecutor"),
    ("lobster.core.notebook_exporter", "lobster.core.notebooks.exporter", "NotebookExporter"),
    ("lobster.core.notebook_validator", "lobster.core.notebooks.validator", "NotebookValidator"),
    ("lobster.core.license_manager", "lobster.core.governance.license_manager", "get_current_tier"),
    ("lobster.core.aquadif_monitor", "lobster.core.governance.aquadif_monitor", "AquadifMonitor"),
    ("lobster.core.analysis_ir", "lobster.core.provenance.analysis_ir", "AnalysisStep"),
    ("lobster.core.provenance", "lobster.core.provenance.provenance", "ProvenanceTracker"),
    ("lobster.core.lineage", "lobster.core.provenance.lineage", "LineageMetadata"),
    ("lobster.core.ir_coverage", "lobster.core.provenance.ir_coverage", "IRCoverageAnalyzer"),
    ("lobster.core.workspace", "lobster.core.runtime.workspace", "resolve_workspace"),
]


@pytest.mark.parametrize("old_path,new_path,name", SHIM_PAIRS)
def test_shim_reexports_and_warns(old_path, new_path, name):
    """Old import path works via shim and emits DeprecationWarning."""
    import importlib
    # Clear cached module to ensure warning fires
    import sys
    sys.modules.pop(old_path, None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = importlib.import_module(old_path)
        assert hasattr(mod, name), f"{old_path} missing {name}"
        # Verify at least one DeprecationWarning
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, f"No DeprecationWarning from {old_path}"


@pytest.mark.parametrize("old_path,new_path,name", SHIM_PAIRS)
def test_isinstance_identity(old_path, new_path, name):
    """Object imported via old path is identical to object from new path."""
    import importlib
    old_mod = importlib.import_module(old_path)
    new_mod = importlib.import_module(new_path)
    old_obj = getattr(old_mod, name)
    new_obj = getattr(new_mod, name)
    assert old_obj is new_obj, f"{name} identity mismatch between {old_path} and {new_path}"
```

## Dependency Graph (No Cycles Proof)

```
governance/  (0 deps on other new subpackages)
  ├── license_manager.py     -- no lobster.core.* deps
  └── aquadif_monitor.py     -- no lobster.core.* deps

queues/  (0 deps on other new subpackages, internal deps only)
  ├── queue_storage.py       -- no lobster.core.* deps
  ├── download_queue.py      -- depends on queue_storage (SAME subpackage)
  └── publication_queue.py   -- depends on queue_storage (SAME subpackage)

runtime/  (0 deps on other new subpackages)
  └── workspace.py           -- no lobster.core.* deps

provenance/  (0 deps on other new subpackages)
  ├── analysis_ir.py         -- no lobster.core.* deps
  ├── provenance.py          -- depends on core/utils/h5ad_utils (NOT moved)
  ├── lineage.py             -- no lobster.core.* deps
  └── ir_coverage.py         -- no lobster.core.* deps

notebooks/  (depends on provenance/ -- ONE-WAY)
  ├── executor.py            -- depends on core/data_manager_v2 (NOT moved in this phase)
  ├── exporter.py            -- depends on provenance/analysis_ir, provenance/provenance, core/data_manager_v2
  └── validator.py           -- no lobster.core.* deps
```

**Direction:** notebooks -> provenance (one-way). No reverse path exists. No cycles.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `pytest tests/unit/core/ -x -q` |
| Full suite command | `pytest tests/unit/ tests/integration/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CORE-01 | 5 subpackages exist with __init__.py | unit | `pytest tests/unit/core/test_core_subpackage_shims.py::test_subpackages_exist -x` | Wave 0 |
| CORE-02 | 13 files moved, old paths work via shim | unit | `pytest tests/unit/core/test_core_subpackage_shims.py::test_shim_reexports_and_warns -x` | Wave 0 |
| CORE-03 | Shims emit DeprecationWarning with removal version | unit | `pytest tests/unit/core/test_core_subpackage_shims.py::test_shim_reexports_and_warns -x` | Wave 0 |
| CORE-04 | Import-linter passes with updated rules | smoke | `lint-imports` | Existing (.importlinter config) |
| CORE-05 | No import cycles between subpackages | smoke | `lint-imports` + `python -c "import lobster.core.queues.download_queue"` | Existing |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/core/ -x -q`
- **Per wave merge:** `pytest tests/unit/ -x && lint-imports`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/core/test_core_subpackage_shims.py` -- covers CORE-01, CORE-02, CORE-03 (shim validation + isinstance identity)
- [ ] Updated `.importlinter` config -- covers CORE-04 (new subpackage paths in independence/layers contract)

## Open Questions

1. **Provenance subpackage name collision**
   - What we know: `core/provenance.py` moves to `core/provenance/provenance.py`. The subpackage and file share the name "provenance".
   - What's unclear: Whether `from lobster.core.provenance import ProvenanceTracker` (which currently imports from the file) will conflict with `from lobster.core.provenance import provenance` (the subpackage module).
   - Recommendation: The shim at the OLD `core/provenance.py` path handles the backward-compat case. For the new subpackage, do NOT re-export `ProvenanceTracker` from `core/provenance/__init__.py` -- consumers should import from `lobster.core.provenance.provenance`. The shim replaces the old file entirely (it becomes the `core/provenance.py` -> wait, this creates a conflict: you can't have both `core/provenance.py` AND `core/provenance/` directory). **Resolution:** The old `core/provenance.py` file MUST be deleted when creating `core/provenance/` directory (Python cannot have a module and package with the same name). The shim must live inside `core/provenance/__init__.py` instead, which does the deprecation warning + re-export from `core/provenance/provenance.py`. This is a CRITICAL implementation detail.

2. **pytest DeprecationWarning noise**
   - What we know: 125+ importers of `analysis_ir` will trigger deprecation warnings during test runs.
   - What's unclear: Whether this will cause pytest to fail if `filterwarnings = "error"` is set.
   - Recommendation: Check `pyproject.toml` pytest config for warning filters. If warnings are errors, add `ignore::DeprecationWarning:lobster.core` filter for the shim modules.

## Critical Implementation Detail: Module/Package Name Collision

**This is the most important finding of this research.**

Python cannot have both `lobster/core/provenance.py` (module) and `lobster/core/provenance/` (package) simultaneously. The same applies to NO other files in our move list (only `provenance` has this collision).

**Solution for `provenance`:** When creating `core/provenance/` package:
1. Delete `core/provenance.py` (the old file)
2. Move the content to `core/provenance/provenance.py`
3. In `core/provenance/__init__.py`, add the deprecation shim that re-exports from `core/provenance/provenance`:

```python
# lobster/core/provenance/__init__.py
"""Provenance tracking: analysis IR, lineage, coverage analysis.

NOTE: This __init__.py also serves as the backward-compat shim for the old
`from lobster.core.provenance import ProvenanceTracker` import path.
"""
import warnings as _w

# Shim for old `from lobster.core.provenance import ProvenanceTracker` usage
_w.warn(
    "Import from 'lobster.core.provenance.provenance' instead of "
    "'lobster.core.provenance'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.provenance.provenance import *  # noqa: F401,F403
```

**Problem with this approach:** The deprecation warning fires on ANY import of the `provenance` package, including valid new-style imports like `from lobster.core.provenance.analysis_ir import AnalysisStep`. This is because Python imports the package `__init__.py` before accessing submodules.

**Better solution:** Use lazy deprecation -- only warn when accessing names that were part of the OLD `provenance.py` module:

```python
# lobster/core/provenance/__init__.py
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
```

This is the RECOMMENDED approach. It:
- Fires only when someone does `from lobster.core.provenance import ProvenanceTracker` (old-style)
- Does NOT fire when someone does `from lobster.core.provenance.analysis_ir import AnalysisStep` (new-style)
- Preserves isinstance identity

**Note:** `from lobster.core.provenance import *` will NOT work with `__getattr__` alone. If the old module has `__all__`, we'd need to define it in `__init__.py` too. Check: `core/provenance.py` does NOT define `__all__`, so `import *` from it is already unreliable. No action needed for wildcard imports.

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis of all 13 files, their imports, and importer counts
- Existing `core/vector/__init__.py` shim -- verified working pattern in this codebase
- `.importlinter` config -- read directly, violations verified by running `lint-imports`
- `kevin_notes/UNIFIED_CLEANUP_PLAN.md` -- PR-6 specification (authoritative)

### Secondary (MEDIUM confidence)
- Python documentation on `__getattr__` for module-level attribute access (PEP 562)
- Python module/package name collision behavior (well-documented CPython behavior)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, all patterns verified in codebase
- Architecture: HIGH -- move map derived from authoritative cleanup plan + verified with actual file analysis
- Pitfalls: HIGH -- the module/package name collision for `provenance` is the critical finding, verified against Python semantics
- Import-linter: MEDIUM -- pre-existing violations make it harder to verify new rules in isolation

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable -- this is internal restructuring, no external deps)
