# Phase 7: data_manager_v2 Move - Research

**Researched:** 2026-03-04
**Domain:** Python module relocation, backward-compatible shims, mock.patch path migration, CI lint enforcement
**Confidence:** HIGH

## Summary

Phase 7 moves `lobster/core/data_manager_v2.py` (3,999 LOC) to `lobster/core/runtime/data_manager.py` and leaves a backward-compatible shim at the old path. The `runtime/` subpackage already exists from Phase 6 (currently contains only `workspace.py`). The move follows the exact same shim pattern proven across 13 files in Phase 6.

The primary complication versus Phase 6 is the high importer count (~200 files: 80 source + 120 test) and -- critically -- 110 `mock.patch()` references in test files that target `lobster.core.data_manager_v2.*` attributes. These patch strings MUST be updated to `lobster.core.runtime.data_manager.*` because mock.patch patches the name in the module where the code actually lives, not in shim re-exports. This is the #1 risk in this phase and must be handled carefully.

Secondary tasks include updating scaffold templates (2 Jinja2 files reference the old path), updating the import-linter config (the `core-independence` contract references the old module path), and adding a CI check to prevent new code from importing via the deprecated path.

**Primary recommendation:** Move the file, create the shim, update all mock.patch strings in the 2 affected test files, update scaffold templates and import-linter, then add a grep-based CI check for old-path imports in new files.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DMGR-01 | data_manager_v2.py moved to core/runtime/data_manager.py | Move Map section defines exact source/target paths; runtime/ subpackage already exists |
| DMGR-02 | Shim at old path re-exports everything -- zero breakage for 80 source + 120 test importers | Shim Pattern section provides verified template from Phase 6; mock.patch Pitfall section identifies the 110 patch strings that need updating |
| DMGR-03 | Scaffold templates updated to import from new path | Scaffold Templates section identifies exactly 2 .j2 files (agent.py.j2, shared_tools.py.j2) with old-path imports |
| DMGR-04 | New code cannot import from old path (CI/lint check) | CI Enforcement section recommends grep-based check in CI workflow or pre-commit hook |
</phase_requirements>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python warnings module | stdlib | Emit DeprecationWarning from shim | Same mechanism used in all 13 Phase 6 shims |
| import-linter | >=2.1 | Enforce import boundaries | Already declared in pyproject.toml dev deps, config in .importlinter |
| pytest | existing | Validate all imports and patches work post-move | Project test framework |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| grep (CI) | system | Prevent new files from importing old path | CI workflow or pre-commit check |

**Installation:** No new packages required.

## Architecture Patterns

### Move Map

```
BEFORE:
lobster/core/
├── data_manager_v2.py          # 3,999 LOC, the file being moved
├── runtime/
│   ├── __init__.py             # "Runtime infrastructure: workspace resolution."
│   └── workspace.py            # Moved in Phase 6

AFTER:
lobster/core/
├── data_manager_v2.py          # SHIM (warns + re-exports from new path)
├── runtime/
│   ├── __init__.py             # Updated docstring to include data manager
│   ├── workspace.py            # Existing (unchanged)
│   └── data_manager.py         # MOVED from core/data_manager_v2.py (RENAMED)
```

**Filename change:** `data_manager_v2.py` -> `data_manager.py` (drop the `_v2` suffix since there is no v1 in `runtime/`). This follows the Phase 6 precedent where `notebook_executor.py` was renamed to `executor.py` on move.

### Canonical Import Path (after move)

```python
# NEW canonical path (all new code uses this)
from lobster.core.runtime.data_manager import DataManagerV2

# OLD path (shim, emits DeprecationWarning, works until v2.0.0)
from lobster.core.data_manager_v2 import DataManagerV2
```

### Shim Pattern (Phase 6 proven template)

```python
# lobster/core/data_manager_v2.py (becomes shim)
"""Backward-compat shim. Use lobster.core.runtime.data_manager instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.core.runtime.data_manager' instead of "
    "'lobster.core.data_manager_v2'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.runtime.data_manager import *  # noqa: F401,F403
```

This is identical to the `lobster/core/workspace.py` shim created in Phase 6.

### Internal Import Updates (inside the moved file)

The moved `data_manager.py` currently imports from old (shimmed) paths. Update these to canonical paths:

| Old Import | New Import |
|-----------|-----------|
| `from lobster.core.analysis_ir import ...` | `from lobster.core.provenance.analysis_ir import ...` |
| `from lobster.core.provenance import ProvenanceTracker` | `from lobster.core.provenance.provenance import ProvenanceTracker` |
| `from lobster.core.queue_storage import ...` | `from lobster.core.queues.queue_storage import ...` |
| `from lobster.core.workspace import resolve_workspace` | `from lobster.core.runtime.workspace import resolve_workspace` |

Other imports (`adapters.base`, `plot_manager`, `interfaces.*`, `utils.h5ad_utils`) remain unchanged -- those modules were not moved in Phase 6.

### Anti-Patterns to Avoid
- **Moving importers en masse:** Do NOT update the 200+ `from lobster.core.data_manager_v2 import DataManagerV2` lines across the codebase. The shim handles backward compatibility. Mass updates are deferred to v2.0.0 (SHIM-01).
- **Forgetting mock.patch strings:** The shim handles `from ... import` but NOT `mock.patch("lobster.core.data_manager_v2.X")`. These MUST be updated (see Pitfalls section).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Backward-compat shim | Custom `__init__.py` with `__getattr__` | Simple `import * + warnings.warn` | data_manager_v2.py is a file (not a package-name collision), so simple re-export works. `__getattr__` is only needed for package/module name collisions (like provenance/) |
| CI old-path check | Custom AST parser | `grep -r` in CI step | Simple pattern match sufficient for `from lobster.core.data_manager_v2` string |
| Import-linter enforcement | New contract type | Update existing `core-independence` contract module path | Just change the module string |

## Common Pitfalls

### Pitfall 1: mock.patch Strings Break After Move
**What goes wrong:** `mock.patch("lobster.core.data_manager_v2.H5ADBackend")` patches the name in the shim module, not in the actual module where `DataManagerV2` lives. Since `DataManagerV2.__init__` references `H5ADBackend` from its own module namespace (the new `lobster.core.runtime.data_manager`), patching the shim module has no effect.
**Why it happens:** `from ... import *` creates new bindings in the shim module, but `mock.patch` only patches the specified module's namespace. The actual class still sees the original binding in its own module.
**How to avoid:** Update ALL `mock.patch("lobster.core.data_manager_v2.X")` to `mock.patch("lobster.core.runtime.data_manager.X")`.
**Blast radius:** Only 2 test files affected:
- `tests/unit/core/test_data_manager_v2.py` (~108 patch references)
- `tests/unit/agents/test_agent_registry.py` (~2 patch references)
**Warning signs:** Tests pass but patches are silently no-ops (mock objects not injected).

### Pitfall 2: `__name__` and `__module__` Changes
**What goes wrong:** After the move, `DataManagerV2.__module__` becomes `lobster.core.runtime.data_manager` instead of `lobster.core.data_manager_v2`. Code that checks `__module__` for logging or error messages will show the new path.
**How to avoid:** This is expected and correct behavior. No action needed unless specific tests assert on module paths.
**Warning signs:** Log output changes (cosmetic, not functional).

### Pitfall 3: Missing `__all__` in Moved Module
**What goes wrong:** The `from ... import *` in the shim exports EVERYTHING from the moved module, including private helper functions and lazy-import globals. This is fine for backward compatibility but may expose unintended names.
**How to avoid:** Add `__all__` to the moved module listing only public API: `DataManagerV2`, `MetadataEntry`, `SuppressKaleidoLogging`. However, since the existing code has no `__all__` and consumers only import `DataManagerV2` and occasionally `MetadataEntry`, this is LOW risk.
**Recommendation:** Do NOT add `__all__` -- matches Phase 6 precedent where no moved files added `__all__`.

### Pitfall 4: Provenance `__init__.py` Lazy Shim Interaction
**What goes wrong:** `data_manager_v2.py` imports `from lobster.core.provenance import ProvenanceTracker`. The `lobster.core.provenance` package has a `__getattr__` lazy shim in `__init__.py`. When updating this import to `from lobster.core.provenance.provenance import ProvenanceTracker`, ensure the direct module path is used (bypassing the `__getattr__` shim).
**How to avoid:** Import from the concrete module path, not the package: `from lobster.core.provenance.provenance import ProvenanceTracker`.

## Code Examples

### Shim File (lobster/core/data_manager_v2.py after move)

```python
# Source: Phase 6 proven pattern (lobster/core/workspace.py)
"""Backward-compat shim. Use lobster.core.runtime.data_manager instead."""
import warnings as _w

_w.warn(
    "Import from 'lobster.core.runtime.data_manager' instead of "
    "'lobster.core.data_manager_v2'. Shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)
from lobster.core.runtime.data_manager import *  # noqa: F401,F403
```

### Scaffold Template Update (agent.py.j2 line 37)

```python
# BEFORE
from lobster.core.data_manager_v2 import DataManagerV2

# AFTER
from lobster.core.runtime.data_manager import DataManagerV2
```

Same change in `shared_tools.py.j2` line 24.

### mock.patch Update Pattern

```python
# BEFORE
@patch("lobster.core.data_manager_v2.H5ADBackend")
@patch("lobster.core.data_manager_v2.TranscriptomicsAdapter")
@patch("lobster.core.data_manager_v2.ProteomicsAdapter")

# AFTER
@patch("lobster.core.runtime.data_manager.H5ADBackend")
@patch("lobster.core.runtime.data_manager.TranscriptomicsAdapter")
@patch("lobster.core.runtime.data_manager.ProteomicsAdapter")
```

### Import-Linter Config Update (.importlinter line 50)

```ini
# BEFORE
[importlinter:contract:core-independence]
name = Core modules are loosely coupled
type = independence
modules =
    lobster.core.data_manager_v2
    lobster.core.component_registry
    lobster.core.queues.download_queue

# AFTER
[importlinter:contract:core-independence]
name = Core modules are loosely coupled
type = independence
modules =
    lobster.core.runtime.data_manager
    lobster.core.component_registry
    lobster.core.queues.download_queue
```

### CI Check for Old-Path Imports (DMGR-04)

```yaml
# In .github/workflows/ci-basic.yml or pr-validation-basic.yml
- name: Check for deprecated data_manager_v2 imports
  run: |
    # Exclude the shim file itself and existing test files
    VIOLATIONS=$(grep -rn "from lobster.core.data_manager_v2" --include="*.py" \
      --exclude="data_manager_v2.py" \
      --exclude-dir=".claude" --exclude-dir="build" --exclude-dir=".testing" \
      lobster/ packages/ | grep -v "# noqa: deprecated-import" || true)
    if [ -n "$VIOLATIONS" ]; then
      echo "::error::New code must import from lobster.core.runtime.data_manager, not lobster.core.data_manager_v2"
      echo "$VIOLATIONS"
      exit 1
    fi
```

**Alternative (import-linter forbidden contract):**

```ini
[importlinter:contract:no-deprecated-data-manager-import]
name = New code must not import from deprecated data_manager_v2 path
type = forbidden
source_modules =
    lobster.core.runtime
    lobster.agents
    lobster.services
    lobster.tools
forbidden_modules =
    lobster.core.data_manager_v2
ignore_imports =
    lobster.core.data_manager_v2 -> lobster.core.runtime.data_manager
```

**Recommendation:** Use the grep-based CI check. Import-linter forbidden contracts check runtime import chains, not source text, so they may not catch `from lobster.core.data_manager_v2 import X` in files that are not yet imported by the linter root. The grep approach is simpler and catches all source files.

## Shim Test Updates

The existing `test_core_subpackage_shims.py` parametrized test should be extended with the new shim pair:

```python
# Add to SHIM_PAIRS list
("lobster.core.data_manager_v2", "lobster.core.runtime.data_manager", "DataManagerV2"),
```

This gives both `test_shim_reexports_and_warns` and `test_isinstance_identity` coverage automatically.

## Importer Census

### Source Files (80 total, NOT updated -- shim handles them)

| Category | Count | Examples |
|----------|-------|---------|
| tools/ | 14 | download_orchestrator.py, workspace_tool.py, all 11 providers |
| services/ | 14 | geo_service.py, content_access_service.py, modality_management_service.py |
| packages/ agents | 39 | All 10 agent packages (transcriptomics, research, genomics, etc.) |
| core/ | 6 | client.py, notebooks/executor.py, interfaces/*.py |
| agents/ (core) | 2 | supervisor.py, graph.py |
| other | 5 | cli.py, scaffold templates, kevin_notes |

### Test Files (120 total, shim handles imports but mock.patch needs updating in 2 files)

| Category | Files | mock.patch Refs |
|----------|-------|-----------------|
| tests/unit/core/test_data_manager_v2.py | 1 | ~108 |
| tests/unit/agents/test_agent_registry.py | 1 | ~2 |
| All other test files | 118 | 0 (only `from ... import` -- shim handles) |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `pytest tests/unit/core/test_core_subpackage_shims.py tests/unit/core/test_data_manager_v2.py -x -q` |
| Full suite command | `pytest tests/ -x --timeout=60` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DMGR-01 | `from lobster.core.runtime.data_manager import DataManagerV2` works | unit | `pytest tests/unit/core/test_core_subpackage_shims.py -k data_manager -x` | Partial (extend existing parametrized test) |
| DMGR-02 | Old path shim re-exports + warns, identity match, mock.patch works | unit | `pytest tests/unit/core/test_core_subpackage_shims.py -k data_manager -x && pytest tests/unit/core/test_data_manager_v2.py -x` | Partial (shim test needs new pair; dm test needs patch path updates) |
| DMGR-03 | Scaffold templates use new import path | unit | `grep -q "lobster.core.runtime.data_manager" lobster/scaffold/templates/agent.py.j2` | No (manual grep or new test) |
| DMGR-04 | CI blocks old-path imports in new code | integration | CI workflow check | No (new CI step needed) |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/core/test_core_subpackage_shims.py tests/unit/core/test_data_manager_v2.py -x -q`
- **Per wave merge:** `pytest tests/ -x --timeout=60`
- **Phase gate:** Full suite green before verify

### Wave 0 Gaps
- [ ] Extend `SHIM_PAIRS` in `test_core_subpackage_shims.py` with data_manager_v2 pair
- [ ] Update mock.patch strings in `test_data_manager_v2.py` (108 references)
- [ ] Update mock.patch strings in `test_agent_registry.py` (2 references)
- [ ] Add CI check step for old-path import prevention

## Open Questions

1. **Should existing 200 importers be updated to new path?**
   - What we know: Phase 6 did NOT update existing importers (shim handles them). SHIM-01 in v2 requirements covers retirement.
   - Recommendation: Do NOT update existing importers. Follow Phase 6 precedent. Shim handles backward compat.

2. **File rename: data_manager_v2.py -> data_manager.py?**
   - What we know: Phase 6 renamed notebook_executor.py -> executor.py on move. The `_v2` suffix is historical (there was a v1 that no longer exists).
   - Recommendation: Rename to `data_manager.py`. The `_v2` suffix adds no value in the new location.

3. **Should we also update data_manager_v2.py internal imports to canonical paths?**
   - What we know: The file imports from 4 old paths that now have shims. This causes unnecessary DeprecationWarning noise when the module loads.
   - Recommendation: YES, update internal imports. This silences spurious warnings and models good practice.

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection of `lobster/core/data_manager_v2.py` (3,999 LOC, imports, exports)
- Phase 6 research and shim pattern from `lobster/core/workspace.py`, `lobster/core/provenance/__init__.py`
- Existing test patterns from `tests/unit/core/test_core_subpackage_shims.py`
- Import-linter config from `.importlinter`
- Scaffold templates from `lobster/scaffold/templates/*.j2`

### Secondary (MEDIUM confidence)
- Python mock.patch documentation: patches the name in the specified module, not at the call site

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - identical to Phase 6, no new libraries
- Architecture: HIGH - follows proven Phase 6 shim pattern exactly
- Pitfalls: HIGH - mock.patch behavior is well-documented Python semantics; verified by inspecting actual test code
- CI enforcement: MEDIUM - grep-based approach is pragmatic but not import-linter level; sufficient for the requirement

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable -- pure mechanical restructuring, no external dependencies)
