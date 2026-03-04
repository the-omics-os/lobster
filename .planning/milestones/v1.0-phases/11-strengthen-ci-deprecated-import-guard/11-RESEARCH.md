# Phase 11: Strengthen CI Deprecated-Import Guard - Research

**Researched:** 2026-03-04
**Domain:** CI enforcement, import migration, shell scripting
**Confidence:** HIGH

## Summary

Phase 11 is a mechanical migration + CI hardening task with no ambiguity. The deprecated import path (`lobster.core.data_manager_v2`) was established as a shim in Phase 07; the CI guard was added at the same time but intentionally left with `|| true` to allow pre-existing violations to pass. Phase 11 closes that gap by migrating all 39 violations in `packages/` to the canonical path (`lobster.core.runtime.data_manager`) and then removing the `|| true` bypass from the CI step.

The work splits cleanly into two concerns: (1) a sed-style bulk import replacement across 10 packages, and (2) a one-line CI yaml edit. No logic changes, no new tests required — the existing shim ensures all imports continue to resolve identically before and after migration. The only structural test worth adding is a pytest-based canonical-import check (mirroring the geo decomposition test pattern already in the codebase) to give local developers a fast feedback loop.

**Primary recommendation:** Migrate all 39 violations with a single-pass sed replacement per file (or bulk find+sed), verify tests still pass, then remove `|| true` from ci-basic.yml. Done in one plan, one wave.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DMGR-04 | New code cannot import from old path (CI/lint check) | CI guard exists but has || true bypass; 39 pre-existing violations in packages/ must be migrated before the guard can be made strict |
</phase_requirements>

## Standard Stack

### Core
| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| `sed` / Python string replace | stdlib | Bulk import path substitution | Fastest for mechanical single-line replacements |
| `grep -rn` | system | Verification that no violations remain | Already used by CI step |
| GitHub Actions bash | n/a | CI yaml step modification | In-place edit, no new tooling |

### No New Dependencies

This phase introduces zero new libraries. All work is:
- File edits (sed-style import replacement)
- CI yaml edit (remove one `|| true`)
- Optional: one new pytest structural test

**Installation:** None required.

## Architecture Patterns

### Current State (What Exists)

```
.github/workflows/ci-basic.yml  — deprecated-import guard (has || true bypass)
lobster/core/data_manager_v2.py — backward-compat shim (re-exports via wildcard from canonical)
lobster/core/runtime/data_manager.py — canonical location (DataManagerV2 class lives here)
lobster/scaffold/templates/agent.py.j2 — already uses canonical path
lobster/scaffold/templates/shared_tools.py.j2 — already uses canonical path
packages/                        — 39 files still importing from deprecated path
```

### Target State (After Phase 11)

```
.github/workflows/ci-basic.yml  — deprecated-import guard (NO || true)
packages/                        — 0 files importing from deprecated path
```

### Canonical Import (The Replacement)

Every occurrence of:
```python
from lobster.core.data_manager_v2 import DataManagerV2
```

Must become:
```python
from lobster.core.runtime.data_manager import DataManagerV2
```

The shim at `lobster/core/data_manager_v2.py` performs a `from lobster.core.runtime.data_manager import *` wildcard re-export, so both paths are functionally identical during and after migration.

### Pattern: Bulk Sed Replacement

```bash
# Find all 39 files and replace in-place
grep -rln "from lobster\.core\.data_manager_v2 import" \
  --include="*.py" \
  packages/ | xargs sed -i 's/from lobster\.core\.data_manager_v2 import/from lobster.core.runtime.data_manager import/g'
```

**Verification after replacement:**
```bash
grep -rn "from lobster.core.data_manager_v2 import" packages/ --include="*.py"
# Should return zero results
```

### Pattern: CI Step Hardening

Remove `|| true` from the grep subshell in `.github/workflows/ci-basic.yml`:

**Before (lines 50-55):**
```yaml
        VIOLATIONS=$(grep -rn "from lobster\.core\.data_manager_v2 import" \
          --include="*.py" \
          --exclude="data_manager_v2.py" \
          --exclude-dir=".claude" --exclude-dir="build" --exclude-dir=".testing" \
          --exclude-dir="kevin_notes" --exclude-dir="skills" \
          lobster/scaffold/ packages/ || true)
```

**After:**
```yaml
        VIOLATIONS=$(grep -rn "from lobster\.core\.data_manager_v2 import" \
          --include="*.py" \
          --exclude="data_manager_v2.py" \
          --exclude-dir=".claude" --exclude-dir="build" --exclude-dir=".testing" \
          --exclude-dir="kevin_notes" --exclude-dir="skills" \
          lobster/scaffold/ packages/ 2>/dev/null || true)
```

Wait — this needs careful reasoning. See **Pitfall 1** below. The correct fix is:

```yaml
        if grep -rn "from lobster\.core\.data_manager_v2 import" \
          --include="*.py" \
          --exclude="data_manager_v2.py" \
          --exclude-dir=".claude" --exclude-dir="build" --exclude-dir=".testing" \
          --exclude-dir="kevin_notes" --exclude-dir="skills" \
          lobster/scaffold/ packages/; then
          echo "::error::New code must import from lobster.core.runtime.data_manager, not lobster.core.data_manager_v2"
          exit 1
        fi
        echo "No deprecated data_manager_v2 imports found in scaffold/packages"
```

Using `if grep` is idiomatic: grep exits 0 (found) = fail CI; grep exits 1 (not found) = pass. This eliminates the VIOLATIONS variable entirely and makes intent explicit.

### Pattern: Structural Test (Optional but Recommended)

Mirror the `test_geo_decomposition.py` grep-based structural test:

```python
# tests/unit/core/test_canonical_imports.py (or add to test_core_subpackage_shims.py)
import subprocess

class TestCanonicalImportEnforcement:
    def test_no_deprecated_data_manager_imports_in_packages(self):
        """packages/ must not import from deprecated lobster.core.data_manager_v2."""
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py",
             "--exclude=data_manager_v2.py",
             "from lobster.core.data_manager_v2 import",
             "packages/"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 1, (  # 1 = no matches = clean
            f"Deprecated imports found in packages/:\n{result.stdout}"
        )
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bulk import replacement | Custom Python AST rewriter | sed -i in-place | Single-line pattern, no AST needed — string replacement is exact |
| CI guard | A new lint plugin/ruff rule | grep-based step already in ci-basic.yml | Pattern already established, removing || true is sufficient |
| Verification | Custom test runner | pytest + subprocess.run grep | Existing pattern in test_geo_decomposition.py |

## Violation Inventory

All 39 violations are in `packages/`, split by package:

| Package | Source Files | Test Files | Total |
|---------|-------------|------------|-------|
| lobster-metadata | 3 | 6 | 9 |
| lobster-drug-discovery | 7 | 1 | 8 |
| lobster-transcriptomics | 4 | 0 | 4 |
| lobster-proteomics | 4 | 0 | 4 |
| lobster-ml | 4 | 0 | 4 |
| lobster-research | 2 | 1 | 3 |
| lobster-structural-viz | 1 | 1 | 2 |
| lobster-metabolomics | 2 | 0 | 2 |
| lobster-genomics | 2 | 0 | 2 |
| lobster-visualization | 1 | 0 | 1 |
| **Total** | **30** | **9** | **39** |

**Note:** All 39 violations are `DataManagerV2` imports only. No other symbols from `data_manager_v2` are imported across packages/. Confirmed by reviewing violation list — every line is `from lobster.core.data_manager_v2 import DataManagerV2`.

**Also note:** `lobster/tests/unit/` (core tests, NOT in packages/) has ~10 additional deprecated imports but they are EXCLUDED from the CI grep scope by design (they import via shim, Phase 07 decision). These are out of scope for Phase 11.

## Common Pitfalls

### Pitfall 1: Misunderstanding the `|| true` Semantics

**What goes wrong:** The `|| true` is inside the `$()` subshell: `VIOLATIONS=$(grep ... || true)`. This does NOT mean "ignore all errors". It means: if grep exits nonzero (which it does when it finds NO matches = exit 1), use `true` exit code so the subshell doesn't fail. The VIOLATIONS check `if [ -n "$VIOLATIONS" ]` still correctly fails when matches ARE found.

**Why it matters:** The guard IS currently blocking CI when violations exist (39 violations = grep exits 0 = VIOLATIONS non-empty = exit 1). The `|| true` is only problematic if: (a) packages/ doesn't exist and grep exits 2, or (b) we want to prove the guard can never be accidentally bypassed.

**How to avoid:** After migrating all violations, rewrite the guard using `if grep` pattern (see Architecture Patterns above) — cleaner semantics, no variable needed, impossible to accidentally bypass.

**Warning signs:** Using `|| true` after grep when you want strict enforcement is an antipattern. Always use `if grep; then fail; fi` for CI enforcement steps.

### Pitfall 2: Forgetting the `--exclude="data_manager_v2.py"` Guard

**What goes wrong:** The shim file itself (`lobster/core/data_manager_v2.py`) contains `from lobster.core.runtime.data_manager import *` — if you remove the `--exclude` flag, it would never trigger (correct), but if you forget it when the shim still has internal references, grep might match the shim itself.

**How to avoid:** Keep `--exclude="data_manager_v2.py"` in the CI grep. The shim file is intentionally excluded because it IS the deprecated path (it's the target, not a violator).

### Pitfall 3: Missing the One Test File with Inline Import

**What goes wrong:** One test file uses an inline import inside a function:
```python
# packages/lobster-research/tests/agents/test_data_expert_tools.py:13
        from lobster.core.data_manager_v2 import DataManagerV2  # inside function body
```

**How to avoid:** The bulk sed command handles this correctly since sed operates on line content, not scope. Verify this file is in the violation list and confirm the replacement is correct after migration.

### Pitfall 4: Assuming VIOLATIONS Variable is Always Right

**What goes wrong:** If someone adds `packages/` to `.gitignore` or changes directory structure, the grep might exit with code 2 (error), which `|| true` would suppress, making VIOLATIONS empty even though the check is broken.

**How to avoid:** Use `if grep` pattern instead (see Architecture Patterns). grep exit 1 = no matches (pass), grep exit 0 = matches found (fail CI), grep exit 2 = error (also fail CI since no `|| true`).

## Code Examples

### Exact Replacement Pattern (verified against all 39 violations)

All 39 violations follow exactly this pattern:
```python
from lobster.core.data_manager_v2 import DataManagerV2
```

Replacement:
```python
from lobster.core.runtime.data_manager import DataManagerV2
```

The `DataManagerV2` class is defined in `lobster/core/runtime/data_manager.py` at class line 205. The shim performs `from lobster.core.runtime.data_manager import *` so both paths resolve to the same class object. Migration is purely cosmetic — zero behavior change.

### Canonical Location Confirmation

```python
# lobster/core/runtime/data_manager.py (line 205)
class DataManagerV2:
    """DataManagerV2: Modular orchestration layer for multi-omics data management."""
    ...
```

```python
# lobster/core/data_manager_v2.py (shim — 11 lines)
"""Backward-compat shim. Use lobster.core.runtime.data_manager instead."""
import warnings as _w
_w.warn(
    "Import from 'lobster.core.runtime.data_manager' instead of "
    "'lobster.core.data_manager_v2'. Shim will be removed in v2.0.0.",
    DeprecationWarning, stacklevel=2,
)
from lobster.core.runtime.data_manager import *  # noqa: F401,F403
```

### CI Step After Hardening (Complete)

```yaml
    - name: Check for deprecated data_manager_v2 imports
      run: |
        # Fail if any file in scaffold/ or packages/ imports from deprecated path
        # grep exit 0 = violations found (bad), exit 1 = no violations (good)
        if grep -rn "from lobster\.core\.data_manager_v2 import" \
          --include="*.py" \
          --exclude="data_manager_v2.py" \
          --exclude-dir=".claude" --exclude-dir="build" --exclude-dir=".testing" \
          --exclude-dir="kevin_notes" --exclude-dir="skills" \
          lobster/scaffold/ packages/; then
          echo "::error::New code must import from lobster.core.runtime.data_manager, not lobster.core.data_manager_v2"
          exit 1
        fi
        echo "No deprecated data_manager_v2 imports found in scaffold/packages"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `VIOLATIONS=$(grep ... \|\| true)` pattern | `if grep; then fail; fi` pattern | Phase 11 | Cleaner semantics, grep error (exit 2) now also fails CI |
| `from lobster.core.data_manager_v2 import DataManagerV2` | `from lobster.core.runtime.data_manager import DataManagerV2` | Phase 07 created canonical, Phase 11 migrates packages/ | No behavior change — shim still works |

**Not deprecated (keep as-is):**
- The shim at `lobster/core/data_manager_v2.py` — stays until v2.0.0 (SHIM-01 is deferred)
- Test files in `lobster/tests/unit/` (core, not packages/) — excluded from CI scope, out of scope for Phase 11

## Open Questions

1. **Should tests in packages/ also be migrated?**
   - What we know: 9 of 39 violations are in `packages/*/tests/` — these ARE in the CI grep scope (packages/ is the scope, tests/ subdirs are included)
   - What's unclear: Phase 07 decision said "existing importers handled by shim" but only scoped to lobster/ core tests; packages/ tests are explicitly in scope
   - Recommendation: Yes, migrate all 39 including test files. The canonical path works identically, and tests should model correct import patterns.

2. **One structural test or two?**
   - What we know: No existing pytest test checks for deprecated imports; geo decomposition tests use subprocess.run + grep pattern
   - What's unclear: Whether to add a new test file or extend test_core_subpackage_shims.py
   - Recommendation: Add to `tests/unit/core/test_core_subpackage_shims.py` (already handles import-path structural tests) with one test method `test_no_deprecated_data_manager_imports_in_packages`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/unit/core/test_core_subpackage_shims.py -v` |
| Full suite command | `python -m pytest tests/unit/core/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DMGR-04 | No deprecated imports in packages/ | structural | `python -m pytest tests/unit/core/test_core_subpackage_shims.py::TestCoreSubpackageShims::test_no_deprecated_data_manager_imports_in_packages -v` | Wave 0 (add to existing file) |
| DMGR-04 | CI step fails on violation (no `\|\| true`) | manual/CI | `grep -rn "from lobster.core.data_manager_v2 import" packages/ --include="*.py"; echo $?` should be 1 | n/a (CI verification) |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/unit/core/test_core_subpackage_shims.py -v`
- **Per wave merge:** `python -m pytest tests/unit/core/ -v`
- **Phase gate:** `grep -rn "from lobster.core.data_manager_v2 import" packages/ --include="*.py"` returns exit 1 (zero matches)

### Wave 0 Gaps
- [ ] `tests/unit/core/test_core_subpackage_shims.py` — add `test_no_deprecated_data_manager_imports_in_packages` method (file exists, method is new)

## Sources

### Primary (HIGH confidence)
- Direct file inspection: `/Users/tyo/Omics-OS/lobster/.github/workflows/ci-basic.yml` — CI step exact content verified
- Direct file inspection: `lobster/core/data_manager_v2.py` — shim content verified
- Direct file inspection: `lobster/core/runtime/data_manager.py` — canonical location verified
- Direct grep: 39 violations enumerated, package-by-package breakdown verified
- `tests/unit/services/data_access/test_geo_decomposition.py:181` — `subprocess.run(["grep", ...])` structural test pattern verified

### Secondary (MEDIUM confidence)
- `.planning/STATE.md` Phase 07 decisions — explains `|| true` was intentional for bootstrapping
- `.planning/v1.0-MILESTONE-AUDIT.md` — confirms 39 violations, classifies gap as MEDIUM severity
- `07-02-SUMMARY.md` — confirms Phase 07 scope was "new code only", existing packages/ deferred to Phase 11

## Metadata

**Confidence breakdown:**
- Violation inventory: HIGH — direct grep on filesystem, exact 39 count verified
- Canonical path: HIGH — both shim and target file inspected directly
- CI mechanics: HIGH — yaml file read directly, `|| true` semantics analyzed
- Structural test pattern: HIGH — existing `test_geo_decomposition.py` pattern confirmed

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable — no external dependencies, pure codebase migration)
