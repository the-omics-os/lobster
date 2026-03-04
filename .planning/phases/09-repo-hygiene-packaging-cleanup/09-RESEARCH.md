# Phase 9: Repo Hygiene & Packaging Cleanup - Research

**Researched:** 2026-03-04
**Domain:** Repository maintenance, .gitignore normalization, Makefile clean targets, artifact removal
**Confidence:** HIGH

## Summary

Phase 9 is a pure mechanical cleanup phase -- no behavioral changes, no new features. The repo has accumulated stale build artifacts (MagicMock/, dist/, .ruff_cache/, *.egg-info/ across 10 package dirs), empty placeholder directories from early GEO decomposition planning, deprecated shim files that are nearly unused, and a .gitignore that is structurally disordered (comments sorted alphabetically but patterns are not grouped by purpose).

The current `make clean` target only handles root-level artifacts. It does NOT descend into `packages/*/` to clean dist/, .ruff_cache/, or *.egg-info/ directories. The `make clean-all` target only adds workspace cleanup. Neither target handles MagicMock/ (test artifact leak from pytest mocks creating real directories).

**Primary recommendation:** Single-plan phase. Normalize .gitignore with clear grouped sections, expand Makefile clean targets, remove empty dirs and deprecated shims, delete stale on-disk artifacts.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| HYGN-01 | .gitignore sections normalized | Current file has 61 comment lines acting as section headers, many single-line "sections". Needs consolidation into ~10 logical groups. |
| HYGN-02 | `make clean` and `make clean-all` expanded for package-local artifacts | Current clean only handles root build/dist/*.egg-info. Must add `packages/*/dist/`, `packages/*/*.egg-info/`, `packages/*/.ruff_cache/`, root `MagicMock/` |
| HYGN-03 | Empty placeholder dirs removed (those not populated by GEO decomposition) | 11 empty dirs in lobster/, 1 in tests/. GEO decomposition populated 5 real modules; 3 empty geo subdirs (metadata/, sample/, utils/) were never used. |
| HYGN-04 | Deprecated shim files cleaned (geo_parser.py, geo_downloader.py) -- verified unused first | geo_parser.py: ZERO importers. geo_downloader.py: 2 importers (1 integration test, 1 unit test). Both must be updated before removal. |
| HYGN-05 | Stale build artifacts removed from package dirs | Found: 8 packages with dist/, 8 with .ruff_cache/, 10 with *.egg-info/, root MagicMock/ with leaked mock paths, root build/ dir |
</phase_requirements>

## Standard Stack

Not applicable -- this phase uses only git, make, and shell commands. No new libraries needed.

## Architecture Patterns

### .gitignore Normalization Pattern

**Current state:** 265 lines, 61 comment headers, 196 patterns, 2 negation rules. Comments are alphabetically sorted but this creates fragmented sections (e.g., "# Byte-compiled" followed by "# C extensions" as separate sections). Many comments are copy-pasted from GitHub's Python template and are not relevant (Django, Flask, Celery, Scrapy, etc.).

**Target structure:** Group into ~10 logical sections:
```
# === Python ===
# Byte-compiled, caches, packaging

# === Build Artifacts ===
# dist, eggs, wheels, egg-info

# === IDE & Editor ===
# .idea, .vscode, .DS_Store

# === Testing ===
# pytest, coverage, htmlcov

# === Environment ===
# .venv, .env, secrets

# === Lobster-Specific ===
# workspace, cache, sessions, model files

# === Private/Premium Packages ===
# packages that are not public yet

# === Data Files ===
# scientific data, downloads, exports

# === OS Files ===
# macOS, Windows, Linux artifacts

# === Negation Rules ===
# Explicit inclusions (!pattern)
```

### Makefile Clean Pattern

**Current `clean` target (lines 734-741):**
```makefile
clean:
    rm -rf build dist *.egg-info
    rm -rf .pytest_cache .coverage htmlcov
    rm -rf .mypy_cache .ruff_cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
```

**Missing from clean:**
- `packages/*/dist/`
- `packages/*/*.egg-info/` (naming varies: lobster_transcriptomics.egg-info, etc.)
- `packages/*/.ruff_cache/`
- `MagicMock/` (test mock directory leak)
- `coverage.xml` (listed in addopts)

**Missing from clean-all:**
- `test_output/`
- `build/` is in clean but `build/bdist.*` persists

### Deprecated Shim Removal Pattern

**Step 1: Verify importers**
- `geo_parser.py`: 0 importers (safe to remove)
- `geo_downloader.py`: 2 importers that must be updated:
  - `tests/integration/test_gse248556_bug_fixes.py:482` -- `from lobster.tools.geo_downloader import GEODownloadManager`
  - `tests/unit/tools/test_geo_downloader.py:21` -- `from lobster.tools.geo_downloader import GEODownloadManager`

**Step 2: Update imports to canonical path**
```python
# Old (deprecated):
from lobster.tools.geo_downloader import GEODownloadManager
# New (canonical):
from lobster.services.data_access.geo.downloader import GEODownloadManager
```

**Step 3: Remove shim files and git-rm them**

### Empty Directory Inventory

| Directory | Tracked in git? | GEO decomposition? | Action |
|-----------|----------------|--------------------|---------|
| `lobster/.benchmarks/` | No | No | Remove |
| `lobster/data/cache/` | No | No | Remove |
| `lobster/services/visualization/` | No | No | Remove |
| `lobster/services/quality/` | No | No | Remove |
| `lobster/services/data_access/geo/utils/` | No | No | Remove |
| `lobster/services/data_access/geo/sample/` | No | No | Remove |
| `lobster/services/data_access/geo/metadata/` | No | No | Remove |
| `lobster/services/ml/` | No | No | Remove |
| `lobster/services/templates/` | No | No | Remove |
| `lobster/services/metadata/protocol_extraction/mass_spec/resources/` | No | No | Remove |
| `lobster/services/metadata/protocol_extraction/rnaseq/resources/` | No | No | Remove |
| `tests/resilience/` | No | No | Remove |
| `test_output/` | No | No | Remove |

None of these are tracked in git. None are from GEO decomposition (Phase 4 populated `archive_processing.py`, `concatenation.py`, `download_execution.py`, `matrix_parsing.py`, `metadata_fetch.py`, `helpers.py`, `soft_download.py`). All are safe to remove.

### Anti-Patterns to Avoid
- **Removing .gitkeep files that serve a purpose:** `data/exports/.gitkeep` is explicitly negated in .gitignore -- preserve it
- **Breaking negation rules:** The `!data/exports/.gitkeep` and `!skills/**/MANIFEST` negation patterns must stay and be placed AFTER their covering rules
- **Removing directories that look empty but have untracked content:** Always verify with `git ls-tree` before claiming empty

## Don't Hand-Roll

Not applicable -- this phase is pure file manipulation with git and make.

## Common Pitfalls

### Pitfall 1: Negation Rule Ordering in .gitignore
**What goes wrong:** Moving negation rules (`!pattern`) before the pattern they negate causes them to have no effect.
**Why it happens:** .gitignore processes rules top-to-bottom. A negation must come AFTER the matching ignore rule.
**How to avoid:** Keep `!data/exports/.gitkeep` after `data/*` and `!skills/**/MANIFEST` after `MANIFEST`.

### Pitfall 2: MagicMock Directory Reappearing
**What goes wrong:** Deleting MagicMock/ but it comes back after running tests.
**Why it happens:** Some test mocks create `MagicMock()/` paths when `workspace_path` is a MagicMock object and `os.makedirs()` is called on it.
**How to avoid:** Add `MagicMock/` to .gitignore AND to `make clean`. The root cause is in test fixtures but fixing that is out of scope for this phase.

### Pitfall 3: Removing Shim Before Updating Importers
**What goes wrong:** `git rm lobster/tools/geo_downloader.py` then tests fail.
**Why it happens:** 2 test files still import from old path.
**How to avoid:** Update imports FIRST, verify tests pass, THEN remove shim.

## Code Examples

### Expanded Makefile clean target
```makefile
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build dist *.egg-info lobster_ai.egg-info
	rm -rf .pytest_cache .coverage htmlcov coverage.xml
	rm -rf .mypy_cache .ruff_cache
	rm -rf MagicMock test_output
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	# Package-local artifacts
	find packages -maxdepth 2 -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find packages -maxdepth 2 -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	find packages -maxdepth 2 -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
```

### Import update for geo_downloader consumers
```python
# tests/unit/tools/test_geo_downloader.py
# Old:
from lobster.tools.geo_downloader import GEODownloadManager
# New:
from lobster.services.data_access.geo.downloader import GEODownloadManager
```

## State of the Art

Not applicable -- standard repository maintenance patterns.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via pyproject.toml) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `pytest tests/unit/ -x --no-cov -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HYGN-01 | .gitignore has clear section headers, no redundant patterns | manual-only | Visual inspection of .gitignore | N/A |
| HYGN-02 | `make clean` removes package-local artifacts | smoke | `make clean && find packages -name dist -o -name '*.egg-info' -o -name '.ruff_cache' \| wc -l` (expect 0) | N/A |
| HYGN-03 | No empty placeholder dirs remain | smoke | `find lobster tests -type d -empty -not -path '*__pycache__*' \| wc -l` (expect 0) | N/A |
| HYGN-04 | geo_parser.py and geo_downloader.py shims removed, imports updated | unit | `pytest tests/unit/tools/test_geo_downloader.py -x --no-cov` | Yes |
| HYGN-05 | No stale build artifacts in package dirs | smoke | `find packages -name dist -o -name '*.egg-info' -o -name '.ruff_cache' \| wc -l` (expect 0 after clean) | N/A |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/ -x --no-cov -q` (verify nothing broken by import changes)
- **Per wave merge:** `pytest tests/ -v --no-cov` (full suite)
- **Phase gate:** Full suite green + manual verification of .gitignore structure

### Wave 0 Gaps
None -- existing test infrastructure covers the only testable requirement (HYGN-04 import updates). Other requirements are verified by shell commands, not unit tests.

## Sources

### Primary (HIGH confidence)
- Direct filesystem inspection via `find`, `git ls-files`, `git ls-tree`
- Current `.gitignore` (265 lines, read in full)
- Current `Makefile` (827 lines, read in full)
- `lobster/tools/geo_parser.py` and `lobster/tools/geo_downloader.py` (shim files, read in full)
- `grep` across all `.py` files for import verification

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no libraries involved, pure file operations
- Architecture: HIGH - direct inspection of all affected files
- Pitfalls: HIGH - verified all import paths and git tracking status

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable -- repo structure changes rarely)
