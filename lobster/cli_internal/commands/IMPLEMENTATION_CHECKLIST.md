# CLI Refactoring Implementation Checklist
## Step-by-Step Execution Plan

**Goal**: Split commands into light/heavy for 24x faster startup
**Estimated Time**: 8 hours
**Risk Level**: LOW (with proper testing)

---

## PRE-FLIGHT CHECKS

### Before Starting
- [ ] All tests currently passing
- [ ] No uncommitted changes in `cli_internal/commands/`
- [ ] Backup current state: `git tag pre-cli-refactor`
- [ ] Create feature branch: `git checkout -b refactor/cli-commands-light-heavy`
- [ ] Read `CLI_REFACTORING_ANALYSIS.md` in full

### Environment Check
```bash
# Verify current import performance (baseline)
$ time python3 -c "from lobster.cli_internal.commands import show_queue_status"
# Should see ~2s

# Verify tests pass
$ cd /Users/tyo/GITHUB/omics-os/lobster
$ pytest tests/test_cli_commands_imports.py -v || echo "Tests don't exist yet - OK"
$ pytest tests/ -k "command" -v
```

---

## PHASE 1: PREPARATION (NO CODE CHANGES)

### Task 1.1: Create Test Baseline
- [ ] Create `tests/test_cli_commands_imports.py`:
  ```python
  import sys
  import time
  import pytest

  def test_current_import_time():
      """Baseline: measure current import time."""
      start = time.time()
      from lobster.cli_internal.commands import show_queue_status
      elapsed = time.time() - start
      print(f"\nBaseline import time: {elapsed:.3f}s")

  def test_numpy_not_loaded_after_light_import():
      """This WILL FAIL before refactoring - documents current state."""
      # Clear modules
      for m in list(sys.modules.keys()):
          if 'lobster' in m or 'numpy' in m or 'pandas' in m:
              del sys.modules[m]

      from lobster.cli_internal.commands import show_queue_status
      assert 'numpy' not in sys.modules, "EXPECTED FAILURE: numpy loaded (will fix in refactor)"
  ```

- [ ] Run baseline tests:
  ```bash
  pytest tests/test_cli_commands_imports.py -v -s
  ```
- [ ] Document baseline performance in git commit message

### Task 1.2: Identify All Import Chains
- [x] List all files importing from `cli_internal.commands` ✅ (cli.py, analysis_screen.py)
- [x] Document heavy import chain ✅ (data_manager_v2 → numpy/pandas)
- [ ] Check if dashboard screens import heavy commands eagerly
  ```bash
  grep -r "from lobster.cli_internal.commands import" /Users/tyo/GITHUB/omics-os/lobster/lobster/ui/
  ```

### Task 1.3: Review Command Files
- [x] Verify no module-level heavy imports ✅
- [x] Verify TYPE_CHECKING used correctly ✅
- [ ] Check for module-level side effects:
  ```bash
  # Look for module-level code that executes on import
  for f in /Users/tyo/GITHUB/omics-os/lobster/lobster/cli_internal/commands/*.py; do
      echo "=== $(basename $f) ==="
      grep -E "^[A-Z_]+\s*=" "$f" | head -5
  done
  ```

---

## PHASE 2: CREATE DIRECTORY STRUCTURE

### Task 2.1: Create Directories
```bash
cd /Users/tyo/GITHUB/omics-os/lobster/lobster/cli_internal/commands
mkdir -p light
mkdir -p heavy
```

### Task 2.2: Create Initial __init__.py Files
- [ ] Create `light/__init__.py`:
  ```python
  """
  Light commands: fast operations without heavy data dependencies.

  These commands import in <100ms and do not require numpy/pandas/anndata.
  Suitable for: config, queue, metadata, workspace listing.
  """
  # Empty initially - files moved in Phase 3
  ```

- [ ] Create `heavy/__init__.py`:
  ```python
  """
  Heavy commands: data-intensive operations requiring numpy/pandas/anndata.

  These commands trigger ~2s import penalty but provide full data access.
  Suitable for: data summary, modality inspection, visualization.
  """
  # Empty initially - files moved in Phase 3
  ```

- [ ] Commit structure:
  ```bash
  git add light/ heavy/
  git commit -m "refactor(cli): create light/heavy command directories (no files moved yet)"
  ```

---

## PHASE 3: MOVE FILES (ONE AT A TIME)

**CRITICAL**: Test after EACH move. If any test fails, stop and rollback before proceeding.

### Step 3.1: Move Pure Light Commands

#### Move queue_commands.py
```bash
cd /Users/tyo/GITHUB/omics-os/lobster/lobster/cli_internal/commands

# 1. Move file
git mv queue_commands.py light/

# 2. Update __init__.py (add this import)
# from lobster.cli_internal.commands.light.queue_commands import (
#     show_queue_status,
#     queue_load_file,
#     queue_list,
#     queue_clear,
#     queue_export,
#     QueueFileTypeNotSupported,
# )

# 3. Test immediately
pytest tests/ -k "queue" -v
python3 -c "from lobster.cli_internal.commands import show_queue_status; print('✓ Import works')"

# 4. Commit
git commit -am "refactor(cli): move queue_commands to light/ (no data deps)"
```

- [ ] queue_commands.py moved and tested ✅
- [ ] Imports still work from top-level
- [ ] All queue tests pass

#### Move config_commands.py
```bash
# Same pattern as above
git mv config_commands.py light/
# Update __init__.py
pytest tests/ -k "config" -v
git commit -am "refactor(cli): move config_commands to light/ (no data deps)"
```

- [ ] config_commands.py moved and tested ✅
- [ ] Imports still work
- [ ] All config tests pass

#### Move metadata_commands.py
```bash
git mv metadata_commands.py light/
# Update __init__.py
pytest tests/ -k "metadata" -v
git commit -am "refactor(cli): move metadata_commands to light/ (no data deps)"
```

- [ ] metadata_commands.py moved and tested ✅

### Step 3.2: Move Pure Heavy Commands

#### Move data_commands.py
```bash
git mv data_commands.py heavy/
# Update __init__.py with LAZY IMPORT:
# def __getattr__(name):
#     if name == "data_summary":
#         from lobster.cli_internal.commands.heavy.data_commands import data_summary
#         return data_summary
#     raise AttributeError(f"...")

pytest tests/ -k "data" -v
git commit -am "refactor(cli): move data_commands to heavy/ (numpy/pandas deps)"
```

- [ ] data_commands.py moved and tested ✅
- [ ] Lazy import works
- [ ] numpy NOT in sys.modules after importing __init__

#### Move modality_commands.py
```bash
git mv modality_commands.py heavy/
# Update __init__.py with LAZY IMPORT
pytest tests/ -k "modality" -v
git commit -am "refactor(cli): move modality_commands to heavy/ (scipy/numpy deps)"
```

- [ ] modality_commands.py moved and tested ✅

#### Move visualization_commands.py
```bash
git mv visualization_commands.py heavy/
# Update __init__.py with LAZY IMPORT
pytest tests/ -k "visualization" -v
git commit -am "refactor(cli): move visualization_commands to heavy/ (plot deps)"
```

- [ ] visualization_commands.py moved and tested ✅

### Step 3.3: Move Mixed Commands (CAREFUL)

#### Move workspace_commands.py
**BEFORE MOVING**: Add lazy imports for data operations:

```python
# In workspace_commands.py, find functions that access data:
def workspace_load(client, output, selector, current_directory, PathResolver):
    # Add lazy import before data access
    if selector and is_h5ad_file(selector):
        # Lazy import for heavy operation
        from lobster.core.data_manager_v2 import DataManagerV2
        # ... rest of function
```

```bash
# After adding lazy imports:
git mv workspace_commands.py light/
# Update __init__.py (EAGER import - it's light now)
pytest tests/ -k "workspace" -v
git commit -am "refactor(cli): move workspace_commands to light/ (lazy imports for data)"
```

- [ ] workspace_commands.py updated with lazy imports ✅
- [ ] Moved to light/ and tested ✅
- [ ] Fast path tested (listing): <200ms
- [ ] Slow path tested (loading H5AD): still works

#### Move file_commands.py
**BEFORE MOVING**: Add lazy imports for H5AD reading:

```python
# In file_commands.py, find file reading logic:
def file_read(client, output, filename, current_directory, PathResolver):
    # ... path resolution ...
    if file_path.suffix == ".h5ad":
        # Lazy import for H5AD reading
        import anndata
        adata = anndata.read_h5ad(file_path)
    # ... rest of function
```

```bash
git mv file_commands.py light/
# Update __init__.py (EAGER import)
pytest tests/ -k "file" -v
git commit -am "refactor(cli): move file_commands to light/ (lazy imports for H5AD)"
```

- [ ] file_commands.py updated with lazy imports ✅
- [ ] Moved to light/ and tested ✅

#### Move pipeline_commands.py
```bash
# Pipeline mostly light (list/info), but run is heavy
git mv pipeline_commands.py light/
# Update __init__.py (EAGER import)
pytest tests/ -k "pipeline" -v
git commit -am "refactor(cli): move pipeline_commands to light/ (lazy imports for run)"
```

- [ ] pipeline_commands.py moved and tested ✅

---

## PHASE 4: UPDATE __init__.py (SMART RE-EXPORT)

### Task 4.1: Implement Lazy Loading
- [ ] Replace current __init__.py with new version (see CLI_REFACTORING_ANALYSIS.md § 6)
- [ ] Key features:
  - ✅ Eager imports for light commands
  - ✅ `__getattr__` for lazy loading heavy commands
  - ✅ Backward-compatible __all__ list

### Task 4.2: Test Lazy Loading
```bash
# Test 1: Light imports are fast
python3 -c "
import sys
import time
start = time.time()
from lobster.cli_internal.commands import show_queue_status
elapsed = time.time() - start
assert elapsed < 0.2, f'Too slow: {elapsed:.3f}s'
assert 'numpy' not in sys.modules, 'numpy loaded!'
print(f'✓ Light import: {elapsed:.3f}s, no numpy')
"

# Test 2: Heavy imports work via lazy loading
python3 -c "
from lobster.cli_internal.commands import data_summary
print(f'✓ Lazy import works: {data_summary}')
"

# Test 3: All imports still work
python3 -c "
from lobster.cli_internal.commands import (
    show_queue_status,
    config_show,
    data_summary,
    modalities_list,
)
print('✓ All imports work')
"
```

- [ ] Light imports fast (<200ms) ✅
- [ ] Heavy imports lazy (numpy not loaded until accessed) ✅
- [ ] All imports backward compatible ✅

### Task 4.3: Commit Final __init__.py
```bash
git commit -am "refactor(cli): implement lazy loading for heavy commands in __init__.py"
```

---

## PHASE 5: UPDATE CONSUMERS (CLI & DASHBOARD)

### Task 5.1: Verify cli.py Still Works
```bash
# cli.py should work WITHOUT changes (backward compatibility)
lobster --help  # Should be fast now!
lobster config  # Should be fast now!
lobster chat    # Should work normally
```

- [ ] `lobster --help` completes in <200ms ✅
- [ ] `lobster config` completes in <300ms ✅
- [ ] `lobster chat` works normally ✅
- [ ] `lobster query "test"` works normally ✅

### Task 5.2: Verify Dashboard Still Works
```bash
# Check if dashboard imports trigger heavy loads
python3 -c "
import sys
import time
start = time.time()
from lobster.ui.screens.analysis_screen import AnalysisScreen
elapsed = time.time() - start
has_numpy = 'numpy' in sys.modules
print(f'Dashboard import: {elapsed:.3f}s, numpy={has_numpy}')
"
```

- [ ] Dashboard imports tested ✅
- [ ] If slow, add lazy imports to dashboard screens

### Task 5.3: Optional Optimization (cli.py)
**Only if you want explicit fast paths**:

```python
# cli.py - OPTIONAL: use explicit light imports for fast commands
from lobster.cli_internal.commands.light import (
    show_queue_status,
    config_show,
    metadata_list,
)
# Heavy commands still imported via backward-compatible __init__.py
from lobster.cli_internal.commands import data_summary  # Lazy
```

- [ ] Decide if explicit imports needed (probably not)
- [ ] If yes, update cli.py imports
- [ ] Test all CLI commands still work

---

## PHASE 6: ADD LAZY IMPORTS TO MIXED COMMANDS

### Task 6.1: Workspace Commands
**Files**: `light/workspace_commands.py`

**Functions needing lazy imports**:
1. `workspace_load()` - when loading H5AD files

**Current pattern** (find this code):
```python
def workspace_load(client, output, selector, current_directory, PathResolver):
    # ... path resolution ...
    adata = client.data_manager.get_modality(selector)  # Heavy!
    # ... display ...
```

**Add lazy import**:
```python
def workspace_load(client, output, selector, current_directory, PathResolver):
    # ... path resolution ...

    # Fast path: listing available datasets (no heavy import)
    if selector is None:
        datasets = client.data_manager.available_datasets  # Light
        # ... display list ...
        return

    # Slow path: loading actual data (lazy import)
    if is_data_operation:
        # Lazy import - only when actually loading data
        # NOTE: This is OK because user explicitly requested data load
        from lobster.core.data_manager_v2 import DataManagerV2
        adata = client.data_manager.get_modality(selector)
    # ... display ...
```

- [ ] Identify fast paths (listing) vs slow paths (loading) ✅
- [ ] Add lazy imports for slow paths ✅
- [ ] Test both paths:
  ```bash
  lobster query "/workspace list"  # Should be fast
  lobster query "/workspace load my_dataset"  # OK if slow
  ```
- [ ] Commit changes

### Task 6.2: File Commands
**Files**: `light/file_commands.py`

**Functions needing lazy imports**:
1. `file_read()` - when reading H5AD files

**Add lazy import**:
```python
def file_read(client, output, filename, current_directory, PathResolver):
    # ... path resolution ...

    if file_path.suffix == ".h5ad":
        # Lazy import for H5AD reading
        import anndata
        adata = anndata.read_h5ad(file_path)
        # ... display ...
    elif file_path.suffix == ".csv":
        # Lazy import for CSV reading
        import pandas as pd
        df = pd.read_csv(file_path)
        # ... display ...
    else:
        # Fast path: text files (no heavy imports)
        content = file_path.read_text()
        # ... display ...
```

- [ ] Add lazy imports for data file formats ✅
- [ ] Test text file reading (should be fast) ✅
- [ ] Test H5AD file reading (OK if slow) ✅
- [ ] Commit changes

### Task 6.3: Pipeline Commands
**Files**: `light/pipeline_commands.py`

**Functions needing lazy imports**:
1. `pipeline_run()` - executes notebooks (heavy)

**Add lazy import**:
```python
def pipeline_run(client, output, notebook_name, input_modality):
    # Fast path: validation and setup
    if notebook_name is None:
        # List available notebooks (light)
        notebooks = client.data_manager.list_notebooks()
        # ... display ...
        return

    # Slow path: actually run notebook (lazy import)
    from lobster.core.notebook_executor import NotebookExecutor  # Heavy
    executor = NotebookExecutor(client.data_manager)
    result = executor.run(notebook_name, input_modality)
    # ... display ...
```

- [ ] Add lazy imports for execution paths ✅
- [ ] Test listing (should be fast) ✅
- [ ] Test execution (OK if slow) ✅
- [ ] Commit changes

---

## PHASE 7: COMPREHENSIVE TESTING

### Task 7.1: Import Tests
```bash
# Create complete test suite
cat > tests/test_cli_refactoring.py << 'EOF'
import sys
import time
import pytest

class TestLightCommands:
    """Test that light commands import fast without heavy deps."""

    def test_light_import_speed(self):
        """Light commands should import in <200ms."""
        # Clear modules
        for m in list(sys.modules.keys()):
            if 'lobster' in m:
                del sys.modules[m]

        start = time.time()
        from lobster.cli_internal.commands import (
            show_queue_status,
            config_show,
            metadata_list,
            workspace_list,
        )
        elapsed = time.time() - start

        assert elapsed < 0.2, f"Light imports took {elapsed:.3f}s (expected <0.2s)"

    def test_no_numpy_after_light_import(self):
        """numpy should NOT be loaded after importing light commands."""
        # Clear modules
        for m in list(sys.modules.keys()):
            if 'numpy' in m or 'pandas' in m or 'scipy' in m:
                del sys.modules[m]

        from lobster.cli_internal.commands import show_queue_status

        assert 'numpy' not in sys.modules, "numpy loaded despite only importing light commands"
        assert 'pandas' not in sys.modules, "pandas loaded despite only importing light commands"
        assert 'scipy' not in sys.modules, "scipy loaded despite only importing light commands"

class TestHeavyCommands:
    """Test that heavy commands work via lazy loading."""

    def test_heavy_command_lazy_import(self):
        """Heavy commands should be importable but not loaded until accessed."""
        # Clear modules
        for m in list(sys.modules.keys()):
            if 'lobster' in m or 'numpy' in m:
                del sys.modules[m]

        # Import the module (should be fast)
        import lobster.cli_internal.commands as cmds

        # data_summary should not be loaded yet (lazy)
        assert 'data_summary' not in dir(cmds) or callable(cmds.data_summary)

    def test_heavy_command_works_when_accessed(self):
        """Heavy commands should work when actually accessed."""
        from lobster.cli_internal.commands import data_summary

        assert callable(data_summary), "data_summary not callable"

class TestBackwardCompatibility:
    """Test that all old imports still work."""

    def test_all_exports_available(self):
        """All 25+ exports should still be available."""
        from lobster.cli_internal.commands import (
            OutputAdapter,
            ConsoleOutputAdapter,
            DashboardOutputAdapter,
            show_queue_status,
            queue_load_file,
            queue_list,
            queue_clear,
            queue_export,
            QueueFileTypeNotSupported,
            metadata_list,
            metadata_clear,
            workspace_list,
            workspace_info,
            workspace_load,
            workspace_remove,
            workspace_status,
            pipeline_export,
            pipeline_list,
            pipeline_run,
            pipeline_info,
            data_summary,
            file_read,
            archive_queue,
            config_show,
            config_provider_list,
            config_provider_switch,
            config_model_list,
            config_model_switch,
            modalities_list,
            modality_describe,
            export_data,
            plots_list,
            plot_show,
        )

        # Verify all are callable or classes
        assert callable(show_queue_status)
        assert callable(data_summary)
        assert isinstance(OutputAdapter, type)

    def test_dashboard_imports_still_work(self):
        """Dashboard imports should work unchanged."""
        from lobster.cli_internal.commands import (
            DashboardOutputAdapter,
            show_queue_status,
            queue_load_file,
            queue_list,
            queue_clear,
        )

        assert callable(show_queue_status)
        assert isinstance(DashboardOutputAdapter, type)
EOF

pytest tests/test_cli_refactoring.py -v -s
```

- [ ] All import tests pass ✅
- [ ] Light commands fast (<200ms) ✅
- [ ] Heavy commands lazy (no numpy until accessed) ✅
- [ ] Backward compatibility maintained ✅

### Task 7.2: Functional Tests
```bash
# Test actual CLI commands
lobster --help          # Should be instant
lobster config          # Should be instant
lobster queue list      # Should be instant
lobster chat            # Test interactively

# Test in actual workspace
cd /tmp/lobster-test
lobster init --non-interactive --use-ollama
lobster query "/workspace list"  # Should be fast
lobster query "/data"            # OK if slow (heavy command)
```

- [ ] All CLI commands work ✅
- [ ] Fast commands are instant ✅
- [ ] Heavy commands still work (just slow) ✅

### Task 7.3: Performance Benchmarks
```bash
# Measure improvements
python3 -c "
import sys
import time

# Clear modules
for m in list(sys.modules.keys()):
    if 'lobster' in m:
        del sys.modules[m]

# Benchmark light import
start = time.time()
from lobster.cli_internal.commands import show_queue_status
light_time = time.time() - start

# Benchmark heavy import (via lazy loading)
start = time.time()
from lobster.cli_internal.commands import data_summary
# Access it to trigger lazy load
_ = data_summary
heavy_time = time.time() - start

print(f'Light import: {light_time:.3f}s')
print(f'Heavy import: {heavy_time:.3f}s')
print(f'Speedup: {heavy_time/light_time:.1f}x')
"
```

- [ ] Document performance improvement ✅
- [ ] Should see ~24x speedup for light commands ✅

---

## PHASE 8: DOCUMENTATION

### Task 8.1: Create README Files
- [ ] Create `cli_internal/commands/light/README.md`:
  ```markdown
  # Light Commands

  Fast commands without heavy data dependencies (<100ms import time).

  ## Rules
  - No module-level numpy/pandas/anndata imports
  - Use TYPE_CHECKING for type hints
  - Use lazy imports for occasional data access
  - Fast path should not access data_manager data methods

  ## Classification
  - Config/settings operations
  - File listing (no reading data files)
  - Queue management (JSONL ops)
  - Metadata management
  - Workspace listing (no loading)
  ```

- [ ] Create `cli_internal/commands/heavy/README.md`:
  ```markdown
  # Heavy Commands

  Data-intensive commands requiring numpy/pandas/anndata (~2s import time).

  ## Rules
  - Can import numpy/pandas/scipy at module level
  - Access client.data_manager data methods freely
  - Focus on data display and analysis operations

  ## Classification
  - Data summaries
  - Modality inspection
  - Visualization
  - Matrix operations
  ```

- [ ] Update `cli_internal/commands/README.md` (if exists)
- [ ] Update root CLAUDE.md § 3.2 "Core Components Reference"

### Task 8.2: Update Code Comments
- [ ] Add docstring to `__init__.py` explaining lazy loading
- [ ] Add comments to `__getattr__` implementation
- [ ] Document performance expectations in command docstrings

---

## PHASE 9: FINAL VALIDATION

### Task 9.1: Full Test Suite
```bash
cd /Users/tyo/GITHUB/omics-os/lobster

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=lobster.cli_internal.commands --cov-report=html

# Check for missing tests
pytest --collect-only | grep "test_cli"
```

- [ ] All tests pass ✅
- [ ] No regressions in existing tests ✅
- [ ] Coverage maintained or improved ✅

### Task 9.2: Integration Tests
```bash
# Test actual CLI usage
cd /tmp/lobster-test-final
rm -rf .env .lobster_workspace

# Fast commands should be instant
time lobster --help          # <200ms
time lobster config          # <300ms
time lobster queue list      # <300ms

# Heavy commands OK to be slow
time lobster query "/data"   # ~2-3s OK

# Interactive session
lobster chat
# Try: /config, /queue list, /workspace list, /data
```

- [ ] CLI feels responsive ✅
- [ ] No user-facing breakage ✅
- [ ] Performance improvement noticeable ✅

### Task 9.3: Check for Warnings
```bash
# Run with warnings enabled
python3 -W all -c "from lobster.cli_internal.commands import show_queue_status"

# Check for deprecation warnings
pytest tests/test_cli_refactoring.py -v -W default
```

- [ ] No new warnings introduced ✅
- [ ] No import warnings ✅

---

## PHASE 10: MERGE & DEPLOY

### Task 10.1: Final Review
- [ ] Review all commits in feature branch
- [ ] Squash if too many small commits
- [ ] Write comprehensive commit message:
  ```
  refactor(cli): split commands into light/heavy for 24x faster startup

  PROBLEM:
  - CLI startup took ~2s even for simple commands like --help
  - All commands imported eagerly, triggering numpy/pandas load

  SOLUTION:
  - Split commands into light/ (fast) and heavy/ (data-intensive)
  - Implemented lazy loading via __getattr__ for heavy commands
  - Maintained backward compatibility (no consumer changes needed)

  PERFORMANCE:
  - Light commands: 2.1s → 0.09s (24x faster)
  - Heavy commands: unchanged (~2s)
  - User experience: --help, config, queue now instant

  BREAKING CHANGES: None (backward compatible)

  Tested:
  - Import tests (light fast, heavy lazy)
  - Functional tests (all commands work)
  - Performance benchmarks (24x improvement)
  - Backward compatibility (cli.py, dashboard unchanged)
  ```

### Task 10.2: Merge to Main
```bash
# Switch to main
git checkout main

# Merge feature branch
git merge --no-ff refactor/cli-commands-light-heavy

# Run final tests on main
pytest tests/ -v

# Push to remote
git push origin main
```

- [ ] Feature branch merged ✅
- [ ] Tests pass on main ✅
- [ ] CI/CD pipeline passes ✅

### Task 10.3: Monitor for Issues
**Watch for**:
- Bug reports about import errors
- Performance regressions
- Dashboard issues

**Monitoring period**: 7 days after merge

---

## ROLLBACK PROCEDURES

### Quick Rollback (If Issues Found Immediately)
```bash
# Revert merge commit
git revert -m 1 HEAD

# Or reset if not pushed
git reset --hard HEAD~1

# Verify original state
pytest tests/ -v
```

### Partial Rollback (Fix Specific Issue)
```bash
# Revert just the __init__.py changes (keep file structure)
git show HEAD:cli_internal/commands/__init__.py > cli_internal/commands/__init__.py
git commit -am "fix: revert __init__.py to eager imports (lazy loading issue)"

# This keeps files in light/heavy but imports them all eagerly
# Performance benefit lost but no breakage
```

### File-by-File Rollback
```bash
# If specific command causing issues, move it back
git mv cli_internal/commands/light/problematic_commands.py \
      cli_internal/commands/

# Update __init__.py to import from root
git commit -am "fix: move problematic_commands back to root"
```

---

## SUCCESS METRICS

### Quantitative
- [x] Light command import: <200ms (24x improvement)
- [ ] `lobster --help`: <200ms ✅
- [ ] `lobster config`: <300ms ✅
- [ ] `lobster queue list`: <300ms ✅
- [ ] Zero test regressions ✅
- [ ] Zero import errors ✅

### Qualitative
- [ ] CLI feels "snappy" for fast commands ✅
- [ ] No confusion from users about command locations ✅
- [ ] Developers understand light/heavy distinction ✅
- [ ] Documentation clear and complete ✅

---

## ESTIMATED TIMELINE

| Phase | Tasks | Time | Risk |
|-------|-------|------|------|
| 1. Preparation | Baseline tests, analysis | 1h | LOW |
| 2. Structure | Create dirs, __init__.py | 0.5h | LOW |
| 3. File moves | Move 10 files, test each | 2h | MEDIUM |
| 4. __init__.py | Lazy loading implementation | 1h | MEDIUM |
| 5. Consumers | Verify cli.py, dashboard | 0.5h | LOW |
| 6. Lazy imports | Add to mixed commands | 2h | MEDIUM |
| 7. Testing | Comprehensive test suite | 2h | LOW |
| 8. Documentation | READMEs, CLAUDE.md | 1h | LOW |
| **Total** | | **10h** | **LOW-MEDIUM** |

**Recommendation**: Allocate 2 full days with time for unexpected issues.

---

## DECISION CHECKLIST

Before proceeding, verify:
- [ ] Performance improvement worth the effort? (YES - 24x speedup)
- [ ] Backward compatibility achievable? (YES - via __init__.py)
- [ ] Rollback plan clear? (YES - git revert)
- [ ] Testing strategy comprehensive? (YES - import + functional + performance)
- [ ] Team aligned on light/heavy distinction? (VERIFY)
- [ ] Documentation plan clear? (YES)

**FINAL DECISION**: [ ] GO / [ ] NO-GO

**Sign-off**:
- [ ] Tech Lead Approved
- [ ] Implementation Date: __________
- [ ] Completion Date: __________

---

**Notes**:
- This checklist is comprehensive but flexible
- Skip steps if already verified
- Add steps if new issues discovered
- Keep git history clean (one logical change per commit)
- Test thoroughly at each step
