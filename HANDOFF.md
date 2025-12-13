# HANDOFF: Data Persistence Bug Investigation

**Date**: 2025-12-13
**Branch**: `dev_todo`
**Commit**: `152c1fe` - feat: add todo planning system and fix H5AD serialization bug
**Status**: ✅ H5AD NoneType bug FIXED, ⚠️ Data persistence bug remains

---

## What Was Completed

### 1. Todo Planning System ✅

**Problem Solved**: Supervisor was directly fetching large workspace content (10K-50K tokens per entry) instead of delegating to specialists.

**Solution Implemented**: LangGraph Command pattern for atomic state-based todo tracking.

**Files Modified**:
- `lobster/agents/state.py` - TodoItem schema + _todo_reducer
- `lobster/tools/todo_tools.py` - write_todos/read_todos with validation
- `lobster/agents/graph.py` - Integration into supervisor tools
- `lobster/agents/supervisor.py` - Planning section + workspace guardrails
- `lobster/config/supervisor_config.py` - Config options (enable_todo_planning, etc.)
- `tests/unit/tools/test_todo_tools.py` - 12 unit tests (all passing)

**Validation**: Real-world test confirmed supervisor creates multi-step plans, updates atomically, and prevents impulsive workspace fetching.

### 2. H5AD Serialization NoneType Bug ✅

**Problem Solved**: Downloads crashed with `TypeError: PurePosixPath() argument must be str, not 'NoneType'`

**Root Cause**: `pd.Index()` constructor in `_convert_arrow_to_standard()` was resetting `index.name` to None during PyArrow→object conversion.

**Solution Implemented**: 4-layer defense-in-depth approach.

**Files Modified**:
- `lobster/core/backends/h5ad_backend.py` (lines 347, 358) - **CRITICAL FIX**: Preserve/default index names during Arrow conversion
- `lobster/core/backends/h5ad_backend.py` (lines 148-176) - Remove None columns, set None index names
- `lobster/core/backends/h5ad_backend.py` (lines 532-543) - Emergency pre-write check
- `lobster/services/data_access/geo_service.py` (lines 2834-2840, 2884-2890) - Skip None metadata keys

**Test Result**: No more NoneType errors! GSE75748 downloads successfully. GSE84133 now fails with **different** errors (data format issues, not serialization).

---

## ⚠️ REMAINING BUG: Data Persistence Issue

### Problem Statement

**Symptom**: Downloads report "✅ Success" and load data into memory, but **H5AD files are NOT created on disk**.

**Evidence**:
```bash
# After "successful" download:
$ find .lobster_workspace -name "*.h5ad" -type f -mmin -10
# Returns: EMPTY (no files)

$ lobster query "list modalities" | grep GSE75748
# Returns: EMPTY (data not persisted)
```

**Logs Show**:
```
✅ Download Complete - GSE75748 Ready
• Modality: geo_GSE75748
• Dimensions: 19,097 samples × 1,018 features
• Status: Ready for immediate analysis

BUT: No H5AD file on disk, modality not accessible in new sessions
```

### Test Cases

**Test 1: GSE84133 (Complex dataset)**
- **Status**: ❌ Fails with data format errors (not NoneType)
- **Strategy Tried**: SAMPLES_FIRST → failed, MATRIX_FIRST → failed, SUPPLEMENTARY_FIRST → failed
- **Errors**: "too many values to unpack (expected 2)", "Could not parse expression files"
- **Conclusion**: This dataset has inherent format issues beyond the NoneType bug

**Test 2: GSE75748 (Simple processed matrix)**
- **Status**: ⚠️ Reports success, loads in memory, but NO H5AD file created
- **Strategy**: MATRIX_FIRST
- **Error**: None logged, but file not persisted
- **Conclusion**: Silent failure in save path or false success reporting

### Why This Is Separate from NoneType Bug

1. **NoneType errors eliminated**: No more `PurePosixPath(None)` crashes
2. **Different failure mode**: Completes without errors but doesn't persist
3. **Inconsistent behavior**: Some datasets work, others fail silently

---

## Investigation Plan for Next Agent

### Priority 1: Understand Save Flow

**Key Question**: Why does `data_manager.save_modality()` succeed but not create files?

**Files to Investigate**:

1. **`lobster/core/data_manager_v2.py`** (lines 730-780):
   ```python
   def save_modality(self, modality_name: str, path: str, **kwargs) -> Optional[Path]:
       """Save modality to file."""
   ```
   - Check return value (does it return None on silent failure?)
   - Verify file path resolution
   - Check if exceptions are caught and suppressed

2. **`lobster/services/data_access/geo_service.py`** (line 1268):
   ```python
   saved_file = self.data_manager.save_modality(modality_name, save_path)
   ```
   - Is `saved_file` checked? Does it validate the file exists?
   - What happens if `save_modality()` returns None?

3. **`lobster/core/backends/h5ad_backend.py`** (line 555):
   ```python
   adata_to_save.write_h5ad(resolved_path, ...)
   ```
   - Does `write_h5ad()` actually write the file?
   - Check if path resolution fails silently
   - Verify parent directory exists

### Priority 2: Test Minimal Reproduction

**Create a minimal test case** to isolate the issue:

```python
# Test script: test_save_minimal.py
from pathlib import Path
import anndata
import numpy as np
import pandas as pd

from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.data_manager_v2 import DataManagerV2

# Create minimal AnnData
adata = anndata.AnnData(
    X=np.random.randn(100, 50),
    obs=pd.DataFrame({"batch": ["A"]*100}, index=[f"cell_{i}" for i in range(100)]),
    var=pd.DataFrame({"gene": [f"gene_{i}" for i in range(50)]}, index=[f"gene_{i}" for i in range(50)])
)

# Test 1: Direct H5AD save
backend = H5ADBackend()
test_path = Path("/tmp/test_direct.h5ad")
backend.save(adata, test_path)
print(f"Direct save: {test_path.exists()}")  # Should be True

# Test 2: Via DataManagerV2
workspace = Path("/tmp/test_workspace")
workspace.mkdir(exist_ok=True)
dm = DataManagerV2(workspace_path=workspace)
dm.add_modality("test_modality", adata)
saved = dm.save_modality("test_modality", "test_via_dm.h5ad")
print(f"Via DataManager: {saved}")  # Should return Path
print(f"File exists: {(workspace / 'data' / 'test_via_dm.h5ad').exists()}")
```

**Run**:
```bash
python test_save_minimal.py
```

**Expected**: Both methods create H5AD files
**Actual**: TBD - will reveal where the persistence fails

### Priority 3: Check Error Suppression

**Search for silent exception handling**:

```bash
cd /Users/tyo/GITHUB/omics-os/lobster
grep -n "except.*:" lobster/core/data_manager_v2.py | grep -A2 "save_modality"
grep -n "except.*pass" lobster/core/backends/h5ad_backend.py
grep -n "except.*:" lobster/services/data_access/geo_service.py | grep -B5 -A5 "save_modality"
```

Look for patterns like:
```python
try:
    saved_file = self.data_manager.save_modality(...)
except Exception as e:
    logger.error(...)
    # BUT: Does it re-raise? Or silently continue?
```

### Priority 4: Verify Path Resolution

**Check workspace path resolution**:

```python
# In data_manager_v2.py:save_modality()
print(f"Workspace path: {self.workspace_path}")
print(f"Data dir: {self.data_dir}")
print(f"Requested path: {path}")
print(f"Resolved path: {resolved_path}")
print(f"Parent exists: {resolved_path.parent.exists()}")
```

**Common Issues**:
- Relative paths incorrectly resolved
- Data directory not created
- Path conflicts between raw/processed files

---

## Quick Start for Next Agent

### Step 1: Verify Current State

```bash
cd /Users/tyo/GITHUB/omics-os/lobster
git status  # Should be on dev_todo branch, commit 152c1fe

# Clear all caches
rm -rf .lobster_workspace/cache/geo/*
rm -f .lobster_workspace/download_queue.jsonl
rm -f .lobster_workspace/.session.json
```

### Step 2: Reproduce the Bug

```bash
source .venv/bin/activate

# Test with simple dataset
lobster query "ADMIN SUPERUSER: Download GSE75748"

# Check results
ls -lh .lobster_workspace/data/*.h5ad  # Should show files but won't
lobster query "list modalities"  # Should show geo_GSE75748 but won't
```

### Step 3: Add Diagnostic Logging

Modify `data_manager_v2.py:save_modality()` to add:
```python
logger.warning(f"🔍 SAVE: Attempting to save {modality_name} to {path}")
logger.warning(f"🔍 SAVE: Resolved path: {resolved_path}")
logger.warning(f"🔍 SAVE: Backend used: {backend.__class__.__name__}")
# After save:
logger.warning(f"🔍 SAVE: File exists: {resolved_path.exists()}")
logger.warning(f"🔍 SAVE: File size: {resolved_path.stat().st_size if resolved_path.exists() else 0}")
```

### Step 4: Run Minimal Test

Use the `test_save_minimal.py` script above to isolate whether:
- H5AD backend works directly (likely ✅)
- DataManagerV2 wrapper has the bug (likely ❌)

---

## Key Files Reference

### Data Flow (Save Operation)

```
geo_service.py:download_dataset()
    ↓ line 1268
self.data_manager.save_modality(modality_name, save_path)
    ↓
data_manager_v2.py:save_modality()  [lines 730-780]
    ↓ line 755
backend_instance.save(adata, resolved_path, **save_kwargs)
    ↓
h5ad_backend.py:save()  [lines 419-560]
    ↓ line 555
adata_to_save.write_h5ad(resolved_path, ...)
```

### Critical Code Locations

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `core/data_manager_v2.py` | 730-780 | save_modality() orchestration | ⚠️ INVESTIGATE |
| `core/backends/h5ad_backend.py` | 419-560 | H5AD save implementation | ✅ FIXED |
| `services/data_access/geo_service.py` | 1260-1280 | GEO download + save call | ✅ OK |
| `core/backends/h5ad_backend.py` | 545-560 | Exception handling | ⚠️ CHECK |

### Error Handling to Check

```python
# Line 545 in h5ad_backend.py
except Exception as e:
    # Remove failed file if it was created
    if resolved_path.exists():
        try:
            resolved_path.unlink()
        except Exception:
            pass
    raise ValueError(f"Failed to save H5AD file {resolved_path}: {e}") from e
```

**Question**: Is this exception being caught upstream without re-raising?

---

## Hypotheses to Test

### Hypothesis 1: Silent Exception Catch
**Likelihood**: 🔴 HIGH

`geo_service.py` or `data_manager_v2.py` catches save exceptions but continues as if successful.

**Test**:
```bash
grep -n "except.*:" lobster/services/data_access/geo_service.py | grep -B10 -A10 "1268"
```

### Hypothesis 2: Path Resolution Failure
**Likelihood**: 🟡 MEDIUM

`resolved_path` is None or incorrect, causing silent failure.

**Test**:
Add logging at line 755 in data_manager_v2.py:
```python
logger.error(f"SAVE DEBUG: resolved_path={resolved_path}, type={type(resolved_path)}")
```

### Hypothesis 3: In-Memory Only Mode
**Likelihood**: 🟢 LOW

DataManagerV2 might have a mode that skips file persistence.

**Test**:
Check for flags like `persist=False` or `memory_only=True` in DataManagerV2 initialization.

### Hypothesis 4: Async Save Race Condition
**Likelihood**: 🟢 LOW

Save completes but file is immediately deleted or moved.

**Test**:
Check for cleanup routines that remove *_raw.h5ad files.

---

## Debugging Commands

```bash
# 1. Test direct H5AD save (bypasses DataManagerV2)
python -c "
import anndata, numpy as np, pandas as pd
from lobster.core.backends.h5ad_backend import H5ADBackend
adata = anndata.AnnData(X=np.random.randn(10,5))
backend = H5ADBackend()
backend.save(adata, '/tmp/test_direct.h5ad')
print('Exists:', Path('/tmp/test_direct.h5ad').exists())
"

# 2. Trace save_modality execution
grep -n "def save_modality" lobster/core/data_manager_v2.py
# Add logger.warning() at every step

# 3. Check for exception suppression
grep -rn "except.*pass" lobster/core/data_manager_v2.py lobster/services/data_access/geo_service.py

# 4. Find where "Download Complete" message is generated
grep -rn "Download Complete" lobster/
```

---

## Expected Next Steps

1. **Read** `lobster/core/data_manager_v2.py` lines 730-780 (save_modality method)
2. **Add** diagnostic logging to trace save execution
3. **Run** minimal reproduction test
4. **Identify** where save succeeds in-memory but fails to persist
5. **Fix** the persistence issue
6. **Test** with GSE75748 to verify H5AD file is created
7. **Document** the fix and update this handoff

---

## Test Datasets

### GSE75748 (Recommended for testing)
- **Type**: Simple processed matrix
- **Size**: 19,097 cells × 1,018 genes
- **Status**: Downloads successfully, loads in memory, but NO file on disk
- **Use for**: Testing persistence fix

### GSE84133 (Complex, has other issues)
- **Type**: Raw single-cell data with supplementary files
- **Size**: 3,605 cells × 20,125 genes
- **Status**: Multiple format errors (unrelated to NoneType bug)
- **Use for**: Validation after GSE75748 works

---

## Success Criteria

You've succeeded when:
- ✅ `lobster query "ADMIN SUPERUSER: Download GSE75748"` creates H5AD file on disk
- ✅ `ls .lobster_workspace/data/*.h5ad` shows the file
- ✅ `lobster query "list modalities"` shows geo_GSE75748 in new session
- ✅ File is readable: `python -c "import anndata; adata = anndata.read_h5ad('path'); print(adata.shape)"`

---

## Additional Context

### Why GSE84133 Still Fails

GSE84133 has **inherent data format issues** separate from NoneType bug:
- Complex TAR structure with nested files
- Mixed human/mouse samples (incompatible for analysis)
- Missing critical metadata (cell_type, condition, tissue)
- "too many values to unpack" error in concatenation service

**Recommendation**: Use GSE75748 for persistence testing, defer GSE84133 format issues to separate ticket.

### Log Analysis Hints

Look for these patterns indicating silent failures:
```python
# Pattern 1: Exception caught but not re-raised
try:
    result = operation()
except Exception as e:
    logger.error(f"Error: {e}")
    return None  # ⚠️ Silent failure!

# Pattern 2: Success assumed without verification
saved_file = self.save_modality(...)
# No check if saved_file is None or if file actually exists

# Pattern 3: Transient vs. persistent storage confusion
self.modalities[name] = adata  # In-memory ✅
self.save_modality(name, path)  # File save ❌ (might fail silently)
```

---

## Contact/Questions

If you need clarification on:
- Todo planning system implementation → See `lobster/tools/todo_tools.py` docstrings
- H5AD bug fix logic → See commit message and inline comments in `h5ad_backend.py`
- Test methodology → See `tests/unit/tools/test_todo_tools.py` (12 passing tests)

**Previous Session Context**: Available in git log and commit 152c1fe message

---

## Quick Reference Commands

```bash
# Test persistence
rm -rf .lobster_workspace/cache/geo/* .lobster_workspace/download_queue.jsonl
source .venv/bin/activate
lobster query "ADMIN SUPERUSER: Download GSE75748"
ls -lh .lobster_workspace/data/*.h5ad  # Should show file (currently doesn't)

# Check logs for save operations
grep -i "save.*modality" ~/.lobster/*.log

# Verify H5AD backend works directly
python test_save_minimal.py  # Create this using script above
```

---

**End of Handoff**

Good luck! The NoneType bug is resolved - now it's about finding why successful downloads don't persist to disk.
