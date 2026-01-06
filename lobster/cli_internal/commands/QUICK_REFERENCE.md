# CLI Refactoring: Quick Reference Card
## One-Page Implementation Guide

---

## COMMAND CLASSIFICATION

### ✅ LIGHT (goes to `light/`)
```
config_commands.py      - Config access only
queue_commands.py       - JSONL file ops
metadata_commands.py    - File listing
workspace_commands.py   - Listing (add lazy imports for load)
file_commands.py        - Text files (add lazy imports for H5AD)
pipeline_commands.py    - Listing (add lazy imports for run)
```

### ❌ HEAVY (goes to `heavy/`)
```
data_commands.py        - Requires numpy/pandas
modality_commands.py    - Matrix/dataframe ops
visualization_commands.py - Plot operations
```

### 📌 SHARED (stays at root)
```
output_adapter.py       - Base class for all commands
```

---

## FILE MOVE COMMANDS

```bash
cd /Users/tyo/GITHUB/omics-os/lobster/lobster/cli_internal/commands

# Create structure
mkdir -p light heavy
touch light/__init__.py heavy/__init__.py

# Move light commands
git mv queue_commands.py light/
git mv config_commands.py light/
git mv metadata_commands.py light/
git mv workspace_commands.py light/
git mv file_commands.py light/
git mv pipeline_commands.py light/

# Move heavy commands
git mv data_commands.py heavy/
git mv modality_commands.py heavy/
git mv visualization_commands.py heavy/

# output_adapter.py stays at root
```

---

## __init__.py TEMPLATE

```python
"""Shared commands with lazy loading for performance."""

# Shared (always imported)
from lobster.cli_internal.commands.output_adapter import (
    OutputAdapter,
    ConsoleOutputAdapter,
    DashboardOutputAdapter,
)

# Light commands (eager import)
from lobster.cli_internal.commands.light.queue_commands import *
from lobster.cli_internal.commands.light.config_commands import *
from lobster.cli_internal.commands.light.metadata_commands import *
from lobster.cli_internal.commands.light.workspace_commands import *
from lobster.cli_internal.commands.light.file_commands import *
from lobster.cli_internal.commands.light.pipeline_commands import *

# Heavy commands (lazy import)
def __getattr__(name):
    """Lazy-load heavy commands to avoid ~2s numpy/pandas import."""
    # Data commands
    if name == "data_summary":
        from lobster.cli_internal.commands.heavy.data_commands import data_summary
        return data_summary

    # Modality commands
    if name in ("modalities_list", "modality_describe"):
        from lobster.cli_internal.commands.heavy.modality_commands import (
            modalities_list,
            modality_describe,
        )
        return locals()[name]

    # Visualization commands
    if name in ("export_data", "plots_list", "plot_show"):
        from lobster.cli_internal.commands.heavy.visualization_commands import (
            export_data,
            plots_list,
            plot_show,
        )
        return locals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Preserve __all__ for backward compatibility
__all__ = [
    # Output adapters
    "OutputAdapter",
    "ConsoleOutputAdapter",
    "DashboardOutputAdapter",
    # Queue commands
    "show_queue_status",
    "queue_load_file",
    "queue_list",
    "queue_clear",
    "queue_export",
    "QueueFileTypeNotSupported",
    # Metadata commands
    "metadata_list",
    "metadata_clear",
    # Workspace commands
    "workspace_list",
    "workspace_info",
    "workspace_load",
    "workspace_remove",
    "workspace_status",
    # Pipeline commands
    "pipeline_export",
    "pipeline_list",
    "pipeline_run",
    "pipeline_info",
    # Data commands (lazy)
    "data_summary",
    # File commands
    "file_read",
    "archive_queue",
    # Config commands
    "config_show",
    "config_provider_list",
    "config_provider_switch",
    "config_model_list",
    "config_model_switch",
    # Modality commands (lazy)
    "modalities_list",
    "modality_describe",
    # Visualization commands (lazy)
    "export_data",
    "plots_list",
    "plot_show",
]
```

---

## LAZY IMPORT PATTERN (for mixed commands)

### workspace_commands.py
```python
def workspace_load(client, output, selector, current_directory, PathResolver):
    """Load workspace item - fast listing, slow data loading."""

    # Fast path: list available (no heavy import)
    if selector is None:
        datasets = client.data_manager.available_datasets
        # ... show list ...
        return

    # Slow path: load actual data (lazy import OK - user requested it)
    # NOTE: Lazy import here is acceptable because this is an explicit data load operation
    if needs_data_access:
        from lobster.core.data_manager_v2 import DataManagerV2  # Heavy
        adata = client.data_manager.get_modality(selector)
        # ... display data ...
```

### file_commands.py
```python
def file_read(client, output, filename, current_directory, PathResolver):
    """Read file - fast for text, slow for data formats."""

    # Fast path: text files (no heavy imports)
    if file_path.suffix in [".txt", ".md", ".log"]:
        content = file_path.read_text()
        output.print(content)
        return

    # Slow path: data files (lazy import OK - user requested data read)
    if file_path.suffix == ".h5ad":
        import anndata  # Lazy import
        adata = anndata.read_h5ad(file_path)
        # ... display data ...
    elif file_path.suffix == ".csv":
        import pandas as pd  # Lazy import
        df = pd.read_csv(file_path)
        # ... display data ...
```

---

## TEST COMMANDS

### Quick Smoke Test
```bash
# Test 1: Light import is fast
time python3 -c "from lobster.cli_internal.commands import show_queue_status"
# Expected: <0.2s

# Test 2: numpy not loaded
python3 -c "
import sys
from lobster.cli_internal.commands import show_queue_status
assert 'numpy' not in sys.modules, 'FAIL: numpy loaded!'
print('✓ PASS: numpy not loaded')
"

# Test 3: Heavy commands work
python3 -c "
from lobster.cli_internal.commands import data_summary
assert callable(data_summary), 'FAIL: not callable!'
print('✓ PASS: heavy command works')
"

# Test 4: CLI commands work
lobster --help          # <200ms
lobster config          # <300ms
lobster queue list      # <300ms
```

### Full Test Suite
```bash
pytest tests/test_cli_refactoring.py -v -s
pytest tests/ -k "command" -v
```

---

## ROLLBACK COMMANDS

### Emergency Rollback (if everything breaks)
```bash
git revert -m 1 HEAD
git push origin main
pytest tests/ -v  # Verify original state restored
```

### Partial Rollback (if __getattr__ breaks)
```bash
# Revert just __init__.py
git show HEAD~1:cli_internal/commands/__init__.py > cli_internal/commands/__init__.py
git commit -am "fix: revert to eager imports (lazy loading issue)"
```

---

## SUCCESS CRITERIA

### Must Have (blocking)
- [ ] All imports backward compatible ✅
- [ ] All tests pass ✅
- [ ] Light commands <200ms ✅
- [ ] Zero import errors ✅

### Nice to Have (non-blocking)
- [ ] Dashboard startup improved
- [ ] Documentation complete
- [ ] Performance benchmarks documented

---

## COMMON PITFALLS

### ❌ DON'T: Import heavy from light
```python
# light/config_commands.py
from lobster.cli_internal.commands.heavy.data_commands import data_summary  # BAD!
```

### ✅ DO: Use lazy imports
```python
# light/config_commands.py
def config_show(client, output):
    if needs_data:
        from lobster.cli_internal.commands.heavy.data_commands import data_summary  # OK
        data_summary(client, output)
```

### ❌ DON'T: Move files without testing
```bash
git mv *.py light/  # BAD! Move one at a time
```

### ✅ DO: Move and test one file at a time
```bash
git mv queue_commands.py light/
pytest tests/ -k "queue" -v
git commit -m "..."
```

### ❌ DON'T: Forget __all__ exports
```python
# __init__.py
from .light.queue_commands import *  # BAD! Not explicit
```

### ✅ DO: Explicit imports and __all__
```python
# __init__.py
from .light.queue_commands import show_queue_status, queue_list
__all__ = ["show_queue_status", "queue_list", ...]  # Explicit
```

---

## KEY FILES

| File | Purpose | Must Edit? |
|------|---------|-----------|
| `__init__.py` | Re-export layer | ✅ YES |
| `light/__init__.py` | Light commands index | ⚠️ MAYBE |
| `heavy/__init__.py` | Heavy commands index | ⚠️ MAYBE |
| `cli.py` | Main CLI (consumer) | ❌ NO (backward compat) |
| `ui/screens/analysis_screen.py` | Dashboard consumer | ⚠️ CHECK |

---

## DEBUGGING TIPS

### If imports break:
```python
# Check what's actually imported
python3 -c "
from lobster.cli_internal.commands import *
print('Available:', dir())
"
```

### If lazy loading fails:
```python
# Check __getattr__ is working
python3 -c "
import lobster.cli_internal.commands as cmds
print('Has __getattr__:', hasattr(cmds, '__getattr__'))
print('data_summary:', cmds.data_summary)
"
```

### If tests fail:
```bash
# Run with verbose output
pytest tests/test_cli_refactoring.py -v -s --tb=long

# Check import order
python3 -c "
import sys
print('Before import:', 'numpy' in sys.modules)
from lobster.cli_internal.commands import show_queue_status
print('After light import:', 'numpy' in sys.modules)
from lobster.cli_internal.commands import data_summary
print('After heavy import:', 'numpy' in sys.modules)
"
```

---

## ESTIMATED TIMINGS

| Phase | Duration | Critical? |
|-------|----------|-----------|
| Preparation | 1h | YES |
| Structure creation | 0.5h | YES |
| File moves | 2h | YES |
| __init__.py update | 1h | YES |
| Lazy imports | 2h | MEDIUM |
| Testing | 2h | YES |
| Documentation | 1h | MEDIUM |
| **TOTAL** | **9.5h** | - |

**Buffer**: Add 2h for unexpected issues → **12h total (1.5 days)**

---

## ONE-LINER DECISION

**Proceed?** YES ✅ - Clear win, low risk, backward compatible, good testing plan

**When?** After current sprint (need 2-day focused block)

**Who?** Senior engineer familiar with CLI + data_manager

**Blocker?** None identified

---

**QUICK START**

```bash
# 1. Create branch
git checkout -b refactor/cli-commands-light-heavy

# 2. Create structure
mkdir -p light heavy
touch light/__init__.py heavy/__init__.py

# 3. Move ONE file and test
git mv queue_commands.py light/
# Update __init__.py (add: from .light.queue_commands import *)
pytest tests/ -k "queue" -v

# 4. If tests pass, continue. If fail, rollback and investigate.

# 5. Repeat for all files (see IMPLEMENTATION_CHECKLIST.md)
```

---

**For detailed analysis, see**: `CLI_REFACTORING_ANALYSIS.md` (13 sections, comprehensive)
**For step-by-step guide, see**: `IMPLEMENTATION_CHECKLIST.md` (10 phases, detailed)
