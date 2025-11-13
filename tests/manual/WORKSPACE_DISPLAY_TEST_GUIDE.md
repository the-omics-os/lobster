# Workspace Display Enhancement Testing Guide

## Test Coverage: Phase 1 (Quick Fix) Features

This document provides comprehensive testing scenarios for the enhanced `/workspace` commands introduced in Phase 1:
- **Intelligent truncation** with middle ellipsis
- **Index-based selection** for list/info/load
- **Pattern matching** for flexible dataset selection
- **Full details display** without truncation

---

## Prerequisites

1. **Environment Setup**
```bash
cd /Users/tyo/GITHUB/omics-os/lobster
source .venv/bin/activate
lobster chat  # Start interactive CLI
```

2. **Test Workspace Preparation**

Create a test workspace with datasets of varying name lengths:

```python
# In lobster chat:
from lobster.core.data_manager_v2 import DataManagerV2
from tests.mock_data.factories import SingleCellDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG
from pathlib import Path

# Create test workspace
test_workspace = Path.home() / ".lobster_test_workspace"
dm = DataManagerV2(workspace_path=test_workspace)

# Create datasets with different name lengths
test_datasets = {
    # Short name (15 chars)
    "geo_gse12345": SingleCellDataFactory(config=SMALL_DATASET_CONFIG),

    # Medium name (45 chars)
    "geo_gse12345_quality_assessed_filtered": SingleCellDataFactory(config=SMALL_DATASET_CONFIG),

    # Long name (80 chars)
    "geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered": SingleCellDataFactory(config=SMALL_DATASET_CONFIG),

    # Very long name (110+ chars)
    "geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered_markers_annotated_pseudobulk": SingleCellDataFactory(config=SMALL_DATASET_CONFIG),

    # Complex name with underscores (95 chars)
    "bulk_rna_seq_project_alpha_experimental_condition_treatment_vs_control_filtered_normalized_de_analyzed": SingleCellDataFactory(config=SMALL_DATASET_CONFIG),
}

# Save all datasets
for name, adata in test_datasets.items():
    dm.modalities[name] = adata
    dm.save_modality(name, f"{name}.h5ad")
    print(f"Created: {name} ({len(name)} chars)")
```

---

## Test Scenarios

### Test 1: Basic `/workspace list` Display

**Command:**
```bash
/workspace list
```

**Expected Behavior:**
- ✅ Table displays with 6 columns: `#`, `Status`, `Name`, `Size`, `Shape`, `Modified`
- ✅ Index column shows sequential numbers (1, 2, 3, ...)
- ✅ Short names (≤60 chars) display in full
- ✅ Long names (>60 chars) are truncated with middle ellipsis (e.g., `geo_gse12345_quality_assessed_filt...alized_pseudobulk`)
- ✅ Truncated names preserve:
  - **Start**: Dataset origin (e.g., `geo_gse12345`, `bulk_rna_seq`)
  - **End**: Processing stage (e.g., `annotated_pseudobulk`, `de_analyzed`)
- ✅ Help hint displayed: `"Use '/workspace info <#>' to see full details"`
- ✅ All rows aligned properly (no overflow)

**Edge Cases to Verify:**
- [ ] Exactly 60 character names (no truncation)
- [ ] 61 character names (minimal truncation)
- [ ] Names with many underscores
- [ ] Empty workspace (no datasets)
- [ ] Single dataset
- [ ] 20+ datasets (ensure consistent indexing)

**Visual Validation Checklist:**
```
┌────┬────────┬──────────────────────────────────────────────────────────────┬────────────┬─────────────────┬──────────────┐
│ #  │ Status │ Name                                                         │ Size       │ Shape           │ Modified     │
├────┼────────┼──────────────────────────────────────────────────────────────┼────────────┼─────────────────┼──────────────┤
│ 1  │ 📁     │ geo_gse12345                                                 │ 1.23 MB    │ 1000 × 2000     │ 2025-01-09   │
│ 2  │ 📁     │ geo_gse12345_quality_assessed_filtered                       │ 1.15 MB    │ 950 × 2000      │ 2025-01-09   │
│ 3  │ 📁     │ geo_gse12345_quality_assessed_filt...ered_markers_annotated  │ 1.05 MB    │ 900 × 1800      │ 2025-01-09   │
│ 4  │ 💾     │ geo_gse12345_quality_assessed_filt...alized_pseudobulk       │ 0.95 MB    │ 50 × 1800       │ 2025-01-09   │
│ 5  │ 📁     │ bulk_rna_seq_project_alpha_experim...filtered_normalized     │ 0.80 MB    │ 200 × 3000      │ 2025-01-09   │
└────┴────────┴──────────────────────────────────────────────────────────────┴────────────┴─────────────────┴──────────────┘

Use '/workspace info <#>' to see full details
```

---

### Test 2: `/workspace info` with Index Selection

**Command:**
```bash
/workspace info 1
/workspace info 3
/workspace info 5
```

**Expected Behavior:**
- ✅ Displays **full, untruncated** dataset name in title
- ✅ Shows comprehensive details:
  - Full name (no truncation)
  - Status (available/loaded)
  - Full file path
  - File size in MB (2 decimal places)
  - Shape with thousands separator (e.g., `1,000 × 2,000`)
  - File type (h5ad)
  - Last modified date (human-readable)
  - Processing stages (if detected from filename)
- ✅ Table formatting is clean and professional

**Edge Cases to Verify:**
- [ ] Index 0 (invalid, should show error)
- [ ] Index beyond range (e.g., `info 99`)
- [ ] Negative index (should show error)
- [ ] Non-numeric index (should show error)

**Example Output:**
```
┌────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
│ Dataset: geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered_markers_annotated │
├────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ Name               │ geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered_mar... │
│ Status             │ 📁 Available                                                                         │
│ Path               │ /Users/tyo/.lobster_test_workspace/data/geo_gse12345_q...annotated.h5ad             │
│ Size               │ 1.05 MB                                                                              │
│ Shape              │ 900 observations × 1,800 variables                                                   │
│ Type               │ h5ad                                                                                 │
│ Modified           │ 2025-01-09 14:32:15                                                                  │
│ Processing Stages  │ quality_assessed → filtered → normalized → doublets_detected → clustered → markers  │
└────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Test 3: `/workspace info` with Pattern Matching

**Command:**
```bash
/workspace info gse12345
/workspace info *clustered*
/workspace info *pseudobulk
/workspace info bulk*
```

**Expected Behavior:**
- ✅ Pattern `gse12345` matches all datasets containing "gse12345"
- ✅ Pattern `*clustered*` matches all datasets with "clustered" anywhere in name
- ✅ Pattern `*pseudobulk` matches all datasets ending with "pseudobulk"
- ✅ Pattern `bulk*` matches all datasets starting with "bulk"
- ✅ Multiple matches display all matching datasets
- ✅ Case-insensitive matching (e.g., `GSE12345` matches `gse12345`)
- ✅ No matches shows helpful error message

**Edge Cases to Verify:**
- [ ] Pattern with no matches: `info xyz999`
- [ ] Wildcard-only pattern: `info *` (matches all)
- [ ] Multiple wildcards: `info *quality*filtered*`
- [ ] Special characters in pattern
- [ ] Very specific pattern matching only one dataset

---

### Test 4: `/workspace load` with Index Selection

**Command:**
```bash
/workspace load 1
/workspace load 3
```

**Expected Behavior:**
- ✅ Loads the dataset at the specified index
- ✅ Shows success message with full dataset name
- ✅ Displays loading time and size
- ✅ Updates status (📁 → 💾)
- ✅ Subsequent `/workspace list` shows "Loaded" status
- ✅ Dataset accessible via CLI commands

**Edge Cases to Verify:**
- [ ] Loading already-loaded dataset (should skip or reload)
- [ ] Loading with invalid index
- [ ] Loading large dataset (>100MB)
- [ ] Memory limits (load multiple large datasets)

**Example Output:**
```
Loading dataset #3: geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered...
✓ Loaded successfully (1.05 MB in 0.23s)

Dataset 'geo_gse12345_quality_assessed_filtered_normalized_doublets_detected_clustered' is now available.
Use '/data' to verify or start analysis.
```

---

### Test 5: `/workspace load` with Pattern Matching

**Command:**
```bash
/workspace load recent
/workspace load *clustered*
/workspace load bulk*
```

**Expected Behavior:**
- ✅ `load recent` loads datasets from last session (if available)
- ✅ Pattern matching loads all matching datasets
- ✅ Shows count of loaded datasets
- ✅ Lists each loaded dataset with size
- ✅ Respects memory limits (default 1000MB)
- ✅ Skips datasets if memory limit reached

**Edge Cases to Verify:**
- [ ] Pattern matches 0 datasets (error message)
- [ ] Pattern matches 10+ datasets (batch loading)
- [ ] Memory limit exceeded during batch load
- [ ] Mix of already-loaded and new datasets

---

### Test 6: Truncation Logic Validation

**Manual Verification Table:**

| Original Length | Truncation Expected | Middle Ellipsis Position | Start Length | End Length |
|-----------------|---------------------|--------------------------|--------------|------------|
| 15 chars        | No                  | -                        | 15           | 0          |
| 45 chars        | No                  | -                        | 45           | 0          |
| 60 chars        | No                  | -                        | 60           | 0          |
| 61 chars        | Yes                 | ~30                      | 30           | 27         |
| 80 chars        | Yes                 | ~30                      | 30           | 27         |
| 110 chars       | Yes                 | ~30                      | 30           | 27         |

**Validation Steps:**
1. For each dataset name, note the original length
2. Check truncated display in `/workspace list`
3. Verify start + "..." + end = ~60 chars
4. Confirm start shows dataset origin
5. Confirm end shows processing stage

---

### Test 7: Integration with Data Loading Workflow

**Complete Workflow Test:**

```bash
# 1. Start fresh session
/workspace list

# 2. View details of a long-named dataset
/workspace info 4

# 3. Load by index
/workspace load 4

# 4. Verify loaded
/workspace list

# 5. Use the dataset
/data

# 6. Load multiple by pattern
/workspace load *clustered*

# 7. Verify all loaded
/workspace list
```

**Expected Behavior:**
- ✅ Seamless workflow without typing long names
- ✅ Consistent indexing across commands
- ✅ Status updates reflect current state
- ✅ No confusion between similar names

---

### Test 8: Error Handling

**Invalid Commands to Test:**

```bash
/workspace info              # Missing selector
/workspace info xyz          # Non-existent pattern
/workspace info 0            # Invalid index (0)
/workspace info -1           # Negative index
/workspace info 999          # Out of range
/workspace info abc          # Non-numeric, non-matching pattern
/workspace load              # Missing selector
/workspace load xyz999       # Non-existent pattern
/workspace load 0            # Invalid index
```

**Expected Error Messages:**
- ✅ Clear, helpful error text
- ✅ Usage examples shown
- ✅ No stack traces (graceful handling)

---

### Test 9: Performance Validation

**Scenarios:**

1. **Large Workspace (100+ datasets)**
   - Create 100+ dummy datasets
   - Run `/workspace list`
   - Measure response time (should be <2 seconds)
   - Verify pagination if implemented

2. **Very Long Names (200+ characters)**
   - Create dataset with 200-character name
   - Verify truncation doesn't break display
   - Check `/workspace info` shows full name

3. **Concurrent Access**
   - Open two terminal sessions
   - Load different datasets in each
   - Verify no conflicts

---

### Test 10: Cross-Platform Compatibility

**Platforms to Test:**

- [ ] macOS (Darwin)
- [ ] Linux (Ubuntu 20.04+)
- [ ] Windows WSL2

**Verification:**
- [ ] Rich library renders correctly
- [ ] Unicode characters display (📁, 💾, ✅)
- [ ] Path separators correct
- [ ] Date formatting consistent

---

## Regression Testing Checklist

Ensure existing functionality still works:

- [ ] `/workspace` without subcommand (shows help)
- [ ] `/workspace list` with no datasets (empty state)
- [ ] `/workspace load recent` (backward compatibility)
- [ ] `/workspace load <full_name>` (legacy name-based loading)
- [ ] `/restore all` (system-level restore)
- [ ] `/restore <pattern>` (pattern-based restore)

---

## Known Issues & Limitations

### Current Implementation Constraints

1. **Index Stability**: Indices are based on alphabetically sorted dataset names. Adding/removing datasets changes indices.
   - **Workaround**: Re-run `/workspace list` after any changes

2. **No Lineage Tracking**: Processing stages detected via filename parsing (basic heuristics)
   - **Future**: Phase 3 will add proper lineage chain extraction from provenance

3. **No Hierarchical View**: All datasets shown in flat list
   - **Future**: Phase 3 optional feature `--tree` view

4. **Memory Limits**: Pattern-based loading respects 1000MB default limit
   - **Workaround**: Load datasets individually by index

---

## Testing Automation

### Create Test Script (Optional)

```python
# tests/manual/test_workspace_display.py
import subprocess
import time
from pathlib import Path

def run_cli_command(command: str) -> str:
    """Run CLI command and capture output."""
    # Implementation would use subprocess or direct CLI invocation
    pass

def test_workspace_list():
    output = run_cli_command("/workspace list")
    assert "Available Datasets" in output
    assert "#" in output  # Index column present
    assert "Name" in output
    assert "Use '/workspace info" in output  # Help hint

def test_workspace_info_by_index():
    output = run_cli_command("/workspace info 1")
    assert "Dataset:" in output
    assert "Path" in output
    assert "Shape" in output

# ... more tests
```

---

## Sign-Off Criteria

Phase 1 implementation is considered complete when:

- ✅ All 10 test scenarios pass
- ✅ No regressions in existing `/workspace` commands
- ✅ Error handling is graceful for all edge cases
- ✅ Performance acceptable for 100+ dataset workspaces
- ✅ Visual formatting is clean and professional
- ✅ User feedback is positive (no confusion, faster workflow)

---

## Next Steps After Testing

1. **User Feedback Collection**
   - Observe real users interacting with new features
   - Note any confusion or unexpected behavior
   - Collect feature requests

2. **Phase 2 Consideration** (if user requests):
   - Lineage visualization
   - Processing stage timeline
   - Dataset dependency graph

3. **Phase 3 Consideration** (if user requests):
   - Full lineage chain extraction from provenance
   - Hierarchical tree view (`/workspace list --tree`)
   - Interactive dataset comparison

---

## Testing Log Template

```
Test Date: __________
Tester: __________
Environment: macOS / Linux / Windows
Lobster Version: __________

| Test # | Scenario | Pass/Fail | Notes |
|--------|----------|-----------|-------|
| 1      | Basic list display | ☐ | |
| 2      | Info by index | ☐ | |
| 3      | Info by pattern | ☐ | |
| 4      | Load by index | ☐ | |
| 5      | Load by pattern | ☐ | |
| 6      | Truncation logic | ☐ | |
| 7      | Integration workflow | ☐ | |
| 8      | Error handling | ☐ | |
| 9      | Performance | ☐ | |
| 10     | Cross-platform | ☐ | |

Overall Status: ☐ PASS  ☐ FAIL  ☐ PARTIAL

Issues Found:
1. _________________
2. _________________

Recommendations:
1. _________________
2. _________________
```

---

## Conclusion

This testing guide ensures comprehensive validation of all Phase 1 workspace display enhancements. Following these scenarios will verify that the implementation meets professional standards and provides a significantly improved user experience for managing datasets with long names.

**Key Improvements Delivered:**
- Zero-typing convenience (index-based selection)
- Full visibility (no information loss)
- Intelligent truncation (semantic preservation)
- Flexible selection (pattern matching)
- Professional presentation (Rich formatting)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-09
**Author**: Lobster AI Development Team
**Status**: Ready for Testing
