# CLI Refactoring: Remaining Commands Analysis

**Current State**: cli.py = 7,650 lines (down from 8,696)
**Extracted So Far**: metadata, workspace, pipeline, data (-1,046 lines, -12%)

---

## High-Impact Extraction Candidates

### Tier 1: Large Commands (300+ lines) - HIGH PRIORITY

| Command | Lines | Complexity | Impact | Notes |
|---------|-------|------------|--------|-------|
| `/config` | ~445 | High | HUGE | Config management, provider setup |
| `/read` | ~323 | Medium | LARGE | File reading with multiple formats |
| `/archive` | ~287 | Medium | LARGE | Queue archiving functionality |
| `/describe` | ~218 | Medium | MEDIUM | Modality details |

**Total Tier 1**: ~1,273 lines (17% of original cli.py)

### Tier 2: Medium Commands (100-300 lines) - MEDIUM PRIORITY

| Command | Lines | Complexity | Impact | Notes |
|---------|-------|------------|--------|-------|
| `/load` | ~133 | Medium | MEDIUM | Deprecated, delegates to /queue |
| `/plot` | ~109 | Low | SMALL | Show single plot |
| `/export` | ~86 | Low | SMALL | Export command |
| `/tokens` | ~80 | Low | SMALL | Token usage display |
| `/input-features` | ~78 | Low | SMALL | Feature list |
| `/modalities` | ~74 | Low | SMALL | Modality details |

**Total Tier 2**: ~560 lines (6% of original cli.py)

### Tier 3: Small Commands (50-100 lines) - LOW PRIORITY

| Command | Lines | Complexity | Impact | Notes |
|---------|-------|------------|--------|-------|
| `/status-panel` | ~56 | Low | TINY | Status display |
| `/open` | ~56 | Low | TINY | Open files |
| `/reset` | ~55 | Low | TINY | Reset session |

**Total Tier 3**: ~167 lines (2% of original cli.py)

### Tier 4: Trivial Commands (<50 lines) - SKIP

Commands like `/clear`, `/exit`, `/help`, `/status`, `/dashboard`, `/analysis-dash`, `/progress`, `/files`, `/tree`, `/plots`, `/save`, `/restore`, `/modes`, `/mode` are too small to extract individually.

**Strategy**: Group into `misc_commands.py` or leave in cli.py

---

## Strategic Extraction Plan

### Option A: Maximum Reduction (Extract Everything)

**Week 2 Plan**:
- Day 1: `/config` (445 lines) → config_commands.py
- Day 2: `/read` + `/archive` (610 lines) → file_commands.py
- Day 3: `/describe` + `/modalities` (292 lines) → modality_commands.py
- Day 4: `/plot` + `/export` (195 lines) → visualization_commands.py
- Day 5: Tier 2 remainder → misc_commands.py

**Result**: cli.py → ~5,000 lines (-2,650 total, **-30% from original**)

### Option B: Pragmatic Reduction (Extract High-Impact Only)

**Focus on Tier 1 only** (1,273 lines):
- `/config` (445 lines)
- `/read` (323 lines)
- `/archive` (287 lines)
- `/describe` (218 lines)

**Result**: cli.py → ~6,400 lines (-2,300 total, **-26% from original**)

### Option C: Quick Wins (Current + 2 More)

Extract 2 more high-impact commands:
- `/config` (445 lines)
- `/read` (323 lines)

**Result**: cli.py → ~6,900 lines (-1,800 total, **-21% from original**)

---

## Recommended: Option A (Maximum Reduction)

### Why?
1. **Maintainability**: Smaller file easier to navigate
2. **Reusability**: All commands work in Dashboard
3. **Testing**: Isolated command testing
4. **Pattern established**: Quick to execute now
5. **Future-proof**: Ready for plugin system

### Grouping Strategy

**file_commands.py**: File operations
- `/read` (323 lines)
- `/archive` (287 lines)
- `/open` (56 lines)
- **Total**: ~666 lines

**config_commands.py**: Configuration
- `/config` (445 lines)
- `/config show`, `/config set`, etc.

**modality_commands.py**: Modality operations
- `/describe` (218 lines)
- `/modalities` (74 lines)
- **Total**: ~292 lines

**visualization_commands.py**: Visualization
- `/plot` (109 lines)
- `/plots` (small)
- `/export` (86 lines)
- **Total**: ~200 lines

**misc_commands.py**: Small utilities
- All Tier 3 + Tier 4 commands
- **Total**: ~300 lines

---

## Extraction Priority (Next 5 Commands)

| Priority | Command | Lines | Time | Cumulative Reduction |
|----------|---------|-------|------|---------------------|
| **1** | `/config` | 445 | 2h | -1,491 lines (-17%) |
| **2** | `/read` | 323 | 1.5h | -1,814 lines (-21%) |
| **3** | `/archive` | 287 | 1.5h | -2,101 lines (-24%) |
| **4** | `/describe` | 218 | 1h | -2,319 lines (-27%) |
| **5** | `/plot` + `/export` | 195 | 1h | -2,514 lines (-29%) |

**Total Time**: ~7 hours
**Final Size**: cli.py → ~6,200 lines (**-2,500 lines, -29% reduction**)

---

## What to Extract Next?

### Immediate Next Steps (Today):

**Recommended**: `/config` (445 lines)
- Highest impact single command
- Complex but self-contained
- Provider setup, show config, etc.
- 2 hours estimated

**Alternative**: `/read` (323 lines)
- File reading with multiple formats
- Simpler than /config
- Good momentum builder
- 1.5 hours estimated

### This Week Target:

Extract Tier 1 commands: `/config`, `/read`, `/archive`, `/describe`
- **Total**: -1,273 lines
- **Result**: cli.py → 6,400 lines (-26% from original)
- **Time**: ~1 day

---

## Commands NOT Worth Extracting

These are too small or tightly coupled to main CLI loop:
- `/status` (3 lines) - trivial
- `/clear` (3 lines) - trivial
- `/exit` (5 lines) - trivial
- `/help` (handled by help system)
- `/dashboard` (launches different UI)
- `/files`, `/tree` (simple listings)

**Leave these in cli.py** - extraction overhead > benefit

---

## Final Target Architecture

```
lobster/cli_internal/commands/
├── __init__.py
├── output_adapter.py          (✅ exists)
├── queue_commands.py           (✅ exists, 433 lines)
├── metadata_commands.py        (✅ created, 273 lines)
├── workspace_commands.py       (✅ created, 679 lines)
├── pipeline_commands.py        (✅ created, 364 lines)
├── data_commands.py            (✅ created, 228 lines)
├── config_commands.py          (🎯 NEXT, ~445 lines)
├── file_commands.py            (📋 /read, /archive, /open, ~666 lines)
├── modality_commands.py        (📋 /describe, /modalities, ~292 lines)
├── visualization_commands.py   (📋 /plot, /export, ~200 lines)
└── misc_commands.py            (📋 small commands, ~300 lines)
```

**Final Estimate**: cli.py → **5,000-5,500 lines** (-3,200 lines, **-37% reduction**)

---

## Next Action

**Your choice**:
1. Continue with `/config` (445 lines, high impact) ← RECOMMENDED
2. Continue with `/read` (323 lines, simpler)
3. Batch extract: `/describe` + `/modalities` (292 lines, quick)
4. Stop here (12% reduction achieved)

Which would you like?
