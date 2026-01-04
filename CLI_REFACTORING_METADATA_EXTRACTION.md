# CLI Refactoring: Metadata Commands Extraction

**Date**: 2026-01-04
**Status**: ✅ COMPLETED
**Pattern**: Option 1 (Function-based, following `queue_commands.py`)

---

## Summary

Successfully extracted `/metadata` commands from `cli.py` (8,696 lines) to dedicated module `metadata_commands.py` (273 lines), reducing `cli.py` by 175 lines (-2%).

This is the **first extraction** following the established pattern, proving the refactoring approach and creating a template for future command extractions.

---

## Changes Made

### 1. Created `lobster/cli_internal/commands/metadata_commands.py` ✅

**Lines**: 273
**Functions**: 2
- `metadata_list(client, output)` - Show metadata store, current metadata, workspace files, exports
- `metadata_clear(client, output)` - Clear metadata store with confirmation

**Pattern Followed**:
```python
# Standard imports
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from lobster.core.client import AgentClient
from lobster.cli_internal.commands.output_adapter import OutputAdapter

# Functions with signature: (client, output, ...) -> Optional[str]
def metadata_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """Full docstring with Args and Returns."""
    # Use output.print() for messages (not console.print)
    # Use output.print_table() for tables (structured data, not Rich Table objects)
    # Return summary string for history or None
```

**Key Improvements**:
- **UI-agnostic**: Uses `OutputAdapter` instead of direct `console.print()`
- **Reusable**: Works in CLI + Dashboard
- **Testable**: Pure functions, easy to mock
- **Structured tables**: Uses `table_data` dict pattern instead of Rich Table objects

### 2. Updated `lobster/cli_internal/commands/__init__.py` ✅

Added exports:
```python
from lobster.cli_internal.commands.metadata_commands import (
    metadata_list,
    metadata_clear,
)

__all__ = [
    # ... existing
    "metadata_list",
    "metadata_clear",
]
```

### 3. Updated `lobster/cli.py` imports ✅

```python
from lobster.cli_internal.commands import (
    # ... existing
    metadata_list,
    metadata_clear,
)
```

### 4. Refactored `/metadata` dispatch in `cli.py` ✅

**Before** (192 lines):
```python
elif cmd.startswith("/metadata"):
    parts = cmd.split()
    subcommand = parts[1] if len(parts) > 1 else None

    if subcommand == "clear":
        # 15 lines of inline logic...
        if not hasattr(client.data_manager, "metadata_store"):
            console.print("[yellow]⚠️  Metadata store not available[/yellow]")
            return None
        # ...

    elif subcommand == "list" or subcommand is None:
        # 177 lines of inline logic...
        console.print("[bold red]📋 Metadata Information[/bold red]\n")
        # ... massive implementation block ...
```

**After** (15 lines):
```python
elif cmd.startswith("/metadata"):
    # Use shared command implementation (unified with dashboard)
    output = ConsoleOutputAdapter(console)
    parts = cmd.split()
    subcommand = parts[1] if len(parts) > 1 else None

    if subcommand == "clear":
        return metadata_clear(client, output)
    elif subcommand == "list" or subcommand is None:
        return metadata_list(client, output)
    else:
        console.print(f"[yellow]Unknown metadata subcommand: {subcommand}[/yellow]")
        console.print("[cyan]Available: list, clear[/cyan]")
        return None
```

**Result**: Clean dispatch, all logic in dedicated module ✅

### 5. Created Unit Tests ✅

**File**: `tests/unit/cli_internal/commands/test_metadata_commands.py`
**Tests**: 7 (all passing ✅)

```
TestMetadataList (3 tests):
- test_empty_metadata_store ✅
- test_with_metadata_store_entries ✅
- test_with_current_metadata ✅

TestMetadataClear (4 tests):
- test_clear_empty_store ✅
- test_clear_with_confirmation_yes ✅
- test_clear_with_confirmation_no ✅
- test_no_metadata_store_attribute ✅
```

**Coverage**: Core functionality (empty state, populated state, confirmations, error handling)

---

## Metrics

| Metric | Value |
|--------|-------|
| **cli.py original** | 8,696 lines |
| **cli.py after** | 8,521 lines |
| **Reduction** | -175 lines (-2%) |
| **New module** | metadata_commands.py (273 lines) |
| **Unit tests** | 7 tests (100% pass) |
| **Test time** | 0.08s |
| **Functions extracted** | 2 |
| **Breaking changes** | 0 (backward compatible) |

---

## Pattern Documentation (For Future Extractions)

### Step-by-Step Extraction Process

**Duration**: ~2 hours for metadata commands (first extraction, includes pattern learning)
**Expected**: ~1 hour for subsequent extractions (pattern established)

#### Phase 1: Analyze & Prepare (15 min)

1. **Identify command section** in `cli.py`
   - Search for `elif cmd.startswith("/your_command"):`
   - Note line numbers and total lines
   - Identify sub-commands (if any)

2. **Review existing pattern**
   - Read `queue_commands.py` to refresh on pattern
   - Note function signatures and return types
   - Check `OutputAdapter` usage patterns

#### Phase 2: Create Command Module (30 min)

1. **Create file**: `lobster/cli_internal/commands/{name}_commands.py`

2. **Write module docstring**:
```python
"""
Shared {name} commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""
```

3. **Add imports**:
```python
from pathlib import Path  # if needed
from typing import TYPE_CHECKING, Optional
from datetime import datetime  # if needed

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter
```

4. **Extract functions** following pattern:
   - Signature: `def command_name(client: "AgentClient", output: OutputAdapter, ...) -> Optional[str]:`
   - Docstring: Include Args and Returns sections
   - Replace `console.print()` → `output.print()`
   - Replace Rich `Table` → `output.print_table(table_data)`
   - Replace `Confirm.ask()` → `output.confirm()`
   - Return summary string or None

5. **Convert tables** to structured format:
```python
table_data = {
    "title": "Optional Title",
    "columns": [
        {"name": "Col1", "style": "cyan", "width": 30},
        {"name": "Col2", "style": "white", "max_width": 50, "overflow": "ellipsis"},
    ],
    "rows": [
        ["value1", "value2"],
        ["value3", "value4"],
    ]
}
output.print_table(table_data)
```

#### Phase 3: Update Exports & Imports (10 min)

1. **Update `cli_internal/commands/__init__.py`**:
```python
from lobster.cli_internal.commands.{name}_commands import (
    {function1},
    {function2},
)

__all__ = [
    # ... existing
    "{function1}",
    "{function2}",
]
```

2. **Update `cli.py` imports**:
```python
from lobster.cli_internal.commands import (
    # ... existing
    {function1},
    {function2},
)
```

#### Phase 4: Refactor Dispatch (15 min)

Replace inline implementation with dispatch:

```python
elif cmd.startswith("/{name}"):
    # Use shared command implementation (unified with dashboard)
    output = ConsoleOutputAdapter(console)
    parts = cmd.split()
    subcommand = parts[1] if len(parts) > 1 else None

    if subcommand == "{sub1}":
        return {function1}(client, output, ...)
    elif subcommand == "{sub2}" or subcommand is None:
        return {function2}(client, output, ...)
    else:
        console.print(f"[yellow]Unknown {name} subcommand: {subcommand}[/yellow]")
        console.print(f"[cyan]Available: {sub1}, {sub2}[/cyan]")
        return None
```

#### Phase 5: Test (30 min)

1. **Syntax check**:
```bash
python -m py_compile lobster/cli_internal/commands/{name}_commands.py
python -m py_compile lobster/cli.py
```

2. **Create unit tests**: `tests/unit/cli_internal/commands/test_{name}_commands.py`
   - Test empty/populated states
   - Test confirmations (yes/no)
   - Test error conditions
   - Use mocks for client and output

3. **Run tests**:
```bash
python -m pytest tests/unit/cli_internal/commands/test_{name}_commands.py -v
```

4. **Integration test** (optional):
```bash
lobster chat
> /{name} list
> /{name} {subcommand}
```

#### Phase 6: Document & Verify (15 min)

1. **Verify line count**:
```bash
wc -l lobster/cli.py lobster/cli_internal/commands/{name}_commands.py
```

2. **Update tracking document**:
   - Lines removed from cli.py
   - Lines in new module
   - Tests created
   - Breaking changes (should be 0)

3. **Commit changes**:
```bash
git add lobster/cli_internal/commands/{name}_commands.py
git add lobster/cli_internal/commands/__init__.py
git add lobster/cli.py
git add tests/unit/cli_internal/commands/test_{name}_commands.py
git commit -m "refactor(cli): extract /{name} commands to dedicated module

- Create {name}_commands.py with {function_count} functions
- Reduce cli.py by {reduction} lines (-{percentage}%)
- Add {test_count} unit tests (all passing)
- Follow established pattern from queue_commands.py
- Zero breaking changes, backward compatible"
```

---

## Key Learnings

### What Worked Well ✅

1. **Existing pattern** (queue_commands.py) provided clear template
2. **OutputAdapter abstraction** made conversion straightforward
3. **Structured table data** cleaner than Rich Table objects
4. **Unit tests** easy to write with mocks
5. **Zero breaking changes** - users see no difference

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Complex table rendering** | Use `table_data` dict pattern, OutputAdapter handles rendering |
| **Datetime imports** | Keep datetime imports in command module (lightweight) |
| **Multiple sections in list** | Break into logical sections with comments, use empty lines |
| **Return value consistency** | Always return `Optional[str]` summary or None |

### Pattern Validation

✅ **Follows established pattern**: Yes (queue_commands.py)
✅ **UI-agnostic**: Yes (OutputAdapter)
✅ **Reusable**: Yes (CLI + Dashboard)
✅ **Testable**: Yes (7 tests, all passing)
✅ **Maintainable**: Yes (273 lines, focused module)
✅ **Backward compatible**: Yes (zero breaking changes)

---

## Next Extraction Candidates

Based on line count and complexity (from largest to smallest):

| Command | Est. Lines | Est. Time | Priority |
|---------|-----------|-----------|----------|
| `/workspace` | ~300-400 | 1.5h | HIGH (high impact) |
| `/pipeline` | ~250-300 | 1h | HIGH (complex logic) |
| `/plots` | ~400-500 | 1.5h | MEDIUM (visualization) |
| `/analysis-dash` | ~300 | 1h | MEDIUM (dashboard) |
| `/data` | ~200 | 1h | MEDIUM (info display) |
| `/status` | ~300 | 1h | LOW (mostly display) |
| `/files` | ~150 | 45min | LOW (simple list) |
| `/tree` | ~100 | 30min | LOW (simple display) |

**Recommended Order** (Week 1):
1. `/workspace` (Day 1-2) - Complex, high impact
2. `/pipeline` (Day 2-3) - Medium complexity, frequent use
3. `/data` (Day 3) - Quick win, practice pattern
4. `/files` + `/tree` (Day 4) - Two simple commands, bundle together
5. Documentation & cleanup (Day 5)

**Expected Result (Week 1)**: cli.py reduces from 8,521 → ~7,500 lines (-1,021 lines, -12%)

---

## Testing Checklist

For each extraction, verify:

- [ ] Python syntax check passes (both files)
- [ ] Unit tests written (minimum 3-5 tests)
- [ ] All unit tests pass
- [ ] Integration test (lobster chat) successful
- [ ] Line count reduced in cli.py
- [ ] New module follows pattern exactly
- [ ] Imports updated correctly
- [ ] __all__ exports updated
- [ ] Backward compatible (no breaking changes)
- [ ] Documentation updated

---

## References

- **This extraction**: `lobster/cli_internal/commands/metadata_commands.py`
- **Pattern source**: `lobster/cli_internal/commands/queue_commands.py`
- **Output adapter**: `lobster/cli_internal/commands/output_adapter.py`
- **Unit tests**: `tests/unit/cli_internal/commands/test_metadata_commands.py`
- **Refactoring guide**: `CLI_REFACTORING_OPTIONS.md`
- **Quick reference**: `CLI_REFACTORING_SUMMARY.md`

---

## Success Metrics (This Extraction)

| Goal | Target | Actual | Status |
|------|--------|--------|--------|
| **Extract functions** | 2 | 2 | ✅ |
| **Reduce cli.py** | -150+ lines | -175 lines | ✅ |
| **Create tests** | 5+ | 7 | ✅ |
| **Test pass rate** | 100% | 100% | ✅ |
| **Breaking changes** | 0 | 0 | ✅ |
| **Pattern consistency** | Match queue_commands.py | ✅ | ✅ |
| **Time to complete** | 2h | ~2h | ✅ |

---

## Conclusion

✅ **Metadata commands successfully extracted**
✅ **Pattern validated and documented**
✅ **Template created for future extractions**
✅ **Zero breaking changes**
✅ **All tests passing**
✅ **Ready for next extraction (/workspace)**

**Next Steps**:
1. Review this document with team
2. Get approval for Week 1 roadmap
3. Schedule `/workspace` extraction (Day 1-2)
4. Continue refactoring following this pattern
5. Update wiki/docs after each extraction

---

**Questions?** Reference `CLI_REFACTORING_OPTIONS.md` for detailed analysis or `CLI_REFACTORING_SUMMARY.md` for quick reference.

