# CLI Refactoring Options

**Problem**: `cli.py` is 8,696 lines (356KB) - difficult to maintain and navigate.

**Goal**: Separate command logic into modular files while maintaining functionality.

---

## Current Architecture

```
lobster/
├── cli.py (8,696 lines)           # ❌ Too large
├── cli_internal/                   # ✅ Already exists!
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── output_adapter.py      # ✅ UI abstraction (CLI + Dashboard)
│   │   └── queue_commands.py      # ✅ Queue commands extracted
│   └── utils/
│       └── path_resolution.py     # ✅ Secure path handling
```

**Pattern Already Established**:
- Queue commands (`/queue load`, `/queue list`, `/queue clear`) → `queue_commands.py`
- Uses `OutputAdapter` abstraction for UI-agnostic rendering
- Functions accept `(client, output, ...)` signature
- Returns `Optional[str]` for conversation history

---

## Option 1: Follow Existing Pattern (RECOMMENDED)

**Approach**: Extract commands to `cli_internal/commands/` following the queue commands pattern.

### Proposed Structure

```
lobster/cli_internal/commands/
├── __init__.py                    # Export all command functions
├── output_adapter.py              # ✅ Already exists
├── queue_commands.py              # ✅ Already exists
├── metadata_commands.py           # NEW: /metadata list, /metadata clear
├── workspace_commands.py          # NEW: /workspace load, /workspace list
├── data_commands.py               # NEW: /data, /files, /tree
├── pipeline_commands.py           # NEW: /pipeline export, /pipeline run
├── plot_commands.py               # NEW: /plots, /plot
├── status_commands.py             # NEW: /status, /status-panel
├── analysis_commands.py           # NEW: /analysis-dash, /progress
└── misc_commands.py               # NEW: /help, /history, /read, /open
```

### Implementation Pattern

**1. Create command module** (e.g., `metadata_commands.py`):

```python
"""Metadata management commands for CLI and Dashboard."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter


def metadata_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show metadata store contents.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    if not hasattr(client.data_manager, "metadata_store"):
        output.print("[yellow]⚠️  Metadata store not available[/yellow]", style="warning")
        return None

    metadata_store = client.data_manager.metadata_store

    if not metadata_store:
        output.print("[grey50]No cached metadata in metadata store[/grey50]", style="info")
        return "No metadata in store"

    # Build table data
    table_data = {
        "title": "🗄️ Metadata Store",
        "columns": [
            {"name": "Dataset ID", "style": "bold white"},
            {"name": "Type", "style": "cyan"},
            {"name": "Title", "style": "white", "max_width": 40, "overflow": "ellipsis"},
            {"name": "Samples", "style": "grey74"},
            {"name": "Cached", "style": "grey50"},
        ],
        "rows": []
    }

    for dataset_id, metadata_info in metadata_store.items():
        metadata = metadata_info.get("metadata", {})
        validation = metadata_info.get("validation", {})

        title = metadata.get("title", "N/A")
        data_type = validation.get("predicted_data_type", "unknown").replace("_", " ").title()
        samples = len(metadata.get("samples", {})) if metadata.get("samples") else "N/A"
        timestamp = metadata_info.get("fetch_timestamp", "N/A")[:16]

        table_data["rows"].append([dataset_id, data_type, title, str(samples), timestamp])

    output.print_table(table_data)
    return f"Listed {len(metadata_store)} metadata entries"


def metadata_clear(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Clear metadata store.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    if not hasattr(client.data_manager, "metadata_store"):
        output.print("[yellow]⚠️  Metadata store not available[/yellow]", style="warning")
        return None

    metadata_store = client.data_manager.metadata_store
    num_entries = len(metadata_store)

    if num_entries == 0:
        output.print("[grey50]Metadata store is already empty[/grey50]", style="info")
        return "Metadata store already empty"

    # Confirm with user
    confirm = output.confirm(f"[yellow]Clear all {num_entries} metadata entries?[/yellow]")

    if confirm:
        metadata_store.clear()
        output.print(f"[green]✓ Cleared {num_entries} metadata entries from store[/green]", style="success")
        return f"Cleared {num_entries} metadata entries"
    else:
        output.print("[cyan]Operation cancelled[/cyan]", style="info")
        return None
```

**2. Export from `__init__.py`**:

```python
# lobster/cli_internal/commands/__init__.py

from lobster.cli_internal.commands.output_adapter import (
    OutputAdapter,
    ConsoleOutputAdapter,
    DashboardOutputAdapter,
)

from lobster.cli_internal.commands.queue_commands import (
    show_queue_status,
    queue_load_file,
    queue_list,
    queue_clear,
    queue_export,
    QueueFileTypeNotSupported,
)

from lobster.cli_internal.commands.metadata_commands import (
    metadata_list,
    metadata_clear,
)

# ... more imports as you extract commands

__all__ = [
    # Adapters
    "OutputAdapter",
    "ConsoleOutputAdapter",
    "DashboardOutputAdapter",
    # Queue
    "show_queue_status",
    "queue_load_file",
    "queue_list",
    "queue_clear",
    "queue_export",
    "QueueFileTypeNotSupported",
    # Metadata
    "metadata_list",
    "metadata_clear",
    # ... more exports
]
```

**3. Update cli.py to use extracted commands**:

```python
# In cli.py imports section (around line 53)
from lobster.cli_internal.commands import (
    ConsoleOutputAdapter,
    show_queue_status,
    queue_load_file,
    queue_list,
    queue_clear,
    queue_export,
    metadata_list,    # NEW
    metadata_clear,   # NEW
    QueueFileTypeNotSupported,
)

# In _execute_command() function
elif cmd.startswith("/metadata"):
    # Metadata management commands
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

### Benefits
✅ **Consistency**: Follows established pattern
✅ **Reusability**: Works in CLI + Dashboard
✅ **Testability**: Each command module independently testable
✅ **Maintainability**: Smaller, focused files
✅ **Low Risk**: Pattern already proven in queue_commands.py

### Estimated Impact
- **Current**: 8,696 lines in cli.py
- **After refactoring**: ~3,000-4,000 lines in cli.py (command dispatch only)
- **New files**: ~10 command modules (~200-500 lines each)

---

## Option 2: Command Registry Pattern

**Approach**: Similar to `agent_registry.py`, create a command registry with metadata.

### Structure

```python
# lobster/cli_internal/command_registry.py

from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class CommandConfig:
    """Configuration for a CLI command."""
    name: str                           # "/metadata"
    description: str                    # "Show metadata information"
    handler_function: str               # "metadata_commands.metadata_list"
    subcommands: Optional[dict] = None  # {"list": "...", "clear": "..."}
    aliases: Optional[list] = None      # ["/md"]
    requires_client: bool = True
    requires_workspace: bool = False

COMMAND_REGISTRY = {
    "/metadata": CommandConfig(
        name="/metadata",
        description="Metadata management commands",
        handler_function="lobster.cli_internal.commands.metadata_commands.metadata_dispatch",
        subcommands={
            "list": "Show metadata store",
            "clear": "Clear metadata store"
        }
    ),
    # ... more commands
}
```

### Benefits
✅ **Dynamic**: Easy to add/remove commands
✅ **Documentation**: Registry serves as command catalog
✅ **Validation**: Centralized command validation

### Drawbacks
❌ **Complexity**: More abstraction layers
❌ **Over-engineering**: Current pattern simpler and sufficient

---

## Option 3: Class-Based Commands

**Approach**: Each command as a class with `execute()` method.

### Structure

```python
# lobster/cli_internal/commands/base.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter

class BaseCommand(ABC):
    """Base class for CLI commands."""

    def __init__(self, client: "AgentClient", output: OutputAdapter):
        self.client = client
        self.output = output

    @abstractmethod
    def execute(self, *args, **kwargs) -> Optional[str]:
        """Execute the command."""
        pass


# lobster/cli_internal/commands/metadata_commands.py

class MetadataListCommand(BaseCommand):
    """List metadata store entries."""

    def execute(self) -> Optional[str]:
        if not hasattr(self.client.data_manager, "metadata_store"):
            self.output.print("[yellow]⚠️  Metadata store not available[/yellow]")
            return None

        # ... implementation
        return f"Listed {len(metadata_store)} entries"


class MetadataClearCommand(BaseCommand):
    """Clear metadata store."""

    def execute(self) -> Optional[str]:
        # ... implementation
        return f"Cleared {num_entries} entries"
```

### Benefits
✅ **OOP**: Encapsulation, inheritance
✅ **State management**: Commands can hold state
✅ **Testing**: Mock-friendly

### Drawbacks
❌ **Verbose**: More boilerplate
❌ **Inconsistent**: Different from current pattern
❌ **Overkill**: Commands are simple, don't need classes

---

## Option 4: Plugin System (Advanced)

**Approach**: Commands as plugins loaded via entry points (like `component_registry.py`).

### Structure

```python
# pyproject.toml
[project.entry-points."lobster.cli.commands"]
metadata = "lobster.cli_internal.commands.metadata_commands:METADATA_COMMANDS"
workspace = "lobster.cli_internal.commands.workspace_commands:WORKSPACE_COMMANDS"

# lobster/cli_internal/command_loader.py
def load_commands():
    """Load all CLI commands from entry points."""
    # Similar to component_registry.py
```

### Benefits
✅ **Extensibility**: Custom packages can add commands
✅ **Modularity**: Commands truly decoupled

### Drawbacks
❌ **Complexity**: Significant architectural change
❌ **Over-engineering**: CLI commands don't need plugin system
❌ **Breaking change**: Risky for stable CLI

---

## Recommendation: Option 1 (Follow Existing Pattern)

### Why?
1. **Already proven**: queue_commands.py works well
2. **Low risk**: Incremental refactoring
3. **Consistent**: Matches codebase philosophy
4. **Maintainable**: Simple function-based approach
5. **Reusable**: Dashboard already uses OutputAdapter

### Migration Path

**Phase 1: Extract High-Impact Commands** (Quick wins)
```
cli_internal/commands/
├── metadata_commands.py      # ~200 lines
├── workspace_commands.py     # ~300 lines
├── pipeline_commands.py      # ~250 lines
```

**Phase 2: Extract Visualization Commands**
```
├── plot_commands.py          # ~400 lines
├── analysis_commands.py      # ~300 lines
```

**Phase 3: Extract Remaining Commands**
```
├── data_commands.py          # ~200 lines
├── status_commands.py        # ~300 lines
├── misc_commands.py          # ~250 lines
```

**Phase 4: Refactor cli.py**
- Keep only: Typer setup, `_execute_command()` dispatch, session management
- Move helpers to `cli_internal/utils/`

**Expected Result**: cli.py reduces from 8,696 → ~3,500 lines (60% reduction)

---

## Implementation Checklist

For each command group:

- [ ] Create `cli_internal/commands/{name}_commands.py`
- [ ] Define functions following `(client, output, ...)` signature
- [ ] Use `OutputAdapter` for all output (no direct `console.print()`)
- [ ] Return `Optional[str]` for history logging
- [ ] Add type hints with `TYPE_CHECKING`
- [ ] Export functions in `__init__.py`
- [ ] Update cli.py imports
- [ ] Update cli.py command dispatch
- [ ] Add unit tests for extracted functions
- [ ] Update documentation

---

## Testing Strategy

```python
# tests/unit/cli_internal/commands/test_metadata_commands.py

from unittest.mock import MagicMock
from lobster.cli_internal.commands import metadata_list, metadata_clear
from lobster.cli_internal.commands.output_adapter import OutputAdapter

def test_metadata_list_empty_store():
    """Test metadata_list with empty store."""
    client = MagicMock()
    client.data_manager.metadata_store = {}
    output = MagicMock(spec=OutputAdapter)

    result = metadata_list(client, output)

    assert result == "No metadata in store"
    output.print.assert_called_once()

def test_metadata_clear_with_confirmation():
    """Test metadata_clear with user confirmation."""
    client = MagicMock()
    client.data_manager.metadata_store = {"GSE123": {}}
    output = MagicMock(spec=OutputAdapter)
    output.confirm.return_value = True

    result = metadata_clear(client, output)

    assert result == "Cleared 1 metadata entries"
    assert len(client.data_manager.metadata_store) == 0
```

---

## References

- Existing pattern: `lobster/cli_internal/commands/queue_commands.py`
- Output adapter: `lobster/cli_internal/commands/output_adapter.py`
- Path utilities: `lobster/cli_internal/utils/path_resolution.py`
- Registry pattern: `lobster/config/agent_registry.py`
- Component pattern: `lobster/core/component_registry.py`

---

## Questions?

1. **Why not Option 3 (classes)?**
   - Commands are simple CRUD operations, don't need OOP complexity
   - Function-based approach is more Pythonic for simple operations
   - Existing pattern uses functions successfully

2. **Why not Option 4 (plugins)?**
   - CLI commands are core functionality, not extensions
   - Plugin system adds complexity without clear benefit
   - Entry points better suited for agent/service extensions

3. **Can we mix patterns?**
   - Stick to one pattern for consistency
   - Option 1 is the established pattern (queue_commands.py)
   - All new extractions should follow this pattern

4. **What about backward compatibility?**
   - Zero impact: commands stay in cli.py namespace
   - Only implementation location changes
   - Users see no difference

5. **How long will refactoring take?**
   - Phase 1: 1-2 days (metadata + workspace + pipeline)
   - Phase 2: 1 day (plots + analysis)
   - Phase 3: 1-2 days (data + status + misc)
   - Phase 4: 1 day (cleanup cli.py)
   - **Total: ~1 week** for complete refactoring

