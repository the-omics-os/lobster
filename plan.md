# Plan: Enhanced `/metadata` Command UX

## Problem Statement

After publication processing or metadata processing runs, users need to understand:
- What metadata is available and where it lives
- Status of processing workflows (publication queue, downloads)
- Sample statistics (counts, disease coverage, filter effectiveness)
- Actionable next steps

**Current `/metadata` command limitations:**
1. Shows only raw file listings without context
2. No publication queue status breakdown
3. No aggregated sample statistics or disease coverage metrics
4. No filter effectiveness visibility
5. No hierarchical view of the data flow (publications → datasets → samples)
6. Missing guidance for next actions

## Solution Overview

Create a **hierarchical `/metadata` command** with subcommands for different views:

```
/metadata                    # Smart overview with key stats
/metadata publications       # Publication queue status breakdown
/metadata samples            # Aggregated sample statistics + disease coverage
/metadata workspace          # File inventory with categorization
/metadata exports            # Export files with usage guidance
/metadata <identifier>       # Drill-down into specific entry
```

## Implementation Plan

### Phase 1: Create MetadataOverviewService (New Service)

**File**: `lobster/services/metadata/metadata_overview_service.py`

**Purpose**: Centralized metadata statistics aggregation (reusable by CLI and agents)

**Methods**:
```python
class MetadataOverviewService:
    def __init__(self, data_manager: DataManagerV2):
        self.data_manager = data_manager
        self.workspace_service = WorkspaceContentService(data_manager)

    def get_publication_queue_summary(self) -> Dict[str, Any]:
        """Return status breakdown, identifier coverage, extracted dataset counts."""

    def get_sample_statistics(self) -> Dict[str, Any]:
        """Return aggregated sample counts, disease coverage, filter retention."""

    def get_workspace_inventory(self) -> Dict[str, Any]:
        """Return categorized file inventory across all storage locations."""

    def get_export_summary(self) -> Dict[str, Any]:
        """Return export files with categories and usage hints."""

    def get_quick_overview(self) -> Dict[str, Any]:
        """Return compact summary for default /metadata command."""
```

**Output structure** (for `get_quick_overview`):
```python
{
    "publication_queue": {
        "total": 47,
        "status_breakdown": {"pending": 12, "handoff_ready": 8, "completed": 25, "failed": 2},
        "ready_for_action": 8  # handoff_ready count
    },
    "samples": {
        "total_samples": 1247,
        "filtered_samples": 527,
        "disease_coverage": 0.785,
        "bioproject_count": 89
    },
    "workspace": {
        "metadata_files": 156,
        "export_files": 12,
        "total_size_mb": 45.3
    },
    "next_steps": [
        "8 entries ready for metadata filtering",
        "2 failed entries need attention"
    ]
}
```

### Phase 2: Enhance metadata_commands.py

**File**: `lobster/cli_internal/commands/light/metadata_commands.py`

#### New Functions:

1. **`metadata_overview()`** - Default smart overview
   - Shows: Quick stats grid + next steps guidance
   - Calls: `MetadataOverviewService.get_quick_overview()`

2. **`metadata_publications()`** - Publication queue breakdown
   - Shows: Status pie, identifier coverage, extracted datasets by type
   - Calls: `MetadataOverviewService.get_publication_queue_summary()`

3. **`metadata_samples()`** - Sample statistics
   - Shows: Total/filtered counts, disease coverage bar, filter retention breakdown
   - Calls: `MetadataOverviewService.get_sample_statistics()`

4. **`metadata_workspace()`** - File inventory
   - Shows: Categorized tree view (metadata files, exports, deprecated warnings)
   - Calls: `MetadataOverviewService.get_workspace_inventory()`

5. **`metadata_exports()`** - Export files
   - Shows: Export files with categories, sizes, and usage commands
   - Calls: `MetadataOverviewService.get_export_summary()`

6. **`metadata_detail(identifier: str)`** - Single entry detail
   - Shows: Full details for specific publication/dataset/sample file
   - Auto-detects type from identifier pattern

#### Output Format (Rich tables with emojis):

```
┌─────────────────────────────────────────────────────────────┐
│ 📋 Metadata Overview                                        │
├─────────────────────────────────────────────────────────────┤
│ Publication Queue                                           │
│ ├─ Total: 47 entries                                        │
│ ├─ ⏳ Pending: 12    │ 🤝 Handoff Ready: 8                  │
│ ├─ ✅ Completed: 25  │ ❌ Failed: 2                         │
│                                                             │
│ Sample Statistics                                           │
│ ├─ Total Samples: 1,247 (89 BioProjects)                   │
│ ├─ Filtered Samples: 527 (42.3% retention)                 │
│ ├─ Disease Coverage: ████████░░ 78.5%                      │
│                                                             │
│ Workspace Files                                             │
│ ├─ Metadata files: 156 (12.3 MB)                           │
│ ├─ Export files: 12 (8.7 MB)                               │
│                                                             │
│ 💡 Next Steps:                                              │
│ • 8 entries ready for filtering: /metadata publications     │
│ • 2 failed entries: /metadata publications --status=failed  │
│ • View exports: /metadata exports                           │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: CLI Router Integration

**File**: `lobster/cli.py`

Update the `/metadata` command handler to support subcommands:

```python
elif user_input.startswith("/metadata"):
    parts = user_input.split()
    if len(parts) == 1:
        # Default: /metadata
        result = metadata_overview(client, output_adapter)
    elif parts[1] == "publications":
        result = metadata_publications(client, output_adapter, status_filter=...)
    elif parts[1] == "samples":
        result = metadata_samples(client, output_adapter)
    elif parts[1] == "workspace":
        result = metadata_workspace(client, output_adapter)
    elif parts[1] == "exports":
        result = metadata_exports(client, output_adapter)
    elif parts[1] == "clear":
        # Existing clear functionality
        result = metadata_clear(client, output_adapter)
    else:
        # Treat as identifier lookup
        result = metadata_detail(client, output_adapter, identifier=parts[1])
```

### Phase 4: Help Text and Documentation

1. Update `/help` output to document new subcommands
2. Add inline guidance in command output (next steps suggestions)
3. Update wiki documentation

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `lobster/services/metadata/metadata_overview_service.py` | **CREATE** | New service for metadata aggregation |
| `lobster/services/metadata/__init__.py` | MODIFY | Export new service |
| `lobster/cli_internal/commands/light/metadata_commands.py` | MODIFY | Add new subcommand functions |
| `lobster/cli_internal/commands/__init__.py` | MODIFY | Export new functions |
| `lobster/cli.py` | MODIFY | Add subcommand routing |

## Detailed Implementation Steps

### Step 1: Create MetadataOverviewService

```python
# lobster/services/metadata/metadata_overview_service.py

from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import Counter

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.data_access.workspace_content_service import (
    WorkspaceContentService,
    ContentType,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataOverviewService:
    """
    Centralized metadata statistics aggregation service.

    Provides unified access to metadata state across:
    - Publication queue (status, identifiers, extracted datasets)
    - Sample metadata (counts, disease coverage, filter stats)
    - Workspace files (categorized inventory)
    - Export files (with usage guidance)
    """

    def __init__(self, data_manager: DataManagerV2):
        self.data_manager = data_manager
        self.workspace_service = WorkspaceContentService(data_manager)
        self._workspace_path = Path(data_manager.workspace_path)

    def get_publication_queue_summary(self) -> Dict[str, Any]:
        """
        Get publication queue status breakdown with identifier coverage.

        Returns:
            Dict with:
            - total: Total entries
            - status_breakdown: Counter of statuses
            - identifier_coverage: Dict of identifier type -> count/percentage
            - extracted_datasets: Counter of extracted database identifiers
            - recent_errors: List of recent error messages
        """
        ...

    def get_sample_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated sample statistics across all processed metadata.

        Returns:
            Dict with:
            - total_samples: Total samples across all BioProjects
            - filtered_samples: Samples after filtering
            - bioproject_count: Number of BioProjects
            - disease_coverage: Percentage with disease annotation
            - filter_criteria: Active filter string
            - filter_breakdown: Per-filter retention stats
            - aggregation_sources: List of source identifiers
        """
        ...

    def get_workspace_inventory(self) -> Dict[str, Any]:
        """
        Get categorized file inventory across all storage locations.

        Returns:
            Dict with:
            - metadata_store: In-memory metadata count
            - workspace_files: Dict by category (sra_samples, publication_*, etc.)
            - exports: Export files with categories
            - deprecated_warnings: Files in old locations
            - total_size_mb: Total workspace size
        """
        ...

    def get_export_summary(self) -> Dict[str, Any]:
        """
        Get export files with categories and usage guidance.

        Returns:
            Dict with:
            - files: List of export file info (name, size, category, modified)
            - categories: Counter of file categories
            - usage_hints: Dict of category -> usage command
        """
        ...

    def get_quick_overview(self) -> Dict[str, Any]:
        """
        Get compact summary for default /metadata command.

        Combines key metrics from all summaries into single overview.
        """
        ...
```

### Step 2: Implement Publication Queue Summary

Reading from `publication_queue.jsonl`:
```python
def get_publication_queue_summary(self) -> Dict[str, Any]:
    queue_path = self._workspace_path / "publication_queue.jsonl"

    if not queue_path.exists():
        return {"total": 0, "status_breakdown": {}, "message": "No publication queue found"}

    entries = []
    with open(queue_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Status breakdown
    status_counts = Counter(e.get("status", "unknown") for e in entries)

    # Identifier coverage
    pmid_count = sum(1 for e in entries if e.get("pmid"))
    doi_count = sum(1 for e in entries if e.get("doi"))
    pmc_count = sum(1 for e in entries if e.get("pmc_id"))

    # Extracted identifiers (aggregated)
    extracted = Counter()
    for e in entries:
        ids = e.get("extracted_identifiers", {})
        for db_type, id_list in ids.items():
            if id_list:
                extracted[db_type] += len(id_list)

    # Recent errors
    recent_errors = []
    for e in entries:
        if e.get("status") == "failed" and e.get("error"):
            recent_errors.append({
                "entry_id": e.get("entry_id"),
                "error": e.get("error")[:100]
            })

    return {
        "total": len(entries),
        "status_breakdown": dict(status_counts),
        "identifier_coverage": {
            "pmid": {"count": pmid_count, "pct": pmid_count / len(entries) * 100 if entries else 0},
            "doi": {"count": doi_count, "pct": doi_count / len(entries) * 100 if entries else 0},
            "pmc_id": {"count": pmc_count, "pct": pmc_count / len(entries) * 100 if entries else 0},
        },
        "extracted_datasets": dict(extracted),
        "workspace_ready": sum(1 for e in entries if e.get("workspace_metadata_keys")),
        "recent_errors": recent_errors[:5]
    }
```

### Step 3: Implement Sample Statistics

Reading from metadata_store and aggregated files:
```python
def get_sample_statistics(self) -> Dict[str, Any]:
    # Check for aggregated_filtered_samples in metadata_store
    aggregated = self.data_manager.metadata_store.get("aggregated_filtered_samples", {})

    if aggregated:
        samples = aggregated.get("samples", [])
        stats = aggregated.get("stats", {})
        return {
            "total_samples": stats.get("total_extracted", len(samples)),
            "filtered_samples": stats.get("total_after_filter", len(samples)),
            "disease_coverage": stats.get("disease_coverage", 0),
            "filter_criteria": aggregated.get("filter_criteria", ""),
            "filter_breakdown": stats.get("filter_breakdown", {}),
            "has_aggregated": True
        }

    # Fallback: count samples from sra_*_samples files
    total_samples = 0
    bioproject_count = 0

    for key, data in self.data_manager.metadata_store.items():
        if key.startswith("sra_") and key.endswith("_samples"):
            samples = data.get("samples", [])
            total_samples += len(samples)
            bioproject_count += 1

    return {
        "total_samples": total_samples,
        "filtered_samples": 0,
        "disease_coverage": 0,
        "filter_criteria": "",
        "bioproject_count": bioproject_count,
        "has_aggregated": False,
        "message": "Run metadata filtering to generate aggregated statistics"
    }
```

### Step 4: Update metadata_commands.py

```python
def metadata_overview(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """Smart overview with key stats and next steps."""
    from lobster.services.metadata.metadata_overview_service import MetadataOverviewService

    service = MetadataOverviewService(client.data_manager)
    overview = service.get_quick_overview()

    # Build Rich output
    output.print("\n[bold cyan]📋 Metadata Overview[/bold cyan]\n")

    # Publication Queue section
    pq = overview.get("publication_queue", {})
    if pq.get("total", 0) > 0:
        output.print("[bold white]Publication Queue[/bold white]")

        status_table = {
            "columns": [
                {"name": "Status", "style": "bold"},
                {"name": "Count", "style": "cyan"},
            ],
            "rows": []
        }

        status_emojis = {
            "pending": "⏳", "extracting": "🔄", "metadata_extracted": "📄",
            "metadata_enriched": "✨", "handoff_ready": "🤝",
            "completed": "✅", "failed": "❌"
        }

        for status, count in pq.get("status_breakdown", {}).items():
            emoji = status_emojis.get(status, "📌")
            status_table["rows"].append([f"{emoji} {status}", str(count)])

        output.print_table(status_table)

    # Sample Statistics section
    samples = overview.get("samples", {})
    if samples.get("total_samples", 0) > 0:
        output.print("\n[bold white]Sample Statistics[/bold white]")
        output.print(f"  Total: {samples['total_samples']:,} samples")

        if samples.get("has_aggregated"):
            filtered = samples.get("filtered_samples", 0)
            retention = filtered / samples["total_samples"] * 100 if samples["total_samples"] else 0
            output.print(f"  Filtered: {filtered:,} ({retention:.1f}% retention)")

            coverage = samples.get("disease_coverage", 0)
            bar = _make_progress_bar(coverage)
            output.print(f"  Disease Coverage: {bar} {coverage:.1f}%")

    # Next Steps
    next_steps = overview.get("next_steps", [])
    if next_steps:
        output.print("\n[bold yellow]💡 Next Steps[/bold yellow]")
        for step in next_steps:
            output.print(f"  • {step}")

    return "Metadata overview displayed"


def _make_progress_bar(pct: float, width: int = 10) -> str:
    """Create ASCII progress bar."""
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)
```

### Step 5: CLI Routing

Update `cli.py` command handler:
```python
elif user_input.startswith("/metadata"):
    parts = shlex.split(user_input)  # Handle quoted args

    if len(parts) == 1:
        result = metadata_overview(client, output_adapter)
    elif parts[1] in ("pub", "publications"):
        status_filter = None
        if len(parts) > 2 and parts[2].startswith("--status="):
            status_filter = parts[2].split("=")[1]
        result = metadata_publications(client, output_adapter, status_filter)
    elif parts[1] in ("samples", "sample"):
        result = metadata_samples(client, output_adapter)
    elif parts[1] in ("workspace", "ws"):
        result = metadata_workspace(client, output_adapter)
    elif parts[1] in ("exports", "export"):
        result = metadata_exports(client, output_adapter)
    elif parts[1] == "clear":
        # Existing clear logic (with subcommands)
        if len(parts) > 2 and parts[2] == "exports":
            result = metadata_clear_exports(client, output_adapter)
        elif len(parts) > 2 and parts[2] == "all":
            result = metadata_clear_all(client, output_adapter)
        else:
            result = metadata_clear(client, output_adapter)
    else:
        # Treat as identifier lookup
        result = metadata_detail(client, output_adapter, identifier=parts[1])
```

## Testing Plan

1. **Unit tests** for MetadataOverviewService methods
2. **Integration tests** with mock workspace data
3. **Manual testing** with real publication processing output

## Success Criteria

1. `/metadata` shows meaningful overview within 2 seconds
2. Users can quickly identify what's ready for next action
3. Disease coverage and filter retention visible at a glance
4. Clear guidance on what commands to run next
5. Backward compatible with existing `/metadata clear` commands
