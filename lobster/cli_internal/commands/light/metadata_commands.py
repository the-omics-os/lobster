"""
Shared metadata commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import (
    OutputAdapter,
    OutputBlock,
    alert_block,
    hint_block,
    kv_block,
    list_block,
    section_block,
    table_block,
)
from lobster.i18n import h


def _make_progress_bar(pct: float, width: int = 10) -> str:
    """Create ASCII progress bar for percentages."""
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _render_blocks(output: OutputAdapter, blocks: list[OutputBlock]) -> None:
    output.render_blocks(blocks)


def metadata_overview(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show smart metadata overview with key stats and next steps.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.services.metadata.metadata_overview_service import (
        MetadataOverviewService,
    )

    service = MetadataOverviewService(client.data_manager)
    overview = service.get_quick_overview()

    blocks: list[OutputBlock] = [section_block(title=h("Metadata Overview"))]

    # Publication Queue section
    pq = overview.get("publication_queue", {})
    if pq.get("total", 0) > 0:
        status_rows = []
        status_emojis = {
            "pending": "⏳",
            "extracting": "🔄",
            "metadata_extracted": "📄",
            "metadata_enriched": "✨",
            "handoff_ready": "🤝",
            "completed": "✅",
            "failed": "❌",
            "paywalled": "🔒",
        }

        for status, count in pq.get("status_breakdown", {}).items():
            emoji = status_emojis.get(status, "📌")
            status_rows.append([f"{emoji} {status}", str(count)])

        blocks.append(section_block(title=h("Publication Queue")))
        if status_rows:
            blocks.append(
                table_block(
                    columns=[{"name": h("Status")}, {"name": h("Count")}],
                    rows=status_rows,
                )
            )
            blocks.append(
                hint_block(
                    " | ".join(
                        [
                            h("Total: {total}").format(total=pq["total"]),
                            h("Workspace-ready: {count}").format(
                                count=pq.get("workspace_ready", 0)
                            ),
                            h("Extracted datasets: {count}").format(
                                count=pq.get("extracted_datasets", 0)
                            ),
                        ]
                    )
                )
            )

    # Sample Statistics section
    samples = overview.get("samples", {})
    if samples.get("total_samples", 0) > 0:
        sample_rows = [
            (h("Total Samples"), f"{samples['total_samples']:,}"),
            ("BioProjects", str(samples.get("bioproject_count", 0))),
        ]

        if samples.get("has_aggregated"):
            filtered = samples.get("filtered_samples", 0)
            retention = samples.get("retention_rate", 0)
            sample_rows.append((h("Filtered Samples"), f"{filtered:,}"))
            sample_rows.append((h("Retention"), f"{retention:.1f}%"))

            coverage = samples.get("disease_coverage", 0)
            bar = _make_progress_bar(coverage)
            sample_rows.append((h("Disease Coverage"), f"{bar} {coverage:.1f}%"))
        else:
            blocks.append(
                hint_block(
                    h("Run metadata filtering to generate aggregated statistics.")
                )
            )
        blocks.append(kv_block(sample_rows, title=h("Sample Statistics")))

    # Workspace Files section
    workspace = overview.get("workspace", {})
    if workspace.get("metadata_files", 0) > 0 or workspace.get("export_files", 0) > 0:
        workspace_rows = []
        if workspace.get("metadata_files", 0) > 0:
            workspace_rows.append(
                (
                    h("Metadata Files"),
                    f"{workspace['metadata_files']} ({workspace.get('total_size_mb', 0):.1f} MB)",
                )
            )
        if workspace.get("export_files", 0) > 0:
            workspace_rows.append((h("Export Files"), str(workspace["export_files"])))
        if workspace.get("in_memory_entries", 0) > 0:
            workspace_rows.append(
                (h("In-memory Entries"), str(workspace["in_memory_entries"]))
            )
        blocks.append(kv_block(workspace_rows, title=h("Workspace Files")))

    # Next Steps
    next_steps = overview.get("next_steps", [])
    if next_steps:
        blocks.append(list_block(next_steps, title=h("Next Steps")))

    # Deprecated warnings
    if overview.get("has_deprecated"):
        blocks.append(
            alert_block(
                h(
                    "Found files in deprecated metadata/exports/ location. Use /metadata workspace for details."
                ),
                level="warning",
            )
        )

    blocks.append(
        hint_block(
            f"{h('Commands:')} /metadata publications | samples | workspace | exports | clear"
        )
    )
    _render_blocks(output, blocks)

    return f"Metadata overview: {pq.get('total', 0)} publications, {samples.get('total_samples', 0)} samples"


def metadata_publications(
    client: "AgentClient", output: OutputAdapter, status_filter: Optional[str] = None
) -> Optional[str]:
    """
    Show publication queue status breakdown with identifier coverage.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        status_filter: Optional status to filter by

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.services.metadata.metadata_overview_service import (
        MetadataOverviewService,
    )

    service = MetadataOverviewService(client.data_manager)
    summary = service.get_publication_queue_summary(status_filter=status_filter)

    if summary.get("total", 0) == 0:
        _render_blocks(
            output,
            [
                alert_block(
                    h(
                        "No publication queue found. Use research_agent to process publications."
                    ),
                    level="warning",
                )
            ],
        )
        return None

    blocks: list[OutputBlock] = [
        section_block(
            title=h("Publication Queue ({total} entries)").format(
                total=summary["total"]
            )
        )
    ]
    status_rows = []

    status_emojis = {
        "pending": "⏳",
        "extracting": "🔄",
        "metadata_extracted": "📄",
        "metadata_enriched": "✨",
        "handoff_ready": "🤝",
        "completed": "✅",
        "failed": "❌",
        "paywalled": "🔒",
    }

    for status, count in summary.get("status_breakdown", {}).items():
        emoji = status_emojis.get(status, "📌")
        status_rows.append([f"{emoji} {status}", str(count)])
    blocks.append(
        table_block(
            title=h("Status Breakdown"),
            columns=[{"name": h("Status")}, {"name": h("Count")}],
            rows=status_rows,
        )
    )

    # Identifier coverage
    id_cov = summary.get("identifier_coverage", {})
    if id_cov:
        coverage_rows = []
        for id_type, stats in id_cov.items():
            count = stats.get("count", 0)
            pct = stats.get("pct", 0)
            bar = _make_progress_bar(pct, width=15)
            coverage_rows.append(
                [id_type.upper(), f"{count}/{summary['total']}", f"{bar} {pct:.1f}%"]
            )
        blocks.append(
            table_block(
                title=h("Identifier Coverage"),
                columns=[
                    {"name": h("Type")},
                    {"name": h("Count")},
                    {"name": h("Coverage")},
                ],
                rows=coverage_rows,
            )
        )

    # Extracted datasets
    extracted = summary.get("extracted_datasets", {})
    if extracted:
        blocks.append(
            table_block(
                title=h("Extracted Identifiers"),
                columns=[{"name": h("Database")}, {"name": h("Count")}],
                rows=[
                    [db_type.upper(), str(count)]
                    for db_type, count in sorted(extracted.items(), key=lambda x: -x[1])
                ],
            )
        )

    # Workspace readiness
    ws_ready = summary.get("workspace_ready", 0)
    if ws_ready > 0:
        blocks.append(
            section_block(
                body=h("Workspace Status: {count} entries with metadata files").format(
                    count=ws_ready
                )
            )
        )

    # Recent errors
    errors = summary.get("recent_errors", [])
    if errors:
        blocks.append(section_block(title=h("Recent Errors")))
        blocks.append(
            list_block(
                [
                    f"{err['entry_id']}: {err['title']} - {err['error']}"
                    for err in errors
                ]
            )
        )

    # Filter hint
    if not status_filter:
        blocks.append(
            hint_block(
                h("Tip: Filter by status with /metadata publications --status=<status>")
            )
        )
    _render_blocks(output, blocks)

    return f"Publication queue: {summary['total']} entries"


def metadata_samples(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show aggregated sample statistics with disease coverage.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.services.metadata.metadata_overview_service import (
        MetadataOverviewService,
    )

    service = MetadataOverviewService(client.data_manager)
    stats = service.get_sample_statistics()

    if stats.get("total_samples", 0) == 0:
        _render_blocks(
            output,
            [
                alert_block(
                    h(
                        "No sample metadata found. Process publications with research_agent first."
                    ),
                    level="warning",
                )
            ],
        )
        return None

    total = stats.get("total_samples", 0)
    bioproject_count = stats.get("bioproject_count", 0)
    blocks: list[OutputBlock] = [
        kv_block(
            [
                (h("Total Samples"), f"{total:,}"),
                ("BioProjects", str(bioproject_count)),
            ],
            title=h("Sample Statistics"),
        )
    ]

    if stats.get("has_aggregated"):
        filtered = stats.get("filtered_samples", 0)
        retention = stats.get("retention_rate", 0)
        coverage = stats.get("disease_coverage", 0)
        bar = _make_progress_bar(coverage, width=20)
        detail_rows = [
            (h("Filtered Samples"), f"{filtered:,}"),
            (h("Retention"), f"{retention:.1f}%"),
            (h("Disease Coverage"), f"{bar} {coverage:.1f}%"),
        ]

        criteria = stats.get("filter_criteria", "")
        if criteria:
            detail_rows.append((h("Filter Criteria"), criteria))
        blocks.append(kv_block(detail_rows, title=h("Aggregated Statistics")))

        breakdown = stats.get("filter_breakdown", {})
        if breakdown:
            breakdown_rows = []
            for filter_name, filter_stats in breakdown.items():
                if isinstance(filter_stats, dict):
                    retained = filter_stats.get("retained", 0)
                    total_filtered = filter_stats.get("total", 0)
                    pct = retained / total_filtered * 100 if total_filtered > 0 else 0
                    bar = _make_progress_bar(pct, width=15)
                    breakdown_rows.append(
                        [
                            filter_name,
                            f"{retained}/{total_filtered}",
                            f"{bar} {pct:.1f}%",
                        ]
                    )
            if breakdown_rows:
                blocks.append(
                    table_block(
                        title=h("Filter Breakdown"),
                        columns=[
                            {"name": h("Filter")},
                            {"name": h("Retained")},
                            {"name": h("Coverage")},
                        ],
                        rows=breakdown_rows,
                    )
                )
        blocks.append(
            section_block(
                body=h(
                    "Aggregated metadata available. Use /metadata exports to see export files."
                )
            )
        )
    else:
        blocks.append(
            alert_block(
                h(
                    "Samples not yet filtered. Use metadata_assistant to apply filters and generate aggregated statistics."
                ),
                level="warning",
            )
        )

        sources = stats.get("sources", [])
        if sources:
            source_items = list(sources[:10])
            if len(sources) > 10:
                source_items.append(
                    h("... and {count} more").format(count=len(sources) - 10)
                )
            blocks.append(list_block(source_items, title=h("Sample Sources")))

    _render_blocks(output, blocks)

    return f"Sample stats: {total:,} samples, {bioproject_count} BioProjects"


def metadata_workspace(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show categorized file inventory across all storage locations.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.services.metadata.metadata_overview_service import (
        MetadataOverviewService,
    )

    service = MetadataOverviewService(client.data_manager)
    inventory = service.get_workspace_inventory()
    blocks: list[OutputBlock] = [section_block(title=h("Workspace Inventory"))]

    mem_count = inventory.get("metadata_store_count", 0)
    if mem_count > 0:
        rows = [(h("Entries"), str(mem_count))]
        categories = inventory.get("metadata_store_categories", {})
        for cat, count in sorted(
            categories.items(), key=lambda item: (-item[1], item[0])
        ):
            rows.append((cat, str(count)))
        blocks.append(
            kv_block(
                rows,
                title=h("In-Memory Store"),
            )
        )
    else:
        blocks.append(hint_block(h("In-memory metadata store is empty.")))

    ws_files = inventory.get("workspace_files", {})
    if ws_files:
        total_files = inventory.get("workspace_files_total", 0)
        size_mb = inventory.get("total_size_mb", 0)
        rows = [
            (h("Total Files"), str(total_files)),
            (h("Total Size"), f"{size_mb:.1f} MB"),
        ]
        for cat, count in sorted(
            ws_files.items(), key=lambda item: (-item[1], item[0])
        ):
            rows.append((cat, str(count)))
        blocks.append(
            kv_block(
                rows,
                title=h("Workspace Files"),
            )
        )
    else:
        blocks.append(hint_block(h("No workspace metadata files found.")))

    exports = inventory.get("exports", [])
    if exports:
        total_exports = inventory.get("exports_total", 0)
        rows = [
            [exp["name"], f"{exp['size_kb']:.1f} KB", exp["modified"]]
            for exp in exports[:10]
        ]
        blocks.append(
            table_block(
                title=h("Export Files ({total})").format(total=total_exports),
                columns=[
                    {"name": h("File")},
                    {"name": h("Size")},
                    {"name": h("Modified")},
                ],
                rows=rows,
            )
        )
        if len(exports) > 10:
            blocks.append(
                hint_block(
                    h("... and {count} more files").format(count=total_exports - 10)
                )
            )

    warnings = inventory.get("deprecated_warnings", [])
    if warnings:
        for warn in warnings:
            blocks.append(alert_block(warn, level="warning"))
        blocks.append(
            hint_block(
                f"{h('Consider migrating:')} mv workspace/metadata/exports/* workspace/exports/"
            )
        )

    _render_blocks(output, blocks)
    return f"Workspace: {mem_count} in-memory, {inventory.get('workspace_files_total', 0)} files"


def metadata_exports(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show export files with categories and usage guidance.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.services.metadata.metadata_overview_service import (
        MetadataOverviewService,
    )

    service = MetadataOverviewService(client.data_manager)
    exports = service.get_export_summary()

    if exports.get("total_count", 0) == 0:
        _render_blocks(
            output,
            [
                alert_block(
                    exports.get(
                        "message",
                        "No export files found. Use write_to_workspace() to export data.",
                    ),
                    level="warning",
                )
            ],
        )
        return None

    blocks: list[OutputBlock] = [
        section_block(
            title=h("Export Files ({total} files)").format(total=exports["total_count"])
        )
    ]

    categories = exports.get("categories", {})
    if categories:
        rows = [
            (cat, str(count))
            for cat, count in sorted(
                categories.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        blocks.append(
            kv_block(
                rows,
                title=h("File Categories"),
                key_label=h("Category"),
                value_label=h("Count"),
            )
        )

    files = exports.get("files", [])
    if files:
        blocks.append(
            table_block(
                title=h("Recent Files"),
                columns=[
                    {
                        "name": h("File"),
                        "style": "cyan",
                        "width": 50,
                        "overflow": "ellipsis",
                    },
                    {"name": h("Size"), "style": "grey50", "width": 10},
                    {"name": h("Modified"), "style": "grey50", "width": 18},
                ],
                rows=[
                    [f["name"], f"{f['size_kb']:.1f} KB", f["modified"]]
                    for f in files[:15]
                ],
            )
        )
        if len(files) > 15:
            blocks.append(
                hint_block(
                    h("... and {count} more files").format(
                        count=exports["total_count"] - 15
                    )
                )
            )

    hints = exports.get("usage_hints", {})
    if hints:
        blocks.append(
            kv_block(
                [
                    (h("List exports"), hints.get("list", "N/A")),
                    (h("Access in code"), hints.get("access", "N/A")),
                    (h("CLI command"), hints.get("cli", "N/A")),
                ],
                title=h("Usage Tips"),
                key_label=h("Action"),
                value_label=h("Command"),
            )
        )

    _render_blocks(output, blocks)
    return f"Export files: {exports['total_count']} files"


def metadata_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show metadata store contents and workspace metadata files.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    blocks: list[OutputBlock] = [section_block(title=h("Metadata Information"))]
    entries_shown = 0

    if hasattr(client.data_manager, "metadata_store"):
        metadata_store = client.data_manager.metadata_store
        if metadata_store:
            rows = []
            for dataset_id, metadata_info in metadata_store.items():
                metadata = metadata_info.get("metadata", {})
                validation = metadata_info.get("validation", {})

                # Extract key information
                title = str(metadata.get("title", "N/A"))
                if len(title) > 40:
                    title = title[:40] + "..."

                data_type = (
                    validation.get("predicted_data_type", "unknown")
                    .replace("_", " ")
                    .title()
                )

                samples = (
                    len(metadata.get("samples", {}))
                    if metadata.get("samples")
                    else "N/A"
                )

                # Parse timestamp
                timestamp = metadata_info.get("fetch_timestamp", "")
                try:
                    cached_time = datetime.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    )
                    cached_str = cached_time.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    cached_str = timestamp[:16] if timestamp else "N/A"

                rows.append([dataset_id, data_type, title, str(samples), cached_str])

            blocks.append(
                table_block(
                    title=h("Metadata Store"),
                    columns=[
                        {"name": h("Dataset ID"), "style": "bold white"},
                        {"name": h("Type"), "style": "cyan"},
                        {
                            "name": h("Title"),
                            "style": "white",
                            "max_width": 40,
                            "overflow": "ellipsis",
                        },
                        {"name": h("Samples"), "style": "grey74"},
                        {"name": h("Cached"), "style": "grey50"},
                    ],
                    rows=rows,
                )
            )
            entries_shown += len(metadata_store)
        else:
            blocks.append(hint_block(h("No cached metadata in metadata store")))

    if (
        hasattr(client.data_manager, "current_metadata")
        and client.data_manager.current_metadata
    ):
        metadata = client.data_manager.current_metadata
        rows = []
        for key, value in metadata.items():
            if isinstance(value, dict):
                display_value = h("Dict with {count} keys: {keys}").format(
                    count=len(value), keys=", ".join(list(value.keys())[:3])
                )
                if len(value) > 3:
                    display_value += h(" ... (+{count} more)").format(
                        count=len(value) - 3
                    )
            elif isinstance(value, list):
                display_value = h("List with {count} items").format(count=len(value))
                if len(value) > 0:
                    display_value += h(": {items}").format(
                        items=", ".join(str(v) for v in value[:3])
                    )
                    if len(value) > 3:
                        display_value += h(" ... (+{count} more)").format(
                            count=len(value) - 3
                        )
            else:
                display_value = str(value)
                if len(display_value) > 60:
                    display_value = display_value[:60] + "..."

            rows.append((key, display_value))

        blocks.append(
            kv_block(
                rows,
                title=h("Current Data Metadata"),
                key_label=h("Key"),
                value_label=h("Value"),
            )
        )
        entries_shown += len(metadata)
    else:
        blocks.append(hint_block(h("No current data metadata available")))

    workspace_path = Path(client.data_manager.workspace_path)
    metadata_dir = workspace_path / "metadata"

    if metadata_dir.exists():
        json_files = sorted(metadata_dir.glob("*.json"))
        if json_files:
            rows = []
            for json_file in json_files[:20]:
                stat = json_file.stat()
                size_kb = stat.st_size / 1024
                modified = datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )
                rows.append([json_file.name, f"{size_kb:.1f} KB", modified])

            blocks.append(
                table_block(
                    title=h("Workspace Metadata Files"),
                    columns=[
                        {
                            "name": h("File"),
                            "style": "cyan",
                            "width": 50,
                            "overflow": "ellipsis",
                        },
                        {"name": h("Size"), "style": "grey50", "width": 10},
                        {"name": h("Modified"), "style": "grey50", "width": 20},
                    ],
                    rows=rows,
                )
            )

            if len(json_files) > 20:
                blocks.append(
                    hint_block(
                        h("... and {count} more files").format(
                            count=len(json_files) - 20
                        )
                    )
                )
            blocks.append(hint_block(h("Path: {path}").format(path=metadata_dir)))

    exports_dir = workspace_path / "exports"
    if exports_dir.exists():
        export_files = sorted(
            [
                f
                for f in exports_dir.iterdir()
                if f.is_file() and f.suffix in {".csv", ".tsv", ".xlsx"}
            ],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if export_files:
            rows = []
            for export_file in export_files[:15]:
                stat = export_file.stat()
                size_kb = stat.st_size / 1024
                modified = datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )
                rows.append([export_file.name, f"{size_kb:.1f} KB", modified])

            blocks.append(
                table_block(
                    title=h("Export Files"),
                    columns=[
                        {
                            "name": h("File"),
                            "style": "green",
                            "width": 50,
                            "overflow": "ellipsis",
                        },
                        {"name": h("Size"), "style": "grey50", "width": 10},
                        {"name": h("Modified"), "style": "grey50", "width": 20},
                    ],
                    rows=rows,
                )
            )

            if len(export_files) > 15:
                blocks.append(
                    hint_block(
                        h("... and {count} more files").format(
                            count=len(export_files) - 15
                        )
                    )
                )
            blocks.append(hint_block(h("Path: {path}").format(path=exports_dir)))

    old_exports_dir = workspace_path / "metadata" / "exports"
    if old_exports_dir.exists():
        old_files = list(old_exports_dir.glob("*"))
        if old_files:
            blocks.append(
                alert_block(
                    h("Found {count} file(s) in old location: {path}").format(
                        count=len(old_files), path=old_exports_dir
                    ),
                    level="warning",
                )
            )
            blocks.append(hint_block(h("New exports go to: workspace/exports/")))
            blocks.append(
                hint_block(
                    f"{h('Migration:')} mv workspace/metadata/exports/* workspace/exports/"
                )
            )

    _render_blocks(output, blocks)

    if entries_shown > 0:
        return f"Displayed metadata information ({entries_shown} entries)"
    else:
        return "No metadata available"


def metadata_clear(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Clear metadata store (memory) AND workspace metadata files (disk).

    Clears:
    1. In-memory metadata_store (DataManagerV2)
    2. In-memory current_metadata (legacy)
    3. Workspace metadata files (workspace/metadata/*.json)

    NOTE: Does NOT clear export files. Use /metadata clear exports for that.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    # Count in-memory entries (metadata_store)
    metadata_store_count = 0
    if hasattr(client.data_manager, "metadata_store"):
        metadata_store_count = len(client.data_manager.metadata_store)

    # Count in-memory current_metadata (legacy)
    current_metadata_count = 0
    if hasattr(client.data_manager, "current_metadata"):
        current_metadata_count = len(client.data_manager.current_metadata)

    memory_count = metadata_store_count + current_metadata_count

    # Count disk files in workspace/metadata/
    disk_files = []
    metadata_dir = None
    if hasattr(client.data_manager, "workspace_path"):
        metadata_dir = Path(client.data_manager.workspace_path) / "metadata"
        if metadata_dir.exists():
            disk_files = list(metadata_dir.glob("*.json"))

    disk_count = len(disk_files)
    total_count = memory_count + disk_count

    if total_count == 0:
        _render_blocks(
            output,
            [
                alert_block(
                    h("Metadata is already empty (memory + disk)"), level="warning"
                )
            ],
        )
        return "Metadata already empty"

    _render_blocks(
        output,
        [
            kv_block(
                [
                    (
                        h("Memory (metadata_store)"),
                        h("{count} entries").format(count=metadata_store_count),
                    ),
                    (
                        h("Memory (current_metadata)"),
                        h("{count} entries").format(count=current_metadata_count),
                    ),
                    (
                        h("Disk (workspace/metadata/)"),
                        h("{count} files").format(count=disk_count),
                    ),
                    (h("Total"), h("{count} items").format(count=total_count)),
                ],
                title=h("About to Clear"),
            )
        ],
    )

    confirm = output.confirm(
        h("Clear all {count} metadata items?").format(count=total_count)
    )

    if confirm:
        cleared_store = 0
        if metadata_store_count > 0 and hasattr(client.data_manager, "metadata_store"):
            client.data_manager.metadata_store.clear()
            cleared_store = metadata_store_count

        cleared_current = 0
        if current_metadata_count > 0 and hasattr(
            client.data_manager, "current_metadata"
        ):
            client.data_manager.current_metadata.clear()
            cleared_current = current_metadata_count

        deleted_files = 0
        failures = []
        for json_file in disk_files:
            try:
                json_file.unlink()
                deleted_files += 1
            except Exception as e:
                failures.append(f"{json_file.name}: {e}")

        result_parts = []
        if cleared_store > 0:
            result_parts.append(
                h("{count} metadata_store entries").format(count=cleared_store)
            )
        if cleared_current > 0:
            result_parts.append(
                h("{count} current_metadata entries").format(count=cleared_current)
            )
        if deleted_files > 0:
            result_parts.append(h("{count} disk files").format(count=deleted_files))

        blocks = [
            alert_block(
                h("Cleared {items}").format(items=" + ".join(result_parts)),
                level="success",
            )
        ]
        if failures:
            blocks.append(
                alert_block(
                    h("{count} files could not be deleted").format(count=len(failures)),
                    level="warning",
                )
            )
            blocks.append(list_block(failures[:5], title=h("Delete Failures")))
        _render_blocks(output, blocks)

        return f"Cleared {total_count} metadata items (memory + disk)"
    else:
        _render_blocks(output, [section_block(body=h("Operation cancelled"))])
        return None


def metadata_clear_exports(
    client: "AgentClient", output: OutputAdapter
) -> Optional[str]:
    """
    Clear export files from workspace/exports/.

    Clears:
    - All files in workspace/exports/ (*.csv, *.tsv, *.xlsx)

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    # Count export files
    export_files = []
    exports_dir = None
    if hasattr(client.data_manager, "workspace_path"):
        exports_dir = Path(client.data_manager.workspace_path) / "exports"
        if exports_dir.exists():
            export_files = [
                f
                for f in exports_dir.iterdir()
                if f.is_file() and f.suffix in {".csv", ".tsv", ".xlsx"}
            ]

    if not export_files:
        _render_blocks(
            output, [alert_block(h("No export files to clear"), level="warning")]
        )
        return "No export files to clear"

    blocks: list[OutputBlock] = [
        kv_block(
            [
                (h("Location"), str(exports_dir)),
                (h("Files"), str(len(export_files))),
            ],
            title=h("About to Clear Export Files"),
        )
    ]
    preview_items = [
        f"{file.name} ({file.stat().st_size / 1024:.1f} KB)"
        for file in export_files[:5]
    ]
    if preview_items:
        blocks.append(list_block(preview_items, title=h("Preview")))
    if len(export_files) > 5:
        blocks.append(
            hint_block(
                h("... and {count} more files").format(count=len(export_files) - 5)
            )
        )
    _render_blocks(output, blocks)

    confirm = output.confirm(
        h("Delete all {count} export files?").format(count=len(export_files))
    )

    if confirm:
        deleted = 0
        failures = []
        for f in export_files:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                failures.append(f"{f.name}: {e}")

        blocks = [
            alert_block(
                h("Deleted {count} export files").format(count=deleted), level="success"
            )
        ]
        if failures:
            blocks.append(
                alert_block(
                    h("{count} files could not be deleted").format(count=len(failures)),
                    level="warning",
                )
            )
            blocks.append(list_block(failures[:5], title=h("Delete Failures")))
        _render_blocks(output, blocks)

        return f"Cleared {deleted} export files"
    else:
        _render_blocks(output, [section_block(body=h("Operation cancelled"))])
        return None


def metadata_clear_all(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Clear ALL metadata: memory, workspace/metadata/, and workspace/exports/.

    This is the most comprehensive clear operation. Equivalent to:
    - /metadata clear + /metadata clear exports

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    metadata_store_count = 0
    if hasattr(client.data_manager, "metadata_store"):
        metadata_store_count = len(client.data_manager.metadata_store)

    current_metadata_count = 0
    if hasattr(client.data_manager, "current_metadata"):
        current_metadata_count = len(client.data_manager.current_metadata)

    disk_files = []
    metadata_dir = None
    if hasattr(client.data_manager, "workspace_path"):
        metadata_dir = Path(client.data_manager.workspace_path) / "metadata"
        if metadata_dir.exists():
            disk_files = list(metadata_dir.glob("*.json"))

    export_files = []
    exports_dir = None
    if hasattr(client.data_manager, "workspace_path"):
        exports_dir = Path(client.data_manager.workspace_path) / "exports"
        if exports_dir.exists():
            export_files = [
                f
                for f in exports_dir.iterdir()
                if f.is_file() and f.suffix in {".csv", ".tsv", ".xlsx"}
            ]

    memory_count = metadata_store_count + current_metadata_count
    disk_count = len(disk_files)
    export_count = len(export_files)
    total_count = memory_count + disk_count + export_count

    if total_count == 0:
        _render_blocks(
            output,
            [
                alert_block(
                    h("Nothing to clear (memory, metadata, exports all empty)"),
                    level="warning",
                )
            ],
        )
        return "Nothing to clear"

    _render_blocks(
        output,
        [
            alert_block(h("This cannot be undone."), level="warning"),
            kv_block(
                [
                    (
                        h("Memory (metadata_store)"),
                        h("{count} entries").format(count=metadata_store_count),
                    ),
                    (
                        h("Memory (current_metadata)"),
                        h("{count} entries").format(count=current_metadata_count),
                    ),
                    (
                        h("Disk (workspace/metadata/)"),
                        h("{count} files").format(count=disk_count),
                    ),
                    (
                        h("Disk (workspace/exports/)"),
                        h("{count} files").format(count=export_count),
                    ),
                    (h("Total"), h("{count} items").format(count=total_count)),
                ],
                title=h("About to Clear All Metadata"),
            ),
        ],
    )
    confirm = output.confirm(
        h("Clear ALL {count} items? This cannot be undone!").format(count=total_count)
    )

    if confirm:
        results = []

        if metadata_store_count > 0 and hasattr(client.data_manager, "metadata_store"):
            client.data_manager.metadata_store.clear()
            results.append(
                h("{count} metadata_store entries").format(count=metadata_store_count)
            )

        if current_metadata_count > 0 and hasattr(
            client.data_manager, "current_metadata"
        ):
            client.data_manager.current_metadata.clear()
            results.append(
                h("{count} current_metadata entries").format(
                    count=current_metadata_count
                )
            )

        deleted_metadata = 0
        metadata_failures = []
        for f in disk_files:
            try:
                f.unlink()
                deleted_metadata += 1
            except Exception as e:
                metadata_failures.append(f"{f.name}: {e}")
        if deleted_metadata > 0:
            results.append(h("{count} metadata files").format(count=deleted_metadata))

        deleted_exports = 0
        export_failures = []
        for f in export_files:
            try:
                f.unlink()
                deleted_exports += 1
            except Exception as e:
                export_failures.append(f"{f.name}: {e}")
        if deleted_exports > 0:
            results.append(h("{count} export files").format(count=deleted_exports))

        blocks = [
            alert_block(
                h("Cleared: {items}").format(items=", ".join(results)), level="success"
            )
        ]
        all_failures = metadata_failures + export_failures
        if all_failures:
            blocks.append(
                alert_block(
                    h("{count} files could not be deleted").format(
                        count=len(all_failures)
                    ),
                    level="warning",
                )
            )
            blocks.append(list_block(all_failures[:5], title=h("Delete Failures")))
        _render_blocks(output, blocks)

        return f"Cleared all metadata ({total_count} items)"
    else:
        _render_blocks(output, [section_block(body=h("Operation cancelled"))])
        return None
