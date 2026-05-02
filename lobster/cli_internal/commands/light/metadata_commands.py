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

    blocks: list[OutputBlock] = [section_block(title="Metadata Overview")]

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

        blocks.append(section_block(title="Publication Queue"))
        if status_rows:
            blocks.append(
                table_block(
                    columns=[{"name": "Status"}, {"name": "Count"}],
                    rows=status_rows,
                )
            )
            blocks.append(
                hint_block(
                    " | ".join(
                        [
                            f"Total: {pq['total']}",
                            f"Workspace-ready: {pq.get('workspace_ready', 0)}",
                            f"Extracted datasets: {pq.get('extracted_datasets', 0)}",
                        ]
                    )
                )
            )

    # Sample Statistics section
    samples = overview.get("samples", {})
    if samples.get("total_samples", 0) > 0:
        sample_rows = [
            ("Total Samples", f"{samples['total_samples']:,}"),
            ("BioProjects", str(samples.get("bioproject_count", 0))),
        ]

        if samples.get("has_aggregated"):
            filtered = samples.get("filtered_samples", 0)
            retention = samples.get("retention_rate", 0)
            sample_rows.append(("Filtered Samples", f"{filtered:,}"))
            sample_rows.append(("Retention", f"{retention:.1f}%"))

            coverage = samples.get("disease_coverage", 0)
            bar = _make_progress_bar(coverage)
            sample_rows.append(("Disease Coverage", f"{bar} {coverage:.1f}%"))
        else:
            blocks.append(
                hint_block("Run metadata filtering to generate aggregated statistics.")
            )
        blocks.append(kv_block(sample_rows, title="Sample Statistics"))

    # Workspace Files section
    workspace = overview.get("workspace", {})
    if workspace.get("metadata_files", 0) > 0 or workspace.get("export_files", 0) > 0:
        workspace_rows = []
        if workspace.get("metadata_files", 0) > 0:
            workspace_rows.append(
                (
                    "Metadata Files",
                    f"{workspace['metadata_files']} ({workspace.get('total_size_mb', 0):.1f} MB)",
                )
            )
        if workspace.get("export_files", 0) > 0:
            workspace_rows.append(("Export Files", str(workspace["export_files"])))
        if workspace.get("in_memory_entries", 0) > 0:
            workspace_rows.append(
                ("In-memory Entries", str(workspace["in_memory_entries"]))
            )
        blocks.append(kv_block(workspace_rows, title="Workspace Files"))

    # Next Steps
    next_steps = overview.get("next_steps", [])
    if next_steps:
        blocks.append(list_block(next_steps, title="Next Steps"))

    # Deprecated warnings
    if overview.get("has_deprecated"):
        blocks.append(
            alert_block(
                "Found files in deprecated metadata/exports/ location. Use /metadata workspace for details.",
                level="warning",
            )
        )

    blocks.append(
        hint_block(
            "Commands: /metadata publications | samples | workspace | exports | clear"
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
                    "No publication queue found. Use research_agent to process publications.",
                    level="warning",
                )
            ],
        )
        return None

    blocks: list[OutputBlock] = [
        section_block(title=f"Publication Queue ({summary['total']} entries)")
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
            title="Status Breakdown",
            columns=[{"name": "Status"}, {"name": "Count"}],
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
                title="Identifier Coverage",
                columns=[{"name": "Type"}, {"name": "Count"}, {"name": "Coverage"}],
                rows=coverage_rows,
            )
        )

    # Extracted datasets
    extracted = summary.get("extracted_datasets", {})
    if extracted:
        blocks.append(
            table_block(
                title="Extracted Identifiers",
                columns=[{"name": "Database"}, {"name": "Count"}],
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
                body=f"Workspace Status: {ws_ready} entries with metadata files"
            )
        )

    # Recent errors
    errors = summary.get("recent_errors", [])
    if errors:
        blocks.append(section_block(title="Recent Errors"))
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
                "Tip: Filter by status with /metadata publications --status=<status>"
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
                    "No sample metadata found. Process publications with research_agent first.",
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
                ("Total Samples", f"{total:,}"),
                ("BioProjects", str(bioproject_count)),
            ],
            title="Sample Statistics",
        )
    ]

    if stats.get("has_aggregated"):
        filtered = stats.get("filtered_samples", 0)
        retention = stats.get("retention_rate", 0)
        coverage = stats.get("disease_coverage", 0)
        bar = _make_progress_bar(coverage, width=20)
        detail_rows = [
            ("Filtered Samples", f"{filtered:,}"),
            ("Retention", f"{retention:.1f}%"),
            ("Disease Coverage", f"{bar} {coverage:.1f}%"),
        ]

        criteria = stats.get("filter_criteria", "")
        if criteria:
            detail_rows.append(("Filter Criteria", criteria))
        blocks.append(kv_block(detail_rows, title="Aggregated Statistics"))

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
                        title="Filter Breakdown",
                        columns=[
                            {"name": "Filter"},
                            {"name": "Retained"},
                            {"name": "Coverage"},
                        ],
                        rows=breakdown_rows,
                    )
                )
        blocks.append(
            section_block(
                body="Aggregated metadata available. Use /metadata exports to see export files."
            )
        )
    else:
        blocks.append(
            alert_block(
                "Samples not yet filtered. Use metadata_assistant to apply filters and generate aggregated statistics.",
                level="warning",
            )
        )

        sources = stats.get("sources", [])
        if sources:
            source_items = list(sources[:10])
            if len(sources) > 10:
                source_items.append(f"... and {len(sources) - 10} more")
            blocks.append(list_block(source_items, title="Sample Sources"))

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
    blocks: list[OutputBlock] = [section_block(title="Workspace Inventory")]

    mem_count = inventory.get("metadata_store_count", 0)
    if mem_count > 0:
        rows = [("Entries", str(mem_count))]
        categories = inventory.get("metadata_store_categories", {})
        for cat, count in sorted(
            categories.items(), key=lambda item: (-item[1], item[0])
        ):
            rows.append((cat, str(count)))
        blocks.append(
            kv_block(
                rows,
                title="In-Memory Store",
            )
        )
    else:
        blocks.append(hint_block("In-memory metadata store is empty."))

    ws_files = inventory.get("workspace_files", {})
    if ws_files:
        total_files = inventory.get("workspace_files_total", 0)
        size_mb = inventory.get("total_size_mb", 0)
        rows = [
            ("Total Files", str(total_files)),
            ("Total Size", f"{size_mb:.1f} MB"),
        ]
        for cat, count in sorted(
            ws_files.items(), key=lambda item: (-item[1], item[0])
        ):
            rows.append((cat, str(count)))
        blocks.append(
            kv_block(
                rows,
                title="Workspace Files",
            )
        )
    else:
        blocks.append(hint_block("No workspace metadata files found."))

    exports = inventory.get("exports", [])
    if exports:
        total_exports = inventory.get("exports_total", 0)
        rows = [
            [exp["name"], f"{exp['size_kb']:.1f} KB", exp["modified"]]
            for exp in exports[:10]
        ]
        blocks.append(
            table_block(
                title=f"Export Files ({total_exports})",
                columns=[
                    {"name": "File"},
                    {"name": "Size"},
                    {"name": "Modified"},
                ],
                rows=rows,
            )
        )
        if len(exports) > 10:
            blocks.append(hint_block(f"... and {total_exports - 10} more files"))

    warnings = inventory.get("deprecated_warnings", [])
    if warnings:
        for warn in warnings:
            blocks.append(alert_block(warn, level="warning"))
        blocks.append(
            hint_block(
                "Consider migrating: mv workspace/metadata/exports/* workspace/exports/"
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
        section_block(title=f"Export Files ({exports['total_count']} files)")
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
                rows, title="File Categories", key_label="Category", value_label="Count"
            )
        )

    files = exports.get("files", [])
    if files:
        blocks.append(
            table_block(
                title="Recent Files",
                columns=[
                    {
                        "name": "File",
                        "style": "cyan",
                        "width": 50,
                        "overflow": "ellipsis",
                    },
                    {"name": "Size", "style": "grey50", "width": 10},
                    {"name": "Modified", "style": "grey50", "width": 18},
                ],
                rows=[
                    [f["name"], f"{f['size_kb']:.1f} KB", f["modified"]]
                    for f in files[:15]
                ],
            )
        )
        if len(files) > 15:
            blocks.append(
                hint_block(f"... and {exports['total_count'] - 15} more files")
            )

    hints = exports.get("usage_hints", {})
    if hints:
        blocks.append(
            kv_block(
                [
                    ("List exports", hints.get("list", "N/A")),
                    ("Access in code", hints.get("access", "N/A")),
                    ("CLI command", hints.get("cli", "N/A")),
                ],
                title="Usage Tips",
                key_label="Action",
                value_label="Command",
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
    blocks: list[OutputBlock] = [section_block(title="Metadata Information")]
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
                    title="Metadata Store",
                    columns=[
                        {"name": "Dataset ID", "style": "bold white"},
                        {"name": "Type", "style": "cyan"},
                        {
                            "name": "Title",
                            "style": "white",
                            "max_width": 40,
                            "overflow": "ellipsis",
                        },
                        {"name": "Samples", "style": "grey74"},
                        {"name": "Cached", "style": "grey50"},
                    ],
                    rows=rows,
                )
            )
            entries_shown += len(metadata_store)
        else:
            blocks.append(hint_block("No cached metadata in metadata store"))

    if (
        hasattr(client.data_manager, "current_metadata")
        and client.data_manager.current_metadata
    ):
        metadata = client.data_manager.current_metadata
        rows = []
        for key, value in metadata.items():
            if isinstance(value, dict):
                display_value = (
                    f"Dict with {len(value)} keys: {', '.join(list(value.keys())[:3])}"
                )
                if len(value) > 3:
                    display_value += f" ... (+{len(value) - 3} more)"
            elif isinstance(value, list):
                display_value = f"List with {len(value)} items"
                if len(value) > 0:
                    display_value += f": {', '.join(str(v) for v in value[:3])}"
                    if len(value) > 3:
                        display_value += f" ... (+{len(value) - 3} more)"
            else:
                display_value = str(value)
                if len(display_value) > 60:
                    display_value = display_value[:60] + "..."

            rows.append((key, display_value))

        blocks.append(
            kv_block(
                rows,
                title="Current Data Metadata",
                key_label="Key",
                value_label="Value",
            )
        )
        entries_shown += len(metadata)
    else:
        blocks.append(hint_block("No current data metadata available"))

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
                    title="Workspace Metadata Files",
                    columns=[
                        {
                            "name": "File",
                            "style": "cyan",
                            "width": 50,
                            "overflow": "ellipsis",
                        },
                        {"name": "Size", "style": "grey50", "width": 10},
                        {"name": "Modified", "style": "grey50", "width": 20},
                    ],
                    rows=rows,
                )
            )

            if len(json_files) > 20:
                blocks.append(hint_block(f"... and {len(json_files) - 20} more files"))
            blocks.append(hint_block(f"Path: {metadata_dir}"))

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
                    title="Export Files",
                    columns=[
                        {
                            "name": "File",
                            "style": "green",
                            "width": 50,
                            "overflow": "ellipsis",
                        },
                        {"name": "Size", "style": "grey50", "width": 10},
                        {"name": "Modified", "style": "grey50", "width": 20},
                    ],
                    rows=rows,
                )
            )

            if len(export_files) > 15:
                blocks.append(
                    hint_block(f"... and {len(export_files) - 15} more files")
                )
            blocks.append(hint_block(f"Path: {exports_dir}"))

    old_exports_dir = workspace_path / "metadata" / "exports"
    if old_exports_dir.exists():
        old_files = list(old_exports_dir.glob("*"))
        if old_files:
            blocks.append(
                alert_block(
                    f"Found {len(old_files)} file(s) in old location: {old_exports_dir}",
                    level="warning",
                )
            )
            blocks.append(hint_block("New exports go to: workspace/exports/"))
            blocks.append(
                hint_block(
                    "Migration: mv workspace/metadata/exports/* workspace/exports/"
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
            [alert_block("Metadata is already empty (memory + disk)", level="warning")],
        )
        return "Metadata already empty"

    _render_blocks(
        output,
        [
            kv_block(
                [
                    ("Memory (metadata_store)", f"{metadata_store_count} entries"),
                    ("Memory (current_metadata)", f"{current_metadata_count} entries"),
                    ("Disk (workspace/metadata/)", f"{disk_count} files"),
                    ("Total", f"{total_count} items"),
                ],
                title="About to Clear",
            )
        ],
    )

    confirm = output.confirm(f"Clear all {total_count} metadata items?")

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
            result_parts.append(f"{cleared_store} metadata_store entries")
        if cleared_current > 0:
            result_parts.append(f"{cleared_current} current_metadata entries")
        if deleted_files > 0:
            result_parts.append(f"{deleted_files} disk files")

        blocks = [alert_block(f"Cleared {' + '.join(result_parts)}", level="success")]
        if failures:
            blocks.append(
                alert_block(
                    f"{len(failures)} files could not be deleted",
                    level="warning",
                )
            )
            blocks.append(list_block(failures[:5], title="Delete Failures"))
        _render_blocks(output, blocks)

        return f"Cleared {total_count} metadata items (memory + disk)"
    else:
        _render_blocks(output, [section_block(body="Operation cancelled")])
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
            output, [alert_block("No export files to clear", level="warning")]
        )
        return "No export files to clear"

    blocks: list[OutputBlock] = [
        kv_block(
            [
                ("Location", str(exports_dir)),
                ("Files", str(len(export_files))),
            ],
            title="About to Clear Export Files",
        )
    ]
    preview_items = [
        f"{file.name} ({file.stat().st_size / 1024:.1f} KB)"
        for file in export_files[:5]
    ]
    if preview_items:
        blocks.append(list_block(preview_items, title="Preview"))
    if len(export_files) > 5:
        blocks.append(hint_block(f"... and {len(export_files) - 5} more files"))
    _render_blocks(output, blocks)

    confirm = output.confirm(f"Delete all {len(export_files)} export files?")

    if confirm:
        deleted = 0
        failures = []
        for f in export_files:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                failures.append(f"{f.name}: {e}")

        blocks = [alert_block(f"Deleted {deleted} export files", level="success")]
        if failures:
            blocks.append(
                alert_block(
                    f"{len(failures)} files could not be deleted",
                    level="warning",
                )
            )
            blocks.append(list_block(failures[:5], title="Delete Failures"))
        _render_blocks(output, blocks)

        return f"Cleared {deleted} export files"
    else:
        _render_blocks(output, [section_block(body="Operation cancelled")])
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
                    "Nothing to clear (memory, metadata, exports all empty)",
                    level="warning",
                )
            ],
        )
        return "Nothing to clear"

    _render_blocks(
        output,
        [
            alert_block("This cannot be undone.", level="warning"),
            kv_block(
                [
                    ("Memory (metadata_store)", f"{metadata_store_count} entries"),
                    ("Memory (current_metadata)", f"{current_metadata_count} entries"),
                    ("Disk (workspace/metadata/)", f"{disk_count} files"),
                    ("Disk (workspace/exports/)", f"{export_count} files"),
                    ("Total", f"{total_count} items"),
                ],
                title="About to Clear All Metadata",
            ),
        ],
    )
    confirm = output.confirm(f"Clear ALL {total_count} items? This cannot be undone!")

    if confirm:
        results = []

        if metadata_store_count > 0 and hasattr(client.data_manager, "metadata_store"):
            client.data_manager.metadata_store.clear()
            results.append(f"{metadata_store_count} metadata_store entries")

        if current_metadata_count > 0 and hasattr(
            client.data_manager, "current_metadata"
        ):
            client.data_manager.current_metadata.clear()
            results.append(f"{current_metadata_count} current_metadata entries")

        deleted_metadata = 0
        metadata_failures = []
        for f in disk_files:
            try:
                f.unlink()
                deleted_metadata += 1
            except Exception as e:
                metadata_failures.append(f"{f.name}: {e}")
        if deleted_metadata > 0:
            results.append(f"{deleted_metadata} metadata files")

        deleted_exports = 0
        export_failures = []
        for f in export_files:
            try:
                f.unlink()
                deleted_exports += 1
            except Exception as e:
                export_failures.append(f"{f.name}: {e}")
        if deleted_exports > 0:
            results.append(f"{deleted_exports} export files")

        blocks = [alert_block(f"Cleared: {', '.join(results)}", level="success")]
        all_failures = metadata_failures + export_failures
        if all_failures:
            blocks.append(
                alert_block(
                    f"{len(all_failures)} files could not be deleted",
                    level="warning",
                )
            )
            blocks.append(list_block(all_failures[:5], title="Delete Failures"))
        _render_blocks(output, blocks)

        return f"Cleared all metadata ({total_count} items)"
    else:
        _render_blocks(output, [section_block(body="Operation cancelled")])
        return None
