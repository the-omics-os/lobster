"""
Shared queue commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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

# Import status display configurations (single source of truth)
from lobster.core.schemas.download_queue import DOWNLOAD_STATUS_DISPLAY
from lobster.core.schemas.publication_queue import PUBLICATION_STATUS_DISPLAY
from lobster.i18n import h


class QueueFileTypeNotSupported(Exception):
    """Raised when unsupported file type is provided to queue load."""

    pass


def _render_blocks(output: OutputAdapter, blocks: list[OutputBlock]) -> None:
    output.render_blocks(blocks)


def _status_label(status_name: str, display_config: Dict[str, tuple]) -> str:
    icon, _style = display_config.get(status_name, ("?", "dim"))
    display_name = status_name.replace("_", " ").title()
    return f"{icon} {display_name}"


def _resolve_queue_file_path(
    client: "AgentClient", filename: str, current_directory: Optional[Path]
) -> tuple[Optional[Path], Optional[list[OutputBlock]]]:
    if current_directory:
        from lobster.cli_internal.utils.path_resolution import PathResolver

        resolver = PathResolver(
            current_directory=current_directory,
            workspace_path=(
                client.data_manager.workspace_path
                if hasattr(client, "data_manager")
                else None
            ),
        )
        resolved = resolver.resolve(filename, search_workspace=True, must_exist=True)

        if not resolved.is_safe:
            return None, [
                alert_block(
                    h("Security error: {error}").format(error=resolved.error),
                    level="error",
                )
            ]

        if not resolved.exists:
            return None, [
                alert_block(
                    h("File not found: {filename}").format(filename=filename),
                    level="error",
                )
            ]

        return resolved.path, None

    file_path = Path(filename)
    if not file_path.is_absolute():
        file_path = client.data_manager.workspace_path / filename

    if not file_path.exists():
        return None, [
            alert_block(
                h("File not found: {filename}").format(filename=filename), level="error"
            )
        ]

    return file_path, None


def _get_download_queue_stats(client: "AgentClient") -> Tuple[Dict, int]:
    """
    Get download queue statistics.

    Args:
        client: AgentClient instance

    Returns:
        Tuple of (stats dict, total count)
    """
    if not hasattr(client, "data_manager") or client.data_manager is None:
        return {}, 0

    queue = client.data_manager.download_queue
    if queue is None:
        return {}, 0

    stats = queue.get_statistics()
    return stats, stats.get("total_entries", 0)


def _get_publication_queue_stats(client: "AgentClient") -> Tuple[Dict, int]:
    """
    Get publication queue statistics.

    Args:
        client: AgentClient instance

    Returns:
        Tuple of (stats dict, total count)
    """
    if client.publication_queue is None:
        return {}, 0

    stats = client.publication_queue.get_statistics()
    return stats, stats.get("total_entries", 0)


def _build_status_table(
    by_status: Dict[str, int],
    display_config: Dict[str, tuple],
    total: int,
) -> OutputBlock:
    """
    Build table data for queue status display.

    Args:
        by_status: Status -> count mapping
        display_config: Status -> (icon, style) mapping
        total: Total entry count

    Returns:
        Structured table block.
    """
    rows = []
    for status_name, count in by_status.items():
        rows.append([_status_label(status_name, display_config), str(count)])

    # Add total row
    rows.append([h("Total"), str(total)])

    return table_block(
        columns=[
            {"name": h("Status"), "style": "cyan"},
            {"name": h("Count"), "style": "white", "justify": "right"},
        ],
        rows=rows,
    )


def show_queue_status(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Display status of download and publication queues.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    # Get statistics from both queues
    dl_stats, dl_total = _get_download_queue_stats(client)
    pub_stats, pub_total = _get_publication_queue_stats(client)

    # Check if any queue is available
    has_download_queue = bool(dl_stats)
    has_publication_queue = bool(pub_stats)

    if not has_download_queue and not has_publication_queue:
        _render_blocks(
            output, [alert_block(h("No queues initialized"), level="warning")]
        )
        return None
    blocks: list[OutputBlock] = [section_block(title=h("Queue Status"))]

    blocks.append(section_block(title=h("Download Queue")))
    if has_download_queue:
        dl_rows = [
            [_status_label(status_name, DOWNLOAD_STATUS_DISPLAY), str(count)]
            for status_name, count in dl_stats.get("by_status", {}).items()
        ]
        dl_rows.append([h("Total"), str(dl_total)])
        blocks.append(
            table_block(
                title=h("Status Breakdown"),
                columns=[{"name": h("Status")}, {"name": h("Count")}],
                rows=dl_rows,
            )
        )
        by_database = dl_stats.get("by_database", {})
        if by_database:
            db_items = [f"{db}: {cnt}" for db, cnt in by_database.items() if cnt > 0]
            if db_items:
                blocks.append(
                    hint_block(
                        h("Databases: {items}").format(items=", ".join(db_items))
                    )
                )
    else:
        blocks.append(hint_block(h("Not initialized")))

    blocks.append(section_block(title=h("Publication Queue")))
    if has_publication_queue:
        pub_rows = [
            [_status_label(status_name, PUBLICATION_STATUS_DISPLAY), str(count)]
            for status_name, count in pub_stats.get("by_status", {}).items()
        ]
        pub_rows.append([h("Total"), str(pub_total)])
        blocks.append(
            table_block(
                title=h("Status Breakdown"),
                columns=[{"name": h("Status")}, {"name": h("Count")}],
                rows=pub_rows,
            )
        )
        by_level = pub_stats.get("by_extraction_level", {})
        if by_level:
            level_items = [f"{lvl}: {cnt}" for lvl, cnt in by_level.items() if cnt > 0]
            if level_items:
                blocks.append(
                    hint_block(
                        h("Extraction levels: {items}").format(
                            items=", ".join(level_items)
                        )
                    )
                )
        ids_extracted = pub_stats.get("identifiers_extracted", 0)
        if ids_extracted > 0:
            blocks.append(
                hint_block(
                    h("Identifiers extracted: {count}").format(count=ids_extracted)
                )
            )
    else:
        blocks.append(hint_block(h("Not initialized")))

    blocks.extend(
        [
            list_block(
                [
                    f"/queue load <file> - {h('Load .ris file into publication queue')}",
                    f"/queue list - {h('List publication queue items')}",
                    f"/queue list download - {h('List download queue items')}",
                    f"/queue clear - {h('Clear publication queue')}",
                    f"/queue clear download - {h('Clear download queue')}",
                    f"/queue clear all - {h('Clear all queues')}",
                    f"/queue export - {h('Export publication queue to workspace')}",
                ],
                title=h("Commands"),
            )
        ]
    )
    _render_blocks(output, blocks)

    # Build summary
    grand_total = dl_total + pub_total
    return f"Queue status: {dl_total} downloads, {pub_total} publications ({grand_total} total)"


def queue_load_file(
    client: "AgentClient",
    filename: str,
    output: OutputAdapter,
    current_directory: Optional[Path] = None,
) -> Optional[str]:
    """
    Load file into queue - type determines handler.

    Args:
        client: AgentClient instance
        filename: File path to load
        output: OutputAdapter for rendering
        current_directory: Current working directory (optional for dashboard)

    Returns:
        Summary string for conversation history, or None

    Raises:
        QueueFileTypeNotSupported: For unsupported file types
    """
    if not filename:
        _render_blocks(
            output, [alert_block(h("Usage: /queue load <file>"), level="warning")]
        )
        return None

    # Resolve file path
    if current_directory:
        # CLI mode: use PathResolver for secure resolution
        from lobster.cli_internal.utils.path_resolution import PathResolver

        resolver = PathResolver(
            current_directory=current_directory,
            workspace_path=(
                client.data_manager.workspace_path
                if hasattr(client, "data_manager")
                else None
            ),
        )
        resolved = resolver.resolve(filename, search_workspace=True, must_exist=True)

        if not resolved.is_safe:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Security error: {error}").format(error=resolved.error),
                        level="error",
                    )
                ],
            )
            return None

        if not resolved.exists:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("File not found: {filename}").format(filename=filename),
                        level="error",
                    )
                ],
            )
            return None

        file_path = resolved.path
    else:
        # Dashboard mode: treat as absolute or workspace-relative path
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = client.data_manager.workspace_path / filename

        if not file_path.exists():
            _render_blocks(
                output,
                [
                    alert_block(
                        h("File not found: {filename}").format(filename=filename),
                        level="error",
                    )
                ],
            )
            return None

    ext = file_path.suffix.lower()

    # Supported: .ris files
    if ext in [".ris", ".txt"]:
        _render_blocks(
            output,
            [
                section_block(
                    title=h("Loading into queue: {name}").format(name=file_path.name)
                )
            ],
        )

        try:
            result = client.load_publication_list(
                file_path=str(file_path),
                priority=5,
                schema_type="general",
                extraction_level="methods",
            )

            added = result["added_count"]
            duplicates = result.get("duplicate_count", 0)
            file_dups = result.get("file_duplicates", 0)
            queue_dups = result.get("queue_duplicates", 0)
            malformed = result.get("skipped_count", 0)

            if added > 0:
                blocks: list[OutputBlock] = [
                    section_block(
                        body=h("Loaded {count} items into queue").format(count=added)
                    )
                ]

                # Show deduplication info (informational, not warning)
                if duplicates > 0:
                    dup_details = []
                    if file_dups > 0:
                        dup_details.append(
                            h("{count} duplicates in file").format(count=file_dups)
                        )
                    if queue_dups > 0:
                        dup_details.append(
                            h("{count} already in queue").format(count=queue_dups)
                        )
                    blocks.append(
                        hint_block(
                            h("Deduplicated: {items}").format(
                                items=", ".join(dup_details)
                            )
                        )
                    )

                if malformed > 0:
                    blocks.append(
                        alert_block(
                            h("Skipped {count} malformed entries").format(
                                count=malformed
                            ),
                            level="warning",
                        )
                    )

                blocks.append(
                    list_block(
                        [
                            h("Extract methods and parameters"),
                            h("Search for related datasets (GEO)"),
                            h("Build citation network"),
                            h("Custom analysis (describe your intent)"),
                        ],
                        title=h("Next Steps"),
                    )
                )
                _render_blocks(output, blocks)

                return f"Loaded {added} publications into queue from {file_path.name}. Awaiting user intent."

            elif duplicates > 0:
                # All entries were duplicates - not an error, just informational
                blocks = [
                    section_block(
                        body=h(
                            "No new items added - all {count} entries already in queue or duplicated"
                        ).format(count=duplicates)
                    )
                ]
                if queue_dups > 0:
                    blocks.append(
                        hint_block(
                            h(
                                "{queue} already in queue, {file} duplicates in file"
                            ).format(queue=queue_dups, file=file_dups)
                        )
                    )
                _render_blocks(output, blocks)
                return f"No new publications added from {file_path.name} - all entries were duplicates."

            else:
                blocks = [
                    alert_block(h("No items could be loaded from file"), level="error")
                ]
                if result.get("errors"):
                    blocks.append(
                        list_block(
                            [str(error) for error in result["errors"][:3]],
                            title=h("Errors"),
                        )
                    )
                _render_blocks(output, blocks)
                return None

        except Exception as e:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Failed to load file: {error}").format(error=str(e)),
                        level="error",
                    )
                ],
            )
            return None

    # Placeholder: .bib files (BibTeX)
    elif ext == ".bib":
        raise QueueFileTypeNotSupported(
            h(
                "BibTeX (.bib) support coming soon. Convert to .ris format or wait for future release."
            )
        )

    # Placeholder: .csv files (custom lists)
    elif ext == ".csv":
        raise QueueFileTypeNotSupported(
            h(
                "CSV queue loading coming soon. Expected format: columns for DOI, PMID, or title."
            )
        )

    # Placeholder: .json files (API exports)
    elif ext == ".json":
        raise QueueFileTypeNotSupported(
            h("JSON queue loading coming soon. Planned support for PubMed API exports.")
        )

    # Unknown type
    else:
        raise QueueFileTypeNotSupported(
            h(
                "Unsupported file type: {ext}. Currently supported: .ris. Coming soon: .bib, .csv, .json"
            ).format(ext=ext)
        )


def queue_list(
    client: "AgentClient", output: OutputAdapter, queue_type: str = "publication"
) -> Optional[str]:
    """
    List items in the specified queue.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        queue_type: "publication" or "download"

    Returns:
        Summary string for conversation history, or None
    """
    if queue_type == "download":
        return _queue_list_download(client, output)
    else:
        return _queue_list_publication(client, output)


def _queue_list_publication(
    client: "AgentClient", output: OutputAdapter
) -> Optional[str]:
    """
    List items in the publication queue.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    if client.publication_queue is None:
        _render_blocks(
            output,
            [alert_block(h("Publication queue not initialized"), level="warning")],
        )
        return None

    entries = client.publication_queue.list_entries()

    if not entries:
        _render_blocks(
            output, [alert_block(h("Publication queue is empty"), level="warning")]
        )
        return "Publication queue is empty"

    # Limit display to first 20 entries
    display_entries = entries[:20]
    total_count = len(entries)

    rows = []

    for i, entry in enumerate(display_entries, 1):
        title = (
            entry.title[:47] + "..."
            if entry.title and len(entry.title) > 50
            else (entry.title or "N/A")
        )
        year = str(entry.year) if entry.year else "N/A"
        status = (
            entry.status.value if hasattr(entry.status, "value") else str(entry.status)
        )
        # Get icon for status
        icon, _style = PUBLICATION_STATUS_DISPLAY.get(status, ("?", "dim"))
        status_display = f"{icon} {status}"
        identifier = entry.pmid or entry.doi or "N/A"
        rows.append([str(i), title, year, status_display, identifier])

    blocks: list[OutputBlock] = [
        section_block(
            title=h("Publication Queue ({shown} of {total} shown)").format(
                shown=len(display_entries), total=total_count
            )
        ),
        table_block(
            columns=[
                {"name": "#", "width": 4},
                {"name": h("Title")},
                {"name": h("Year"), "width": 6},
                {"name": h("Status")},
                {"name": "PMID/DOI"},
            ],
            rows=rows,
        ),
    ]
    if total_count > 20:
        blocks.append(
            hint_block(h("... and {count} more items").format(count=total_count - 20))
        )
    _render_blocks(output, blocks)

    return f"Listed {len(display_entries)} of {total_count} publication queue items"


def _queue_list_download(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    List items in the download queue.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    if not hasattr(client, "data_manager") or client.data_manager is None:
        _render_blocks(
            output, [alert_block(h("Data manager not initialized"), level="warning")]
        )
        return None

    download_queue = client.data_manager.download_queue
    if download_queue is None:
        _render_blocks(
            output, [alert_block(h("Download queue not initialized"), level="warning")]
        )
        return None

    entries = download_queue.list_entries()

    if not entries:
        _render_blocks(
            output, [alert_block(h("Download queue is empty"), level="warning")]
        )
        return "Download queue is empty"

    # Limit display to first 20 entries
    display_entries = entries[:20]
    total_count = len(entries)

    rows = []

    for i, entry in enumerate(display_entries, 1):
        accession = entry.dataset_id or "N/A"
        database = entry.database or "N/A"
        status = (
            entry.status.value if hasattr(entry.status, "value") else str(entry.status)
        )
        # Get icon for status
        icon, _style = DOWNLOAD_STATUS_DISPLAY.get(status, ("?", "dim"))
        status_display = f"{icon} {status}"

        # Get strategy info
        strategy = "N/A"
        if entry.recommended_strategy and entry.recommended_strategy.strategy_name:
            strategy = entry.recommended_strategy.strategy_name

        priority = str(entry.priority) if entry.priority else "5"

        rows.append([str(i), accession, database, status_display, strategy, priority])

    # Show summary by status
    stats = download_queue.get_statistics()
    by_status = stats.get("by_status", {})
    status_summary = []
    for status_name, count in by_status.items():
        if count > 0:
            icon, _style = DOWNLOAD_STATUS_DISPLAY.get(status_name, ("?", "dim"))
            status_summary.append(f"{icon} {status_name}: {count}")

    blocks = [
        section_block(
            title=h("Download Queue ({shown} of {total} shown)").format(
                shown=len(display_entries), total=total_count
            )
        ),
        table_block(
            columns=[
                {"name": "#", "width": 4},
                {"name": h("Accession")},
                {"name": h("Database")},
                {"name": h("Status")},
                {"name": h("Strategy")},
                {"name": h("Priority")},
            ],
            rows=rows,
        ),
    ]
    if total_count > 20:
        blocks.append(
            hint_block(h("... and {count} more items").format(count=total_count - 20))
        )
    if status_summary:
        blocks.append(
            hint_block(h("Summary: {items}").format(items=" | ".join(status_summary)))
        )
    _render_blocks(output, blocks)

    return f"Listed {len(display_entries)} of {total_count} download queue items"


def queue_clear(
    client: "AgentClient", output: OutputAdapter, queue_type: str = "publication"
) -> Optional[str]:
    """
    Clear items from specified queue(s).

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        queue_type: "publication", "download", or "all"

    Returns:
        Summary string for conversation history, or None
    """
    if queue_type == "all":
        # Clear both queues
        pub_total = 0
        dl_total = 0

        # Get publication queue stats
        if client.publication_queue is not None:
            pub_stats = client.publication_queue.get_statistics()
            pub_total = pub_stats.get("total_entries", 0)

        # Get download queue stats
        if hasattr(client, "data_manager") and client.data_manager:
            dl_stats = client.data_manager.download_queue.get_statistics()
            dl_total = dl_stats.get("total_entries", 0)

        grand_total = pub_total + dl_total

        if grand_total == 0:
            _render_blocks(
                output,
                [alert_block(h("All queues are already empty"), level="warning")],
            )
            return "All queues were already empty"

        # Confirm with user
        _render_blocks(
            output,
            [
                kv_block(
                    [
                        ("Publication queue", f"{pub_total} items"),
                        ("Download queue", f"{dl_total} items"),
                        (h("Total"), f"{grand_total} items"),
                    ],
                    title=h("About to Clear"),
                )
            ],
        )
        confirm = output.confirm(
            h("Clear all {count} items from both queues?").format(count=grand_total)
        )

        if confirm:
            cleared = []
            if pub_total > 0:
                client.publication_queue.clear_queue()
                cleared.append(f"{h('publication')} ({pub_total})")
            if dl_total > 0:
                client.data_manager.download_queue.clear_queue()
                cleared.append(f"{h('download')} ({dl_total})")

            _render_blocks(
                output,
                [
                    alert_block(
                        h("Cleared {count} items from {queues} queues").format(
                            count=grand_total, queues=", ".join(cleared)
                        ),
                        level="success",
                    )
                ],
            )
            return f"Cleared {grand_total} items from all queues"
        else:
            _render_blocks(output, [section_block(body=h("Operation cancelled"))])
            return None

    elif queue_type == "download":
        # Clear download queue
        if not hasattr(client, "data_manager") or not client.data_manager:
            _render_blocks(
                output,
                [alert_block(h("Data manager not initialized"), level="warning")],
            )
            return None

        download_queue = client.data_manager.download_queue
        stats = download_queue.get_statistics()
        total = stats.get("total_entries", 0)

        if total == 0:
            _render_blocks(
                output,
                [alert_block(h("Download queue is already empty"), level="warning")],
            )
            return "Download queue was already empty"

        # Confirm with user
        confirm = output.confirm(
            h("Clear all {count} items from download queue?").format(count=total)
        )

        if confirm:
            download_queue.clear_queue()
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Cleared {count} items from download queue").format(
                            count=total
                        ),
                        level="success",
                    )
                ],
            )
            return f"Cleared {total} items from download queue"
        else:
            _render_blocks(output, [section_block(body=h("Operation cancelled"))])
            return None

    else:  # publication (default)
        if client.publication_queue is None:
            _render_blocks(
                output,
                [alert_block(h("Publication queue not initialized"), level="warning")],
            )
            return None

        # Get count before clearing
        stats = client.publication_queue.get_statistics()
        total = stats.get("total_entries", 0)

        if total == 0:
            _render_blocks(
                output,
                [alert_block(h("Publication queue is already empty"), level="warning")],
            )
            return "Publication queue was already empty"

        # Confirm with user
        confirm = output.confirm(
            h("Clear all {count} items from publication queue?").format(count=total)
        )

        if confirm:
            client.publication_queue.clear_queue()
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Cleared {count} items from publication queue").format(
                            count=total
                        ),
                        level="success",
                    )
                ],
            )
            return f"Cleared {total} items from publication queue"
        else:
            _render_blocks(output, [section_block(body=h("Operation cancelled"))])
            return None


def queue_export(
    client: "AgentClient", name: Optional[str], output: OutputAdapter
) -> Optional[str]:
    """
    Export queue to workspace for persistence.

    Args:
        client: AgentClient instance
        name: Optional name for the exported dataset
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    if client.publication_queue is None:
        _render_blocks(
            output,
            [alert_block(h("Publication queue not initialized"), level="warning")],
        )
        return None

    stats = client.publication_queue.get_statistics()
    if stats.get("total_entries", 0) == 0:
        _render_blocks(
            output,
            [alert_block(h("Queue is empty, nothing to export"), level="warning")],
        )
        return None

    # Generate export name if not provided
    if not name:
        from datetime import datetime

        name = f"queue_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    _render_blocks(
        output,
        [
            section_block(
                body=h("Exporting queue to workspace as '{name}'...").format(name=name)
            )
        ],
    )

    try:
        # Export queue data by copying the queue file to workspace
        source_path = client.publication_queue.queue_file
        export_path = client.data_manager.workspace_path / f"{name}.jsonl"

        # Copy the queue file
        shutil.copy2(source_path, export_path)

        _render_blocks(
            output,
            [
                alert_block(
                    h("Exported {count} items to: {path}").format(
                        count=stats.get("total_entries", 0), path=export_path
                    ),
                    level="success",
                )
            ],
        )
        return f"Exported {stats.get('total_entries', 0)} queue items to workspace as '{name}'"
    except Exception as e:
        _render_blocks(
            output,
            [
                alert_block(
                    h("Export failed: {error}").format(error=str(e)), level="error"
                )
            ],
        )
        return None


def _check_metadata_files(
    entries: List,
    workspace_path: Path,
) -> Tuple[int, List[str]]:
    """
    Check which workspace_metadata_keys have missing files.

    Args:
        entries: List of PublicationQueueEntry objects
        workspace_path: Path to workspace directory

    Returns:
        Tuple of (missing_count, list of missing file basenames)
    """
    metadata_dir = workspace_path / "metadata"
    missing_files = []

    for entry in entries:
        if hasattr(entry, "workspace_metadata_keys") and entry.workspace_metadata_keys:
            for key in entry.workspace_metadata_keys:
                file_path = metadata_dir / key
                if not file_path.exists():
                    missing_files.append(key)

    return len(missing_files), missing_files


def queue_import(
    client: "AgentClient",
    filename: str,
    output: OutputAdapter,
    current_directory: Optional[Path] = None,
) -> Optional[str]:
    """
    Import previously exported queue from JSONL file.

    Args:
        client: AgentClient instance
        filename: Path to JSONL file to import
        output: OutputAdapter for rendering
        current_directory: Current working directory (optional for dashboard)

    Returns:
        Summary string for conversation history, or None
    """
    import json

    from lobster.core.schemas.publication_queue import PublicationQueueEntry

    if not filename:
        _render_blocks(
            output,
            [alert_block(h("Usage: /queue import <file.jsonl>"), level="warning")],
        )
        return None

    # Resolve file path
    if current_directory:
        # CLI mode: use PathResolver for secure resolution
        from lobster.cli_internal.utils.path_resolution import PathResolver

        resolver = PathResolver(
            current_directory=current_directory,
            workspace_path=(
                client.data_manager.workspace_path
                if hasattr(client, "data_manager")
                else None
            ),
        )
        resolved = resolver.resolve(filename, search_workspace=True, must_exist=True)

        if not resolved.is_safe:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Security error: {error}").format(error=resolved.error),
                        level="error",
                    )
                ],
            )
            return None

        if not resolved.exists:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("File not found: {filename}").format(filename=filename),
                        level="error",
                    )
                ],
            )
            return None

        file_path = resolved.path
    else:
        # Dashboard mode: treat as absolute or workspace-relative path
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = client.data_manager.workspace_path / filename

        if not file_path.exists():
            _render_blocks(
                output,
                [
                    alert_block(
                        h("File not found: {filename}").format(filename=filename),
                        level="error",
                    )
                ],
            )
            return None

    ext = file_path.suffix.lower()

    # Only support .jsonl files for import
    if ext != ".jsonl":
        _render_blocks(
            output,
            [
                alert_block(
                    h("Unsupported file type: {ext}").format(ext=ext), level="error"
                ),
                section_block(
                    body=h(
                        "The /queue import command only accepts .jsonl files (exported via /queue export)."
                    )
                ),
                hint_block(h("To load .ris files, use: /queue load <file.ris>")),
            ],
        )
        return None

    if client.publication_queue is None:
        _render_blocks(
            output, [alert_block(h("Publication queue not initialized"), level="error")]
        )
        return None

    _render_blocks(
        output,
        [
            section_block(
                body=h("Importing queue from: {name}").format(name=file_path.name)
            )
        ],
    )

    # Parse JSONL file
    entries = []
    parse_blocks: list[OutputBlock] = []
    parse_errors = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    entry = PublicationQueueEntry.from_dict(data)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    parse_blocks.append(
                        alert_block(
                            h("Line {line}: Invalid JSON - {error}").format(
                                line=line_num, error=e
                            ),
                            level="warning",
                        )
                    )
                except Exception as e:
                    parse_errors += 1
                    parse_blocks.append(
                        alert_block(
                            h("Line {line}: Invalid entry - {error}").format(
                                line=line_num, error=e
                            ),
                            level="warning",
                        )
                    )

        if parse_blocks:
            _render_blocks(output, parse_blocks)

        if not entries:
            _render_blocks(
                output,
                [alert_block(h("No valid entries found in file"), level="warning")],
            )
            return None

        # Import entries to queue
        result = client.publication_queue.import_entries(entries, skip_duplicates=True)

        imported = result.get("imported", 0)
        skipped = result.get("skipped", 0)
        errors = result.get("errors", 0)

        # Display results
        result_blocks: list[OutputBlock] = []
        if imported > 0:
            result_blocks.append(
                alert_block(
                    h("Imported {count} entries into queue").format(count=imported),
                    level="success",
                )
            )

        if skipped > 0:
            result_blocks.append(
                alert_block(
                    h("Skipped {count} duplicate entries (already in queue)").format(
                        count=skipped
                    ),
                    level="warning",
                )
            )

        if errors > 0 or parse_errors > 0:
            total_errors = errors + parse_errors
            result_blocks.append(
                alert_block(
                    h("{count} entries failed to import").format(count=total_errors),
                    level="error",
                )
            )

        # Check for missing metadata files
        if imported > 0 and hasattr(client, "data_manager"):
            workspace_path = client.data_manager.workspace_path
            missing_count, missing_files = _check_metadata_files(
                entries, workspace_path
            )

            if missing_count > 0:
                result_blocks.append(
                    alert_block(
                        h(
                            "{count} referenced metadata files not found in workspace"
                        ).format(count=missing_count),
                        level="warning",
                    )
                )
                # Show first few missing files
                preview = missing_files[:5]
                result_blocks.append(
                    list_block(preview, title=h("Missing Metadata Files"))
                )
                if missing_count > 5:
                    result_blocks.append(
                        hint_block(
                            h("... and {count} more").format(count=missing_count - 5)
                        )
                    )

                result_blocks.append(
                    hint_block(
                        h(
                            "Tip: Re-run processing for affected entries to regenerate metadata."
                        )
                    )
                )

        if result_blocks:
            _render_blocks(output, result_blocks)

        # Build summary
        summary_parts = []
        if imported > 0:
            summary_parts.append(f"imported {imported}")
        if skipped > 0:
            summary_parts.append(f"skipped {skipped} duplicates")
        if errors + parse_errors > 0:
            summary_parts.append(f"{errors + parse_errors} errors")

        return f"Queue import complete: {', '.join(summary_parts)}"

    except Exception as e:
        _render_blocks(
            output,
            [
                alert_block(
                    h("Import failed: {error}").format(error=str(e)), level="error"
                )
            ],
        )
        return None
