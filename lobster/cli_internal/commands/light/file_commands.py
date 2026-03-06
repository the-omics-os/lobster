"""
Shared file commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

import glob as glob_module
import itertools
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import (
    OutputAdapter,
    OutputBlock,
    alert_block,
    code_block,
    hint_block,
    kv_block,
    list_block,
    section_block,
)
from lobster.cli_internal.utils.path_resolution import PathResolver

# NOTE: component_registry is imported lazily in archive_queue() to avoid
# triggering heavy dependency loads (pandas/numpy) at module import time.
# This keeps light commands fast (<300ms startup).


def build_read_usage_blocks() -> list[OutputBlock]:
    return [
        section_block(body="/read views file contents without loading data into memory."),
        section_block(body="Usage: /read <filename>"),
        hint_block("Examples: /read my_data.h5ad, /read config.yaml, /read data/*.csv"),
        hint_block("To load data for analysis, use /workspace load <name>."),
    ]


def _render_blocks(output: OutputAdapter, blocks: list[OutputBlock]) -> None:
    output.render_blocks(blocks)


def file_read(
    client: "AgentClient",
    output: OutputAdapter,
    filename: str,
    current_directory: Path,
    path_resolver_class=PathResolver,
) -> Optional[str]:
    """
    Read workspace files with multiple format support.

    Handles text files, code files, data files, archives, and glob patterns.
    For non-text files, provides file info and guidance on how to load them.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        filename: File path or glob pattern to read
        current_directory: Current working directory
        path_resolver_class: PathResolver class for secure path resolution (default: PathResolver)

    Returns:
        Summary string for conversation history, or None

    Features:
        - Glob pattern support (*.py, data/*.csv, etc.)
        - Syntax highlighting for code files
        - Security via PathResolver
        - Binary file detection
        - File size limits (10MB for text display)
        - Suggestions for loading data files
    """
    if not filename:
        output.render_blocks(
            [
                alert_block("Usage: /read <file|pattern>", level="warning"),
                section_block(body="View file contents (text files only)."),
                hint_block("Use /workspace load <file> to load data files."),
            ]
        )
        return None

    # Check if filename contains glob patterns (before path resolution)
    is_glob_pattern = any(char in filename for char in ["*", "?", "[", "]"])

    # BUG FIX #6: Use PathResolver for secure path resolution (non-glob paths)
    if not is_glob_pattern:
        resolver = path_resolver_class(
            current_directory=current_directory,
            workspace_path=(
                client.data_manager.workspace_path
                if hasattr(client, "data_manager")
                else None
            ),
        )
        resolved = resolver.resolve(filename, search_workspace=True, must_exist=False)

        if not resolved.is_safe:
            _render_blocks(
                output,
                [alert_block(f"Security error: {resolved.error}", level="error")],
            )
            return None

        file_path = resolved.path
    else:
        # Glob patterns need special handling - construct search pattern
        file_path = (
            Path(filename)
            if Path(filename).is_absolute()
            else current_directory / filename
        )

    if is_glob_pattern:
        # Handle glob patterns - show contents of matching text files
        if not Path(filename).is_absolute():
            base_path = current_directory
            search_pattern = str(base_path / filename)
        else:
            search_pattern = filename

        # BUG FIX #3: Use lazy evaluation to prevent memory explosion
        # Only load first 10 file paths instead of all matches
        matching_files = list(itertools.islice(glob_module.iglob(search_pattern), 10))

        if not matching_files:
            _render_blocks(
                output,
                [
                    alert_block(
                        f"No files found matching pattern: {filename}",
                        level="error",
                    ),
                    section_block(body=f"Searched in: {current_directory}"),
                ],
            )
            return None

        # Count total matches without loading all paths
        total_count = sum(1 for _ in glob_module.iglob(search_pattern))

        matching_files.sort()
        blocks: list[OutputBlock] = [
            section_block(
                body=f"Found {total_count} files matching '{filename}', displaying first 10."
            )
        ]

        displayed_count = 0
        for match_path in matching_files:  # Already limited to 10
            match_file = Path(match_path)
            file_info = client.detect_file_type(match_file)

            # Only display text files
            if not file_info.get("binary", True):
                try:
                    # BUG FIX #3: Add file size check before reading (10MB limit)
                    file_size = match_file.stat().st_size
                    if file_size > 10_000_000:  # 10MB
                        blocks.append(
                            alert_block(
                                (
                                    f"{match_file.name} is too large to display "
                                    f"({file_size / 1_000_000:.1f}MB, limit: 10MB)."
                                ),
                                level="warning",
                            )
                        )
                        continue

                    content = match_file.read_text(encoding="utf-8")
                    lines = content.splitlines()

                    # Language detection
                    ext = match_file.suffix.lower()
                    language_map = {
                        ".py": "python",
                        ".js": "javascript",
                        ".ts": "typescript",
                        ".html": "html",
                        ".css": "css",
                        ".json": "json",
                        ".xml": "xml",
                        ".yaml": "yaml",
                        ".yml": "yaml",
                        ".sh": "bash",
                        ".bash": "bash",
                        ".md": "markdown",
                        ".txt": "text",
                        ".log": "text",
                        ".r": "r",
                        ".csv": "csv",
                        ".tsv": "csv",
                        ".ris": "text",
                    }
                    language = language_map.get(ext, "text")

                    blocks.extend(
                        [
                            section_block(body=f"{match_file.name} ({len(lines)} lines)"),
                            code_block(content, language=language),
                        ]
                    )
                    displayed_count += 1
                except Exception as e:
                    blocks.append(
                        alert_block(
                            f"Could not read {match_file.name}: {e}",
                            level="warning",
                        )
                    )
            else:
                blocks.append(
                    hint_block(f"{match_file.name} (binary file - skipped)")
                )

        if total_count > 10:
            blocks.append(
                hint_block(f"... and {total_count - 10} more files (not loaded)")
            )

        _render_blocks(output, blocks)

        return f"Displayed {displayed_count} text files matching '{filename}' (total: {total_count})"

    # Single file processing
    if not file_path.exists():
        # Try to locate via client (searches workspace directories)
        file_info = client.locate_file(filename)
        if not file_info["found"]:
            blocks = [alert_block(file_info["error"], level="error")]
            if "searched_paths" in file_info:
                blocks.extend(
                    [
                        section_block(body="Searched in:"),
                        list_block(
                            [str(path) for path in file_info["searched_paths"][:5]]
                        ),
                    ]
                )
            _render_blocks(output, blocks)
            return f"File '{filename}' not found"
        file_path = file_info["path"]

    # Get file info
    file_info = client.detect_file_type(file_path)
    file_description = file_info.get("description", "Unknown")
    file_category = file_info.get("category", "unknown")
    is_binary = file_info.get("binary", True)

    output.render_blocks(
        [
            kv_block(
                [
                    ("Name", file_path.name),
                    ("Path", str(file_path)),
                    ("Type", file_description),
                ],
                title="File",
            )
        ]
    )

    # Handle text files - display content
    if not is_binary:
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()

            # Language detection
            ext = file_path.suffix.lower()
            language_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".html": "html",
                ".css": "css",
                ".json": "json",
                ".xml": "xml",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".sh": "bash",
                ".bash": "bash",
                ".md": "markdown",
                ".txt": "text",
                ".log": "text",
                ".r": "r",
                ".csv": "csv",
                ".tsv": "csv",
                ".ris": "text",
            }
            language = language_map.get(ext, "text")

            # Display using code block
            output.render_blocks(
                [
                    section_block(body=f"Contents: {file_path.name}"),
                    code_block(content, language=language),
                ]
            )

            return f"Displayed text file '{filename}' ({file_description}, {len(lines)} lines)"

        except UnicodeDecodeError:
            _render_blocks(
                output,
                [
                    alert_block(
                        "File appears to be binary despite its extension.",
                        level="warning",
                    )
                ],
            )
            is_binary = True
        except Exception as e:
            _render_blocks(
                output,
                [alert_block(f"Error reading file: {e}", level="error")],
            )
            return f"Error reading file '{filename}': {str(e)}"

    # Handle binary/data files - show info only, suggest /workspace load
    if is_binary:
        # Format file size
        size_bytes = file_path.stat().st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} bytes"
        elif size_bytes < 1024**2:
            size_str = f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024**3:
            size_str = f"{size_bytes / 1024**2:.1f} MB"
        else:
            size_str = f"{size_bytes / 1024**3:.1f} GB"

        output.render_blocks(
            [
                kv_block(
                    [
                        ("Name", file_path.name),
                        ("Path", str(file_path)),
                        ("Type", file_description),
                        ("Category", file_category),
                        ("Size", size_str),
                    ],
                    title="File Info",
                )
            ]
        )

        # Provide guidance based on file type
        if file_category == "bioinformatics":
            output.render_blocks(
                [
                    section_block(
                        body=f"This is a bioinformatics data file ({file_description})."
                    ),
                    hint_block(
                        f"To load it into the workspace: /workspace load {filename}"
                    ),
                ]
            )
        elif file_category == "tabular":
            output.render_blocks(
                [
                    section_block(
                        body=f"This is a tabular data file ({file_description})."
                    ),
                    hint_block(
                        f"To load it into the workspace: /workspace load {filename}"
                    ),
                ]
            )
        elif file_category == "archive":
            output.render_blocks(
                [
                    section_block(body=f"This is an archive file ({file_description})."),
                    hint_block(f"To extract and load it: /workspace load {filename}"),
                ]
            )
        elif file_category == "image":
            output.render_blocks(
                [
                    section_block(
                        body="This is an image file. Use your system image viewer to open it."
                    )
                ]
            )
        else:
            output.render_blocks(
                [section_block(body="Binary file - use external tools to view it.")]
            )

        return f"Inspected file '{filename}' ({file_description}, {size_str}) - use /workspace load to load data files"


def archive_queue(
    client: "AgentClient",
    output: OutputAdapter,
    subcommand: str = "help",
    args: Optional[str] = None,
) -> Optional[str]:
    """
    Archive queue functionality for cached extractions.

    Manages extraction cache for nested archive files, allowing users to
    list, load, and inspect cached archive contents without re-extracting.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        subcommand: Archive subcommand (list, groups, load, status, cleanup, help)
        args: Additional arguments for subcommands (e.g., pattern for load, limit flags)

    Returns:
        Summary string for conversation history, or None

    Subcommands:
        - list: Show all samples in cached archive
        - groups: Show condition groups summary
        - load <pattern>: Load samples by pattern/GSM ID/condition
        - status: Show extraction cache status
        - cleanup: Clear old cached extractions (>7 days)
        - help: Show usage information

    Premium Feature:
        Archive caching requires premium distribution.
    """
    # BUG FIX #1: Handle nested archive commands with proper cache management
    # Use ExtractionCacheManager instead of client instance variables to prevent race conditions

    # Lazy import to avoid loading heavy dependencies at module import time
    from lobster.core.component_registry import component_registry

    ExtractionCacheManager = component_registry.get_service("extraction_cache")

    # Check if extraction cache is available (premium feature)
    if ExtractionCacheManager is None:
        output.print(
            "[yellow]Archive caching is a premium feature not available in this distribution.[/yellow]",
            style="warning",
        )
        output.print(
            "[dim]Use extract_and_load_archive() to load entire archives instead.[/dim]",
            style="info",
        )
        return None

    # Initialize cache manager (thread-safe, per-request instance)
    cache_manager = ExtractionCacheManager(client.data_manager.workspace_path)
    recent_caches = cache_manager.list_all_caches()

    if not recent_caches:
        output.print("[red]❌ No cached archives found[/red]", style="error")
        output.print(
            "[yellow]💡 Run /read <archive.tar> first to inspect an archive[/yellow]",
            style="warning",
        )
        return None

    # Select cache: use most recent if only one, otherwise prompt user
    if len(recent_caches) == 1:
        cache_id = recent_caches[0]["cache_id"]
    else:
        # Multiple caches available - show list and use most recent
        output.print(
            f"\n[cyan]📦 Found {len(recent_caches)} cached archives (using most recent):[/cyan]",
            style="info",
        )
        for i, cache in enumerate(recent_caches[:3], 1):
            age_hours = (time.time() - cache.get("timestamp", 0)) / 3600
            output.print(
                f"  {i}. {cache['cache_id']} ({age_hours:.1f}h ago)", style="info"
            )
        cache_id = recent_caches[0]["cache_id"]  # Most recent

    # Get cache info and nested_info
    cache_info = cache_manager.get_cache_info(cache_id)
    if not cache_info:
        output.print(
            f"[red]❌ Cache {cache_id} metadata not found[/red]", style="error"
        )
        return None

    nested_info = cache_info.get("nested_info")
    if not nested_info:
        output.print(
            f"[red]❌ Cache {cache_id} missing nested structure info[/red]",
            style="error",
        )
        return None

    if subcommand == "list":
        # Show detailed list of all nested samples
        output.print("\n[bold white]📋 Archive Contents:[/bold white]", style="info")
        output.print(f"[dim]Cache ID: {cache_id}[/dim]\n", style="info")

        samples_table_data = {
            "title": "All Samples",
            "columns": [
                {"name": "GSM ID", "style": "bold orange1"},
                {"name": "Condition", "style": "white"},
                {"name": "Number", "style": "grey70"},
                {"name": "Filename", "style": "dim"},
            ],
            "rows": [],
        }

        for condition, samples in nested_info.groups.items():
            for sample in samples:
                samples_table_data["rows"].append(
                    [
                        sample["gsm_id"],
                        condition,
                        sample["number"],
                        sample["filename"],
                    ]
                )

        output.print_table(samples_table_data)
        return f"Listed {nested_info.total_count} samples from cached archive"

    elif subcommand == "groups":
        # Show condition groups summary (nested_info already loaded above)
        output.print("\n[bold white]📂 Condition Groups:[/bold white]\n", style="info")

        groups_table_data = {
            "title": None,
            "columns": [
                {"name": "Condition", "style": "bold orange1"},
                {"name": "Sample Count", "style": "white"},
                {"name": "GSM IDs", "style": "grey70"},
            ],
            "rows": [],
        }

        for condition, samples in nested_info.groups.items():
            gsm_ids = [s["gsm_id"] for s in samples]
            groups_table_data["rows"].append(
                [
                    condition,
                    str(len(samples)),
                    f"{min(gsm_ids)}-{max(gsm_ids)}" if gsm_ids else "N/A",
                ]
            )

        output.print_table(groups_table_data)
        return f"Displayed {len(nested_info.groups)} condition groups"

    elif subcommand == "load":
        # Load samples by pattern
        if not args:
            output.print(
                "[yellow]Usage: /archive load <pattern|GSM_ID|condition>[/yellow]",
                style="warning",
            )
            output.print("[dim]Examples:[/dim]", style="info")
            output.print("[dim]  /archive load GSM4710689[/dim]", style="info")
            output.print("[dim]  /archive load TISSUE[/dim]", style="info")
            output.print("[dim]  /archive load PDAC_* --limit 3[/dim]", style="info")
            return None

        pattern_arg = args

        # Parse limit flag
        limit = None
        if "--limit" in pattern_arg:
            pattern, limit_str = pattern_arg.split("--limit")
            pattern = pattern.strip()
            try:
                limit = int(limit_str.strip())
            except ValueError:
                output.print("[red]❌ Invalid limit value[/red]", style="error")
                return None
        else:
            pattern = pattern_arg

        output.print(
            f"[cyan]🔄 Loading samples matching '[bold]{pattern}[/bold]'...[/cyan]",
            style="info",
        )

        # Note: Using status context is not supported by OutputAdapter abstraction
        # Real implementation would need client.load_from_cache to handle this
        result = client.load_from_cache(cache_id, pattern, limit)

        if result["success"]:
            output.print(f"\n[green]✓ {result['message']}[/green]", style="success")

            # Display merged dataset if auto-concatenation occurred
            if "merged_modality" in result:
                merged_name = result["merged_modality"]

                # Get merged dataset details
                try:
                    merged_adata = client.data_manager.get_modality(merged_name)

                    # Create prominent merged dataset panel info
                    merged_info = f"""[bold white]Merged Dataset:[/bold white] [orange1]{merged_name}[/orange1]

[white]Shape:[/white] [cyan]{merged_adata.n_obs:,} cells × {merged_adata.n_vars:,} genes[/cyan]
[white]Batches:[/white] [cyan]{result["loaded_count"]} samples merged[/cyan]
[white]Batch key:[/white] [cyan]sample_id[/cyan]

[bold white]🎯 Ready for Analysis![/bold white]
[grey70]  • Say: "Show me a UMAP of this dataset"[/grey70]
[grey70]  • Say: "Perform quality control"[/grey70]
[grey70]  • Or use: /data to inspect the dataset[/grey70]"""

                    output.print(
                        f"\n[bold green]✨ Auto-Merged Dataset[/bold green]\n{merged_info}",
                        style="success",
                    )

                except Exception as e:
                    output.print(
                        f"\n[yellow]⚠️  Could not display merged dataset details: {e}[/yellow]",
                        style="warning",
                    )

                # Show individual modalities in collapsed format
                output.print(
                    f"\n[dim]Individual modalities (merged into '{merged_name}'):[/dim]",
                    style="info",
                )
                for i, modality in enumerate(result["modalities"][:5], 1):
                    output.print(f"  [dim]{i}. {modality}[/dim]", style="info")
                if len(result["modalities"]) > 5:
                    output.print(
                        f"  [dim]... and {len(result['modalities']) - 5} more[/dim]",
                        style="info",
                    )

            else:
                # Single sample or no auto-concatenation
                output.print(
                    "\n[bold white]Loaded Modalities:[/bold white]", style="info"
                )
                for modality in result["modalities"]:
                    output.print(f"  • [cyan]{modality}[/cyan]", style="info")

                # Suggest next steps
                output.print("\n[bold white]🎯 Next Steps:[/bold white]", style="info")
                output.print(
                    "[grey70]  • Use /data to inspect the dataset[/grey70]",
                    style="info",
                )
                output.print(
                    "[grey70]  • Say: 'Analyze this dataset' for natural language analysis[/grey70]",
                    style="info",
                )

            if result["failed"]:
                output.print(
                    f"\n[yellow]⚠️  Failed to load {len(result['failed'])} samples:[/yellow]",
                    style="warning",
                )
                for failed in result["failed"][:5]:
                    output.print(f"  • [dim]{failed}[/dim]", style="info")
                if len(result["failed"]) > 5:
                    output.print(
                        f"  • [dim]... and {len(result['failed']) - 5} more[/dim]",
                        style="info",
                    )

            # Return summary
            if "merged_modality" in result:
                return f"Merged {result['loaded_count']} samples into '{result['merged_modality']}'"
            else:
                return f"Loaded {result['loaded_count']} samples: {', '.join(result['modalities'][:3])}{'...' if len(result['modalities']) > 3 else ''}"

        else:
            output.print(f"\n[red]❌ {result['error']}[/red]", style="error")
            if "suggestion" in result:
                output.print(
                    f"[yellow]💡 {result['suggestion']}[/yellow]", style="warning"
                )
            return f"Failed to load samples: {result['error']}"

    elif subcommand == "status":
        # Show cache status (uses top-level import, already checked HAS_EXTRACTION_CACHE)
        cache_manager = ExtractionCacheManager(client.workspace_path)
        all_caches = cache_manager.list_all_caches()

        output.print(
            "\n[bold white]📊 Extraction Cache Status:[/bold white]\n", style="info"
        )
        output.print(
            f"[white]Total cached extractions: [yellow]{len(all_caches)}[/yellow][/white]",
            style="info",
        )

        if all_caches:
            cache_table_data = {
                "title": None,
                "columns": [
                    {"name": "Cache ID", "style": "bold orange1"},
                    {"name": "Archive", "style": "white"},
                    {"name": "Samples", "style": "yellow"},
                    {"name": "Extracted At", "style": "dim"},
                ],
                "rows": [],
            }

            for cache in all_caches:
                extracted_at = datetime.fromisoformat(cache["extracted_at"])
                cache_table_data["rows"].append(
                    [
                        cache["cache_id"],
                        Path(cache["archive_path"]).name,
                        str(cache["nested_info"]["total_count"]),
                        extracted_at.strftime("%Y-%m-%d %H:%M"),
                    ]
                )

            output.print_table(cache_table_data)

        return f"Cache status: {len(all_caches)} active extractions"

    elif subcommand == "cleanup":
        # Clean up old caches (uses top-level import, already checked HAS_EXTRACTION_CACHE)
        cache_manager = ExtractionCacheManager(client.workspace_path)

        output.print(
            "[cyan]🧹 Cleaning up old cached extractions...[/cyan]", style="info"
        )
        removed_count = cache_manager.cleanup_old_caches(max_age_days=7)

        output.print(
            f"[green]✓ Removed {removed_count} old cache(s)[/green]", style="success"
        )
        return f"Cleaned up {removed_count} old cached extractions"

    else:
        # Show help
        output.print("\n[bold white]📦 /archive Commands:[/bold white]\n", style="info")
        output.print(
            "[orange1]/archive list[/orange1]             - List all samples in inspected archive",
            style="info",
        )
        output.print(
            "[orange1]/archive groups[/orange1]           - Show condition groups",
            style="info",
        )
        output.print(
            "[orange1]/archive load <pattern>[/orange1]   - Load samples by pattern",
            style="info",
        )
        output.print(
            "[orange1]/archive status[/orange1]           - Show extraction cache status",
            style="info",
        )
        output.print(
            "[orange1]/archive cleanup[/orange1]          - Clear old cached extractions\n",
            style="info",
        )

        output.print("[bold white]Loading Patterns:[/bold white]", style="info")
        output.print("[grey70]• GSM ID:[/grey70]        GSM4710689", style="info")
        output.print(
            "[grey70]• Condition:[/grey70]     TISSUE, PDAC_TISSUE", style="info"
        )
        output.print(
            "[grey70]• Glob:[/grey70]          PDAC_*, *_TISSUE_*", style="info"
        )
        output.print("[grey70]• With limit:[/grey70]    TISSUE --limit 3", style="info")

        return None
