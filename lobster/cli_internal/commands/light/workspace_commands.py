"""
Shared workspace commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import (
    OutputAdapter,
    OutputBlock,
    alert_block,
    hint_block,
    list_block,
    section_block,
    table_block,
)
from lobster.i18n import h


def truncate_middle(text: str, max_length: int = 60) -> str:
    """
    Truncate text in the middle with ellipsis, preserving start and end.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text with middle ellipsis
    """
    if len(text) <= max_length:
        return text

    # Calculate how much to keep on each side
    # Reserve 3 characters for "..."
    available_chars = max_length - 3
    start_length = (available_chars + 1) // 2  # Slightly prefer start
    end_length = available_chars // 2

    return f"{text[:start_length]}...{text[-end_length:]}"


def _render_blocks(output: OutputAdapter, blocks: list[OutputBlock]) -> None:
    output.render_blocks(blocks)


def _build_workspace_list_blocks(
    available: dict, loaded: set[str]
) -> list[OutputBlock]:
    rows = []
    for idx, (name, info) in enumerate(sorted(available.items()), start=1):
        status = h("Loaded") if name in loaded else h("Available")
        size = f"{info['size_mb']:.1f} MB"
        shape = (
            f"{info['shape'][0]:,} × {info['shape'][1]:,}" if info["shape"] else "N/A"
        )
        modified = info["modified"].split("T")[0]
        rows.append(
            [
                str(idx),
                status,
                truncate_middle(name, max_length=60),
                size,
                shape,
                modified,
            ]
        )

    return [
        table_block(
            title=h("Available Datasets"),
            columns=[
                {"name": "#", "style": "dim", "width": 4},
                {"name": h("Status"), "style": "green", "width": 10},
                {"name": h("Name"), "style": "bold", "no_wrap": False},
                {"name": h("Size"), "style": "cyan", "width": 10},
                {"name": h("Shape"), "style": "white", "width": 15},
                {"name": h("Modified"), "style": "dim", "width": 12},
            ],
            rows=rows,
        ),
        hint_block(h("Use '/workspace info <#>' to see full details.")),
    ]


def _build_workspace_info_blocks(
    matched_datasets: list[tuple[str, dict]],
    loaded: set[str],
) -> list[OutputBlock]:
    blocks: list[OutputBlock] = []
    for name, info in matched_datasets:
        rows = [
            [h("Name"), name],
            [h("Status"), h("Loaded") if name in loaded else h("Not Loaded")],
            [h("Path"), info["path"]],
            [h("Size"), f"{info['size_mb']:.2f} MB"],
            [
                h("Shape"),
                (
                    h("{obs} observations × {vars} variables").format(
                        obs=f"{info['shape'][0]:,}", vars=f"{info['shape'][1]:,}"
                    )
                    if info["shape"]
                    else "N/A"
                ),
            ],
            [h("Type"), info["type"]],
            [h("Modified"), info["modified"]],
        ]

        if "_" in name:
            parts_list = name.split("_")
            possible_stages = [
                part
                for part in parts_list
                if any(
                    keyword in part.lower()
                    for keyword in [
                        "quality",
                        "filter",
                        "normal",
                        "doublet",
                        "cluster",
                        "marker",
                        "annot",
                        "pseudobulk",
                    ]
                )
            ]
            if possible_stages:
                rows.append([h("Processing Stages"), " -> ".join(possible_stages)])

        blocks.append(
            table_block(
                title=h("Dataset: {name}").format(name=name),
                columns=[
                    {"name": h("Property"), "style": "bold cyan"},
                    {"name": h("Value"), "style": "white"},
                ],
                rows=rows,
            )
        )
    return blocks


def _build_workspace_status_blocks(
    workspace_status_dict: dict,
    modality_detail_rows: list[tuple[str, list[list[str]]]],
    *,
    compact: bool = False,
    available_dataset_count: Optional[int] = None,
) -> list[OutputBlock]:
    max_visible_modalities = 12
    max_detail_tables = 3
    workspace_path = workspace_status_dict.get("workspace_path", "N/A")
    modalities_count = workspace_status_dict.get("modalities_loaded", 0)
    provenance = (
        h("Enabled")
        if workspace_status_dict.get("provenance_enabled")
        else h("Disabled")
    )
    mudata = (
        h("Available")
        if workspace_status_dict.get("mudata_available")
        else h("Not installed")
    )

    blocks: list[OutputBlock] = [
        section_block(title=h("Workspace Status")),
        table_block(
            title=h("Summary"),
            columns=[{"name": h("Field")}, {"name": h("Value")}],
            rows=[
                [h("Workspace"), workspace_path],
                *(
                    [[h("Datasets Available"), str(available_dataset_count)]]
                    if available_dataset_count is not None
                    else []
                ),
                [h("Modalities Loaded"), str(modalities_count)],
                [h("Provenance"), provenance],
                ["MuData", mudata],
            ],
        ),
    ]

    if workspace_status_dict.get("directories"):
        dir_icons = {
            "data": "Data",
            "exports": "Exports",
            "cache": "Cache",
            "literature_cache": "Literature Cache",
            "metadata": "Metadata",
            "notebooks": "Notebooks",
            "queues": "Queues",
        }
        dir_rows = []
        for dir_type, path in workspace_status_dict["directories"].items():
            file_count, size_str, exists = _get_directory_stats(path)
            display_path = truncate_middle(path, 45)
            if not exists:
                display_path = h("{path} (not created)").format(path=display_path)
            dir_rows.append(
                [
                    dir_icons.get(dir_type, dir_type.replace("_", " ").title()),
                    str(file_count) if exists else "-",
                    size_str,
                    display_path,
                ]
            )
        if compact:
            preferred_dirs = {"Data", "Queues", "Metadata", "Exports"}
            compact_dir_rows = [
                row[:3]
                for row in dir_rows
                if row[0] in preferred_dirs or row[1] != "-" or row[2] != "0 B"
            ]
            if not compact_dir_rows:
                compact_dir_rows = [row[:3] for row in dir_rows[:4]]
            compact_dir_rows = compact_dir_rows[:4]
            blocks.append(
                table_block(
                    title=h("Directories"),
                    columns=[
                        {"name": h("Directory"), "style": "bold white"},
                        {
                            "name": h("Files"),
                            "style": "cyan",
                            "justify": "right",
                            "width": 8,
                        },
                        {
                            "name": h("Size"),
                            "style": "green",
                            "justify": "right",
                            "width": 10,
                        },
                    ],
                    rows=compact_dir_rows,
                )
            )
            if len(dir_rows) > len(compact_dir_rows):
                blocks.append(
                    hint_block(
                        h("... and {count} more workspace directories").format(
                            count=len(dir_rows) - len(compact_dir_rows)
                        )
                    )
                )
        else:
            blocks.append(
                table_block(
                    title=h("Directories"),
                    columns=[
                        {"name": h("Directory"), "style": "bold white"},
                        {"name": h("Files"), "style": "cyan", "justify": "right"},
                        {"name": h("Size"), "style": "green", "justify": "right"},
                        {"name": h("Path"), "style": "grey70"},
                    ],
                    rows=dir_rows,
                )
            )

    modality_names = workspace_status_dict.get("modality_names", [])
    if modality_names:
        visible_modalities = modality_names[:max_visible_modalities]
        blocks.append(
            list_block(
                visible_modalities,
                title=h("Active Modalities") if compact else h("Loaded Modalities"),
            )
        )
        if len(modality_names) > max_visible_modalities:
            remaining = len(modality_names) - max_visible_modalities
            blocks.append(
                hint_block(
                    h("... and {count} more loaded modalities").format(count=remaining)
                )
            )
    else:
        blocks.append(hint_block(h("No modalities currently loaded")))

    if compact:
        blocks.append(
            hint_block(
                h(
                    "Use `/workspace list` for dataset inventory and `/workspace info <#>` for file details."
                )
            )
        )
        return blocks

    backends = workspace_status_dict.get("registered_backends", [])
    adapters = workspace_status_dict.get("registered_adapters", [])
    blocks.append(
        table_block(
            title=h("System Capabilities"),
            columns=[{"name": h("Field")}, {"name": h("Value")}],
            rows=[
                [h("Backends"), ", ".join(backends) if backends else h("None")],
                [h("Adapters"), ", ".join(adapters[:5]) if adapters else h("None")],
            ],
        )
    )
    if len(adapters) > 5:
        blocks.append(
            hint_block(
                h("... and {count} more adapters").format(count=len(adapters) - 5)
            )
        )

    if modality_detail_rows:
        blocks.append(section_block(title=h("Modality Details")))
        visible_detail_rows = modality_detail_rows[:max_detail_tables]
        for modality_name, rows in visible_detail_rows:
            blocks.append(
                table_block(
                    title=modality_name,
                    columns=[
                        {"name": h("Property"), "style": "bold grey93"},
                        {"name": h("Value"), "style": "white"},
                    ],
                    rows=rows,
                )
            )
        if len(modality_detail_rows) > max_detail_tables:
            total = len(modality_detail_rows)
            blocks.append(
                hint_block(
                    h(
                        "Showing {shown} of {total} modalities. Use `/describe <modality>` or `/workspace info <name>` for the rest."
                    ).format(shown=max_detail_tables, total=total)
                )
            )

    return blocks


def _build_workspace_load_result_blocks(result: dict) -> list[OutputBlock]:
    restored = [str(name) for name in result.get("restored", [])]
    skipped = [str(name) for name in result.get("skipped", [])]
    total_size_mb = float(result.get("total_size_mb", 0.0))

    if restored:
        blocks: list[OutputBlock] = [
            section_block(
                body=h("Loaded {count} datasets ({size} MB)").format(
                    count=len(restored), size=f"{total_size_mb:.1f}"
                )
            ),
            list_block(restored, title=h("Loaded Datasets")),
        ]
        if skipped:
            blocks.extend(
                [
                    hint_block(
                        h(
                            "Skipped {count} dataset(s) that were already loaded or unavailable."
                        ).format(count=len(skipped))
                    ),
                    list_block(skipped, title=h("Skipped Datasets")),
                ]
            )
        return blocks

    blocks = [alert_block(h("No datasets loaded"), level="warning")]
    if skipped:
        blocks.extend(
            [
                hint_block(
                    h(
                        "Skipped {count} dataset(s) that were already loaded or unavailable."
                    ).format(count=len(skipped))
                ),
                list_block(skipped, title=h("Skipped Datasets")),
            ]
        )
    return blocks


def workspace_list(
    client: "AgentClient", output: OutputAdapter, force_refresh: bool = False
) -> Optional[str]:
    """
    List available datasets in workspace.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        force_refresh: Force refresh of workspace scan (default: False)

    Returns:
        Summary string for conversation history, or None
    """
    # BUG FIX #2: Use cached scan instead of explicit rescan (75% faster)
    if hasattr(client.data_manager, "get_available_datasets"):
        available = client.data_manager.get_available_datasets(
            force_refresh=force_refresh
        )
    else:
        # Fallback for older DataManager versions
        if hasattr(client.data_manager, "_scan_workspace"):
            client.data_manager._scan_workspace()
        available = client.data_manager.available_datasets

    loaded = set(client.data_manager.modalities.keys())

    if not available:
        # Handle empty case with helpful information
        workspace_path = client.data_manager.workspace_path
        data_dir = workspace_path / "data"
        blocks: list[OutputBlock] = [
            alert_block(h("No datasets found in workspace"), level="warning"),
            section_block(body=h("Workspace: {path}").format(path=workspace_path)),
            section_block(body=h("Data directory: {path}").format(path=data_dir)),
        ]

        if not data_dir.exists():
            blocks.append(
                alert_block(h("Data directory does not exist"), level="error")
            )
            blocks.append(
                section_block(
                    body=h("Create it with: mkdir -p {path}").format(path=data_dir)
                )
            )
        else:
            # Check what files are actually in the data directory
            files = list(data_dir.glob("*"))
            if files:
                blocks.append(
                    section_block(
                        body=(
                            h(
                                "Found {count} files in data directory, but none are supported datasets (.h5ad)."
                            ).format(count=len(files))
                        )
                    )
                )
                blocks.append(
                    list_block([f.name for f in files[:5]], title=h("Files found"))
                )
                if len(files) > 5:
                    blocks.append(
                        hint_block(
                            h("... and {count} more").format(count=len(files) - 5)
                        )
                    )
            else:
                blocks.append(
                    section_block(
                        body=h("Add .h5ad files to {path} to see them here.").format(
                            path=data_dir
                        )
                    )
                )

        _render_blocks(output, blocks)
        return "No datasets found in workspace"
    _render_blocks(output, _build_workspace_list_blocks(available, loaded))
    return f"Listed {len(available)} available datasets"


def workspace_info(
    client: "AgentClient", output: OutputAdapter, selector: str
) -> Optional[str]:
    """
    Show detailed information for specific dataset(s).

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        selector: Index (#) or pattern to match datasets

    Returns:
        Summary string for conversation history, or None
    """
    if not selector:
        _render_blocks(
            output,
            [
                alert_block(h("Usage: /workspace info <#|pattern>"), level="error"),
                hint_block(
                    h(
                        "Examples: /workspace info 1, /workspace info gse12345, /workspace info *clustered*"
                    )
                ),
            ],
        )
        return None

    # BUG FIX #2: Use cached scan for info command
    if hasattr(client.data_manager, "get_available_datasets"):
        available = client.data_manager.get_available_datasets(force_refresh=False)
    else:
        # Fallback for older DataManager versions
        if hasattr(client.data_manager, "_scan_workspace"):
            client.data_manager._scan_workspace()
        available = client.data_manager.available_datasets

    loaded = set(client.data_manager.modalities.keys())

    if not available:
        _render_blocks(
            output,
            [alert_block(h("No datasets found in workspace"), level="warning")],
        )
        return None

    # Determine if selector is an index or pattern
    matched_datasets = []

    if selector.isdigit():
        # Index-based selection
        idx = int(selector)
        sorted_names = sorted(available.keys())
        if 1 <= idx <= len(sorted_names):
            matched_datasets = [
                (sorted_names[idx - 1], available[sorted_names[idx - 1]])
            ]
        else:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Index {idx} out of range (1-{max})").format(
                            idx=idx, max=len(sorted_names)
                        ),
                        level="error",
                    )
                ],
            )
            return None
    else:
        # Pattern-based selection
        for name, info in sorted(available.items()):
            if fnmatch.fnmatch(name.lower(), selector.lower()):
                matched_datasets.append((name, info))

        if not matched_datasets:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("No datasets match pattern: {selector}").format(
                            selector=selector
                        ),
                        level="warning",
                    )
                ],
            )
            return None

    _render_blocks(output, _build_workspace_info_blocks(matched_datasets, loaded))
    return f"Displayed details for {len(matched_datasets)} dataset(s)"


def workspace_load(
    client: "AgentClient",
    output: OutputAdapter,
    selector: str,
    current_directory: Path,
    path_resolver_class,
) -> Optional[str]:
    """
    Load specific datasets by index, pattern, or file path.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        selector: Index (#), pattern, or file path
        current_directory: Current working directory for path resolution
        path_resolver_class: PathResolver class for secure path resolution

    Returns:
        Summary string for conversation history, or None
    """
    if not selector:
        _render_blocks(
            output,
            [
                alert_block(
                    h("Usage: /workspace load <#|pattern|file>"), level="error"
                ),
                hint_block(
                    h(
                        "Examples: /workspace load 1, /workspace load recent, /workspace load data.h5ad"
                    )
                ),
            ],
        )
        return None

    # BUG FIX #6: Use PathResolver for secure path resolution
    resolver = path_resolver_class(
        current_directory=current_directory,
        workspace_path=(
            client.data_manager.workspace_path
            if hasattr(client, "data_manager")
            else None
        ),
    )
    resolved = resolver.resolve(selector, search_workspace=True, must_exist=False)

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

    file_path = resolved.path

    if file_path.exists() and file_path.is_file():
        # Load file directly into workspace
        _render_blocks(
            output,
            [
                section_block(
                    body=h("Loading file into workspace: {name}").format(
                        name=file_path.name
                    )
                )
            ],
        )

        try:
            result = client.load_data_file(str(file_path))

            if result.get("success"):
                _render_blocks(
                    output,
                    [
                        section_block(
                            body=(
                                h("Loaded '{name}' ({rows} × {cols})").format(
                                    name=result["modality_name"],
                                    rows=f"{result['data_shape'][0]:,}",
                                    cols=f"{result['data_shape'][1]:,}",
                                )
                            )
                        )
                    ],
                )
                return f"Loaded file '{file_path.name}' as modality '{result['modality_name']}'"
            blocks = [alert_block(result.get("error", "Unknown error"), level="error")]
            if result.get("suggestion"):
                blocks.append(section_block(body=result["suggestion"]))
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

    # BUG FIX #2: Use cached scan for load command
    if hasattr(client.data_manager, "get_available_datasets"):
        available = client.data_manager.get_available_datasets(force_refresh=False)
    else:
        # Fallback for older DataManager versions
        if hasattr(client.data_manager, "_scan_workspace"):
            client.data_manager._scan_workspace()
        available = client.data_manager.available_datasets

    if not available:
        _render_blocks(
            output,
            [
                alert_block(h("No datasets found in workspace"), level="warning"),
                hint_block(
                    h(
                        "Tip: If '{selector}' is a file, ensure the path is correct."
                    ).format(selector=selector)
                ),
            ],
        )
        return None

    # Determine if selector is an index or pattern
    if selector.isdigit():
        # Index-based loading (single dataset)
        idx = int(selector)
        sorted_names = sorted(available.keys())
        if 1 <= idx <= len(sorted_names):
            dataset_name = sorted_names[idx - 1]

            _render_blocks(
                output,
                [
                    section_block(
                        body=h("Loading dataset: {name}...").format(name=dataset_name)
                    )
                ],
            )

            # Load single dataset directly
            success = client.data_manager.load_dataset(dataset_name)

            if success:
                _render_blocks(
                    output,
                    [
                        section_block(
                            body=(
                                h("Loaded dataset: {name} ({size} MB)").format(
                                    name=dataset_name,
                                    size=f"{available[dataset_name]['size_mb']:.1f}",
                                )
                            )
                        )
                    ],
                )
                return "Loaded dataset from workspace"
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Failed to load dataset: {name}").format(name=dataset_name),
                        level="error",
                    )
                ],
            )
            return None
        _render_blocks(
            output,
            [
                alert_block(
                    h("Index {idx} out of range (1-{max})").format(
                        idx=idx, max=len(sorted_names)
                    ),
                    level="error",
                )
            ],
        )
        return None
    # Pattern-based loading (potentially multiple datasets)
    _render_blocks(
        output,
        [
            section_block(
                body=h("Loading workspace datasets (pattern: {selector})...").format(
                    selector=selector
                )
            )
        ],
    )

    # Note: Progress bar creation is CLI-specific, so we skip it here
    # The CLI layer can add progress bars if needed

    # Perform workspace loading
    result = client.data_manager.restore_session(selector)

    # Display results
    if result["restored"]:
        _render_blocks(output, _build_workspace_load_result_blocks(result))
        return f"Loaded {len(result['restored'])} datasets from workspace"

    _render_blocks(output, _build_workspace_load_result_blocks(result))
    return None


def workspace_remove(
    client: "AgentClient", output: OutputAdapter, selector: str
) -> Optional[str]:
    """
    Remove modality(ies) from memory by index, pattern, or exact name.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        selector: Index (#), pattern (with wildcards), or exact modality name

    Returns:
        Summary string for conversation history, or None
    """
    if not selector:
        _render_blocks(
            output,
            [
                alert_block(
                    h("Usage: /workspace remove <#|pattern|name>"), level="error"
                ),
                list_block(
                    [
                        h("/workspace remove 1 - Remove by index"),
                        h("/workspace remove * - Remove all modalities"),
                        h("/workspace remove *clustered* - Remove matching pattern"),
                        h("/workspace remove geo_gse12345 - Remove by exact name"),
                    ],
                    title=h("Examples"),
                ),
                hint_block(
                    h("Tip: Use '/modalities' to see loaded modalities with indexes")
                ),
            ],
        )
        return None

    # Check if modality management is available
    if not hasattr(client.data_manager, "list_modalities"):
        _render_blocks(
            output,
            [
                alert_block(
                    h("Modality management not available in this client"),
                    level="error",
                )
            ],
        )
        return None

    available_modalities = client.data_manager.list_modalities()

    if not available_modalities:
        _render_blocks(
            output,
            [alert_block(h("No modalities currently loaded"), level="warning")],
        )
        return None

    # Determine which modalities to remove
    modalities_to_remove = []
    sorted_modalities = sorted(available_modalities)

    if selector.isdigit():
        # Index-based removal
        idx = int(selector)
        if 1 <= idx <= len(sorted_modalities):
            modalities_to_remove = [sorted_modalities[idx - 1]]
        else:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Index {idx} out of range (1-{max})").format(
                            idx=idx, max=len(sorted_modalities)
                        ),
                        level="error",
                    ),
                    list_block(
                        sorted_modalities,
                        title=h("Available modalities ({count})").format(
                            count=len(available_modalities)
                        ),
                        ordered=True,
                    ),
                ],
            )
            return None
    elif "*" in selector or "?" in selector or "[" in selector:
        # Pattern-based removal (wildcards detected)
        for name in sorted_modalities:
            if fnmatch.fnmatch(name.lower(), selector.lower()):
                modalities_to_remove.append(name)

        if not modalities_to_remove:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("No modalities match pattern: {selector}").format(
                            selector=selector
                        ),
                        level="warning",
                    ),
                    list_block(
                        sorted_modalities,
                        title=h("Available modalities ({count})").format(
                            count=len(available_modalities)
                        ),
                        ordered=True,
                    ),
                ],
            )
            return None
    else:
        # Exact name match
        if selector in available_modalities:
            modalities_to_remove = [selector]
        else:
            _render_blocks(
                output,
                [
                    alert_block(
                        h("Modality '{selector}' not found").format(selector=selector),
                        level="error",
                    ),
                    list_block(
                        sorted_modalities,
                        title=h("Available modalities ({count})").format(
                            count=len(available_modalities)
                        ),
                        ordered=True,
                    ),
                ],
            )
            return None

    # Confirm removal for multiple modalities
    if len(modalities_to_remove) > 1:
        _render_blocks(
            output,
            [
                alert_block(
                    h("About to remove {count} modalities:").format(
                        count=len(modalities_to_remove)
                    ),
                    level="warning",
                ),
                list_block(modalities_to_remove),
            ],
        )

    try:
        # Import the service
        from lobster.services.data_management.modality_management_service import (
            ModalityManagementService,
        )

        # Create service instance
        service = ModalityManagementService(client.data_manager)

        # Remove each modality
        removed_count = 0
        failed_count = 0

        for modality_name in modalities_to_remove:
            success, stats, ir = service.remove_modality(modality_name)

            if success:
                # Log to provenance
                client.data_manager.log_tool_usage(
                    tool_name="remove_modality",
                    parameters={"modality_name": modality_name},
                    description=f"Removed modality {stats['removed_modality']}: {stats['shape']['n_obs']} obs x {stats['shape']['n_vars']} vars",
                    ir=ir,
                )

                # Display success message
                _render_blocks(
                    output,
                    [
                        section_block(
                            body=h("Removed: {name}").format(
                                name=stats["removed_modality"]
                            )
                        )
                    ],
                )
                if len(modalities_to_remove) == 1:
                    # Show detailed info only for single removal
                    _render_blocks(
                        output,
                        [
                            hint_block(
                                h("Shape: {n_obs} obs × {n_vars} vars").format(
                                    n_obs=stats["shape"]["n_obs"],
                                    n_vars=stats["shape"]["n_vars"],
                                )
                            )
                        ],
                    )
                removed_count += 1
            else:
                _render_blocks(
                    output,
                    [
                        alert_block(
                            h("Failed to remove: {name}").format(name=modality_name),
                            level="error",
                        )
                    ],
                )
                failed_count += 1

        # Summary for multiple removals
        if len(modalities_to_remove) > 1:
            remaining = client.data_manager.list_modalities()
            _render_blocks(
                output,
                [
                    hint_block(
                        h(
                            "Summary: {removed} removed, {failed} failed, {remaining} remaining"
                        ).format(
                            removed=removed_count,
                            failed=failed_count,
                            remaining=len(remaining),
                        )
                    )
                ],
            )

        if removed_count > 0:
            if removed_count == 1:
                return f"Removed modality: {modalities_to_remove[0]}"
            else:
                return f"Removed {removed_count} modalities"
        return None

    except Exception as e:
        _render_blocks(
            output,
            [
                alert_block(
                    h("Error removing modality: {error}").format(error=str(e)),
                    level="error",
                )
            ],
        )
        return None


def _get_directory_stats(dir_path: str) -> tuple:
    """
    Get file count and total size for a directory.

    Args:
        dir_path: Path to directory

    Returns:
        Tuple of (file_count, total_size_str, exists)
    """
    path = Path(dir_path)
    if not path.exists():
        return 0, "-", False

    try:
        files = list(path.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        # Format size
        if total_size < 1024:
            size_str = f"{total_size} B"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        elif total_size < 1024 * 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"

        return file_count, size_str, True
    except Exception:
        return 0, "-", True


def workspace_status(
    client: "AgentClient", output: OutputAdapter, *, compact: bool = False
) -> Optional[str]:
    """
    Show workspace status and information.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    # Check if using DataManagerV2
    workspace_status_dict = {}
    if hasattr(client.data_manager, "get_workspace_status"):
        workspace_status_dict = client.data_manager.get_workspace_status()
    available_dataset_count: Optional[int] = None
    if compact and hasattr(client.data_manager, "get_available_datasets"):
        try:
            available_dataset_count = len(
                client.data_manager.get_available_datasets(force_refresh=False)
            )
        except Exception:
            available_dataset_count = None

    modality_detail_rows: list[tuple[str, list[list[str]]]] = []
    if hasattr(client.data_manager, "list_modalities"):
        modalities = client.data_manager.list_modalities()

        if modalities and not compact:
            for modality_name in modalities:
                try:
                    adata = client.data_manager.get_modality(modality_name)
                    rows = [
                        [h("Shape"), f"{adata.n_obs:,} obs × {adata.n_vars:,} vars"]
                    ]

                    # Show obs columns
                    obs_cols = list(adata.obs.columns)
                    if obs_cols:
                        cols_preview = ", ".join(obs_cols[:5])
                        if len(obs_cols) > 5:
                            cols_preview += " " + h("... (+{count} more)").format(
                                count=len(obs_cols) - 5
                            )
                        rows.append([h("Obs Columns"), cols_preview])

                    # Show var columns
                    var_cols = list(adata.var.columns)
                    if var_cols:
                        var_preview = ", ".join(var_cols[:5])
                        if len(var_cols) > 5:
                            var_preview += " " + h("... (+{count} more)").format(
                                count=len(var_cols) - 5
                            )
                        rows.append([h("Var Columns"), var_preview])

                    # Show layers
                    from lobster.core.utils.layer_keys import layer_names

                    layers = layer_names(adata)
                    if layers:
                        rows.append([h("Layers"), ", ".join(layers)])

                    # Show obsm
                    if adata.obsm:
                        obsm_str = ", ".join(list(adata.obsm.keys()))
                        rows.append(["Obsm", obsm_str])

                    # Show varm
                    if hasattr(adata, "varm") and adata.varm:
                        varm_str = ", ".join(list(adata.varm.keys()))
                        rows.append(["Varm", varm_str])

                    # Show some uns info
                    if adata.uns:
                        uns_keys = list(adata.uns.keys())[:5]
                        uns_str = ", ".join(uns_keys)
                        if len(adata.uns) > 5:
                            uns_str += " " + h("... (+{count} more)").format(
                                count=len(adata.uns) - 5
                            )
                        rows.append([h("Uns Keys"), uns_str])

                    modality_detail_rows.append((modality_name, rows))
                except Exception as e:
                    output.print(
                        h("Error accessing {name}: {error}").format(
                            name=modality_name, error=e
                        ),
                        style="error",
                    )

    _render_blocks(
        output,
        _build_workspace_status_blocks(
            workspace_status_dict,
            modality_detail_rows,
            compact=compact,
            available_dataset_count=available_dataset_count,
        ),
    )
    return "Displayed workspace status and information"
