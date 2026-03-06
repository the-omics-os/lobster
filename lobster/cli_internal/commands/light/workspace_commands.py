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
        status = "Loaded" if name in loaded else "Available"
        size = f"{info['size_mb']:.1f} MB"
        shape = (
            f"{info['shape'][0]:,} × {info['shape'][1]:,}" if info["shape"] else "N/A"
        )
        modified = info["modified"].split("T")[0]
        rows.append(
            [str(idx), status, truncate_middle(name, max_length=60), size, shape, modified]
        )

    return [
        table_block(
            title="Available Datasets",
            columns=[
                {"name": "#", "style": "dim", "width": 4},
                {"name": "Status", "style": "green", "width": 10},
                {"name": "Name", "style": "bold", "no_wrap": False},
                {"name": "Size", "style": "cyan", "width": 10},
                {"name": "Shape", "style": "white", "width": 15},
                {"name": "Modified", "style": "dim", "width": 12},
            ],
            rows=rows,
        ),
        hint_block("Use '/workspace info <#>' to see full details."),
    ]


def _build_workspace_info_blocks(
    matched_datasets: list[tuple[str, dict]],
    loaded: set[str],
) -> list[OutputBlock]:
    blocks: list[OutputBlock] = []
    for name, info in matched_datasets:
        rows = [
            ["Name", name],
            ["Status", "Loaded" if name in loaded else "Not Loaded"],
            ["Path", info["path"]],
            ["Size", f"{info['size_mb']:.2f} MB"],
            [
                "Shape",
                (
                    f"{info['shape'][0]:,} observations × {info['shape'][1]:,} variables"
                    if info["shape"]
                    else "N/A"
                ),
            ],
            ["Type", info["type"]],
            ["Modified", info["modified"]],
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
                rows.append(["Processing Stages", " -> ".join(possible_stages)])

        blocks.append(
            table_block(
                title=f"Dataset: {name}",
                columns=[
                    {"name": "Property", "style": "bold cyan"},
                    {"name": "Value", "style": "white"},
                ],
                rows=rows,
            )
        )
    return blocks


def _build_workspace_status_blocks(
    workspace_status_dict: dict,
    modality_detail_rows: list[tuple[str, list[list[str]]]],
) -> list[OutputBlock]:
    workspace_path = workspace_status_dict.get("workspace_path", "N/A")
    modalities_count = workspace_status_dict.get("modalities_loaded", 0)
    provenance = (
        "Enabled" if workspace_status_dict.get("provenance_enabled") else "Disabled"
    )
    mudata = (
        "Available" if workspace_status_dict.get("mudata_available") else "Not installed"
    )

    blocks: list[OutputBlock] = [
        section_block(title="Workspace Status"),
        table_block(
            title="Summary",
            columns=[{"name": "Field"}, {"name": "Value"}],
            rows=[
                ["Workspace", workspace_path],
                ["Modalities Loaded", str(modalities_count)],
                ["Provenance", provenance],
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
                display_path = f"{display_path} (not created)"
            dir_rows.append(
                [
                    dir_icons.get(dir_type, dir_type.replace("_", " ").title()),
                    str(file_count) if exists else "-",
                    size_str,
                    display_path,
                ]
            )
        blocks.append(
            table_block(
                title="Directories",
                columns=[
                    {"name": "Directory", "style": "bold white"},
                    {"name": "Files", "style": "cyan", "justify": "right"},
                    {"name": "Size", "style": "green", "justify": "right"},
                    {"name": "Path", "style": "grey70"},
                ],
                rows=dir_rows,
            )
        )

    modality_names = workspace_status_dict.get("modality_names", [])
    if modality_names:
        blocks.append(list_block(modality_names, title="Loaded Modalities"))
    else:
        blocks.append(hint_block("No modalities currently loaded"))

    backends = workspace_status_dict.get("registered_backends", [])
    adapters = workspace_status_dict.get("registered_adapters", [])
    blocks.append(
        table_block(
            title="System Capabilities",
            columns=[{"name": "Field"}, {"name": "Value"}],
            rows=[
                ["Backends", ", ".join(backends) if backends else "None"],
                ["Adapters", ", ".join(adapters[:5]) if adapters else "None"],
            ],
        )
    )
    if len(adapters) > 5:
        blocks.append(hint_block(f"... and {len(adapters) - 5} more adapters"))

    if modality_detail_rows:
        blocks.append(section_block(title="Modality Details"))
        for modality_name, rows in modality_detail_rows:
            blocks.append(
                table_block(
                    title=modality_name,
                    columns=[
                        {"name": "Property", "style": "bold grey93"},
                        {"name": "Value", "style": "white"},
                    ],
                    rows=rows,
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
                body=f"Loaded {len(restored)} datasets ({total_size_mb:.1f} MB)"
            ),
            list_block(restored, title="Loaded Datasets"),
        ]
        if skipped:
            blocks.extend(
                [
                    hint_block(
                        f"Skipped {len(skipped)} dataset(s) that were already loaded or unavailable."
                    ),
                    list_block(skipped, title="Skipped Datasets"),
                ]
            )
        return blocks

    blocks = [alert_block("No datasets loaded", level="warning")]
    if skipped:
        blocks.extend(
            [
                hint_block(
                    f"Skipped {len(skipped)} dataset(s) that were already loaded or unavailable."
                ),
                list_block(skipped, title="Skipped Datasets"),
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
            alert_block("No datasets found in workspace", level="warning"),
            section_block(body=f"Workspace: {workspace_path}"),
            section_block(body=f"Data directory: {data_dir}"),
        ]

        if not data_dir.exists():
            blocks.append(alert_block("Data directory does not exist", level="error"))
            blocks.append(section_block(body=f"Create it with: mkdir -p {data_dir}"))
        else:
            # Check what files are actually in the data directory
            files = list(data_dir.glob("*"))
            if files:
                blocks.append(
                    section_block(
                        body=(
                            f"Found {len(files)} files in data directory, but none are "
                            "supported datasets (.h5ad)."
                        )
                    )
                )
                blocks.append(
                    list_block([f.name for f in files[:5]], title="Files found")
                )
                if len(files) > 5:
                    blocks.append(
                        hint_block(f"... and {len(files) - 5} more")
                    )
            else:
                blocks.append(
                    section_block(
                        body=f"Add .h5ad files to {data_dir} to see them here."
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
                alert_block("Usage: /workspace info <#|pattern>", level="error"),
                hint_block(
                    "Examples: /workspace info 1, /workspace info gse12345, /workspace info *clustered*"
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
            [alert_block("No datasets found in workspace", level="warning")],
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
                        f"Index {idx} out of range (1-{len(sorted_names)})",
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
                        f"No datasets match pattern: {selector}",
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
                alert_block("Usage: /workspace load <#|pattern|file>", level="error"),
                hint_block(
                    "Examples: /workspace load 1, /workspace load recent, /workspace load data.h5ad"
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
            [alert_block(f"Security error: {resolved.error}", level="error")],
        )
        return None

    file_path = resolved.path

    if file_path.exists() and file_path.is_file():
        # Load file directly into workspace
        _render_blocks(
            output, [section_block(body=f"Loading file into workspace: {file_path.name}")]
        )

        try:
            result = client.load_data_file(str(file_path))

            if result.get("success"):
                _render_blocks(
                    output,
                    [
                        section_block(
                            body=(
                                f"Loaded '{result['modality_name']}' "
                                f"({result['data_shape'][0]:,} × {result['data_shape'][1]:,})"
                            )
                        )
                    ],
                )
                return f"Loaded file '{file_path.name}' as modality '{result['modality_name']}'"
            blocks = [
                alert_block(result.get("error", "Unknown error"), level="error")
            ]
            if result.get("suggestion"):
                blocks.append(section_block(body=result["suggestion"]))
            _render_blocks(output, blocks)
            return None

        except Exception as e:
            _render_blocks(
                output,
                [alert_block(f"Failed to load file: {str(e)}", level="error")],
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
                alert_block("No datasets found in workspace", level="warning"),
                hint_block(f"Tip: If '{selector}' is a file, ensure the path is correct."),
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
                [section_block(body=f"Loading dataset: {dataset_name}...")],
            )

            # Load single dataset directly
            success = client.data_manager.load_dataset(dataset_name)

            if success:
                _render_blocks(
                    output,
                    [
                        section_block(
                            body=(
                                f"Loaded dataset: {dataset_name} "
                                f"({available[dataset_name]['size_mb']:.1f} MB)"
                            )
                        )
                    ],
                )
                return "Loaded dataset from workspace"
            _render_blocks(
                output,
                [alert_block(f"Failed to load dataset: {dataset_name}", level="error")],
            )
            return None
        _render_blocks(
            output,
            [alert_block(f"Index {idx} out of range (1-{len(sorted_names)})", level="error")],
        )
        return None
    # Pattern-based loading (potentially multiple datasets)
    _render_blocks(
        output,
        [section_block(body=f"Loading workspace datasets (pattern: {selector})...")],
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
                alert_block("Usage: /workspace remove <#|pattern|name>", level="error"),
                list_block(
                    [
                        "/workspace remove 1 - Remove by index",
                        "/workspace remove * - Remove all modalities",
                        "/workspace remove *clustered* - Remove matching pattern",
                        "/workspace remove geo_gse12345 - Remove by exact name",
                    ],
                    title="Examples",
                ),
                hint_block("Tip: Use '/modalities' to see loaded modalities with indexes"),
            ],
        )
        return None

    # Check if modality management is available
    if not hasattr(client.data_manager, "list_modalities"):
        _render_blocks(
            output,
            [
                alert_block(
                    "Modality management not available in this client",
                    level="error",
                )
            ],
        )
        return None

    available_modalities = client.data_manager.list_modalities()

    if not available_modalities:
        _render_blocks(
            output,
            [alert_block("No modalities currently loaded", level="warning")],
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
                        f"Index {idx} out of range (1-{len(sorted_modalities)})",
                        level="error",
                    ),
                    list_block(
                        sorted_modalities,
                        title=f"Available modalities ({len(available_modalities)})",
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
                        f"No modalities match pattern: {selector}",
                        level="warning",
                    ),
                    list_block(
                        sorted_modalities,
                        title=f"Available modalities ({len(available_modalities)})",
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
                    alert_block(f"Modality '{selector}' not found", level="error"),
                    list_block(
                        sorted_modalities,
                        title=f"Available modalities ({len(available_modalities)})",
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
                    f"About to remove {len(modalities_to_remove)} modalities:",
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
                        section_block(body=f"Removed: {stats['removed_modality']}")
                    ],
                )
                if len(modalities_to_remove) == 1:
                    # Show detailed info only for single removal
                    _render_blocks(
                        output,
                        [
                            hint_block(
                                f"Shape: {stats['shape']['n_obs']} obs × {stats['shape']['n_vars']} vars"
                            )
                        ],
                    )
                removed_count += 1
            else:
                _render_blocks(
                    output,
                    [alert_block(f"Failed to remove: {modality_name}", level="error")],
                )
                failed_count += 1

        # Summary for multiple removals
        if len(modalities_to_remove) > 1:
            remaining = client.data_manager.list_modalities()
            _render_blocks(
                output,
                [
                    hint_block(
                        f"Summary: {removed_count} removed, {failed_count} failed, {len(remaining)} remaining"
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
            [alert_block(f"Error removing modality: {str(e)}", level="error")],
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


def workspace_status(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
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
    modality_detail_rows: list[tuple[str, list[list[str]]]] = []
    if hasattr(client.data_manager, "list_modalities"):
        modalities = client.data_manager.list_modalities()

        if modalities:
            for modality_name in modalities:
                try:
                    adata = client.data_manager.get_modality(modality_name)
                    rows = [["Shape", f"{adata.n_obs:,} obs × {adata.n_vars:,} vars"]]

                    # Show obs columns
                    obs_cols = list(adata.obs.columns)
                    if obs_cols:
                        cols_preview = ", ".join(obs_cols[:5])
                        if len(obs_cols) > 5:
                            cols_preview += f" ... (+{len(obs_cols) - 5} more)"
                        rows.append(["Obs Columns", cols_preview])

                    # Show var columns
                    var_cols = list(adata.var.columns)
                    if var_cols:
                        var_preview = ", ".join(var_cols[:5])
                        if len(var_cols) > 5:
                            var_preview += f" ... (+{len(var_cols) - 5} more)"
                        rows.append(["Var Columns", var_preview])

                    # Show layers
                    if adata.layers:
                        layers_str = ", ".join(list(adata.layers.keys()))
                        rows.append(["Layers", layers_str])

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
                            uns_str += f" ... (+{len(adata.uns) - 5} more)"
                        rows.append(["Uns Keys", uns_str])

                    modality_detail_rows.append((modality_name, rows))
                except Exception as e:
                    output.print(f"Error accessing {modality_name}: {e}", style="error")

    _render_blocks(
        output,
        _build_workspace_status_blocks(workspace_status_dict, modality_detail_rows),
    )
    return "Displayed workspace status and information"
