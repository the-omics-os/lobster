"""
Data preview formatting and status display.

Display helpers for data matrices, DataFrames, arrays, and system status.
Extracted from cli.py for modularity.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rich import box
from rich.table import Table

from lobster.cli_internal.commands.output_adapter import (
    ConsoleOutputAdapter,
    OutputBlock,
    alert_block,
    hint_block,
    kv_block,
    list_block,
    section_block,
    table_block,
)
from lobster.i18n import h, t
from lobster.ui.console_manager import get_console_manager

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

console_manager = get_console_manager()
console = console_manager.console


def _format_data_preview(matrix, max_rows: int = 5, max_cols: int = 5) -> Table:
    """Format a data matrix preview as a Rich table."""
    import numpy as np
    import scipy.sparse as sp

    # Convert sparse to dense for preview if needed
    if sp.issparse(matrix):
        preview_rows = min(max_rows, matrix.shape[0])
        preview_cols = min(max_cols, matrix.shape[1])
        preview_data = matrix[:preview_rows, :preview_cols].toarray()
    else:
        preview_rows = min(max_rows, matrix.shape[0])
        preview_cols = min(max_cols, matrix.shape[1])
        preview_data = matrix[:preview_rows, :preview_cols]

    # Create table
    table = Table(box=box.SIMPLE)

    # Add columns
    table.add_column("", style="bold grey50")  # Row index
    for i in range(preview_cols):
        table.add_column(f"[{i}]", style="cyan")

    # Add rows
    for i in range(preview_rows):
        row_values = ["[" + str(i) + "]"]
        for j in range(preview_cols):
            val = preview_data[i, j]
            if isinstance(val, (int, np.integer)):
                formatted = str(val)
            elif isinstance(val, (float, np.floating)):
                formatted = f"{val:.2f}"
            else:
                formatted = str(val)
            row_values.append(formatted)
        table.add_row(*row_values)

    # Add ellipsis row if there are more rows
    if matrix.shape[0] > max_rows or matrix.shape[1] > max_cols:
        ellipsis_row = ["..."] * (min(preview_cols, matrix.shape[1]) + 1)
        table.add_row(*ellipsis_row, style="dim")

    return table


def _format_dataframe_preview(df: "pd.DataFrame", max_rows: int = 5) -> Table:
    """Format a DataFrame preview as a Rich table."""
    import numpy as np
    import pandas as pd

    table = Table(box=box.SIMPLE)

    # Add index column
    table.add_column("Index", style="bold grey50")

    # Add data columns
    for col in df.columns[:10]:  # Limit to first 10 columns
        dtype_str = str(df[col].dtype)
        style = (
            "cyan"
            if dtype_str.startswith("int") or dtype_str.startswith("float")
            else "white"
        )
        table.add_column(str(col), style=style)

    # Add rows
    preview_rows = min(max_rows, len(df))
    for idx in range(preview_rows):
        row_data = [str(df.index[idx])]
        for col in df.columns[:10]:
            val = df.iloc[idx][col]
            if pd.isna(val):
                formatted = "NaN"
            elif isinstance(val, (int, np.integer)):
                formatted = str(val)
            elif isinstance(val, (float, np.floating)):
                formatted = f"{val:.2f}"
            else:
                formatted = str(val)[:20]  # Truncate long strings
            row_data.append(formatted)
        table.add_row(*row_data)

    # Add ellipsis if there are more rows
    if len(df) > max_rows:
        ellipsis_row = ["..."] * (min(10, len(df.columns)) + 1)
        table.add_row(*ellipsis_row, style="dim")

    # Add more columns indicator
    if len(df.columns) > 10:
        table.add_column(f"... +{len(df.columns) - 10} more", style="dim")

    return table


def _format_array_info(arrays_dict: Dict[str, "np.ndarray"]) -> Table:
    """Format array information (obsm/varm) as a table."""
    if not arrays_dict:
        return None

    table = Table(box=box.SIMPLE)
    table.add_column("Key", style="bold cyan")
    table.add_column("Shape", style="white")
    table.add_column("Dtype", style="grey70")

    for key, arr in arrays_dict.items():
        shape_str = " x ".join(str(d) for d in arr.shape)
        dtype_str = str(arr.dtype)
        table.add_row(key, shape_str, dtype_str)

    return table


def _get_matrix_info(matrix) -> Dict[str, Any]:
    """Get information about a matrix (sparse or dense)."""
    import scipy.sparse as sp

    info = {}
    info["shape"] = matrix.shape
    info["dtype"] = str(matrix.dtype)

    if sp.issparse(matrix):
        info["sparse"] = True
        info["format"] = matrix.format.upper()
        info["nnz"] = matrix.nnz
        info["density"] = (matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100
        info["memory_mb"] = (
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        ) / (1024**2)
    else:
        info["sparse"] = False
        info["format"] = "Dense"
        info["memory_mb"] = matrix.nbytes / (1024**2)
        info["density"] = 100.0

    return info


def _kv(rows, *, title: Optional[str] = None) -> OutputBlock:
    """kv block whose column headers follow the UI language."""
    return kv_block(rows, title=title, key_label=h("Field"), value_label=h("Value"))


def build_status_blocks(
    *,
    compact: bool = False,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
) -> List[OutputBlock]:
    """
    Build structured status blocks for console, JSON, and protocol adapters.

    Shared implementation for both `lobster status` and `/status`.
    """
    blocks: List[OutputBlock] = [
        section_block(title=h("Lobster Status")),
    ]
    # Resolve provider configuration exactly as the local runtime does. A
    # resolved configuration does not establish valid credentials/account access.
    from lobster.core.config_resolver import ConfigResolver, ConfigurationError
    from lobster.core.runtime.workspace import resolve_workspace

    resolved_workspace = resolve_workspace(workspace_path, create=False)
    try:
        provider, source = ConfigResolver(resolved_workspace).resolve_provider(
            runtime_override=provider_override
        )
    except ConfigurationError:
        init_rows = [(h("Initialization"), h("Not configured"))]
        blocks.append(
            alert_block(
                t("cli.status.not_initialized")
                + "\n"
                + t("cli.status.no_config", path=str(resolved_workspace)),
                level="warning",
                title=h("Initialization"),
            )
        )
        blocks.append(hint_block(t("cli.status.init_hint")))
    else:
        init_rows = [
            (h("Initialization"), h("Configured")),
            (h("Provider"), provider),
            (h("Configuration Source"), source),
            (h("Account Access"), h("Not checked (configuration only)")),
        ]
    blocks.append(_kv(init_rows, title=h("Initialization")))

    # Get entitlement status
    from lobster.core.license_manager import get_entitlement_status

    entitlement = get_entitlement_status()

    # Get installed packages
    from lobster.core.plugin_loader import get_installed_packages

    packages = get_installed_packages()

    # Get available agents
    from lobster.config.agent_registry import get_worker_agents
    from lobster.config.subscription_tiers import is_agent_available

    worker_agents = get_worker_agents()
    tier = entitlement.get("tier", "free")
    available = [name for name in worker_agents if is_agent_available(name, tier)]

    tier_rows = [
        (h("Subscription Tier"), entitlement.get("tier_display", "Free")),
        (h("Source"), entitlement.get("source", "default")),
    ]

    if entitlement.get("expires_at"):
        days = entitlement.get("days_until_expiry")
        if days is not None and days < 30:
            blocks.append(
                alert_block(
                    t("cli.status.license_expires", days=days),
                    level="warning",
                )
            )
        else:
            tier_rows.append((h("Expires"), entitlement.get("expires_at")))

    if entitlement.get("warnings"):
        for warning in entitlement["warnings"]:
            blocks.append(alert_block(warning, level="error"))
    blocks.append(_kv(tier_rows, title=h("Subscription")))

    package_rows = []
    package_counts = {"installed": 0, "development": 0, "missing": 0}
    for pkg_name, version in packages.items():
        if version == "missing":
            status_str = h("Missing")
            package_counts["missing"] += 1
        elif version == "dev":
            status_str = h("Development")
            package_counts["development"] += 1
        else:
            status_str = h("Installed")
            package_counts["installed"] += 1
        package_rows.append([pkg_name, version, status_str])
    if compact:
        blocks.append(
            _kv(
                [
                    (h("Installed Packages"), str(package_counts["installed"])),
                    (h("Development Packages"), str(package_counts["development"])),
                    (h("Missing Packages"), str(package_counts["missing"])),
                    (h("Available Agents"), str(len(available))),
                ],
                title=h("Runtime Summary"),
            )
        )
    else:
        blocks.append(
            table_block(
                [
                    {"name": h("Package")},
                    {"name": h("Version")},
                    {"name": h("Status")},
                ],
                package_rows,
                title=h("Installed Packages"),
            )
        )

    capabilities = []
    try:
        from lobster.vector.service import VectorSearchService  # noqa: F401

        capabilities.append(
            (
                h("available"),
                h("Semantic Search"),
                h("Vector backend available"),
            )
        )
    except ImportError:
        from lobster.core.component_registry import get_install_command

        vector_cmd = get_install_command("vector-search", is_extra=True)
        capabilities.append(
            (
                h("optional"),
                h("Semantic Search"),
                f"{vector_cmd} {h('(+ lobster-metadata backend, dev-only)')}",
            )
        )
    try:
        import docling  # noqa: F401

        capabilities.append((h("available"), h("Document Intelligence"), "docling"))
    except ImportError:
        from lobster.core.component_registry import get_install_command

        docling_cmd = get_install_command("docling", is_extra=True)
        capabilities.append(
            (
                h("optional"),
                h("Document Intelligence"),
                docling_cmd,
            )
        )
    if compact:
        capability_items = [
            f"{label}: {status} ({details})" for status, label, details in capabilities
        ]
        blocks.append(list_block(capability_items, title=h("Optional Capabilities")))
        blocks.append(
            hint_block(h("Use `lobster status` for the full package and agent roster."))
        )
        return blocks

    blocks.append(
        table_block(
            [
                {"name": h("Status")},
                {"name": h("Capability")},
                {"name": h("Details")},
            ],
            capabilities,
            title=h("Optional Capabilities"),
        )
    )

    if available:
        available_names = sorted(available)
        available_preview = available_names[:6] if compact else available_names
        blocks.append(
            list_block(
                available_preview,
                title=t("cli.status.available_agents", n=len(available)),
            )
        )
        if compact and len(available_names) > len(available_preview):
            blocks.append(
                hint_block(
                    t(
                        "cli.status.more_agents",
                        n=len(available_names) - len(available_preview),
                    )
                )
            )

    features = entitlement.get("features", [])
    if features:
        feature_items = [str(feature) for feature in features]
        feature_preview = feature_items[:6] if compact else feature_items
        blocks.append(list_block(feature_preview, title=h("Enabled Features")))
        if compact and len(feature_items) > len(feature_preview):
            blocks.append(
                hint_block(
                    t(
                        "cli.status.more_features",
                        n=len(feature_items) - len(feature_preview),
                    )
                )
            )

    return blocks


def _display_status_info(output=None):
    """
    Display subscription tier, installed packages, and available agents.

    Shared implementation for both 'lobster status' CLI command and '/status' chat command.
    This function is client-independent and shows installation/license information.
    """
    if output is None:
        output = ConsoleOutputAdapter(console)
    output.render_blocks(build_status_blocks())
