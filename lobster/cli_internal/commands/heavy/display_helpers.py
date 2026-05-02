"""
Data preview formatting and status display.

Display helpers for data matrices, DataFrames, arrays, and system status.
Extracted from cli.py for modularity.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

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


def build_status_blocks(*, compact: bool = False) -> List[OutputBlock]:
    """
    Build structured status blocks for console, JSON, and protocol adapters.

    Shared implementation for both `lobster status` and `/status`.
    """
    blocks: List[OutputBlock] = [
        section_block(title="Lobster Status"),
    ]
    # Check initialization status
    env_file = Path.cwd() / ".env"
    is_initialized = env_file.exists()

    init_rows = [
        ("Initialization", "Configured" if is_initialized else "Not configured")
    ]
    if is_initialized:
        try:
            from dotenv import dotenv_values

            env_vars = dotenv_values(env_file)
            provider = env_vars.get("LOBSTER_LLM_PROVIDER")
            if provider:
                init_rows.append(("Provider", provider))
            else:
                if env_vars.get("ANTHROPIC_API_KEY"):
                    init_rows.append(("Provider", "anthropic (auto-detected)"))
                elif env_vars.get("AWS_BEDROCK_ACCESS_KEY"):
                    init_rows.append(("Provider", "bedrock (auto-detected)"))
                elif env_vars.get("OLLAMA_BASE_URL"):
                    init_rows.append(("Provider", "ollama (auto-detected)"))
        except Exception:
            pass
        init_rows.append(("Config File", str(env_file)))
    else:
        blocks.append(
            alert_block(
                "Lobster is not initialized yet",
                level="warning",
                title="Initialization",
            )
        )
        blocks.append(
            hint_block(
                "Run `lobster init` to configure your LLM provider (Anthropic/Bedrock/Ollama)."
            )
        )
    blocks.append(kv_block(init_rows, title="Initialization"))

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
    restricted = [name for name in worker_agents if not is_agent_available(name, tier)]

    tier_rows = [
        ("Subscription Tier", entitlement.get("tier_display", "Free")),
        ("Source", entitlement.get("source", "default")),
    ]

    if entitlement.get("expires_at"):
        days = entitlement.get("days_until_expiry")
        if days is not None and days < 30:
            blocks.append(
                alert_block(
                    f"License expires in {days} days",
                    level="warning",
                )
            )
        else:
            tier_rows.append(("Expires", entitlement.get("expires_at")))

    if entitlement.get("warnings"):
        for warning in entitlement["warnings"]:
            blocks.append(alert_block(warning, level="error"))
    blocks.append(kv_block(tier_rows, title="Subscription"))

    package_rows = []
    package_counts = {"installed": 0, "development": 0, "missing": 0}
    for pkg_name, version in packages.items():
        if version == "missing":
            status_str = "Missing"
            package_counts["missing"] += 1
        elif version == "dev":
            status_str = "Development"
            package_counts["development"] += 1
        else:
            status_str = "Installed"
            package_counts["installed"] += 1
        package_rows.append([pkg_name, version, status_str])
    if compact:
        blocks.append(
            kv_block(
                [
                    ("Installed Packages", str(package_counts["installed"])),
                    ("Development Packages", str(package_counts["development"])),
                    ("Missing Packages", str(package_counts["missing"])),
                    ("Available Agents", str(len(available))),
                    ("Premium Agents", str(len(restricted))),
                ],
                title="Runtime Summary",
            )
        )
    else:
        blocks.append(
            table_block(
                [
                    {"name": "Package"},
                    {"name": "Version"},
                    {"name": "Status"},
                ],
                package_rows,
                title="Installed Packages",
            )
        )

    capabilities = []
    try:
        from lobster.vector.service import VectorSearchService  # noqa: F401

        capabilities.append(
            ("available", "Semantic Search", "Vector backend available")
        )
    except ImportError:
        from lobster.core.component_registry import get_install_command

        vector_cmd = get_install_command("vector-search", is_extra=True)
        capabilities.append(
            (
                "optional",
                "Semantic Search",
                f"{vector_cmd} (+ lobster-metadata backend, dev-only)",
            )
        )
    try:
        import docling  # noqa: F401

        capabilities.append(("available", "Document Intelligence", "docling"))
    except ImportError:
        from lobster.core.component_registry import get_install_command

        docling_cmd = get_install_command("docling", is_extra=True)
        capabilities.append(
            (
                "optional",
                "Document Intelligence",
                docling_cmd,
            )
        )
    if compact:
        capability_items = [
            f"{label}: {status} ({details})" for status, label, details in capabilities
        ]
        blocks.append(list_block(capability_items, title="Optional Capabilities"))
        blocks.append(
            hint_block("Use `lobster status` for the full package and agent roster.")
        )
        return blocks

    blocks.append(
        table_block(
            [
                {"name": "Status"},
                {"name": "Capability"},
                {"name": "Details"},
            ],
            capabilities,
            title="Optional Capabilities",
        )
    )

    if available:
        available_names = sorted(available)
        available_preview = available_names[:6] if compact else available_names
        blocks.append(
            list_block(
                available_preview,
                title=f"Available Agents ({len(available)})",
            )
        )
        if compact and len(available_names) > len(available_preview):
            blocks.append(
                hint_block(
                    f"... and {len(available_names) - len(available_preview)} more available agents"
                )
            )

    if restricted:
        restricted_names = sorted(restricted)
        restricted_preview = restricted_names[:4] if compact else restricted_names
        blocks.append(
            list_block(
                restricted_preview,
                title=f"Premium Agents ({len(restricted)})",
            )
        )
        if compact and len(restricted_names) > len(restricted_preview):
            blocks.append(
                hint_block(
                    f"... and {len(restricted_names) - len(restricted_preview)} more premium agents"
                )
            )
        blocks.append(
            hint_block(
                f"Upgrade to Premium to unlock {len(restricted)} additional agents. Visit https://omics-os.com/pricing or run 'lobster activate <code>'."
            )
        )

    features = entitlement.get("features", [])
    if features:
        feature_items = [str(feature) for feature in features]
        feature_preview = feature_items[:6] if compact else feature_items
        blocks.append(list_block(feature_preview, title="Enabled Features"))
        if compact and len(feature_items) > len(feature_preview):
            blocks.append(
                hint_block(
                    f"... and {len(feature_items) - len(feature_preview)} more enabled features"
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
