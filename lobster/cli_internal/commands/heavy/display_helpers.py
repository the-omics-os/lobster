"""
Data preview formatting and status display.

Display helpers for data matrices, DataFrames, arrays, and system status.
Extracted from cli.py for modularity.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from rich import box
from rich.panel import Panel
from rich.table import Table

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


def _display_status_info():
    """
    Display subscription tier, installed packages, and available agents.

    Shared implementation for both 'lobster status' CLI command and '/status' chat command.
    This function is client-independent and shows installation/license information.
    """
    from lobster.ui import LobsterTheme

    console.print()
    console.print(
        Panel.fit(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]Lobster Status[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            padding=(0, 2),
        )
    )
    console.print()

    # Check initialization status
    env_file = Path.cwd() / ".env"
    is_initialized = env_file.exists()

    if is_initialized:
        console.print("[bold]Initialization:[/bold] Configured")
        try:
            from dotenv import dotenv_values

            env_vars = dotenv_values(env_file)
            provider = env_vars.get("LOBSTER_LLM_PROVIDER")
            if provider:
                console.print(f"[dim]Provider: {provider}[/dim]")
            else:
                if env_vars.get("ANTHROPIC_API_KEY"):
                    console.print("[dim]Provider: anthropic (auto-detected)[/dim]")
                elif env_vars.get("AWS_BEDROCK_ACCESS_KEY"):
                    console.print("[dim]Provider: bedrock (auto-detected)[/dim]")
                elif env_vars.get("OLLAMA_BASE_URL"):
                    console.print("[dim]Provider: ollama (auto-detected)[/dim]")
        except Exception:
            pass
        console.print(f"[dim]Config file: {env_file}[/dim]")
    else:
        console.print("[bold]Initialization:[/bold] Not configured")
        console.print(
            Panel.fit(
                "[yellow]Lobster is not initialized yet[/yellow]\n\n"
                f"Run: [bold {LobsterTheme.PRIMARY_ORANGE}]lobster init[/bold {LobsterTheme.PRIMARY_ORANGE}]\n\n"
                "[dim]This will configure your LLM provider (Anthropic/Bedrock/Ollama)[/dim]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    console.print()

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

    # Subscription tier section
    tier_display = entitlement.get("tier_display", "Free")
    tier_emoji = {"free": "free", "premium": "premium", "enterprise": "enterprise"}.get(
        entitlement.get("tier", "free"), "free"
    )

    console.print(f"[bold]Subscription Tier:[/bold] {tier_display}")
    console.print(f"[dim]Source: {entitlement.get('source', 'default')}[/dim]")

    if entitlement.get("expires_at"):
        days = entitlement.get("days_until_expiry")
        if days is not None and days < 30:
            console.print(f"[yellow]License expires in {days} days[/yellow]")
        else:
            console.print(f"[dim]Expires: {entitlement.get('expires_at')}[/dim]")

    if entitlement.get("warnings"):
        for warning in entitlement["warnings"]:
            console.print(f"[red]{warning}[/red]")

    console.print()

    # Installed packages table
    console.print("[bold]Installed Packages:[/bold]")
    pkg_table = Table(box=box.ROUNDED, border_style="cyan", show_header=True)
    pkg_table.add_column("Package", style="white")
    pkg_table.add_column("Version", style="cyan")
    pkg_table.add_column("Status", style="green")

    for pkg_name, version in packages.items():
        if version == "missing":
            status_str = "[red]Missing[/red]"
        elif version == "dev":
            status_str = "[yellow]Development[/yellow]"
        else:
            status_str = "[green]Installed[/green]"
        pkg_table.add_row(pkg_name, version, status_str)

    console.print(pkg_table)
    console.print()

    # Optional capabilities
    console.print("[bold]Optional Capabilities:[/bold]")
    capabilities = []
    try:
        import chromadb  # noqa: F401

        capabilities.append(
            ("[green]v[/green]", "Semantic Search", "chromadb + sentence-transformers")
        )
    except ImportError:
        capabilities.append(
            (
                "[dim]o[/dim]",
                "Semantic Search",
                "pip install 'lobster-ai\\[vector-search]'",
            )
        )
    try:
        import docling  # noqa: F401

        capabilities.append(("[green]v[/green]", "Document Intelligence", "docling"))
    except ImportError:
        capabilities.append(
            (
                "[dim]o[/dim]",
                "Document Intelligence",
                "pip install 'lobster-ai\\[docling]'",
            )
        )

    cap_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    cap_table.add_column("Status", width=3)
    cap_table.add_column("Capability", style="white")
    cap_table.add_column("Details", style="dim")
    for status, name, detail in capabilities:
        cap_table.add_row(status, name, detail)
    console.print(cap_table)
    console.print()

    # Available agents
    if available:
        console.print(f"[bold]Available Agents ({len(available)}):[/bold]")
        agent_list = ", ".join(sorted(available))
        console.print(f"[green]{agent_list}[/green]")
        console.print()

    # Restricted agents (upgrade prompt)
    if restricted:
        console.print(f"[bold]Premium Agents ({len(restricted)}):[/bold]")
        restricted_list = ", ".join(sorted(restricted))
        console.print(f"[dim]{restricted_list}[/dim]")
        console.print()
        console.print(
            Panel.fit(
                f"[yellow]Upgrade to Premium to unlock {len(restricted)} additional agents[/yellow]\n"
                f"[dim]Visit https://omics-os.com/pricing or run 'lobster activate <code>'[/dim]",
                border_style="yellow",
                padding=(0, 2),
            )
        )

    # Features
    features = entitlement.get("features", [])
    if features:
        console.print()
        console.print("[bold]Enabled Features:[/bold]")
        console.print(f"[cyan]{', '.join(features)}[/cyan]")
