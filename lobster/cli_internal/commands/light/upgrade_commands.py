"""Implementation of the ``lobster upgrade`` command.

Light command: no heavy dependencies, safe to import from thin contexts.
"""

import typer
from rich.console import Console

from lobster.core.self_upgrade import run_upgrade


def upgrade_impl(dry_run: bool = False) -> None:
    """Upgrade lobster-ai to the latest published version.

    Args:
        dry_run: Print the upgrade command instead of running it.
    """
    console = Console()
    raise typer.Exit(code=run_upgrade(console=console, dry_run=dry_run))
