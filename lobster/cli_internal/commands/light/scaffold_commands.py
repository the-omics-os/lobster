"""Scaffold commands for generating plugin packages."""

from pathlib import Path
from typing import Optional

import typer

scaffold_app = typer.Typer(
    name="scaffold",
    help="Generate Lobster AI plugin packages",
    no_args_is_help=True,
)


@scaffold_app.command(name="agent")
def scaffold_agent_cmd(
    name: str = typer.Option(
        ..., "--name", help="Agent name in snake_case (e.g., epigenomics_expert)"
    ),
    display_name: str = typer.Option(
        ..., "--display-name", help="Human-readable name (e.g., 'Epigenomics Expert')"
    ),
    description: str = typer.Option(
        ..., "--description", help="Agent capabilities description"
    ),
    tier: str = typer.Option(
        "free", "--tier", help="Subscription tier: free, premium, enterprise"
    ),
    children: Optional[str] = typer.Option(
        None, "--children", help="Comma-separated child agent names"
    ),
    output_dir: Path = typer.Option(
        Path("."), "--output-dir", "-o", help="Output directory"
    ),
    author_name: str = typer.Option(
        "Lobster AI Community", "--author-name", help="Package author name"
    ),
    author_email: str = typer.Option(
        "community@lobsterbio.com", "--author-email", help="Package author email"
    ),
):
    """Generate a complete agent plugin package.

    Examples:
        lobster scaffold agent --name epigenomics_expert --display-name "Epigenomics Expert" --description "Epigenomics analysis" --tier free

        lobster scaffold agent --name epigenomics_expert --display-name "Epigenomics Expert" --description "Epigenomics analysis" --children methylation_expert,chromatin_expert
    """
    from lobster.scaffold import scaffold_agent

    children_list = [c.strip() for c in children.split(",")] if children else []

    pkg_dir = scaffold_agent(
        name=name,
        display_name=display_name,
        description=description,
        tier=tier,
        children=children_list,
        output_dir=output_dir,
        author_name=author_name,
        author_email=author_email,
    )

    typer.echo(f"Plugin package created at: {pkg_dir}")
    typer.echo(f"\nNext steps:")
    typer.echo(f"  cd {pkg_dir}")
    typer.echo(f"  uv pip install -e '.[dev]'")
    typer.echo(f"  python -m pytest tests/ -v -m contract")
    typer.echo(f"  lobster validate-plugin .")
