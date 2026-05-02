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
        "free", "--tier", hidden=True, help="Deprecated — all agents are open source"
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
        lobster scaffold agent --name <domain>_expert --display-name "<Domain> Expert" --description "<capabilities>"

        lobster scaffold agent --name <domain>_expert --display-name "<Domain> Expert" --description "<capabilities>" --children <sub>_expert,<sub2>_expert
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

    # Show relative path to avoid confusing agents/users
    try:
        rel_path = pkg_dir.relative_to(Path.cwd())
    except ValueError:
        rel_path = pkg_dir
    typer.echo(f"Plugin package created at: ./{rel_path}")
    typer.echo("\nNext steps:")
    typer.echo(
        f"  1. Fill in domain-specific tools in ./{rel_path}/lobster/agents/*/shared_tools.py"
    )
    typer.echo("  2. Add services and platform configs")
    typer.echo(f"  3. Validate: lobster validate-plugin ./{rel_path}")
