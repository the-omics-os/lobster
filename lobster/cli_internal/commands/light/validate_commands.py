"""Validate-plugin command for checking plugin package compliance."""

from pathlib import Path

import typer

validate_app = typer.Typer(
    name="validate-plugin",
    help="Validate Lobster AI plugin packages for contract compliance",
    invoke_without_command=True,
)


@validate_app.callback(invoke_without_command=True)
def validate_plugin_cmd(
    plugin_dir: Path = typer.Argument(
        ..., help="Path to plugin package directory", exists=True
    ),
):
    """Validate a plugin package for Lobster AI contract compliance.

    Runs 7 structural checks:
    1. PEP 420 compliance (no __init__.py at namespace boundaries)
    2. Entry points (lobster.agents group in pyproject.toml)
    3. AGENT_CONFIG position (before heavy imports)
    4. Factory signature (standard parameters)
    5. AQUADIF metadata (tools have categories and provenance)
    6. Provenance calls (log_tool_usage with ir=)
    7. Import boundaries (no cross-agent imports)

    Example:
        lobster validate-plugin ./lobster-epigenomics/
    """
    from lobster.scaffold.validators import validate_plugin

    results = validate_plugin(plugin_dir)

    if not results:
        typer.echo("No checks could be performed. Is this a valid plugin directory?")
        raise typer.Exit(1)

    passed = 0
    failed = 0

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        icon = "✓" if result.passed else "✗"
        typer.echo(f"  {icon} [{result.check}] {result.message}")
        if result.passed:
            passed += 1
        else:
            failed += 1

    typer.echo("")
    if failed == 0:
        typer.echo(f"All {passed} checks passed.")
    else:
        typer.echo(f"{failed} check(s) failed, {passed} passed.")
        raise typer.Exit(1)
