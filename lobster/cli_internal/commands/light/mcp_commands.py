"""``lobster mcp``: Lobster as an MCP server (machine interface for MCP hosts)."""

import sys
from pathlib import Path
from typing import Optional

import typer

from lobster.i18n import h


def mcp_command(
    stdio: bool = typer.Option(
        False, "--stdio", help=h("Speak MCP (JSON-RPC) on stdin and stdout.")
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        help=h(
            "Folder the Lobster project is found from (default: the current folder)."
        ),
    ),
):
    if not stdio:
        print(
            h(
                "`lobster mcp` is a machine interface for MCP hosts; run it with --stdio."
            ),
            file=sys.stderr,
        )
        raise typer.Exit(2)
    if workspace is not None and not workspace.is_dir():
        print(h("--workspace must be an existing folder."), file=sys.stderr)
        raise typer.Exit(2)
    from lobster.cli_internal.mcp_stdio import enter_start_folder, run_mcp_stdio

    start = enter_start_folder(workspace)
    raise typer.Exit(run_mcp_stdio(start=start, follow_project=True))


mcp_command.__doc__ = h(
    "Run Lobster as an MCP server for desktop and agent hosts (JSON-RPC on stdin and stdout)."
)
