"""``lobster mcp``: Lobster as an MCP server (machine interface for MCP hosts)."""

import sys

import typer

from lobster.i18n import h


def mcp_command(
    stdio: bool = typer.Option(
        False, "--stdio", help=h("Speak MCP (JSON-RPC) on stdin and stdout.")
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
    from lobster.cli_internal.mcp_stdio import run_mcp_stdio

    raise typer.Exit(run_mcp_stdio())


mcp_command.__doc__ = h(
    "Run Lobster as an MCP server for desktop and agent hosts (JSON-RPC on stdin and stdout)."
)
