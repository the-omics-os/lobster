"""``lobster commands``: list the engine's commands (``--json`` for hosts)."""

import json

import typer

from lobster.i18n import h

_REEXEC_MARKER = "LOBSTER_COMMANDS_REEXEC"


def commands_command(
    as_json: bool = typer.Option(
        False, "--json", help=h("Print the lobster.commands/1 manifest as JSON.")
    ),
):
    import os

    from lobster.i18n import current_lang

    # The marker holds the re-running parent's pid, so one inherited from an
    # unrelated ancestor does not skip the English re-run.
    reexeced = os.environ.get(_REEXEC_MARKER) == str(os.getppid())
    if as_json and current_lang() != "en" and not reexeced:
        # The manifest is a machine contract: help strings are translated when
        # the CLI loads, so build it in a process whose UI language is English.
        # shortcut: costs a second CLI start-up under a non-English UI and re-runs
        # at most once (marker); upgrade = emit the source strings without re-exec.
        import subprocess  # nosec B404 # re-runs this CLI with a fixed argv
        import sys

        env = dict(os.environ, LOBSTER_LANG="en", **{_REEXEC_MARKER: str(os.getpid())})
        result = subprocess.run(  # nosec B603
            [sys.executable, "-m", "lobster", "commands", "--json"],
            env=env,
            capture_output=True,
            text=True,
        )
        sys.stderr.write(result.stderr)
        typer.echo(result.stdout, nl=False)
        raise typer.Exit(result.returncode)

    from lobster.cli import app
    from lobster.cli_internal.command_registry import build_manifest

    manifest = build_manifest(app)
    if as_json:
        typer.echo(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    for command in manifest["commands"]:
        typer.echo(f"{command['name']:<32} {command.get('help', '')}")


commands_command.__doc__ = h(
    "List Lobster's commands; --json prints the machine-readable manifest."
)
