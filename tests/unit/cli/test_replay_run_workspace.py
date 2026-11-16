"""``replay -w <folder>`` names its workspace by the one rule the projects module has."""

import pytest
import typer
from typer.testing import CliRunner

from lobster.cli_internal.commands.light import replay_commands
from lobster.cli_internal.commands.light.replay_commands import (
    JsonUsageCommand,
    _anchored_file,
    replay_command,
)
from lobster.core import projects

pytestmark = pytest.mark.unit

app = typer.Typer()
app.command(cls=JsonUsageCommand)(replay_command)


def _file(folder, name="raw.h5ad"):
    (folder / "data").mkdir(parents=True)
    (folder / "data" / name).write_bytes(b"x")
    return folder / "data" / name


@pytest.mark.parametrize("case", ["run", "inside", "own_data"])
def test_the_three_folders_a_workspace_is_named_by(tmp_path, case):
    if case == "run":  # runs/<n> -> the workspace nested in it
        folder = tmp_path / "runs" / "r1"
        want = _file(folder / ".lobster_workspace")
    elif case == "inside":  # the workspace itself
        folder = tmp_path / "runs" / "r1" / ".lobster_workspace"
        want = _file(folder)
    else:  # a folder with its own data wins over a nested workspace
        folder = tmp_path / "plain"
        want = _file(folder)
        (folder / ".lobster_workspace").mkdir()
        _file(folder / ".lobster_workspace", "other.h5ad")
    assert _anchored_file(folder, "raw.h5ad") == want
    assert replay_commands.run_workspace is projects.run_workspace


def test_the_command_asks_the_projects_rule_for_the_workspace(tmp_path, monkeypatch):
    seen = []
    real = projects.run_workspace
    monkeypatch.setattr(
        replay_commands,
        "run_workspace",
        lambda folder: seen.append(folder) or real(folder),
    )
    folder = tmp_path / "runs" / "r1"
    _file(folder / ".lobster_workspace")
    _anchored_file(folder, "raw.h5ad")
    assert seen == [folder]


def test_a_replay_in_a_project_run_is_recorded_in_the_project_catalog(
    tmp_path, monkeypatch
):
    from lobster.core.catalog.layout import catalog_layout

    project, _ = projects.init_project(tmp_path / "study", "study")
    run = tmp_path / "study" / "runs" / "r1"
    (run / ".lobster_workspace" / "data").mkdir(parents=True)
    got = []
    monkeypatch.setattr(
        replay_commands, "_catalog_record", lambda ws: got.append(ws) or None
    )
    notebook = tmp_path / "nb.ipynb"
    notebook.write_text("{}")
    CliRunner().invoke(app, ["--verify", str(notebook), "-w", str(run), "--json"])
    (ws,) = got
    assert ws == (run / ".lobster_workspace").resolve()
    assert catalog_layout(ws).project == project.root.resolve()
