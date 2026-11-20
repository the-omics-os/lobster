"""Clearing exports in a folder that is not a Lobster project shows every file first."""

import json
from types import SimpleNamespace

import pytest
from rich.console import Console

from lobster.cli_internal.commands.light.metadata_commands import (
    metadata_clear_all,
    metadata_clear_exports,
)
from lobster.cli_internal.commands.output_adapter import ConsoleOutputAdapter

WARNING = (
    "This folder is not a Lobster project; "
    "files in its exports folder may be your own."
)
NAMES = [f"table_{i}.csv" for i in range(6)] + ["thesis_data.xlsx"]


def _workspace(folder, project):
    exports = folder / "exports"
    exports.mkdir(parents=True)
    for name in NAMES:
        (exports / name).write_text("a,b\n")
    if project:
        (folder / "project.json").write_text(json.dumps({"name": "study"}))
    return folder


def _ask(clear, workspace):
    """Run a clear and answer no; returns what was shown."""
    console = Console(record=True, width=300)
    output = ConsoleOutputAdapter(console)
    output.confirm = lambda question: False
    client = SimpleNamespace(
        data_manager=SimpleNamespace(
            workspace_path=str(workspace), metadata_store={}, current_metadata={}
        )
    )
    clear(client, output)
    return " ".join(console.export_text().split())


@pytest.mark.parametrize("clear", [metadata_clear_exports, metadata_clear_all])
def test_a_plain_folder_shows_every_file_and_warns(tmp_path, clear):
    workspace = _workspace(tmp_path / "docs", project=False)

    shown = _ask(clear, workspace)

    assert WARNING in shown
    assert all(name in shown for name in NAMES)
    assert sorted(p.name for p in (workspace / "exports").iterdir()) == sorted(NAMES)


def test_a_project_shows_the_preview_as_before(tmp_path):
    workspace = _workspace(tmp_path / "study" / ".lobster_workspace", project=True)

    shown = _ask(metadata_clear_exports, workspace)

    assert WARNING not in shown
    assert "... and 2 more files" in shown


def test_clearing_all_in_a_project_does_not_warn(tmp_path):
    workspace = _workspace(tmp_path / "study" / ".lobster_workspace", project=True)

    assert WARNING not in _ask(metadata_clear_all, workspace)
