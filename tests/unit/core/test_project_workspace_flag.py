"""``-w <project folder>`` names the project's own workspace, for every command."""

from pathlib import Path

import pytest

from lobster.core import projects
from lobster.core.workspace import resolve_workspace


@pytest.fixture
def project(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    from lobster.config import global_config

    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "cfg" / "lobster")
    root = tmp_path / "study"
    projects.init_project(root)
    return root


@pytest.mark.parametrize("stray_data", [False, True])
def test_resolve_workspace_of_a_project_folder_is_its_workspace(project, stray_data):
    if stray_data:  # left by an earlier `query -w <project>`
        (project / "data").mkdir()

    assert resolve_workspace(project, create=False) == project / ".lobster_workspace"


def test_run_workspace_of_a_project_with_a_stray_data_folder_is_its_workspace(
    project,
):
    (project / "data").mkdir()

    assert projects.run_workspace(project) == project / ".lobster_workspace"


def test_an_explicit_workspace_folder_stays_itself(project):
    workspace = project / ".lobster_workspace"

    assert resolve_workspace(workspace, create=False) == workspace
    assert projects.run_workspace(workspace) == workspace


def test_a_plain_folder_stays_itself(tmp_path):
    assert resolve_workspace(tmp_path / "plain", create=False) == tmp_path / "plain"
    assert projects.run_workspace(tmp_path / "plain") == tmp_path / "plain"


def test_a_run_folder_keeps_its_rules(project):
    run = project / "runs" / "r1"
    (run / ".lobster_workspace").mkdir(parents=True)

    assert resolve_workspace(run, create=False) == run
    assert projects.run_workspace(run) == run / ".lobster_workspace"


def test_an_older_workspace_with_its_own_data_stays_itself(tmp_path):
    old = tmp_path / "old"
    (old / "data").mkdir(parents=True)
    (old / ".lobster_workspace").mkdir()

    assert resolve_workspace(old, create=False) == old
    assert projects.run_workspace(old) == old


def test_a_linked_project_workspace_is_not_followed(tmp_path, project):
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / ".lobster_workspace").symlink_to(project / ".lobster_workspace")

    assert resolve_workspace(linked, create=False) == linked


def test_the_workspace_variable_with_a_project_folder_is_its_workspace(
    project, monkeypatch
):
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(project))

    assert resolve_workspace(create=False) == project / ".lobster_workspace"


def test_the_workspace_variable_with_a_plain_folder_stays_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(tmp_path / "plain"))

    assert resolve_workspace(create=False) == tmp_path / "plain"
