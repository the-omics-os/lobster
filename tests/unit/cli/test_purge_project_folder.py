"""``lobster purge`` removes a workspace, never a project folder or the files around it."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lobster.cli_internal.commands.light import purge_commands
from lobster.cli_internal.commands.light.purge_commands import (
    PurgeScope,
    PurgeTarget,
    discover_purge_targets,
    execute_purge,
)
from lobster.core import projects


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    from lobster.config import global_config

    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "cfg" / "lobster")
    return home


@pytest.fixture
def project(tmp_path, home):
    """A project whose folder also holds the user's own data/ and exports/."""
    root = tmp_path / "study"
    projects.init_project(root)
    (root / ".lobster_workspace" / "provider_config.json").write_text("{}")
    (root / "data").mkdir()
    (root / "data" / "x.csv").write_text("a,b\n")
    (root / "exports").mkdir()
    (root / "my_notes.txt").write_text("notes")
    return root


def purge(*args, env=None):
    from lobster.cli import app

    return CliRunner().invoke(app, ["purge", "--scope", "workspace", *args], env=env)


def assert_user_files_kept(root: Path):
    assert (root / "my_notes.txt").read_text() == "notes"
    assert (root / "data" / "x.csv").exists()


def test_purge_with_a_project_folder_removes_only_its_workspace(project):
    result = purge("-w", str(project), "--force")

    assert result.exit_code == 0, result.output
    assert_user_files_kept(project)
    assert not (project / ".lobster_workspace").exists()


def test_purge_with_the_variable_set_to_a_project_folder_removes_only_its_workspace(
    project,
):
    result = purge("--force", env={"LOBSTER_WORKSPACE": str(project)})

    assert result.exit_code == 0, result.output
    assert_user_files_kept(project)
    assert not (project / ".lobster_workspace").exists()


def test_a_dry_run_on_a_project_folder_lists_only_its_workspace(project):
    targets = discover_purge_targets(PurgeScope.WORKSPACE, project)
    result = purge("-w", str(project), "--dry-run")

    assert [t.path for t in targets] == [project / ".lobster_workspace"]
    assert result.exit_code == 0, result.output
    text = "".join(result.output.split())  # long paths wrap across lines
    nested = str(project / ".lobster_workspace")
    assert nested in text
    assert text.count(str(project)) == text.count(nested)  # the root only inside it


def test_purge_never_removes_the_home_folder(home):
    (home / "data").mkdir()
    (home / "exports").mkdir()
    (home / "provider_config.json").write_text("{}")  # even with a marker

    result = purge("-w", str(home), "--force")

    assert result.exit_code == 0, result.output
    assert (home / "data").is_dir() and (home / "provider_config.json").exists()
    assert "is your home folder or contains it" in " ".join(result.output.split())


def test_the_filesystem_root_is_never_a_workspace(home):
    reason = purge_commands._workspace_refusal(Path("/"))

    assert reason and "is your home folder or contains it" in reason


def test_a_folder_with_only_standard_subfolders_is_not_removed(tmp_path, home):
    folder = tmp_path / "analysis"
    (folder / "data").mkdir(parents=True)
    (folder / "exports").mkdir()
    (folder / "data" / "x.csv").write_text("a,b\n")

    result = purge("-w", str(folder), "--force")

    assert result.exit_code == 0, result.output
    assert (folder / "data" / "x.csv").exists()
    assert "is not a Lobster workspace" in " ".join(result.output.split())


@pytest.mark.parametrize("name", [".lobster_workspace", "my-workspace"])
def test_a_real_workspace_is_purged_as_before(tmp_path, home, name):
    workspace = tmp_path / name
    (workspace / "data").mkdir(parents=True)
    (workspace / ".session.json").write_text("{}")

    result = purge("-w", str(workspace), "--force")

    assert result.exit_code == 0, result.output
    assert not workspace.exists()


def test_removal_refuses_a_project_folder_even_when_told_it_is_verified(project):
    target = PurgeTarget(path=project, description="", is_verified=True)

    result = execute_purge([target])

    assert not result.success and result.targets_removed == []
    assert "is a project folder" in " ".join(result.errors)
    assert_user_files_kept(project)
    assert (project / ".lobster_workspace" / "project.json").exists()
