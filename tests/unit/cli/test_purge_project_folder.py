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


def link(at: Path, to: Path):
    try:
        at.symlink_to(to, target_is_directory=to.is_dir())
    except OSError:
        pytest.skip("this system cannot make links")


def said(result, text):
    return text in " ".join(result.output.split())


def test_a_folder_that_contains_a_project_is_not_removed(tmp_path, home):
    # A pre-1.3 `-w ~/work` left .session.json there; a project was made inside later.
    work = home / "work"
    work.mkdir()
    (work / ".session.json").write_text("{}")
    (work / "thesis.docx").write_text("x")
    inner = work / "projA"
    projects.init_project(inner)

    result = purge("-w", str(work), "--force")

    assert result.exit_code == 0, result.output
    assert (work / "thesis.docx").exists()
    assert (inner / ".lobster_workspace" / "project.json").exists()
    assert said(result, "contains a project or workspace")


def test_a_registered_project_deep_inside_keeps_the_folder(tmp_path, home):
    work = tmp_path / "work"
    work.mkdir()
    (work / ".session.json").write_text("{}")
    deep = work / "a" / "b" / "c" / "proj"
    projects.init_project(deep)

    result = purge("-w", str(work), "--force")

    assert (deep / ".lobster_workspace" / "project.json").exists(), result.output
    assert said(result, "contains a project or workspace")


def test_an_unregistered_project_inside_keeps_the_folder(tmp_path, home):
    work = tmp_path / "work"
    work.mkdir()
    (work / ".session.json").write_text("{}")
    inner = work / "a" / "proj"
    projects.init_project(inner)
    projects.forget_project(inner)

    result = purge("-w", str(work), "--force")

    assert (inner / ".lobster_workspace" / "project.json").exists(), result.output
    assert said(result, "contains a project or workspace")


@pytest.mark.parametrize("depth", [3, 5])
def test_a_moved_project_deep_inside_keeps_the_folder(tmp_path, home, depth):
    work = tmp_path / "work"
    work.mkdir()
    (work / ".session.json").write_text("{}")
    inner = work.joinpath(*[f"l{i}" for i in range(depth - 1)], "proj")
    projects.init_project(inner)
    projects.forget_project(inner)  # moved or cloned: the registry does not know it

    result = purge("-w", str(work), "--force")

    assert (inner / ".lobster_workspace" / "project.json").exists(), result.output
    assert said(result, "contains a project or workspace")


def test_a_folder_too_large_to_check_is_not_removed(tmp_path, home, monkeypatch):
    monkeypatch.setattr(purge_commands, "_SCAN_LIMIT", 3)
    work = tmp_path / "work"
    for i in range(5):
        (work / f"d{i}").mkdir(parents=True)
    (work / ".session.json").write_text("{}")

    result = purge("-w", str(work), "--force")

    assert work.is_dir() and len(list(work.iterdir())) == 6, result.output
    assert said(result, "holds too many folders to check for projects")


def test_a_large_workspace_by_name_is_still_purged(tmp_path, home, monkeypatch):
    monkeypatch.setattr(purge_commands, "_SCAN_LIMIT", 3)
    workspace = tmp_path / "old" / ".lobster_workspace"
    for i in range(5):
        (workspace / f"d{i}").mkdir(parents=True)

    result = purge("-w", str(workspace), "--force")

    assert result.exit_code == 0, result.output
    assert not workspace.exists()


def test_a_project_whose_project_file_is_a_link_is_not_removed(tmp_path, home):
    root = tmp_path / "study"
    projects.init_project(root)
    (root / "my_notes.txt").write_text("notes")
    (root / ".session.json").write_text("{}")
    pj = root / ".lobster_workspace" / "project.json"
    real = tmp_path / "pj.json"
    real.write_text(pj.read_text())
    pj.unlink()
    link(pj, real)

    result = purge("-w", str(root), "--force")

    assert (root / "my_notes.txt").exists(), result.output
    assert said(result, "holds a Lobster workspace")


def test_a_folder_holding_an_old_workspace_is_not_removed(tmp_path, home):
    old = tmp_path / "old"
    (old / ".lobster_workspace" / "data").mkdir(parents=True)
    (old / ".session.json").write_text("{}")
    (old / "thesis.docx").write_text("x")

    result = purge("-w", str(old), "--force")

    assert (old / "thesis.docx").exists(), result.output
    assert said(result, "holds a Lobster workspace")


def test_purge_never_follows_a_link_into_another_project(tmp_path, home):
    other = tmp_path / "other"
    projects.init_project(other)
    (other / ".lobster_workspace" / "provider_config.json").write_text("{}")
    study = tmp_path / "study"
    study.mkdir()
    link(study / ".lobster_workspace", other)

    result = purge("-w", str(study / ".lobster_workspace"), "--force")

    assert (other / ".lobster_workspace" / "project.json").exists(), result.output
    assert said(result, "is reached through a link")


def test_a_linked_workspace_gets_a_reason_not_an_error(tmp_path, home, monkeypatch):
    docs = home / "Documents"
    docs.mkdir()
    (docs / "a.txt").write_text("a")
    study = tmp_path / "study"
    study.mkdir()
    link(study / ".lobster_workspace", docs)
    monkeypatch.chdir(study)

    result = purge("--force")

    assert (docs / "a.txt").exists()
    assert said(result, "is reached through a link"), result.output
    assert "Errno" not in result.output


def test_an_error_naming_a_bracketed_path_is_printed_as_written(
    tmp_path, home, monkeypatch
):
    workspace = tmp_path / "[bold]ws"
    workspace.mkdir()
    (workspace / ".session.json").write_text("{}")
    failed = purge_commands.PurgeResult(
        success=False, errors=[f"Failed to remove {workspace}: busy"]
    )
    monkeypatch.setattr(purge_commands, "execute_purge", lambda *a, **k: failed)

    result = purge("-w", str(workspace), "--force")

    text = "".join(result.output.split())  # long paths wrap across lines
    assert f"Failedtoremove{workspace}:busy" in text, result.output
    assert text.count("[bold]ws") == 2  # the listing and the error


def test_purge_never_goes_through_a_linked_workspace_above(tmp_path, home):
    other = tmp_path / "other"
    projects.init_project(other)
    old = other / ".lobster_workspace" / "old"
    old.mkdir()
    (old / ".session.json").write_text("{}")
    study = tmp_path / "study"
    study.mkdir()
    link(study / ".lobster_workspace", other / ".lobster_workspace")

    result = purge("-w", str(study / ".lobster_workspace" / "old"), "--force")

    assert (old / ".session.json").exists(), result.output
    assert said(result, "is reached through a link")
