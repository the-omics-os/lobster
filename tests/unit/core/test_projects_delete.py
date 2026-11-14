"""Deleting a project removes only what Lobster created."""

import json
import os
import subprocess
import sys

import pytest

from lobster.core import projects
from lobster.core.projects import ProjectError


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    from lobster.config import global_config

    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "cfg" / "lobster")
    return h


@pytest.fixture
def proj(home):
    project, _ = projects.init_project(home / "p")
    (home / "p" / "README.md").write_text("mine")
    (home / "p" / "data").mkdir()
    (home / "p" / "data" / "in.csv").write_text("a,b")
    (home / "p" / ".env").write_text("OPENAI_API_KEY=x\n")
    run = home / "p" / "runs" / "r1"
    (run / ".lobster_workspace").mkdir(parents=True)
    (run / ".lobster_workspace" / "mcp_turns.json").write_text("{}")
    (run / "result.json").write_text("{}")
    projects.record_created_path(project, run)
    (project.workspace / "provider_config.json").write_text("{}")
    return project


def test_plan_lists_workspace_and_recorded_runs(proj):
    plan = projects.plan_delete(proj.root)
    assert plan.paths == [proj.root / "runs" / "r1", proj.workspace]
    assert plan.files >= 3
    assert plan.size_bytes > 0
    assert set(plan.kept) == {".env", "data", "runs"}
    assert not plan.skipped


def test_delete_removes_only_lobster_things(proj):
    errors = projects.delete_project(projects.plan_delete(proj.root))
    assert errors == []
    assert not proj.workspace.exists()
    assert not (proj.root / "runs" / "r1").exists()
    assert (proj.root / "README.md").read_text() == "mine"
    assert (proj.root / "data" / "in.csv").read_text() == "a,b"
    assert (proj.root / ".env").exists()
    assert proj.root.is_dir()
    assert projects.registered_projects() == []


def test_plan_does_not_remove_anything(proj):
    projects.plan_delete(proj.root)
    assert proj.workspace.is_dir()
    assert (proj.root / "runs" / "r1").is_dir()


def test_refuses_folder_lobster_did_not_create(home):
    (home / "repo").mkdir()
    (home / "repo" / "main.py").write_text("x")
    with pytest.raises(ProjectError) as exc:
        projects.plan_delete(home / "repo")
    assert exc.value.code == "not_a_project"


def test_refuses_legacy_workspace_without_identity(home):
    (home / "old" / ".lobster_workspace").mkdir(parents=True)
    with pytest.raises(ProjectError) as exc:
        projects.plan_delete(home / "old")
    assert exc.value.code == "not_a_project"
    assert (home / "old" / ".lobster_workspace").is_dir()


def test_refuses_project_folder_that_is_a_link(proj, home):
    os.symlink(proj.root, home / "link")
    with pytest.raises(ProjectError) as exc:
        projects.plan_delete(home / "link")
    assert exc.value.code == "unsafe_path"
    assert proj.workspace.is_dir()


def test_refuses_symlinked_workspace(home, tmp_path):
    real, _ = projects.init_project(tmp_path / "store")
    (home / "q").mkdir()
    os.symlink(real.workspace, home / "q" / ".lobster_workspace")
    with pytest.raises(ProjectError) as exc:
        projects.plan_delete(home / "q")
    assert exc.value.code == "unsafe_path"
    assert (real.workspace / "project.json").is_file()


def test_refuses_symlinked_project_file(home, tmp_path):
    ws = home / "q" / ".lobster_workspace"
    ws.mkdir(parents=True)
    real = tmp_path / "id.json"
    real.write_text(json.dumps({}))
    os.symlink(real, ws / "project.json")
    with pytest.raises(ProjectError) as exc:
        projects.plan_delete(home / "q")
    assert exc.value.code == "unsafe_path"


def test_symlink_inside_a_run_folder_is_not_followed(proj, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    os.symlink(outside, proj.root / "runs" / "r1" / "linked")
    os.symlink(outside / "keep.txt", proj.workspace / "file_link")
    assert projects.delete_project(projects.plan_delete(proj.root)) == []
    assert (outside / "keep.txt").read_text() == "keep"


def test_recorded_path_that_became_a_link_is_skipped(proj, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    run = proj.root / "runs" / "r1"
    for child in run.iterdir():
        if child.is_dir():
            import shutil

            shutil.rmtree(child)
        else:
            child.unlink()
    run.rmdir()
    os.symlink(outside, run)
    plan = projects.plan_delete(proj.root)
    assert run not in plan.paths
    assert plan.skipped and plan.skipped[0][0] == "runs/r1"
    projects.delete_project(plan)
    assert (outside / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("entry", ["../escape", "/etc", ".", "", ".lobster_workspace"])
def test_unsafe_recorded_paths_are_skipped(proj, entry):
    path = proj.workspace / "created_paths.json"
    data = json.loads(path.read_text())
    data["paths"].append(entry)
    path.write_text(json.dumps(data))
    plan = projects.plan_delete(proj.root)
    assert plan.paths == [proj.root / "runs" / "r1", proj.workspace]
    assert any(rel == entry for rel, _ in plan.skipped)


def test_refuses_while_a_run_holds_the_folder(proj):
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        lock = proj.root / "runs" / "r1" / ".lobster_workspace" / "mcp.lock"
        lock.write_text(f"{sleeper.pid}\n")
        with pytest.raises(ProjectError) as exc:
            projects.plan_delete(proj.root)
        assert exc.value.code == "in_use"
        assert proj.workspace.is_dir()
    finally:
        sleeper.kill()
        sleeper.wait()


def test_forget_project_drops_only_the_entry(home):
    projects.init_project(home / "a")
    projects.init_project(home / "b")
    assert projects.forget_project(home / "a") is True
    assert projects.forget_project(home / "a") is False
    assert [e.name for e in projects.registered_projects()] == ["b"]
    assert (home / "a" / ".lobster_workspace" / "project.json").is_file()
