"""Project discovery: which workspace a folder resolves to without a flag."""

import json
import os
from pathlib import Path

import pytest

from lobster.core.runtime import workspace as ws

PROJECT = {
    "schema_version": "lobster.project/1",
    "project_id": "6f1c1f2e-3b8a-4d0e-9a55-0a1b2c3d4e5f",
    "name": "p",
    "created_at": "2026-11-13T17:00:00Z",
    "cloud": None,
}


def make_project(folder: Path) -> Path:
    (folder / ".lobster_workspace").mkdir(parents=True)
    (folder / ".lobster_workspace" / "project.json").write_text(json.dumps(PROJECT))
    return folder


def make_legacy(folder: Path) -> Path:
    (folder / ".lobster_workspace").mkdir(parents=True)
    return folder


@pytest.fixture
def tree(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    return home


def resolved(start: Path, monkeypatch) -> Path:
    monkeypatch.chdir(start)
    return ws.resolve_workspace(create=False)


def test_start_folder_project_is_found(tree):
    make_project(tree / "a")
    assert ws.find_project_root(tree / "a") == tree / "a"


def test_parent_project_found_from_subfolder(tree, monkeypatch):
    make_project(tree / "a")
    sub = tree / "a" / "x" / "y"
    sub.mkdir(parents=True)
    assert ws.find_project_root(sub) == tree / "a"
    assert resolved(sub, monkeypatch) == tree / "a" / ".lobster_workspace"


def test_nearest_project_wins(tree):
    make_project(tree / "a")
    make_project(tree / "a" / "b")
    (tree / "a" / "b" / "c").mkdir()
    assert ws.find_project_root(tree / "a" / "b" / "c") == tree / "a" / "b"


def test_explicit_path_beats_discovery(tree, tmp_path):
    make_project(tree / "a")
    explicit = tmp_path / "elsewhere"
    assert ws.resolve_workspace(explicit, create=False) == explicit


def test_env_beats_discovery(tree, tmp_path, monkeypatch):
    make_project(tree / "a")
    env_ws = tmp_path / "env_ws"
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(env_ws))
    assert resolved(tree / "a", monkeypatch) == env_ws


def test_legacy_workspace_in_start_folder_is_used_as_today(tree, monkeypatch):
    make_project(tree / "a")
    make_legacy(tree / "a" / "b")
    assert ws.find_project_root(tree / "a" / "b") is None
    assert (
        resolved(tree / "a" / "b", monkeypatch)
        == tree / "a" / "b" / ".lobster_workspace"
    )


def test_parent_legacy_workspace_stops_the_walk(tree, monkeypatch):
    make_project(tree / "a")
    make_legacy(tree / "a" / "b")
    sub = tree / "a" / "b" / "c"
    sub.mkdir()
    assert ws.find_project_root(sub) is None
    # same result as before discovery existed: cwd/.lobster_workspace
    assert resolved(sub, monkeypatch) == sub / ".lobster_workspace"


def test_legacy_only_tree_resolves_as_before(tree, monkeypatch):
    make_legacy(tree / "a")
    sub = tree / "a" / "sub"
    sub.mkdir()
    assert resolved(sub, monkeypatch) == sub / ".lobster_workspace"


def test_nothing_found_resolves_as_before(tree, monkeypatch):
    sub = tree / "a" / "b"
    sub.mkdir(parents=True)
    assert ws.find_project_root(sub) is None
    assert resolved(sub, monkeypatch) == sub / ".lobster_workspace"


def test_the_home_workspace_is_never_a_project(tree):
    make_project(tree)
    sub = tree / "a"
    sub.mkdir()
    assert ws.find_project_root(sub) is None
    assert ws.find_project_root(tree) is None


def test_never_above_home(tree, tmp_path):
    make_project(tmp_path)  # a project above the home folder
    sub = tree / "a"
    sub.mkdir()
    assert ws.find_project_root(sub) is None


def test_stops_on_device_change(tree, monkeypatch):
    make_project(tree / "a")
    sub = tree / "a" / "mnt" / "deep"
    sub.mkdir(parents=True)
    real = ws._device

    def fake(path):
        # the folder "mnt" and everything under it is another filesystem
        return -1 if "mnt" in Path(path).parts else real(path)

    monkeypatch.setattr(ws, "_device", fake)
    assert ws.find_project_root(sub) is None
    monkeypatch.setattr(ws, "_device", real)
    assert ws.find_project_root(sub) == tree / "a"


def test_root_is_never_inspected(tree, monkeypatch):
    seen = []
    real = Path.is_dir

    def spy(self, *a, **k):
        seen.append(str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "is_dir", spy)
    ws.find_project_root(Path("/private/tmp"))
    assert "/.lobster_workspace" not in seen


def test_symlinked_start_folder(tree):
    make_project(tree / "real")
    (tree / "real" / "sub").mkdir()
    os.symlink(tree / "real", tree / "link")
    assert ws.find_project_root(tree / "link" / "sub") == tree / "link"


def test_symlinked_workspace_stops_the_walk(tree):
    make_project(tree / "a")
    target = make_project(tree / "store")
    (tree / "a" / "b").mkdir()
    os.symlink(target / ".lobster_workspace", tree / "a" / "b" / ".lobster_workspace")
    sub = tree / "a" / "b" / "c"
    sub.mkdir()
    assert ws.find_project_root(tree / "a" / "b") is None
    assert ws.find_project_root(sub) is None


def test_symlinked_project_file_is_not_a_project(tree):
    folder = make_legacy(tree / "a")
    elsewhere = tree / "real.json"
    elsewhere.write_text(json.dumps(PROJECT))
    os.symlink(elsewhere, folder / ".lobster_workspace" / "project.json")
    assert ws.find_project_root(folder) is None
