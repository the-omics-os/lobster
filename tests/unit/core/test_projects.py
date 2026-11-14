"""Local projects: identity file, settings read from the workspace, registry."""

import json
import os
import re
import shutil
import uuid
from pathlib import Path

import pytest

from lobster.core import projects
from lobster.core.projects import ProjectError


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    from lobster.config import global_config

    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "cfg" / "lobster")
    return home


def project_file(root: Path) -> Path:
    return root / ".lobster_workspace" / "project.json"


def good_identity() -> dict:
    return {
        "schema_version": "lobster.project/1",
        "project_id": str(uuid.uuid4()),
        "name": "n",
        "created_at": "2026-11-13T17:00:00Z",
        "cloud": None,
    }


def edit_identity(root: Path, **changes) -> None:
    path = project_file(root)
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def test_init_writes_identity_only(env):
    project, created = projects.init_project(env / "pbmc")
    assert created is True
    data = json.loads(project_file(env / "pbmc").read_text())
    assert set(data) == {
        "schema_version",
        "project_id",
        "name",
        "created_at",
        "cloud",
    }
    assert data["schema_version"] == "lobster.project/1"
    assert uuid.UUID(data["project_id"]).version == 4
    assert data["name"] == "pbmc"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", data["created_at"])
    assert data["cloud"] is None
    assert project.root == env / "pbmc"
    assert project.workspace == env / "pbmc" / ".lobster_workspace"


def test_init_again_keeps_identity(env):
    first, _ = projects.init_project(env / "p", name="One")
    again, created = projects.init_project(env / "p")
    assert created is False
    assert again.project_id == first.project_id
    assert again.created_at == first.created_at
    assert again.name == "One"


def test_init_again_with_name_renames_only(env):
    first, _ = projects.init_project(env / "p", name="One")
    renamed, created = projects.init_project(env / "p", name="Two")
    assert (created, renamed.name) == (False, "Two")
    assert renamed.project_id == first.project_id


def test_init_adopts_legacy_workspace_and_keeps_its_files(env):
    ws = env / "old" / ".lobster_workspace"
    ws.mkdir(parents=True)
    (ws / "provider_config.json").write_text('{"global_provider": "ollama"}')
    project, created = projects.init_project(env / "old")
    assert created is True
    assert (ws / "provider_config.json").read_text() == '{"global_provider": "ollama"}'
    assert project.project_id


@pytest.mark.parametrize("name", ["", "   ", "x" * 257])
def test_init_refuses_bad_name(env, name):
    with pytest.raises(ProjectError) as exc:
        projects.init_project(env / "p", name=name)
    assert exc.value.code == "bad_argument"
    assert not (env / "p" / ".lobster_workspace").exists()


def test_load_missing_is_not_a_project(env):
    (env / "x").mkdir()
    with pytest.raises(ProjectError) as exc:
        projects.load_project(env / "x")
    assert exc.value.code == "not_a_project"


@pytest.mark.parametrize(
    "changes",
    [
        {"project_id": "not-a-uuid"},
        {"project_id": str(uuid.uuid1())},
        {"name": ""},
        {"created_at": None},
        {"schema_version": "other/9"},
    ],
)
def test_load_refuses_invalid_identity(env, changes):
    projects.init_project(env / "p")
    edit_identity(env / "p", **changes)
    with pytest.raises(ProjectError) as exc:
        projects.load_project(env / "p")
    assert exc.value.code == "invalid_project"


def test_load_refuses_garbage(env):
    projects.init_project(env / "p")
    project_file(env / "p").write_text("{not json")
    with pytest.raises(ProjectError) as exc:
        projects.load_project(env / "p")
    assert exc.value.code == "invalid_project"


@pytest.mark.parametrize(
    "key",
    [
        "provider",
        "model",
        "api_key",
        "openai_api_key",
        "ANTHROPIC_KEY",
        "secret",
        "client_secret",
        "token",
        "access_token",
        "password",
        "credentials",
        "apiKey",
        "model_name",
    ],
)
def test_read_refuses_credential_like_keys(env, key):
    projects.init_project(env / "p")
    edit_identity(env / "p", **{key: "sentinel-value"})
    with pytest.raises(ProjectError) as exc:
        projects.load_project(env / "p")
    assert exc.value.code == "invalid_project"
    assert key in exc.value.message
    assert "sentinel-value" not in exc.value.message


def test_read_refuses_nested_credential_like_keys(env):
    projects.init_project(env / "p")
    edit_identity(
        env / "p", cloud={"api_base": "https://x", "auth": {"api_key": "sentinel"}}
    )
    with pytest.raises(ProjectError) as exc:
        projects.load_project(env / "p")
    assert "sentinel" not in exc.value.message


@pytest.mark.parametrize(
    "key", ["keywords", "monkey", "models_dir", "api_base", "note"]
)
def test_read_accepts_names_that_only_contain_a_word_part(env, key):
    projects.init_project(env / "p")
    edit_identity(env / "p", **{key: "fine"})
    assert projects.load_project(env / "p").name == "p"


def test_write_refuses_unknown_and_credential_keys(env):
    ws = env / "p" / ".lobster_workspace"
    ws.mkdir(parents=True)
    for extra in ({"provider": "x"}, {"model": "m"}, {"api_key": "k"}, {"other": 1}):
        with pytest.raises(ProjectError):
            projects.write_project_file(ws, {**good_identity(), **extra})
    assert not (ws / "project.json").exists()
    good = good_identity()
    projects.write_project_file(ws, good)
    assert json.loads((ws / "project.json").read_text()) == good


def test_write_refuses_cloud_block_in_this_release(env):
    ws = env / "p" / ".lobster_workspace"
    ws.mkdir(parents=True)
    with pytest.raises(ProjectError):
        projects.write_project_file(ws, {**good_identity(), "cloud": {"id": "x"}})


def test_find_project_and_project_for_workspace(env):
    made, _ = projects.init_project(env / "p")
    (env / "p" / "sub").mkdir()
    assert projects.find_project(env / "p" / "sub").project_id == made.project_id
    assert projects.find_project(env) is None
    ws = env / "p" / ".lobster_workspace"
    assert projects.project_for_workspace(ws).project_id == made.project_id
    assert projects.project_for_workspace(env / "p").project_id == made.project_id
    assert projects.project_for_workspace(env) is None
    (env / "legacy" / ".lobster_workspace").mkdir(parents=True)
    assert projects.project_for_workspace(env / "legacy") is None


def test_settings_read_from_workspace_files(env):
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig
    from lobster.config.workspace_config import WorkspaceProviderConfig

    project, _ = projects.init_project(env / "p")
    assert projects.project_settings(project) == projects.ProjectSettings(
        provider=None, model=None, experts=[], preset=None
    )
    cfg = WorkspaceProviderConfig(global_provider="ollama-cloud")
    cfg.set_model_for_provider("ollama-cloud", "glm-5.2")
    cfg.save(project.workspace)
    WorkspaceAgentConfig(enabled=["research_agent", "data_expert"]).save(
        project.workspace
    )
    got = projects.project_settings(project)
    assert got.provider == "ollama-cloud"
    assert got.model == "glm-5.2"
    assert got.experts == ["research_agent", "data_expert"]
    assert got.preset is None


def test_settings_preset_stands_for_its_experts(env):
    from lobster.config.agent_presets import expand_preset
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    project, _ = projects.init_project(env / "p")
    WorkspaceAgentConfig(preset="scrna-basic").save(project.workspace)
    got = projects.project_settings(project)
    assert got.preset == "scrna-basic"
    assert got.experts == expand_preset("scrna-basic")
    assert got.experts


def test_runs_dir_and_list_runs(env):
    project, _ = projects.init_project(env / "p")
    assert projects.runs_dir(project) == env / "p" / "runs"
    assert projects.list_runs(project) == []
    (env / "p" / "runs" / "b").mkdir(parents=True)
    (env / "p" / "runs" / "a").mkdir()
    (env / "p" / "runs" / "file.txt").write_text("x")
    assert projects.list_runs(project) == ["a", "b"]


def test_record_created_path(env):
    project, _ = projects.init_project(env / "p")
    run = env / "p" / "runs" / "r1"
    run.mkdir(parents=True)
    projects.record_created_path(project, run)
    projects.record_created_path(project, run)  # idempotent
    data = json.loads((project.workspace / "created_paths.json").read_text())
    assert data["paths"] == ["runs/r1"]
    assert "paths" not in json.loads(project_file(env / "p").read_text())


def test_record_created_path_refuses_outside_and_root(env, tmp_path):
    project, _ = projects.init_project(env / "p")
    for bad in (tmp_path / "elsewhere", env / "p", env / "p" / "..", env):
        with pytest.raises(ProjectError) as exc:
            projects.record_created_path(project, bad)
        assert exc.value.code == "unsafe_path"


def test_record_created_path_refuses_symlink_and_escape(env, tmp_path):
    project, _ = projects.init_project(env / "p")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, env / "p" / "runs_link")
    with pytest.raises(ProjectError):
        projects.record_created_path(project, env / "p" / "runs_link")
    with pytest.raises(ProjectError):
        projects.record_created_path(project, env / "p" / "runs_link" / "x")


def test_registry_lists_projects_and_marks_missing(env):
    a, _ = projects.init_project(env / "a", name="A")
    projects.init_project(env / "b", name="B")
    shutil.rmtree(env / "b")
    entries = {e.name: e for e in projects.registered_projects()}
    assert entries["A"].project_id == a.project_id
    assert entries["A"].missing is False
    assert entries["B"].missing is True
    assert Path(entries["A"].root) == env / "a"


def test_registry_is_not_duplicated_and_follows_rename(env):
    projects.init_project(env / "a", name="A")
    projects.init_project(env / "a", name="A2")
    assert [e.name for e in projects.registered_projects()] == ["A2"]


def test_registry_survives_garbage_file(env):
    path = projects.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[[[")
    assert projects.registered_projects() == []
    projects.init_project(env / "a")
    assert len(projects.registered_projects()) == 1


def test_module_import_is_light():
    import subprocess
    import sys

    code = (
        f"import sys; sys.path[:0] = {sys.path!r};"
        "import lobster.core.projects;"
        "print([m for m in ('langchain','langgraph','lobster.agents',"
        "'lobster.core.client') if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "[]", out.stderr


@pytest.mark.parametrize("name", ["token counts", "api key test", "model: provider"])
def test_credential_words_in_values_are_fine(env, name):
    project, _ = projects.init_project(env / "p", name=name)
    assert projects.load_project(env / "p").name == name
    assert project.name == name


def test_init_refuses_to_nest(env):
    outer, _ = projects.init_project(env / "p", name="Outer")
    for inner in (env / "p" / "sub", env / "p" / "new" / "deeper"):
        with pytest.raises(ProjectError) as exc:
            projects.init_project(inner)
        assert exc.value.code == "nested_project"
        assert "Outer" in exc.value.message and str(outer.root) in exc.value.message
        assert not (inner / ".lobster_workspace").exists()
    # the project itself can be initialised again
    assert projects.init_project(env / "p")[1] is False
