"""Review fixes: what delete may remove, home folders, names, ids and key words."""

import json
import os
import subprocess
import sys
import uuid

import pytest

from lobster.core import projects
from lobster.core.projects import ProjectError
from lobster.core.runtime.workspace import find_project_root

MARKER = "mcp_turns.json"


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
    return project


def make_run(project, name="r1", marker=True):
    run = project.root / "runs" / name
    (run / ".lobster_workspace").mkdir(parents=True)
    if marker:
        (run / ".lobster_workspace" / MARKER).write_text("{}")
    return run


def record(project, *entries):
    path = project.workspace / "created_paths.json"
    path.write_text(json.dumps({"paths": list(entries)}))


# only server run folders --------------------------------------------------


def test_only_a_run_folder_with_the_server_marker_is_removed(proj):
    run = make_run(proj, "r1")
    (proj.root / "src").mkdir()
    (proj.root / "data").mkdir()
    (proj.root / ".git").mkdir()
    (proj.root / "runs" / "plain").mkdir()
    (proj.root / "runs" / "a" / "b" / ".lobster_workspace").mkdir(parents=True)
    (proj.root / "runs" / "a" / "b" / ".lobster_workspace" / MARKER).write_text("{}")
    entries = ["src", "data", ".git", "runs", "runs/plain", "runs/a/b", "runs/r1"]
    record(proj, *entries)
    plan = projects.plan_delete(proj.root)
    assert plan.paths == [run, proj.workspace]
    skipped = dict(plan.skipped)
    assert set(skipped) == set(entries) - {"runs/r1"}
    assert all(why == "not a run folder Lobster made" for why in skipped.values())
    assert projects.delete_project(plan) == []
    for name in ("src", "data", ".git", "runs/plain", "runs/a/b"):
        assert (proj.root / name).is_dir()


def test_dot_git_is_never_removed_even_with_the_marker(proj):
    git = proj.root / "runs" / ".git"
    (git / ".lobster_workspace").mkdir(parents=True)
    (git / ".lobster_workspace" / MARKER).write_text("{}")
    record(proj, "runs/.git")
    assert projects.plan_delete(proj.root).paths == [proj.workspace]


def test_an_unfinished_leftover_marks_a_run_folder_too(proj):
    run = proj.root / "runs" / "u"
    (run / ".lobster_workspace.unfinished-abc123").mkdir(parents=True)
    record(proj, "runs/u")
    assert projects.plan_delete(proj.root).paths == [run, proj.workspace]


def test_a_marker_that_is_a_link_does_not_count(proj, tmp_path):
    run = proj.root / "runs" / "l"
    (run / ".lobster_workspace").mkdir(parents=True)
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    os.symlink(target, run / ".lobster_workspace" / MARKER)
    record(proj, "runs/l")
    assert projects.plan_delete(proj.root).paths == [proj.workspace]


# a live lock below a recorded folder ------------------------------------


@pytest.mark.parametrize("below", ["sub", "sub/deeper"])
def test_a_live_lock_below_a_run_folder_refuses_the_delete(proj, below):
    make_run(proj, "r1")
    record(proj, "runs/r1")
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        ws = proj.root / "runs" / "r1" / below / ".lobster_workspace"
        ws.mkdir(parents=True)
        (ws / "mcp.lock").write_text(f"{sleeper.pid}\n")
        with pytest.raises(ProjectError) as exc:
            projects.plan_delete(proj.root)
        assert exc.value.code == "in_use"
        assert proj.workspace.is_dir()
    finally:
        sleeper.kill()
        sleeper.wait()


# an unreadable record of run folders ------------------------------------


@pytest.mark.parametrize(
    "content", ['{"paths": 5}', '{"paths": null}', "[1]", "{nope", '{"paths": [1, {}]}']
)
def test_a_malformed_created_paths_file_keeps_run_folders_and_warns(proj, content):
    run = make_run(proj, "r1")
    (proj.workspace / "created_paths.json").write_text(content)
    plan = projects.plan_delete(proj.root)
    assert plan.paths == [proj.workspace]
    assert any("unreadable" in w for w in plan.warnings)
    assert projects.delete_project(plan) == []
    assert run.is_dir()


def test_no_record_means_no_warning(proj):
    assert projects.plan_delete(proj.root).warnings == []


# repeats, files, a path gone at delete time ------------------------------


def test_repeated_and_file_records_are_planned_once_and_never_fail(proj):
    run = make_run(proj, "r1")
    (proj.root / "README.md").write_text("mine")
    record(proj, "runs/r1", "runs/r1", "runs/./r1", "README.md")
    plan = projects.plan_delete(proj.root)
    assert plan.paths == [run, proj.workspace]
    assert projects.delete_project(plan) == []
    assert (proj.root / "README.md").read_text() == "mine"


def test_a_path_that_is_already_gone_is_done_and_the_entry_is_forgotten(proj):
    run = make_run(proj, "r1")
    record(proj, "runs/r1")
    plan = projects.plan_delete(proj.root)
    import shutil

    shutil.rmtree(run)
    assert projects.delete_project(plan) == []
    assert projects.registered_projects() == []


def test_the_entry_goes_when_the_workspace_is_gone_even_if_a_run_failed(
    proj, monkeypatch
):
    run = make_run(proj, "r1")
    record(proj, "runs/r1")
    plan = projects.plan_delete(proj.root)
    real = projects.shutil.rmtree

    def flaky(path, *a, **k):
        if os.fspath(path) == os.fspath(run):
            raise OSError(13, "Permission denied")
        return real(path, *a, **k)

    monkeypatch.setattr(projects.shutil, "rmtree", flaky)
    errors = projects.delete_project(plan)
    assert errors and "Permission denied" in errors[0]
    assert not proj.workspace.exists()
    assert projects.registered_projects() == []


# an unreadable project.json ----------------------------------------------


@pytest.mark.parametrize(
    "content", ["{nope", '{"schema_version": "lobster.project/1", "project_id": "x"}']
)
def test_a_broken_project_file_can_still_be_deleted(proj, content):
    (proj.workspace / "project.json").write_text(content)
    run = make_run(proj, "r1")
    record(proj, "runs/r1")
    plan = projects.plan_delete(proj.root)
    assert plan.paths == [run, proj.workspace]
    assert plan.project.name == "p"
    assert any("project.json" in w for w in plan.warnings)
    assert projects.delete_project(plan) == []
    assert not proj.workspace.exists()
    assert projects.registered_projects() == []


# names -------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["a\x1b[31mb", "a‮b", "a​b", "a\nb", "a\x00b"])
def test_a_name_with_control_or_format_characters_is_refused(home, bad):
    with pytest.raises(ProjectError) as exc:
        projects.init_project(home / "x", name=bad)
    assert exc.value.code == "bad_argument"
    assert not (home / "x" / ".lobster_workspace").exists()


def test_a_hostile_name_read_from_disk_is_shown_without_the_characters(proj):
    data = json.loads((proj.workspace / "project.json").read_text())
    data["name"] = "x\x1b[31mRED‮evil"
    (proj.workspace / "project.json").write_text(json.dumps(data))
    assert projects.load_project(proj.root).name == "x[31mREDevil"
    registry = projects.registry_path()
    entries = json.loads(registry.read_text())
    entries["projects"][0]["name"] = data["name"]
    registry.write_text(json.dumps(entries))
    assert projects.registered_projects()[0].name == "x[31mREDevil"


# ids and dates ------------------------------------------------------------


def identity(**over):
    base = {
        "schema_version": "lobster.project/1",
        "project_id": str(uuid.uuid4()),
        "name": "n",
        "created_at": "2026-11-13T17:00:00Z",
        "cloud": None,
    }
    base.update(over)
    return base


@pytest.mark.parametrize(
    "change",
    [
        lambda i: str(uuid.UUID(i)).upper(),
        lambda i: "{" + i + "}",
        lambda i: "urn:uuid:" + i,
        lambda i: i.replace("-", ""),
    ],
)
def test_project_id_must_be_a_canonical_lowercase_uuid4(proj, change):
    new = change(str(uuid.uuid4()))
    (proj.workspace / "project.json").write_text(json.dumps(identity(project_id=new)))
    with pytest.raises(ProjectError) as exc:
        projects.load_project(proj.root)
    assert exc.value.code == "invalid_project"
    with pytest.raises(ProjectError):
        projects.write_project_file(proj.workspace, identity(project_id=new))


def test_created_at_takes_ascii_digits_only(proj):
    fullwidth = "２０２６-11-13T17:00:00Z"
    with pytest.raises(ProjectError):
        projects.write_project_file(proj.workspace, identity(created_at=fullwidth))


# credential words ---------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["authorization", "Authorization", "bearer", "bearer_token", "passphrase", "pwd"]
    + ["auth", "cred", "creds", "x-auth", "credentials"],
)
def test_more_credential_words_are_refused(proj, key):
    (proj.workspace / "project.json").write_text(json.dumps({**identity(), key: "v"}))
    with pytest.raises(ProjectError) as exc:
        projects.load_project(proj.root)
    assert "credential-like keys" in exc.value.message


@pytest.mark.parametrize("key", ["author", "authority", "keywords", "credits"])
def test_near_misses_are_not_credential_words(proj, key):
    (proj.workspace / "project.json").write_text(json.dumps({**identity(), key: "v"}))
    assert projects.load_project(proj.root).name == "n"


# the home folder and its parents ----------------------------------------


def test_home_and_its_parents_cannot_be_projects(home, tmp_path):
    for folder in (home, tmp_path, home.parent):
        with pytest.raises(ProjectError) as exc:
            projects.init_project(folder)
        assert exc.value.code == "unsafe_path"
        assert not (folder / ".lobster_workspace").exists()
    with pytest.raises(ProjectError):
        projects.init_project("/")


def test_a_project_file_in_the_home_workspace_is_ignored_by_discovery(home):
    (home / ".lobster_workspace").mkdir()
    (home / ".lobster_workspace" / "project.json").write_text(json.dumps(identity()))
    (home / "work" / "deep").mkdir(parents=True)
    assert find_project_root(home) is None
    assert find_project_root(home / "work" / "deep") is None
    assert projects.find_project(home / "work") is None


def test_a_folder_below_home_still_finds_its_project(home):
    project, _ = projects.init_project(home / "p")
    (home / "p" / "a" / "b").mkdir(parents=True)
    assert find_project_root(home / "p" / "a" / "b") == project.root


# second round -------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every folder")
def test_an_unreadable_run_folder_is_kept_with_a_warning_and_no_traceback(proj):
    run = make_run(proj, "r1")
    other = make_run(proj, "r2")
    record(proj, "runs/r1", "runs/r2")
    run.chmod(0)
    try:
        plan = projects.plan_delete(proj.root)
        assert plan.paths == [other, proj.workspace]
        assert any(f"could not read {run}; kept" in w for w in plan.warnings)
        assert projects.delete_project(plan) == []
    finally:
        run.chmod(0o755)
    assert run.is_dir()


def test_the_home_folder_is_recognised_by_identity_not_by_text(home, tmp_path):
    link = tmp_path / "alias"
    os.symlink(home, link)
    (home / "x").mkdir()
    for folder in (link, home / "x" / "..", tmp_path / "home" / ".", link / "."):
        assert projects.is_home_or_above(folder)
        with pytest.raises(ProjectError) as exc:
            projects.init_project(folder)
        assert exc.value.code == "unsafe_path"
    assert not projects.is_home_or_above(home / "x")
    assert not projects.is_home_or_above(home / "not-yet-made")


def test_a_case_variant_of_home_is_still_home(home):
    swapped = type(home)(str(home).swapcase())
    if not swapped.exists():
        pytest.skip("case-sensitive filesystem")
    assert projects.is_home_or_above(swapped)
    (home / ".lobster_workspace").mkdir()
    (home / ".lobster_workspace" / "project.json").write_text(json.dumps(identity()))
    (swapped / "sub").mkdir()
    assert find_project_root(swapped / "sub") is None


def test_a_marker_reached_through_a_link_does_not_count(proj, tmp_path):
    real = tmp_path / "real_ws"
    real.mkdir()
    (real / MARKER).write_text("{}")
    run = proj.root / "runs" / "viaLink"
    run.mkdir(parents=True)
    os.symlink(real, run / ".lobster_workspace")
    record(proj, "runs/viaLink")
    plan = projects.plan_delete(proj.root)
    assert plan.paths == [proj.workspace]
    assert dict(plan.skipped)["runs/viaLink"] == "not a run folder Lobster made"


def test_a_broken_identity_has_no_id_and_no_date(proj):
    (proj.workspace / "project.json").write_text("{nope")
    plan = projects.plan_delete(proj.root)
    block = projects.project_to_dict(plan.project)
    assert block["project_id"] is None and block["created_at"] is None


# the .env note ---------------------------------------------------------------------


def test_env_note_is_key_when_init_wrote_it(proj):
    (proj.root / ".env").write_text("FOO=1\n")
    projects.record_created_file(proj, ".env")
    assert projects.plan_delete(proj.root).env_note == "key"


@pytest.mark.parametrize(
    "text, note",
    [
        ("FOO=1\n", "plain"),
        ("OPENAI_API_KEY=\n", "key"),
        ("export GITHUB_TOKEN = x\n", "key"),
        ("# ANTHROPIC_API_KEY=x\nFOO=1\n", "plain"),
        ("API_KEY_NOTE=1\n", "plain"),
        ("AWS_SECRET_ACCESS_KEY=B\n", "key"),
        ("DB_PASSWORD=x\n", "key"),
        ("DB_SECRET=x\n", "key"),
        ("export client_secret = x\n", "key"),
        ("DB_PASSWD=x\n", "key"),
        ("SECRET_FOLDER=x\n", "plain"),
        ("PASSWORD_HINT=x\n", "plain"),
    ],
)
def test_env_note_reads_names_only(proj, text, note):
    (proj.root / ".env").write_text(text)
    assert projects.plan_delete(proj.root).env_note == note


def test_no_env_no_note(proj):
    assert projects.plan_delete(proj.root).env_note is None


def test_recording_a_file_keeps_the_run_folder_record(proj):
    run = make_run(proj, "r1")
    projects.record_created_path(proj, run)
    projects.record_created_file(proj, ".env")
    projects.record_created_path(proj, make_run(proj, "r2"))
    data = json.loads((proj.workspace / "created_paths.json").read_text())
    assert data["paths"] == ["runs/r1", "runs/r2"] and data["files"] == [".env"]
