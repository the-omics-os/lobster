"""``lobster projects list | info | delete`` and their ``--json`` schema."""

import json
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from lobster.core import projects

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "lobster"
        / "schemas"
        / "lobster.project.v1.schema.json"
    ).read_text(encoding="utf-8")
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.chdir(h)
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    from lobster.config import global_config

    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "cfg" / "lobster")
    return h


def invoke(*args):
    from lobster.cli import app

    return CliRunner().invoke(app, list(args))


def valid(payload):
    Draft202012Validator(SCHEMA).validate(payload)
    return payload


@pytest.fixture
def proj(home):
    project, _ = projects.init_project(home / "p", name="PBMC")
    run = home / "p" / "runs" / "r1"
    (run / ".lobster_workspace").mkdir(parents=True)
    (run / ".lobster_workspace" / "mcp_turns.json").write_text("{}")
    projects.record_created_path(project, run)
    (home / "p" / "keep.txt").write_text("mine")
    return project


def test_list_json_and_text(home, proj):
    projects.init_project(home / "gone")
    import shutil

    shutil.rmtree(home / "gone")
    out = valid(json.loads(invoke("projects", "list", "--json").output))
    assert out["total"] == 2
    by_name = {p["name"]: p for p in out["projects"]}
    assert by_name["PBMC"]["missing"] is False
    assert by_name["gone"]["missing"] is True
    text = invoke("projects", "list").output
    assert "PBMC" in text and "(missing)" in text


def test_list_empty(home):
    assert valid(json.loads(invoke("projects", "list", "--json").output)) == {
        "schema_version": "lobster.project/1",
        "projects": [],
        "total": 0,
    }
    assert "lobster init" in invoke("projects", "list").output


def test_info_json_from_folder_and_from_subfolder(home, proj):
    out = valid(json.loads(invoke("projects", "info", str(proj.root), "--json").output))
    assert out["project"]["name"] == "PBMC"
    assert out["project"]["project_id"] == proj.project_id
    assert out["project"]["status"] == "active"
    assert out["local"]["root"] == str(proj.root)
    assert out["local"]["runs"] == ["r1"]
    sub = proj.root / "sub"
    sub.mkdir()
    os.chdir(sub)
    again = valid(json.loads(invoke("projects", "info", "--json").output))
    assert again["project"]["project_id"] == proj.project_id


def test_info_json_never_carries_a_credential_field(home, proj):
    text = invoke("projects", "info", str(proj.root), "--json").output.lower()
    for word in ("api_key", "secret", "password", "token"):
        assert word not in text


def test_info_text(home, proj):
    out = invoke("projects", "info", str(proj.root)).output
    assert "PBMC" in out and str(proj.root) in out and "runs:" in out


def test_info_outside_a_project_is_a_json_error(home):
    result = invoke("projects", "info", "--json")
    assert result.exit_code == 1
    err = valid(json.loads(result.output))
    assert err["code"] == "not_a_project" and err["kind"] == "not_found"
    assert err["status"] is None and err["retryable"] is False


def test_info_outside_a_project_text_error_shape(home):
    result = invoke("projects", "info")
    assert result.exit_code == 1
    assert "✗" in result.output and "lobster init" in result.output


def test_info_refuses_credential_keys_in_project_file(home, proj):
    path = proj.workspace / "project.json"
    data = json.loads(path.read_text())
    data["api_key"] = "sentinel-secret"
    path.write_text(json.dumps(data))
    result = invoke("projects", "info", str(proj.root), "--json")
    assert result.exit_code == 1
    assert "sentinel-secret" not in result.output
    assert valid(json.loads(result.output))["code"] == "invalid_project"


def test_hidden_singular_alias(home, proj):
    assert invoke("project", "info", str(proj.root), "--json").exit_code == 0
    from lobster.cli import app

    help_text = CliRunner().invoke(app, ["--help"]).output
    assert "projects" in help_text
    assert " project " not in help_text


def test_delete_without_terminal_and_without_yes_changes_nothing(home, proj):
    result = invoke("projects", "delete", str(proj.root))
    assert result.exit_code == 2
    assert proj.workspace.is_dir()
    assert (proj.root / "runs" / "r1").is_dir()
    result = invoke("projects", "delete", str(proj.root), "--json")
    assert result.exit_code == 2
    assert valid(json.loads(result.output))["kind"] == "usage"


def test_delete_dry_run_lists_and_changes_nothing(home, proj):
    result = invoke("projects", "delete", str(proj.root), "--dry-run", "--json")
    assert result.exit_code == 0
    out = valid(json.loads(result.output))
    assert out["dry_run"] is True and out["deleted"] is False
    assert str(proj.workspace) in out["removed"]
    assert proj.workspace.is_dir()
    text = invoke("projects", "delete", str(proj.root), "--dry-run").output
    assert "would remove" in text and str(proj.workspace) in text


def test_delete_yes(home, proj):
    result = invoke("projects", "delete", str(proj.root), "--yes", "--json")
    assert result.exit_code == 0, result.output
    out = valid(json.loads(result.output))
    assert out["deleted"] is True
    assert not proj.workspace.exists()
    assert not (proj.root / "runs" / "r1").exists()
    assert (proj.root / "keep.txt").read_text() == "mine"
    assert proj.root.is_dir()
    assert projects.registered_projects() == []


def test_delete_on_a_terminal_asks_first(home, proj, monkeypatch):
    monkeypatch.setattr(
        "lobster.cli_internal.commands.light.project_commands._tty", lambda: True
    )
    from lobster.cli import app

    refused = CliRunner().invoke(
        app, ["projects", "delete", str(proj.root)], input="n\n"
    )
    assert refused.exit_code == 1
    assert "Nothing deleted" in refused.output
    assert proj.workspace.is_dir()
    accepted = CliRunner().invoke(
        app, ["projects", "delete", str(proj.root)], input="y\n"
    )
    assert accepted.exit_code == 0, accepted.output
    assert not proj.workspace.exists()


def test_delete_refuses_a_folder_lobster_did_not_create(home):
    repo = home / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("x")
    result = invoke("projects", "delete", str(repo), "--yes", "--json")
    assert result.exit_code == 1
    assert valid(json.loads(result.output))["code"] == "not_a_project"
    assert (repo / "main.py").read_text() == "x"


def test_delete_refuses_a_home_folder_without_a_project(home):
    (home / "notes.txt").write_text("x")
    assert invoke("projects", "delete", str(home), "--yes").exit_code == 1
    assert (home / "notes.txt").exists()


def test_delete_in_a_subfolder_needs_the_folder_named(home, proj):
    sub = proj.root / "sub"
    sub.mkdir()
    os.chdir(sub)
    result = invoke("projects", "delete", "--yes", "--json")
    assert result.exit_code == 2
    assert proj.workspace.is_dir()
    os.chdir(proj.root)
    assert invoke("projects", "delete", "--yes").exit_code == 0
    assert not proj.workspace.exists()


def test_delete_of_a_linked_folder_is_refused(home, proj):
    os.symlink(proj.root, home / "link")
    result = invoke("projects", "delete", str(home / "link"), "--yes", "--json")
    assert result.exit_code == 1
    assert valid(json.loads(result.output))["code"] == "unsafe_path"
    assert proj.workspace.is_dir()


def test_delete_forgets_a_missing_entry(home):
    projects.init_project(home / "gone")
    import shutil

    shutil.rmtree(home / "gone")
    result = invoke("projects", "delete", str(home / "gone"), "--yes", "--json")
    assert result.exit_code == 0, result.output
    assert valid(json.loads(result.output))["deleted"] is True
    assert projects.registered_projects() == []


def test_help_lists_the_three_verbs(home):
    out = invoke("projects", "--help").output
    for verb in ("list", "info", "delete"):
        assert verb in out
    assert "create" not in out and "update" not in out


def test_manifest_lists_the_project_verbs_for_scientists(home):
    from lobster.cli import app
    from lobster.cli_internal.command_registry import build_manifest

    commands = {c["name"]: c for c in build_manifest(app)["commands"]}
    for verb in ("list", "info", "delete"):
        assert commands[f"projects {verb}"]["scientist_facing"] is True
    assert not [n for n in commands if n.startswith("project ")]
    assert "projects create" not in commands and "projects update" not in commands
    assert commands["init"]["args"]
    assert any(a["name"] == "name" for a in commands["init"]["args"])


# review fixes: wording, warnings, broken files ----------------------------------


def test_delete_text_says_what_goes_and_what_stays(home, proj):
    (proj.root / ".env").write_text("OPENAI_API_KEY=x\n")
    text = invoke("projects", "delete", str(proj.root), "--dry-run").output
    assert (
        "This removes Lobster's workspace, including the data and sessions saved in it."
        in text
    )
    assert (
        ".env kept: it holds an API key; delete it yourself if you no longer need it"
        in text
    )
    assert "your own files" not in text


def test_delete_success_does_not_claim_the_workspace_data_survives(home, proj):
    text = invoke("projects", "delete", str(proj.root), "--yes").output
    assert "Deleted the Lobster workspace" in text
    assert "untouched" not in text or "outside the workspace" in text


def test_a_recorded_folder_that_is_not_a_run_is_listed_as_kept(home, proj):
    (proj.root / "src").mkdir()
    path = proj.workspace / "created_paths.json"
    data = json.loads(path.read_text())
    data["paths"].append("src")
    path.write_text(json.dumps(data))
    result = invoke("projects", "delete", str(proj.root), "--yes", "--json")
    out = valid(json.loads(result.output))
    assert {"path": "src", "reason": "not a run folder Lobster made"} in out["skipped"]
    assert (proj.root / "src").is_dir()
    text_proj = projects.init_project(home / "q")[0]
    (home / "q" / "src").mkdir()
    (text_proj.workspace / "created_paths.json").write_text('{"paths": ["src"]}')
    text = invoke("projects", "delete", str(home / "q"), "--dry-run").output
    assert "kept src: not a run folder Lobster made" in text


def test_an_unreadable_record_warns_and_never_shows_a_traceback(home, proj):
    (proj.workspace / "created_paths.json").write_text('{"paths": 5}')
    result = invoke("projects", "delete", str(proj.root), "--yes", "--json")
    assert result.exit_code == 0, result.output
    out = valid(json.loads(result.output))
    assert any("unreadable" in w for w in out["warnings"])
    assert (proj.root / "runs" / "r1").is_dir()
    assert "Traceback" not in result.output
    again = projects.init_project(home / "w")[0]
    (again.workspace / "created_paths.json").write_text("{nope")
    text = invoke("projects", "delete", str(home / "w"), "--yes").output
    assert "the record of run folders is unreadable; run folders are kept" in text


def test_a_broken_project_file_is_deleted_with_yes(home, proj):
    (proj.workspace / "project.json").write_text('{"schema_version": "x"}')
    result = invoke("projects", "delete", str(proj.root), "--yes", "--json")
    assert result.exit_code == 0, result.output
    out = valid(json.loads(result.output))
    assert out["deleted"] is True and out["warnings"]
    assert not proj.workspace.exists()


def test_hostile_names_print_without_escape_sequences(home, proj):
    data = json.loads((proj.workspace / "project.json").read_text())
    data["name"] = "x\x1b[31mRED‮evil"
    (proj.workspace / "project.json").write_text(json.dumps(data))
    for argv in (("projects", "info", str(proj.root)), ("projects", "list")):
        out = invoke(*argv).output
        assert "\x1b" not in out and "‮" not in out


def test_schema_types_the_warnings_and_allows_a_null_id_and_date():
    delete = next(s for s in SCHEMA["oneOf"] if s["title"] == "delete")
    assert delete["properties"]["warnings"]["items"] == {"type": "string"}
    strict = SCHEMA["$defs"]["project"]["properties"]
    assert strict["project_id"]["type"] == "string"
    assert strict["created_at"]["type"] == "string"
    assert delete["properties"]["project"]["$ref"] == "#/$defs/deleted_project"
    loose = SCHEMA["$defs"]["deleted_project"]["properties"]
    assert "null" in loose["project_id"]["type"]
    assert "null" in loose["created_at"]["type"]
    from jsonschema import ValidationError

    base = {
        "schema_version": "lobster.project/1",
        "removed": [],
        "kept": [],
        "skipped": [],
        "dry_run": False,
        "deleted": True,
    }
    valid({**base, "warnings": ["a"]})
    with pytest.raises(ValidationError):
        valid({**base, "warnings": [1]})


def test_a_broken_identity_reports_null_id_and_date(home, proj):
    (proj.workspace / "project.json").write_text("{nope")
    out = valid(
        json.loads(
            invoke("projects", "delete", str(proj.root), "--yes", "--json").output
        )
    )
    assert out["project"]["project_id"] is None
    assert out["project"]["created_at"] is None


def test_env_line_depends_on_what_the_file_holds(home, proj):
    (proj.root / ".env").write_text("FOO=1\n")
    text = invoke("projects", "delete", str(proj.root), "--dry-run").output
    assert ".env kept" in text and "API key" not in text
    (proj.root / ".env").write_text("OPENAI_API_KEY=x\n")
    text = invoke("projects", "delete", str(proj.root), "--dry-run").output
    assert (
        ".env kept: it holds an API key; delete it yourself if you no longer need it"
        in text
    )


def test_only_the_delete_result_may_carry_a_null_id_or_date():
    from jsonschema import ValidationError

    nulled = {
        "project_id": None,
        "name": "p",
        "status": "active",
        "org_id": None,
        "created_at": None,
    }
    valid(
        {
            "schema_version": "lobster.project/1",
            "project": nulled,
            "removed": [],
            "kept": [],
            "skipped": [],
            "dry_run": False,
            "deleted": True,
        }
    )
    local = {
        "root": "/p",
        "workspace": "/p/.lobster_workspace",
        "provider": None,
        "model": None,
        "experts": [],
        "preset": None,
        "runs_folder": "/p/runs",
        "runs": [],
        "runs_total": 0,
    }
    with pytest.raises(ValidationError):
        valid(
            {"schema_version": "lobster.project/1", "project": nulled, "local": local}
        )
