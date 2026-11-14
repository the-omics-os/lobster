"""``lobster skills list|install`` (Phase 0 B5)."""

import json

import pytest
from typer.testing import CliRunner

from lobster.cli import app
from lobster.cli_internal.commands.light.skills_commands import MARKER

runner = CliRunner()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".claude").mkdir()
    return tmp_path


def _install(*args):
    result = runner.invoke(app, ["skills", "install", "--json", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_list_finds_shipped_skills():
    result = runner.invoke(app, ["skills", "list", "--json"])
    assert result.exit_code == 0, result.output
    names = [s["name"] for s in json.loads(result.stdout)["skills"]]
    assert {"lobster-use", "lobster-dev"} <= set(names)


def test_dry_run_plans_detected_harness_only_and_writes_nothing(home):
    payload = _install("--dry-run")
    assert payload["schema_version"] == "lobster.skills-install/1"
    assert payload["dry_run"] is True
    assert {a["harness"] for a in payload["actions"]} == {"claude-code"}
    assert {a["action"] for a in payload["actions"]} == {"install"}
    assert not (home / ".claude" / "skills").exists()


def test_install_copies_and_replaces_only_ours(home):
    skills = home / ".claude" / "skills"
    user_skill = skills / "lobster-dev"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("mine")

    first = {a["skill"]: a for a in _install()["actions"]}
    assert first["lobster-use"]["action"] == "install"
    assert first["lobster-dev"]["action"] == "skip"
    assert (user_skill / "SKILL.md").read_text() == "mine"
    assert not (user_skill / MARKER).exists()
    assert (skills / "lobster-use" / "SKILL.md").is_file()
    marker = json.loads((skills / "lobster-use" / MARKER).read_text())
    assert marker["installed_by"] == "lobster-ai"

    stale = skills / "lobster-use" / "stale.md"
    stale.write_text("old")
    second = {a["skill"]: a for a in _install()["actions"]}
    assert second["lobster-use"]["action"] == "replace"
    assert not stale.exists()
    assert (skills / "lobster-use" / MARKER).is_file()


def test_json_shape_and_target(tmp_path):
    target = tmp_path / "custom"
    payload = _install("--target", str(target), "--dry-run")
    assert set(payload) == {"schema_version", "dry_run", "actions"}
    for action in payload["actions"]:
        assert set(action) >= {"harness", "skill", "source", "destination", "action"}
        assert action["destination"].startswith(str(target))


def test_unknown_harness_is_rejected(home):
    result = runner.invoke(app, ["skills", "install", "--harness", "nope"])
    assert result.exit_code == 2


def test_marker_must_be_lobster_json(home):
    skills = home / ".claude" / "skills"
    for name, text in (
        ("lobster-use", "not json"),
        ("lobster-dev", '{"installed_by": "x"}'),
    ):
        (skills / name).mkdir(parents=True)
        (skills / name / MARKER).write_text(text)
    actions = {a["skill"]: a["action"] for a in _install("--dry-run")["actions"]}
    assert actions["lobster-use"] == "skip"
    assert actions["lobster-dev"] == "skip"


def test_symlinked_destination_is_skipped(home, tmp_path):
    skills = home / ".claude" / "skills"
    skills.mkdir(parents=True)
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / MARKER).write_text('{"installed_by": "lobster-ai"}')
    (skills / "lobster-use").symlink_to(real, target_is_directory=True)
    actions = {a["skill"]: a["action"] for a in _install()["actions"]}
    assert actions["lobster-use"] == "skip"
    assert (real / MARKER).is_file()


def test_destination_changed_after_the_plan_is_left_alone(tmp_path):
    from lobster.cli_internal.commands.light.skills_commands import (
        apply_install,
        plan_install,
    )

    apply_install(plan_install([], tmp_path))
    actions = plan_install([], tmp_path)
    replaced = [a for a in actions if a["action"] == "replace"]
    assert replaced
    # Between plan and apply, someone else's folder takes the place of ours.
    dest = tmp_path / replaced[0]["skill"]
    (dest / MARKER).unlink()
    (dest / "notes.md").write_text("mine")
    apply_install(actions)
    assert (dest / "notes.md").read_text() == "mine"
    assert replaced[0]["action"] == "skip"
    assert "changed" in replaced[0]["reason"]
