"""A project's saved language is read from the workspace the command will use."""

import json

import pytest

from lobster.i18n import resolve_lang


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project whose workspace says zh; the user-wide config says nothing."""
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "study"
    workspace = root / ".lobster_workspace"
    workspace.mkdir(parents=True)
    (workspace / "project.json").write_text("{}")
    (workspace / "provider_config.json").write_text(json.dumps({"lang": "zh"}))
    return root


def _lang(*args, **environ):
    return resolve_lang(
        ["lobster", *args], {"XDG_CONFIG_HOME": "/nonexistent", **environ}
    )


def test_a_project_named_with_w_reads_its_workspace(project):
    assert _lang("query", "-w", str(project)) == "zh"
    assert _lang("query", f"--workspace={project}") == "zh"
    assert _lang("query", LOBSTER_WORKSPACE=str(project)) == "zh"


def test_a_project_folder_with_a_file_of_the_same_name_reads_its_workspace(project):
    # the user's own file in the project folder is not the project's settings
    (project / "provider_config.json").write_text(json.dumps({"tool": "mine"}))
    assert _lang("query", "-w", str(project)) == "zh"


def test_a_command_run_inside_a_project_reads_its_workspace(project, monkeypatch):
    inner = project / "data" / "raw"
    inner.mkdir(parents=True)
    monkeypatch.chdir(inner)
    assert _lang("query") == "zh"


def test_a_folder_outside_any_project_keeps_the_default(project, monkeypatch):
    outside = project.parent / "elsewhere"
    outside.mkdir()
    monkeypatch.chdir(outside)
    assert _lang("query") == "en"
