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
    (project / "provider_config.json").write_text(json.dumps({"tool": "mine"}))
    monkeypatch.chdir(inner)
    assert _lang("query") == "zh"


def test_a_folder_outside_any_project_keeps_the_default(project, monkeypatch):
    outside = project.parent / "elsewhere"
    outside.mkdir()
    monkeypatch.chdir(outside)
    assert _lang("query") == "en"


# a language that cannot be read is no language: the CLI still starts


UTF16 = b"\xff\xfe{\x00}\x00"  # PowerShell 5.1's default for a redirected file


def test_a_config_saved_as_utf16_in_a_project_is_skipped(project, monkeypatch):
    (project / ".lobster_workspace" / "provider_config.json").write_bytes(UTF16)
    monkeypatch.chdir(project / ".lobster_workspace")
    assert _lang("query") == "en"
    assert _lang("query", "-w", str(project)) == "en"


def test_a_user_wide_config_saved_as_utf16_is_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "xdg" / "lobster").mkdir(parents=True)
    (tmp_path / "xdg" / "lobster" / "providers.json").write_bytes(UTF16)
    assert resolve_lang(["lobster"], {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}) == "en"


def test_an_unreadable_workspace_is_skipped(project):
    workspace = project / ".lobster_workspace"
    workspace.chmod(0)
    try:
        assert _lang("query", "-w", str(project)) == "en"
        assert _lang("query", "-w", str(workspace)) == "en"
    finally:
        workspace.chmod(0o755)


def test_a_deleted_current_folder_is_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    assert _lang("query") == "en"


def test_an_unreadable_user_wide_config_folder_is_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "xdg" / "lobster"
    folder.mkdir(parents=True)
    folder.chmod(0)
    try:
        env = {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}
        assert resolve_lang(["lobster"], env) == "en"
    finally:
        folder.chmod(0o755)


def test_a_workspace_under_the_home_folder_is_found(project, monkeypatch):
    monkeypatch.setenv("HOME", str(project.parent))
    assert _lang("query", "-w", "~/study") == "zh"


def test_a_workspace_under_an_unknown_user_keeps_the_default(project):
    assert _lang("query", "-w", "~nosuchuser/x") == "en"
    assert _lang("query", "--workspace=~nosuchuser/x") == "en"
    assert _lang("query", LOBSTER_WORKSPACE="~nosuchuser/x") == "en"


BOM = b"\xef\xbb\xbf"  # PowerShell 5.1's Set-Content -Encoding UTF8


def test_a_project_config_saved_with_a_bom_is_read(project):
    path = project / ".lobster_workspace" / "provider_config.json"
    path.write_bytes(BOM + json.dumps({"lang": "zh"}).encode())
    assert _lang("query", "-w", str(project)) == "zh"


def test_a_user_wide_config_saved_with_a_bom_is_read(tmp_path, monkeypatch):
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "xdg" / "lobster").mkdir(parents=True)
    path = tmp_path / "xdg" / "lobster" / "providers.json"
    path.write_bytes(BOM + json.dumps({"lang": "zh"}).encode())
    assert resolve_lang(["lobster"], {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}) == "zh"
