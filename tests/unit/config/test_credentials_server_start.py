"""What the server got at its own start must not become a project's key.

``Settings()`` loads ``./.env`` and the ``.env`` found upward from the package before any
project is named. When a PROJECT workspace is then loaded, those file loads are undone
(same guard: only a variable that still holds what we set); a non-project workspace or a
no-workspace load re-reads them by the old rules. Shell exports and the global file are
shared by every project and never undone.
"""

from __future__ import annotations

import os

import pytest

from lobster.config.settings import settings

KEY = "ANTHROPIC_API_KEY"


@pytest.fixture
def clean(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("LOBSTER_") or name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("lobster.config.global_config.CONFIG_DIR", tmp_path / "global")
    upward = tmp_path / "venv_home" / ".env"
    upward.parent.mkdir()
    monkeypatch.setattr("lobster.config.settings.find_dotenv", lambda: str(upward))
    server = tmp_path / "server_cwd"
    server.mkdir()
    monkeypatch.chdir(server)
    yield tmp_path
    settings.reload_credentials(tmp_path / "nowhere")


def project(root, name, env_text=None):
    ws = root / name / ".lobster_workspace"
    ws.mkdir(parents=True)
    (ws / "project.json").write_text("{}")
    if env_text is not None:
        (root / name / ".env").write_text(env_text)
    return ws


def plain(root, name):
    ws = root / name / ".lobster_workspace"
    ws.mkdir(parents=True)
    return ws


def test_a_cwd_env_loaded_at_start_is_not_lent_to_a_project_without_a_key(clean):
    (clean / "server_cwd" / ".env").write_text(f"{KEY}=sk-cwd\n")
    settings._load_credentials()  # the import-time load
    assert os.environ[KEY] == "sk-cwd"
    settings.reload_credentials(project(clean, "f"))
    assert KEY not in os.environ and settings.ANTHROPIC_API_KEY == ""


@pytest.mark.parametrize("fallback", ["none", "shell", "global"])
def test_an_upward_env_loaded_at_start_is_not_lent_to_a_project(
    clean, monkeypatch, fallback
):
    (clean / "venv_home" / ".env").write_text(f"{KEY}=sk-up\n")
    if fallback == "shell":
        monkeypatch.setenv(KEY, "sk-shell")
    if fallback == "global":
        (clean / "global").mkdir()
        (clean / "global" / "credentials.env").write_text(f"{KEY}=sk-global\n")
    settings._load_credentials()
    settings.reload_credentials(project(clean, "f"))
    assert (
        os.environ.get(KEY)
        == {
            "none": None,
            "shell": "sk-shell",
            "global": "sk-global",
        }[fallback]
    )


def test_a_shell_value_the_start_cwd_env_replaced_comes_back_for_a_project(
    clean, monkeypatch
):
    monkeypatch.setenv(KEY, "sk-shell")
    (clean / "server_cwd" / ".env").write_text(f"{KEY}=sk-cwd\n")
    settings._load_credentials()
    assert os.environ[KEY] == "sk-cwd"
    settings.reload_credentials(project(clean, "f"))
    assert os.environ[KEY] == "sk-shell"


def test_the_start_keys_come_back_for_a_non_project_workspace(clean):
    (clean / "server_cwd" / ".env").write_text(f"{KEY}=sk-cwd\n")
    (clean / "venv_home" / ".env").write_text("LINKUP_API_KEY=lk-up\n")
    settings._load_credentials()
    settings.reload_credentials(project(clean, "f"))
    assert KEY not in os.environ and "LINKUP_API_KEY" not in os.environ
    settings.reload_credentials(plain(clean, "legacy"))
    assert os.environ[KEY] == "sk-cwd" and os.environ["LINKUP_API_KEY"] == "lk-up"
    settings._load_credentials()  # a no-workspace load as well
    assert os.environ[KEY] == "sk-cwd"


def test_a_switch_between_projects_never_shows_the_start_key_or_the_other_project(
    clean,
):
    (clean / "server_cwd" / ".env").write_text(f"{KEY}=sk-cwd\n")
    settings._load_credentials()
    a = project(clean, "a", f"{KEY}=sk-a\n")
    b = project(clean, "b")
    seen = []
    for ws in (a, b, a, b):
        settings.reload_credentials(ws)
        seen.append(os.environ.get(KEY))
    assert seen == ["sk-a", None, "sk-a", None]


def test_a_start_key_changed_since_is_left_alone(clean):
    (clean / "server_cwd" / ".env").write_text(f"{KEY}=sk-cwd\n")
    settings._load_credentials()
    os.environ[KEY] = "set-by-someone-else"
    settings.reload_credentials(project(clean, "f"))
    assert os.environ[KEY] == "set-by-someone-else"


def test_the_global_key_is_shared_and_never_undone(clean):
    (clean / "global").mkdir()
    (clean / "global" / "credentials.env").write_text("NCBI_API_KEY=ncbi-g\n")
    settings._load_credentials()
    settings.reload_credentials(project(clean, "f"))
    assert os.environ["NCBI_API_KEY"] == "ncbi-g"


def test_reading_the_start_file_again_does_not_forget_the_shell_value(
    clean, monkeypatch
):
    monkeypatch.setenv(KEY, "sk-shell")
    (clean / "server_cwd" / ".env").write_text(f"{KEY}=sk-cwd\n")
    settings._load_credentials()
    settings._load_credentials()  # reads ./.env a second time
    settings.reload_credentials(project(clean, "f"))
    assert os.environ[KEY] == "sk-shell"


def test_an_upward_fill_then_a_start_override_of_the_same_key_leaves_nothing(clean):
    (clean / "venv_home" / ".env").write_text(f"{KEY}=sk-up\n")
    settings.reload_credentials(plain(clean, "legacy"))  # no .env: fills from upward
    (clean / "server_cwd" / ".env").write_text(f"{KEY}=sk-cwd\n")
    settings._load_credentials()  # the start file overrides the upward value
    assert os.environ[KEY] == "sk-cwd"
    settings.reload_credentials(project(clean, "f"))
    assert KEY not in os.environ  # the key was absent before either file


def test_the_project_layer_is_undone_before_the_start_layer(clean, monkeypatch):
    monkeypatch.setenv(KEY, "sk-shell")
    (clean / "server_cwd" / ".env").write_text(f"{KEY}=sk-cwd\n")
    settings._load_credentials()
    legacy = plain(clean, "legacy")
    (legacy / ".env").write_text(f"{KEY}=sk-plain\n")
    settings.reload_credentials(legacy)
    assert os.environ[KEY] == "sk-plain"
    settings.reload_credentials(project(clean, "f"))
    assert os.environ[KEY] == "sk-shell"
