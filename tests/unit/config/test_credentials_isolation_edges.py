"""Edges of project credential isolation: what is restored, what is not, and what is logged.

Companion to test_credentials_between_projects.py (the five rows there). Only a load that
overrides (a workspace ``.env``, or the ``./.env`` fallback of a reload) is undone when the
next load starts; the import-time ``./.env``, the upward ``.env`` and the global file belong
to the process, not to a project.
"""

from __future__ import annotations

import logging
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
    monkeypatch.setattr("lobster.config.settings.find_dotenv", lambda: "")
    cwd = tmp_path / "server_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    yield tmp_path
    settings.reload_credentials(tmp_path / "nowhere")


def project(root, name, env_text):
    ws = root / name / ".lobster_workspace"
    ws.mkdir(parents=True)
    if env_text is not None:
        (ws / ".env").write_text(env_text)
    return ws


def global_file(root, text):
    (root / "global").mkdir(exist_ok=True)
    (root / "global" / "credentials.env").write_text(text)


def test_the_server_cwd_env_loaded_at_import_survives_leaving_a_project(clean):
    (clean / "server_cwd" / ".env").write_text(f"{KEY}=sk-cwd\n")
    settings._load_credentials()  # the import-time load: no workspace
    a = project(clean, "a", f"{KEY}=sk-a\n")
    b = project(clean, "b", "OPENAI_API_KEY=sk-b\n")

    settings.reload_credentials(a)
    assert os.environ[KEY] == "sk-a"
    settings.reload_credentials(b)
    assert os.environ[KEY] == "sk-cwd"  # W > C: b has no such key, the cwd file does


def test_a_project_without_env_falls_back_to_the_cwd_env_not_to_the_last_project(clean):
    (clean / "server_cwd" / ".env").write_text(f"{KEY}=sk-cwd\n")
    settings._load_credentials()
    a = project(clean, "a", f"{KEY}=sk-a\n")
    b = project(clean, "b", None)

    settings.reload_credentials(a)
    settings.reload_credentials(b)
    assert os.environ[KEY] == "sk-cwd"


def test_a_value_changed_since_the_project_load_is_not_clobbered(clean):
    a = project(clean, "a", f"{KEY}=sk-a\n")
    b = project(clean, "b", None)

    settings.reload_credentials(a)
    os.environ[KEY] = "set-by-someone-else"
    settings.reload_credentials(b)
    assert os.environ[KEY] == "set-by-someone-else"


def test_the_global_file_key_a_project_overrode_comes_back(clean):
    global_file(clean, f"{KEY}=sk-global\nNCBI_API_KEY=ncbi-global\n")
    a = project(clean, "a", f"{KEY}=sk-a\n")
    b = project(clean, "b", None)

    settings.reload_credentials(a)
    assert os.environ[KEY] == "sk-a"
    assert os.environ["NCBI_API_KEY"] == "ncbi-global"  # G is shared, never undone
    settings.reload_credentials(b)
    assert os.environ[KEY] == "sk-global"
    assert os.environ["NCBI_API_KEY"] == "ncbi-global"


def test_the_upward_env_is_shared_by_every_project(clean, monkeypatch):
    upward = clean / "upward.env"
    upward.write_text("LINKUP_API_KEY=up\n")
    monkeypatch.setattr("lobster.config.settings.find_dotenv", lambda: str(upward))
    b = project(clean, "b", None)
    a = project(clean, "a", None)

    settings.reload_credentials(a)
    settings.reload_credentials(b)
    assert os.environ["LINKUP_API_KEY"] == "up"


def test_a_project_that_sets_the_value_the_shell_already_had_leaves_it_alone(
    clean, monkeypatch
):
    monkeypatch.setenv(KEY, "sk-shell")
    a = project(clean, "a", f"{KEY}=sk-shell\n")
    b = project(clean, "b", None)

    settings.reload_credentials(a)
    settings.reload_credentials(b)
    assert os.environ[KEY] == "sk-shell"


def test_a_load_never_logs_a_value(clean, caplog):
    a = project(clean, "a", f"{KEY}=sk-secret-value\n")
    b = project(clean, "b", None)
    with caplog.at_level(logging.DEBUG, logger="lobster.config.settings"):
        settings.reload_credentials(a)
        settings.reload_credentials(b)
    assert "sk-secret-value" not in caplog.text
