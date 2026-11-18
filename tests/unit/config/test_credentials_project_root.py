"""A project's credentials file is the ``.env`` beside its ``.lobster_workspace``.

``lobster init`` writes the key there. A workspace is a project when it holds a
``project.json``; any other workspace keeps today's rule (its own ``.env``, else ``./.env``).
Whatever the file, it is a W-rank file: it overrides the environment and is undone when the
next workspace loads.
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
    monkeypatch.setattr("lobster.config.settings.find_dotenv", lambda: "")
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    monkeypatch.chdir(foreign)
    yield tmp_path
    settings.reload_credentials(tmp_path / "nowhere")


def make(root, name, *, project=True, root_env=None, ws_env=None):
    ws = root / name / ".lobster_workspace"
    ws.mkdir(parents=True)
    if project:
        (ws / "project.json").write_text("{}")
    if root_env is not None:
        (root / name / ".env").write_text(root_env)
    if ws_env is not None:
        (ws / ".env").write_text(ws_env)
    return ws


def test_the_root_env_of_a_project_is_found_from_a_foreign_cwd(clean):
    ws = make(clean, "p", root_env=f"{KEY}=sk-root\n")
    settings.reload_credentials(ws)
    assert os.environ[KEY] == "sk-root"
    assert settings.ANTHROPIC_API_KEY == "sk-root"


def test_the_root_env_of_a_project_is_found_from_a_subfolder(clean, monkeypatch):
    ws = make(clean, "p", root_env=f"{KEY}=sk-root\n")
    sub = clean / "p" / "src" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    settings.reload_credentials(ws)
    assert os.environ[KEY] == "sk-root"


def test_the_workspace_env_wins_over_the_root_env(clean):
    ws = make(clean, "p", root_env=f"{KEY}=sk-root\n", ws_env=f"{KEY}=sk-ws\n")
    settings.reload_credentials(ws)
    assert os.environ[KEY] == "sk-ws"


def test_the_root_env_beats_a_cwd_env_and_a_shell_export(clean, monkeypatch):
    monkeypatch.setenv(KEY, "sk-shell")
    (clean / "elsewhere" / ".env").write_text(f"{KEY}=sk-cwd\n")
    ws = make(clean, "p", root_env=f"{KEY}=sk-root\n")
    settings.reload_credentials(ws)
    assert os.environ[KEY] == "sk-root"


def test_a_workspace_that_is_not_a_project_keeps_the_old_rule(clean):
    ws = make(clean, "legacy", project=False, root_env=f"{KEY}=sk-root\n")
    settings.reload_credentials(ws)
    assert KEY not in os.environ  # the root .env is NOT read for a legacy workspace


def test_a_symlinked_project_file_does_not_make_a_project(clean):
    ws = make(clean, "p", project=False, root_env=f"{KEY}=sk-root\n")
    target = clean / "somewhere.json"
    target.write_text("{}")
    (ws / "project.json").symlink_to(target)
    settings.reload_credentials(ws)
    assert KEY not in os.environ


def test_a_root_env_key_is_undone_when_the_next_project_loads(clean, monkeypatch):
    monkeypatch.setenv(KEY, "sk-shell")
    a = make(clean, "a", root_env=f"{KEY}=sk-a\n")
    b = make(clean, "b")
    settings.reload_credentials(a)
    assert os.environ[KEY] == "sk-a"
    settings.reload_credentials(b)
    assert os.environ[KEY] == "sk-shell"
    settings.reload_credentials(a)
    assert os.environ[KEY] == "sk-a"


def test_a_symlinked_workspace_does_not_make_a_project(clean):
    real = make(clean, "real", root_env=f"{KEY}=sk-real\n")
    (clean / "link").mkdir()
    link = clean / "link" / ".lobster_workspace"
    link.symlink_to(real)
    (clean / "link" / ".env").write_text(f"{KEY}=sk-link\n")
    settings.reload_credentials(link)
    assert KEY not in os.environ


def test_only_a_folder_named_lobster_workspace_lends_its_parent(clean):
    ws = clean / "home" / "custom_ws"
    ws.mkdir(parents=True)
    (ws / "project.json").write_text("{}")
    (clean / "home" / ".env").write_text(f"{KEY}=sk-home\n")
    settings.reload_credentials(ws)
    assert KEY not in os.environ


def test_a_relative_workspace_path_finds_the_project_root_env(clean, monkeypatch):
    ws = make(clean, "p", root_env=f"{KEY}=sk-root\n")
    monkeypatch.chdir(ws)  # `--workspace .` from inside the workspace
    from pathlib import Path

    settings.reload_credentials(Path("."))
    assert os.environ[KEY] == "sk-root"


def test_the_upward_env_does_not_fill_what_a_project_root_env_leaves_out(
    clean, monkeypatch
):
    up = clean / "up.env"
    up.write_text("LINKUP_API_KEY=from-upward\n")
    monkeypatch.setattr("lobster.config.settings.find_dotenv", lambda: str(up))
    ws = make(clean, "p", root_env=f"{KEY}=sk-root\n")
    settings.reload_credentials(ws)
    assert os.environ[KEY] == "sk-root"
    assert "LINKUP_API_KEY" not in os.environ  # the project file is the W file


def test_a_project_root_env_cannot_set_the_shell_only_keys(clean):
    ws = make(clean, "p", root_env=f"{KEY}=sk-root\nLOBSTER_LAMIN=1\n")
    settings.reload_credentials(ws)
    assert os.environ[KEY] == "sk-root" and "LOBSTER_LAMIN" not in os.environ


def test_a_directory_named_project_json_is_not_a_project(clean):
    ws = make(clean, "p", project=False, root_env=f"{KEY}=sk-root\n")
    (ws / "project.json").mkdir()
    settings.reload_credentials(ws)
    assert KEY not in os.environ


# --- a project's W file is ONLY <ws>/.env or the root .env: no cwd file, no upward search ---


def global_file(root, text):
    (root / "global").mkdir(exist_ok=True)
    (root / "global" / "credentials.env").write_text(text)


def test_a_venv_inside_project_a_does_not_hand_a_key_to_project_b(clean, monkeypatch):
    a = make(clean, "a", root_env=f"{KEY}=sk-a\nLINKUP_API_KEY=lk-a\n")
    b = make(clean, "b")
    monkeypatch.setattr(
        "lobster.config.settings.find_dotenv", lambda: str(clean / "a" / ".env")
    )
    settings.reload_credentials(a)
    settings.reload_credentials(b)
    assert KEY not in os.environ and "LINKUP_API_KEY" not in os.environ


@pytest.mark.parametrize("fallback", ["none", "shell", "global"])
def test_project_b_gets_none_shell_or_global_never_a(clean, monkeypatch, fallback):
    make(clean, "a", root_env=f"{KEY}=sk-a\n")
    b = make(clean, "b")
    monkeypatch.setattr(  # a venv inside A's folder
        "lobster.config.settings.find_dotenv", lambda: str(clean / "a" / ".env")
    )
    if fallback == "shell":
        monkeypatch.setenv(KEY, "sk-shell")
    if fallback == "global":
        global_file(clean, f"{KEY}=sk-global\n")
    settings.reload_credentials(
        b
    )  # first load of the process: A's file is only "upward"
    assert (
        os.environ.get(KEY)
        == {"none": None, "shell": "sk-shell", "global": "sk-global"}[fallback]
    )


def test_a_server_whose_cwd_is_project_a_does_not_hand_a_key_to_project_b(
    clean, monkeypatch
):
    a = make(clean, "a", root_env=f"{KEY}=sk-a\n")
    b = make(clean, "b")
    monkeypatch.chdir(clean / "a")
    settings.reload_credentials(b)
    assert KEY not in os.environ


def test_under_a_tracer_the_search_starts_in_the_cwd_and_still_hands_nothing_over(
    clean, monkeypatch
):
    import sys

    make(clean, "a", root_env=f"{KEY}=sk-a\n")
    b = make(clean, "b")
    monkeypatch.chdir(clean / "a")
    import dotenv

    monkeypatch.setattr("lobster.config.settings.find_dotenv", dotenv.find_dotenv)
    previous = sys.gettrace()
    sys.settrace(lambda *args: None)
    try:
        settings.reload_credentials(b)
    finally:
        sys.settrace(previous)
    assert KEY not in os.environ


def test_a_non_project_workspace_still_reads_the_cwd_env_and_the_upward_env(
    clean, monkeypatch
):
    up = clean / "up.env"
    up.write_text("LINKUP_API_KEY=from-upward\n")
    monkeypatch.setattr("lobster.config.settings.find_dotenv", lambda: str(up))
    (clean / "elsewhere" / ".env").write_text(f"{KEY}=sk-cwd\n")
    ws = make(clean, "legacy", project=False)
    settings.reload_credentials(ws)
    assert os.environ[KEY] == "sk-cwd" and os.environ["LINKUP_API_KEY"] == "from-upward"


def test_a_lamin_only_workspace_env_does_not_hide_the_project_key(clean):
    # `lobster init --lamin` / `lobster lamin init` write LOBSTER_LAMIN there.
    ws = make(clean, "p", root_env=f"{KEY}=sk-root\n", ws_env="LOBSTER_LAMIN=1\n")
    settings.reload_credentials(ws)
    assert os.environ[KEY] == "sk-root"
