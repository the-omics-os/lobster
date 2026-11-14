"""One long-lived process serves several projects: a key of one must never reach another."""

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
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    settings.reload_credentials(tmp_path / "nowhere")


def _project(root, name, env_text):
    ws = root / name / ".lobster_workspace"
    ws.mkdir(parents=True)
    if env_text is not None:
        (ws / ".env").write_text(env_text)
    return ws


def test_a_key_of_project_a_does_not_reach_project_b(clean):
    a = _project(clean, "a", f"{KEY}=sk-ant-a-0001\n")
    b = _project(clean, "b", None)

    settings.reload_credentials(a)
    assert os.environ[KEY] == "sk-ant-a-0001"
    settings.reload_credentials(b)
    assert KEY not in os.environ
    assert settings.ANTHROPIC_API_KEY == ""

    settings.reload_credentials(a)
    assert os.environ[KEY] == "sk-ant-a-0001"


def test_a_shell_key_comes_back_when_the_project_that_overrode_it_is_left(
    clean, monkeypatch
):
    monkeypatch.setenv(KEY, "sk-ant-shell-0001")
    a = _project(clean, "a", f"{KEY}=sk-ant-a-0001\n")
    b = _project(clean, "b", None)

    settings.reload_credentials(a)
    assert os.environ[KEY] == "sk-ant-a-0001"
    settings.reload_credentials(b)
    assert os.environ[KEY] == "sk-ant-shell-0001"


def test_project_b_falls_to_the_global_file_not_to_project_a(clean):
    global_dir = clean / "global"
    global_dir.mkdir()
    (global_dir / "credentials.env").write_text(f"{KEY}=sk-ant-global-0001\n")
    a = _project(clean, "a", f"{KEY}=sk-ant-a-0001\n")
    b = _project(clean, "b", None)

    settings.reload_credentials(a)
    settings.reload_credentials(b)
    assert os.environ[KEY] == "sk-ant-global-0001"


def test_every_key_of_a_project_file_is_removed_not_only_the_provider_key(clean):
    a = _project(
        clean,
        "a",
        f"{KEY}=sk-ant-a-0001\nNCBI_API_KEY=ncbi-a\nLOBSTER_LLM_PROVIDER=anthropic\n",
    )
    b = _project(clean, "b", "OPENAI_API_KEY=sk-b\n")

    settings.reload_credentials(a)
    settings.reload_credentials(b)
    assert KEY not in os.environ and "NCBI_API_KEY" not in os.environ
    assert "LOBSTER_LLM_PROVIDER" not in os.environ
    assert os.environ["OPENAI_API_KEY"] == "sk-b"
    assert settings.get_ncbi_key_count() == 0


def test_a_key_removed_from_a_project_file_does_not_linger_in_the_same_project(clean):
    a = _project(clean, "a", f"{KEY}=sk-ant-a-0001\n")
    settings.reload_credentials(a)
    (a / ".env").write_text("OPENAI_API_KEY=sk-a\n")
    settings.reload_credentials(a)
    assert KEY not in os.environ and os.environ["OPENAI_API_KEY"] == "sk-a"
