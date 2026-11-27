"""An MCP run inside a project reads the keys that project's ``lobster init``
wrote: ``<project>/.env``, then ``<project>/.lobster_workspace/.env``, which
wins a conflict, and from them only the names Lobster writes. Never the .env of
the folder the server was started in, never a folder above it."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_stdio
from lobster.config import global_config
from lobster.config.settings import settings
from lobster.core import projects
from lobster.core.config_resolver import ConfigResolver

pytestmark = pytest.mark.no_auto_config

KEY = "OLLAMA_API_KEY"
OTHER = "OTHER_TOOL_TOKEN"
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in (KEY, OTHER, "PYTHON_DOTENV_DISABLED"):
        monkeypatch.setenv(name, "")  # recorded, so teardown removes what a load sets
        monkeypatch.delenv(name)
    monkeypatch.setattr(
        global_config,
        "get_global_credentials_path",
        lambda: tmp_path / "home" / "credentials.env",
    )
    monkeypatch.setattr("lobster.config.settings.find_dotenv", lambda: "")
    root = tmp_path / "study"
    root.mkdir()
    made, _ = projects.init_project(root, name="study")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    yield made
    monkeypatch.chdir(tmp_path)  # a folder without a .env
    settings.reload_credentials(tmp_path / "nowhere")  # undo the project's keys


def env_of_run(project, tmp_path, monkeypatch):
    """The environment when the run's engine resolves its provider."""
    seen = {}

    def stop(*args, **kwargs):
        seen.update(os.environ)
        raise RuntimeError("stop after the keys were loaded")

    monkeypatch.setattr(ConfigResolver, "get_instance", stop)
    with pytest.raises(RuntimeError, match="stop after"):
        mcp_stdio._build_client(tmp_path / "session", project.workspace, None)
    return seen


def test_the_workspace_env_wins_over_the_project_env(project, tmp_path, monkeypatch):
    (project.root / ".env").write_text(f"{KEY}=from-project\n")
    (project.workspace / ".env").write_text(f"{KEY}=from-workspace\n")
    assert env_of_run(project, tmp_path, monkeypatch)[KEY] == "from-workspace"


def test_a_name_lobster_does_not_write_stays_out_of_the_run(
    project, tmp_path, monkeypatch
):
    (project.root / ".env").write_text(f"{KEY}=from-project\n{OTHER}=x\n")
    seen = env_of_run(project, tmp_path, monkeypatch)
    assert seen[KEY] == "from-project"
    assert OTHER not in seen


def test_the_env_of_the_server_folder_is_not_read(project, tmp_path, monkeypatch):
    (Path.cwd() / ".env").write_text(f"{KEY}=from-server-folder\n")
    assert KEY not in env_of_run(project, tmp_path, monkeypatch)


def test_the_global_file_fills_a_gap(project, tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "credentials.env").write_text(f"{KEY}=from-global\n")
    assert env_of_run(project, tmp_path, monkeypatch)[KEY] == "from-global"


def test_a_folder_that_is_no_longer_a_project_gives_no_keys(project, tmp_path):
    (project.root / ".env").write_text(f"{KEY}=from-project\n")
    (project.workspace / ".env").write_text(f"{KEY}=from-workspace\n")
    (Path.cwd() / ".env").write_text(f"{KEY}=from-server-folder\n")
    (project.workspace / "project.json").unlink()
    settings.reload_credentials(project.workspace, lobster_names_only=True)
    assert KEY not in os.environ


def test_python_dotenv_disabled_is_obeyed(project, tmp_path, monkeypatch):
    (project.root / ".env").write_text(f"{KEY}=from-project\n")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    assert KEY not in env_of_run(project, tmp_path, monkeypatch)


def test_lobster_chat_still_reads_every_name_of_the_project(project):
    (project.root / ".env").write_text(f"{OTHER}=x\n")
    settings.reload_credentials(project.workspace)
    assert os.environ.get(OTHER) == "x"


def test_every_name_lobster_init_writes_is_read():
    from lobster.config.settings import _lobster_written
    from lobster.ui.bridge.init_adapter import _PROVIDER_ENV_KEYS

    names = {n for keys in _PROVIDER_ENV_KEYS.values() for n in keys}
    setup = (ROOT / "lobster/config/provider_setup.py").read_text()
    names |= set(re.findall(r'"([A-Z][A-Z0-9_]+)"\s*[:\]]', setup))
    for source in (
        "lobster/cli_internal/commands/heavy/init_commands.py",
        "lobster/ui/bridge/init_adapter.py",
    ):
        text = (ROOT / source).read_text()
        names |= set(re.findall(r'append\(f?"([A-Z][A-Z0-9_]+)=', text))
    assert len(names) >= 20, names  # the scan found what init writes
    assert sorted(n for n in names if not _lobster_written(n)) == []
