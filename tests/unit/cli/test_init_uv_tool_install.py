"""`lobster init --non-interactive` under `uv tool` installs what it was asked for.

It used to print a `uv tool command:` line and install nothing, so an agent
running init got a configured workspace without its analysis packages.
"""

import importlib.util
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from lobster.cli_internal.commands.heavy import init_commands
from lobster.config import credentials, global_config
from lobster.core import uv_tool_env

pytestmark = pytest.mark.no_auto_config
# The tool must stay on the Python it runs on; uv would pick the newest one.
PYTHON = f"{sys.version_info[0]}.{sys.version_info[1]}"


@pytest.fixture
def tool_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", tmp_path / "profiles")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "profiles/credentials.json"
    )
    monkeypatch.setattr(init_commands, "console", Console(width=2000))
    for name in ("OMICS_OS_API_KEY", "LOBSTER_LLM_PROVIDER", "LOBSTER_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)

    prefix = tmp_path / "tool-prefix"
    prefix.mkdir()
    (prefix / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{ name = "lobster-ai" }]\n'
    )
    monkeypatch.setattr(uv_tool_env.sys, "prefix", str(prefix))
    monkeypatch.setattr(uv_tool_env.shutil, "which", lambda _name: "uv")
    monkeypatch.setattr("lobster.version.__version__", "1.3.0")
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "lamindb_setup" else find_spec(name, *a),
    )

    def _exit(code):
        raise SystemExit(code)

    monkeypatch.setattr("os._exit", _exit)
    uv = SimpleNamespace(installs=[], returncode=0)

    def _run(cmd, *args, **kwargs):
        if list(cmd[1:3]) == ["tool", "install"]:
            uv.installs.append(list(cmd))
            return subprocess.CompletedProcess(cmd, uv.returncode)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _run)
    return uv


def _init(*options):
    from lobster.cli import app

    return CliRunner().invoke(
        app,
        [
            "init",
            "--non-interactive",
            "--skip-ssl-test",
            "--use-omics-os",
            "--omics-os-key",
            "omk_synthetic",
            "--agents",
            "transcriptomics_expert",
            "--lamin",
            *options,
        ],
    )


def test_non_interactive_init_runs_the_uv_tool_install(tool_env):
    result = _init()

    assert result.exit_code == 0, result.output
    assert len(tool_env.installs) == 1
    command = tool_env.installs[0]
    assert "--reinstall" in command
    assert "lobster-ai[lamin]>=1.3.0" in command
    assert command[command.index("--with") + 1] == "lobster-transcriptomics>=1.3.0"
    assert "lobster chat" in result.output
    assert "Restart lobster" not in result.output


def test_skip_extras_installs_nothing(tool_env):
    result = _init("--skip-extras")

    assert result.exit_code == 0, result.output
    assert tool_env.installs == []


def test_a_failed_uv_tool_install_exits_non_zero_with_the_command(tool_env):
    tool_env.returncode = 2

    result = _init()

    assert result.exit_code == 2
    # uv already removed the tool; the printed line must rebuild all of it.
    assert "uv tool install --reinstall 'lobster-ai[lamin]>=1.3.0'" in result.output
    assert f"--with 'lobster-transcriptomics>=1.3.0' --python {PYTHON}" in result.output


def test_agents_install_under_uv_tool_adds_to_the_tool(tool_env, tmp_path):
    from lobster.cli_internal.commands.light.agent_commands import agents_app
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    workspace = tmp_path / ".lobster_workspace"
    result = CliRunner().invoke(
        agents_app, ["install", "lobster-research", "--workspace", str(workspace)]
    )

    assert result.exit_code == 0, result.output
    assert tool_env.installs == [
        [
            "uv",
            "tool",
            "install",
            "--reinstall",
            "lobster-ai>=1.3.0",
            "--with",
            "lobster-research>=1.3.0",
            "--python",
            PYTHON,
        ]
    ]
    enabled = WorkspaceAgentConfig.load(workspace).enabled_agents
    assert {"research_agent", "data_expert_agent"} <= set(enabled)


def test_every_uv_tool_install_keeps_the_running_python(tool_env):
    _init()

    assert tool_env.installs[0][-2:] == ["--python", PYTHON]


@pytest.mark.parametrize("skip_extras", [False, True])
def test_lamin_in_a_venv_installs_the_extra(tool_env, monkeypatch, skip_extras):
    from lobster.cli_internal.commands.light import agent_commands, lamin_commands

    monkeypatch.setattr(uv_tool_env, "is_uv_tool_env", lambda: False)
    monkeypatch.setattr(uv_tool_env, "detect_uv_tool_env", lambda: None)
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: False)
    installed = []
    monkeypatch.setattr(
        agent_commands,
        "_uv_pip_install",
        lambda package: installed.append(package) or (True, ""),
    )

    result = _init(*(["--skip-extras"] if skip_extras else []))

    assert result.exit_code == 0, result.output
    assert ("lobster-ai[lamin]" in installed) is not skip_extras


def test_the_project_is_made_before_the_tool_install(tool_env, tmp_path):
    # The install ends the process, so the project must exist before it runs.
    result = _init()

    assert result.exit_code == 0, result.output
    assert len(tool_env.installs) == 1
    assert (tmp_path / ".lobster_workspace" / "project.json").exists()


def test_printed_install_hints_are_quoted_and_keep_the_tool(tool_env, tmp_path):
    from lobster.core.component_registry import get_install_command

    (tmp_path / "tool-prefix" / "uv-receipt.toml").write_text(
        "[tool]\nrequirements = [\n"
        '  { name = "lobster-ai", extras = ["anthropic"] },\n'
        '  { name = "lobster-research" },\n]\n'
    )
    hints = [
        get_install_command("lamin", is_extra=True),
        get_install_command("lobster-transcriptomics"),
    ]

    for hint in hints:
        # zsh: an unquoted "[" is a glob ("no matches found").
        assert not re.search(r"(?:^|\s)[^'\s]*\[", hint), hint
        assert "anthropic" in hint and "--with lobster-research" in hint, hint
