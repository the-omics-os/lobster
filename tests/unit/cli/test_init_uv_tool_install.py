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
    # One next step, and only after the install.
    assert result.output.count("Next: run lobster chat") == 1
    assert result.output.index("Packages installed") < result.output.index(
        "Next: run lobster chat"
    )


def test_skip_extras_installs_nothing(tool_env):
    result = _init("--skip-extras")

    assert result.exit_code == 0, result.output
    assert tool_env.installs == []
    # It says what was skipped and how to add it, never "will be added".
    assert "--skip-extras prevented package installation" in result.output
    assert "uv tool install --reinstall" in result.output
    assert "will be added" not in result.output


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


def test_ctrl_c_at_the_install_prompt_still_leaves_a_project(
    tool_env, monkeypatch, tmp_path
):
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    (workspace / "provider_config.json").write_text("{}")

    def _interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("rich.prompt.Confirm.ask", _interrupt)

    @init_commands._creates_project
    def _wizard(folder=None, project_name=None, global_config=False):
        init_commands._uv_tool_env_handoff(
            "anthropic", ["transcriptomics_expert"], skip_extras=False
        )

    with pytest.raises(KeyboardInterrupt):
        _wizard()

    assert (workspace / "project.json").exists()
    assert tool_env.installs == []


# --- The next step comes after the uv tool install, on every init path ---

RESULT = {
    "provider": "omics-os",
    "api_key": "omk_synthetic",
    "agents": ["transcriptomics_expert"],
    "lamin_enabled": False,
    "cancelled": False,
}


def _init_kwargs(**overrides):
    import inspect

    import typer

    options = {
        name: (
            param.default.default
            if isinstance(param.default, typer.models.OptionInfo)
            else param.default
        )
        for name, param in inspect.signature(init_commands.init_impl).parameters.items()
    }
    options.update(skip_ssl_test=True, **overrides)
    return options


def _run_interactive(monkeypatch, capsys, ui_mode, **overrides):
    import typer

    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda *a, **k: False)
    answers = iter(
        ["1", "1", "1", "omk_synthetic"]
    )  # language, provider, has account, key
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **k: next(answers, "n"))
    monkeypatch.setattr(
        init_commands,
        "_perform_agent_selection_interactive",
        lambda *a, **k: (["transcriptomics_expert"], None),
    )
    monkeypatch.setattr(init_commands, "_ask_language", lambda **k: None)
    if ui_mode == "go":
        monkeypatch.setattr(
            "lobster.ui.bridge.binary_finder.find_tui_binary", lambda: "/fake/tui"
        )
        monkeypatch.setattr(
            "lobster.ui.bridge.go_tui_bridge.run_init_wizard",
            lambda *a, **k: dict(RESULT),
        )
    if ui_mode == "auto":  # no Go binary: the questionary screens
        monkeypatch.setattr(
            "lobster.ui.bridge.binary_finder.find_tui_binary", lambda: None
        )
        monkeypatch.setattr(
            "lobster.ui.bridge.questionary_fallback.run_questionary_init",
            lambda *a, **k: dict(RESULT),
        )
    try:
        init_commands.init_impl(**_init_kwargs(ui_mode=ui_mode, **overrides))
    except typer.Exit as exc:
        assert not exc.exit_code
    except SystemExit as exc:  # the faked os._exit after a tool install
        assert not exc.code
    return capsys.readouterr().out


@pytest.mark.parametrize("ui_mode", ["go", "auto", "classic"])
def test_declining_the_install_never_says_run_lobster_chat_alone(
    tool_env, monkeypatch, capsys, ui_mode
):
    out = _run_interactive(monkeypatch, capsys, ui_mode)

    assert tool_env.installs == []
    assert "Package installation needed" in out, out
    # The only next step names the install first.
    assert "Next: install them, then run lobster chat" in out
    assert "Next: run lobster chat" not in out
    assert "Next step:" not in out


def test_classic_panel_shows_a_failed_install_instead_of_next_step(
    monkeypatch, capsys, tmp_path
):
    import typer

    from lobster.cli_internal.commands.light import agent_commands

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_commands, "console", Console(width=2000))
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda *a, **k: False)
    answers = iter(["1", "1", "1", "omk_synthetic"])
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **k: next(answers, "n"))
    monkeypatch.setattr(agent_commands, "FAILED_INSTALLS", ["lobster-research"])
    monkeypatch.setattr(
        init_commands,
        "_perform_agent_selection_interactive",
        lambda *a, **k: ([], None),
    )

    try:
        init_commands.init_impl(**_init_kwargs(ui_mode="classic"))
    except typer.Exit as exc:
        assert not exc.exit_code

    out = capsys.readouterr().out
    assert "lobster-research" in out
    assert "Next step:" not in out


def test_lamin_under_uv_tool_goes_into_the_tool_not_pip(tool_env, monkeypatch):
    from lobster.cli_internal.commands.light import agent_commands

    pip = []
    monkeypatch.setattr(agent_commands, "_uv_pip_install", lambda p: pip.append(p))

    init_commands._print_lamin_notes(True, global_config=False, install=True)

    # A pip install would vanish with the next tool rebuild.
    assert pip == []


def test_questionary_ticks_only_the_default_packages(monkeypatch):
    from types import ModuleType

    from lobster.cli_internal.agent_packages import DEFAULT_AGENT_PACKAGES
    from lobster.ui.bridge.questionary_fallback import run_questionary_init
    from lobster.ui.wizard.manifest import (
        AgentPackageDef,
        ProviderDef,
        WizardManifest,
    )

    questionary = ModuleType("questionary")
    choices = []

    def _choice(title, **kwargs):
        choices.append(kwargs)
        return SimpleNamespace(title=title, **kwargs)

    questionary.Choice = _choice
    questionary.select = lambda prompt, **k: SimpleNamespace(
        ask=lambda: "yes" if "account" in prompt else "omics-os"
    )
    questionary.checkbox = lambda *a, **k: SimpleNamespace(ask=lambda: [])
    questionary.confirm = lambda *a, **k: SimpleNamespace(ask=lambda: False)
    questionary.password = lambda *a, **k: SimpleNamespace(ask=lambda: "omk_x")
    monkeypatch.setitem(sys.modules, "questionary", questionary)
    packages = ["lobster-research", "lobster-transcriptomics", "lobster-genomics"]
    manifest = WizardManifest(
        providers=[
            ProviderDef(
                name="omics-os", display_name="Omics-OS", model_selection="managed"
            )
        ],
        agent_packages=[
            AgentPackageDef(package_name=name, description=name, agents=[name])
            for name in packages
        ],
    )

    run_questionary_init(manifest)

    ticked = {
        c["value"] for c in choices if c.get("value") in packages and c["checked"]
    }
    assert ticked == set(DEFAULT_AGENT_PACKAGES)


def test_nothing_to_install_still_ends_with_the_next_step(tool_env, capsys):
    init_commands._uv_tool_env_handoff(None, [], skip_extras=False)

    assert tool_env.installs == []
    assert "Next: run lobster chat" in capsys.readouterr().out


def test_windows_prints_the_tool_install_instead_of_running_it(
    tool_env, monkeypatch, tmp_path
):
    # Windows locks the running lobster.exe and its venv: a reinstall from
    # inside it can fail halfway and leave no lobster at all.
    monkeypatch.setattr(uv_tool_env, "windows_locks_running_tool", lambda: True)
    spaced = r"C:\Users\First Last\.local\bin\uv.exe"
    monkeypatch.setattr(uv_tool_env.shutil, "which", lambda _name: spaced)

    result = _init()

    assert result.exit_code == 0, result.output
    assert tool_env.installs == []
    assert (tmp_path / ".lobster_workspace" / "project.json").exists()
    # Double quotes work in PowerShell and cmd; ">=" and "[" unquoted do not.
    assert '"lobster-ai[lamin]>=1.3.0"' in result.output
    assert f'"lobster-transcriptomics>=1.3.0" --python {PYTHON}' in result.output
    assert "Close Lobster" in result.output
    # PowerShell reads a leading quoted path as an expression: print plain uv.
    assert "Then run: uv tool install --reinstall" in result.output
    assert "First Last" not in result.output


def test_windows_interactive_init_prints_the_command_once_and_asks_nothing(
    tool_env, monkeypatch, capsys
):
    monkeypatch.setattr(uv_tool_env, "windows_locks_running_tool", lambda: True)
    # The prompt is answered "no": a Windows run must not reach it.
    out = _run_interactive(monkeypatch, capsys, "auto")  # questionary on Windows

    assert tool_env.installs == []
    assert out.count("Close Lobster") == 1
    assert "Running uv tool install" not in out
    assert "Next: install them" not in out


@pytest.mark.parametrize("options", [(), ("--skip-extras",)])
def test_from_another_folder_the_next_step_goes_there_first(
    tool_env, tmp_path, options
):
    # `lobster init ~/my-study` from ~: chat in ~ finds no provider.
    study = tmp_path / "my study"
    result = _init(*options, str(study))

    assert result.exit_code == 0, result.output
    assert f"cd '{study}' && lobster chat" in result.output
    assert "Next: run lobster chat" not in result.output


@pytest.mark.parametrize("ui_mode", ["go", "auto", "classic"])
def test_declining_from_another_folder_names_the_folder_after_the_install(
    tool_env, monkeypatch, capsys, tmp_path, ui_mode
):
    study = tmp_path / "study"
    out = _run_interactive(monkeypatch, capsys, ui_mode, folder=study)

    assert tool_env.installs == []
    before, after = out.split("Package installation needed")
    assert "Next" not in before, before  # the next step belongs to the install
    assert after.index("Next: install them") < after.index(
        f"cd {study} && lobster chat"
    )


def test_the_classic_panel_from_another_folder_names_the_folder(
    monkeypatch, capsys, tmp_path
):
    import typer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_commands, "console", Console(width=2000))
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda *a, **k: False)
    answers = iter(["1", "1", "1", "omk_synthetic"])
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **k: next(answers, "n"))
    monkeypatch.setattr(
        init_commands,
        "_perform_agent_selection_interactive",
        lambda *a, **k: ([], None),
    )
    study = tmp_path / "study"

    try:
        init_commands.init_impl(**_init_kwargs(ui_mode="classic", folder=study))
    except typer.Exit as exc:
        assert not exc.exit_code

    out = capsys.readouterr().out
    assert f"cd {study} && lobster chat" in out
    assert "Next step:" not in out


def test_on_windows_the_next_step_runs_in_powershell(monkeypatch, tmp_path):
    # Windows PowerShell 5.1 rejects && as a statement separator.
    monkeypatch.setattr(init_commands, "_init_folder", tmp_path / "study")
    monkeypatch.setattr(init_commands.sys, "platform", "win32")

    text = init_commands._next_chat_text()

    assert f"cd {tmp_path / 'study'}; lobster chat" in text
    assert "&&" not in text
