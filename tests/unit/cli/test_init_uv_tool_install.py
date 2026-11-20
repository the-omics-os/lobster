"""`lobster init --non-interactive` under `uv tool` installs what it was asked for.

It used to print a `uv tool command:` line and install nothing, so an agent
running init got a configured workspace without its analysis packages.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
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


def test_windows_install_later_quotes_for_powershell_and_cmd(
    tool_env, monkeypatch, capsys
):
    monkeypatch.setattr("sys.platform", "win32")

    init_commands._uv_tool_env_handoff(None, ["transcriptomics_expert"], True)

    out = capsys.readouterr().out
    # Single quotes are literal in cmd.exe; ">=" bare is a redirect in both.
    assert '"lobster-transcriptomics>=1.3.0"' in out
    assert "'" not in out


def test_windows_failed_install_line_quotes_for_powershell_and_cmd(monkeypatch):
    from lobster.cli_internal.commands.light import agent_commands

    monkeypatch.setattr(agent_commands, "FAILED_INSTALLS", ["lobster-ai[lamin]"])
    monkeypatch.setattr("shutil.which", lambda _name: "uv")
    monkeypatch.setattr("sys.platform", "win32")

    lines = init_commands._failed_install_lines()

    assert '"lobster-ai[lamin]"' in lines[-1]
    assert "'" not in lines[-1]


def test_windows_agents_install_prints_one_command_without_the_uv_path(
    tool_env, monkeypatch, tmp_path
):
    from lobster.cli_internal.commands.light.agent_commands import agents_app

    monkeypatch.setattr(uv_tool_env, "windows_locks_running_tool", lambda: True)
    spaced = r"C:\Users\First Last\.local\bin\uv.exe"
    monkeypatch.setattr(uv_tool_env.shutil, "which", lambda _name: spaced)

    result = CliRunner().invoke(
        agents_app,
        ["install", "lobster-research", "--workspace", str(tmp_path / "ws")],
    )

    assert result.exit_code == 0, result.output
    assert tool_env.installs == []
    # A quoted path first is an expression in PowerShell; one command only.
    assert "First Last" not in result.output
    assert result.output.count("tool install") == 1


# A copy of the installer's check (PS_INSTALLER, step 5: the pending install
# it runs after init); keep both in step.
INSTALLER_SPEC = re.compile(
    r"^lobster-[a-z-]+(\[[a-z0-9,-]+\])?([<>=~!]=?[0-9.]+(,[<>=~!]=?[0-9.]+)*)?$"
)


def _installer_accepts(argv):
    if argv[:4] != ["uv", "tool", "install", "--reinstall"]:
        return False
    rest = argv[4:]
    if rest[-2:] not in (["--python", "3.12"], ["--python", "3.13"]):
        return False
    specs, rest = [rest[0]], rest[1:-2]
    while rest[:1] == ["--with"]:
        specs.append(rest[1])
        rest = rest[2:]
    return not rest and all(INSTALLER_SPEC.match(s) for s in specs)


@pytest.mark.parametrize(
    "argv, accepted",
    [
        (
            ["uv", "tool", "install", "--reinstall", "lobster-ai", "--python", "3.12"],
            True,
        ),
        (
            ["uv", "tool", "install", "--reinstall", "lobster-ai[anthropic]>=1.3.0"]
            + ["--python", "3.13"],
            True,
        ),
        (
            ["uv", "tool", "install", "--reinstall", "lobster-ai"]
            + ["--with", "lobster-transcriptomics>=1.3.0", "--python", "3.12"],
            True,
        ),
        (["uv", "tool", "install", "--reinstall", "lobster-ai"], False),
        (["uv", "tool", "install", "--reinstall", "lobster-ai", "--with"], False),
        (
            ["uv", "tool", "install", "--reinstall", "lobster-ai", "--with"]
            + ["requests", "--python", "3.12"],
            False,
        ),
        (
            ["uv", "tool", "install", "--reinstall", "lobster-ai; calc"]
            + ["--python", "3.12"],
            False,
        ),
    ],
)
def test_the_installer_check_accepts_zero_or_more_with_packages(argv, accepted):
    """The check copied from the installer: no --with is valid, anything else not."""
    assert _installer_accepts(argv) is accepted


def _pending():
    import json

    path = global_config.CONFIG_DIR / "pending_install.json"
    return json.loads(path.read_text()) if path.exists() else None


@pytest.mark.parametrize("ui_mode", [None, "auto"])
def test_windows_init_leaves_the_reinstall_for_the_installer(
    tool_env, monkeypatch, capsys, ui_mode
):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(uv_tool_env, "windows_locks_running_tool", lambda: True)
    # On Windows uv is a full path, often with a space; the installer wants "uv".
    spaced = r"C:\Users\First Last\.local\bin\uv.exe"
    monkeypatch.setattr(uv_tool_env.shutil, "which", lambda _name: spaced)
    if ui_mode is None:
        assert _init().exit_code == 0
    else:
        _run_interactive(monkeypatch, capsys, ui_mode)

    pending = _pending()
    assert pending["version"] == 1
    assert pending["lobster"] == "1.3.0"
    # ISO 8601 with an explicit UTC offset, as datetime.isoformat writes it.
    assert pending["created"].endswith("+00:00")
    created = datetime.fromisoformat(pending["created"])
    now = datetime.now(timezone.utc)
    assert now - timedelta(minutes=1) < created <= now
    assert "lobster-transcriptomics>=1.3.0" in pending["argv"]
    assert pending["argv"][0] == "uv"
    assert _installer_accepts(pending["argv"]), pending["argv"]


def test_init_that_runs_the_install_itself_leaves_no_pending_install(tool_env):
    stale = global_config.CONFIG_DIR / "pending_install.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"version": 1, "argv": ["uv"]}')

    assert _init().exit_code == 0

    assert len(tool_env.installs) == 1
    assert not stale.exists()


def test_init_without_an_install_removes_an_old_pending_install(tool_env):
    stale = global_config.CONFIG_DIR / "pending_install.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"version": 1, "argv": ["uv"]}')

    assert _init("--skip-extras").exit_code == 0

    assert not stale.exists()


def test_windows_agents_install_leaves_no_pending_install(
    tool_env, monkeypatch, tmp_path
):
    from lobster.cli_internal.commands.light.agent_commands import agents_app

    monkeypatch.setattr(uv_tool_env, "windows_locks_running_tool", lambda: True)
    CliRunner().invoke(
        agents_app,
        ["install", "lobster-research", "--workspace", str(tmp_path / "ws")],
    )

    assert _pending() is None


def test_picked_packages_under_uv_tool_are_left_for_the_one_reinstall(
    tool_env, monkeypatch, capsys
):
    """Under uv tool, a direct install can only fail; the reinstall line adds them."""
    from lobster.cli_internal.commands.light import agent_commands

    direct = []
    monkeypatch.setattr(
        agent_commands, "_uv_pip_install", lambda pkg: direct.append(pkg) or (False, "")
    )
    monkeypatch.setattr(init_commands, "_get_installed_agents", lambda: {})
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **k: k["default"])

    selected, _ = init_commands._perform_agent_selection_interactive(Path("ws"))

    assert direct == []
    out = capsys.readouterr().out
    assert "failed" not in out and "Installing" not in out
    assert "already installed" not in out
    assert "research_agent" in selected and "transcriptomics_expert" in selected
    command = init_commands._build_uv_tool_init_command(None, selected)
    assert "lobster-research>=1.3.0" in command
    assert "lobster-transcriptomics>=1.3.0" in command


def test_lamin_under_uv_tool_is_in_the_one_reinstall_only(tool_env):
    """A separate lamin line would drop the picked packages when run."""
    result = _init()

    assert result.exit_code == 0, result.output
    assert "LaminDB catalog: on." in result.output
    assert "LaminDB is not installed yet" not in result.output
    [command] = tool_env.installs
    assert "lobster-ai[lamin]>=1.3.0" in command
    assert "lobster-transcriptomics>=1.3.0" in command


def _install_later(out):
    return [line for line in out.splitlines() if "Install later:" in line]


def test_skip_extras_under_uv_tool_prints_one_install_line(tool_env):
    """Two separate commands each drop what the other adds."""
    result = _init("--skip-extras")

    assert result.exit_code == 0, result.output
    assert tool_env.installs == []
    assert "LaminDB is not installed yet" not in result.output
    [line] = _install_later(result.output)
    assert "uv tool install --reinstall 'lobster-ai[lamin]>=1.3.0'" in line
    assert "--with 'lobster-transcriptomics>=1.3.0'" in line


@pytest.mark.parametrize("ui_mode", ["go", "auto", "classic"])
def test_interactive_skip_extras_under_uv_tool_prints_one_install_line(
    tool_env, monkeypatch, capsys, ui_mode
):
    out = _run_interactive(monkeypatch, capsys, ui_mode, skip_extras=True, lamin=True)

    assert tool_env.installs == []
    assert "LaminDB is not installed yet" not in out
    [line] = _install_later(out)
    assert "uv tool install --reinstall 'lobster-ai[lamin]>=1.3.0'" in line
    assert "--with 'lobster-transcriptomics>=1.3.0'" in line


@pytest.mark.parametrize("ui_mode", [None, "go", "auto", "classic"])
def test_skip_extras_under_uv_tool_says_the_next_step_after_install_later(
    tool_env, monkeypatch, capsys, ui_mode
):
    if ui_mode is None:
        out = _init("--skip-extras").output
    else:
        out = _run_interactive(
            monkeypatch, capsys, ui_mode, skip_extras=True, lamin=True
        )

    lines = out.splitlines()
    [install] = [i for i, line in enumerate(lines) if "Install later:" in line]
    nexts = [i for i, line in enumerate(lines) if "lobster chat" in line]
    assert len(nexts) == 1, out
    assert install < nexts[0], out
