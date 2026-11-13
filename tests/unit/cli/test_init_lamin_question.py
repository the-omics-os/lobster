"""`lobster init` and the LaminDB catalog question (no LaminDB needed).

Every init path writes the answer where the catalog reads it: the workspace .env
(``LOBSTER_LAMIN``) or, with --global, ``lamin_enabled`` in providers.json.
"""

import inspect
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import typer
from dotenv import dotenv_values
from rich.console import Console

from lobster.cli_internal.commands.heavy import init_commands
from lobster.cli_internal.commands.heavy.init_commands import init_impl
from lobster.cli_internal.commands.light import lamin_commands
from lobster.cli_internal.commands.light.lamin_commands import (
    apply_init_choice,
    init_default,
    init_notes,
    init_question,
)
from lobster.config import global_config
from lobster.config.global_config import GlobalProviderConfig
from lobster.core.catalog.factory import resolve_catalog_config

pytestmark = pytest.mark.unit


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """An isolated global config folder; no catalog setting in the shell."""
    for name in ("LOBSTER_LAMIN", "LOBSTER_LAMIN_INSTANCE", "LAMIN_CURRENT_INSTANCE"):
        monkeypatch.delenv(name, raising=False)
    folder = tmp_path / "config"
    monkeypatch.setattr(global_config, "CONFIG_DIR", folder)
    return folder


@pytest.fixture
def ws(tmp_path, cfg):
    path = tmp_path / ".lobster_workspace"
    path.mkdir()
    return path


def _flag(ws):
    env = ws / ".env"
    return dotenv_values(env).get("LOBSTER_LAMIN") if env.is_file() else None


def _enabled(ws):
    return resolve_catalog_config(ws).enabled


# --- the answer, saved -----------------------------------------------------


def test_yes_switches_the_workspace_on_in_a_private_env(ws):
    apply_init_choice(True, ws, previous={})
    assert _flag(ws) == "1" and _enabled(ws)
    assert (ws / ".env").stat().st_mode & 0o777 == 0o600


def test_no_writes_nothing_when_nothing_would_switch_it_on(ws):
    apply_init_choice(False, ws, previous={})
    assert not (ws / ".env").exists() and not _enabled(ws)


def test_no_switches_off_a_workspace_that_was_on(ws):
    (ws / ".env").write_text("OTHER=x\nLOBSTER_LAMIN=1\n")
    apply_init_choice(False, ws, previous={"LOBSTER_LAMIN": "1"})
    assert _flag(ws) == "0" and not _enabled(ws)
    assert dotenv_values(ws / ".env")["OTHER"] == "x"


def test_no_overrides_a_global_yes(ws, cfg):
    GlobalProviderConfig(lamin_enabled=True).save()
    assert _enabled(ws)
    apply_init_choice(False, ws, previous={})
    assert _flag(ws) == "0" and not _enabled(ws)


def test_no_answer_puts_back_a_setting_init_replaced(ws):
    # The workspace is the folder whose .env init rewrites.
    (ws / ".env").write_text("ANTHROPIC_API_KEY=k\n")
    apply_init_choice(None, ws, previous={"LOBSTER_LAMIN": "1"})
    assert _flag(ws) == "1"
    assert dotenv_values(ws / ".env")["ANTHROPIC_API_KEY"] == "k"
    apply_init_choice(None, ws, previous={})
    assert _flag(ws) == "1"


def test_an_answer_keeps_the_instance_init_replaced(ws):
    previous = {"LOBSTER_LAMIN": "1", "LOBSTER_LAMIN_INSTANCE": "me/lab"}
    (ws / ".env").write_text("ANTHROPIC_API_KEY=k\n")
    apply_init_choice(False, ws, previous=previous)
    values = dotenv_values(ws / ".env")
    assert values["LOBSTER_LAMIN"] == "0"
    assert values["LOBSTER_LAMIN_INSTANCE"] == "me/lab"
    assert values["ANTHROPIC_API_KEY"] == "k"


def test_the_question_starts_at_the_saved_answer_not_the_shell(ws, monkeypatch):
    assert init_default(ws, global_config=False) is False
    (ws / ".env").write_text("LOBSTER_LAMIN=1\n")
    monkeypatch.setenv("LOBSTER_LAMIN", "0")
    assert init_default(ws, global_config=False) is True
    assert init_default(None, global_config=True) is False
    GlobalProviderConfig(lamin_enabled=True).save()
    assert init_default(None, global_config=True) is True


def test_question_text_is_factual():
    title, text = init_question()
    assert title == "LaminDB catalog"
    assert "contacts no server" in text and "lamin.ai" in text
    for claim in ("compliant", "audit", "provenance", "GxP", "tamper"):
        assert claim not in text


def test_notes_name_the_install_when_lamindb_is_missing(monkeypatch):
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: False)
    notes = init_notes(True, global_config=False)
    assert notes[0].startswith("LaminDB catalog: on.")
    assert "lobster-ai[lamin]" in notes[1]
    assert init_notes(None, global_config=False) == []
    assert init_notes(False, global_config=True) == ["LaminDB catalog: off."]


# --- N6: the global setting is a model field --------------------------------


def test_global_lamin_enabled_survives_load_and_save(cfg):
    GlobalProviderConfig(default_provider="ollama", lamin_enabled=True).save()
    config = GlobalProviderConfig.load()
    assert config.lamin_enabled is True
    config.default_profile = "development"
    config.save()
    assert json.loads((cfg / "providers.json").read_text())["lamin_enabled"] is True


@pytest.mark.parametrize(
    "raw,parsed", [("yes", True), ("off", False), ("maybe", None), (1, None)]
)
def test_an_unreadable_global_value_does_not_cost_the_file(cfg, raw, parsed):
    cfg.mkdir()
    (cfg / "providers.json").write_text(
        json.dumps({"default_provider": "ollama", "lamin_enabled": raw})
    )
    config = GlobalProviderConfig.load()
    assert config.default_provider == "ollama" and config.lamin_enabled is parsed


def test_global_init_keeps_a_setting_it_did_not_ask_about(cfg):
    GlobalProviderConfig(lamin_enabled=True).save()
    init_commands._create_global_config({"provider": "ollama"})
    assert GlobalProviderConfig.load().lamin_enabled is True
    init_commands._create_global_config({"provider": "ollama", "lamin_enabled": False})
    assert GlobalProviderConfig.load().lamin_enabled is False


# --- non-interactive init ----------------------------------------------------


def _init_kwargs(**overrides):
    kwargs = {}
    for name, param in inspect.signature(init_impl).parameters.items():
        default = param.default
        if isinstance(default, typer.models.OptionInfo):
            default = default.default
        kwargs[name] = None if default is inspect.Parameter.empty else default
    kwargs.update(
        non_interactive=True,
        force=True,
        skip_extras=True,
        skip_ssl_test=True,
        nebius_key="test-nebius-key",
    )
    kwargs.update(overrides)
    return kwargs


def _run_init(**overrides):
    with pytest.raises(typer.Exit) as exc_info:
        init_impl(**_init_kwargs(**overrides))
    assert exc_info.value.exit_code == 0
    for backup in Path.cwd().glob(".env.backup.*"):  # named by the second
        backup.unlink()


@pytest.fixture
def project(tmp_path, cfg, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_commands, "console", Console(file=io.StringIO()))
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: True)
    return tmp_path


@pytest.mark.parametrize("separate_workspace", [True, False])
def test_non_interactive_flags_write_and_keep_the_setting(
    project, monkeypatch, separate_workspace
):
    ws = project / ".lobster_workspace" if separate_workspace else project
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(ws))
    _run_init(lamin=True)
    assert _flag(ws) == "1" and _enabled(ws)
    assert "NEBIUS_API_KEY=test-nebius-key" in (project / ".env").read_text()
    _run_init()  # a later init without the flag keeps the answer
    assert _flag(ws) == "1"
    assert "NEBIUS_API_KEY=test-nebius-key" in (project / ".env").read_text()
    _run_init(lamin=False)
    assert _flag(ws) == "0" and not _enabled(ws)


def test_non_interactive_global_flag_and_round_trip(project):
    _run_init(global_config=True, lamin=True)
    assert GlobalProviderConfig.load().lamin_enabled is True
    _run_init(global_config=True)  # N6: re-init without the flag keeps it
    assert GlobalProviderConfig.load().lamin_enabled is True
    _run_init(global_config=True, lamin=False)
    assert GlobalProviderConfig.load().lamin_enabled is False


# --- Go wizard and questionary results ---------------------------------------


@pytest.mark.parametrize(
    "answer,expected",
    [(True, "1"), (False, None), (None, None), ("1", None), (1, None)],
)
def test_wizard_result_reaches_the_workspace(project, answer, expected):
    from lobster.ui.bridge.init_adapter import apply_tui_init_result

    ws = project / ".lobster_workspace"
    result = {"provider": "ollama", "lamin_enabled": answer}
    apply_tui_init_result(result, ws, project / ".env")
    assert _flag(ws) == expected


def test_wizard_result_reaches_the_global_config(project):
    from lobster.ui.bridge.init_adapter import apply_tui_init_result

    ws = project / ".lobster_workspace"
    apply_tui_init_result(
        {"provider": "ollama", "lamin_enabled": True}, ws, None, global_config=True
    )
    assert GlobalProviderConfig.load().lamin_enabled is True
    apply_tui_init_result({"provider": "ollama"}, ws, None, global_config=True)
    assert GlobalProviderConfig.load().lamin_enabled is True


@pytest.mark.parametrize("default", [None, False, True])
def test_go_wizard_is_asked_with_the_saved_answer(tmp_path, monkeypatch, default):
    import lobster.ui.bridge.go_tui_bridge as bridge

    monkeypatch.setenv("LOBSTER_INIT_LAMIN", "1")  # a stale value must not leak
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs["env"])
        result_file = cmd[cmd.index("--result-file") + 1]
        Path(result_file).write_text('{"provider": "ollama"}', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    bridge.run_init_wizard("lobster-tui", lamin_default=default)
    expected = None if default is None else ("1" if default else "0")
    assert seen.get("LOBSTER_INIT_LAMIN") == expected


@pytest.mark.parametrize("flag", [None, True, False])
def test_init_go_path_passes_the_default_and_the_flag_wins(project, monkeypatch, flag):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(project / ".lobster_workspace"))
    seen = {}

    def fake_wizard(binary_path, *, lamin_default=None, **_):
        seen["default"] = lamin_default
        return {"provider": "ollama", "agents": [], "lamin_enabled": True}

    def fake_apply(result, workspace_path, env_path, global_config=False):
        seen["answer"] = result.get("lamin_enabled")

    monkeypatch.setattr(
        "lobster.ui.bridge.binary_finder.find_tui_binary", lambda: "/fake/lobster-tui"
    )
    monkeypatch.setattr("lobster.ui.bridge.go_tui_bridge.run_init_wizard", fake_wizard)
    monkeypatch.setattr(
        "lobster.ui.bridge.init_adapter.apply_tui_init_result", fake_apply
    )
    _run_init(non_interactive=False, ui_mode="go", lamin=flag, nebius_key=None)
    assert seen["default"] == (False if flag is None else None)
    assert seen["answer"] is (True if flag is None else flag)


@pytest.mark.parametrize("answered", [True, False])
def test_an_older_go_wizard_that_did_not_ask_says_so(project, monkeypatch, answered):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(project / ".lobster_workspace"))
    result = {"provider": "ollama", "agents": []}
    if answered:
        result["lamin_enabled"] = False
    monkeypatch.setattr(
        "lobster.ui.bridge.binary_finder.find_tui_binary", lambda: "/fake/lobster-tui"
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.go_tui_bridge.run_init_wizard",
        lambda binary_path, **_: dict(result),
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.init_adapter.apply_tui_init_result", lambda *a, **k: None
    )
    _run_init(non_interactive=False, ui_mode="go", nebius_key=None)
    out = init_commands.console.file.getvalue()
    assert ("did not ask about the LaminDB catalog" in out) is not answered


def _questionary(monkeypatch, lamin_answer):
    questionary = ModuleType("questionary")
    questionary.Choice = lambda title, **kwargs: SimpleNamespace(title=title, **kwargs)
    questionary.select = lambda *a, **k: SimpleNamespace(ask=lambda: "ollama")
    questionary.checkbox = lambda *a, **k: SimpleNamespace(ask=lambda: [])
    questionary.text = lambda *a, **k: SimpleNamespace(ask=lambda: "")
    questionary.password = lambda *a, **k: SimpleNamespace(ask=lambda: "")
    asked = []

    def confirm(prompt, *, default):
        asked.append((prompt, default))
        answer = lamin_answer if "LaminDB" in prompt else False
        return SimpleNamespace(ask=lambda: answer)

    questionary.confirm = confirm
    monkeypatch.setitem(sys.modules, "questionary", questionary)
    return asked


@pytest.mark.parametrize("default", [None, False, True])
def test_questionary_asks_only_when_given_a_default(monkeypatch, default):
    from lobster.ui.bridge.questionary_fallback import run_questionary_init
    from lobster.ui.wizard.manifest import ProviderDef, WizardManifest

    asked = _questionary(monkeypatch, lamin_answer=True)
    manifest = WizardManifest(
        providers=[ProviderDef(name="ollama", display_name="Ollama")]
    )
    result = run_questionary_init(manifest, lamin_default=default)
    lamin = [(p, d) for p, d in asked if "LaminDB" in p]
    if default is None:
        assert lamin == [] and result["lamin_enabled"] is None
    else:
        assert lamin == [("Keep a LaminDB catalog?", default)]
        assert result["lamin_enabled"] is True


def test_classic_question_starts_at_the_saved_answer(ws, monkeypatch):
    monkeypatch.setattr(init_commands, "console", Console(file=io.StringIO()))
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(ws))
    asked = []

    def ask(prompt, default):
        asked.append((prompt, default))
        return default

    monkeypatch.setattr(init_commands.Confirm, "ask", ask)
    assert init_commands._ask_lamin(False) is False
    (ws / ".env").write_text("LOBSTER_LAMIN=1\n")
    assert init_commands._ask_lamin(False) is True
    assert asked == [
        ("Keep a LaminDB catalog?", False),
        ("Keep a LaminDB catalog?", True),
    ]


def test_uv_tool_command_adds_the_lamin_extra_only_when_missing(monkeypatch):
    import importlib.util

    monkeypatch.setattr(
        "lobster.core.uv_tool_env.detect_uv_tool_env", lambda: None, raising=False
    )
    real = importlib.util.find_spec
    for installed in (False, True):
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: (
                (real("json") if installed else None)
                if name == "lamindb_setup"
                else real(name, *a)
            ),
        )
        cmd = init_commands._build_uv_tool_init_command(None, None, include_lamin=True)
        assert any("lamin" in part for part in cmd) is (not installed), cmd


# --- the command line ----------------------------------------------------------


def test_init_has_lamin_flags_and_passes_every_option():
    from lobster.cli import app, init

    command = typer.main.get_command(app).commands["init"]
    flags = {
        opt for param in command.params for opt in param.opts + param.secondary_opts
    }
    assert {"--lamin", "--no-lamin"} <= flags
    assert list(inspect.signature(init).parameters) == list(
        inspect.signature(init_impl).parameters
    )
    # init passes dict(locals()) on: it must bind no local before that line.
    source = inspect.getsource(init)
    body = source[source.index('"""', source.index('"""') + 3) + 3 :]
    assert body.split("options = dict(locals())")[0].strip() == ""
