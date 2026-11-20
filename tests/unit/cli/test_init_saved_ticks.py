"""`lobster init` pickers start from the workspace's saved experts, not the defaults.

Without a saved expert set they start from DEFAULT_AGENT_PACKAGES as before.
"""

import inspect
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import typer
from rich.console import Console

from lobster.cli_internal.commands.heavy import init_commands
from lobster.cli_internal.commands.heavy.init_commands import init_impl
from lobster.config import global_config
from lobster.config.workspace_agent_config import WorkspaceAgentConfig

pytestmark = pytest.mark.unit

GENOMICS = ["research_agent", "data_expert_agent", "genomics_expert"]


@pytest.fixture
def ws(tmp_path, monkeypatch):
    for name in ("LOBSTER_LAMIN", "LOBSTER_INIT_PACKAGES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_commands, "console", Console(file=io.StringIO()))
    path = tmp_path / ".lobster_workspace"
    path.mkdir()
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(path))
    return path


# --- classic ------------------------------------------------------------------


@pytest.mark.parametrize(
    "saved, ticks",
    [
        (None, "1,2"),
        (WorkspaceAgentConfig(enabled_agents=GENOMICS), "1,4"),
        (WorkspaceAgentConfig(preset="scrna-basic"), "1,2,3"),
    ],
)
def test_classic_picker_starts_from_the_saved_experts(ws, monkeypatch, saved, ticks):
    if saved is not None:
        saved.save(ws)
    asked = []

    def ask(prompt, default):
        asked.append(default)
        return default

    monkeypatch.setattr(init_commands.Prompt, "ask", ask)
    monkeypatch.setattr(init_commands, "_get_installed_agents", lambda: {})
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: True)
    init_commands._prompt_manual_agent_selection(ws)
    assert asked == [ticks]


# --- questionary --------------------------------------------------------------


def _questionary(monkeypatch):
    questionary = ModuleType("questionary")
    questionary.Choice = lambda title, **kwargs: SimpleNamespace(title=title, **kwargs)
    questionary.select = lambda *a, **k: SimpleNamespace(ask=lambda: "ollama")
    shown = []

    def checkbox(prompt, **kwargs):
        shown.extend(kwargs["choices"])
        return SimpleNamespace(ask=lambda: [])

    questionary.checkbox = checkbox
    questionary.text = lambda *a, **k: SimpleNamespace(ask=lambda: "")
    questionary.password = lambda *a, **k: SimpleNamespace(ask=lambda: "")
    questionary.confirm = lambda *a, **k: SimpleNamespace(ask=lambda: False)
    monkeypatch.setitem(sys.modules, "questionary", questionary)
    return shown


@pytest.mark.parametrize(
    "packages, ticked",
    [
        (None, {"lobster-research", "lobster-transcriptomics"}),
        (
            ["lobster-research", "lobster-genomics"],
            {"lobster-research", "lobster-genomics"},
        ),
    ],
)
def test_questionary_picker_ticks_the_packages_it_is_given(
    monkeypatch, packages, ticked
):
    from lobster.ui.bridge.questionary_fallback import run_questionary_init
    from lobster.ui.wizard.manifest import build_init_manifest

    choices = _questionary(monkeypatch)
    run_questionary_init(build_init_manifest(), agent_packages=packages)
    assert {c.value for c in choices if c.checked} == ticked


# --- Go wizard (the bridge) ---------------------------------------------------


@pytest.mark.parametrize(
    "packages, sent",
    [
        (None, None),
        (["lobster-research", "lobster-genomics"], "lobster-research,lobster-genomics"),
    ],
)
def test_go_wizard_is_sent_the_saved_packages(monkeypatch, packages, sent):
    import lobster.ui.bridge.go_tui_bridge as bridge

    monkeypatch.setenv("LOBSTER_INIT_PACKAGES", "lobster-ml")  # stale: must not leak
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs["env"])
        result_file = cmd[cmd.index("--result-file") + 1]
        Path(result_file).write_text('{"provider": "ollama"}', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    bridge.run_init_wizard("lobster-tui", agent_packages=packages)
    assert seen.get("LOBSTER_INIT_PACKAGES") == sent


# --- init --force hands the saved packages to the picker ----------------------


def _init_kwargs(**overrides):
    kwargs = {}
    for name, param in inspect.signature(init_impl).parameters.items():
        default = param.default
        if isinstance(default, typer.models.OptionInfo):
            default = default.default
        kwargs[name] = None if default is inspect.Parameter.empty else default
    kwargs.update(force=True, skip_extras=True, non_interactive=False)
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("ui_mode", ["go", "auto"])
@pytest.mark.parametrize("saved", [False, True])
def test_init_force_hands_the_saved_packages_to_the_picker(
    ws, monkeypatch, ui_mode, saved
):
    (ws.parent / ".env").write_text("OLLAMA_BASE_URL=http://localhost:11434\n")
    if saved:
        WorkspaceAgentConfig(enabled_agents=GENOMICS).save(ws)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    seen = {}

    def picker(*_args, agent_packages=None, **_kwargs):
        seen["packages"] = agent_packages
        return {"provider": "ollama", "agents": [], "cancelled": False}

    monkeypatch.setattr(
        "lobster.ui.bridge.binary_finder.find_tui_binary",
        lambda: "/fake/lobster-tui" if ui_mode == "go" else None,
    )
    monkeypatch.setattr("lobster.ui.bridge.go_tui_bridge.run_init_wizard", picker)
    monkeypatch.setattr(
        "lobster.ui.bridge.questionary_fallback.run_questionary_init", picker
    )
    monkeypatch.setattr(init_commands, "_ask_language", lambda **_: None)
    monkeypatch.setattr(
        "lobster.ui.bridge.init_adapter.apply_tui_init_result", lambda *a, **k: None
    )
    with pytest.raises(typer.Exit) as exc_info:
        init_impl(**_init_kwargs(ui_mode=ui_mode))
    assert exc_info.value.exit_code == 0
    expected = ["lobster-research", "lobster-genomics"] if saved else None
    assert seen["packages"] == expected


# --- an interactive --preset starts every picker and is recorded --------------

SCRNA_FULL = [
    "lobster-research",
    "lobster-transcriptomics",
    "lobster-visualization",
    "lobster-metadata",
]
SCRNA_FULL_AGENTS = [
    "research_agent",
    "data_expert_agent",
    "transcriptomics_expert",
    "annotation_expert",
    "de_analysis_expert",
    "visualization_expert_agent",
    "metadata_assistant",
]


@pytest.mark.parametrize(
    "answer, kept", [(None, "scrna-full"), ("1,2", None)], ids=["kept", "unticked"]
)
def test_classic_picker_starts_from_the_preset(ws, monkeypatch, answer, kept):
    WorkspaceAgentConfig(enabled_agents=GENOMICS).save(ws)  # --preset wins
    asked = []

    def ask(prompt, default):
        asked.append(default)
        return answer or default

    monkeypatch.setattr(init_commands.Prompt, "ask", ask)
    monkeypatch.setattr(init_commands, "_get_installed_agents", lambda: {})
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: True)
    selected, preset = init_commands._perform_agent_selection_interactive(
        ws, skip_extras=True, preset="scrna-full"
    )
    assert asked == ["1,2,3,9"]
    assert preset == kept
    out = init_commands.console.file.getvalue()
    replaced = "Preset 'scrna-full' will be saved as explicit agents instead."
    assert (replaced in out) is (kept is None)
    if kept:
        assert sorted(selected) == sorted(SCRNA_FULL_AGENTS)


@pytest.mark.parametrize("ui_mode", ["go", "auto"])
@pytest.mark.parametrize(
    "picked, kept",
    [
        (SCRNA_FULL_AGENTS, "scrna-full"),
        (["research_agent", "data_expert_agent"], None),
    ],
    ids=["kept", "unticked"],
)
def test_init_preset_starts_the_picker_and_is_recorded(
    ws, monkeypatch, ui_mode, picked, kept
):
    (ws.parent / ".env").write_text("OLLAMA_BASE_URL=http://localhost:11434\n")
    WorkspaceAgentConfig(enabled_agents=GENOMICS).save(ws)  # --preset wins
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    seen = {}

    def picker(*_args, agent_packages=None, **_kwargs):
        seen["packages"] = agent_packages
        return {"provider": "ollama", "agents": list(picked), "cancelled": False}

    monkeypatch.setattr(
        "lobster.ui.bridge.binary_finder.find_tui_binary",
        lambda: "/fake/lobster-tui" if ui_mode == "go" else None,
    )
    monkeypatch.setattr("lobster.ui.bridge.go_tui_bridge.run_init_wizard", picker)
    monkeypatch.setattr(
        "lobster.ui.bridge.questionary_fallback.run_questionary_init", picker
    )
    monkeypatch.setattr(init_commands, "_ask_language", lambda **_: None)
    with pytest.raises(typer.Exit) as exc_info:
        init_impl(**_init_kwargs(ui_mode=ui_mode, preset="scrna-full"))
    assert exc_info.value.exit_code == 0
    assert seen["packages"] == SCRNA_FULL
    saved = WorkspaceAgentConfig.load(ws)
    assert saved.preset == kept
    if kept is None:
        assert saved.enabled_agents == picked
        out = init_commands.console.file.getvalue()
        assert "Preset 'scrna-full' will be saved as explicit agents instead." in out


@pytest.mark.parametrize("ui_mode", ["go", "classic"])
def test_interactive_init_refuses_an_unknown_preset(ws, monkeypatch, ui_mode):
    (ws.parent / ".env").write_text("OLLAMA_BASE_URL=http://localhost:11434\n")
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))

    def picker(*_args, **_kwargs):
        raise AssertionError("no picker for an unknown preset")

    monkeypatch.setattr(
        "lobster.ui.bridge.binary_finder.find_tui_binary", lambda: "/fake/lobster-tui"
    )
    monkeypatch.setattr("lobster.ui.bridge.go_tui_bridge.run_init_wizard", picker)
    monkeypatch.setattr(init_commands, "_ask_language", picker)
    monkeypatch.setattr(init_commands.Prompt, "ask", picker)
    with pytest.raises(typer.Exit) as exc_info:
        init_impl(**_init_kwargs(ui_mode=ui_mode, preset="scrna-everything"))
    assert exc_info.value.exit_code == 1
    assert "Invalid preset: scrna-everything" in init_commands.console.file.getvalue()
