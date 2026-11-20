"""`lobster agents` in a project whose config.toml holds a preset, not a list.

config.toml keeps either ``preset`` or ``enabled``, never both; an install or
uninstall there turns the preset into the explicit list it changes.
"""

import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from lobster.cli_internal.commands.light import agent_commands
from lobster.config.agent_presets import expand_preset
from lobster.config.workspace_agent_config import WorkspaceAgentConfig
from lobster.core.component_registry import component_registry

pytestmark = pytest.mark.unit

PACKAGES = {
    "research_agent": "lobster-research",
    "data_expert_agent": "lobster-research",
    "transcriptomics_expert": "lobster-transcriptomics",
    "annotation_expert": "lobster-transcriptomics",
    "de_analysis_expert": "lobster-transcriptomics",
    "visualization_expert_agent": "lobster-visualization",
    "genomics_expert": "lobster-genomics",
    "variant_analysis_expert": "lobster-genomics",
}
CHILDREN = {"transcriptomics_expert": ["annotation_expert", "de_analysis_expert"]}
BASIC = expand_preset("scrna-basic")
GENOMICS = {"genomics_expert", "variant_analysis_expert"}
REPLACED = "Preset 'scrna-basic' will be saved as explicit agents instead."


@pytest.fixture
def ws(tmp_path, monkeypatch):
    agents = {
        name: SimpleNamespace(
            name=name,
            tier_requirement="free",
            package_name=pkg,
            child_agents=CHILDREN.get(name, []),
        )
        for name, pkg in PACKAGES.items()
    }
    monkeypatch.setattr(component_registry, "list_agents", lambda: dict(agents))
    monkeypatch.setattr(component_registry, "reset", lambda: None)
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)
    monkeypatch.setattr(
        agent_commands, "console", Console(file=io.StringIO(), width=200)
    )
    WorkspaceAgentConfig(preset="scrna-basic").save(tmp_path)
    return tmp_path


def _out():
    return agent_commands.console.file.getvalue()


def test_list_ticks_the_agents_the_preset_enables(ws):
    agent_commands.agents_list(workspace=ws)

    rows = {
        line.split()[1]: line
        for line in _out().splitlines()
        if line.startswith("│") and line.split()[1] in PACKAGES
    }
    assert {name for name, row in rows.items() if " v " in row} == set(BASIC)


def test_install_adds_to_the_preset_as_an_explicit_list(ws, monkeypatch):
    monkeypatch.setattr(agent_commands, "_uv_pip_install", lambda pkg: (True, ""))

    agent_commands.agents_install("lobster-genomics", workspace=ws)

    config = WorkspaceAgentConfig.load(ws)
    assert config.preset is None
    assert set(config.enabled_agents) == set(BASIC) | GENOMICS
    assert REPLACED in _out()
    # the session's experts include the new ones
    from lobster.agents.graph import _resolve_worker_agents

    workers, _, _ = _resolve_worker_agents(None, config, lambda *a: True, "free")
    assert GENOMICS <= set(workers)


def test_tool_install_adds_to_the_preset_as_an_explicit_list(ws, monkeypatch):
    from lobster.core import uv_tool_env

    def ran(cmd, *a, **k):
        raise SystemExit(0)

    monkeypatch.setattr(uv_tool_env, "run_tool_install", ran)
    monkeypatch.setattr(
        uv_tool_env, "build_tool_install_command", lambda **k: ["uv", "tool", "x"]
    )
    monkeypatch.setattr(uv_tool_env, "windows_locks_running_tool", lambda: True)

    with pytest.raises(SystemExit):
        agent_commands._install_into_uv_tool("lobster-genomics", ws)

    config = WorkspaceAgentConfig.load(ws)
    assert config.preset is None
    assert set(config.enabled_agents) == set(BASIC) | GENOMICS
    assert REPLACED in _out()


def test_uninstall_takes_from_the_preset_as_an_explicit_list(ws, monkeypatch):
    monkeypatch.setattr(agent_commands, "_uv_pip_uninstall", lambda pkg: (True, ""))

    agent_commands.agents_uninstall("lobster-visualization", workspace=ws)

    config = WorkspaceAgentConfig.load(ws)
    assert config.preset is None
    assert set(config.enabled_agents) == set(BASIC) - {"visualization_expert_agent"}
    assert REPLACED in _out()


def test_a_listed_project_is_changed_as_before(ws, monkeypatch):
    WorkspaceAgentConfig(enabled_agents=["research_agent"]).save(ws)
    monkeypatch.setattr(agent_commands, "_uv_pip_install", lambda pkg: (True, ""))

    agent_commands.agents_install("lobster-genomics", workspace=ws)

    config = WorkspaceAgentConfig.load(ws)
    assert set(config.enabled_agents) == {"research_agent"} | GENOMICS
    assert "explicit agents" not in _out()


def test_a_change_that_changes_nothing_keeps_the_preset(ws, monkeypatch):
    monkeypatch.setattr(agent_commands, "_uv_pip_install", lambda pkg: (True, ""))
    monkeypatch.setattr(agent_commands, "_uv_pip_uninstall", lambda pkg: (True, ""))

    agent_commands.agents_install("lobster-visualization", workspace=ws)  # in it
    agent_commands.agents_uninstall("lobster-genomics", workspace=ws)  # not in it

    config = WorkspaceAgentConfig.load(ws)
    assert (config.preset, config.enabled_agents) == ("scrna-basic", [])
    assert "explicit agents" not in _out()


def _tool_install(monkeypatch, ok):
    from lobster.core import uv_tool_env

    def ran(cmd, *a, after=None, **k):
        if after is not None:
            after(ok)
        raise SystemExit(0 if ok else 1)

    monkeypatch.setattr(uv_tool_env, "run_tool_install", ran)
    monkeypatch.setattr(
        uv_tool_env, "build_tool_install_command", lambda **k: ["uv", "tool", "x"]
    )
    monkeypatch.setattr(uv_tool_env, "windows_locks_running_tool", lambda: False)


@pytest.mark.parametrize("tool", [False, True])
def test_installing_a_preset_package_with_child_agents_keeps_the_preset(
    ws, monkeypatch, tool
):
    before = (ws / "config.toml").read_bytes()
    if tool:
        _tool_install(monkeypatch, ok=True)
        with pytest.raises(SystemExit):
            agent_commands._install_into_uv_tool("lobster-transcriptomics", ws)
    else:
        monkeypatch.setattr(agent_commands, "_uv_pip_install", lambda pkg: (True, ""))
        agent_commands.agents_install("lobster-transcriptomics", workspace=ws)

    assert (ws / "config.toml").read_bytes() == before
    assert "explicit agents" not in _out()


def test_a_failed_tool_install_leaves_the_config_alone(ws, monkeypatch):
    before = (ws / "config.toml").read_bytes()
    _tool_install(monkeypatch, ok=False)

    with pytest.raises(SystemExit):
        agent_commands._install_into_uv_tool("lobster-genomics", ws)

    assert (ws / "config.toml").read_bytes() == before
    assert sorted(p.name for p in ws.iterdir()) == ["config.toml"]


def test_a_tool_install_that_works_enables_the_package(ws, monkeypatch):
    _tool_install(monkeypatch, ok=True)

    with pytest.raises(SystemExit):
        agent_commands._install_into_uv_tool("lobster-genomics", ws)

    config = WorkspaceAgentConfig.load(ws)
    assert set(config.enabled_agents) == set(BASIC) | GENOMICS
    assert sorted(p.name for p in ws.iterdir()) == ["config.toml"]
