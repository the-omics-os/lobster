"""Per-agent switch for ``execute_custom_code``.

Covers the resolver order (each layer winning over the next), its exact inputs (the
process environment, the agent's own workspace config.toml and the declared default;
never a user-level config), invalid values, the ``code_tool`` workspace setting, the
helper (nothing is built while off) and the supervisor graph.
"""

import importlib
import os
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lobster.config import workspace_agent_config as wac
from lobster.config.workspace_agent_config import AgentSettings, WorkspaceAgentConfig
from lobster.tools import code_tool_switch as switch
from lobster.tools.code_tool_switch import (
    GLOBAL_ENV_VAR,
    SUPERVISOR_CODE_TOOL_DEFAULT,
    agent_env_var,
    attach_code_tool,
    code_tool_enabled,
)

AGENT = "proteomics_expert"
CODE_EXEC = {"categories": ["CODE_EXEC"], "provenance": False}


@pytest.fixture(autouse=True)
def no_switch_env(monkeypatch):
    for key in list(os.environ):
        if key == GLOBAL_ENV_VAR or key.startswith(GLOBAL_ENV_VAR + "_"):
            monkeypatch.delenv(key)


@pytest.fixture
def warnings_seen(monkeypatch):
    seen = []

    def record(msg, *args, **kwargs):
        seen.append(msg % args if args else msg)

    monkeypatch.setattr(switch.logger, "warning", record)
    monkeypatch.setattr(wac.logger, "warning", record)
    return seen


def _workspace(tmp_path, agent=AGENT, line=None, extra=""):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    if line is not None or extra:
        body = f"[agent_settings.{agent}]\n"
        if line is not None:
            body += f"code_tool = {line}\n"
        (ws / "config.toml").write_text(body + extra)
    return ws


# --------------------------------------------------------------------------- order


ORDER = [
    # (per-agent env, global env, config.toml value, declared default, expected)
    pytest.param("1", "none", "false", False, True, id="agent_env_1_beats_global_none"),
    pytest.param("0", "all", "true", True, False, id="agent_env_0_beats_global_all"),
    pytest.param(None, "none", "true", True, False, id="global_none_beats_config"),
    pytest.param(None, "all", "false", False, True, id="global_all_beats_config"),
    pytest.param(None, None, "false", True, False, id="config_false_beats_default"),
    pytest.param(None, None, "true", False, True, id="config_true_beats_default"),
    pytest.param(None, "default", "false", True, False, id="global_default_defers"),
    pytest.param(None, "default", None, True, True, id="global_default_to_default"),
    pytest.param(None, None, None, True, True, id="default_true_beats_fallback"),
    pytest.param(None, None, None, False, False, id="fallback_false"),
]


@pytest.mark.parametrize("agent_env,global_env,config,default,expected", ORDER)
def test_resolution_order(
    tmp_path, monkeypatch, agent_env, global_env, config, default, expected
):
    if agent_env is not None:
        monkeypatch.setenv(agent_env_var(AGENT), agent_env)
    if global_env is not None:
        monkeypatch.setenv(GLOBAL_ENV_VAR, global_env)
    ws = _workspace(tmp_path, line=config)
    assert code_tool_enabled(AGENT, default=default, workspace_path=ws) is expected


def test_env_var_names():
    assert GLOBAL_ENV_VAR == "LOBSTER_CODE_TOOL"
    assert agent_env_var("data_expert_agent") == "LOBSTER_CODE_TOOL_DATA_EXPERT_AGENT"


def test_unknown_agent_is_off_and_supervisor_default_is_on():
    assert code_tool_enabled("an_agent_nobody_declared") is False
    assert SUPERVISOR_CODE_TOOL_DEFAULT is True
    assert code_tool_enabled("supervisor", default=SUPERVISOR_CODE_TOOL_DEFAULT)


def test_global_mode_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv(GLOBAL_ENV_VAR, "  NONE ")
    assert code_tool_enabled(AGENT, default=True) is False


def test_empty_env_values_are_unset(tmp_path, monkeypatch, warnings_seen):
    monkeypatch.setenv(agent_env_var(AGENT), "")
    monkeypatch.setenv(GLOBAL_ENV_VAR, " ")
    assert code_tool_enabled(AGENT, default=True) is True
    assert warnings_seen == []


# ------------------------------------------------------------------ invalid values


@pytest.mark.parametrize("raw", ["yes", "true", "2", "on", "-1"])
def test_invalid_agent_env_warns_and_falls_through(monkeypatch, warnings_seen, raw):
    monkeypatch.setenv(agent_env_var(AGENT), raw)
    monkeypatch.setenv(GLOBAL_ENV_VAR, "none")
    assert code_tool_enabled(AGENT, default=True) is False  # the global layer decides
    assert len(warnings_seen) == 1 and agent_env_var(AGENT) in warnings_seen[0]


@pytest.mark.parametrize("raw", ["sometimes", "on", "1", "true"])
def test_invalid_global_env_warns_and_falls_through(
    tmp_path, monkeypatch, warnings_seen, raw
):
    monkeypatch.setenv(GLOBAL_ENV_VAR, raw)
    ws = _workspace(tmp_path, line="true")
    assert code_tool_enabled(AGENT, default=False, workspace_path=ws) is True
    assert len(warnings_seen) == 1 and GLOBAL_ENV_VAR in warnings_seen[0]


@pytest.mark.parametrize("line", ['"true"', '"false"', "1", "0", "1.0", '"yes"'])
def test_non_boolean_config_value_drops_only_that_setting(
    tmp_path, warnings_seen, line
):
    ws = _workspace(tmp_path, line=line, extra='temperature = 0.5\nmodel = "m"\n')
    for default in (True, False):
        assert code_tool_enabled(AGENT, default=default, workspace_path=ws) is default
    settings = WorkspaceAgentConfig.load(ws).agent_settings[AGENT]
    assert settings.code_tool is None
    assert settings.temperature == 0.5 and settings.model == "m"
    assert any("code_tool" in w for w in warnings_seen)


def test_non_boolean_default_warns_and_is_off(warnings_seen):
    assert code_tool_enabled(AGENT, default="yes") is False
    assert len(warnings_seen) == 1


@pytest.mark.parametrize("default", [True, False])
def test_corrupt_or_unreadable_config_falls_back(tmp_path, default):
    ws = _workspace(tmp_path)
    (ws / "config.toml").write_text("[agent_settings.proteomics_expert\ncode_tool = ")
    assert code_tool_enabled(AGENT, default=default, workspace_path=ws) is default
    (ws / "config.toml").unlink()
    (ws / "config.toml").mkdir()  # exists but cannot be read as a file
    assert code_tool_enabled(AGENT, default=default, workspace_path=ws) is default


# ---------------------------------------------------------------------- the inputs


def test_workspace_path_may_be_str_or_pathlike(tmp_path):
    ws = _workspace(tmp_path, line="true")
    assert code_tool_enabled(AGENT, workspace_path=str(ws)) is True
    assert code_tool_enabled(AGENT, workspace_path=Path(ws)) is True


@pytest.mark.parametrize("value", [None, 7, MagicMock(), object()])
def test_non_path_workspace_skips_the_config_layer(value):
    assert code_tool_enabled(AGENT, default=True, workspace_path=value) is True


def test_only_the_agents_own_workspace_config_is_read(tmp_path, monkeypatch):
    """No user-level or global config: HOME, XDG and the cwd are never consulted."""
    home = tmp_path / "home"
    off = f"[agent_settings.{AGENT}]\ncode_tool = false\n"
    for rel in (
        ".lobster/config.toml",
        ".lobster_workspace/config.toml",
        ".config/lobster/config.toml",
        "config.toml",
    ):
        target = home / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(off)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.chdir(home)
    ws = _workspace(tmp_path)  # this agent's workspace has no config.toml
    assert code_tool_enabled(AGENT, default=True, workspace_path=ws) is True
    assert code_tool_enabled(AGENT, default=True) is True


@pytest.fixture
def config_elsewhere(tmp_path):
    """A code_tool = true config in every conventional place except the workspace."""
    elsewhere = tmp_path / "elsewhere"
    on = f"[agent_settings.{AGENT}]\ncode_tool = true\n"
    for rel in (
        "config.toml",
        ".lobster_workspace/config.toml",
        ".lobster/config.toml",
        ".config/lobster/config.toml",
    ):
        target = elsewhere / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(on)
    return elsewhere


@pytest.mark.parametrize("where", ["cwd", "LOBSTER_WORKSPACE", "HOME"])
def test_config_outside_the_data_manager_workspace_has_no_effect(
    tmp_path, monkeypatch, config_elsewhere, where
):
    """The switch reads config.toml only from data_manager.workspace_path."""
    if where == "cwd":
        monkeypatch.chdir(config_elsewhere)
    elif where == "LOBSTER_WORKSPACE":
        monkeypatch.setenv("LOBSTER_WORKSPACE", str(config_elsewhere))
    else:
        monkeypatch.setenv("HOME", str(config_elsewhere))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_elsewhere / ".config"))
    dm = SimpleNamespace(workspace_path=_workspace(tmp_path))  # no config.toml here
    assert attach_code_tool([], dm, AGENT, default=False) == []
    assert attach_code_tool([], SimpleNamespace(), AGENT, default=False) == []
    assert code_tool_enabled(AGENT, default=False) is False


def test_config_in_the_data_manager_workspace_takes_effect(tmp_path, spies):
    dm = SimpleNamespace(workspace_path=_workspace(tmp_path, line="true"))
    assert len(attach_code_tool([], dm, AGENT, default=False)) == 1


@pytest.mark.parametrize("value", [None, 7, object()])
def test_no_workspace_path_never_opens_a_file(monkeypatch, value):
    def refuse(*args, **kwargs):
        raise AssertionError("no file may be opened without a workspace path")

    monkeypatch.setattr(WorkspaceAgentConfig, "load", refuse)
    monkeypatch.setattr(Path, "read_bytes", refuse)
    monkeypatch.setattr(Path, "open", refuse)
    monkeypatch.setattr("builtins.open", refuse)
    monkeypatch.setattr("io.open", refuse)
    assert attach_code_tool([], SimpleNamespace(), AGENT, default=False) == []
    dm = SimpleNamespace(workspace_path=value)
    assert attach_code_tool([], dm, AGENT, default=False) == []
    assert code_tool_enabled(AGENT, default=True, workspace_path=value) is True


# Valid TOML whose other settings fail validation; none of them is this agent's
# code_tool, so they must never change it. (text, placed before this agent's table)
UNRELATED_INVALID = [
    pytest.param(
        "[agent_settings.other_expert]\ntemperature = 3.0\n", False, id="other_agent"
    ),
    pytest.param(
        "[agent_settings.other_expert]\nmodel = 5\n", False, id="other_agent_type"
    ),
    pytest.param('enabled_agents = "not a list"\n', True, id="top_level_field"),
    pytest.param(
        '[provider]\ndefault = ["not", "a", "string"]\n', False, id="provider"
    ),
    pytest.param("temperature = 3.0\n", False, id="same_agent_sibling"),
]


def _with_unrelated(tmp_path, agent, line, unrelated, before):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    own = f"[agent_settings.{agent}]\ncode_tool = {line}\n"
    (ws / "config.toml").write_text(unrelated + own if before else own + unrelated)
    # The whole-config loader rejects this file, so it would lose the setting.
    assert agent not in WorkspaceAgentConfig.load(ws).agent_settings
    return ws


@pytest.mark.parametrize("unrelated,before", UNRELATED_INVALID)
def test_an_unrelated_invalid_setting_keeps_a_valid_disable(
    tmp_path, unrelated, before
):
    ws = _with_unrelated(tmp_path, "supervisor", "false", unrelated, before)
    assert (
        code_tool_enabled(
            "supervisor", default=SUPERVISOR_CODE_TOOL_DEFAULT, workspace_path=ws
        )
        is False
    )
    assert (
        attach_code_tool(
            [], SimpleNamespace(workspace_path=ws), "supervisor", default=True
        )
        == []
    )


@pytest.mark.parametrize("unrelated,before", UNRELATED_INVALID)
def test_an_unrelated_invalid_setting_keeps_a_valid_enable(
    tmp_path, unrelated, before, spies
):
    ws = _with_unrelated(tmp_path, AGENT, "true", unrelated, before)
    assert code_tool_enabled(AGENT, default=False, workspace_path=ws) is True
    dm = SimpleNamespace(workspace_path=ws)
    assert len(attach_code_tool([], dm, AGENT, default=False)) == 1


def test_the_switch_does_not_depend_on_the_whole_config_loading(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the switch reads its own key only")

    monkeypatch.setattr(WorkspaceAgentConfig, "load", refuse)
    ws = _workspace(tmp_path, line="false")
    assert code_tool_enabled(AGENT, default=True, workspace_path=ws) is False


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("line", ['"false"', "0", '"no"', "[]", "{ on = true }"])
def test_an_invalid_value_warns_with_the_agent_and_its_default(
    tmp_path, warnings_seen, default, line
):
    ws = _workspace(tmp_path, line=line)
    assert code_tool_enabled(AGENT, default=default, workspace_path=ws) is default
    mine = [w for w in warnings_seen if AGENT in w]
    assert len(mine) == 1, warnings_seen
    assert "code_tool" in mine[0] and ("on" if default else "off") in mine[0]
    assert "default" in mine[0]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("agent_settings = 5\n", id="agent_settings_not_a_table"),
        pytest.param(f'[agent_settings]\n{AGENT} = "on"\n', id="agent_not_a_table"),
        pytest.param("[agent_settings.x\n", id="bad_toml"),
        pytest.param(b"\xff\xfe code_tool = true", id="not_utf8"),
    ],
)
def test_an_unusable_config_warns_and_falls_through(tmp_path, warnings_seen, body):
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "config.toml"
    target.write_bytes(body) if isinstance(body, bytes) else target.write_text(body)
    for default in (True, False):
        assert code_tool_enabled(AGENT, default=default, workspace_path=ws) is default
    assert len([w for w in warnings_seen if AGENT in w]) == 2, warnings_seen


def test_a_missing_config_or_key_is_silent(tmp_path, warnings_seen):
    ws = _workspace(tmp_path)
    assert code_tool_enabled(AGENT, default=True, workspace_path=ws) is True
    (ws / "config.toml").write_text("[agent_settings.other_expert]\ncode_tool = 1\n")
    assert code_tool_enabled(AGENT, default=True, workspace_path=ws) is True
    (ws / "config.toml").write_text(f"[agent_settings.{AGENT}]\ntemperature = 0.1\n")
    assert code_tool_enabled(AGENT, default=False, workspace_path=ws) is False
    assert warnings_seen == []
    missing = tmp_path / "does_not_exist"
    assert code_tool_enabled(AGENT, default=True, workspace_path=missing) is True
    assert warnings_seen == []


# ------------------------------------------------------------ the workspace setting


def test_code_tool_setting_accepts_only_real_booleans(warnings_seen):
    assert AgentSettings(code_tool=True).code_tool is True
    assert AgentSettings(code_tool=False).code_tool is False
    assert AgentSettings().code_tool is None
    for value in ("true", "false", 1, 0, 1.0, "1"):
        assert AgentSettings(code_tool=value, temperature=0.2).code_tool is None
    assert len(warnings_seen) == 6


def test_false_survives_save_and_load_and_none_is_not_written(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    WorkspaceAgentConfig(
        agent_settings={
            AGENT: AgentSettings(code_tool=False),
            "other_expert": AgentSettings(temperature=0.3),
        }
    ).save(ws)
    text = (ws / "config.toml").read_text()
    assert text.count("code_tool") == 1
    loaded = WorkspaceAgentConfig.load(ws).agent_settings
    assert loaded[AGENT].code_tool is False
    assert loaded["other_expert"].code_tool is None
    assert code_tool_enabled(AGENT, default=True, workspace_path=ws) is False


# ---------------------------------------------------------------------- the helper


@pytest.fixture
def spies(monkeypatch):
    from lobster.services.execution import custom_code_execution_service as service_mod
    from lobster.tools import custom_code_tool as tool_mod

    record = SimpleNamespace(services=[], calls=[])

    class CountingService:
        def __init__(self, data_manager):
            record.services.append(data_manager)

    def factory(*args, **kwargs):
        record.calls.append((args, kwargs))
        return SimpleNamespace(name="execute_custom_code", metadata=None, tags=None)

    monkeypatch.setattr(service_mod, "CustomCodeExecutionService", CountingService)
    monkeypatch.setattr(tool_mod, "create_execute_custom_code_tool", factory)
    record.service_class = CountingService
    return record


def _dm(tmp_path):
    return SimpleNamespace(workspace_path=_workspace(tmp_path))


def test_off_builds_nothing_and_returns_the_same_tools(tmp_path, spies):
    tools = ["a", "b"]
    out = attach_code_tool(tools, _dm(tmp_path), AGENT, default=False)
    assert out == ["a", "b"] and out is not tools and tools == ["a", "b"]
    assert spies.services == [] and spies.calls == []


def test_on_appends_one_tool_with_code_exec_metadata(tmp_path, spies):
    dm = _dm(tmp_path)
    tools = ["a", "b"]
    out = attach_code_tool(tools, dm, AGENT, default=True)
    assert tools == ["a", "b"]
    assert out[:2] == ["a", "b"] and len(out) == 3
    assert out[2].metadata == CODE_EXEC and out[2].tags == ["CODE_EXEC"]
    assert spies.services == [dm]
    [(args, kwargs)] = spies.calls
    assert kwargs == {} and len(args) == 4
    assert args[0] is dm and isinstance(args[1], spies.service_class)
    assert args[2:] == (AGENT, None)


def test_tool_agent_name_and_post_processor_are_passed_through(tmp_path, spies):
    def post(*args):
        return None

    attach_code_tool(
        [],
        _dm(tmp_path),
        "data_expert_agent",
        default=True,
        tool_agent_name="data_expert",
        post_processor=post,
    )
    [(args, _)] = spies.calls
    assert args[2] == "data_expert" and args[3] is post


def test_switch_is_keyed_on_switch_name_not_tool_agent_name(
    tmp_path, spies, monkeypatch
):
    monkeypatch.setenv(agent_env_var("data_expert"), "0")
    out = attach_code_tool(
        [],
        _dm(tmp_path),
        "data_expert_agent",
        default=True,
        tool_agent_name="data_expert",
    )
    assert len(out) == 1
    monkeypatch.setenv(agent_env_var("data_expert_agent"), "0")
    out = attach_code_tool(
        [],
        _dm(tmp_path),
        "data_expert_agent",
        default=True,
        tool_agent_name="data_expert",
    )
    assert out == []


def test_real_tool_when_on(tmp_path):
    dm = MagicMock()
    dm.workspace_path = _workspace(tmp_path)
    [tool] = attach_code_tool([], dm, AGENT, default=True)
    assert tool.name == "execute_custom_code"
    assert tool.metadata == CODE_EXEC and tool.tags == ["CODE_EXEC"]


# ------------------------------------------------------------- supervisor and graph


class _FakeAgent:
    def __init__(self, tools):
        self.nodes = {"tools": SimpleNamespace(runnable=SimpleNamespace(tools=tools))}

    def with_config(self, *args, **kwargs):
        return self


def _installed_agents():
    return {
        e.name: e.value
        for e in entry_points(group="lobster.agents")
        if e.value.startswith("lobster.agents.")
    }


def _build_graph(tmp_path, enabled_agents, expected_workers):
    """Build the real graph with real agent factories; no model is created or run.

    Every agent passes the tier filter, and the build must create exactly
    ``expected_workers``: a worker whose factory failed (the graph skips it) fails here.
    """
    from lobster.agents import graph as graph_module
    from lobster.core.aquadif_monitor import AquadifMonitor

    configs = {}
    for value in _installed_agents().values():
        module_name, attribute = value.split(":")
        config = getattr(importlib.import_module(module_name), attribute)
        configs[config.factory_function] = config

    def import_factory(path):
        config = configs[path]
        module_name, factory_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)

        def factory(**kwargs):
            with (
                patch.object(module, "create_llm", return_value=MagicMock()),
                patch.object(module, "create_react_agent", side_effect=lambda **kw: kw),
            ):
                agent = getattr(module, factory_name)(
                    data_manager=kwargs["data_manager"],
                    agent_name=kwargs.get("agent_name", config.name),
                    workspace_path=tmp_path,
                    delegation_tools=[],
                )
            built[config.name] = [t.name for t in agent["tools"]]
            return _FakeAgent(agent["tools"])

        return factory

    captured = []
    built = {}

    def create_react_agent(**kwargs):
        captured.append(kwargs)
        agent = MagicMock()
        agent.with_config.return_value = agent
        return agent

    data_manager = MagicMock()
    data_manager.workspace_path = _workspace(tmp_path)
    data_manager.get_modality_ids.return_value = []
    monitor = AquadifMonitor(tool_metadata_map={})
    with (
        patch.object(graph_module, "create_llm", return_value=MagicMock()),
        patch.object(
            graph_module, "create_react_agent", side_effect=create_react_agent
        ),
        patch.object(graph_module, "import_agent_factory", side_effect=import_factory),
    ):
        _, metadata = graph_module.create_bioinformatics_graph(
            data_manager=data_manager,
            enabled_agents=enabled_agents,
            aquadif_monitor=monitor,
            subscription_tier="free",
            agent_filter=lambda name, config: True,
        )
    supervisor_tools = [kw for kw in captured if "tools" in kw][-1]["tools"]
    tool_map = monitor._tool_metadata_map
    assert sorted(built) == sorted(expected_workers), "a worker was not built"
    assert sorted(a.name for a in metadata.available_agents) == sorted(built)
    for worker, names in built.items():  # its tools reached the AQUADIF map
        assert names and all(name in tool_map for name in names), worker
    return supervisor_tools, tool_map


SUPERVISOR_UTILITY = [
    "list_available_modalities",
    "get_content_from_workspace",
    "delete_from_workspace",
    "write_todos",
    "read_todos",
    "execute_custom_code",
    "ask_user",
    "get_dashboard_schema",
]


@pytest.mark.parametrize("mode", [None, "default", "all", "none"])
def test_supervisor_tools_per_mode(tmp_path, monkeypatch, mode):
    if mode is not None:
        monkeypatch.setenv(GLOBAL_ENV_VAR, mode)
    tools, _ = _build_graph(tmp_path, enabled_agents=[], expected_workers=[])
    names = [t.name for t in tools]
    expected = list(SUPERVISOR_UTILITY)
    if mode == "none":
        expected.remove("execute_custom_code")
    assert names[-len(expected) :] == expected
    for tool in tools:
        if tool.name == "execute_custom_code":
            assert tool.metadata == CODE_EXEC and tool.tags == ["CODE_EXEC"]


def test_supervisor_switch_off_by_its_own_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(agent_env_var("supervisor"), "0")
    tools, _ = _build_graph(tmp_path, enabled_agents=[], expected_workers=[])
    assert "execute_custom_code" not in [t.name for t in tools]


# The AQUADIF monitor map entry for execute_custom_code, per graph and mode.
# (enabled agents, mode, expected map entry); the workers the build must create are the
# enabled agents plus their installed child agents, or every installed agent.
MAP_ROWS = [
    pytest.param(None, None, CODE_EXEC, id="default_full_graph"),
    pytest.param(["research_agent"], None, CODE_EXEC, id="research_only"),
    pytest.param(["transcriptomics_expert"], None, CODE_EXEC, id="default_off_experts"),
    pytest.param(None, "none", "<absent>", id="mode_none"),
    pytest.param(None, "all", CODE_EXEC, id="mode_all"),
]


def _expected_workers(enabled):
    installed = _installed_agents()
    if enabled is None:
        return sorted(installed)
    missing = set(enabled) - set(installed)
    if missing:
        pytest.skip(f"agent package not installed: {sorted(missing)}")
    workers = set(enabled)
    for (
        name
    ) in enabled:  # the graph auto-includes installed children of enabled parents
        module_name, attribute = installed[name].split(":")
        config = getattr(importlib.import_module(module_name), attribute)
        workers.update(c for c in config.child_agents or [] if c in installed)
    return sorted(workers)


def test_expected_workers_follow_the_installed_children():
    installed = _installed_agents()
    if "transcriptomics_expert" in installed:
        assert {"annotation_expert", "de_analysis_expert"} & set(installed) <= set(
            _expected_workers(["transcriptomics_expert"])
        )
    if "metadata_assistant" in installed:
        assert "metadata_assistant" in _expected_workers(["research_agent"])


@pytest.mark.parametrize("enabled,mode,expected", MAP_ROWS)
def test_aquadif_map_entry_for_the_code_tool(
    tmp_path, monkeypatch, enabled, mode, expected
):
    if mode is not None:
        monkeypatch.setenv(GLOBAL_ENV_VAR, mode)
    workers = _expected_workers(enabled)
    tools, tool_map = _build_graph(
        tmp_path, enabled_agents=enabled, expected_workers=workers
    )
    assert tool_map.get("execute_custom_code", "<absent>") == expected
