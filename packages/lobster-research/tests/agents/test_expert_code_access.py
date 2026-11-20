"""Inspect actual factory tool lists without invoking models or scientific tools.

Every agent whose factory offers ``execute_custom_code`` adds it through
``lobster.tools.code_tool_switch.attach_code_tool``. These tests build the real
factories of the installed agents and check, per agent, the exact presence of the code
tool in each global mode, its metadata, the arguments it is built with, and the switch
keys. Static checks read the agent sources that are present in this checkout.

An agent package that is not installed (or whose source is absent) is skipped; with
``LOBSTER_EXPECT_ALL_AGENTS=1`` every expected agent must be present instead.
"""

import ast
import importlib
import os
import tomllib
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

GLOBAL_ENV_VAR = "LOBSTER_CODE_TOOL"
CODE_EXEC = {"categories": ["CODE_EXEC"], "provenance": False}

#: Every first-party agent (entry point name), installed or not. A release install
#: must provide all of them.
EXPECTED_AGENTS = {
    "annotation_expert",
    "biomarker_discovery_expert",
    "cheminformatics_expert",
    "clinical_dev_expert",
    "data_expert_agent",
    "de_analysis_expert",
    "drug_discovery_expert",
    "feature_selection_expert",
    "genomics_expert",
    "machine_learning_expert",
    "metabolomics_expert",
    "metadata_assistant",
    "peptide_expert",
    "pharmacogenomics_expert",
    "protein_structure_visualization_expert",
    "proteomics_de_analysis_expert",
    "proteomics_expert",
    "research_agent",
    "survival_analysis_expert",
    "transcriptomics_expert",
    "variant_analysis_expert",
    "visualization_expert_agent",
}
#: Agents that get the code tool by default (the supervisor is covered in
#: tests/unit/tools/test_code_tool_switch.py).
DEFAULT_ON = {"data_expert_agent", "metadata_assistant", "visualization_expert_agent"}
#: Never offered the code tool: external text must not reach custom code.
NEVER = {"research_agent"}
#: The agent name each code tool records, where it is not the factory's agent_name.
ATTRIBUTION = {
    "data_expert_agent": "data_expert",
    "metadata_assistant": "metadata_assistant",
}

PACKAGES_ROOT = Path(__file__).resolve().parents[3]
EXPECT_ALL = os.environ.get("LOBSTER_EXPECT_ALL_AGENTS") == "1"

INSTALLED = {
    entry.name: entry.value
    for entry in entry_points(group="lobster.agents")
    if entry.value.startswith("lobster.agents.")
}

#: Executable per-agent cases: the installed agents only.
AGENTS = sorted(INSTALLED.items())
MODES = ["default", "all", "none"]
# Flat parameter lists, so a case index always maps to the same agent.
AGENT_MODES = [(name, value, mode) for name, value in AGENTS for mode in MODES]
CALL_FORMS = [True, False]  # graph call (agent_name=config.name), factory default
AGENT_CALLS = [(name, value, form) for name, value in AGENTS for form in CALL_FORMS]
DEFAULT_ON_CALLS = [(name, form) for name in sorted(DEFAULT_ON) for form in CALL_FORMS]


@pytest.fixture(autouse=True)
def no_switch_env(monkeypatch):
    import os

    for key in list(os.environ):
        if key == GLOBAL_ENV_VAR or key.startswith(GLOBAL_ENV_VAR + "_"):
            monkeypatch.delenv(key)


def _absent(what):
    if EXPECT_ALL:
        pytest.fail(f"{what} (LOBSTER_EXPECT_ALL_AGENTS=1)")
    pytest.skip(what)


def _installed(name):
    """Entry point value of an installed agent; otherwise skip (or fail, see above)."""
    if name not in INSTALLED:
        _absent(f"{name}: agent package not installed")
    return INSTALLED[name]


def _config(entry_value):
    config_module, attribute = entry_value.split(":")
    return getattr(importlib.import_module(config_module), attribute)


def _build(entry_value, tmp_path, *, pass_agent_name=True):
    """Build the real factory; return (config, tools)."""
    config = _config(entry_value)
    factory_module, factory_name = config.factory_function.rsplit(".", 1)
    module = importlib.import_module(factory_module)
    data_manager = MagicMock()
    data_manager.workspace_path = tmp_path
    data_manager.workspace_dir = tmp_path
    data_manager.plot_manager = MagicMock()
    kwargs = dict(
        data_manager=data_manager, workspace_path=tmp_path, delegation_tools=[]
    )
    if pass_agent_name:  # as graph.py does: agent_name=agent_config.name
        kwargs["agent_name"] = config.name
    # Capture the real assembled tools rather than trusting source text/metadata.
    with (
        patch.object(module, "create_llm", return_value=MagicMock()),
        patch.object(module, "create_react_agent", side_effect=lambda **kw: kw),
    ):
        agent = getattr(module, factory_name)(**kwargs)
    return config, agent["tools"]


def _code_tools(tools):
    return [t for t in tools if t.name == "execute_custom_code"]


def _expected(name, mode):
    if name in NEVER or mode == "none":
        return False
    return mode == "all" or name in DEFAULT_ON


def test_installed_agent_inventory_is_not_empty():
    """This package's own agents; other packages are optional here."""
    assert {"research_agent", "data_expert_agent"} <= set(INSTALLED)


def test_installed_agents_are_exactly_the_expected_set():
    """No unexpected agent anywhere; with LOBSTER_EXPECT_ALL_AGENTS=1, none missing.

    Isolated package installs carry only some agents, so equality is asserted only
    where every agent package is expected to be installed.
    """
    installed = set(INSTALLED)
    assert installed - EXPECTED_AGENTS == set(), "unexpected agents installed"
    if os.environ.get("LOBSTER_EXPECT_ALL_AGENTS") == "1":
        assert EXPECTED_AGENTS - installed == set(), "expected agents not installed"


@pytest.mark.parametrize("entry_name,entry_value,mode", AGENT_MODES)
def test_expert_factory_code_access(
    entry_name, entry_value, mode, tmp_path, monkeypatch
):
    if mode != "default":
        monkeypatch.setenv(GLOBAL_ENV_VAR, mode)
    config, tools = _build(entry_value, tmp_path)
    code_tools = _code_tools(tools)
    if config.name == "data_expert_agent":
        assert sum(t.name == "create_interactive_plot" for t in tools) == 1
    if config.name in NEVER:
        assert not code_tools, "External-text research must not receive custom code"
    elif _expected(entry_name, mode):
        assert len(code_tools) == 1, f"{entry_name}: missing or duplicate code tool"
        assert code_tools[0].metadata == CODE_EXEC
        assert code_tools[0].tags == ["CODE_EXEC"]
    else:
        assert not code_tools, f"{entry_name}: code tool present in mode {mode}"


@pytest.mark.parametrize("entry_name,entry_value", AGENTS)
def test_declared_default_matches_the_default_set(entry_name, entry_value):
    assert _config(entry_value).code_tool_default is (entry_name in DEFAULT_ON)


@pytest.mark.parametrize("entry_name,entry_value", AGENTS)
def test_switch_removes_only_the_code_tool(
    entry_name, entry_value, tmp_path, monkeypatch
):
    monkeypatch.setenv(GLOBAL_ENV_VAR, "all")
    _, on = _build(entry_value, tmp_path)
    monkeypatch.setenv(GLOBAL_ENV_VAR, "none")
    _, off = _build(entry_value, tmp_path)
    assert [t.name for t in on if t.name != "execute_custom_code"] == [
        t.name for t in off
    ]


@pytest.mark.parametrize("entry_name,pass_agent_name", DEFAULT_ON_CALLS)
def test_default_on_agents_resolve_on_by_their_real_name(
    entry_name, pass_agent_name, tmp_path
):
    _, tools = _build(_installed(entry_name), tmp_path, pass_agent_name=pass_agent_name)
    assert len(_code_tools(tools)) == 1


# ------------------------------------------------- static checks over the agent sources


def _source_declarations():
    """{entry point name: (package dir, value)} for the agent sources present here."""
    declared = {}
    for pyproject in sorted(PACKAGES_ROOT.glob("lobster-*/pyproject.toml")):
        data = tomllib.loads(pyproject.read_text())
        entries = data.get("project", {}).get("entry-points", {})
        for name, value in entries.get("lobster.agents", {}).items():
            declared[name] = (pyproject.parent, value)
    return declared


def _declared(name):
    declared = _source_declarations()
    if name not in declared:
        _absent(f"{name}: agent source not present")
    return declared[name]


def _source_module(package_dir, module):
    base = package_dir.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    raise AssertionError(f"{module}: no source file under {package_dir}")


def _agent_config_literals(package_dir, path, target="AGENT_CONFIG"):
    """Literal keyword arguments of the module-level ``AGENT_CONFIG = ...(...)``,
    following ``AGENT_CONFIG = OTHER`` to an assignment or import of ``OTHER``."""
    body = ast.parse(path.read_text()).body
    for node in body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == target for t in node.targets
        ):
            if isinstance(node.value, ast.Call):
                return {
                    kw.arg: kw.value.value
                    for kw in node.value.keywords
                    if isinstance(kw.value, ast.Constant)
                }
            if isinstance(node.value, ast.Name):
                return _agent_config_literals(package_dir, path, node.value.id)
    for node in body:
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                if (alias.asname or alias.name) == target:
                    source = _source_module(package_dir, node.module)
                    return _agent_config_literals(package_dir, source, alias.name)
    raise AssertionError(f"{path}: no module-level {target}")


def _source_config(name):
    package_dir, value = _declared(name)
    config_path = _source_module(package_dir, value.split(":")[0])
    literals = _agent_config_literals(package_dir, config_path)
    factory_module = literals["factory_function"].rsplit(".", 1)[0]
    return literals, _source_module(package_dir, factory_module)


def _source_agent_config_name(name):
    """``name=`` of the agent's ``AGENT_CONFIG`` in its source file."""
    return _source_config(name)[0]["name"]


def _helper_calls(path):
    calls, direct = [], False
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            called = getattr(node.func, "id", getattr(node.func, "attr", None))
            direct = direct or called == "create_execute_custom_code_tool"
            if called == "attach_code_tool":
                calls.append(node)
    return calls, direct


@pytest.mark.parametrize("entry_name", sorted(DEFAULT_ON))
def test_default_on_source_name_equals_entry_point_name(entry_name):
    """The switch keys on AGENT_CONFIG.name; for a default-ON agent it must not drift."""
    assert _source_agent_config_name(entry_name) == entry_name


@pytest.mark.parametrize("entry_name", sorted(EXPECTED_AGENTS))
def test_agent_source_uses_the_switch(entry_name):
    """Static: the factory offers the tool through one switch call on its AGENT_CONFIG."""
    literals, factory_path = _source_config(entry_name)
    assert literals.get("code_tool_default", False) is (entry_name in DEFAULT_ON)
    calls, direct = _helper_calls(factory_path)
    assert not direct, f"{factory_path}: builds the code tool directly"
    if entry_name in NEVER:
        assert calls == []
        return
    [call] = calls
    keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert ast.unparse(call.args[2]) == "AGENT_CONFIG.name"
    assert keywords.get("default") == "AGENT_CONFIG.code_tool_default"
    expected = (
        repr(ATTRIBUTION[entry_name]) if entry_name in ATTRIBUTION else "agent_name"
    )
    assert keywords.get("tool_agent_name") == expected


def test_no_other_source_builds_or_attaches_the_code_tool():
    """Static, over the agent sources present here: no direct build, and every switch
    call sits in the factory module of a declared agent."""
    declared = _source_declarations()
    assert set(declared) <= EXPECTED_AGENTS, "unexpected agent declared in source"
    factories = set()
    for name in declared:
        factories.add(_source_config(name)[1].resolve())
    for path in sorted(PACKAGES_ROOT.glob("lobster-*/lobster/agents/**/*.py")):
        calls, direct = _helper_calls(path)
        assert not direct, f"{path}: builds the code tool directly"
        assert len(calls) <= 1, f"{path}: more than one switch call"
        if calls:
            assert path.resolve() in factories, f"{path}: not a declared agent factory"
    if EXPECT_ALL:
        assert set(declared) == EXPECTED_AGENTS, "agent sources missing"


@pytest.mark.parametrize("entry_name,entry_value,pass_agent_name", AGENT_CALLS)
def test_code_tool_is_built_with_unchanged_arguments(
    entry_name, entry_value, pass_agent_name, tmp_path, monkeypatch
):
    if entry_name in NEVER:
        pytest.skip("never offered the code tool")
    import inspect

    from lobster.services.execution.custom_code_execution_service import (
        CustomCodeExecutionService,
    )
    from lobster.tools import custom_code_tool

    calls = []
    real = custom_code_tool.create_execute_custom_code_tool

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(custom_code_tool, "create_execute_custom_code_tool", spy)
    monkeypatch.setenv(GLOBAL_ENV_VAR, "all")
    config, tools = _build(entry_value, tmp_path, pass_agent_name=pass_agent_name)
    [(args, kwargs)] = calls
    assert kwargs == {} and len(args) == 4
    assert isinstance(args[1], CustomCodeExecutionService)
    factory = getattr(
        importlib.import_module(config.factory_function.rsplit(".", 1)[0]),
        config.factory_function.rsplit(".", 1)[1],
    )
    factory_default = inspect.signature(factory).parameters["agent_name"].default
    expected_name = ATTRIBUTION.get(
        entry_name, config.name if pass_agent_name else factory_default
    )
    assert args[2] == expected_name
    if entry_name == "metadata_assistant":
        assert args[3] is custom_code_tool.metadata_store_post_processor
    else:
        assert args[3] is None


def _workspace_setting(tmp_path, agent, value):
    (tmp_path / "config.toml").write_text(
        f"[agent_settings.{agent}]\ncode_tool = {value}\n"
    )


@pytest.mark.parametrize(
    "agent,value,expected",
    [
        ("visualization_expert_agent", "false", 0),
        ("proteomics_expert", "true", 1),
    ],
)
def test_workspace_config_switch(agent, value, expected, tmp_path):
    _workspace_setting(tmp_path, agent, value)
    _, tools = _build(_installed(agent), tmp_path)
    assert len(_code_tools(tools)) == expected


def test_env_override_is_keyed_on_the_registry_name(tmp_path, monkeypatch):
    value = _installed("visualization_expert_agent")
    monkeypatch.setenv("LOBSTER_CODE_TOOL_VISUALIZATION_EXPERT", "0")
    _, tools = _build(value, tmp_path)
    assert len(_code_tools(tools)) == 1  # not the registry name: no effect
    monkeypatch.setenv("LOBSTER_CODE_TOOL_VISUALIZATION_EXPERT_AGENT", "0")
    _, tools = _build(value, tmp_path)
    assert _code_tools(tools) == []


def test_research_agent_never_gets_the_code_tool(tmp_path, monkeypatch):
    monkeypatch.setenv(GLOBAL_ENV_VAR, "all")
    monkeypatch.setenv("LOBSTER_CODE_TOOL_RESEARCH_AGENT", "1")
    _workspace_setting(tmp_path, "research_agent", "true")
    _, tools = _build(_installed("research_agent"), tmp_path)
    assert _code_tools(tools) == []


STRUCTURAL = "protein_structure_visualization_expert"


@pytest.fixture
def structural(monkeypatch):
    value = _installed(STRUCTURAL)
    config = _config(value)
    # restore whatever discovery left on the shared config object
    monkeypatch.setattr(config, "name", config.name)
    monkeypatch.setattr(config, "package_name", config.package_name)
    return SimpleNamespace(config=config, value=value)


def test_structural_switch_uses_the_entry_name_after_discovery(
    structural, tmp_path, monkeypatch
):
    from lobster.core.component_registry import ComponentRegistry

    ComponentRegistry()._annotate_agent_package(STRUCTURAL, structural.config, None)
    assert structural.config.name == STRUCTURAL
    _, tools = _build(structural.value, tmp_path)
    assert _code_tools(tools) == []  # default stays off
    monkeypatch.setenv("LOBSTER_CODE_TOOL_PROTEIN_STRUCTURE_VISUALIZATION_EXPERT", "1")
    _, tools = _build(structural.value, tmp_path)
    assert len(_code_tools(tools)) == 1


def test_structural_direct_call_before_discovery_uses_the_source_name(
    structural, tmp_path, monkeypatch
):
    """Known quirk (default OFF either way): the source name differs from the entry name."""
    source_name = _source_agent_config_name(STRUCTURAL)
    assert source_name == "protein_structure_visualization_expert_agent"
    monkeypatch.setattr(structural.config, "name", source_name)
    monkeypatch.setenv("LOBSTER_CODE_TOOL_PROTEIN_STRUCTURE_VISUALIZATION_EXPERT", "1")
    _, tools = _build(structural.value, tmp_path)
    assert _code_tools(tools) == []
    monkeypatch.setenv(
        "LOBSTER_CODE_TOOL_PROTEIN_STRUCTURE_VISUALIZATION_EXPERT_AGENT", "1"
    )
    _, tools = _build(structural.value, tmp_path)
    assert len(_code_tools(tools)) == 1
