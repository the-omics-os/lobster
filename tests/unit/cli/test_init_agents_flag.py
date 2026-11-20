"""What init says about --agents: ignored next to --preset, unknown names refused,
and experts of the same package that the list leaves off."""

import pytest

from tests.unit.cli.test_init_project import (  # noqa: F401  (fixture)
    KEY,
    _init_impl,
    _wizard,
    dirs,
)

pytestmark = pytest.mark.unit

IGNORED = "--agents is ignored because --preset is given."
UNKNOWN = "--agents names an unknown expert: {names}. Nothing was changed; lobster agents list shows the experts."
LEFT_OFF = "--agents leaves out {names} from the same package, so they stay off; add them to --agents to turn them on."


@pytest.fixture
def said(monkeypatch, capsys):
    from rich.console import Console

    from lobster.cli_internal.commands.heavy import init_commands

    console = Console(width=400, record=True)
    monkeypatch.setattr(init_commands, "console", console)
    return lambda: " ".join((console.export_text() + capsys.readouterr().out).split())


def _run(folder, mode, monkeypatch, picked=(), **flags):
    if mode == "non-interactive":
        return _init_impl(
            folder, non_interactive=True, anthropic_key=KEY, skip_ssl_test=True, **flags
        )
    answers = {"provider": "anthropic", "api_key": KEY, "cancelled": False}
    _wizard(monkeypatch, {**answers, "agents": list(picked)})
    return _init_impl(folder, **flags)


def _saved(folder):
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig

    return WorkspaceAgentConfig.load(folder / ".lobster_workspace")


MODES = ["non-interactive", "interactive"]


@pytest.mark.parametrize("mode", MODES)
def test_agents_next_to_a_preset_is_said_to_be_ignored(
    dirs, monkeypatch, said, mode  # noqa: F811
):
    _run(
        dirs / "study",
        mode,
        monkeypatch,
        preset="scrna-basic",
        agents="genomics_expert",
    )

    assert IGNORED in said()


@pytest.mark.parametrize("mode", MODES)
def test_a_preset_alone_says_nothing_about_agents(
    dirs, monkeypatch, said, mode  # noqa: F811
):
    _run(dirs / "study", mode, monkeypatch, preset="scrna-basic")

    assert "--agents" not in said()


@pytest.mark.parametrize("mode", MODES)
def test_agents_alone_is_not_said_to_be_ignored(
    dirs, monkeypatch, said, mode  # noqa: F811
):
    _run(dirs / "study", mode, monkeypatch, agents="genomics_expert")

    assert IGNORED not in said()


@pytest.mark.parametrize("mode", MODES)
def test_an_unknown_expert_is_refused_before_anything_is_written(
    dirs, monkeypatch, said, mode  # noqa: F811
):
    study = dirs / "study"

    code = _run(study, mode, monkeypatch, agents="research_agent,transcriptomic_expert")

    assert code == 1
    assert UNKNOWN.format(names="transcriptomic_expert") in said()
    assert not (study / ".lobster_workspace").exists()


def test_an_unknown_expert_next_to_a_preset_is_not_refused(
    dirs, monkeypatch, said  # noqa: F811
):
    code = _run(
        dirs / "study",
        "non-interactive",
        monkeypatch,
        preset="scrna-basic",
        agents="no_such_expert",
    )

    assert code == 0
    assert "unknown expert" not in said()


def test_an_installed_expert_from_another_package_is_known(
    dirs, monkeypatch, said  # noqa: F811
):
    from lobster.cli_internal.commands.heavy import init_commands

    monkeypatch.setattr(
        init_commands, "_get_installed_agents", lambda: {"my_lab_expert": object()}
    )

    assert (
        _run(dirs / "study", "non-interactive", monkeypatch, agents="my_lab_expert")
        == 0
    )
    assert "unknown expert" not in said()


def test_a_package_name_is_saved_as_its_experts(dirs, monkeypatch, said):  # noqa: F811
    study = dirs / "study"

    assert _run(study, "non-interactive", monkeypatch, agents="lobster-research") == 0

    assert _saved(study).enabled_agents == ["research_agent", "data_expert_agent"]


@pytest.mark.parametrize("mode", MODES)
def test_an_expert_the_list_leaves_off_its_package_is_named(
    dirs, monkeypatch, said, mode  # noqa: F811
):
    study = dirs / "study"

    # the picker starts from the list's packages and the user keeps them
    picked = ["lobster-research", "lobster-genomics"]
    _run(study, mode, monkeypatch, picked, agents="research_agent,genomics_expert")

    assert LEFT_OFF.format(names="data_expert_agent") in said()
    assert _saved(study).enabled_agents == ["research_agent", "genomics_expert"]


@pytest.mark.parametrize(
    "agents",
    [
        "transcriptomics_expert",  # its children come with it
        "drug_discovery_expert",
        "research_agent,data_expert_agent",
        "lobster-research",
    ],
)
def test_a_list_that_covers_its_packages_names_nothing(
    dirs, monkeypatch, said, agents  # noqa: F811
):
    _run(dirs / "study", "non-interactive", monkeypatch, agents=agents)

    assert "leaves out" not in said()


def test_a_child_without_its_parent_names_the_parent(
    dirs, monkeypatch, said  # noqa: F811
):
    _run(dirs / "study", "non-interactive", monkeypatch, agents="annotation_expert")

    assert LEFT_OFF.format(names="transcriptomics_expert, de_analysis_expert") in said()


def test_the_child_agents_match_the_installed_experts():
    from lobster.cli_internal.agent_packages import (
        AVAILABLE_AGENT_PACKAGES,
        CHILD_AGENTS,
    )
    from lobster.core.component_registry import component_registry

    agents = component_registry.list_agents()
    checked = 0
    for _, _, provided, _, _ in AVAILABLE_AGENT_PACKAGES:
        for name in provided:
            if name in agents:
                children = tuple(agents[name].child_agents or ())
                assert CHILD_AGENTS.get(name, ()) == children, name
                checked += 1
    assert checked
