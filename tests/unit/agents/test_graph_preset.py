"""The session's experts follow a preset saved in config.toml."""

from types import SimpleNamespace

import pytest

from lobster.agents.graph import _resolve_worker_agents
from lobster.config.workspace_agent_config import WorkspaceAgentConfig
from lobster.core.component_registry import component_registry

pytestmark = pytest.mark.unit

# scrna-basic: research, data_expert, transcriptomics, visualization
INSTALLED = {
    "research_agent": [],
    "data_expert_agent": [],
    "transcriptomics_expert": ["annotation_expert"],
    "annotation_expert": [],
    "genomics_expert": [],
    "proteomics_expert": [],
}


@pytest.fixture(autouse=True)
def installed(monkeypatch):
    agents = {
        name: SimpleNamespace(name=name, child_agents=children)
        for name, children in INSTALLED.items()
    }
    monkeypatch.setattr(component_registry, "list_agents", lambda: dict(agents))


def _workers(config):
    workers, _, _ = _resolve_worker_agents(None, config, lambda *a: True, "free")
    return set(workers)


def test_a_preset_runs_only_its_installed_agents():
    # visualization_expert_agent is not installed: skipped, no crash
    assert _workers(WorkspaceAgentConfig(preset="scrna-basic")) == {
        "research_agent",
        "data_expert_agent",
        "transcriptomics_expert",
        "annotation_expert",
    }


def test_an_uninstalled_name_in_the_list_is_skipped():
    config = WorkspaceAgentConfig(enabled_agents=["research_agent", "gone_expert"])
    assert _workers(config) == {"research_agent"}


def test_no_preference_runs_every_installed_agent():
    assert _workers(WorkspaceAgentConfig()) == set(INSTALLED)
    assert _workers(None) == set(INSTALLED)


NONE_INSTALLED = (
    "None of this project's experts (gone_expert, lost_expert) are installed; "
    "run lobster agents install <package> or lobster init --force."
)


def _metadata(config, tmp_path, **options):
    from unittest.mock import MagicMock, patch

    import lobster.agents.graph as graph_module

    def react(**_):
        agent = MagicMock()
        agent.with_config.return_value = agent
        return agent

    data_manager = MagicMock()
    data_manager.workspace_path = tmp_path
    data_manager.get_modality_ids.return_value = []
    data_manager.list_modalities.return_value = []
    with (
        patch.object(graph_module, "create_llm", return_value=MagicMock()),
        patch.object(graph_module, "create_react_agent", side_effect=react),
    ):
        _, metadata = graph_module.create_bioinformatics_graph(
            data_manager=data_manager,
            config=config,
            **{
                "subscription_tier": "free",
                "agent_filter": lambda name, cfg: True,
                **options,
            },
        )
    return metadata


def test_a_config_with_no_installed_expert_says_so_once(tmp_path, monkeypatch, caplog):
    import lobster.agents.graph as graph_module

    monkeypatch.setattr(graph_module, "_told_uninstalled", set())
    config = WorkspaceAgentConfig(enabled_agents=["lost_expert", "gone_expert"])

    with caplog.at_level("WARNING", logger="lobster"):
        first = _metadata(config, tmp_path)
        again = _metadata(config, tmp_path)

    assert first.uninstalled_experts == ["gone_expert", "lost_expert"]
    assert again.uninstalled_experts == ["gone_expert", "lost_expert"]
    told = [r for r in caplog.records if r.levelname == "WARNING"]
    assert [r.getMessage() for r in told if "None of" in r.getMessage()] == [
        NONE_INSTALLED
    ]


def test_an_installed_expert_or_no_preference_says_nothing(tmp_path, monkeypatch):
    import lobster.agents.graph as graph_module

    monkeypatch.setattr(graph_module, "_told_uninstalled", set())
    monkeypatch.setattr(component_registry, "list_agents", lambda: {})

    # nothing installed and nothing named: the supervisor's own line covers it
    assert _metadata(WorkspaceAgentConfig(), tmp_path).uninstalled_experts == []
    assert graph_module._told_uninstalled == set()


def test_a_tier_filter_or_an_explicit_list_is_not_an_uninstalled_expert(
    tmp_path, monkeypatch
):
    import lobster.agents.graph as graph_module

    monkeypatch.setattr(graph_module, "_told_uninstalled", set())
    installed = WorkspaceAgentConfig(enabled_agents=["research_agent"])
    gone = WorkspaceAgentConfig(enabled_agents=["gone_expert"])

    tier = _metadata(installed, tmp_path, agent_filter=lambda name, cfg: False)
    explicit = _metadata(gone, tmp_path, enabled_agents=[])
    some = _metadata(
        WorkspaceAgentConfig(enabled_agents=["research_agent", "gone_expert"]),
        tmp_path,
    )

    # research_agent is installed (its fake entry builds no agent)
    assert [m.uninstalled_experts for m in (tier, explicit, some)] == [[], [], []]
    assert graph_module._told_uninstalled == set()
