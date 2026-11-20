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
