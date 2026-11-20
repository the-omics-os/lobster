"""The hardcoded tier lists name agents as their packages register them."""

import tomllib
from pathlib import Path

import pytest

from lobster.config.subscription_tiers import SUBSCRIPTION_TIERS, is_agent_available

REPO = Path(__file__).resolve().parents[3]


def _entry_point_names():
    names = set()
    for pyproject in REPO.glob("packages/*/pyproject.toml"):
        project = tomllib.loads(pyproject.read_text())["project"]
        names |= set(project.get("entry-points", {}).get("lobster.agents", {}))
    return names


ENTRY_POINT_NAMES = _entry_point_names()


@pytest.mark.parametrize("tier", ["free", "premium"])
def test_every_fallback_agent_is_a_registered_name(tier):
    assert ENTRY_POINT_NAMES  # the packages were found
    unknown = set(SUBSCRIPTION_TIERS[tier]["agents"]) - ENTRY_POINT_NAMES
    assert unknown == set()


@pytest.mark.parametrize(
    "name", ["machine_learning_expert", "protein_structure_visualization_expert"]
)
def test_the_fallback_finds_an_agent_by_its_registered_name(monkeypatch, name):
    """When the registry lookup fails, the list still knows the agent."""
    from lobster.core import component_registry as registry_module

    def lookup_fails(agent_name):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(registry_module.component_registry, "get_agent", lookup_fails)
    assert is_agent_available(name, "free")
