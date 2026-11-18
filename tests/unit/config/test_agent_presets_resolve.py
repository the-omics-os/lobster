"""Every agent a preset names is one an official package registers.

multiomics-full named machine_learning_expert_agent; lobster-ml registers
machine_learning_expert, so init said "unknown expert" and left ML out.
"""

import tomllib
from pathlib import Path

from lobster.config.agent_presets import AGENT_PRESETS

PACKAGES = Path(__file__).resolve().parents[3] / "packages"


def _registered_agents():
    names = set()
    for pyproject in PACKAGES.glob("*/pyproject.toml"):
        project = tomllib.loads(pyproject.read_text())["project"]
        names |= set(project.get("entry-points", {}).get("lobster.agents", {}))
    return names


def test_every_preset_agent_is_registered():
    registered = _registered_agents()
    assert registered  # the packages are in this checkout

    unknown = {
        (preset, agent)
        for preset, config in AGENT_PRESETS.items()
        for agent in config["agents"]
        if agent not in registered
    }
    assert unknown == set()
