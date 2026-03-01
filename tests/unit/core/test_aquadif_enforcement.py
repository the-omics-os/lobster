"""Test that ComponentRegistry warns on agents missing contract fields."""

import logging
from unittest.mock import MagicMock

from lobster.core.component_registry import ComponentRegistry


def test_registry_warns_missing_tier_requirement(caplog):
    """Registry should warn when agent config lacks tier_requirement."""
    registry = ComponentRegistry()

    # Simulate a loaded agent config missing tier_requirement
    bad_config = MagicMock(spec=[])  # Empty spec = no attributes
    registry._agents["bad_agent"] = bad_config
    registry._loaded = False  # Won't actually load, we'll call check directly

    with caplog.at_level(logging.WARNING):
        registry._check_agent_contract_compliance()

    assert any("bad_agent" in r.message and "tier_requirement" in r.message for r in caplog.records)


def test_registry_warns_empty_name(caplog):
    """Registry should warn when agent config has empty name."""
    registry = ComponentRegistry()

    bad_config = MagicMock()
    bad_config.name = ""
    bad_config.tier_requirement = "free"
    registry._agents["empty_name_agent"] = bad_config

    with caplog.at_level(logging.WARNING):
        registry._check_agent_contract_compliance()

    assert any("empty_name_agent" in r.message and "empty" in r.message for r in caplog.records)


def test_registry_no_warning_for_compliant_agent(caplog):
    """Registry should not warn for properly configured agents."""
    registry = ComponentRegistry()

    good_config = MagicMock()
    good_config.name = "test_agent"
    good_config.tier_requirement = "free"
    registry._agents["test_agent"] = good_config

    with caplog.at_level(logging.WARNING):
        registry._check_agent_contract_compliance()

    assert not any("test_agent" in r.message for r in caplog.records)
