"""
Unit tests for agent configuration system.
"""
import io
import sys
from unittest.mock import patch

import pytest

from lobster.config.agent_config import LobsterAgentConfigurator


def test_print_current_config_filters_by_tier_free(monkeypatch):
    """Test that print_current_config respects free tier license."""
    # Mock license tier as 'free'
    monkeypatch.setattr(
        "lobster.core.license_manager.get_current_tier", lambda: "free"
    )

    configurator = LobsterAgentConfigurator(profile="production")

    # Capture output
    captured = io.StringIO()
    sys.stdout = captured

    configurator.print_current_config()

    sys.stdout = sys.__stdout__
    output = captured.getvalue()

    # Verify license tier is displayed
    assert "License Tier: Free" in output

    # Verify free tier agents are shown
    assert "research_agent" in output.lower()
    assert "data_expert_agent" in output.lower()
    assert "transcriptomics_expert" in output.lower()
    assert "visualization_expert_agent" in output.lower()

    # Verify premium agents are NOT shown
    assert "metadata_assistant" not in output.lower()
    assert "proteomics_expert" not in output.lower()
    assert "machine_learning_expert_agent" not in output.lower()
    assert "protein_structure_visualization_expert_agent" not in output.lower()

    # Verify summary message is shown
    assert "6 agents available for free tier" in output
    assert "premium agents hidden" in output


def test_print_current_config_show_all_ignores_tier(monkeypatch):
    """Test that print_current_config with show_all=True shows all agents regardless of tier."""
    # Mock license tier as 'free'
    monkeypatch.setattr(
        "lobster.core.license_manager.get_current_tier", lambda: "free"
    )

    configurator = LobsterAgentConfigurator(profile="production")

    # Capture output
    captured = io.StringIO()
    sys.stdout = captured

    configurator.print_current_config(show_all=True)

    sys.stdout = sys.__stdout__
    output = captured.getvalue()

    # Verify license tier is displayed
    assert "License Tier: Free" in output

    # Verify free tier agents are shown
    assert "research_agent" in output.lower()
    assert "data_expert_agent" in output.lower()

    # Verify premium agents ARE shown with show_all=True
    assert "metadata_assistant" in output.lower()
    assert "proteomics_expert" in output.lower()
    assert "machine_learning_expert_agent" in output.lower()

    # Verify no summary message is shown (since show_all=True)
    assert "premium agents hidden" not in output


def test_print_current_config_premium_tier_shows_all(monkeypatch):
    """Test that premium tier users see all premium agents (but not enterprise-only)."""
    # Mock license tier as 'premium'
    monkeypatch.setattr(
        "lobster.core.license_manager.get_current_tier", lambda: "premium"
    )

    configurator = LobsterAgentConfigurator(profile="production")

    # Capture output
    captured = io.StringIO()
    sys.stdout = captured

    configurator.print_current_config()

    sys.stdout = sys.__stdout__
    output = captured.getvalue()

    # Verify license tier is displayed
    assert "License Tier: Premium" in output

    # Verify free tier agents are shown
    assert "research_agent" in output.lower()
    assert "data_expert_agent" in output.lower()

    # Verify premium agents ARE shown for premium tier
    assert "metadata_assistant" in output.lower()
    assert "proteomics_expert" in output.lower()
    assert "machine_learning_expert_agent" in output.lower()
    assert "protein_structure_visualization_expert_agent" in output.lower()

    # Verify summary shows 10 agents (premium includes free + 4 premium-only)
    assert "10 agents available for premium tier" in output

    # Note: Some system/enterprise agents (assistant, supervisor, custom_feature_agent)
    # may still be filtered out even at premium tier
    assert "agents hidden" in output  # Some filtering occurs


def test_print_current_config_enterprise_tier_shows_all(monkeypatch):
    """Test that enterprise tier users see all agents."""
    # Mock license tier as 'enterprise'
    monkeypatch.setattr(
        "lobster.core.license_manager.get_current_tier", lambda: "enterprise"
    )

    configurator = LobsterAgentConfigurator(profile="production")

    # Capture output
    captured = io.StringIO()
    sys.stdout = captured

    configurator.print_current_config()

    sys.stdout = sys.__stdout__
    output = captured.getvalue()

    # Verify license tier is displayed
    assert "License Tier: Enterprise" in output

    # Verify all agents are shown (wildcard in enterprise tier)
    assert "research_agent" in output.lower()
    assert "data_expert_agent" in output.lower()
    assert "metadata_assistant" in output.lower()
    assert "proteomics_expert" in output.lower()

    # Verify no summary message for enterprise (no agents filtered)
    assert "premium agents hidden" not in output


def test_print_current_config_consistent_across_profiles(monkeypatch):
    """Test that tier filtering works consistently across different profiles."""
    # Mock license tier as 'free'
    monkeypatch.setattr(
        "lobster.core.license_manager.get_current_tier", lambda: "free"
    )

    # Test with different profiles
    for profile in ["development", "production", "ultra"]:
        configurator = LobsterAgentConfigurator(profile=profile)

        # Capture output
        captured = io.StringIO()
        sys.stdout = captured

        configurator.print_current_config()

        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        # Verify profile is shown
        assert f"Profile: {profile}" in output

        # Verify free tier filtering applies to all profiles
        assert "License Tier: Free" in output
        assert "6 agents available for free tier" in output
        assert "metadata_assistant" not in output.lower()
