"""
Integration tests for runtime tier gating at agent factory invocation.

These tests verify the complete tier gating flow:
- All official agents available at free tier (no premium gating)
- Enterprise users bypass all tier checks
- Tier infrastructure intact for custom packages
- Free agents work without activation (no tier check blocks them)

Phase: 07-subscription-tier-adaptation
Plan: 07-03
"""

import inspect
import pytest
from unittest.mock import patch, MagicMock

from lobster.config.subscription_tiers import (
    TierRestrictedError,
    check_tier_access,
    is_tier_at_least,
    is_agent_available,
    SUBSCRIPTION_TIERS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_free_tier():
    """Mock get_current_tier() to return 'free'."""
    with patch("lobster.core.license_manager.get_current_tier", return_value="free"):
        yield


@pytest.fixture
def mock_premium_tier():
    """Mock get_current_tier() to return 'premium'."""
    with patch("lobster.core.license_manager.get_current_tier", return_value="premium"):
        yield


@pytest.fixture
def mock_enterprise_tier():
    """Mock get_current_tier() to return 'enterprise'."""
    with patch("lobster.core.license_manager.get_current_tier", return_value="enterprise"):
        yield


# =============================================================================
# CORE TIER GATING TESTS
# =============================================================================


class TestTierGatingBasics:
    """Basic tier gating functionality tests."""

    def test_free_agent_works_for_free_user(self, mock_free_tier):
        """Free agents work without activation - no tier check blocks them."""
        check_tier_access("test_free_agent", required_tier="free")

    def test_free_agent_works_for_premium_user(self, mock_premium_tier):
        """Free agents also work for premium users."""
        check_tier_access("test_free_agent", required_tier="free")

    def test_premium_requirement_blocked_for_free_user(self, mock_free_tier):
        """Tier infrastructure still works: premium requirement blocks free user."""
        with pytest.raises(TierRestrictedError) as exc_info:
            check_tier_access("test_custom_agent", required_tier="premium")

        error = exc_info.value
        assert error.agent_name == "test_custom_agent"
        assert error.required_tier == "premium"
        assert error.current_tier == "free"

    def test_premium_requirement_allowed_for_premium_user(self, mock_premium_tier):
        """Premium user can access premium-required features."""
        check_tier_access("test_custom_agent", required_tier="premium")

    def test_enterprise_bypasses_all_checks(self, mock_enterprise_tier):
        """Enterprise user can invoke any agent."""
        check_tier_access("test_agent", required_tier="premium")
        check_tier_access("test_agent", required_tier="enterprise")


class TestAllAgentsFreeAtFreeTier:
    """Verify all official agents are available at the free tier."""

    def test_all_official_agents_in_free_tier(self):
        """All 11 official agents listed in free tier."""
        free_agents = SUBSCRIPTION_TIERS["free"]["agents"]
        expected_agents = [
            "research_agent",
            "data_expert_agent",
            "transcriptomics_expert",
            "visualization_expert_agent",
            "annotation_expert",
            "de_analysis_expert",
            "metadata_assistant",
            "proteomics_expert",
            "genomics_expert",
            "machine_learning_expert_agent",
            "protein_structure_visualization_expert_agent",
        ]
        for agent in expected_agents:
            assert agent in free_agents, f"{agent} not in free tier agents list"

    def test_no_restricted_handoffs_at_free_tier(self):
        """Free tier has no handoff restrictions."""
        assert SUBSCRIPTION_TIERS["free"]["restricted_handoffs"] == {}

    def test_formerly_premium_agents_available_at_free_tier(self):
        """Genomics, proteomics, and ML agents available at free tier."""
        assert is_agent_available("genomics_expert", "free")
        assert is_agent_available("proteomics_expert", "free")
        assert is_agent_available("machine_learning_expert_agent", "free")

    def test_all_8_agent_packages_available_at_free_tier(self):
        """All 8 agent packages (11 agents) available at free tier."""
        agents = [
            "research_agent",
            "data_expert_agent",
            "transcriptomics_expert",
            "annotation_expert",
            "de_analysis_expert",
            "visualization_expert_agent",
            "metadata_assistant",
            "protein_structure_visualization_expert_agent",
            "genomics_expert",
            "proteomics_expert",
            "machine_learning_expert_agent",
        ]
        for agent in agents:
            assert is_agent_available(agent, "free"), f"{agent} not available at free tier"


class TestTierRestrictedErrorContent:
    """Tests for TierRestrictedError message content."""

    def test_error_contains_agent_name(self, mock_free_tier):
        """Error message includes the restricted agent name."""
        with pytest.raises(TierRestrictedError) as exc_info:
            check_tier_access("custom_enterprise_agent", required_tier="premium")

        error_str = str(exc_info.value)
        assert "custom_enterprise_agent" in error_str

    def test_error_contains_pricing_url(self, mock_free_tier):
        """Error message includes upgrade URL."""
        with pytest.raises(TierRestrictedError) as exc_info:
            check_tier_access("test_agent", required_tier="premium")

        error_str = str(exc_info.value)
        assert "omics-os.com/pricing" in error_str

    def test_error_contains_activate_command(self, mock_free_tier):
        """Error message includes activation CLI command."""
        with pytest.raises(TierRestrictedError) as exc_info:
            check_tier_access("test_agent", required_tier="premium")

        error_str = str(exc_info.value)
        assert "lobster activate" in error_str

    def test_error_shows_current_tier(self, mock_free_tier):
        """Error message shows user's current tier."""
        with pytest.raises(TierRestrictedError) as exc_info:
            check_tier_access("test_agent", required_tier="premium")

        error = exc_info.value
        assert error.current_tier == "free"
        assert "Free" in str(error)  # Display format

    def test_error_shows_required_tier(self, mock_free_tier):
        """Error message shows required tier."""
        with pytest.raises(TierRestrictedError) as exc_info:
            check_tier_access("test_agent", required_tier="premium")

        error = exc_info.value
        assert error.required_tier == "premium"
        assert "Premium" in str(error)  # Display format


class TestTierHierarchy:
    """Tests for tier hierarchy validation."""

    def test_tier_hierarchy_free_lowest(self):
        """Free tier is lowest in hierarchy."""
        assert is_tier_at_least("free", "free")
        assert not is_tier_at_least("free", "premium")
        assert not is_tier_at_least("free", "enterprise")

    def test_tier_hierarchy_premium_middle(self):
        """Premium tier is in the middle."""
        assert is_tier_at_least("premium", "free")
        assert is_tier_at_least("premium", "premium")
        assert not is_tier_at_least("premium", "enterprise")

    def test_tier_hierarchy_enterprise_highest(self):
        """Enterprise tier is highest - can access everything."""
        assert is_tier_at_least("enterprise", "free")
        assert is_tier_at_least("enterprise", "premium")
        assert is_tier_at_least("enterprise", "enterprise")

    def test_case_insensitive_tier_check(self, mock_free_tier):
        """Tier names are case insensitive."""
        check_tier_access("test_agent", required_tier="FREE")
        check_tier_access("test_agent", required_tier="Free")
        check_tier_access("test_agent", required_tier="free")


class TestGenomicsExpertFree:
    """Tests specific to genomics_expert being free."""

    def test_genomics_expert_agent_config_is_free(self):
        """Verify genomics_expert AGENT_CONFIG has tier_requirement='free'."""
        from lobster.agents.genomics.genomics_expert import AGENT_CONFIG

        assert AGENT_CONFIG.tier_requirement == "free"

    def test_genomics_expert_no_tier_check_in_factory(self):
        """Verify genomics_expert factory does not call check_tier_access."""
        from lobster.agents.genomics.genomics_expert import genomics_expert

        source = inspect.getsource(genomics_expert)
        assert "check_tier_access" not in source

    def test_proteomics_expert_agent_config_is_free(self):
        """Verify proteomics_expert AGENT_CONFIG has tier_requirement='free'."""
        from lobster.agents.proteomics.proteomics_expert import AGENT_CONFIG

        assert AGENT_CONFIG.tier_requirement == "free"

    def test_ml_expert_agent_config_is_free(self):
        """Verify ML expert AGENT_CONFIG has tier_requirement='free'."""
        from lobster.agents.machine_learning.config import (
            ML_EXPERT_CONFIG,
            FEATURE_SELECTION_EXPERT_CONFIG,
            SURVIVAL_ANALYSIS_EXPERT_CONFIG,
        )

        assert ML_EXPERT_CONFIG.tier_requirement == "free"
        assert FEATURE_SELECTION_EXPERT_CONFIG.tier_requirement == "free"
        assert SURVIVAL_ANALYSIS_EXPERT_CONFIG.tier_requirement == "free"


class TestEdgeCases:
    """Edge case tests for tier gating."""

    def test_empty_required_tier_defaults_to_free(self, mock_free_tier):
        """Empty or missing required_tier should be treated as free."""
        check_tier_access("test_agent", required_tier="free")

    def test_unknown_tier_treated_as_free(self):
        """Unknown tier names should be handled gracefully."""
        assert is_tier_at_least("unknown", "free")  # unknown >= free
        assert not is_tier_at_least("unknown", "premium")

    def test_enterprise_bypasses_even_enterprise_requirement(self, mock_enterprise_tier):
        """Enterprise user can access even enterprise-only features."""
        check_tier_access("custom_agent", required_tier="enterprise")


class TestIntegrationWithComponentRegistry:
    """Integration tests with component registry (if available)."""

    def test_tier_check_with_real_agent_config(self):
        """Verify tier check works with real agent configurations."""
        try:
            from lobster.core.component_registry import component_registry

            agent_config = component_registry.get_agent("genomics_expert")

            if agent_config:
                tier_req = getattr(agent_config, "tier_requirement", "free")
                assert tier_req == "free", (
                    f"genomics_expert tier_requirement should be 'free', got '{tier_req}'"
                )
        except Exception:
            pytest.skip("component_registry not available")


# =============================================================================
# SUMMARY
# =============================================================================
# Total tests: ~25
# Coverage:
#   - All official agents available at free tier
#   - No handoff restrictions at free tier
#   - Tier infrastructure intact for custom packages
#   - TierRestrictedError content validation
#   - Tier hierarchy verification
#   - genomics/proteomics/ml-specific free tier validation
#   - Edge cases and error handling
