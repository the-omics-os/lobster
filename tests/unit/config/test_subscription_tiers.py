"""Unit tests for subscription tier checking functionality.

Tests for TierRestrictedError and check_tier_access() in subscription_tiers.py.
These functions provide runtime tier gating for premium agent access.
"""

from unittest.mock import patch

import pytest

from lobster.config.subscription_tiers import (
    TierRestrictedError,
    check_tier_access,
    is_tier_at_least,
)


# =============================================================================
# TierRestrictedError Tests
# =============================================================================


class TestTierRestrictedError:
    """Tests for TierRestrictedError exception class."""

    def test_tier_restricted_error_message_format(self):
        """Verify error message contains all required components."""
        error = TierRestrictedError(
            agent_name="proteomics_expert",
            required_tier="premium",
            current_tier="free",
        )

        message = str(error)

        # Check all required components
        assert "proteomics_expert" in message
        assert "Premium" in message  # Title case for display
        assert "Free" in message  # Title case for current tier
        assert "https://omics-os.com/pricing" in message
        assert "lobster activate <your-key>" in message

    def test_tier_restricted_error_attributes(self):
        """Verify exception attributes are set correctly."""
        error = TierRestrictedError(
            agent_name="genomics_expert",
            required_tier="premium",
            current_tier="free",
        )

        assert error.agent_name == "genomics_expert"
        assert error.required_tier == "premium"
        assert error.current_tier == "free"

    def test_tier_restricted_error_message_structure(self):
        """Verify message follows expected structure."""
        error = TierRestrictedError(
            agent_name="test_agent",
            required_tier="enterprise",
            current_tier="premium",
        )

        message = str(error)

        # Message should have clear sections
        assert "requires" in message.lower()
        assert "your current tier" in message.lower()
        assert "upgrade" in message.lower()
        assert "activate" in message.lower()

    def test_tier_restricted_error_case_handling(self):
        """Verify tier names are displayed in title case."""
        error = TierRestrictedError(
            agent_name="test_agent",
            required_tier="premium",
            current_tier="free",
        )

        message = str(error)

        # Should use title case for display
        assert "Premium" in message
        assert "Free" in message


# =============================================================================
# check_tier_access() Tests
# =============================================================================


class TestCheckTierAccess:
    """Tests for check_tier_access() function."""

    @pytest.mark.parametrize(
        "current_tier",
        ["free", "premium", "enterprise"],
    )
    def test_check_tier_access_free_agent_any_tier(self, current_tier):
        """Free required tier should pass for any user tier."""
        with patch(
            "lobster.core.license_manager.get_current_tier",
            return_value=current_tier,
        ):
            # Should not raise any exception
            check_tier_access("any_agent", required_tier="free")

    def test_check_tier_access_premium_agent_free_user(self):
        """Premium required + free user = raises TierRestrictedError."""
        with patch(
            "lobster.core.license_manager.get_current_tier",
            return_value="free",
        ):
            with pytest.raises(TierRestrictedError) as exc_info:
                check_tier_access("proteomics_expert", required_tier="premium")

            error = exc_info.value
            assert error.agent_name == "proteomics_expert"
            assert error.required_tier == "premium"
            assert error.current_tier == "free"

    def test_check_tier_access_premium_agent_premium_user(self):
        """Premium required + premium user = passes."""
        with patch(
            "lobster.core.license_manager.get_current_tier",
            return_value="premium",
        ):
            # Should not raise any exception
            check_tier_access("proteomics_expert", required_tier="premium")

    def test_check_tier_access_enterprise_bypass(self):
        """Enterprise tier should bypass all checks regardless of required_tier."""
        with patch(
            "lobster.core.license_manager.get_current_tier",
            return_value="enterprise",
        ):
            # Enterprise should pass for free requirement
            check_tier_access("free_agent", required_tier="free")

            # Enterprise should pass for premium requirement
            check_tier_access("premium_agent", required_tier="premium")

            # Enterprise should pass for enterprise requirement
            check_tier_access("enterprise_agent", required_tier="enterprise")

    @pytest.mark.parametrize(
        "tier_input,expected_tier",
        [
            ("Premium", "premium"),
            ("PREMIUM", "premium"),
            ("premium", "premium"),
            ("Free", "free"),
            ("FREE", "free"),
            ("free", "free"),
            ("Enterprise", "enterprise"),
            ("ENTERPRISE", "enterprise"),
            ("enterprise", "enterprise"),
        ],
    )
    def test_check_tier_access_case_insensitive(self, tier_input, expected_tier):
        """Tier comparison should be case insensitive."""
        with patch(
            "lobster.core.license_manager.get_current_tier",
            return_value=tier_input,
        ):
            # Free tier requirement should always pass
            check_tier_access("test_agent", required_tier="free")

    def test_check_tier_access_default_required_tier(self):
        """Default required_tier should be 'free'."""
        with patch(
            "lobster.core.license_manager.get_current_tier",
            return_value="free",
        ):
            # Should pass without specifying required_tier (defaults to free)
            check_tier_access("any_agent")

    def test_check_tier_access_enterprise_required_premium_user(self):
        """Enterprise required + premium user = raises TierRestrictedError."""
        with patch(
            "lobster.core.license_manager.get_current_tier",
            return_value="premium",
        ):
            with pytest.raises(TierRestrictedError) as exc_info:
                check_tier_access("enterprise_only_agent", required_tier="enterprise")

            error = exc_info.value
            assert error.required_tier == "enterprise"
            assert error.current_tier == "premium"

    def test_check_tier_access_error_contains_helpful_info(self):
        """TierRestrictedError should contain helpful upgrade information."""
        with patch(
            "lobster.core.license_manager.get_current_tier",
            return_value="free",
        ):
            with pytest.raises(TierRestrictedError) as exc_info:
                check_tier_access("premium_agent", required_tier="premium")

            error_message = str(exc_info.value)

            # Should contain the agent name
            assert "premium_agent" in error_message

            # Should contain upgrade URL
            assert "https://omics-os.com/pricing" in error_message

            # Should contain activation command
            assert "lobster activate" in error_message


# =============================================================================
# is_tier_at_least() Tests (for completeness)
# =============================================================================


class TestIsTierAtLeast:
    """Tests for is_tier_at_least() helper function."""

    def test_free_meets_free(self):
        """Free tier meets free requirement."""
        assert is_tier_at_least("free", "free") is True

    def test_premium_meets_free(self):
        """Premium tier meets free requirement."""
        assert is_tier_at_least("premium", "free") is True

    def test_enterprise_meets_free(self):
        """Enterprise tier meets free requirement."""
        assert is_tier_at_least("enterprise", "free") is True

    def test_free_does_not_meet_premium(self):
        """Free tier does not meet premium requirement."""
        assert is_tier_at_least("free", "premium") is False

    def test_premium_meets_premium(self):
        """Premium tier meets premium requirement."""
        assert is_tier_at_least("premium", "premium") is True

    def test_enterprise_meets_premium(self):
        """Enterprise tier meets premium requirement."""
        assert is_tier_at_least("enterprise", "premium") is True

    def test_free_does_not_meet_enterprise(self):
        """Free tier does not meet enterprise requirement."""
        assert is_tier_at_least("free", "enterprise") is False

    def test_premium_does_not_meet_enterprise(self):
        """Premium tier does not meet enterprise requirement."""
        assert is_tier_at_least("premium", "enterprise") is False

    def test_enterprise_meets_enterprise(self):
        """Enterprise tier meets enterprise requirement."""
        assert is_tier_at_least("enterprise", "enterprise") is True

    def test_case_insensitive_comparison(self):
        """Tier comparison should be case insensitive."""
        assert is_tier_at_least("PREMIUM", "premium") is True
        assert is_tier_at_least("Premium", "PREMIUM") is True
        assert is_tier_at_least("free", "FREE") is True
