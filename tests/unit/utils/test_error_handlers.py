"""
Unit tests for the modular error handling system.

Tests all error handlers, registry functionality, and error guidance generation.
"""

import pytest
from unittest.mock import Mock, patch

from lobster.utils.error_handlers import (
    ErrorGuidance,
    ErrorHandler,
    RateLimitErrorHandler,
    AuthenticationErrorHandler,
    NetworkErrorHandler,
    QuotaExceededErrorHandler,
    ErrorHandlerRegistry,
    get_error_registry,
    reset_error_registry
)


class TestErrorGuidance:
    """Test the ErrorGuidance dataclass."""

    def test_error_guidance_creation(self):
        """Test creating ErrorGuidance with all fields."""
        guidance = ErrorGuidance(
            error_type="test",
            title="Test Error",
            description="Test description",
            solutions=["Solution 1", "Solution 2"],
            severity="error",
            support_email="test@example.com",
            documentation_url="https://example.com",
            can_retry=True,
            retry_delay=30
        )

        assert guidance.error_type == "test"
        assert guidance.title == "Test Error"
        assert len(guidance.solutions) == 2
        assert guidance.can_retry is True
        assert guidance.retry_delay == 30

    def test_error_guidance_defaults(self):
        """Test ErrorGuidance with default values."""
        guidance = ErrorGuidance(
            error_type="test",
            title="Test",
            description="Test",
            solutions=[]
        )

        assert guidance.severity == "error"
        assert guidance.support_email == "info@omics-os.com"
        assert guidance.documentation_url is None
        assert guidance.can_retry is False
        assert guidance.retry_delay is None


class TestRateLimitErrorHandler:
    """Test the RateLimitErrorHandler."""

    def setup_method(self):
        """Set up handler for each test."""
        self.handler = RateLimitErrorHandler()

    def test_can_handle_429_error(self):
        """Test detection of 429 rate limit errors."""
        error = Exception("Error code: 429 - rate_limit_error")
        assert self.handler.can_handle(error, str(error)) is True

    def test_can_handle_rate_limit_text(self):
        """Test detection of 'rate limit' in error message."""
        error = Exception("You have exceeded the rate limit")
        assert self.handler.can_handle(error, str(error)) is True

    def test_can_handle_anthropic_rate_limit(self):
        """Test detection of Anthropic-specific rate limit message."""
        error = Exception(
            "This request would exceed your organization's maximum usage increase rate"
        )
        assert self.handler.can_handle(error, str(error)) is True

    def test_does_not_handle_other_errors(self):
        """Test that non-rate-limit errors are not handled."""
        error = Exception("Connection timeout")
        assert self.handler.can_handle(error, str(error)) is False

    def test_handle_generates_proper_guidance(self):
        """Test that handle() generates appropriate guidance."""
        error = Exception("Error 429: rate_limit_error")
        guidance = self.handler.handle(error, str(error))

        assert guidance.error_type == "rate_limit"
        assert "Rate Limit" in guidance.title
        assert guidance.severity == "warning"
        assert guidance.can_retry is True
        assert guidance.retry_delay == 60
        assert len(guidance.solutions) >= 4
        assert guidance.documentation_url is not None


class TestAuthenticationErrorHandler:
    """Test the AuthenticationErrorHandler."""

    def setup_method(self):
        """Set up handler for each test."""
        self.handler = AuthenticationErrorHandler()

    def test_can_handle_401_error(self):
        """Test detection of 401 authentication errors."""
        error = Exception("Error code: 401 - unauthorized")
        assert self.handler.can_handle(error, str(error)) is True

    def test_can_handle_invalid_key(self):
        """Test detection of invalid API key errors."""
        error = Exception("invalid api key provided")
        assert self.handler.can_handle(error, str(error)) is True

    def test_can_handle_auth_failed(self):
        """Test detection of authentication failed messages."""
        error = Exception("Authentication failed: invalid credentials")
        assert self.handler.can_handle(error, str(error)) is True

    def test_does_not_handle_other_errors(self):
        """Test that non-auth errors are not handled."""
        error = Exception("Connection timeout")
        assert self.handler.can_handle(error, str(error)) is False

    def test_handle_generates_proper_guidance(self):
        """Test that handle() generates appropriate guidance."""
        error = Exception("Error 401: invalid_api_key")
        guidance = self.handler.handle(error, str(error))

        assert guidance.error_type == "authentication"
        assert "Authentication" in guidance.title
        assert guidance.severity == "error"
        assert guidance.can_retry is False
        assert len(guidance.solutions) >= 4
        assert guidance.documentation_url is not None


class TestNetworkErrorHandler:
    """Test the NetworkErrorHandler."""

    def setup_method(self):
        """Set up handler for each test."""
        self.handler = NetworkErrorHandler()

    def test_can_handle_connection_error(self):
        """Test detection of connection errors."""
        error = Exception("Connection error: failed to connect")
        assert self.handler.can_handle(error, str(error)) is True

    def test_can_handle_timeout(self):
        """Test detection of timeout errors."""
        error = Exception("Request timed out after 30 seconds")
        assert self.handler.can_handle(error, str(error)) is True

    def test_can_handle_connection_refused(self):
        """Test detection of connection refused errors."""
        error = Exception("Connection refused by remote server")
        assert self.handler.can_handle(error, str(error)) is True

    def test_can_handle_dns_error(self):
        """Test detection of DNS errors."""
        error = Exception("DNS resolution failed for api.example.com")
        assert self.handler.can_handle(error, str(error)) is True

    def test_does_not_handle_other_errors(self):
        """Test that non-network errors are not handled."""
        error = Exception("Invalid API key")
        assert self.handler.can_handle(error, str(error)) is False

    def test_handle_generates_proper_guidance(self):
        """Test that handle() generates appropriate guidance."""
        error = Exception("Connection timeout")
        guidance = self.handler.handle(error, str(error))

        assert guidance.error_type == "network"
        assert "Network" in guidance.title
        assert guidance.severity == "error"
        assert guidance.can_retry is True
        assert guidance.retry_delay == 30
        assert len(guidance.solutions) >= 4


class TestQuotaExceededErrorHandler:
    """Test the QuotaExceededErrorHandler."""

    def setup_method(self):
        """Set up handler for each test."""
        self.handler = QuotaExceededErrorHandler()

    def test_can_handle_quota_exceeded(self):
        """Test detection of quota exceeded errors."""
        error = Exception("quota exceeded for this month")
        assert self.handler.can_handle(error, str(error)) is True

    def test_can_handle_insufficient_quota(self):
        """Test detection of insufficient quota errors."""
        error = Exception("Error: insufficient_quota")
        assert self.handler.can_handle(error, str(error)) is True

    def test_can_handle_402_error(self):
        """Test detection of 402 payment required errors."""
        error = Exception("Error code: 402 - payment required")
        assert self.handler.can_handle(error, str(error)) is True

    def test_does_not_handle_other_errors(self):
        """Test that non-quota errors are not handled."""
        error = Exception("Connection timeout")
        assert self.handler.can_handle(error, str(error)) is False

    def test_handle_generates_proper_guidance(self):
        """Test that handle() generates appropriate guidance."""
        error = Exception("insufficient_quota")
        guidance = self.handler.handle(error, str(error))

        assert guidance.error_type == "quota"
        assert "Quota" in guidance.title
        assert guidance.severity == "error"
        assert guidance.can_retry is False
        assert len(guidance.solutions) >= 4


class TestErrorHandlerRegistry:
    """Test the ErrorHandlerRegistry."""

    def setup_method(self):
        """Set up fresh registry for each test."""
        reset_error_registry()
        self.registry = ErrorHandlerRegistry()

    def test_registry_initializes_with_default_handlers(self):
        """Test that registry initializes with built-in handlers."""
        assert len(self.registry.handlers) >= 4
        # Verify handler types
        handler_types = [type(h).__name__ for h in self.registry.handlers]
        assert "RateLimitErrorHandler" in handler_types
        assert "AuthenticationErrorHandler" in handler_types
        assert "NetworkErrorHandler" in handler_types
        assert "QuotaExceededErrorHandler" in handler_types

    def test_register_custom_handler(self):
        """Test registering a custom handler."""
        class CustomHandler(ErrorHandler):
            def can_handle(self, error, error_str):
                return "custom" in error_str.lower()

            def handle(self, error, error_str):
                return ErrorGuidance(
                    error_type="custom",
                    title="Custom Error",
                    description="Custom",
                    solutions=[]
                )

        initial_count = len(self.registry.handlers)
        self.registry.register(CustomHandler())
        assert len(self.registry.handlers) == initial_count + 1

    def test_handle_error_routes_to_correct_handler(self):
        """Test that errors are routed to the correct handler."""
        # Test rate limit error
        error = Exception("Error 429: rate_limit_error")
        guidance = self.registry.handle_error(error)
        assert guidance.error_type == "rate_limit"

        # Test authentication error
        error = Exception("Error 401: unauthorized")
        guidance = self.registry.handle_error(error)
        assert guidance.error_type == "authentication"

        # Test network error
        error = Exception("Connection timeout")
        guidance = self.registry.handle_error(error)
        assert guidance.error_type == "network"

    def test_handle_error_first_match_wins(self):
        """Test that first matching handler wins."""
        # Create error that could match multiple patterns
        error = Exception("network timeout")
        guidance = self.registry.handle_error(error)
        # Should match NetworkErrorHandler (which checks for both patterns)
        assert guidance.error_type == "network"

    def test_handle_error_fallback_to_generic(self):
        """Test fallback to generic error for unknown errors."""
        error = Exception("Some completely unknown error xyz123")
        guidance = self.registry.handle_error(error)

        assert guidance.error_type == "unknown"
        assert "Agent Error" in guidance.title
        assert len(guidance.solutions) > 0

    def test_handle_keyboard_interrupt(self):
        """Test special handling of KeyboardInterrupt."""
        error = KeyboardInterrupt()
        guidance = self.registry.handle_error(error)

        assert guidance.error_type == "interrupt"
        assert "Cancelled" in guidance.title
        assert guidance.severity == "warning"
        assert guidance.can_retry is False

    def test_handler_exception_does_not_break_registry(self):
        """Test that handler exceptions don't break the system."""
        class FaultyHandler(ErrorHandler):
            def can_handle(self, error, error_str):
                return True  # Always claims to handle

            def handle(self, error, error_str):
                raise Exception("Handler is broken!")

        # Insert faulty handler at beginning
        self.registry.handlers.insert(0, FaultyHandler())

        # Should still work by falling back to other handlers
        error = Exception("rate limit error")
        guidance = self.registry.handle_error(error)

        # Should have fallen through to rate limit handler
        assert guidance is not None
        assert guidance.error_type == "rate_limit"


class TestGlobalRegistry:
    """Test the global registry singleton."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_error_registry()

    def test_get_error_registry_returns_singleton(self):
        """Test that get_error_registry returns the same instance."""
        registry1 = get_error_registry()
        registry2 = get_error_registry()
        assert registry1 is registry2

    def test_reset_error_registry(self):
        """Test that reset creates a new registry."""
        registry1 = get_error_registry()
        reset_error_registry()
        registry2 = get_error_registry()
        assert registry1 is not registry2

    def test_global_registry_works_across_imports(self):
        """Test that global registry is consistent across calls."""
        from lobster.utils.error_handlers import get_error_registry

        registry1 = get_error_registry()
        registry1.register(Mock(spec=ErrorHandler))

        registry2 = get_error_registry()
        # Should be same instance with same handlers
        assert len(registry1.handlers) == len(registry2.handlers)


class TestErrorGuidanceMetadata:
    """Test error guidance metadata handling."""

    def test_metadata_stored_in_guidance(self):
        """Test that metadata is properly stored."""
        handler = RateLimitErrorHandler()
        error = Exception("Rate limit error with lots of details...")
        guidance = handler.handle(error, str(error))

        assert "original_error" in guidance.metadata
        assert "provider" in guidance.metadata
        assert guidance.metadata["provider"] == "anthropic"

    def test_original_error_truncated(self):
        """Test that very long errors are truncated in metadata."""
        handler = RateLimitErrorHandler()
        long_error = "x" * 500  # Very long error
        error = Exception(long_error)
        guidance = handler.handle(error, str(error))

        # Should be truncated to 200 chars
        assert len(guidance.metadata["original_error"]) == 200


class TestIntegration:
    """Integration tests for the complete error handling flow."""

    def setup_method(self):
        """Set up for integration tests."""
        reset_error_registry()

    def test_end_to_end_rate_limit_flow(self):
        """Test complete flow from error to guidance display."""
        # Simulate real Anthropic rate limit error
        error_message = (
            "Error code: 429 - {'type': 'error', 'error': "
            "{'type': 'rate_limit_error', 'message': 'This request would "
            "exceed your organization's maximum usage increase rate'}}"
        )
        error = Exception(error_message)

        registry = get_error_registry()
        guidance = registry.handle_error(error)

        # Verify complete guidance
        assert guidance.error_type == "rate_limit"
        assert "Rate Limit" in guidance.title
        assert len(guidance.solutions) >= 4
        assert "Anthropic" in guidance.solutions[1]  # Should mention requesting increase
        assert "Bedrock" in guidance.solutions[2]  # Should recommend AWS
        assert guidance.support_email == "info@omics-os.com"
        assert guidance.can_retry is True

    def test_multiple_error_types_in_sequence(self):
        """Test handling different error types in sequence."""
        registry = get_error_registry()

        # Test sequence of different errors
        errors = [
            Exception("429 rate limit"),
            Exception("401 unauthorized"),
            Exception("Connection timeout"),
            Exception("quota exceeded")
        ]

        expected_types = ["rate_limit", "authentication", "network", "quota"]

        for error, expected_type in zip(errors, expected_types):
            guidance = registry.handle_error(error)
            assert guidance.error_type == expected_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
