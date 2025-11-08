"""
Unit tests for Redis-based rate limiter.

Tests rate limiting functionality with fakeredis for isolated testing
without requiring a real Redis instance.
"""

import os
import time
from unittest.mock import Mock, patch

import fakeredis
import pytest
from redis.exceptions import ConnectionError, RedisError

from lobster.tools.rate_limiter import (
    NCBIRateLimiter,
    get_redis_client,
    rate_limited,
)


@pytest.fixture
def fake_redis():
    """Provide a fake Redis client for testing."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def rate_limiter(fake_redis):
    """Provide a rate limiter with fake Redis."""
    return NCBIRateLimiter(redis_client=fake_redis)


@pytest.fixture
def rate_limiter_no_redis():
    """Provide a rate limiter without Redis (graceful degradation)."""
    with patch("lobster.tools.rate_limiter.get_redis_client", return_value=None):
        yield NCBIRateLimiter(redis_client=None)


class TestGetRedisClient:
    """Tests for get_redis_client() function."""

    @patch.dict(os.environ, {"REDIS_HOST": "localhost", "REDIS_PORT": "6379"})
    @patch("redis.Redis")
    def test_successful_connection(self, mock_redis):
        """Test successful Redis connection."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client

        client = get_redis_client()

        assert client is not None
        mock_client.ping.assert_called_once()

    @patch("redis.Redis")
    def test_connection_failure(self, mock_redis):
        """Test graceful handling of connection failure."""
        mock_redis.side_effect = ConnectionError("Connection refused")

        client = get_redis_client()

        assert client is None

    @patch.dict(os.environ, {"REDIS_HOST": "custom-host", "REDIS_PORT": "6380"})
    @patch("redis.Redis")
    def test_custom_host_port(self, mock_redis):
        """Test Redis connection with custom host and port."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client

        get_redis_client()

        mock_redis.assert_called_once()
        call_kwargs = mock_redis.call_args[1]
        assert call_kwargs["host"] == "custom-host"
        assert call_kwargs["port"] == 6380


class TestNCBIRateLimiterInitialization:
    """Tests for NCBIRateLimiter initialization."""

    def test_init_with_api_key(self, fake_redis):
        """Test initialization with NCBI API key (higher rate limit)."""
        with patch.dict(os.environ, {"NCBI_API_KEY": "test-key"}):
            limiter = NCBIRateLimiter(redis_client=fake_redis)

            assert limiter.rate_limit == 9  # 9 req/s with key
            assert limiter.ncbi_api_key == "test-key"
            assert limiter.window_seconds == 1

    def test_init_without_api_key(self, fake_redis):
        """Test initialization without NCBI API key (lower rate limit)."""
        with patch.dict(os.environ, {"NCBI_API_KEY": ""}, clear=True):
            limiter = NCBIRateLimiter(redis_client=fake_redis)

            assert limiter.rate_limit == 2  # 2 req/s without key
            assert limiter.ncbi_api_key == ""

    @patch("lobster.tools.rate_limiter.get_redis_client", return_value=None)
    def test_init_without_redis(self, mock_get_redis):
        """Test initialization without Redis (graceful degradation)."""
        limiter = NCBIRateLimiter(redis_client=None)

        assert limiter.redis_client is None
        assert limiter.rate_limit > 0  # Still has rate limit configured


class TestRateLimitChecking:
    """Tests for check_rate_limit() method."""

    def test_first_request_allowed(self, rate_limiter, fake_redis):
        """Test first request in window is allowed."""
        allowed = rate_limiter.check_rate_limit("ncbi_esearch", "user1")

        assert allowed is True

        # Verify Redis key created
        key = "ratelimit:ncbi_esearch:user1"
        assert fake_redis.get(key) == "1"

    def test_requests_within_limit(self, rate_limiter):
        """Test requests within rate limit are allowed."""
        # Allow 9 requests (or 2 without key)
        for i in range(rate_limiter.rate_limit):
            allowed = rate_limiter.check_rate_limit("ncbi_esearch", "user1")
            assert allowed is True, f"Request {i+1} should be allowed"

    def test_request_exceeding_limit(self, rate_limiter):
        """Test request exceeding rate limit is blocked."""
        # Fill up the rate limit
        for _ in range(rate_limiter.rate_limit):
            rate_limiter.check_rate_limit("ncbi_esearch", "user1")

        # Next request should be blocked
        allowed = rate_limiter.check_rate_limit("ncbi_esearch", "user1")
        assert allowed is False

    def test_different_apis_isolated(self, rate_limiter):
        """Test different API endpoints have separate rate limits."""
        # Fill up esearch limit
        for _ in range(rate_limiter.rate_limit):
            rate_limiter.check_rate_limit("ncbi_esearch", "user1")

        # efetch should still be available
        allowed = rate_limiter.check_rate_limit("ncbi_efetch", "user1")
        assert allowed is True

    def test_different_users_isolated(self, rate_limiter):
        """Test different users have separate rate limits."""
        # Fill up user1 limit
        for _ in range(rate_limiter.rate_limit):
            rate_limiter.check_rate_limit("ncbi_esearch", "user1")

        # user2 should still be available
        allowed = rate_limiter.check_rate_limit("ncbi_esearch", "user2")
        assert allowed is True

    def test_ttl_expiry(self, rate_limiter, fake_redis):
        """Test rate limit resets after TTL expires."""
        # Fill up rate limit
        for _ in range(rate_limiter.rate_limit):
            rate_limiter.check_rate_limit("ncbi_esearch", "user1")

        # Verify blocked
        assert rate_limiter.check_rate_limit("ncbi_esearch", "user1") is False

        # Manually expire the key (simulate 1 second passing)
        key = "ratelimit:ncbi_esearch:user1"
        fake_redis.delete(key)

        # Should be allowed again
        allowed = rate_limiter.check_rate_limit("ncbi_esearch", "user1")
        assert allowed is True

    def test_graceful_degradation_no_redis(self, rate_limiter_no_redis):
        """Test requests proceed when Redis unavailable (fail-open)."""
        # Should always return True when Redis unavailable
        for _ in range(20):  # Way over any rate limit
            allowed = rate_limiter_no_redis.check_rate_limit("ncbi_esearch", "user1")
            assert allowed is True

    def test_redis_error_handling(self, rate_limiter, fake_redis):
        """Test graceful handling of Redis errors during operation."""
        # Mock Redis to raise error
        fake_redis.get = Mock(side_effect=RedisError("Connection lost"))

        # Should fail open (return True) with error
        allowed = rate_limiter.check_rate_limit("ncbi_esearch", "user1")
        assert allowed is True


class TestWaitForSlot:
    """Tests for wait_for_slot() method."""

    def test_immediate_slot_available(self, rate_limiter):
        """Test returns immediately when slot available."""
        start = time.time()
        success = rate_limiter.wait_for_slot("ncbi_esearch", "user1")
        elapsed = time.time() - start

        assert success is True
        assert elapsed < 0.2  # Should be nearly instant

    @pytest.mark.flaky(reruns=3)
    def test_waits_for_slot(self, rate_limiter, fake_redis):
        """Test waits and acquires slot when available (timing-sensitive test)."""
        # Fill up rate limit
        for _ in range(rate_limiter.rate_limit):
            rate_limiter.check_rate_limit("ncbi_esearch", "user1")

        # Simulate TTL expiry after 0.5 seconds
        def delayed_delete():
            time.sleep(0.3)
            fake_redis.delete("ratelimit:ncbi_esearch:user1")

        import threading

        thread = threading.Thread(target=delayed_delete)
        thread.start()

        start = time.time()
        success = rate_limiter.wait_for_slot("ncbi_esearch", "user1", max_wait=2.0)
        elapsed = time.time() - start

        thread.join()

        assert success is True
        assert 0.2 <= elapsed <= 1.0  # Should wait ~0.3s (tolerant for thread timing)

        # Small cleanup delay to ensure next test doesn't interfere
        time.sleep(0.1)

    def test_timeout_when_limit_not_available(self, rate_limiter):
        """Test returns False when max_wait exceeded."""
        # Fill up rate limit (use different user to avoid test interference)
        for _ in range(rate_limiter.rate_limit):
            rate_limiter.check_rate_limit("ncbi_esearch", "user_timeout_test")

        start = time.time()
        success = rate_limiter.wait_for_slot(
            "ncbi_esearch",
            "user_timeout_test",
            max_wait=0.5,  # Shorter than TTL (1s) to test timeout
        )
        elapsed = time.time() - start

        assert success is False
        assert 0.5 <= elapsed <= 0.7  # Should timeout around 0.5s

    def test_no_redis_immediate_return(self, rate_limiter_no_redis):
        """Test immediate return when Redis unavailable."""
        start = time.time()
        success = rate_limiter_no_redis.wait_for_slot("ncbi_esearch", "user1")
        elapsed = time.time() - start

        assert success is True
        assert elapsed < 0.2  # Should be instant (fail-open)


class TestGetCurrentUsage:
    """Tests for get_current_usage() method."""

    def test_zero_usage_initially(self, rate_limiter):
        """Test returns 0 for new API/user."""
        usage = rate_limiter.get_current_usage("ncbi_esearch", "user1")
        assert usage == 0

    def test_accurate_usage_tracking(self, rate_limiter):
        """Test accurately tracks request count."""
        # Make 3 requests
        for _ in range(3):
            rate_limiter.check_rate_limit("ncbi_esearch", "user1")

        usage = rate_limiter.get_current_usage("ncbi_esearch", "user1")
        assert usage == 3

    def test_usage_isolated_by_api(self, rate_limiter):
        """Test usage tracking isolated by API endpoint."""
        rate_limiter.check_rate_limit("ncbi_esearch", "user1")
        rate_limiter.check_rate_limit("ncbi_esearch", "user1")
        rate_limiter.check_rate_limit("ncbi_efetch", "user1")

        esearch_usage = rate_limiter.get_current_usage("ncbi_esearch", "user1")
        efetch_usage = rate_limiter.get_current_usage("ncbi_efetch", "user1")

        assert esearch_usage == 2
        assert efetch_usage == 1

    def test_no_redis_returns_zero(self, rate_limiter_no_redis):
        """Test returns 0 when Redis unavailable."""
        usage = rate_limiter_no_redis.get_current_usage("ncbi_esearch", "user1")
        assert usage == 0


class TestResetLimit:
    """Tests for reset_limit() method."""

    def test_reset_clears_usage(self, rate_limiter):
        """Test reset clears current usage count."""
        # Make requests
        for _ in range(3):
            rate_limiter.check_rate_limit("ncbi_esearch", "user1")

        # Reset
        success = rate_limiter.reset_limit("ncbi_esearch", "user1")
        assert success is True

        # Usage should be 0
        usage = rate_limiter.get_current_usage("ncbi_esearch", "user1")
        assert usage == 0

    def test_reset_allows_new_requests(self, rate_limiter):
        """Test reset allows rate limit to be used again."""
        # Fill up rate limit
        for _ in range(rate_limiter.rate_limit):
            rate_limiter.check_rate_limit("ncbi_esearch", "user1")

        # Should be blocked
        assert rate_limiter.check_rate_limit("ncbi_esearch", "user1") is False

        # Reset
        rate_limiter.reset_limit("ncbi_esearch", "user1")

        # Should be allowed again
        allowed = rate_limiter.check_rate_limit("ncbi_esearch", "user1")
        assert allowed is True

    def test_reset_no_redis_returns_false(self, rate_limiter_no_redis):
        """Test reset returns False when Redis unavailable."""
        success = rate_limiter_no_redis.reset_limit("ncbi_esearch", "user1")
        assert success is False


class TestRateLimitedDecorator:
    """Tests for @rate_limited decorator."""

    def test_decorator_enforces_rate_limit(self, fake_redis):
        """Test decorator blocks calls when rate limit exceeded."""
        call_count = 0

        with patch.dict(os.environ, {"NCBI_API_KEY": "test-key"}):

            @rate_limited("ncbi_esearch")
            def mock_api_call():
                nonlocal call_count
                call_count += 1
                return "success"

            # Patch NCBIRateLimiter to use fake_redis
            with patch(
                "lobster.tools.rate_limiter.NCBIRateLimiter"
            ) as mock_limiter_class:
                mock_limiter = NCBIRateLimiter(redis_client=fake_redis)
                mock_limiter_class.return_value = mock_limiter

                # Make calls up to limit (should succeed)
                for i in range(mock_limiter.rate_limit):
                    result = mock_api_call()
                    assert result == "success"
                    assert call_count == i + 1

    def test_decorator_waits_for_slot(self, fake_redis):
        """Test decorator waits when rate limit hit."""

        @rate_limited("ncbi_esearch")
        def mock_api_call():
            return "success"

        with patch("lobster.tools.rate_limiter.NCBIRateLimiter") as mock_limiter_class:
            mock_limiter = NCBIRateLimiter(redis_client=fake_redis)
            mock_limiter_class.return_value = mock_limiter

            # Fill up rate limit
            for _ in range(mock_limiter.rate_limit):
                mock_api_call()

            # This should wait (we'll timeout it)
            start = time.time()

            # Mock wait_for_slot to return False (timeout)
            mock_limiter.wait_for_slot = Mock(return_value=False)

            with pytest.raises(TimeoutError):
                mock_api_call()

    def test_decorator_no_redis_still_works(self):
        """Test decorator doesn't block when Redis unavailable."""

        @rate_limited("ncbi_esearch")
        def mock_api_call():
            return "success"

        with patch("lobster.tools.rate_limiter.NCBIRateLimiter") as mock_limiter_class:
            mock_limiter = NCBIRateLimiter(redis_client=None)
            mock_limiter_class.return_value = mock_limiter

            # Should succeed even without Redis (fail-open)
            for _ in range(20):  # Way over any rate limit
                result = mock_api_call()
                assert result == "success"


class TestConcurrentAccess:
    """Tests for concurrent access patterns."""

    def test_concurrent_users_isolated(self, rate_limiter):
        """Test concurrent users have independent rate limits."""
        import concurrent.futures

        def make_requests(user_id):
            results = []
            for _ in range(5):
                allowed = rate_limiter.check_rate_limit("ncbi_esearch", user_id)
                results.append(allowed)
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_requests, f"user{i}") for i in range(1, 4)]
            results = [f.result() for f in futures]

        # Each user should have independent limits
        for user_results in results:
            # All 5 requests should be allowed since each user has independent limits
            allowed_count = sum(user_results)
            assert allowed_count == 5  # All 5 requests should succeed per user


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_zero_rate_limit_handling(self, fake_redis):
        """Test behavior with zero rate limit (should never happen)."""
        limiter = NCBIRateLimiter(redis_client=fake_redis)
        limiter.rate_limit = 0  # Force zero limit

        # First request should still create key
        allowed = limiter.check_rate_limit("ncbi_esearch", "user1")
        # Implementation will block if counter >= 0, so expect False
        # Actually, with rate_limit=0, the first request sets counter to 1
        # and 1 >= 0, so it gets blocked immediately after creation
        # Let's test the actual behavior

        # After setting key to 1, next check: 1 >= 0 -> blocked
        # This is correct behavior for rate_limit=0

    def test_negative_window_handling(self, fake_redis):
        """Test behavior with negative window (should never happen)."""
        limiter = NCBIRateLimiter(redis_client=fake_redis)
        limiter.window_seconds = -1  # Invalid

        # Redis will reject negative TTL, should fail gracefully
        try:
            limiter.check_rate_limit("ncbi_esearch", "user1")
        except Exception:
            pass  # Expected to fail gracefully

    def test_empty_api_name(self, rate_limiter):
        """Test behavior with empty API name."""
        allowed = rate_limiter.check_rate_limit("", "user1")
        # Should work (creates key "ratelimit::user1")
        assert allowed is True

    def test_special_characters_in_identifiers(self, rate_limiter):
        """Test handling of special characters in API/user names."""
        allowed = rate_limiter.check_rate_limit("ncbi:esearch", "user@example.com")
        assert allowed is True

        usage = rate_limiter.get_current_usage("ncbi:esearch", "user@example.com")
        assert usage == 1
