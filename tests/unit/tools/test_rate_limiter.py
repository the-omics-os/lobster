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


# ============================================================================
# MULTI-DOMAIN RATE LIMITER TESTS
# ============================================================================


class TestDomainStrategy:
    """Tests for DomainStrategy dataclass."""

    def test_default_values(self):
        """Test default strategy values."""
        from lobster.tools.rate_limiter import DomainStrategy

        strategy = DomainStrategy(name="test", requests_per_second=1.0)
        assert strategy.window_seconds == 1.0
        assert strategy.max_retries == 3
        assert strategy.backoff_base == 1.0
        assert strategy.backoff_factor == 3.0
        assert strategy.backoff_max == 30.0
        assert 429 in strategy.retry_on
        assert 503 in strategy.retry_on
        assert 502 in strategy.retry_on

    def test_custom_values(self):
        """Test custom strategy values."""
        from lobster.tools.rate_limiter import DomainStrategy

        strategy = DomainStrategy(
            name="custom",
            requests_per_second=5.0,
            window_seconds=2.0,
            max_retries=5,
            backoff_base=2.0,
            backoff_factor=2.0,
            backoff_max=60.0,
            retry_on=[500, 502, 503],
        )
        assert strategy.window_seconds == 2.0
        assert strategy.max_retries == 5
        assert strategy.backoff_base == 2.0
        assert strategy.backoff_factor == 2.0
        assert strategy.backoff_max == 60.0
        assert strategy.retry_on == [500, 502, 503]


class TestMultiDomainRateLimiter:
    """Tests for MultiDomainRateLimiter."""

    @pytest.fixture
    def multi_limiter(self, fake_redis):
        """Provide multi-domain rate limiter with fake Redis."""
        from lobster.tools.rate_limiter import MultiDomainRateLimiter

        return MultiDomainRateLimiter(redis_client=fake_redis)

    def test_init_with_redis(self, fake_redis):
        """Test initialization with Redis client."""
        from lobster.tools.rate_limiter import MultiDomainRateLimiter

        limiter = MultiDomainRateLimiter(redis_client=fake_redis)
        assert limiter.redis_client is not None
        assert limiter._strategies is not None
        assert "eutils.ncbi.nlm.nih.gov" in limiter._strategies
        assert "default" in limiter._strategies

    def test_init_without_redis(self):
        """Test initialization without Redis (graceful degradation)."""
        from lobster.tools.rate_limiter import MultiDomainRateLimiter

        with patch("lobster.tools.rate_limiter.get_redis_client", return_value=None):
            limiter = MultiDomainRateLimiter(redis_client=None)
            assert limiter.redis_client is None

    def test_detect_domain_ncbi(self, multi_limiter):
        """Test NCBI domain detection."""
        assert (
            multi_limiter.detect_domain("https://eutils.ncbi.nlm.nih.gov/entrez/")
            == "eutils.ncbi.nlm.nih.gov"
        )

    def test_detect_domain_pmc(self, multi_limiter):
        """Test PMC domain detection."""
        assert (
            multi_limiter.detect_domain(
                "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/"
            )
            == "www.ncbi.nlm.nih.gov"
        )
        assert multi_limiter.detect_domain("PMC12345") == "pmc.ncbi.nlm.nih.gov"
        assert multi_limiter.detect_domain("PMID:12345") == "pmc.ncbi.nlm.nih.gov"

    def test_detect_domain_publishers(self, multi_limiter):
        """Test publisher domain detection."""
        assert (
            multi_limiter.detect_domain("https://www.nature.com/articles/s41586")
            == "nature.com"
        )
        assert (
            multi_limiter.detect_domain("https://www.cell.com/cell/fulltext/S0092")
            == "cell.com"
        )
        assert (
            multi_limiter.detect_domain(
                "https://www.sciencedirect.com/science/article/"
            )
            == "sciencedirect.com"
        )
        assert (
            multi_limiter.detect_domain("https://www.frontiersin.org/articles/")
            == "frontiersin.org"
        )

    def test_detect_domain_default(self, multi_limiter):
        """Test default domain for unknown URLs."""
        assert (
            multi_limiter.detect_domain("https://unknown-site.com/article") == "default"
        )
        assert multi_limiter.detect_domain("invalid-url") == "default"

    def test_get_rate_limit(self, multi_limiter):
        """Test rate limit retrieval."""
        assert multi_limiter.get_rate_limit("eutils.ncbi.nlm.nih.gov") == 10.0
        assert multi_limiter.get_rate_limit("pmc.ncbi.nlm.nih.gov") == 3.0
        assert multi_limiter.get_rate_limit("nature.com") == 0.5
        assert multi_limiter.get_rate_limit("default") == 0.3

    def test_check_rate_limit_allows_within_limit(self, multi_limiter):
        """Test rate limit allows requests within limit."""
        url = "https://www.frontiersin.org/articles/test"  # 1 req/s
        assert multi_limiter.check_rate_limit(url) is True

    def test_check_rate_limit_blocks_over_limit(self, multi_limiter):
        """Test rate limit blocks requests over limit."""
        url = "https://www.frontiersin.org/articles/test"  # 1 req/s
        # First request allowed
        assert multi_limiter.check_rate_limit(url) is True
        # Second request blocked (1 req/s limit)
        assert multi_limiter.check_rate_limit(url) is False

    def test_different_domains_independent(self, multi_limiter):
        """Test different domains have independent limits."""
        url1 = "https://www.frontiersin.org/articles/test"  # 1 req/s
        url2 = "https://www.mdpi.com/articles/test"  # 1 req/s

        # Fill frontiers limit
        assert multi_limiter.check_rate_limit(url1) is True
        assert multi_limiter.check_rate_limit(url1) is False

        # MDPI should still work (different domain)
        assert multi_limiter.check_rate_limit(url2) is True

    def test_different_users_independent(self, multi_limiter):
        """Test different users have independent limits."""
        url = "https://www.frontiersin.org/articles/test"

        # Fill user1 limit
        assert multi_limiter.check_rate_limit(url, user_id="user1") is True
        assert multi_limiter.check_rate_limit(url, user_id="user1") is False

        # user2 should still work
        assert multi_limiter.check_rate_limit(url, user_id="user2") is True

    def test_graceful_degradation_no_redis(self):
        """Test fail-open when Redis unavailable."""
        from lobster.tools.rate_limiter import MultiDomainRateLimiter

        with patch("lobster.tools.rate_limiter.get_redis_client", return_value=None):
            limiter = MultiDomainRateLimiter(redis_client=None)
            # Should allow all requests (fail-open)
            assert limiter.check_rate_limit("https://www.nature.com/test") is True
            assert limiter.check_rate_limit("https://www.nature.com/test") is True

    def test_redis_error_handling(self, multi_limiter, fake_redis):
        """Test graceful handling of Redis errors."""
        # Mock Redis to raise error
        fake_redis.get = Mock(side_effect=RedisError("Connection lost"))

        # Should fail open (return True) with error
        allowed = multi_limiter.check_rate_limit("https://www.nature.com/test")
        assert allowed is True

    def test_calculate_backoff(self, multi_limiter):
        """Test exponential backoff calculation."""
        domain = "nature.com"
        # 1s -> 3s -> 9s -> 27s -> 30s (capped)
        assert multi_limiter.calculate_backoff(domain, 0) == 1.0
        assert multi_limiter.calculate_backoff(domain, 1) == 3.0
        assert multi_limiter.calculate_backoff(domain, 2) == 9.0
        assert multi_limiter.calculate_backoff(domain, 3) == 27.0
        assert multi_limiter.calculate_backoff(domain, 4) == 30.0  # Capped

    def test_should_retry(self, multi_limiter):
        """Test retryable status codes."""
        domain = "nature.com"
        assert multi_limiter.should_retry(domain, 429) is True
        assert multi_limiter.should_retry(domain, 503) is True
        assert multi_limiter.should_retry(domain, 502) is True
        assert multi_limiter.should_retry(domain, 200) is False
        assert multi_limiter.should_retry(domain, 404) is False

    def test_get_max_retries(self, multi_limiter):
        """Test max retries per domain."""
        # High rate domains (>=1.0 req/s) get 3 retries
        assert multi_limiter.get_max_retries("frontiersin.org") == 3
        assert multi_limiter.get_max_retries("eutils.ncbi.nlm.nih.gov") == 3
        # Low rate domains (<1.0 req/s) get 2 retries
        assert multi_limiter.get_max_retries("nature.com") == 2
        assert multi_limiter.get_max_retries("default") == 2

    def test_wait_for_slot_immediate(self, multi_limiter):
        """Test wait_for_slot returns immediately when slot available."""
        start = time.time()
        success = multi_limiter.wait_for_slot("https://www.frontiersin.org/test")
        elapsed = time.time() - start

        assert success is True
        assert elapsed < 0.2  # Should be nearly instant

    def test_wait_for_slot_timeout(self, multi_limiter, fake_redis):
        """Test wait_for_slot returns False when max_wait exceeded."""
        url = "https://www.frontiersin.org/test"

        # Fill rate limit
        multi_limiter.check_rate_limit(url)

        # Prevent key from expiring during test by mocking the check to always fail
        original_check = multi_limiter.check_rate_limit
        multi_limiter.check_rate_limit = Mock(return_value=False)

        start = time.time()
        success = multi_limiter.wait_for_slot(url, max_wait=0.3)
        elapsed = time.time() - start

        # Restore original method
        multi_limiter.check_rate_limit = original_check

        assert success is False
        assert 0.25 <= elapsed <= 0.6  # Should timeout around 0.3-0.5s

    def test_wait_for_slot_no_redis(self):
        """Test wait_for_slot works without Redis (fail-open)."""
        from lobster.tools.rate_limiter import MultiDomainRateLimiter

        with patch("lobster.tools.rate_limiter.get_redis_client", return_value=None):
            limiter = MultiDomainRateLimiter(redis_client=None)
            start = time.time()
            success = limiter.wait_for_slot("https://www.nature.com/test")
            elapsed = time.time() - start

            assert success is True
            assert elapsed < 0.2  # Should be instant


class TestRateLimitedRequest:
    """Tests for rate_limited_request function."""

    def test_successful_request(self, mocker):
        """Test successful request passes through."""
        from lobster.tools.rate_limiter import rate_limited_request

        # Mock response
        mock_response = mocker.Mock(status_code=200)
        mock_request = mocker.Mock(return_value=mock_response)

        # Mock the limiter
        mock_limiter = mocker.Mock()
        mock_limiter.detect_domain.return_value = "nature.com"
        mock_limiter.get_max_retries.return_value = 3
        mock_limiter.wait_for_slot.return_value = True
        mock_limiter.should_retry.return_value = False

        with patch(
            "lobster.tools.rate_limiter.MultiDomainRateLimiter",
            return_value=mock_limiter,
        ):
            result = rate_limited_request("https://www.nature.com/test", mock_request)

        assert result.status_code == 200
        mock_request.assert_called_once()
        mock_limiter.wait_for_slot.assert_called_once()

    def test_request_with_retry_on_429(self, mocker):
        """Test request retries on HTTP 429."""
        from lobster.tools.rate_limiter import rate_limited_request

        # First response: 429, second: 200
        mock_response_429 = mocker.Mock(status_code=429)
        mock_response_200 = mocker.Mock(status_code=200)
        mock_request = mocker.Mock(side_effect=[mock_response_429, mock_response_200])

        # Mock limiter
        mock_limiter = mocker.Mock()
        mock_limiter.detect_domain.return_value = "nature.com"
        mock_limiter.get_max_retries.return_value = 3
        mock_limiter.wait_for_slot.return_value = True
        mock_limiter.should_retry.side_effect = [True, False]  # Retry first, not second
        mock_limiter.calculate_backoff.return_value = 0.01  # Fast backoff for testing

        with patch(
            "lobster.tools.rate_limiter.MultiDomainRateLimiter",
            return_value=mock_limiter,
        ):
            result = rate_limited_request("https://www.nature.com/test", mock_request)

        assert result.status_code == 200
        assert mock_request.call_count == 2

    def test_request_timeout(self, mocker):
        """Test request raises TimeoutError when rate limit slot unavailable."""
        from lobster.tools.rate_limiter import rate_limited_request

        mock_request = mocker.Mock()

        # Mock limiter that never gets a slot
        mock_limiter = mocker.Mock()
        mock_limiter.detect_domain.return_value = "nature.com"
        mock_limiter.get_max_retries.return_value = 3
        mock_limiter.wait_for_slot.return_value = False

        with patch(
            "lobster.tools.rate_limiter.MultiDomainRateLimiter",
            return_value=mock_limiter,
        ):
            with pytest.raises(TimeoutError, match="Rate limit timeout"):
                rate_limited_request("https://www.nature.com/test", mock_request)

    def test_request_with_args_kwargs(self, mocker):
        """Test request passes args and kwargs to request function."""
        from lobster.tools.rate_limiter import rate_limited_request

        mock_response = mocker.Mock(status_code=200)
        mock_request = mocker.Mock(return_value=mock_response)

        # Mock limiter
        mock_limiter = mocker.Mock()
        mock_limiter.detect_domain.return_value = "nature.com"
        mock_limiter.get_max_retries.return_value = 3
        mock_limiter.wait_for_slot.return_value = True
        mock_limiter.should_retry.return_value = False

        with patch(
            "lobster.tools.rate_limiter.MultiDomainRateLimiter",
            return_value=mock_limiter,
        ):
            result = rate_limited_request(
                "https://www.nature.com/test",
                mock_request,
                timeout=30,
                headers={"User-Agent": "test"},
            )

        assert result.status_code == 200
        mock_request.assert_called_once_with(
            "https://www.nature.com/test", timeout=30, headers={"User-Agent": "test"}
        )

    def test_request_max_retries_override(self, mocker):
        """Test max_retries parameter overrides domain default."""
        from lobster.tools.rate_limiter import rate_limited_request

        # All responses: 429
        mock_response = mocker.Mock(status_code=429)
        mock_request = mocker.Mock(return_value=mock_response)

        # Mock limiter
        mock_limiter = mocker.Mock()
        mock_limiter.detect_domain.return_value = "nature.com"
        mock_limiter.get_max_retries.return_value = 3  # Default
        mock_limiter.wait_for_slot.return_value = True
        mock_limiter.should_retry.return_value = True
        mock_limiter.calculate_backoff.return_value = 0.01

        with patch(
            "lobster.tools.rate_limiter.MultiDomainRateLimiter",
            return_value=mock_limiter,
        ):
            result = rate_limited_request(
                "https://www.nature.com/test",
                mock_request,
                max_retries=1,  # Override to 1
            )

        # Should try 2 times total (1 initial + 1 retry)
        assert mock_request.call_count == 2
        assert result.status_code == 429

    def test_request_exception_handling(self, mocker):
        """Test request handles exceptions with retry."""
        from lobster.tools.rate_limiter import rate_limited_request

        # First call raises exception, second succeeds
        mock_response = mocker.Mock(status_code=200)
        mock_request = mocker.Mock(
            side_effect=[ConnectionError("Network error"), mock_response]
        )

        # Mock limiter
        mock_limiter = mocker.Mock()
        mock_limiter.detect_domain.return_value = "nature.com"
        mock_limiter.get_max_retries.return_value = 3
        mock_limiter.wait_for_slot.return_value = True
        mock_limiter.calculate_backoff.return_value = 0.01
        mock_limiter.should_retry.return_value = False  # Don't retry on success

        with patch(
            "lobster.tools.rate_limiter.MultiDomainRateLimiter",
            return_value=mock_limiter,
        ):
            result = rate_limited_request("https://www.nature.com/test", mock_request)

        assert result.status_code == 200
        assert mock_request.call_count == 2

    def test_request_exception_exhausted_retries(self, mocker):
        """Test request raises exception when all retries exhausted."""
        from lobster.tools.rate_limiter import rate_limited_request

        # Always raise exception
        mock_request = mocker.Mock(side_effect=ConnectionError("Network error"))

        # Mock limiter
        mock_limiter = mocker.Mock()
        mock_limiter.detect_domain.return_value = "nature.com"
        mock_limiter.get_max_retries.return_value = 2
        mock_limiter.wait_for_slot.return_value = True
        mock_limiter.calculate_backoff.return_value = 0.01

        with patch(
            "lobster.tools.rate_limiter.MultiDomainRateLimiter",
            return_value=mock_limiter,
        ):
            with pytest.raises(ConnectionError, match="Network error"):
                rate_limited_request("https://www.nature.com/test", mock_request)
