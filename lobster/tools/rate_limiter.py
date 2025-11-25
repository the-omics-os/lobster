"""
Redis-based rate limiter for NCBI API endpoints.

This module provides production-grade distributed rate limiting with graceful
degradation for NCBI E-utilities API calls. It ensures compliance with NCBI
rate limits (3 req/s without key, 10 req/s with key) across multiple users
and processes.
"""

import os
import time
from functools import wraps
from typing import Optional

import redis
from redis.exceptions import ConnectionError, RedisError

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Global flag to ensure we only warn about Redis unavailability once
_REDIS_WARNING_SHOWN = False


def get_redis_client() -> Optional[redis.Redis]:
    """
    Get Redis client with health check and graceful degradation.

    Returns None if Redis is unavailable - allows system to continue
    operating (with warning) rather than failing completely.

    Returns:
        Redis client if available, None otherwise
    """
    global _REDIS_WARNING_SHOWN

    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=0,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )

        # Health check
        client.ping()
        logger.info("✓ Redis connection established for rate limiting")
        return client

    except ConnectionError as e:
        # Only show warning once during application startup
        if not _REDIS_WARNING_SHOWN:
            logger.warning(
                "⚠️  Redis unavailable - rate limiting disabled. "
                "For production use, start Redis with: docker-compose up -d redis"
            )
            _REDIS_WARNING_SHOWN = True
        return None
    except Exception as e:
        if not _REDIS_WARNING_SHOWN:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            _REDIS_WARNING_SHOWN = True
        return None


class NCBIRateLimiter:
    """
    Rate limiter for NCBI API endpoints using Redis sliding window.

    Implements distributed rate limiting with automatic expiry and graceful
    degradation. If Redis is unavailable, the system logs a warning but
    continues operating (fail-open design).

    Attributes:
        redis_client: Redis client instance (None if unavailable)
        ncbi_api_key: NCBI API key from environment
        rate_limit: Requests per second limit (9 with key, 2 without)
        window_seconds: Time window for rate limiting (1 second)
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize rate limiter with optional Redis client.

        Args:
            redis_client: Optional Redis client (creates new if not provided)
        """
        self.redis_client = (
            redis_client if redis_client is not None else get_redis_client()
        )

        # NCBI limits: 3 req/s without key, 10 req/s with key
        # Use conservative limits to avoid accidental bans
        self.ncbi_api_key = os.getenv("NCBI_API_KEY")
        self.rate_limit = 9 if self.ncbi_api_key else 2
        self.window_seconds = 1

    def check_rate_limit(self, api_name: str, user_id: str = "default") -> bool:
        """
        Check if request is within rate limit.

        Uses Redis sliding window algorithm with automatic expiry.
        If Redis is unavailable, returns True with warning (fail-open).

        Args:
            api_name: API endpoint name (e.g., "ncbi_esearch", "ncbi_efetch")
            user_id: User identifier for per-user rate limiting (default: "default")

        Returns:
            True if request should be allowed, False if rate limit exceeded
        """
        if self.redis_client is None:
            # Redis unavailable - allow request silently (warning already shown at startup)
            return True  # Fail open - risky but better than blocking all requests

        try:
            key = f"ratelimit:{api_name}:{user_id}"
            current = self.redis_client.get(key)

            if current is None:
                # First request in this 1-second window
                self.redis_client.setex(key, self.window_seconds, 1)
                return True

            current_count = int(current)
            if current_count >= self.rate_limit:
                logger.warning(
                    f"Rate limit exceeded for {api_name} by {user_id}: "
                    f"{current_count}/{self.rate_limit} requests"
                )
                return False

            # Increment counter
            self.redis_client.incr(key)
            return True

        except RedisError as e:
            # Redis error during operation - fail open with warning
            logger.error(f"Redis error during rate limiting: {e}")
            return True
        except Exception as e:
            # Unexpected error - fail open
            logger.error(f"Unexpected error in rate limiter: {e}")
            return True

    def wait_for_slot(
        self, api_name: str, user_id: str = "default", max_wait: float = 10.0
    ) -> bool:
        """
        Block until a rate limit slot is available.

        Polls the rate limiter every 100ms until a slot opens up or
        max_wait time is exceeded.

        Args:
            api_name: API endpoint name
            user_id: User identifier for per-user rate limiting
            max_wait: Maximum time to wait in seconds (default: 10.0)

        Returns:
            True if slot acquired, False if max_wait exceeded
        """
        start_time = time.time()
        wait_count = 0

        while not self.check_rate_limit(api_name, user_id):
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                logger.error(
                    f"Rate limit wait timeout for {api_name} after {elapsed:.1f}s"
                )
                return False

            wait_count += 1
            if wait_count % 10 == 0:  # Log every 1 second
                logger.debug(
                    f"Waiting for rate limit slot ({wait_count * 0.1:.1f}s elapsed)"
                )

            time.sleep(0.1)  # 100ms backoff

        return True

    def get_current_usage(self, api_name: str, user_id: str = "default") -> int:
        """
        Get current request count in the rate limit window.

        Args:
            api_name: API endpoint name
            user_id: User identifier

        Returns:
            Current request count (0 if Redis unavailable)
        """
        if self.redis_client is None:
            return 0

        try:
            key = f"ratelimit:{api_name}:{user_id}"
            current = self.redis_client.get(key)
            return int(current) if current else 0
        except Exception as e:
            logger.error(f"Error getting current usage: {e}")
            return 0

    def reset_limit(self, api_name: str, user_id: str = "default") -> bool:
        """
        Reset rate limit counter for testing purposes.

        Args:
            api_name: API endpoint name
            user_id: User identifier

        Returns:
            True if reset successful, False otherwise
        """
        if self.redis_client is None:
            return False

        try:
            key = f"ratelimit:{api_name}:{user_id}"
            self.redis_client.delete(key)
            logger.debug(f"Reset rate limit for {api_name}:{user_id}")
            return True
        except Exception as e:
            logger.error(f"Error resetting rate limit: {e}")
            return False


def rate_limited(api_name: str):
    """
    Decorator to enforce rate limiting on API calls.

    Automatically waits for a rate limit slot before executing the
    decorated function. Uses default user_id="default" for shared
    rate limiting across all users.

    Args:
        api_name: API endpoint name (e.g., "ncbi_esearch")

    Example:
        @rate_limited("ncbi_esearch")
        def search_pubmed(query: str):
            # API call here
            pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rate_limiter = NCBIRateLimiter()
            if not rate_limiter.wait_for_slot(api_name):
                raise TimeoutError(
                    f"Rate limit wait timeout for {api_name}. "
                    "Too many concurrent requests."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# MULTI-DOMAIN RATE LIMITING (v2.0)
# ============================================================================

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
import urllib.parse

# Domain-specific rate limits (requests per second)
DOMAIN_RATE_LIMITS = {
    "eutils.ncbi.nlm.nih.gov": 10.0,  # NCBI with API key
    "www.ncbi.nlm.nih.gov": 3.0,       # PMC web
    "pmc.ncbi.nlm.nih.gov": 3.0,
    "europepmc.org": 2.0,
    "frontiersin.org": 1.0,
    "mdpi.com": 1.0,
    "peerj.com": 1.0,
    "nature.com": 0.5,
    "cell.com": 0.5,
    "elsevier.com": 0.5,
    "sciencedirect.com": 0.5,
    "default": 0.3,
}


@dataclass
class DomainStrategy:
    """Rate limit strategy for a specific domain."""
    name: str
    requests_per_second: float
    window_seconds: float = 1.0
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_factor: float = 3.0  # 1s -> 3s -> 9s -> 27s
    backoff_max: float = 30.0
    retry_on: List[int] = field(default_factory=lambda: [429, 503, 502])


class MultiDomainRateLimiter:
    """
    Multi-domain rate limiter with exponential backoff.

    Extends NCBIRateLimiter with:
    - Domain-specific rate limits (PMC: 3/s, publishers: 0.5/s)
    - Exponential backoff with retry (1s -> 3s -> 9s -> 27s)
    - Automatic domain detection from URLs
    - Graceful degradation (fail-open if Redis unavailable)
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client if redis_client is not None else get_redis_client()
        self.ncbi_api_key = os.getenv("NCBI_API_KEY")
        self._strategies = self._build_strategies()

    def _build_strategies(self) -> Dict[str, DomainStrategy]:
        """Build domain strategies from DOMAIN_RATE_LIMITS."""
        strategies = {}
        for domain, rate in DOMAIN_RATE_LIMITS.items():
            strategies[domain] = DomainStrategy(
                name=domain,
                requests_per_second=rate,
                max_retries=3 if rate >= 1.0 else 2,
            )
        return strategies

    def detect_domain(self, url: str) -> str:
        """
        Detect rate limit domain from URL.

        Args:
            url: Full URL or identifier (PMC12345, PMID:12345)

        Returns:
            Domain key for rate limiting
        """
        # Handle identifiers
        if url.startswith("PMC") or url.startswith("PMID:"):
            return "pmc.ncbi.nlm.nih.gov"

        try:
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname or ""

            # Check for exact matches
            if hostname in DOMAIN_RATE_LIMITS:
                return hostname

            # Check for partial matches
            for domain in DOMAIN_RATE_LIMITS:
                if domain != "default" and domain in hostname:
                    return domain

            return "default"
        except Exception:
            return "default"

    def get_rate_limit(self, domain: str) -> float:
        """Get rate limit for domain (requests per second)."""
        strategy = self._strategies.get(domain, self._strategies["default"])
        return strategy.requests_per_second

    def check_rate_limit(
        self,
        url_or_domain: str,
        user_id: str = "default"
    ) -> bool:
        """
        Check if request is within rate limit.

        Args:
            url_or_domain: URL or domain key
            user_id: User identifier for per-user limiting

        Returns:
            True if request allowed, False if blocked
        """
        if self.redis_client is None:
            return True  # Fail open

        domain = self.detect_domain(url_or_domain)
        rate_limit = int(self.get_rate_limit(domain))

        try:
            key = f"ratelimit:multi:{domain}:{user_id}"
            current = self.redis_client.get(key)

            if current is None:
                self.redis_client.setex(key, 1, 1)
                return True

            current_count = int(current)
            if current_count >= rate_limit:
                logger.debug(f"Rate limit hit for {domain}: {current_count}/{rate_limit}")
                return False

            self.redis_client.incr(key)
            return True

        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            return True  # Fail open

    def wait_for_slot(
        self,
        url_or_domain: str,
        user_id: str = "default",
        max_wait: float = 30.0
    ) -> bool:
        """
        Block until rate limit slot available.

        Args:
            url_or_domain: URL or domain key
            user_id: User identifier
            max_wait: Maximum wait time in seconds

        Returns:
            True if slot acquired, False if timeout
        """
        start_time = time.time()
        domain = self.detect_domain(url_or_domain)

        while not self.check_rate_limit(url_or_domain, user_id):
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                logger.warning(f"Rate limit timeout for {domain} after {elapsed:.1f}s")
                return False

            # Sleep based on domain rate (faster domains = shorter sleep)
            rate = self.get_rate_limit(domain)
            sleep_time = min(1.0 / rate, 0.5)  # At most 0.5s
            time.sleep(sleep_time)

        return True

    def calculate_backoff(self, domain: str, attempt: int) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            domain: Domain key
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        strategy = self._strategies.get(domain, self._strategies["default"])
        delay = strategy.backoff_base * (strategy.backoff_factor ** attempt)
        return min(delay, strategy.backoff_max)

    def should_retry(self, domain: str, status_code: int) -> bool:
        """
        Check if HTTP status code should trigger retry.

        Args:
            domain: Domain key
            status_code: HTTP status code

        Returns:
            True if should retry
        """
        strategy = self._strategies.get(domain, self._strategies["default"])
        return status_code in strategy.retry_on

    def get_max_retries(self, domain: str) -> int:
        """Get maximum retry attempts for domain."""
        strategy = self._strategies.get(domain, self._strategies["default"])
        return strategy.max_retries


def rate_limited_request(
    url: str,
    request_func: Callable,
    *args,
    max_retries: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Execute HTTP request with rate limiting and exponential backoff.

    Args:
        url: Request URL
        request_func: Function to execute (e.g., requests.get)
        *args, **kwargs: Passed to request_func
        max_retries: Override max retries (uses domain default if None)

    Returns:
        Response from request_func

    Raises:
        Exception: If all retries exhausted

    Example:
        response = rate_limited_request(
            "https://www.nature.com/articles/123",
            requests.get,
            timeout=30
        )
    """
    limiter = MultiDomainRateLimiter()
    domain = limiter.detect_domain(url)
    retries = max_retries if max_retries is not None else limiter.get_max_retries(domain)

    last_error = None

    for attempt in range(retries + 1):
        # Wait for rate limit slot
        if not limiter.wait_for_slot(url, max_wait=30.0):
            raise TimeoutError(f"Rate limit timeout for {domain}")

        try:
            response = request_func(url, *args, **kwargs)

            # Check for retryable status codes
            if hasattr(response, "status_code"):
                if limiter.should_retry(domain, response.status_code):
                    if attempt < retries:
                        backoff = limiter.calculate_backoff(domain, attempt)
                        logger.warning(
                            f"HTTP {response.status_code} for {domain}, "
                            f"retry {attempt + 1}/{retries} after {backoff:.1f}s"
                        )
                        time.sleep(backoff)
                        continue
                    else:
                        logger.error(
                            f"Max retries ({retries}) exhausted for {domain}: "
                            f"HTTP {response.status_code}"
                        )

            return response

        except Exception as e:
            last_error = e
            if attempt < retries:
                backoff = limiter.calculate_backoff(domain, attempt)
                logger.warning(
                    f"Request error for {domain}: {e}, "
                    f"retry {attempt + 1}/{retries} after {backoff:.1f}s"
                )
                time.sleep(backoff)
            else:
                raise

    if last_error:
        raise last_error
