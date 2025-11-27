# Redis Rate Limiter Architecture

This document describes the Redis-based rate limiting implementation for NCBI API endpoints in Lobster AI.

## Overview

The rate limiter ensures compliance with NCBI API limits (3 req/s without key, 10 req/s with key) across all usage scenarios:

- **Interactive**: `lobster chat` (single/multiple sessions)
- **Non-interactive**: `lobster query` (single/multiple instances)
- **Programmatic**: `import lobster` in Python scripts

## Architecture

### Connection Pool Pattern

The rate limiter uses `redis.ConnectionPool` instead of individual Redis clients:

```
┌─────────────────────────────────────────────────────────────┐
│                       Process                                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  Provider1  │   │  Provider2  │   │  Provider3  │       │
│  │  (Limiter)  │   │  (Limiter)  │   │  (Limiter)  │       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Shared Redis ConnectionPool               │   │
│  │  (thread-safe, health_check_interval=30s)           │   │
│  └─────────────────────────┬───────────────────────────┘   │
└──────────────────────────────┼──────────────────────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │   Redis Server   │
                     │ (rate limit keys │
                     │   with TTL)      │
                     └──────────────────┘
```

### Key Features

| Feature | Implementation |
|---------|----------------|
| Thread safety | Double-checked locking with `threading.Lock` |
| Stale connection recovery | `health_check_interval=30` validates connections |
| Graceful degradation | Falls back to fail-open if Redis unavailable |
| Test isolation | `reset_redis_pool()` for clean test state |
| Cross-process coordination | Redis keys with TTL (handled by Redis itself) |

## Code Location

| File | Purpose |
|------|---------|
| `lobster/tools/rate_limiter.py` | Connection pool, rate limiters (`NCBIRateLimiter`, `MultiDomainRateLimiter`) |
| `lobster/tools/providers/pubmed_provider.py` | Uses `NCBIRateLimiter` for NCBI API calls |
| `lobster/services/orchestration/publication_processing_service.py` | Reuses provider via lazy property |

## Usage Guidelines

### For New Providers

When creating providers that make external API calls:

1. **Use the shared rate limiter**:
   ```python
   from lobster.tools.rate_limiter import NCBIRateLimiter

   class MyProvider:
       def __init__(self):
           self._rate_limiter = NCBIRateLimiter()  # Uses shared pool automatically
   ```

2. **Reuse provider instances** (don't create new ones per-call):
   ```python
   # BAD - creates new provider (and rate limiter) per call
   def process_item(item):
       provider = MyProvider()
       return provider.fetch(item)

   # GOOD - reuse via lazy property
   class MyService:
       @property
       def provider(self):
           if self._provider is None:
               self._provider = MyProvider()
           return self._provider

       def process_item(self, item):
           return self.provider.fetch(item)
   ```

### For Testing

Reset the pool between tests to ensure isolation:

```python
from lobster.tools.rate_limiter import reset_redis_pool

def test_something():
    reset_redis_pool()  # Clean state
    # ... test code ...
    reset_redis_pool()  # Cleanup
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `NCBI_API_KEY` | None | NCBI API key (increases rate limit from 3 to 10 req/s) |

## Implementation Details

### Double-Checked Locking

The pool initialization uses double-checked locking for thread safety:

```python
if not _POOL_INITIALIZED:        # First check (fast path)
    with _POOL_LOCK:              # Acquire lock
        if not _POOL_INITIALIZED: # Second check (inside lock)
            _REDIS_POOL = _create_connection_pool()
            _POOL_INITIALIZED = True
```

This pattern ensures:
- Only one thread creates the pool
- Subsequent calls skip the lock entirely (fast path)
- No race conditions during initialization

### Why ConnectionPool Instead of Singleton Client

| Singleton Client | ConnectionPool |
|------------------|----------------|
| Single TCP connection | Multiple connections |
| Bottleneck under load | Efficient concurrency |
| Manual stale connection handling | Auto health checks |
| Custom reconnection logic | Built-in reconnection |

## Troubleshooting

### "Redis unavailable" Warning

If you see this warning, Redis is not running. Options:

1. **Start Redis**: `docker-compose up -d redis`
2. **Continue without Redis**: Rate limiting is disabled (fail-open), but system continues working

### Multiple "Redis connection pool established" Messages

This should NOT happen with the current implementation. If it does:

1. Check if `reset_redis_pool()` is being called unexpectedly
2. Verify providers are being reused (not created per-call)
3. Check for module import issues causing multiple pool initializations

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - See "Redis rate limiting pattern" in Patterns & Abstractions
- [Performance Optimization](22-performance-optimization.md) - General performance guidelines
