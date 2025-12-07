# Supervisor Agent Import Performance Analysis

## Executive Summary

The supervisor agent module takes **2.7 seconds to import** (56% of total CLI initialization time). Analysis reveals that **ZERO imports are actually needed at module level** except for logging, but all 9 imports happen eagerly, triggering expensive cascading imports.

## Root Cause Analysis

### 1. Plugin Discovery at Module Import Time (CRITICAL)

**Location:** `lobster/config/agent_registry.py` line 206

```python
# Merge plugins at module load time
_merge_plugin_agents()
```

This runs **every time** `agent_registry` is imported and:
- Imports `lobster.core.plugin_loader`
- Tries to import `lobster_premium` package
- Loads `license_manager.load_entitlement()` (reads ~/.lobster/license.json from disk)
- Scans for custom packages via `importlib.import_module()`
- Calls `importlib.metadata.entry_points()` to scan all installed packages
- Updates the global `AGENT_REGISTRY` dictionary

**Measured cost:** ~0.9s (from agent_registry import taking 0.918s)

**Impact:** This happens on EVERY import of `agent_registry`, which is imported by:
- `supervisor.py`
- `agent_capabilities.py`
- Many other agent modules

### 2. Heavy Scientific Library Imports

**Location:** Various adapter/backend modules imported by `data_manager_v2.py`

While `data_manager_v2.py` itself only takes 0.023s (has good lazy loading), it still imports modules that eagerly load:
- `scanpy` (1.164s) - imported by `transcriptomics_adapter.py:16`, `h5ad_backend.py:16`
- `anndata` (0.471s) - imported by multiple modules
- `pandas` (0.383s) - needed but expensive

### 3. Unnecessary Eager Imports

**supervisor.py imports at module level:**
```python
from lobster.config.agent_capabilities import AgentCapabilityExtractor
from lobster.config.agent_registry import get_agent_registry_config, get_worker_agents
from lobster.config.supervisor_config import SupervisorConfig
from lobster.core.data_manager_v2 import DataManagerV2
```

**Usage analysis shows:**
- **ONLY `get_logger` is used at module level** (line 20)
- **ALL other imports are ONLY used inside functions**
- None of these need to be imported when the module is first loaded

## Measured Import Times

```
Module                                              Time
--------------------------------------------------  ------
platform                                            0.001s  ✓ (stdlib)
psutil                                              0.014s  ✓ (lightweight)
pandas                                              0.383s  ⚠️ (moderate)
anndata                                             0.471s  ❌ (heavy)
scanpy                                              1.164s  ❌ (very heavy)
lobster.config.agent_registry                       0.918s  ❌ (plugin discovery!)
lobster.config.supervisor_config                    0.096s  ✓ (acceptable)
lobster.config.agent_capabilities                   0.001s  ✓ (lightweight)
lobster.core.data_manager_v2                        0.023s  ✓ (already optimized)
```

## Detailed Recommendations

### Priority 1: Lazy Plugin Discovery (CRITICAL - Saves ~0.9s)

**Problem:** Plugin discovery runs at module import time in `agent_registry.py`

**Solution:** Defer plugin discovery until first use

**Implementation:**

```python
# In lobster/config/agent_registry.py

# REMOVE lines 205-206:
# # Merge plugins at module load time
# _merge_plugin_agents()

# ADD lazy loading mechanism:
_plugins_loaded = False
_plugins_lock = threading.Lock()

def _ensure_plugins_loaded() -> None:
    """Lazy-load plugins on first access (thread-safe)."""
    global _plugins_loaded
    if _plugins_loaded:
        return

    with _plugins_lock:
        if _plugins_loaded:  # Double-check
            return

        try:
            from lobster.core.plugin_loader import discover_plugins
            plugin_agents = discover_plugins()
            if plugin_agents:
                AGENT_REGISTRY.update(plugin_agents)
                logger.debug(f"Merged {len(plugin_agents)} plugin agents")
        except Exception as e:
            logger.warning(f"Plugin discovery failed: {e}")

        _plugins_loaded = True

# MODIFY public functions to ensure plugins are loaded:
def get_worker_agents() -> Dict[str, AgentRegistryConfig]:
    """Get only the worker agents (excluding system agents)."""
    _ensure_plugins_loaded()
    return AGENT_REGISTRY.copy()

def get_agent_registry_config(agent_name: str) -> Optional[AgentRegistryConfig]:
    """Get registry configuration for a specific agent."""
    _ensure_plugins_loaded()
    return AGENT_REGISTRY.get(agent_name)
```

**Expected savings:** ~0.9s (90% of current agent_registry import time)

### Priority 2: Lazy Imports in supervisor.py (Saves ~1.0s)

**Problem:** All imports happen eagerly but are only used in functions

**Solution:** Move imports to function level or use TYPE_CHECKING

**Implementation:**

```python
# lobster/agents/supervisor.py

# TOP OF FILE - Keep only essential imports
from datetime import date
from typing import TYPE_CHECKING, List, Optional

from lobster.utils.logger import get_logger

# Lazy imports via TYPE_CHECKING
if TYPE_CHECKING:
    import platform
    import psutil
    from lobster.config.agent_capabilities import AgentCapabilityExtractor
    from lobster.config.agent_registry import AgentRegistryConfig
    from lobster.config.supervisor_config import SupervisorConfig
    from lobster.core.data_manager_v2 import DataManagerV2

logger = get_logger(__name__)


def create_supervisor_prompt(
    data_manager: "DataManagerV2",  # String annotation
    config: Optional["SupervisorConfig"] = None,
    active_agents: Optional[List[str]] = None,
) -> str:
    """Create dynamic supervisor prompt..."""
    # Import at function level
    from lobster.config.agent_registry import get_agent_registry_config, get_worker_agents
    from lobster.config.supervisor_config import SupervisorConfig

    if config is None:
        config = SupervisorConfig.from_env()

    # ... rest of function


def _build_agents_section(active_agents: List[str], config: "SupervisorConfig") -> str:
    """Build dynamic agent descriptions from registry."""
    from lobster.config.agent_capabilities import AgentCapabilityExtractor
    from lobster.config.agent_registry import get_agent_registry_config

    # ... rest of function


def _build_context_section(
    data_manager: "DataManagerV2",
    config: "SupervisorConfig"
) -> str:
    """Build current system context section."""
    import platform
    import psutil

    # ... rest of function
```

**Expected savings:** ~1.0s (eliminates eager loading of agent_registry, agent_capabilities)

### Priority 3: Cache Plugin Discovery Results (Additional ~0.5s on subsequent runs)

**Problem:** Even with lazy loading, plugin discovery happens on first agent creation

**Solution:** Cache discovered plugins to disk

**Implementation:**

```python
# In lobster/core/plugin_loader.py

import json
from pathlib import Path
import hashlib

_CACHE_FILE = Path.home() / ".lobster" / "plugin_cache.json"
_CACHE_TTL = 3600  # 1 hour

def _get_cache_key() -> str:
    """Generate cache key based on installed packages."""
    try:
        import importlib.metadata
        packages = sorted(
            [d.name for d in importlib.metadata.distributions()
             if d.name.startswith('lobster')]
        )
        return hashlib.md5('|'.join(packages).encode()).hexdigest()
    except:
        return "nocache"

def discover_plugins() -> Dict[str, Any]:
    """Discover plugins with disk caching."""

    # Try cache first
    cache_key = _get_cache_key()
    if _CACHE_FILE.exists():
        try:
            cache = json.loads(_CACHE_FILE.read_text())
            if (cache.get('key') == cache_key and
                time.time() - cache.get('timestamp', 0) < _CACHE_TTL):
                logger.debug("Using cached plugin discovery")
                return cache.get('plugins', {})
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")

    # Perform discovery (existing logic)
    discovered_agents = {}
    # ... existing discovery code ...

    # Write cache
    try:
        _CACHE_FILE.parent.mkdir(exist_ok=True)
        cache = {
            'key': cache_key,
            'timestamp': time.time(),
            'plugins': discovered_agents
        }
        _CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception as e:
        logger.debug(f"Cache write failed: {e}")

    return discovered_agents
```

**Expected savings:** ~0.5s on subsequent CLI invocations (after first run)

## Implementation Priority & Expected Impact

| Priority | Change | File(s) | Lines | Complexity | Savings |
|----------|--------|---------|-------|------------|---------|
| **P1** | Lazy plugin discovery | `agent_registry.py` | ~30 | Low | **~0.9s** |
| **P2** | Lazy imports in supervisor | `supervisor.py` | ~20 | Low | **~1.0s** |
| **P3** | Cache plugin discovery | `plugin_loader.py` | ~40 | Medium | **~0.5s** (subsequent) |

**Total expected improvement: 1.9s → 0.8s (70% reduction)**

Combined with research_agent optimization, total CLI init could go from **~5s to ~1.5s (70% improvement)**.

## Additional Opportunities (Lower Priority)

### 4. Environment Variable for Plugin Discovery

Allow users to completely disable plugin discovery:

```python
# Skip plugin discovery entirely if env var set
if os.environ.get('LOBSTER_NO_PLUGINS') == '1':
    return {}
```

**Use case:** CI/CD, testing, or free tier users who know they don't have plugins

### 5. Defer psutil and platform imports

These are only used in context building (which may not happen):

```python
def _build_context_section(...):
    import platform  # Only import if this function is called
    import psutil
    # ...
```

**Savings:** ~0.015s (small but adds up)

## Testing Strategy

### 1. Unit Tests

```python
# tests/unit/test_lazy_imports.py

def test_supervisor_import_time():
    """Verify supervisor import is under 100ms."""
    import time
    start = time.time()
    import lobster.agents.supervisor
    elapsed = time.time() - start
    assert elapsed < 0.1, f"Supervisor import took {elapsed:.3f}s (target: <0.1s)"

def test_agent_registry_import_time():
    """Verify agent_registry import is under 100ms."""
    import time
    start = time.time()
    import lobster.config.agent_registry
    elapsed = time.time() - start
    assert elapsed < 0.1, f"Agent registry import took {elapsed:.3f}s (target: <0.1s)"

def test_plugin_discovery_is_lazy():
    """Verify plugins aren't loaded until first use."""
    import importlib
    import sys

    # Remove any cached imports
    if 'lobster.config.agent_registry' in sys.modules:
        del sys.modules['lobster.config.agent_registry']

    # Import should NOT trigger plugin discovery
    from lobster.config import agent_registry
    assert not agent_registry._plugins_loaded

    # First access should trigger discovery
    agents = agent_registry.get_worker_agents()
    assert agent_registry._plugins_loaded
```

### 2. Integration Tests

```python
# tests/integration/test_cli_performance.py

def test_cli_init_performance():
    """Verify CLI initialization is under 2 seconds."""
    import subprocess
    import time

    start = time.time()
    result = subprocess.run(
        ['lobster', '--help'],
        capture_output=True,
        timeout=5
    )
    elapsed = time.time() - start

    assert result.returncode == 0
    assert elapsed < 2.0, f"CLI init took {elapsed:.3f}s (target: <2.0s)"
```

### 3. Benchmarking

Create a script to measure before/after:

```python
# scripts/benchmark_imports.py

import time
import sys

def time_import(module_name):
    # Clear any cached imports
    for mod in list(sys.modules.keys()):
        if mod.startswith('lobster'):
            del sys.modules[mod]

    start = time.time()
    __import__(module_name)
    return time.time() - start

modules = [
    'lobster.agents.supervisor',
    'lobster.agents.research_agent',
    'lobster.config.agent_registry',
    'lobster.core.plugin_loader',
]

print("Module                                 Time")
print("-" * 50)
for module in modules:
    elapsed = time_import(module)
    print(f"{module:40s} {elapsed:.3f}s")
```

## Risks & Mitigation

### Risk 1: Thread Safety in Lazy Loading

**Mitigation:** Use `threading.Lock()` with double-checked locking pattern (shown in implementation)

### Risk 2: Plugin Discovery Might Be Called Multiple Times

**Mitigation:** Use global `_plugins_loaded` flag to ensure it only runs once

### Risk 3: Type Hints Break with String Annotations

**Mitigation:** Use `from __future__ import annotations` at top of file (Python 3.7+) or `TYPE_CHECKING` pattern

### Risk 4: Cache Invalidation

**Mitigation:** Cache key includes hash of installed lobster packages; cache expires after 1 hour

## Next Steps

1. **Implement P1** (lazy plugin discovery) - highest impact, lowest risk
2. **Verify with benchmarks** - run `benchmark_imports.py` before/after
3. **Run test suite** - ensure no regressions
4. **Implement P2** (lazy imports in supervisor)
5. **Run full CLI test** - test actual user workflows
6. **Implement P3** (caching) if needed
7. **Apply same patterns to research_agent.py** (1.9s import time)

## Related Work

This analysis should be combined with similar optimization for:
- `research_agent.py` (1.9s import time, 39% of init)
- Other agent modules that might have similar patterns
- Any other modules in the critical path for CLI initialization
