# Supervisor Agent Import Optimization Report

## Executive Summary

**Problem:** Supervisor agent module takes 2.7s to import (56% of total CLI init time)

**Root Cause:** Plugin discovery runs at module import time + unnecessary eager imports

**Solution:** Lazy plugin loading + deferred imports

**Expected Impact:** 2.7s → 0.8s (70% reduction, saving ~1.9s)

---

## 1. Top-Level Imports with Cost Analysis

| Import | Module | Cost | Used at Module Level? | Recommendation |
|--------|--------|------|----------------------|----------------|
| `platform` | stdlib | **LOW** (0.001s) | ❌ No | Move to function |
| `date` | datetime | **LOW** (0.001s) | ✓ Yes | Keep |
| `List, Optional` | typing | **LOW** (0.001s) | ✓ Yes | Keep |
| `psutil` | third-party | **LOW** (0.014s) | ❌ No | Move to function |
| `AgentCapabilityExtractor` | lobster.config | **LOW** (0.001s) | ❌ No | Move to function |
| `get_agent_registry_config` | lobster.config | **HIGH** (0.918s) | ❌ No | **CRITICAL: Lazy load** |
| `get_worker_agents` | lobster.config | **HIGH** (0.918s) | ❌ No | **CRITICAL: Lazy load** |
| `SupervisorConfig` | lobster.config | **LOW** (0.096s) | ❌ No | Move to function |
| `DataManagerV2` | lobster.core | **LOW** (0.023s) | ❌ No | Move to function |
| `get_logger` | lobster.utils | **LOW** (<0.001s) | ✓ Yes | Keep |

### Critical Finding

**ONLY `get_logger` is actually used at module level!** All other imports can be deferred.

The `agent_registry` import is catastrophically slow (0.918s) because it triggers plugin discovery at module load time.

---

## 2. Module-Level Code Analysis

### Current Behavior

```python
# Line 20: ONLY module-level code
logger = get_logger(__name__)
```

**Finding:** No other code runs at module level. All imports are wasted until functions are called.

### Plugin Discovery Problem

```python
# agent_registry.py line 206 - RUNS AT IMPORT TIME!
_merge_plugin_agents()
```

This function:
1. Imports `lobster.core.plugin_loader` (~0.1s)
2. Tries to import `lobster_premium` (~0.2s if installed)
3. Loads entitlement file from disk (~0.05s)
4. Scans installed packages with `importlib.metadata.entry_points()` (~0.3s)
5. Tries to import custom packages (~0.2s per package)

**Total cost: ~0.9s on every import of agent_registry!**

---

## 3. Specific Recommendations

### Priority 1: Lazy Plugin Discovery (CRITICAL)

**File:** `lobster/config/agent_registry.py`

**Problem:** Plugin discovery runs at module import time (line 206)

**Solution:** Defer until first access

**Implementation:**

```python
# ADD: Lazy loading mechanism
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
        except Exception as e:
            logger.warning(f"Plugin discovery failed: {e}")

        _plugins_loaded = True

# MODIFY: Add lazy loading to public functions
def get_worker_agents() -> Dict[str, AgentRegistryConfig]:
    _ensure_plugins_loaded()  # ADD THIS
    return AGENT_REGISTRY.copy()

def get_agent_registry_config(agent_name: str) -> Optional[AgentRegistryConfig]:
    _ensure_plugins_loaded()  # ADD THIS
    return AGENT_REGISTRY.get(agent_name)

# REMOVE: Lines 172-206 (_merge_plugin_agents and its call)
```

**Expected Savings:** ~0.9s (90% of agent_registry import time)

**Risk:** Low - thread-safe implementation with double-checked locking

---

### Priority 2: Lazy Imports in Supervisor

**File:** `lobster/agents/supervisor.py`

**Problem:** All imports happen at module level but are only used in functions

**Solution:** Move to TYPE_CHECKING + function-level imports

**Implementation:**

```python
# REPLACE imports (lines 8-18) with:
from datetime import date
from typing import TYPE_CHECKING, List, Optional

from lobster.utils.logger import get_logger

if TYPE_CHECKING:
    import platform
    import psutil
    from lobster.config.agent_capabilities import AgentCapabilityExtractor
    from lobster.config.agent_registry import AgentRegistryConfig
    from lobster.config.supervisor_config import SupervisorConfig
    from lobster.core.data_manager_v2 import DataManagerV2

logger = get_logger(__name__)


# MODIFY: Use string annotations + function-level imports
def create_supervisor_prompt(
    data_manager: "DataManagerV2",  # String annotation
    config: Optional["SupervisorConfig"] = None,
    active_agents: Optional[List[str]] = None,
) -> str:
    # Import when actually needed
    from lobster.config.agent_registry import get_worker_agents
    from lobster.config.supervisor_config import SupervisorConfig

    # ... rest of function


def _build_agents_section(active_agents: List[str], config: "SupervisorConfig") -> str:
    # Import when needed
    from lobster.config.agent_capabilities import AgentCapabilityExtractor
    from lobster.config.agent_registry import get_agent_registry_config

    # ... rest of function


def _build_context_section(data_manager: "DataManagerV2", config: "SupervisorConfig") -> str:
    # Import only if context section is built
    import platform
    import psutil

    # ... rest of function
```

**Expected Savings:** ~1.0s (eliminates eager agent_registry loading)

**Risk:** Low - TYPE_CHECKING pattern is standard, function imports are valid

---

### Priority 3: Plugin Discovery Caching

**File:** `lobster/core/plugin_loader.py`

**Problem:** Even with lazy loading, first discovery is slow (~0.9s)

**Solution:** Cache discovery results to disk

**Implementation:**

```python
import hashlib
import json
import time
from pathlib import Path

_CACHE_FILE = Path.home() / ".lobster" / "plugin_cache.json"
_CACHE_TTL = 3600  # 1 hour

def _get_plugin_cache_key() -> str:
    """Generate cache key based on installed packages."""
    packages = [
        f"{d.name}={d.version}"
        for d in importlib.metadata.distributions()
        if d.name.startswith('lobster')
    ]
    packages.sort()
    return hashlib.md5('|'.join(packages).encode()).hexdigest()

def _load_plugin_cache() -> Optional[Dict[str, Any]]:
    """Load cached plugins if valid."""
    if not _CACHE_FILE.exists():
        return None

    try:
        cache = json.loads(_CACHE_FILE.read_text())

        # Validate cache key and age
        if cache['key'] != _get_plugin_cache_key():
            return None
        if time.time() - cache['timestamp'] > _CACHE_TTL:
            return None

        return cache['plugins']
    except:
        return None

def _save_plugin_cache(plugins: Dict[str, Any]) -> None:
    """Save plugins to cache."""
    try:
        _CACHE_FILE.parent.mkdir(exist_ok=True)
        cache = {
            'key': _get_plugin_cache_key(),
            'timestamp': time.time(),
            'plugins': plugins
        }
        _CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except:
        pass  # Non-critical

def discover_plugins() -> Dict[str, Any]:
    """Discover plugins with caching."""
    # Try cache first
    cached = _load_plugin_cache()
    if cached is not None:
        return cached

    # Perform discovery (existing code)
    discovered = {}
    # ... existing discovery logic ...

    # Save to cache
    _save_plugin_cache(discovered)
    return discovered
```

**Expected Savings:** ~0.5s on subsequent CLI invocations (after first run)

**Risk:** Low - cache invalidation is robust, failures are non-critical

---

## 4. Implementation Code Examples

### Example 1: Lazy Plugin Loading (CRITICAL)

See `supervisor_optimization_implementations.py` section "PRIORITY 1" for complete code.

**Key pattern:**
```python
_loaded = False
_lock = threading.Lock()

def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        # expensive operation
        _loaded = True

def public_function():
    _ensure_loaded()  # Add to all entry points
    # ... rest of function
```

### Example 2: TYPE_CHECKING Pattern

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expensive.module import ExpensiveClass

def my_function(arg: "ExpensiveClass") -> None:
    from expensive.module import ExpensiveClass  # Import at runtime
    # ... use arg
```

### Example 3: Function-Level Imports

```python
def build_report():
    # Import only when this function is called
    import platform
    import psutil

    return {
        'platform': platform.system(),
        'memory': psutil.virtual_memory().percent
    }
```

---

## 5. Expected Performance Improvement

### Before Optimization

```
Module Import Times:
  agent_registry:     0.918s  (plugin discovery at import)
  supervisor:         2.700s  (includes agent_registry + other imports)
  CLI init:           ~5.0s   (includes supervisor + research_agent)
```

### After Priority 1 + 2

```
Module Import Times:
  agent_registry:     0.020s  (no plugin discovery)
  supervisor:         0.080s  (minimal imports)
  CLI init:           ~3.0s   (still need to optimize research_agent)

First agent creation:  +0.9s  (plugin discovery deferred to here)
```

### After Priority 3 (Cache)

```
Subsequent runs:
  agent_registry:     0.020s
  supervisor:         0.080s
  CLI init:           ~3.0s

First agent creation:  +0.1s  (cache hit)
```

### Combined Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Supervisor import | 2.7s | 0.8s | **-1.9s (70%)** |
| First agent creation | 0s | +0.9s | Deferred cost |
| Subsequent agent creation | 0s | +0.1s | Cached |
| CLI --help | ~5s | ~3s | **-2s (40%)** |

**Note:** First agent creation is slower, but:
1. This only happens once per CLI invocation
2. User doesn't notice (they're waiting for agent response anyway)
3. `--help`, `--version` commands never trigger it (fast!)

---

## 6. Testing Strategy

### Unit Tests

```python
# tests/unit/test_supervisor_import_performance.py

def test_supervisor_import_time():
    """Verify supervisor imports in under 100ms."""
    import time
    start = time.time()
    import lobster.agents.supervisor
    elapsed = time.time() - start
    assert elapsed < 0.1, f"Too slow: {elapsed:.3f}s"

def test_plugin_discovery_is_lazy():
    """Verify plugins aren't loaded at import."""
    import lobster.config.agent_registry as registry
    assert not registry._plugins_loaded

    # Trigger discovery
    agents = registry.get_worker_agents()
    assert registry._plugins_loaded
```

### Integration Tests

```python
# tests/integration/test_cli_init_performance.py

def test_cli_help_fast():
    """Verify CLI --help is under 2 seconds."""
    import subprocess, time
    start = time.time()
    result = subprocess.run(['lobster', '--help'], capture_output=True)
    elapsed = time.time() - start
    assert elapsed < 2.0, f"Too slow: {elapsed:.3f}s"
```

### Benchmark Script

```bash
python scripts/benchmark_imports.py
```

Expected output after optimization:
```
Module                                     Time (ms)    Status
--------------------------------------------------------------
lobster.agents.supervisor                      80.0ms  ✓ PASS
lobster.config.agent_registry                  20.0ms  ✓ PASS
lobster.core.plugin_loader                     10.0ms  ✓ PASS
--------------------------------------------------------------
TOTAL                                         110.0ms
✓ All imports within target performance!
```

---

## 7. Implementation Checklist

- [ ] **Priority 1: Lazy plugin discovery**
  - [ ] Add `_ensure_plugins_loaded()` to `agent_registry.py`
  - [ ] Modify `get_worker_agents()` and `get_agent_registry_config()`
  - [ ] Remove `_merge_plugin_agents()` call at module level
  - [ ] Run unit tests: `pytest tests/unit/test_supervisor_import_performance.py`
  - [ ] Verify: `python -c "import time; s=time.time(); import lobster.config.agent_registry; print(f'{time.time()-s:.3f}s')"` should be <0.1s

- [ ] **Priority 2: Lazy imports in supervisor**
  - [ ] Update imports section with TYPE_CHECKING
  - [ ] Add function-level imports
  - [ ] Update type annotations to use strings
  - [ ] Run unit tests
  - [ ] Verify: `python -c "import time; s=time.time(); import lobster.agents.supervisor; print(f'{time.time()-s:.3f}s')"` should be <0.1s

- [ ] **Priority 3: Plugin caching (optional)**
  - [ ] Add cache functions to `plugin_loader.py`
  - [ ] Update `discover_plugins()` to use cache
  - [ ] Test cache invalidation (install/remove package)
  - [ ] Verify cache file created: `~/.lobster/plugin_cache.json`

- [ ] **Verification**
  - [ ] Run benchmark: `python scripts/benchmark_imports.py`
  - [ ] Run full test suite: `make test`
  - [ ] Test CLI performance: `time lobster --help` (should be <2s)
  - [ ] Test actual workflow: `lobster chat` and verify agents work

- [ ] **Apply to research_agent.py**
  - [ ] Same patterns (saves additional 1.9s)
  - [ ] Total CLI init improvement: ~4s

---

## 8. Files to Modify

1. **`lobster/config/agent_registry.py`** (Priority 1)
   - Lines to add: ~30
   - Lines to remove: ~35
   - Complexity: Low
   - Risk: Low (thread-safe pattern)

2. **`lobster/agents/supervisor.py`** (Priority 2)
   - Lines to modify: ~25
   - Complexity: Low
   - Risk: Low (standard pattern)

3. **`lobster/core/plugin_loader.py`** (Priority 3)
   - Lines to add: ~80
   - Complexity: Medium
   - Risk: Low (non-critical)

4. **`scripts/benchmark_imports.py`** (New)
   - Lines: ~100
   - For testing/validation

5. **`tests/unit/test_supervisor_import_performance.py`** (New)
   - Lines: ~50
   - Unit tests for optimization

---

## 9. Related Work

After completing supervisor optimization, apply same patterns to:

1. **`research_agent.py`** (1.9s import time)
   - Same plugin discovery issue
   - Same unnecessary eager imports
   - Expected savings: ~1.5s

2. **Other agent modules** (if slow)
   - Run benchmark to identify
   - Apply same patterns

3. **Service modules** (if imported eagerly)
   - Check if scanpy/anndata imported at module level
   - Move to function level where possible

---

## 10. References

- **Analysis:** `supervisor_import_analysis.md`
- **Implementation code:** `supervisor_optimization_implementations.py`
- **Thread safety pattern:** https://en.wikipedia.org/wiki/Double-checked_locking
- **TYPE_CHECKING pattern:** https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING

---

## Summary

The supervisor agent's 2.7s import time is caused by:

1. **Plugin discovery running at import time** (0.9s) - CRITICAL
2. **Unnecessary eager imports** (1.0s) - HIGH PRIORITY
3. **No caching of discovery results** (0.5s) - MEDIUM PRIORITY

Implementing the 3 priorities will reduce import time by 70% (2.7s → 0.8s), with the deferred cost appearing only when agents are actually used.

**Next steps:**
1. Apply Priority 1 and 2 changes
2. Run tests and benchmarks
3. Apply same patterns to research_agent.py
4. Target final CLI init time: <2s (currently ~5s)
