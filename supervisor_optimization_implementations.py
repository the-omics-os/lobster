"""
Implementation Code for Supervisor Import Optimization

This file contains the exact code changes needed to optimize the supervisor
agent import time from 2.7s to ~0.8s (70% reduction).

Apply these changes in order:
1. Priority 1: Lazy plugin discovery (agent_registry.py)
2. Priority 2: Lazy imports (supervisor.py)
3. Priority 3: Plugin cache (plugin_loader.py)
"""

# =============================================================================
# PRIORITY 1: Lazy Plugin Discovery
# File: lobster/config/agent_registry.py
# Expected savings: ~0.9s
# =============================================================================

AGENT_REGISTRY_CHANGES = """
# ADD at top of file after imports:
import threading

# ADD after AGENT_REGISTRY definition (around line 135):

# =============================================================================
# LAZY PLUGIN LOADING
# =============================================================================

_plugins_loaded = False
_plugins_lock = threading.Lock()


def _ensure_plugins_loaded() -> None:
    '''
    Lazy-load plugins on first access (thread-safe).

    This defers plugin discovery from module import time to first actual use,
    reducing CLI initialization time by ~0.9s.
    '''
    global _plugins_loaded

    # Fast path: already loaded
    if _plugins_loaded:
        return

    # Slow path: need to load (with thread safety)
    with _plugins_lock:
        # Double-check after acquiring lock
        if _plugins_loaded:
            return

        try:
            from lobster.core.plugin_loader import discover_plugins

            plugin_agents = discover_plugins()
            if plugin_agents:
                AGENT_REGISTRY.update(plugin_agents)
                # Log at debug level to avoid noise during imports
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Merged {len(plugin_agents)} plugin agents into registry")
        except ImportError:
            # plugin_loader not available (shouldn't happen in normal installs)
            pass
        except Exception as e:
            # Don't let plugin discovery failures break the core system
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Plugin discovery failed: {e}")

        _plugins_loaded = True


# MODIFY get_worker_agents() function (around line 147):
def get_worker_agents() -> Dict[str, AgentRegistryConfig]:
    '''Get only the worker agents (excluding system agents).'''
    _ensure_plugins_loaded()  # ADD THIS LINE
    return AGENT_REGISTRY.copy()


# MODIFY get_agent_registry_config() function (around line 152):
def get_agent_registry_config(agent_name: str) -> Optional[AgentRegistryConfig]:
    '''Get registry configuration for a specific agent.'''
    _ensure_plugins_loaded()  # ADD THIS LINE
    return AGENT_REGISTRY.get(agent_name)


# REMOVE OR COMMENT OUT lines 172-206 (the _merge_plugin_agents function and its call):
# def _merge_plugin_agents() -> None:
#     ...
#
# # Merge plugins at module load time
# _merge_plugin_agents()

# These lines are NO LONGER NEEDED because plugin discovery is now lazy
"""

# =============================================================================
# PRIORITY 2: Lazy Imports in supervisor.py
# File: lobster/agents/supervisor.py
# Expected savings: ~1.0s
# =============================================================================

SUPERVISOR_CHANGES = """
# REPLACE the imports section (lines 8-18) with:

from datetime import date
from typing import TYPE_CHECKING, List, Optional

from lobster.utils.logger import get_logger

# Lazy imports - only for type checking, not runtime
if TYPE_CHECKING:
    import platform
    import psutil
    from lobster.config.agent_capabilities import AgentCapabilityExtractor
    from lobster.config.agent_registry import AgentRegistryConfig
    from lobster.config.supervisor_config import SupervisorConfig
    from lobster.core.data_manager_v2 import DataManagerV2

logger = get_logger(__name__)


# MODIFY create_supervisor_prompt() function signature (line 23):
def create_supervisor_prompt(
    data_manager: "DataManagerV2",  # String annotation instead of direct type
    config: Optional["SupervisorConfig"] = None,  # String annotation
    active_agents: Optional[List[str]] = None,
) -> str:
    '''Create dynamic supervisor prompt based on system state and configuration.'''

    # ADD imports at function start:
    from lobster.config.agent_registry import get_worker_agents
    from lobster.config.supervisor_config import SupervisorConfig

    # Rest of function remains the same
    if config is None:
        config = SupervisorConfig.from_env()
        logger.debug(f"Using supervisor config mode: {config.get_prompt_mode()}")
    # ...


# MODIFY _build_agents_section() function (line 135):
def _build_agents_section(active_agents: List[str], config: "SupervisorConfig") -> str:
    '''Build dynamic agent descriptions from registry.'''

    # ADD imports at function start:
    from lobster.config.agent_capabilities import AgentCapabilityExtractor
    from lobster.config.agent_registry import get_agent_registry_config

    section = "<Available Agents>\\n"
    # Rest of function remains the same...


# MODIFY _build_decision_framework() function (line 165):
def _build_decision_framework(
    active_agents: List[str], config: "SupervisorConfig"
) -> str:
    '''Build decision framework with agent-specific delegation rules.'''
    # No imports needed here, function already doesn't use expensive imports


# MODIFY _build_workflow_section() function (line 295):
def _build_workflow_section(active_agents: List[str], config: "SupervisorConfig") -> str:
    '''Build workflow awareness section based on active agents.'''
    # No imports needed here


# MODIFY _build_response_rules() function (line 351):
def _build_response_rules(config: "SupervisorConfig") -> str:
    '''Build response rules based on configuration.'''
    # No imports needed here


# MODIFY _build_context_section() function (line 404):
def _build_context_section(
    data_manager: "DataManagerV2", config: "SupervisorConfig"
) -> str:
    '''Build current system context section.'''

    # ADD imports at function start (only loaded if this function is called):
    import platform
    import psutil

    sections = []
    # Rest of function remains the same...
"""

# =============================================================================
# PRIORITY 3: Plugin Discovery Caching
# File: lobster/core/plugin_loader.py
# Expected savings: ~0.5s on subsequent runs
# =============================================================================

PLUGIN_LOADER_CHANGES = """
# ADD at top of file after existing imports:
import hashlib
import time
from pathlib import Path

# ADD after imports, before functions:

# =============================================================================
# PLUGIN DISCOVERY CACHING
# =============================================================================

_CACHE_FILE = Path.home() / ".lobster" / "plugin_cache.json"
_CACHE_TTL = 3600  # 1 hour in seconds


def _get_plugin_cache_key() -> str:
    '''
    Generate cache key based on installed lobster packages.

    Returns cache key that changes when lobster packages are installed/updated.
    This ensures cache is invalidated when plugins might have changed.
    '''
    try:
        packages = []
        for dist in importlib.metadata.distributions():
            if dist.name.startswith('lobster'):
                # Include name and version in cache key
                packages.append(f"{dist.name}={dist.version}")

        # Sort for consistency
        packages.sort()

        # Hash the package list
        package_str = '|'.join(packages)
        return hashlib.md5(package_str.encode()).hexdigest()
    except Exception as e:
        # If we can't generate a proper key, disable caching
        logger.debug(f"Could not generate cache key: {e}")
        return "nocache"


def _load_plugin_cache() -> Optional[Dict[str, Any]]:
    '''
    Load plugin discovery results from cache if valid.

    Returns:
        Cached plugin dict if valid, None otherwise
    '''
    if not _CACHE_FILE.exists():
        return None

    try:
        cache_data = json.loads(_CACHE_FILE.read_text())

        # Validate cache structure
        if not isinstance(cache_data, dict):
            return None

        required_keys = ['key', 'timestamp', 'plugins']
        if not all(k in cache_data for k in required_keys):
            return None

        # Check if cache key matches current environment
        current_key = _get_plugin_cache_key()
        if cache_data['key'] != current_key:
            logger.debug("Plugin cache key mismatch (packages changed)")
            return None

        # Check if cache has expired
        cache_age = time.time() - cache_data['timestamp']
        if cache_age > _CACHE_TTL:
            logger.debug(f"Plugin cache expired (age: {cache_age:.0f}s)")
            return None

        logger.debug(f"Using cached plugin discovery (age: {cache_age:.0f}s)")
        return cache_data['plugins']

    except Exception as e:
        logger.debug(f"Failed to load plugin cache: {e}")
        return None


def _save_plugin_cache(plugins: Dict[str, Any]) -> None:
    '''
    Save plugin discovery results to cache.

    Args:
        plugins: Discovered plugin agents to cache
    '''
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            'key': _get_plugin_cache_key(),
            'timestamp': time.time(),
            'plugins': plugins,
            'version': 1  # For future cache format changes
        }

        _CACHE_FILE.write_text(json.dumps(cache_data, indent=2))
        logger.debug(f"Saved plugin cache to {_CACHE_FILE}")

    except Exception as e:
        # Don't fail if caching doesn't work
        logger.debug(f"Failed to save plugin cache: {e}")


# MODIFY discover_plugins() function (starts at line 58):
def discover_plugins() -> Dict[str, Any]:
    '''
    Discover all lobster plugin packages and load their registries.

    This function now includes disk caching to speed up subsequent
    discoveries when no packages have changed.

    Discovers packages by naming convention:
    - lobster_premium: shared premium features
    - lobster_custom_*: customer-specific features

    The function checks the entitlement to determine which custom
    packages are authorized for the current installation.

    Returns:
        Dict of agent configurations to merge into AGENT_REGISTRY.
        Keys are agent names, values are AgentRegistryConfig instances.
    '''
    # ADD: Try to load from cache first
    cached_plugins = _load_plugin_cache()
    if cached_plugins is not None:
        return cached_plugins

    # Existing discovery logic (unchanged):
    discovered_agents: Dict[str, Any] = {}

    # 1. Try to load lobster-premium package
    try:
        from lobster_premium import PREMIUM_REGISTRY
        discovered_agents.update(PREMIUM_REGISTRY)
        logger.info(
            f"Loaded {len(PREMIUM_REGISTRY)} premium agents from lobster-premium"
        )
    except ImportError:
        logger.debug("lobster-premium not installed (expected for free tier)")
    except Exception as e:
        logger.warning(f"Failed to load lobster-premium: {e}")

    # 2. Discover lobster-custom-* packages based on entitlement
    entitlement = _load_entitlement()
    custom_packages = entitlement.get("custom_packages", [])

    for pkg_name in custom_packages:
        try:
            # Convert package name to module name (dashes to underscores)
            module_name = pkg_name.replace("-", "_")
            module = importlib.import_module(module_name)

            if hasattr(module, "CUSTOM_REGISTRY"):
                registry = getattr(module, "CUSTOM_REGISTRY")
                discovered_agents.update(registry)
                logger.info(f"Loaded {len(registry)} custom agents from {pkg_name}")
            else:
                logger.debug(f"Package {pkg_name} has no CUSTOM_REGISTRY export")

        except ImportError as e:
            logger.warning(f"Custom package {pkg_name} not installed: {e}")
        except Exception as e:
            logger.warning(f"Failed to load custom package {pkg_name}: {e}")

    # 3. Also discover any packages via entry points (future-proof)
    discovered_agents.update(_discover_via_entry_points())

    # ADD: Save to cache before returning
    _save_plugin_cache(discovered_agents)

    return discovered_agents
"""

# =============================================================================
# ADDITIONAL: Environment Variable to Disable Plugins
# File: lobster/core/plugin_loader.py
# Optional: For CI/CD and testing
# =============================================================================

OPTIONAL_NO_PLUGINS = """
# ADD at very start of discover_plugins() function:
def discover_plugins() -> Dict[str, Any]:
    '''Discover all lobster plugin packages and load their registries.'''

    # ADD: Allow disabling plugin discovery entirely
    if os.environ.get('LOBSTER_NO_PLUGINS') == '1':
        logger.debug("Plugin discovery disabled (LOBSTER_NO_PLUGINS=1)")
        return {}

    # Rest of function continues...
"""

# =============================================================================
# TESTING CODE
# =============================================================================

TEST_CODE = """
# File: tests/unit/test_supervisor_import_performance.py

import time
import sys
import pytest


class TestImportPerformance:
    '''Performance tests for supervisor agent import time.'''

    def test_supervisor_import_time(self):
        '''Verify supervisor import is under 100ms.'''
        # Clear any cached imports
        modules_to_clear = [m for m in sys.modules if m.startswith('lobster.agents.supervisor')]
        for mod in modules_to_clear:
            del sys.modules[mod]

        start = time.time()
        import lobster.agents.supervisor
        elapsed = time.time() - start

        assert elapsed < 0.1, (
            f"Supervisor import took {elapsed:.3f}s (target: <0.1s). "
            f"This suggests expensive imports are still happening at module level."
        )

    def test_agent_registry_import_time(self):
        '''Verify agent_registry import is under 100ms.'''
        # Clear any cached imports
        modules_to_clear = [m for m in sys.modules if m.startswith('lobster.config.agent_registry')]
        for mod in modules_to_clear:
            del sys.modules[mod]

        start = time.time()
        import lobster.config.agent_registry
        elapsed = time.time() - start

        assert elapsed < 0.1, (
            f"Agent registry import took {elapsed:.3f}s (target: <0.1s). "
            f"Plugin discovery should be lazy, not at import time."
        )

    def test_plugin_discovery_is_lazy(self):
        '''Verify plugins aren't loaded until first use.'''
        # Clear any cached imports
        modules_to_clear = [m for m in sys.modules if m.startswith('lobster.config.agent_registry')]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import should NOT trigger plugin discovery
        import lobster.config.agent_registry as registry
        assert not registry._plugins_loaded, "Plugins should not be loaded at import time"

        # First access should trigger discovery
        agents = registry.get_worker_agents()
        assert registry._plugins_loaded, "Plugins should be loaded after first access"
        assert isinstance(agents, dict), "Should return agent dict"


# File: tests/integration/test_cli_init_performance.py

import subprocess
import time
import pytest


def test_cli_help_performance():
    '''Verify CLI --help is under 2 seconds.'''
    start = time.time()
    result = subprocess.run(
        ['lobster', '--help'],
        capture_output=True,
        timeout=5,
        text=True
    )
    elapsed = time.time() - start

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert elapsed < 2.0, (
        f"CLI --help took {elapsed:.3f}s (target: <2.0s). "
        f"Import optimization may not be working correctly."
    )


def test_cli_version_performance():
    '''Verify CLI --version is under 1 second.'''
    start = time.time()
    result = subprocess.run(
        ['lobster', '--version'],
        capture_output=True,
        timeout=5,
        text=True
    )
    elapsed = time.time() - start

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert elapsed < 1.0, (
        f"CLI --version took {elapsed:.3f}s (target: <1.0s)"
    )


# File: scripts/benchmark_imports.py

import time
import sys


def time_import(module_name: str, clear_cache: bool = True) -> float:
    '''
    Time the import of a module.

    Args:
        module_name: Module to import
        clear_cache: If True, clear sys.modules cache first

    Returns:
        Import time in seconds
    '''
    if clear_cache:
        # Clear any cached imports
        modules_to_clear = [m for m in sys.modules if m.startswith(module_name.split('.')[0])]
        for mod in modules_to_clear:
            del sys.modules[mod]

    start = time.time()
    __import__(module_name)
    return time.time() - start


def main():
    '''Benchmark critical import paths.'''
    modules = [
        'lobster.agents.supervisor',
        'lobster.agents.research_agent',
        'lobster.config.agent_registry',
        'lobster.config.agent_capabilities',
        'lobster.core.plugin_loader',
        'lobster.core.data_manager_v2',
    ]

    print("=" * 70)
    print("LOBSTER IMPORT PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"{'Module':<45s} {'Time (ms)':>12s} {'Status':>10s}")
    print("-" * 70)

    targets = {
        'lobster.agents.supervisor': 100,
        'lobster.agents.research_agent': 100,
        'lobster.config.agent_registry': 100,
        'lobster.config.agent_capabilities': 50,
        'lobster.core.plugin_loader': 50,
        'lobster.core.data_manager_v2': 50,
    }

    total_time = 0
    failures = []

    for module in modules:
        try:
            elapsed = time_import(module)
            elapsed_ms = elapsed * 1000
            total_time += elapsed

            target_ms = targets.get(module, 100)
            status = "✓ PASS" if elapsed_ms < target_ms else "✗ SLOW"

            print(f"{module:<45s} {elapsed_ms:>10.1f}ms {status:>10s}")

            if elapsed_ms >= target_ms:
                failures.append((module, elapsed_ms, target_ms))

        except Exception as e:
            print(f"{module:<45s} {'ERROR':>12s} {str(e)[:30]}")

    print("-" * 70)
    print(f"{'TOTAL':<45s} {total_time*1000:>10.1f}ms")
    print("=" * 70)

    if failures:
        print("\\nSLOW IMPORTS DETECTED:")
        for module, actual, target in failures:
            print(f"  - {module}: {actual:.1f}ms (target: {target:.1f}ms)")
        sys.exit(1)
    else:
        print("\\n✓ All imports within target performance!")
        sys.exit(0)


if __name__ == '__main__':
    main()
"""

# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY = """
IMPLEMENTATION SUMMARY
======================

Apply changes in this order:

1. PRIORITY 1 (Required): Lazy plugin discovery in agent_registry.py
   - Add _ensure_plugins_loaded() function
   - Modify get_worker_agents() and get_agent_registry_config()
   - Remove _merge_plugin_agents() call at module level
   - Expected: ~0.9s improvement

2. PRIORITY 2 (Required): Lazy imports in supervisor.py
   - Move imports to TYPE_CHECKING block
   - Add function-level imports where needed
   - Use string annotations for types
   - Expected: ~1.0s improvement

3. PRIORITY 3 (Optional): Plugin caching in plugin_loader.py
   - Add cache read/write functions
   - Modify discover_plugins() to use cache
   - Expected: ~0.5s improvement on subsequent runs

Total expected improvement: 2.7s → 0.8s (70% reduction)

After implementation, run:
  python scripts/benchmark_imports.py
  pytest tests/unit/test_supervisor_import_performance.py

Then apply same patterns to research_agent.py for additional 1.9s improvement.
"""

if __name__ == '__main__':
    print(SUMMARY)
