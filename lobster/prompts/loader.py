"""Prompt section loader with LRU caching."""

import logging
from functools import lru_cache
from importlib.resources import as_file, files
from typing import List

logger = logging.getLogger(__name__)


class PromptLoader:
    """Load prompt sections from packages with LRU caching.

    Uses importlib.resources for package-relative file access,
    enabling both installed packages and development mode.

    Performance targets:
    - First load: <100ms
    - Cached load: <30ms
    - Startup preload: <500ms
    """

    def __init__(self, cache_size: int = 128):
        """Initialize loader with configurable cache size.

        Args:
            cache_size: Maximum number of sections to cache (default: 128)
        """
        self._cache_size = cache_size
        # Note: We use a module-level cache function to enable cache_clear()
        # The instance method wraps it for OOP interface

    def load_section(self, package_name: str, section_path: str) -> str:
        """Load a markdown section from a package.

        Args:
            package_name: Package name (e.g., "lobster.prompts")
            section_path: Relative path within package (e.g., "shared/role_identity.md")

        Returns:
            Section content as string, empty string if not found
        """
        return _load_section_cached(package_name, section_path)

    def preload_shared_sections(self) -> None:
        """Preload all shared sections from core at startup.

        Call this during application initialization to warm the cache.
        """
        shared_sections = [
            "shared/role_identity.md",
            "shared/important_rules.md",
            "shared/tool_usage_patterns.md",
        ]
        for section in shared_sections:
            self.load_section("lobster.prompts", section)
        logger.debug(f"Preloaded {len(shared_sections)} shared prompt sections")

    def clear_cache(self) -> None:
        """Clear the LRU cache (for testing and hot-reload)."""
        _load_section_cached.cache_clear()
        logger.debug("Prompt section cache cleared")

    def get_cache_info(self) -> dict:
        """Get cache statistics for monitoring.

        Returns:
            Dict with hits, misses, maxsize, currsize
        """
        info = _load_section_cached.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "maxsize": info.maxsize,
            "currsize": info.currsize,
        }


@lru_cache(maxsize=128)
def _load_section_cached(package_name: str, section_path: str) -> str:
    """Cached implementation of section loading.

    This is a module-level function to enable proper LRU caching
    across all PromptLoader instances.
    """
    try:
        pkg_files = files(package_name)
        section_file = pkg_files.joinpath(section_path)
        with as_file(section_file) as path:
            return path.read_text()
    except FileNotFoundError:
        logger.warning(f"Prompt section not found: {package_name}/{section_path}")
        return ""
    except ModuleNotFoundError:
        logger.warning(f"Package not found for prompt section: {package_name}")
        return ""
    except Exception as e:
        logger.warning(f"Failed to load section {package_name}/{section_path}: {e}")
        return ""


def get_shared_section_paths() -> List[str]:
    """Get list of shared section paths available in core.

    Returns:
        List of relative paths to shared sections
    """
    return [
        "shared/role_identity.md",
        "shared/important_rules.md",
        "shared/tool_usage_patterns.md",
    ]
