"""Prompt management infrastructure for Lobster AI agents.

This module provides the infrastructure for YAML+MD prompt format:
- PromptConfig/SectionSpec: Pydantic models for YAML configs
- PromptLoader: File I/O with LRU caching
- PromptRegistry: Singleton for config discovery
- PromptComposer: Section assembly with Jinja2 rendering

Example:
    >>> from lobster.prompts import (
    ...     PromptConfig, SectionSpec, PromptLoader,
    ...     get_prompt_registry, get_prompt_composer
    ... )
    >>> registry = get_prompt_registry()
    >>> config = PromptConfig(
    ...     agent_name='test_agent',
    ...     sections=[SectionSpec(source='shared/role_identity.md')]
    ... )
    >>> registry.register_config(config)
    >>> composer = get_prompt_composer()
    >>> prompt = composer.compose('test_agent', {'agent_name': 'Test'})
"""

from lobster.prompts.composer import PromptComposer, get_prompt_composer
from lobster.prompts.config import PromptConfig, SectionSpec
from lobster.prompts.loader import PromptLoader
from lobster.prompts.registry import PromptRegistry, get_prompt_registry

__all__ = [
    "PromptConfig",
    "SectionSpec",
    "PromptLoader",
    "PromptRegistry",
    "get_prompt_registry",
    "PromptComposer",
    "get_prompt_composer",
]
