"""Prompt composer for assembling and rendering agent prompts."""

import logging
from typing import Dict, Optional

from jinja2 import BaseLoader, Environment, StrictUndefined, UndefinedError

from lobster.prompts.config import PromptConfig
from lobster.prompts.loader import PromptLoader
from lobster.prompts.registry import get_prompt_registry

logger = logging.getLogger(__name__)


class PromptComposer:
    """Assembles prompt sections with Jinja2 rendering.

    The composer:
    1. Fetches the PromptConfig from registry (or accepts directly)
    2. Loads each section via PromptLoader
    3. Renders Jinja2 templates with merged variables
    4. Joins sections into final prompt string

    Usage:
        >>> composer = get_prompt_composer()
        >>> prompt = composer.compose("transcriptomics_expert", {
        ...     "active_agents": ["annotation_expert", "de_analysis_expert"],
        ...     "workspace_path": "/data/project"
        ... })
    """

    def __init__(self, loader: Optional[PromptLoader] = None):
        """Initialize composer with optional custom loader.

        Args:
            loader: Custom PromptLoader instance (default: creates new one)
        """
        self.loader = loader or PromptLoader()
        self.registry = get_prompt_registry()
        self.env = Environment(
            loader=BaseLoader(),
            undefined=StrictUndefined,  # Catch missing variables at render time
            autoescape=False,  # Markdown content, no HTML escaping
        )

    def compose(
        self,
        agent_name: str,
        runtime_variables: Optional[Dict[str, str]] = None,
        package_name: Optional[str] = None,
    ) -> str:
        """Compose a prompt for an agent using registered config.

        Args:
            agent_name: Agent identifier (must have registered PromptConfig)
            runtime_variables: Variables to inject at render time
            package_name: Package containing agent-specific sections
                         (default: inferred from agent_name)

        Returns:
            Assembled prompt string, empty string if config not found
        """
        config = self.registry.get_prompt_config(agent_name)
        if config is None:
            logger.warning(f"No prompt config found for agent: {agent_name}")
            return ""

        return self.compose_from_config(
            config=config,
            runtime_variables=runtime_variables,
            package_name=package_name or self._infer_package_name(agent_name),
        )

    def compose_from_config(
        self,
        config: PromptConfig,
        runtime_variables: Optional[Dict[str, str]] = None,
        package_name: str = "lobster.prompts",
    ) -> str:
        """Compose a prompt from a PromptConfig directly.

        This method is useful for:
        - Testing with ad-hoc configs
        - Composing prompts before registration
        - Custom composition without registry lookup

        Args:
            config: PromptConfig to compose from
            runtime_variables: Variables to inject at render time
            package_name: Package for non-shared sections

        Returns:
            Assembled prompt string
        """
        sections = []
        runtime_vars = runtime_variables or {}

        for section_spec in config.sections:
            # Determine package: shared sections always from core
            if section_spec.source.startswith("shared/"):
                pkg = "lobster.prompts"
            else:
                pkg = package_name

            # Load section content
            content = self.loader.load_section(pkg, section_spec.source)
            if not content:
                logger.warning(f"Empty or missing section: {pkg}/{section_spec.source}")
                continue

            # Merge variables: section-specific < runtime
            variables = {**section_spec.variables, **runtime_vars}

            # Render Jinja2 template
            try:
                rendered = self.env.from_string(content).render(**variables)
                sections.append(rendered)
            except UndefinedError as e:
                logger.error(f"Missing variable in section {section_spec.source}: {e}")
                # Include unrendered content to avoid silent failures
                sections.append(content)
            except Exception as e:
                logger.error(f"Failed to render section {section_spec.source}: {e}")
                sections.append(content)

        return "\n\n".join(sections)

    def _infer_package_name(self, agent_name: str) -> str:
        """Infer package name from agent name.

        Convention: transcriptomics_expert -> lobster_transcriptomics.prompts

        Args:
            agent_name: Agent identifier

        Returns:
            Inferred package name for agent-specific sections
        """
        # Remove common suffixes to get domain
        domain = agent_name
        for suffix in ("_expert", "_assistant", "_agent"):
            if domain.endswith(suffix):
                domain = domain[: -len(suffix)]
                break

        return f"lobster_{domain}.prompts"

    def validate_config(
        self,
        config: PromptConfig,
        runtime_variables: Optional[Dict[str, str]] = None,
        package_name: str = "lobster.prompts",
    ) -> Dict[str, list]:
        """Validate a prompt config without composing.

        Checks for:
        - Missing sections (file not found)
        - Missing variables (undefined in template)

        Args:
            config: PromptConfig to validate
            runtime_variables: Variables that would be provided at runtime
            package_name: Package for non-shared sections

        Returns:
            Dict with 'missing_sections' and 'missing_variables' lists
        """
        missing_sections = []
        missing_variables = []
        runtime_vars = runtime_variables or {}

        for section_spec in config.sections:
            # Determine package
            if section_spec.source.startswith("shared/"):
                pkg = "lobster.prompts"
            else:
                pkg = package_name

            # Check section exists
            content = self.loader.load_section(pkg, section_spec.source)
            if not content:
                missing_sections.append(f"{pkg}/{section_spec.source}")
                continue

            # Check for missing variables
            variables = {**section_spec.variables, **runtime_vars}
            try:
                self.env.from_string(content).render(**variables)
            except UndefinedError as e:
                missing_variables.append(f"{section_spec.source}: {e}")

        return {
            "missing_sections": missing_sections,
            "missing_variables": missing_variables,
        }


# Module-level singleton accessor
_composer: Optional[PromptComposer] = None


def get_prompt_composer() -> PromptComposer:
    """Get the global PromptComposer singleton.

    Returns:
        The shared PromptComposer instance
    """
    global _composer
    if _composer is None:
        _composer = PromptComposer()
    return _composer
