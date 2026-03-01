"""Agent plugin package generator."""

from pathlib import Path
from typing import List, Optional

import jinja2


_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render_template(template_name: str, context: dict) -> str:
    """Render a Jinja2 template with the given context."""
    template = _env.get_template(template_name)
    return template.render(**context)


def _derive_domain(agent_name: str) -> str:
    """Derive domain name from agent name (strip _expert suffix)."""
    return agent_name.removesuffix("_expert")


def _to_class_name(snake_name: str) -> str:
    """Convert snake_case to PascalCase (e.g., epigenomics_expert -> EpigenomicsExpert)."""
    return "".join(word.capitalize() for word in snake_name.split("_"))


def scaffold_agent(
    name: str,
    display_name: str,
    description: str,
    tier: str = "free",
    children: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    author_name: str = "Lobster AI Community",
    author_email: str = "community@lobsterbio.com",
    python_version: str = "3.12",
) -> Path:
    """Generate a complete agent plugin package.

    Args:
        name: Agent name in snake_case (e.g., "epigenomics_expert")
        display_name: Human-readable name (e.g., "Epigenomics Expert")
        description: Agent capabilities description
        tier: Subscription tier ("free", "premium", "enterprise")
        children: Optional list of child agent names
        output_dir: Where to create the package (default: current directory)
        author_name: Package author name
        author_email: Package author email
        python_version: Minimum Python version

    Returns:
        Path to the generated package directory
    """
    # Implemented in Task 1.7
    raise NotImplementedError("scaffold_agent() not yet implemented")
