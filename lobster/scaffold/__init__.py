"""
Lobster AI Plugin Scaffold Generator.

Generates structurally correct, AQUADIF-compliant plugin packages
from Jinja2 templates. Called via `lobster scaffold` CLI or programmatically.

Usage:
    # CLI
    lobster scaffold agent --name epigenomics_expert --tier free

    # Programmatic
    from lobster.scaffold import scaffold_agent
    scaffold_agent(name="epigenomics_expert", tier="free", output_dir=Path("."))
"""

from lobster.scaffold.generators.agent import scaffold_agent

__all__ = ["scaffold_agent"]
