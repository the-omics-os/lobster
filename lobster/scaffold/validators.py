"""
Post-generation plugin validation.

Provides AST-level and structural checks for generated plugin packages.
Used by `lobster validate-plugin <dir>` CLI command.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ValidationError:
    """A single validation failure."""

    file: str
    line: int
    rule: str
    message: str
    severity: str  # "error" or "warning"


def validate_plugin(plugin_dir: Path) -> List[ValidationError]:
    """Validate a plugin package for contract compliance.

    Args:
        plugin_dir: Root directory of the plugin package

    Returns:
        List of validation errors (empty = all checks pass)
    """
    # Implemented in Task 3.1
    return []
