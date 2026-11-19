"""
Centralized workspace path resolution for consistent workspace location across all entry points.

Resolution Order (first match wins):
1. Explicit path parameter (passed programmatically)
2. LOBSTER_WORKSPACE environment variable
3. The workspace of the project found by walking up from the current directory
   (see ``find_project_root``)
4. Current working directory + ".lobster_workspace"

Example:
    >>> from lobster.core.workspace import resolve_workspace
    >>> workspace = resolve_workspace()  # Uses env or cwd fallback
    >>> workspace = resolve_workspace("/custom/path")  # Uses explicit path
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Standard workspace folder name
WORKSPACE_FOLDER_NAME = ".lobster_workspace"

# Environment variable for workspace override
WORKSPACE_ENV_VAR = "LOBSTER_WORKSPACE"

# Identity file of a project, inside its workspace folder
PROJECT_FILE_NAME = "project.json"


def _device(path: Path) -> int:
    return os.stat(path).st_dev


def _is_project_workspace(workspace: Path) -> bool:
    # A link is never trusted as the identity of a project: it could point anywhere.
    project_file = workspace / PROJECT_FILE_NAME
    return (
        not workspace.is_symlink()
        and project_file.is_file()
        and not project_file.is_symlink()
    )


def _is_home(folder: Path, home: str) -> bool:
    """Same folder as ``home`` by identity, so links and letter case do not hide it."""
    try:
        return os.path.samefile(folder, home)
    except OSError:
        return False


def find_project_root(start: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Return the project folder for ``start`` (default: the current directory).

    Walks upward from ``start``. The first folder that holds a
    ``.lobster_workspace`` decides: it is a project when it has a
    ``project.json``, otherwise (an older workspace, or a symlinked one) the
    walk stops and nothing is adopted. The walk never goes above the user's home
    folder (whose own workspace is never a project), never crosses into another filesystem and
    never inspects the filesystem root.
    """
    current = Path(os.path.abspath(Path.cwd() if start is None else start))
    home = os.path.realpath(Path.home())
    try:
        while True:
            parent = current.parent
            if parent == current:
                return None
            workspace = current / WORKSPACE_FOLDER_NAME
            if workspace.is_dir():
                if _is_home(current, home):
                    return None  # the home workspace is the per-user fallback, not a project
                return current if _is_project_workspace(workspace) else None
            if _is_home(current, home) or parent.parent == parent:
                return None
            if _device(parent) != _device(current):
                return None
            current = parent
    except OSError:
        return None


def resolve_workspace(
    explicit_path: Optional[Union[str, Path]] = None,
    create: bool = True,
) -> Path:
    """
    Resolve the workspace path using a consistent priority order.

    Args:
        explicit_path: Explicitly provided workspace path (highest priority)
        create: Whether to create the workspace directory if it doesn't exist

    Returns:
        Path: Resolved workspace path (absolute)

    Resolution Order:
        1. explicit_path (if provided); a project folder names its own workspace
        2. LOBSTER_WORKSPACE environment variable
        3. The workspace of the project that contains the current directory
        4. Path.cwd() / ".lobster_workspace" (default fallback)

    Example:
        >>> # Use environment variable or default
        >>> workspace = resolve_workspace()

        >>> # Override with explicit path
        >>> workspace = resolve_workspace("/custom/workspace")

        >>> # Check path without creating directory
        >>> workspace = resolve_workspace(create=False)
    """
    workspace: Path
    source: str

    if explicit_path is not None:
        workspace = Path(explicit_path)
        source = "explicit parameter"
        if _is_project_workspace(workspace / WORKSPACE_FOLDER_NAME):
            workspace = workspace / WORKSPACE_FOLDER_NAME  # -w <project folder>
            source = "explicit project folder"
    elif env_workspace := os.environ.get(WORKSPACE_ENV_VAR):
        workspace = Path(env_workspace)
        source = f"${WORKSPACE_ENV_VAR} environment variable"
    elif project_root := find_project_root():
        workspace = project_root / WORKSPACE_FOLDER_NAME
        source = "project found from the current directory"
    else:
        workspace = Path.cwd() / WORKSPACE_FOLDER_NAME
        source = "current working directory"

    # Convert to absolute path without following symlinks
    # Note: Using absolute() instead of resolve() to preserve symlinks
    # This avoids issues on macOS where /var -> /private/var
    if not workspace.is_absolute():
        workspace = Path.cwd() / workspace
    workspace = Path(os.path.abspath(workspace))

    logger.debug(f"Resolved workspace to {workspace} (source: {source})")

    if create:
        workspace.mkdir(parents=True, exist_ok=True)

    return workspace


def get_workspace_env_var() -> str:
    """Return the name of the workspace environment variable.

    Returns:
        str: The environment variable name (LOBSTER_WORKSPACE)
    """
    return WORKSPACE_ENV_VAR


def get_workspace_folder_name() -> str:
    """Return the default workspace folder name.

    Returns:
        str: The folder name (.lobster_workspace)
    """
    return WORKSPACE_FOLDER_NAME
