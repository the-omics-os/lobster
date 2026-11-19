"""
Purge command for removing all Lobster AI files from the system.

This module provides a comprehensive cleanup command that removes:
- Global configuration (~/.config/lobster/)
- License and history (~/.lobster/)
- Workspace data (.lobster_workspace/)

SAFETY: Only removes global directories that contain Lobster AI specific marker
files, to avoid removing files from other software named "lobster", and only a
workspace folder named .lobster_workspace.
"""

import os
import platform
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from rich.markup import escape

from lobster.core.runtime.workspace import (
    WORKSPACE_FOLDER_NAME,
    _is_project_workspace,
    _named_workspace,
)
from lobster.i18n import h

if TYPE_CHECKING:
    from lobster.cli_internal.commands.output_adapter import OutputAdapter


class PurgeScope(str, Enum):
    """Scope of files to purge."""

    GLOBAL = "global"  # ~/.lobster/ and ~/.config/lobster/
    WORKSPACE = "workspace"  # Current .lobster_workspace/
    ALL = "all"  # Everything


@dataclass
class LobsterPaths:
    """
    Centralized definition of all Lobster AI file locations.

    This class provides platform-aware paths and verification methods
    to ensure we only remove Lobster AI specific files.
    """

    # Platform-specific global config directory
    # Unix: ~/.config/lobster/
    # Windows: %APPDATA%/lobster/
    @staticmethod
    def get_global_config_dir() -> Path:
        """Get the global configuration directory."""
        if platform.system() == "Windows":
            return Path(os.environ.get("APPDATA") or Path.home()) / "lobster"
        else:
            return (
                Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
                / "lobster"
            )

    @staticmethod
    def get_lobster_home_dir() -> Path:
        """Get the ~/.lobster directory (license, history, cache)."""
        return Path.home() / ".lobster"

    @staticmethod
    def get_workspace_dir(workspace_path: Optional[Path] = None) -> Path:
        """Get the workspace directory (a project folder names its own workspace)."""
        if workspace_path:
            return _named_workspace(Path(workspace_path))
        # Check LOBSTER_WORKSPACE env var
        env_workspace = os.environ.get("LOBSTER_WORKSPACE")
        if env_workspace:
            return _named_workspace(Path(env_workspace))
        return Path.cwd() / WORKSPACE_FOLDER_NAME

    # Marker files that identify Lobster AI directories
    GLOBAL_CONFIG_MARKERS = ["providers.json", "credentials.env"]
    HOME_DIR_MARKERS = ["license.json", "lobster_history", ".version_check_cache.json"]
    WORKSPACE_MARKERS = ["provider_config.json", ".session.json"]
    # What Lobster writes in its global folders, taken from the writers; the rest is
    # kept. ~/.lobster: the history (ui/console_manager.py), the version cache
    # (config/version_check.py), lobster-vector's vector_store and ontology_cache, and
    # license.json from older releases. The config folder: config/global_config.py
    # and the project registry (core/projects.py).
    HOME_DIR_ENTRIES = HOME_DIR_MARKERS + ["vector_store", "ontology_cache"]
    GLOBAL_CONFIG_ENTRIES = GLOBAL_CONFIG_MARKERS + ["projects.json"]
    OS_FILES = [".DS_Store", "Thumbs.db", "desktop.ini"]
    WORKSPACE_SUBDIRS = ["data", "cache", "plots", "exports", "metadata", ".lobster"]


@dataclass
class PurgeTarget:
    """Represents a directory or file to be purged."""

    path: Path
    description: str
    size_bytes: int = 0
    file_count: int = 0
    is_verified: bool = False  # True if contains Lobster AI marker files
    marker_found: Optional[str] = None
    refusal: Optional[str] = None  # why an unverified workspace is never removed
    kept: List[Path] = field(default_factory=list)  # entries of a global folder kept


@dataclass
class PurgeResult:
    """Result of a purge operation."""

    success: bool
    targets_removed: List[PurgeTarget] = field(default_factory=list)
    targets_skipped: List[PurgeTarget] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    total_size_freed: int = 0


def _get_dir_size(path: Path) -> Tuple[int, int]:
    """
    Calculate total size and file count of a directory.

    Returns:
        Tuple of (total_bytes, file_count)
    """
    total_size = 0
    file_count = 0

    if not path.exists():
        return 0, 0

    if path.is_file():
        return path.stat().st_size, 1

    try:
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total_size += item.stat().st_size
                    file_count += 1
                except (OSError, PermissionError):
                    continue
    except (OSError, PermissionError):
        pass

    return total_size, file_count


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _verify_lobster_directory(
    path: Path, markers: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Verify a directory belongs to Lobster AI by checking for marker files.

    This prevents accidentally removing files from other software named "lobster".

    Args:
        path: Directory to verify
        markers: List of marker file names to check for

    Returns:
        Tuple of (is_verified, marker_found)
    """
    if not path.exists() or not path.is_dir():
        return False, None

    for marker in markers:
        marker_path = path / marker
        if marker_path.exists():
            return True, marker

    return False, None


# Folders checked for a project inside a folder to remove; more is refused unchecked.
_SCAN_LIMIT = 5000


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _reached_through_link(path: Path) -> bool:
    """The folder, the folder named to find its ``.lobster_workspace``, or any
    ``.lobster_workspace`` above it is a link.

    Other links above (``/tmp`` on macOS, a linked home) are the system's, not a redirect.
    """
    named = [path.parent] if path.name == WORKSPACE_FOLDER_NAME else []
    above = [p for p in path.parents if p.name == WORKSPACE_FOLDER_NAME]
    return any(_is_link(p) for p in [path, *named, *above])


def _project_inside(path: Path) -> Optional[str]:
    """A project or workspace inside ``path``; ``""`` when there are too many folders to
    check; None when there is none."""
    from lobster.core.projects import _same, registered_projects

    for entry in registered_projects():
        if not entry.missing and any(
            _same(path, parent) for parent in Path(entry.root).parents
        ):
            return entry.root
    # Projects never registered (moved, copied): every level, without following links.
    level, seen = [path], 0
    while level:
        below = []
        for folder in level:
            try:
                entries = list(os.scandir(folder))
            except OSError:
                continue
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                seen += 1
                if seen > _SCAN_LIMIT:
                    return ""
                inner = Path(entry.path)
                if os.path.lexists(inner / WORKSPACE_FOLDER_NAME):
                    return str(inner)
                below.append(inner)
        level = below
    return None


def _never_removed(path: Path) -> Optional[str]:
    """Why ``path`` must never be removed, whatever it holds; None if it may be."""
    from lobster.core.projects import is_home_or_above

    if is_home_or_above(path):
        return h("Not removed: {path} is your home folder or contains it").format(
            path=path
        )
    if _reached_through_link(path):
        return h(
            "Not removed: {path} is reached through a link; name the real folder instead"
        ).format(path=path)
    nested = path / WORKSPACE_FOLDER_NAME
    if _is_project_workspace(nested):
        return h(
            "Not removed: {path} is a project folder; its workspace is {workspace}"
        ).format(path=path, workspace=nested)
    # A folder that holds a workspace is never itself the workspace, whatever that is.
    if os.path.lexists(nested):
        return h(
            "Not removed: {path} holds a Lobster workspace; name {workspace} instead"
        ).format(path=path, workspace=nested)
    inner = _project_inside(path)
    # A big workspace has many folders; its name already proves Lobster made it.
    if inner == "" and path.name != WORKSPACE_FOLDER_NAME:
        return h(
            "Not removed: {path} holds too many folders to check for projects"
        ).format(path=path)
    if inner:
        return h(
            "Not removed: {path} contains a project or workspace ({inner})"
        ).format(path=path, inner=inner)
    return None


def _workspace_refusal(path: Path) -> Optional[str]:
    """Why ``path`` is not removed as a workspace; None if it is one."""
    if refusal := _never_removed(path):
        return refusal
    if path.name == WORKSPACE_FOLDER_NAME:
        return None
    # A marker file proves only that Lobster once used the folder, not that
    # everything in it is Lobster's: the user's own files may sit beside it.
    return h(
        "Not removed: {path} is not a standard Lobster workspace (.lobster_workspace). "
        "Check what it holds and delete it yourself if it is yours to delete."
    ).format(path=path)


def _global_entries(path: Path) -> Optional[List[str]]:
    """The names Lobster writes in ``path`` when it is a global folder; else None."""
    from lobster.core.projects import _same

    for folder, entries in (
        (LobsterPaths.get_lobster_home_dir(), LobsterPaths.HOME_DIR_ENTRIES),
        (LobsterPaths.get_global_config_dir(), LobsterPaths.GLOBAL_CONFIG_ENTRIES),
    ):
        if _same(path, folder):
            return entries + LobsterPaths.OS_FILES
    return None


def _global_refusal(path: Path) -> Optional[str]:
    """Why the global folder ``path`` is not touched; None if its files may go."""
    from lobster.core.projects import is_home_or_above

    if is_home_or_above(path):
        return h("Not removed: {path} is your home folder or contains it").format(
            path=path
        )
    if _is_link(path):
        return h(
            "Not removed: {path} is reached through a link; name the real folder instead"
        ).format(path=path)
    return None


def _split_entries(path: Path, known: List[str]) -> Tuple[List[Path], List[Path]]:
    """The entries of ``path`` Lobster wrote, and the ones it did not."""
    try:
        entries = sorted(path.iterdir())
    except OSError:  # not a folder, or unreadable: nothing is listed, nothing goes
        return [], []
    return (
        [e for e in entries if e.name in known],
        [e for e in entries if e.name not in known],
    )


def _global_target(path: Path, markers: List[str], description: str) -> PurgeTarget:
    is_verified, marker = _verify_lobster_directory(path, markers)
    refusal = _global_refusal(path)
    removed, kept = _split_entries(path, _global_entries(path) or [])
    sizes = [_get_dir_size(e) for e in removed] if is_verified and not refusal else []
    return PurgeTarget(
        path=path,
        description=description,
        size_bytes=sum(size for size, _ in sizes),
        file_count=sum(count for _, count in sizes),
        is_verified=is_verified and refusal is None,
        marker_found=marker,
        refusal=refusal,
        kept=kept,
    )


def _verify_workspace_directory(path: Path) -> Tuple[bool, Optional[str]]:
    """
    Verify a workspace directory belongs to Lobster AI.

    It must be named .lobster_workspace, and must not be reached through a link,
    hold or contain a project, or be the home folder.
    """
    if not path.exists() or not path.is_dir() or _workspace_refusal(path):
        return False, None
    for marker in LobsterPaths.WORKSPACE_MARKERS:
        if (path / marker).exists():
            return True, marker
    return True, WORKSPACE_FOLDER_NAME


def discover_purge_targets(
    scope: PurgeScope,
    workspace_path: Optional[Path] = None,
) -> List[PurgeTarget]:
    """
    Discover all Lobster AI files that can be purged.

    Only includes directories that are verified to belong to Lobster AI.

    Args:
        scope: What to include (global, workspace, or all)
        workspace_path: Optional explicit workspace path

    Returns:
        List of PurgeTarget objects
    """
    targets = []

    # Global targets
    if scope in (PurgeScope.GLOBAL, PurgeScope.ALL):
        # ~/.lobster/ (license, history, version cache)
        home_dir = LobsterPaths.get_lobster_home_dir()
        if home_dir.exists():
            targets.append(
                _global_target(
                    home_dir,
                    LobsterPaths.HOME_DIR_MARKERS,
                    h("License, history, and version cache"),
                )
            )

        # ~/.config/lobster/ (global config)
        config_dir = LobsterPaths.get_global_config_dir()
        if config_dir.exists():
            targets.append(
                _global_target(
                    config_dir,
                    LobsterPaths.GLOBAL_CONFIG_MARKERS,
                    h("Global configuration and credentials"),
                )
            )

    # Workspace targets
    if scope in (PurgeScope.WORKSPACE, PurgeScope.ALL):
        workspace_dir = LobsterPaths.get_workspace_dir(workspace_path)
        if workspace_dir.exists():
            is_verified, marker = _verify_workspace_directory(workspace_dir)
            # Never walk a folder that is not removed (it may be the whole disk).
            size, count = _get_dir_size(workspace_dir) if is_verified else (0, 0)
            targets.append(
                PurgeTarget(
                    path=workspace_dir,
                    description=h("Current workspace (data, cache, sessions)"),
                    size_bytes=size,
                    file_count=count,
                    is_verified=is_verified,
                    marker_found=marker,
                    refusal=None if is_verified else _workspace_refusal(workspace_dir),
                )
            )

    return targets


def execute_purge(
    targets: List[PurgeTarget],
    dry_run: bool = False,
) -> PurgeResult:
    """
    Execute the purge operation.

    Args:
        targets: List of PurgeTarget objects to remove
        dry_run: If True, don't actually delete anything

    Returns:
        PurgeResult with details of what was done
    """
    result = PurgeResult(success=True)

    for target in targets:
        if not target.is_verified:
            result.targets_skipped.append(target)
            continue

        # Holds under --force too: --force skips only the prompt.
        known = _global_entries(target.path)
        if refusal := (
            _workspace_refusal(target.path)
            if known is None
            else _global_refusal(target.path)
        ):
            result.errors.append(refusal)
            result.success = False
            continue

        if dry_run:
            result.targets_removed.append(target)
            result.total_size_freed += target.size_bytes
            continue

        try:
            if known is not None:
                # A global folder: only what Lobster wrote goes; the folder only if empty.
                removed, target.kept = _split_entries(target.path, known)
                for entry in removed:
                    if entry.is_dir() and not _is_link(entry):
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                if not target.kept:
                    target.path.rmdir()
            elif target.path.is_dir():
                shutil.rmtree(target.path)
            else:
                target.path.unlink()

            result.targets_removed.append(target)
            result.total_size_freed += target.size_bytes

        except PermissionError:
            result.errors.append(
                h("Permission denied: {path}").format(path=target.path)
            )
            result.success = False
        except Exception as e:
            result.errors.append(
                h("Failed to remove {path}: {error}").format(
                    path=target.path, error=str(e)
                )
            )
            result.success = False

    return result


def _skipped_line(target: PurgeTarget, unmarked: str) -> str:
    if target.refusal:
        return f"  [yellow]{escape(target.refusal)}[/yellow]"
    return unmarked.format(path=escape(str(target.path)))


def purge(
    output: "OutputAdapter",
    scope: str = "all",
    workspace_path: Optional[Path] = None,
    dry_run: bool = False,
    force: bool = False,
) -> Optional[str]:
    """
    Remove all Lobster AI files from the system.

    This command removes:
    - ~/.lobster/ (command history, version cache)
    - ~/.config/lobster/ (global provider config, credentials)
    - .lobster_workspace/ (workspace data, cache, sessions)

    SAFETY: Only removes directories verified to contain Lobster AI files.
    Other software named "lobster" will NOT be affected.

    Args:
        output: OutputAdapter for rendering
        scope: What to purge - "global", "workspace", or "all"
        workspace_path: Optional explicit workspace path
        dry_run: If True, show what would be deleted without deleting
        force: If True, skip confirmation prompt

    Returns:
        Summary string for conversation history
    """
    try:
        purge_scope = PurgeScope(scope.lower())
    except ValueError:
        output.print(h("[red]Invalid scope: {scope}[/red]").format(scope=scope))
        output.print(h("[dim]Valid options: global, workspace, all[/dim]"))
        return None

    # Discover targets
    targets = discover_purge_targets(purge_scope, workspace_path)

    if not targets:
        output.print(h("[yellow]No Lobster AI files found to purge.[/yellow]"))
        return "No files to purge"

    # Check if any targets are verified
    verified_targets = [t for t in targets if t.is_verified]
    unverified_targets = [t for t in targets if not t.is_verified]

    if not verified_targets:
        output.print(
            h("[yellow]Found directories but none verified as Lobster AI:[/yellow]")
        )
        for target in unverified_targets:
            output.print(
                _skipped_line(
                    target, h("  • [dim]{path}[/dim] - No marker files found")
                )
            )
        output.print(
            h("\n[dim]These may belong to other software. Not removing.[/dim]")
        )
        return "No verified Lobster AI files found"

    # Calculate totals
    total_size = sum(t.size_bytes for t in verified_targets)
    total_files = sum(t.file_count for t in verified_targets)

    # Display what will be removed
    mode_label = h("[bold yellow]DRY RUN[/bold yellow] - ") if dry_run else ""
    output.print(
        h("\n{mode}[bold cyan]🧹 Lobster AI Purge[/bold cyan]").format(mode=mode_label)
    )
    output.print("[dim]" + "─" * 60 + "[/dim]")

    # Show verified targets
    output.print(h("\n[bold green]✓ Verified Lobster AI directories:[/bold green]"))
    for target in verified_targets:
        size_str = _format_size(target.size_bytes)
        output.print(f"\n  [cyan]{escape(str(target.path))}[/cyan]")
        output.print(f"    {target.description}")
        output.print(
            h("    [dim]{size}, {count} files[/dim]").format(
                size=size_str, count=target.file_count
            )
        )
        output.print(
            h("    [dim]Verified by: {marker}[/dim]").format(marker=target.marker_found)
        )
        for entry in target.kept:
            output.print(
                h("    [yellow]kept: {entry}[/yellow]").format(entry=escape(entry.name))
            )

    # Show unverified targets (won't be removed)
    if unverified_targets:
        output.print(
            h("\n[bold yellow]⚠ Skipping unverified directories:[/bold yellow]")
        )
        for target in unverified_targets:
            output.print(
                _skipped_line(target, h("  [dim]{path}[/dim] - No marker files"))
            )

    # Summary
    output.print("\n[dim]" + "─" * 60 + "[/dim]")
    output.print(
        h("[bold]Total to remove:[/bold] {size} across {count} files").format(
            size=_format_size(total_size), count=total_files
        )
    )

    # Dry run - don't actually delete
    if dry_run:
        output.print(h("\n[yellow]Dry run complete. No files were deleted.[/yellow]"))
        output.print(h("[dim]Remove --dry-run to execute the purge.[/dim]"))
        return f"Dry run: would remove {_format_size(total_size)}"

    # Confirmation unless --force
    if not force:
        output.print(h("\n[bold red]⚠️  This action cannot be undone![/bold red]"))
        # Note: Interactive confirmation would go here in CLI
        # For now, require --force flag
        output.print(h("[yellow]Use --force to confirm deletion.[/yellow]"))
        return "Purge cancelled - use --force to confirm"

    # Execute purge
    output.print(h("\n[bold]Purging...[/bold]"))
    result = execute_purge(verified_targets, dry_run=False)

    # Report results
    if result.success:
        output.print(h("\n[bold green]✓ Purge complete![/bold green]"))
        for target in result.targets_removed:
            removed = (
                h("  [green]✓[/green] Removed Lobster's files from {path}")
                if target.kept
                else h("  [green]✓[/green] Removed {path}")
            )
            output.print(removed.format(path=escape(str(target.path))))
        output.print(
            h("\n[dim]Freed {size}[/dim]").format(
                size=_format_size(result.total_size_freed)
            )
        )
    else:
        output.print(h("\n[bold yellow]⚠ Purge completed with errors:[/bold yellow]"))
        for error in result.errors:
            output.print(f"  [red]✗[/red] {escape(error)}")

    # Show uninstall instructions
    output.print("\n[dim]" + "─" * 60 + "[/dim]")
    output.print(h("\n[bold cyan]📦 To completely uninstall Lobster AI:[/bold cyan]"))
    output.print(h("\n  [bold]Using uv (recommended):[/bold]"))
    output.print("    [white]uv pip uninstall lobster-ai[/white]")
    output.print(h("\n  [bold]Using pip:[/bold]"))
    output.print("    [white]pip uninstall lobster-ai[/white]")
    output.print(h("\n  [bold]Using pipx (if installed via pipx):[/bold]"))
    output.print("    [white]pipx uninstall lobster-ai[/white]")

    return f"Purged {_format_size(result.total_size_freed)} from {len(result.targets_removed)} locations"
