"""
Purge command for removing all Lobster AI files from the system.

This module provides a comprehensive cleanup command that removes:
- Global configuration (~/.config/lobster/)
- License and history (~/.lobster/)
- Workspace data (.lobster_workspace/)

SAFETY: Only removes directories that contain Lobster AI specific marker files
to avoid removing files from other software named "lobster".
"""

import os
import platform
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

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
            return Path(os.environ.get("APPDATA", Path.home())) / "lobster"
        else:
            return (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "lobster"
            )

    @staticmethod
    def get_lobster_home_dir() -> Path:
        """Get the ~/.lobster directory (license, history, cache)."""
        return Path.home() / ".lobster"

    @staticmethod
    def get_workspace_dir(workspace_path: Optional[Path] = None) -> Path:
        """Get the workspace directory."""
        if workspace_path:
            return workspace_path
        # Check LOBSTER_WORKSPACE env var
        env_workspace = os.environ.get("LOBSTER_WORKSPACE")
        if env_workspace:
            return Path(env_workspace)
        return Path.cwd() / ".lobster_workspace"

    # Marker files that identify Lobster AI directories
    GLOBAL_CONFIG_MARKERS = ["providers.json", "credentials.env"]
    HOME_DIR_MARKERS = ["license.json", "lobster_history", ".version_check_cache.json"]
    WORKSPACE_MARKERS = ["provider_config.json", ".session.json"]
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


def _verify_workspace_directory(path: Path) -> Tuple[bool, Optional[str]]:
    """
    Verify a workspace directory belongs to Lobster AI.

    Checks for:
    1. Marker files (provider_config.json, .session.json)
    2. Standard subdirectory structure (data/, cache/, plots/)
    """
    if not path.exists() or not path.is_dir():
        return False, None

    # Check marker files
    for marker in LobsterPaths.WORKSPACE_MARKERS:
        marker_path = path / marker
        if marker_path.exists():
            return True, marker

    # Check for characteristic subdirectory structure
    # If at least 2 of these exist, it's likely our workspace
    found_subdirs = 0
    for subdir in LobsterPaths.WORKSPACE_SUBDIRS:
        if (path / subdir).exists():
            found_subdirs += 1

    if found_subdirs >= 2:
        return True, f"{found_subdirs} standard subdirs"

    return False, None


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
            is_verified, marker = _verify_lobster_directory(
                home_dir, LobsterPaths.HOME_DIR_MARKERS
            )
            size, count = _get_dir_size(home_dir)
            targets.append(
                PurgeTarget(
                    path=home_dir,
                    description="License, history, and version cache",
                    size_bytes=size,
                    file_count=count,
                    is_verified=is_verified,
                    marker_found=marker,
                )
            )

        # ~/.config/lobster/ (global config)
        config_dir = LobsterPaths.get_global_config_dir()
        if config_dir.exists():
            is_verified, marker = _verify_lobster_directory(
                config_dir, LobsterPaths.GLOBAL_CONFIG_MARKERS
            )
            size, count = _get_dir_size(config_dir)
            targets.append(
                PurgeTarget(
                    path=config_dir,
                    description="Global configuration and credentials",
                    size_bytes=size,
                    file_count=count,
                    is_verified=is_verified,
                    marker_found=marker,
                )
            )

    # Workspace targets
    if scope in (PurgeScope.WORKSPACE, PurgeScope.ALL):
        workspace_dir = LobsterPaths.get_workspace_dir(workspace_path)
        if workspace_dir.exists():
            is_verified, marker = _verify_workspace_directory(workspace_dir)
            size, count = _get_dir_size(workspace_dir)
            targets.append(
                PurgeTarget(
                    path=workspace_dir,
                    description="Current workspace (data, cache, sessions)",
                    size_bytes=size,
                    file_count=count,
                    is_verified=is_verified,
                    marker_found=marker,
                )
            )

    return targets


def execute_purge(
    targets: List[PurgeTarget],
    keep_license: bool = False,
    dry_run: bool = False,
) -> PurgeResult:
    """
    Execute the purge operation.

    Args:
        targets: List of PurgeTarget objects to remove
        keep_license: If True, preserve ~/.lobster/license.json
        dry_run: If True, don't actually delete anything

    Returns:
        PurgeResult with details of what was done
    """
    result = PurgeResult(success=True)

    for target in targets:
        # Skip unverified directories for safety
        if not target.is_verified:
            result.targets_skipped.append(target)
            continue

        if dry_run:
            result.targets_removed.append(target)
            result.total_size_freed += target.size_bytes
            continue

        try:
            # Special handling for ~/.lobster/ with --keep-license
            if keep_license and target.path == LobsterPaths.get_lobster_home_dir():
                license_path = target.path / "license.json"
                license_backup = None

                # Backup license if it exists
                if license_path.exists():
                    license_backup = license_path.read_text()

                # Remove directory
                shutil.rmtree(target.path)

                # Restore license
                if license_backup:
                    target.path.mkdir(parents=True, exist_ok=True)
                    license_path.write_text(license_backup)
            else:
                # Normal removal
                if target.path.is_dir():
                    shutil.rmtree(target.path)
                else:
                    target.path.unlink()

            result.targets_removed.append(target)
            result.total_size_freed += target.size_bytes

        except PermissionError:
            result.errors.append(f"Permission denied: {target.path}")
            result.success = False
        except Exception as e:
            result.errors.append(f"Failed to remove {target.path}: {str(e)}")
            result.success = False

    return result


def purge(
    output: "OutputAdapter",
    scope: str = "all",
    workspace_path: Optional[Path] = None,
    keep_license: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> Optional[str]:
    """
    Remove all Lobster AI files from the system.

    This command removes:
    - ~/.lobster/ (license, command history, version cache)
    - ~/.config/lobster/ (global provider config, credentials)
    - .lobster_workspace/ (workspace data, cache, sessions)

    SAFETY: Only removes directories verified to contain Lobster AI files.
    Other software named "lobster" will NOT be affected.

    Args:
        output: OutputAdapter for rendering
        scope: What to purge - "global", "workspace", or "all"
        workspace_path: Optional explicit workspace path
        keep_license: If True, preserve license.json
        dry_run: If True, show what would be deleted without deleting
        force: If True, skip confirmation prompt

    Returns:
        Summary string for conversation history
    """
    try:
        purge_scope = PurgeScope(scope.lower())
    except ValueError:
        output.print(f"[red]Invalid scope: {scope}[/red]")
        output.print("[dim]Valid options: global, workspace, all[/dim]")
        return None

    # Discover targets
    targets = discover_purge_targets(purge_scope, workspace_path)

    if not targets:
        output.print("[yellow]No Lobster AI files found to purge.[/yellow]")
        return "No files to purge"

    # Check if any targets are verified
    verified_targets = [t for t in targets if t.is_verified]
    unverified_targets = [t for t in targets if not t.is_verified]

    if not verified_targets:
        output.print(
            "[yellow]Found directories but none verified as Lobster AI:[/yellow]"
        )
        for target in unverified_targets:
            output.print(f"  • [dim]{target.path}[/dim] - No marker files found")
        output.print("\n[dim]These may belong to other software. Not removing.[/dim]")
        return "No verified Lobster AI files found"

    # Calculate totals
    total_size = sum(t.size_bytes for t in verified_targets)
    total_files = sum(t.file_count for t in verified_targets)

    # Display what will be removed
    mode_label = "[bold yellow]DRY RUN[/bold yellow] - " if dry_run else ""
    output.print(f"\n{mode_label}[bold cyan]🧹 Lobster AI Purge[/bold cyan]")
    output.print("[dim]" + "─" * 60 + "[/dim]")

    # Show verified targets
    output.print("\n[bold green]✓ Verified Lobster AI directories:[/bold green]")
    for target in verified_targets:
        size_str = _format_size(target.size_bytes)
        output.print(f"\n  [cyan]{target.path}[/cyan]")
        output.print(f"    {target.description}")
        output.print(f"    [dim]{size_str}, {target.file_count} files[/dim]")
        output.print(f"    [dim]Verified by: {target.marker_found}[/dim]")

    # Show unverified targets (won't be removed)
    if unverified_targets:
        output.print("\n[bold yellow]⚠ Skipping unverified directories:[/bold yellow]")
        for target in unverified_targets:
            output.print(f"  [dim]{target.path}[/dim] - No marker files")

    # Summary
    output.print("\n[dim]" + "─" * 60 + "[/dim]")
    output.print(
        f"[bold]Total to remove:[/bold] {_format_size(total_size)} across {total_files} files"
    )

    if keep_license:
        output.print("[dim]License file will be preserved[/dim]")

    # Dry run - don't actually delete
    if dry_run:
        output.print("\n[yellow]Dry run complete. No files were deleted.[/yellow]")
        output.print("[dim]Remove --dry-run to execute the purge.[/dim]")
        return f"Dry run: would remove {_format_size(total_size)}"

    # Confirmation unless --force
    if not force:
        output.print("\n[bold red]⚠️  This action cannot be undone![/bold red]")
        # Note: Interactive confirmation would go here in CLI
        # For now, require --force flag
        output.print("[yellow]Use --force to confirm deletion.[/yellow]")
        return "Purge cancelled - use --force to confirm"

    # Execute purge
    output.print("\n[bold]Purging...[/bold]")
    result = execute_purge(verified_targets, keep_license=keep_license, dry_run=False)

    # Report results
    if result.success:
        output.print("\n[bold green]✓ Purge complete![/bold green]")
        for target in result.targets_removed:
            output.print(f"  [green]✓[/green] Removed {target.path}")
        output.print(f"\n[dim]Freed {_format_size(result.total_size_freed)}[/dim]")
    else:
        output.print("\n[bold yellow]⚠ Purge completed with errors:[/bold yellow]")
        for error in result.errors:
            output.print(f"  [red]✗[/red] {error}")

    # Show uninstall instructions
    output.print("\n[dim]" + "─" * 60 + "[/dim]")
    output.print("\n[bold cyan]📦 To completely uninstall Lobster AI:[/bold cyan]")
    output.print("\n  [bold]Using uv (recommended):[/bold]")
    output.print("    [white]uv pip uninstall lobster-ai[/white]")
    output.print("\n  [bold]Using pip:[/bold]")
    output.print("    [white]pip uninstall lobster-ai[/white]")
    output.print("\n  [bold]Using pipx (if installed via pipx):[/bold]")
    output.print("    [white]pipx uninstall lobster-ai[/white]")

    return f"Purged {_format_size(result.total_size_freed)} from {len(result.targets_removed)} locations"
