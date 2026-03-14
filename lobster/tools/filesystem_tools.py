"""
Filesystem tools factory for data_expert agent.

Cherry-picked from DeepAgents' FilesystemMiddleware patterns, adapted to Lobster
conventions: @tool decorator, AQUADIF UTILITY metadata, workspace-scoped paths.

These tools give agents autonomous file-level capabilities: listing, reading,
writing, searching, and shell execution within the workspace boundary.

Architecture:
    Factory Pattern: create_filesystem_tools(workspace_path) returns list of @tool functions
    Path Safety: All paths resolved relative to workspace_path, traversal blocked
    AQUADIF: All tools are UTILITY category, no provenance IR

Usage:
    tools = create_filesystem_tools(workspace_path=data_manager.workspace_path)
    # Returns: [list_files, read_file, write_file, glob_files, grep_files, shell_execute]

See Also:
    - workspace_tool.py: Established factory pattern for shared tools
    - custom_code_tool.py: Another factory pattern example
    - DeepAgents FilesystemMiddleware: Pattern source (libs/deepagents/middleware/filesystem.py)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Limits
MAX_OUTPUT_CHARS = 30000  # Cap tool output to prevent context overflow
MAX_READ_LINES = 2000  # Default max lines for read_file
DEFAULT_EXECUTE_TIMEOUT = 120  # seconds
MAX_EXECUTE_TIMEOUT = 600  # seconds
MAX_GLOB_RESULTS = 500  # Max files returned by glob
BINARY_EXTENSIONS = frozenset(
    {
        ".h5",
        ".h5ad",
        ".bam",
        ".sam",
        ".bcf",
        ".gz",
        ".tar",
        ".zip",
        ".bz2",
        ".xz",
        ".pkl",
        ".pickle",
        ".npy",
        ".npz",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".pdf",
        ".pyc",
    }
)


def _resolve_safe_path(workspace_path: Path, relative_path: str) -> Path:
    """
    Resolve a relative path safely within the workspace boundary.

    Prevents directory traversal attacks by ensuring the resolved path
    is within workspace_path.

    Args:
        workspace_path: Root workspace directory
        relative_path: User-provided relative path

    Returns:
        Resolved absolute Path

    Raises:
        ValueError: If path escapes workspace boundary
    """
    input_path = Path(relative_path).expanduser()
    resolved = (
        input_path.resolve()
        if input_path.is_absolute()
        else (workspace_path / input_path).resolve()
    )
    workspace_resolved = workspace_path.resolve()

    if resolved != workspace_resolved and workspace_resolved not in resolved.parents:
        raise ValueError(
            f"Path '{relative_path}' resolves outside workspace boundary. "
            f"All paths must be relative to the workspace directory."
        )
    return resolved


def _truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate output with indicator if too long."""
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n\n... [TRUNCATED: {len(text):,} chars total, showing first {max_chars:,}]"
    )


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


def create_filesystem_tools(workspace_path: Path) -> List:
    """
    Factory that creates workspace-scoped filesystem tools.

    All tools operate relative to workspace_path. Path traversal outside
    workspace is blocked. Follows Lobster factory pattern (workspace_tool.py).

    Args:
        workspace_path: Root directory for all file operations

    Returns:
        List of 6 LangChain tools with AQUADIF UTILITY metadata
    """
    workspace_path = Path(workspace_path).resolve()

    # =========================================================================
    # list_files
    # =========================================================================
    @tool
    def list_files(path: str = ".") -> str:
        """List files and directories at the given path within the workspace.

        Returns file names with size and type indicators. Use this to explore
        the workspace structure before reading or editing files.

        Args:
            path: Relative path within workspace (default: workspace root ".")

        Returns:
            Formatted directory listing with file sizes and types
        """
        try:
            resolved = _resolve_safe_path(workspace_path, path)
        except ValueError as e:
            return f"Error: {e}"

        if not resolved.exists():
            return f"Error: Path '{path}' not found in workspace."
        if not resolved.is_dir():
            return f"Error: '{path}' is a file, not a directory. Use read_file to inspect it."

        entries = []
        try:
            for item in sorted(resolved.iterdir()):
                rel = item.relative_to(workspace_path)
                if item.is_dir():
                    entries.append(f"  {rel}/")
                else:
                    size = _format_file_size(item.stat().st_size)
                    entries.append(f"  {rel}  ({size})")
        except PermissionError:
            return f"Error: Permission denied reading '{path}'."

        if not entries:
            return f"Directory '{path}' is empty."

        header = f"Contents of {path}/ ({len(entries)} items):\n"
        return _truncate_output(header + "\n".join(entries))

    list_files.metadata = {"categories": ["UTILITY"], "provenance": False}
    list_files.tags = ["UTILITY"]

    # =========================================================================
    # read_file
    # =========================================================================
    @tool
    def read_file(
        path: str,
        offset: int = 0,
        limit: int = MAX_READ_LINES,
    ) -> str:
        """Read a file's contents with optional pagination.

        Returns content in cat -n format (line numbers). For large files, use
        offset and limit to paginate. Binary files are detected and skipped.

        Args:
            path: Relative path to file within workspace
            offset: Line number to start reading from (0-indexed, default: 0)
            limit: Maximum number of lines to read (default: 2000)

        Returns:
            File contents with line numbers, or error message
        """
        try:
            resolved = _resolve_safe_path(workspace_path, path)
        except ValueError as e:
            return f"Error: {e}"

        if not resolved.exists():
            return f"Error: File '{path}' not found in workspace."
        if resolved.is_dir():
            return f"Error: '{path}' is a directory. Use list_files instead."

        # Binary file detection
        if resolved.suffix.lower() in BINARY_EXTENSIONS:
            size = _format_file_size(resolved.stat().st_size)
            return (
                f"Binary file: {path} ({size})\n"
                f"Type: {resolved.suffix}\n"
                f"Use shell_execute to inspect binary files "
                f"(e.g., 'h5ls {path}' for H5 files, 'file {path}' for type detection)."
            )

        try:
            with open(resolved, "r", errors="replace") as f:
                all_lines = f.readlines()
        except PermissionError:
            return f"Error: Permission denied reading '{path}'."

        total_lines = len(all_lines)
        selected = all_lines[offset : offset + limit]

        if not selected:
            if total_lines == 0:
                return f"File '{path}' exists but is empty."
            return (
                f"Error: offset {offset} is beyond end of file ({total_lines} lines)."
            )

        # Format with line numbers (cat -n style)
        numbered = []
        for i, line in enumerate(selected, start=offset + 1):
            numbered.append(f"{i:6d}\t{line.rstrip()}")

        result = "\n".join(numbered)

        # Add pagination info if file is larger than what we showed
        if total_lines > offset + limit:
            remaining = total_lines - (offset + limit)
            result += (
                f"\n\n[Showing lines {offset + 1}-{offset + len(selected)} "
                f"of {total_lines} total. {remaining} more lines available. "
                f"Use offset={offset + limit} to continue.]"
            )

        return _truncate_output(result)

    read_file.metadata = {"categories": ["UTILITY"], "provenance": False}
    read_file.tags = ["UTILITY"]

    # =========================================================================
    # write_file
    # =========================================================================
    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file in the workspace.

        Creates the file if it doesn't exist, overwrites if it does.
        Parent directories are created automatically.

        Args:
            path: Relative path within workspace
            content: Content to write

        Returns:
            Success or error message
        """
        try:
            resolved = _resolve_safe_path(workspace_path, path)
        except ValueError as e:
            return f"Error: {e}"

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content)
            size = _format_file_size(len(content.encode()))
            return f"Wrote {size} to {path}"
        except PermissionError:
            return f"Error: Permission denied writing '{path}'."
        except Exception as e:
            return f"Error writing '{path}': {e}"

    write_file.metadata = {"categories": ["UTILITY"], "provenance": False}
    write_file.tags = ["UTILITY"]

    # =========================================================================
    # glob_files
    # =========================================================================
    @tool
    def glob_files(pattern: str, path: str = ".") -> str:
        """Find files matching a glob pattern within the workspace.

        Supports standard glob patterns: * (any chars), ** (recursive),
        ? (single char).

        Args:
            pattern: Glob pattern (e.g., "**/*.csv", "data/*.h5ad")
            path: Base directory for search (default: workspace root)

        Returns:
            List of matching file paths relative to workspace
        """
        try:
            resolved = _resolve_safe_path(workspace_path, path)
        except ValueError as e:
            return f"Error: {e}"

        if not resolved.exists():
            return f"Error: Path '{path}' not found."

        matches = []
        try:
            for match in sorted(resolved.glob(pattern)):
                if match.is_file():
                    rel = match.relative_to(workspace_path)
                    size = _format_file_size(match.stat().st_size)
                    matches.append(f"  {rel}  ({size})")
                if len(matches) >= MAX_GLOB_RESULTS:
                    matches.append(f"\n  ... [truncated at {MAX_GLOB_RESULTS} results]")
                    break
        except Exception as e:
            return f"Error during glob: {e}"

        if not matches:
            return f"No matches found for pattern '{pattern}' in {path}/"

        header = f"Found {len(matches)} file(s) matching '{pattern}':\n"
        return _truncate_output(header + "\n".join(matches))

    glob_files.metadata = {"categories": ["UTILITY"], "provenance": False}
    glob_files.tags = ["UTILITY"]

    # =========================================================================
    # grep_files
    # =========================================================================
    @tool
    def grep_files(
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
    ) -> str:
        """Search for a text pattern across files in the workspace.

        Searches for literal text (not regex) and returns matching lines
        with file paths and line numbers. Use glob parameter to filter
        which files to search.

        Args:
            pattern: Text to search for (literal, not regex)
            path: Directory to search in (default: workspace root)
            glob: Optional glob filter for files (e.g., "*.csv", "*.py")

        Returns:
            Matching lines with file:line_number format
        """
        try:
            resolved = _resolve_safe_path(workspace_path, path)
        except ValueError as e:
            return f"Error: {e}"

        if not resolved.exists():
            return f"Error: Path '{path}' not found."

        matches = []
        files_searched = 0
        search_pattern = pattern.lower()

        def _search_file(file_path: Path):
            nonlocal files_searched
            # Skip binary files
            if file_path.suffix.lower() in BINARY_EXTENSIONS:
                return
            files_searched += 1
            try:
                with open(file_path, "r", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        if search_pattern in line.lower():
                            rel = file_path.relative_to(workspace_path)
                            matches.append(f"  {rel}:{line_num}: {line.rstrip()}")
                            if len(matches) >= 200:  # Cap matches
                                return
            except (PermissionError, IsADirectoryError):
                pass

        if resolved.is_file():
            _search_file(resolved)
        else:
            file_iter = resolved.rglob(glob or "*")
            for file_path in file_iter:
                if file_path.is_file():
                    _search_file(file_path)
                    if len(matches) >= 200:
                        break

        if not matches:
            return (
                f"No matches for '{pattern}' "
                f"({files_searched} files searched in {path}/)"
            )

        header = (
            f"Found {len(matches)} match(es) for '{pattern}' "
            f"({files_searched} files searched):\n"
        )
        if len(matches) >= 200:
            header += "  [Results capped at 200 matches]\n"
        return _truncate_output(header + "\n".join(matches))

    grep_files.metadata = {"categories": ["UTILITY"], "provenance": False}
    grep_files.tags = ["UTILITY"]

    # =========================================================================
    # shell_execute
    # =========================================================================
    @tool
    def shell_execute(
        command: str,
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
    ) -> str:
        """Execute a shell command in the workspace directory.

        Runs the command with the workspace as working directory. Use for
        file operations not covered by other tools: tar, gunzip, head, wc,
        file type detection, h5ls, samtools, etc.

        Args:
            command: Shell command to execute
            timeout: Maximum execution time in seconds (default: 120, max: 600)

        Returns:
            Command output (stdout + stderr) with exit code
        """
        timeout = min(timeout, MAX_EXECUTE_TIMEOUT)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "HOME": os.environ.get("HOME", "/tmp")},
            )

            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if result.returncode != 0:
                output = f"[exit_code: {result.returncode}]\n{output}"

            return _truncate_output(output)

        except subprocess.TimeoutExpired:
            return (
                f"Error: Command timed out after {timeout}s. "
                f"Consider increasing timeout or breaking into smaller operations."
            )
        except Exception as e:
            return f"Error executing command: {e}"

    shell_execute.metadata = {"categories": ["UTILITY"], "provenance": False}
    shell_execute.tags = ["UTILITY"]

    # =========================================================================
    # Return all tools
    # =========================================================================
    return [
        list_files,
        read_file,
        write_file,
        glob_files,
        grep_files,
        shell_execute,
    ]
