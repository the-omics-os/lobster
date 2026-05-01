"""Launch the @omicsos/lobster npm CLI for cloud interactive sessions."""

import os
import shutil
import subprocess
import sys
from typing import Optional


def find_npm_binary() -> Optional[str]:
    """Find the @omicsos/lobster npm binary via the lobster-cli discovery name.

    Returns the path to the binary, or None if not installed.
    """
    env = os.environ.get("LOBSTER_CLI_BINARY")
    if env:
        return env
    return shutil.which("lobster-cli")


def launch_cloud_chat(
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> bool:
    """Launch the npm cloud chat TUI.

    Returns True if launched successfully, False if binary not found.
    Exits with the subprocess return code on success.
    """
    binary = find_npm_binary()
    if not binary:
        return False
    cmd = [binary, "chat", "--cloud"]
    if session_id:
        cmd.extend(["--session-id", session_id])
    if project_id:
        cmd.extend(["--project-id", project_id])
    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except FileNotFoundError:
        return False
