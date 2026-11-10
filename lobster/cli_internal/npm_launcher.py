"""Launch the @omicsos/lobster npm CLI for cloud interactive sessions."""

import json
import os
import shutil
import subprocess  # nosec B404 # used only to launch the installed chat client with an argv list
import sys
from typing import Optional

NPM_PACKAGE = "@omicsos/lobster"

# Environment variable the npm client reads its auth token from (argv wins if both are set).
TOKEN_ENV_VAR = "LOBSTER_CLOUD_TOKEN"
CAPABILITIES_TIMEOUT_S = 3.0


def install_hint() -> str:
    """Return the command that installs the npm cloud TUI."""
    return f"npm install -g {NPM_PACKAGE}"


def find_npm_binary() -> Optional[str]:
    """Find the @omicsos/lobster npm binary via the lobster-cli discovery name.

    Returns the path to the binary, or None if not installed.
    """
    env = os.environ.get("LOBSTER_CLI_BINARY")
    if env:
        return env
    return shutil.which("lobster-cli")


def supports_token_env(binary: str) -> bool:
    """Return True if the installed client accepts its token from the environment.

    Runs ``<binary> capabilities``. Only exit 0 within the timeout with a JSON object
    whose ``"token-env"`` is ``true`` counts; anything else (missing binary, non-zero
    exit, timeout, unparsable output) means an older client that needs ``--token``.
    """
    try:
        proc = subprocess.Popen(  # nosec B603 # fixed argv for the installed chat client; no shell
            [binary, "capabilities"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError):
        return False
    try:
        stdout, _ = proc.communicate(timeout=CAPABILITIES_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # Do not drain the pipe after the kill: on Windows a shim's child can keep it open.
        proc.kill()
        return False
    if proc.returncode != 0:
        return False
    try:
        caps = json.loads(stdout)
    except ValueError:  # includes undecodable bytes
        return False
    return isinstance(caps, dict) and caps.get("token-env") is True


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
        result = subprocess.run(cmd)  # nosec B603 # chat client argv list; no shell
        sys.exit(result.returncode)
    except FileNotFoundError:
        return False
