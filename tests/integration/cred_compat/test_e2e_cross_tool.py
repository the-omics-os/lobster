"""E2E cross-tool contract — the real npm CLI and Python Lobster stay apart.

GATED (``@pytest.mark.real_api`` → skipped unless ``--runreal``). Requires the
real npm Cloud client (``omics-os``) on PATH, signed in to a THROWAWAY test
tenant. Since 1.3.2 the two tools keep separate sign-ins: Python Lobster neither
reads nor changes ~/.config/omics-os, so its commands must leave the npm sign-in
byte-identical and working, and must not see it as a Lobster key.

Run pre-release, by a human, against a THROWAWAY tenant:
    uv run pytest tests/integration/cred_compat/test_e2e_cross_tool.py --runreal -v

These tests do NOT run in CI and never write real tokens into fixtures. Python
runs with its own settings in a tmp XDG_CONFIG_HOME, so the developer's saved
Lobster key is neither read nor removed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_api]


@pytest.fixture
def real_sign_in():
    """The npm CLI's REAL sign-in file (throwaway tenant), restored after."""
    cred = Path.home() / ".config" / "omics-os" / "credentials.json"
    if not cred.exists():
        pytest.skip("No npm sign-in: run omics-os login first (throwaway tenant).")
    backup = cred.read_bytes()
    try:
        yield cred
    finally:
        cred.write_bytes(backup)  # restore original bytes


def _py_cli(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "xdg"), "NO_COLOR": "1"}
    env.pop("OMICS_OS_API_KEY", None)
    env.pop("OMICS_OS_ENDPOINT", None)
    return subprocess.run(  # nosec B603 # fixed argv, no shell
        [sys.executable, "-m", "lobster.cli", *args],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        input="",
        check=False,
    )


def test_python_does_not_take_the_npm_sign_in(real_sign_in, tmp_path):
    before = real_sign_in.read_bytes()
    res = _py_cli(["cloud", "account"], tmp_path)
    assert res.returncode == 0, res.stderr
    assert "Not connected" in res.stdout
    assert real_sign_in.read_bytes() == before


def test_npm_stays_signed_in_after_python_commands(real_sign_in, tmp_path):
    npm = shutil.which("omics-os")
    if not npm:
        pytest.skip("npm Cloud client (omics-os) not on PATH")
    before = real_sign_in.read_bytes()
    for args in (["cloud", "status"], ["cloud", "logout"]):
        res = _py_cli(args, tmp_path)
        assert res.returncode == 0, (args, res.stderr[-300:])
    assert real_sign_in.read_bytes() == before
    res = subprocess.run(  # nosec B603 # fixed argv, no shell
        [npm, "status"],
        capture_output=True,
        text=True,
        timeout=60,
        input="",
        check=False,
        env={**os.environ, "NO_COLOR": "1"},
    )
    assert res.returncode == 0, f"npm CLI signed out by Python: {res.stderr[-300:]}"
