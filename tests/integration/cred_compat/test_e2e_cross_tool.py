"""E2E cross-tool contract — the real npm CLI ↔ Python round-trip.

GATED (``@pytest.mark.real_api`` → skipped unless ``--runreal``). Requires the
real npm ``@omicsos/lobster`` on PATH and a logged-in throwaway test tenant. This
is the whole point of CRED_COMPAT_V2: the two tools share one credential file and
must both read what the other writes.

Run pre-release, by a human, against a THROWAWAY tenant:
    uv run pytest tests/integration/cred_compat/test_e2e_cross_tool.py --runreal -v

These tests do NOT run in CI and never write real tokens into fixtures.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_api]

REPO_ROOT = Path(__file__).resolve().parents[3]


def _npm_lobster() -> str | None:
    return shutil.which("lobster")


@pytest.fixture
def real_home():
    """Use the developer's REAL credentials file (throwaway tenant).

    Cross-tool tests need the file the npm CLI actually wrote — we do not
    fabricate it. We snapshot it, run the round-trip, and restore it after so the
    test never mutates the real login permanently.
    """
    cred = Path.home() / ".config" / "omics-os" / "credentials.json"
    if not cred.exists():
        pytest.skip(
            "No real credentials file — log in with the npm CLI first (throwaway tenant)."
        )
    backup = cred.read_bytes()
    try:
        yield cred
    finally:
        cred.write_bytes(backup)  # restore original bytes


def _py_cli(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "lobster.cli", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        input="",
        check=False,
    )


def test_python_reads_npm_written_file(real_home):
    """The Python CLI authenticates against the file the npm CLI wrote."""
    res = _py_cli(["cloud", "account"])
    assert res.returncode == 0, res.stderr
    assert (
        "Not connected" not in res.stdout
    ), "Python sees the npm-written file as logged out"


def test_npm_reads_file_after_python_refresh(real_home):
    """After a Python refresh rotates tokens, the npm CLI still authenticates."""
    npm = _npm_lobster()
    if not npm:
        pytest.skip("npm @omicsos/lobster not on PATH")

    # Force a Python-side refresh (reads → rotates → writes the shared file).
    from lobster.config import credentials

    refreshed = credentials.refresh_token()
    if not refreshed:
        pytest.skip(
            "Refresh did not return a token (token may be non-refreshable in this tenant)."
        )

    # File must still be valid V2 JSON after the Python write.
    data = json.loads(real_home.read_text(encoding="utf-8"))
    assert data.get("version") == 2

    # The real npm CLI must still read it and report authenticated.
    res = subprocess.run(
        [npm, "cloud", "status"],
        capture_output=True,
        text=True,
        timeout=60,
        input="",
        check=False,
        env={**os.environ, "NO_COLOR": "1"},
    )
    assert (
        res.returncode == 0
    ), f"npm CLI failed to read Python-refreshed file: {res.stderr[-300:]}"
