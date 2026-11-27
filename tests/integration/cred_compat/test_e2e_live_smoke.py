"""E2E live smoke — one real round-trip with a Lobster key against app.omics-os.com.

GATED (``@pytest.mark.real_api`` → skipped unless ``--runreal``). Needs
OMICS_OS_API_KEY for a THROWAWAY test tenant in the environment. Python runs
with its own settings in a tmp XDG_CONFIG_HOME. Run pre-release:
    OMICS_OS_API_KEY=<throwaway key> uv run pytest \
        tests/integration/cred_compat/test_e2e_live_smoke.py --runreal -v
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_api]


@pytest.fixture
def live_key():
    key = (os.environ.get("OMICS_OS_API_KEY") or "").strip()
    if not key:
        pytest.skip("Set OMICS_OS_API_KEY to a throwaway tenant's key.")
    return key


def test_live_status_with_a_lobster_key(live_key, tmp_path):
    env = {**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "xdg"), "NO_COLOR": "1"}
    env.pop("OMICS_OS_ENDPOINT", None)
    res = subprocess.run(
        [sys.executable, "-m", "lobster.cli", "cloud", "status"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        input="",
        check=False,
    )
    assert res.returncode == 0, res.stdout[-300:] + res.stderr[-300:]
    assert live_key not in res.stdout + res.stderr


def test_live_get_api_key_returns_the_environment_key(live_key):
    from lobster.config import credentials

    assert credentials.get_api_key() == live_key
