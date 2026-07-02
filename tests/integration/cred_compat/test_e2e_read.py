"""E2E read path — the real subprocess CLI reads each credential shape.

This is the axis-1/axis-2 regression at the subprocess level: `python -m
lobster.cli cloud account` under an isolated HOME must descend into the V2
active profile and report the right email/tier/endpoint (not "unknown", not the
DEFAULT host). Unit tests prove the functions; these prove the assembled CLI.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _account(isolated_home, shape):
    isolated_home.write(shape)
    return isolated_home.run_cli(["cloud", "account"])


def test_v2_single_profile_read(isolated_home, cred_factory):
    res = _account(isolated_home, cred_factory.v2_single())
    assert res.returncode == 0, res.stderr
    assert "app.omics-os.com" in res.stdout
    assert "oauth" in res.stdout


def test_v2_multi_profile_reads_active_tenant_host(isolated_home, cred_factory):
    res = _account(isolated_home, cred_factory.v2_multi())
    assert res.returncode == 0, res.stderr
    # active_profile is databiomix → tenant endpoint, NOT the default's platform host
    assert "databiomix.omics-os.com" in res.stdout


def test_v1_oauth_read(isolated_home, cred_factory):
    res = _account(isolated_home, cred_factory.v1_oauth())
    assert res.returncode == 0, res.stderr
    assert "oauth" in res.stdout
    assert "app.omics-os.com" in res.stdout


def test_v1_apikey_read(isolated_home, cred_factory):
    res = _account(isolated_home, cred_factory.v1_apikey())
    assert res.returncode == 0, res.stderr
    assert "api_key" in res.stdout


def test_malformed_v2_falls_back_not_crash(isolated_home, cred_factory):
    res = _account(isolated_home, cred_factory.malformed_v2())
    # resolver falls back to "default"; must not traceback
    assert res.returncode == 0, res.stderr
    assert "Traceback" not in res.stderr
    assert "app.omics-os.com" in res.stdout


def test_no_credentials_reports_not_connected(isolated_home):
    # no file written
    res = isolated_home.run_cli(["cloud", "account"])
    assert res.returncode == 0
    assert "Not connected" in res.stdout or "cloud login" in res.stdout
