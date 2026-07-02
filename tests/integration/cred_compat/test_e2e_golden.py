"""Golden fixture — the exact byte-shape the npm CLI writes.

``golden/credentials_v2.json`` mirrors what the real npm ``@omicsos/lobster``
writes (token VALUES scrubbed to PLACEHOLDERs; structure/keys/indentation kept).
Two guards:

  1. **Contract fields present** — every field Python's read/refresh/sync path
     depends on exists in the golden. If npm ever drops one, this fails.
  2. **Python parses it** — loading the golden through Python's reader yields the
     active profile, aliases cli_token→access_token, and reports the tenant host.

Regenerate against a THROWAWAY tenant when npm's format changes:
  * ``lobster cloud login`` (npm) → copy ``~/.config/omics-os/credentials.json``
  * scrub every ``cli_token``/``refresh_token``/``credential_id``/``user_id``/
    ``org_id``/``email`` value to a ``*_PLACEHOLDER_*`` string
  * keep keys, nesting, and value FORMAT (e.g. ISO-Z timestamps) intact
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

GOLDEN = Path(__file__).parent / "golden" / "credentials_v2.json"

# Fields Python's read/refresh/sync path relies on, per profile.
_REQUIRED_PROFILE_FIELDS = {
    "credential_type",
    "cli_token",
    "refresh_token",
    "credential_id",
    "endpoint",
    "auth_mode",
    "token_expiry",
}


def test_golden_has_v2_envelope():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert "active_profile" in data
    assert isinstance(data["profiles"], dict) and data["profiles"]
    assert data["active_profile"] in data["profiles"]


def test_golden_profiles_have_contract_fields():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for name, prof in data["profiles"].items():
        missing = _REQUIRED_PROFILE_FIELDS - set(prof)
        assert (
            not missing
        ), f"golden profile {name!r} missing contract fields: {missing}"
        assert prof["credential_type"] == "omics_cli"


def test_golden_tokens_are_scrubbed():
    """Safety: the committed golden must never contain a real-looking secret."""
    raw = GOLDEN.read_text(encoding="utf-8")
    assert "PLACEHOLDER" in raw
    data = json.loads(raw)
    for prof in data["profiles"].values():
        assert "PLACEHOLDER" in prof["cli_token"]
        assert "PLACEHOLDER" in prof["refresh_token"]


def test_python_reads_golden_active_profile(tmp_path, monkeypatch):
    """Python's reader parses npm's byte-shape and resolves the active profile."""
    from lobster.config import credentials

    cred_dir = tmp_path / "omics-os"
    cred_dir.mkdir(parents=True)
    target = cred_dir / "credentials.json"
    target.write_text(GOLDEN.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", target)
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    monkeypatch.delenv("OMICS_OS_ENDPOINT", raising=False)

    view = credentials.load_active_profile()
    assert view is not None
    # active is databiomix → tenant endpoint + its cli_token aliased to access_token
    assert view["endpoint"] == "https://databiomix.omics-os.com"
    assert view["access_token"] == view["cli_token"]
    assert credentials.get_endpoint() == "https://databiomix.omics-os.com"
    # token not expired (far-future golden) → get_api_key returns it without refresh
    assert credentials.get_api_key() == view["cli_token"]
