"""Golden fixture — the exact byte-shape the npm CLI writes.

``golden/credentials_v2.json`` mirrors what the real npm CLI writes (token
VALUES scrubbed to PLACEHOLDERs; structure/keys/indentation kept). It is the
realistic omics-os CLI sign-in that Lobster must ignore:

  1. **Realistic** — the golden keeps the fields the npm CLI writes.
  2. **Ignored** — with the golden as the sign-in, Lobster has no key, sends to
     the platform host only, and leaves the file byte-identical.

Regenerate against a THROWAWAY tenant when npm's format changes:
  * ``omics-os login`` (npm) → copy ``~/.config/omics-os/credentials.json``
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

# Fields the npm CLI writes per profile; they keep the sentinel realistic.
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


def test_python_ignores_the_golden_sign_in(tmp_path, monkeypatch):
    """Lobster reads no key or endpoint from the npm CLI's sign-in."""
    from lobster.config import credentials

    target = tmp_path / ".config" / "omics-os" / "credentials.json"
    target.parent.mkdir(parents=True)
    target.write_text(GOLDEN.read_text(encoding="utf-8"), encoding="utf-8")
    before = target.read_bytes()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    monkeypatch.delenv("OMICS_OS_ENDPOINT", raising=False)

    assert credentials.get_api_key() is None
    assert credentials.get_endpoint() == "https://app.omics-os.com"
    assert target.read_bytes() == before
