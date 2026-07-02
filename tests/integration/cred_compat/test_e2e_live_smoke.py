"""E2E live smoke — real refresh + one real round-trip against app.omics-os.com.

GATED (``@pytest.mark.real_api`` → skipped unless ``--runreal``). Uses the
developer's real credentials for a THROWAWAY test tenant. Run pre-release:
    uv run pytest tests/integration/cred_compat/test_e2e_live_smoke.py --runreal -v
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_api]


@pytest.fixture
def real_creds_guard():
    cred = Path.home() / ".config" / "omics-os" / "credentials.json"
    if not cred.exists():
        pytest.skip("No real credentials file — log in first (throwaway tenant).")
    backup = cred.read_bytes()
    try:
        yield cred
    finally:
        cred.write_bytes(backup)


def test_live_refresh_returns_token_and_pins_platform(real_creds_guard, monkeypatch):
    """A real refresh against app.omics-os.com returns a token; verify the URL host."""
    from lobster.config import credentials

    captured = {}
    real_post = None

    import httpx

    orig_client = httpx.Client

    class _SpyClient(orig_client):
        def post(self, url, *a, **k):
            captured.setdefault("urls", []).append(url)
            return super().post(url, *a, **k)

    monkeypatch.setattr(httpx, "Client", _SpyClient)

    token = credentials.refresh_token()
    if token is None:
        pytest.skip(
            "Token not refreshable in this tenant (may be api_key or already valid)."
        )

    # Every refresh POST must have gone to the platform host — never a tenant.
    for url in captured.get("urls", []):
        host = urlparse(url).hostname or ""
        assert host == "app.omics-os.com", f"refresh hit non-platform host: {host}"


def test_live_get_api_key_returns_usable_token(real_creds_guard):
    from lobster.config import credentials

    token = credentials.get_api_key()
    assert token, "get_api_key returned nothing against the real logged-in file"
    # omc_ (first-party) or omk_ (api key) or a Cognito access token — never empty.
    assert isinstance(token, str) and len(token) > 10
