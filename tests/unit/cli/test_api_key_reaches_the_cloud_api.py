"""`lobster cloud query` and `lobster cloud projects` send an API key as a Bearer token.

app.omics-os.com reaches the API through CloudFront, which forwards only the
Authorization header on /api/*, so a key sent only in X-API-Key never arrives.
"""

import pytest

from lobster.cli_internal.commands.light import cloud_query

KEY = "omk_PLACEHOLDER0000000000000000000000000000000000000000"


@pytest.mark.parametrize("source", ["override", "stored"])
def test_an_api_key_goes_in_the_authorization_header_only(monkeypatch, source):
    monkeypatch.setattr(cloud_query, "get_api_key", lambda: KEY)
    headers = cloud_query.resolve_auth(KEY if source == "override" else None)
    assert headers == {"Authorization": f"Bearer {KEY}"}


def test_a_cli_token_still_goes_as_bearer(monkeypatch):
    monkeypatch.setattr(cloud_query, "get_api_key", lambda: "omc_PLACEHOLDER")
    assert cloud_query.resolve_auth() == {"Authorization": "Bearer omc_PLACEHOLDER"}
