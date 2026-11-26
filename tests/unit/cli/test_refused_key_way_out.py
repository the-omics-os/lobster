"""A key the provider refuses says how to enter a new one, in the Go chat."""

import pytest

from lobster.cli_internal.go_tui_launcher import _friendly_error

pytestmark = pytest.mark.no_auto_config

# Error texts as the SDKs raise them for a placeholder or wrong key.
REFUSED_KEY_ERRORS = [
    # Nebius, seen live with a placeholder key (2026-11-26).
    "Error code: 401 - Couldn't authenticate. Reason: Unable authenticate",
    "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', "
    "'message': 'invalid x-api-key'}}",
    "Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-xx.', "
    "'type': 'invalid_request_error', 'code': 'invalid_api_key'}}",
]


@pytest.mark.parametrize("raw", REFUSED_KEY_ERRORS)
def test_a_refused_key_says_how_to_enter_a_new_one(raw):
    title, message = _friendly_error(raw)
    assert title == "Auth Error"
    assert "lobster init --force" in message


def test_a_number_that_merely_contains_401_is_not_a_refused_key():
    title, _ = _friendly_error("The table has 4012 rows")
    assert title != "Auth Error"
