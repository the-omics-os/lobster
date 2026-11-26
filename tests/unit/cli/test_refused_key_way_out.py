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
    "unauthorized (status code: 401)",  # Ollama Cloud
    "Error code: 401 - {'error': {'message': 'No auth credentials found', "
    "'code': 401}}",  # OpenRouter
]


@pytest.mark.parametrize("raw", REFUSED_KEY_ERRORS)
def test_a_refused_key_says_how_to_enter_a_new_one(raw):
    title, message = _friendly_error(raw)
    assert title == "Auth Error"
    assert "lobster init --force" in message


@pytest.mark.parametrize(
    "raw",
    [
        "The table has 4012 rows",
        "HTTP Error 401: Unauthorized",  # a tool's download, not the model's key
        'File "x.py", line 401, in run',
        "p-value 0.401 exceeded the cut-off",
    ],
)
def test_a_401_that_is_not_the_model_key_is_left_alone(raw):
    title, _ = _friendly_error(raw)
    assert title != "Auth Error"
