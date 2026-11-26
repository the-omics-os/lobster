"""Every `lobster …` command an error hint names exists, with its options."""

import re

import pytest
import typer

from lobster.cli import app
from lobster.cli_internal.go_tui_launcher import _friendly_error
from lobster.utils.error_handlers import ErrorHandlerRegistry

pytestmark = pytest.mark.no_auto_config

# One raw error for each branch of the chat hints and of the classic handlers.
RAW_ERRORS = [
    "The security token included in the request is invalid.",
    "An error occurred (AccessDeniedException) when calling the ConverseStream operation",
    "User is not authorized to use bedrock in this region",
    "Error code: 429 - rate limit reached",
    "Could not connect to the endpoint",
    "Request timed out",
    "insufficient_quota: You exceeded your current quota",
    "Error code: 401 - invalid api key",
    "The provided model identifier is invalid.",
    "prompt is too long: 250000 tokens > 200000 maximum",
    "connection error: no route to host",
    "something nobody planned for",
]

LOBSTER_CALL = re.compile(r"\blobster((?: [a-z][\w-]*| --[\w-]+)+)")


def _hint_texts():
    registry = ErrorHandlerRegistry()
    for raw in RAW_ERRORS:
        yield " ".join(_friendly_error(raw))
        guidance = registry.handle_error(Exception(raw))
        yield " ".join([guidance.description, *guidance.solutions])


def _check_call(root, call: str) -> None:
    tokens = call.split()
    assert tokens[0] in root.commands, f"lobster {call}: no such command"
    # Typer's groups are not click.Group instances; they do have `commands`.
    command, rest = root, tokens
    while rest and rest[0] in getattr(command, "commands", {}):
        command, rest = command.commands[rest[0]], rest[1:]
    options = {name for param in command.params for name in param.opts}
    for token in rest:
        if token.startswith("--"):
            assert token in options, f"lobster {call}: {token} is not an option"


def test_every_hint_names_a_real_command():
    root = typer.main.get_command(app)
    calls = [
        call.strip() for text in _hint_texts() for call in LOBSTER_CALL.findall(text)
    ]
    assert calls, "no hint names a lobster command"
    for call in calls:
        _check_call(root, call)
