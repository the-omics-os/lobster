"""A model the provider does not have gets a next step, in chat and from the shell."""

import pytest
from typer.testing import CliRunner

from lobster.cli_internal.go_tui_launcher import _friendly_error

pytestmark = pytest.mark.no_auto_config

# OpenAI's text without its error code (lobster query, 1.3.1, a tester's run).
OPENAI_WITHOUT_CODE = (
    "Error code: 404 - The model `gpt-6.3-sol` does not exist or you do not have "
    "access to it."
)

# Error texts as the provider SDKs raise them for a model name they do not know.
WRONG_MODEL_ERRORS = [
    "An error occurred (ValidationException) when calling the ConverseStream "
    "operation: The provided model identifier is invalid.",
    "Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', "
    "'message': 'model: claude-sonnet-9'}}",
    "Error code: 404 - {'error': {'message': 'The model `gpt-9` does not exist or "
    "you do not have access to it.', 'type': 'invalid_request_error', 'param': None, "
    "'code': 'model_not_found'}}",
    'model "llama9" not found, try pulling it first (status code: 404)',
    "404 models/gemini-9 is not found for API version v1beta, or is not supported "
    "for generateContent.",
    "An error occurred (ValidationException) when calling the ConverseStream "
    "operation: Omics-OS Cloud: Invalid model ID: claude-sonnet-9",
    "Error code: 400 - {'error': {'message': 'openai/gpt-9 is not a valid model ID', "
    "'code': 400}}",
    OPENAI_WITHOUT_CODE,
    # The Omics-OS Cloud gateway.
    "An error occurred (ValidationException) when calling the Converse operation: "
    'Omics-OS Cloud: Invalid request: {"detail":{"status":422,"code":"MODEL_UNKNOWN",'
    '"message":"Unknown model.","retryable":false}}',
]


@pytest.mark.parametrize("raw", WRONG_MODEL_ERRORS)
def test_a_wrong_model_names_the_commands_that_change_it(raw):
    title, message = _friendly_error(raw)
    assert title == "Model Not Available"
    assert "/config model" in message
    assert "IAM" not in message


@pytest.mark.parametrize("raw", WRONG_MODEL_ERRORS)
def test_classic_chat_gives_the_same_way_out(raw):
    from lobster.utils.error_handlers import ErrorHandlerRegistry

    guidance = ErrorHandlerRegistry().handle_error(Exception(raw))
    assert guidance.error_type == "model_not_found"
    assert any("/config model" in step for step in guidance.solutions)


def test_the_way_out_names_the_option_for_one_run():
    from lobster.utils.error_handlers import ErrorHandlerRegistry

    guidance = ErrorHandlerRegistry().handle_error(Exception(OPENAI_WITHOUT_CODE))
    assert any("lobster query --model <name>" in step for step in guidance.solutions)


def test_shell_config_says_the_slash_commands_are_for_chat(monkeypatch, tmp_path):
    from lobster.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("AWS_BEDROCK_ACCESS_KEY", "AKIATESTNOTAKEY")
    monkeypatch.setenv("AWS_BEDROCK_SECRET_ACCESS_KEY", "test-not-a-secret")
    result = CliRunner().invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert "inside lobster chat" in " ".join(result.output.split())
