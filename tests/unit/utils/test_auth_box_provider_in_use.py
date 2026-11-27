"""The "Authentication Failed" box names the provider the running client uses
(a --provider override or a provider switch), not only the configured one, and
never says "API API"."""

import io

import pytest
from rich.console import Console

from lobster.utils.callbacks import SimpleTerminalCallback
from lobster.utils.error_handlers import AuthenticationErrorHandler, get_error_registry

REFUSED = RuntimeError(
    "Omics-OS Cloud API key rejected or expired by https://app.omics-os.com (401)"
)


@pytest.fixture(autouse=True)
def configured_anthropic(monkeypatch):
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "anthropic")


def test_the_provider_in_use_wins_over_the_configured_one():
    guidance = get_error_registry().handle_error(REFUSED, provider="omics-os")
    assert guidance.title == "🔑 Authentication Failed"
    assert guidance.description == "The Omics-OS Cloud key was refused or is not set."
    assert "OMICS_OS_API_KEY" in guidance.solutions[0]


def test_without_a_provider_in_use_the_configured_one_is_named_once():
    guidance = AuthenticationErrorHandler().handle(REFUSED, str(REFUSED))
    assert guidance.description == (
        "The Anthropic Direct API key was refused or is not set."
    )


def test_a_provider_without_a_key_gets_the_plain_sentence():
    guidance = get_error_registry().handle_error(REFUSED, provider="ollama")
    assert guidance.description == "The API key was refused or is not set."


def test_the_provider_in_use_does_not_outlive_its_error():
    get_error_registry().handle_error(REFUSED, provider="omics-os")
    guidance = get_error_registry().handle_error(REFUSED)
    assert "Anthropic" in guidance.description


@pytest.fixture
def builds(monkeypatch):
    """The client builds no model."""
    from lobster.core import client as client_module

    monkeypatch.setattr(
        client_module, "create_bioinformatics_graph", lambda **_: (object(), {})
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-key")


def test_the_terminal_callback_names_the_clients_provider(
    builds, monkeypatch, tmp_path
):
    from lobster.core.client import AgentClient

    callback = SimpleTerminalCallback(Console(file=io.StringIO()), show_reasoning=False)
    shown = []
    monkeypatch.setattr(callback, "_display_error_guidance", shown.append)
    client = AgentClient(
        workspace_path=tmp_path,
        provider_override="omics-os",
        custom_callbacks=[callback],
    )
    callback.on_llm_error(REFUSED)
    assert "Omics-OS Cloud" in shown[-1].description
    assert client.switch_provider("anthropic")["success"]
    callback.on_llm_error(REFUSED)
    assert "Anthropic" in shown[-1].description
