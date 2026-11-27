"""/config model <name> is for this session unless --save; a save is read at once;
the chat footer names the model in use after a provider switch."""

import pytest

from lobster.cli_internal.commands.light.config_commands import config_model_switch

pytestmark = pytest.mark.no_auto_config  # these tests read the real resolver


class _Output:
    def __init__(self):
        self.messages = []

    def print(self, message, style=None):
        self.messages.append(str(message))

    def text(self):
        return "\n".join(self.messages)


@pytest.fixture
def builds(monkeypatch):
    """Every graph the client builds, by its options; no model is created."""
    from lobster.core import client as client_module
    from lobster.core.config_resolver import ConfigResolver

    seen = []

    def build(**options):
        seen.append(options)
        return object(), {}

    monkeypatch.setattr(client_module, "create_bioinformatics_graph", build)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-key")
    ConfigResolver.reset_instance()
    yield seen
    ConfigResolver.reset_instance()


def _client(tmp_path, **options):
    from lobster.core.client import AgentClient

    return AgentClient(
        workspace_path=tmp_path, provider_override="anthropic", **options
    )


def test_without_save_the_model_is_for_this_session_only(builds, tmp_path):
    from lobster.config.workspace_config import WorkspaceProviderConfig

    client = _client(tmp_path)
    output = _Output()
    assert config_model_switch(client, output, "claude-opus-5-5")
    assert builds[-1]["model_override"] == "claude-opus-5-5"
    saved = WorkspaceProviderConfig.load(tmp_path)
    assert saved.get_model_for_provider("anthropic") != "claude-opus-5-5"
    assert "Using claude-opus-5-5 for this session. Add --save to keep it." in (
        output.text()
    )
    assert "Saved" not in output.text()


def test_save_keeps_it_and_the_running_chat_reads_it_at_once(builds, tmp_path):
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.core.config_resolver import ConfigResolver

    client = _client(tmp_path)
    ConfigResolver.get_instance(tmp_path)  # what the chat read when it started
    output = _Output()
    assert config_model_switch(client, output, "claude-opus-5-5", save=True)
    saved = WorkspaceProviderConfig.load(tmp_path)
    assert saved.get_model_for_provider("anthropic") == "claude-opus-5-5"
    assert "Saved claude-opus-5-5 as your default." in output.text()
    model, _ = ConfigResolver.get_instance(tmp_path).resolve_model(
        agent_name="supervisor", provider="anthropic"
    )
    assert model == "claude-opus-5-5"


def test_a_saved_workspace_config_is_read_again(builds, tmp_path):
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.core.config_resolver import ConfigResolver

    ConfigResolver.get_instance(tmp_path)  # read before the save
    config = WorkspaceProviderConfig.load(tmp_path)
    config.set_model_for_provider("anthropic", "claude-opus-5-5")
    config.save(tmp_path)
    model, _ = ConfigResolver.get_instance(tmp_path).resolve_model(
        agent_name="supervisor", provider="anthropic"
    )
    assert model == "claude-opus-5-5"


@pytest.mark.parametrize("command", ["/config model switch", "/config model --save"])
def test_a_switch_without_a_name_asks_for_one(builds, tmp_path, command):
    from lobster.cli_internal.commands.heavy import slash_commands

    client = _client(tmp_path)
    output = _Output()
    slash_commands._execute_command(command, client, output=output)
    assert "Name a model." in output.text()
    assert "None" not in output.text() and "'--save'" not in output.text()
    assert len(builds) == 1  # nothing was rebuilt


def test_the_footer_names_the_model_after_a_provider_switch(builds, tmp_path):
    from lobster.cli_internal.go_tui_launcher import _emit_provider_status
    from lobster.core.config_resolver import ConfigResolver

    client = _client(tmp_path, model_override="claude-opus-5-5")
    assert client.switch_provider("openai")["success"]
    sent = []

    class Bridge:
        def send(self, kind, payload):
            sent.append(payload.get("text", ""))

    _emit_provider_status(Bridge(), client)
    expected, _ = ConfigResolver.get_instance(tmp_path).resolve_model(
        agent_name="supervisor", provider="openai"
    )
    assert "Provider: openai" in sent
    assert f"Model: {expected}" in sent
