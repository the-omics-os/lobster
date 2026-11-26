"""/config model in a running chat changes the model of the next turn."""

import pytest

from lobster.cli_internal.commands.light.config_commands import config_model_switch


class _Output:
    def __init__(self):
        self.messages = []

    def print(self, message, style=None):
        self.messages.append((style, message))


@pytest.fixture
def builds(monkeypatch):
    """Every graph the client builds, by its options; no model is created."""
    from lobster.core import client as client_module

    seen = []

    def build(**options):
        seen.append(options)
        return object(), {}

    monkeypatch.setattr(client_module, "create_bioinformatics_graph", build)
    return seen


def test_model_switch_rebuilds_the_graph_with_the_new_model(builds, tmp_path):
    from lobster.core.client import AgentClient

    client = AgentClient(
        workspace_path=tmp_path,
        provider_override="anthropic",
        model_override="claude-sonnet-4-20250514",
    )
    first_graph = client.graph
    assert config_model_switch(client, _Output(), "claude-sonnet-5-5")
    assert client.graph is not first_graph
    assert builds[-1]["model_override"] == "claude-sonnet-5-5"
    assert builds[-1]["provider_override"] == "anthropic"
    assert client.model_override == "claude-sonnet-5-5"


def test_failed_rebuild_keeps_the_running_model(builds, monkeypatch, tmp_path):
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.core import client as client_module
    from lobster.core.client import AgentClient

    client = AgentClient(
        workspace_path=tmp_path,
        provider_override="anthropic",
        model_override="claude-sonnet-4-20250514",
    )
    first_graph = client.graph

    def broken(**options):
        raise RuntimeError("model could not be created")

    monkeypatch.setattr(client_module, "create_bioinformatics_graph", broken)
    output = _Output()
    assert config_model_switch(client, output, "claude-sonnet-5-5") is None
    assert client.graph is first_graph
    assert client.model_override == "claude-sonnet-4-20250514"
    assert not any("Switched" in str(message) for _, message in output.messages)
    saved = WorkspaceProviderConfig.load(tmp_path)
    assert saved.get_model_for_provider("anthropic") != "claude-sonnet-5-5"
