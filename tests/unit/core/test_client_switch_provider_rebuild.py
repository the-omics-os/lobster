"""A provider switch rebuilds the graph like a model switch: the AQUADIF monitor stays."""

import pytest


@pytest.fixture
def builds(monkeypatch):
    """Every graph the client builds, by its options; no model is created."""
    from lobster.core import client as client_module

    seen = []

    def build(**options):
        seen.append(options)
        return object(), {}

    monkeypatch.setattr(client_module, "create_bioinformatics_graph", build)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-key")
    return seen


def test_provider_switch_keeps_the_aquadif_monitor(builds, tmp_path):
    from lobster.core.client import AgentClient

    client = AgentClient(workspace_path=tmp_path, provider_override="bedrock")
    result = client.switch_provider("anthropic")
    assert result["success"], result
    assert builds[-1]["provider_override"] == "anthropic"
    assert builds[-1]["aquadif_monitor"] is client.aquadif_monitor
    assert builds[-1]["checkpointer"] is client.checkpointer
    assert builds[-1]["model_override"] is None


def test_failed_provider_switch_keeps_the_running_graph(
    builds, monkeypatch, tmp_path
):
    from lobster.core import client as client_module
    from lobster.core.client import AgentClient

    client = AgentClient(workspace_path=tmp_path, provider_override="bedrock")
    first_graph = client.graph

    def broken(**options):
        raise RuntimeError("model could not be created")

    monkeypatch.setattr(client_module, "create_bioinformatics_graph", broken)
    result = client.switch_provider("anthropic")
    assert not result["success"]
    assert client.graph is first_graph
    assert client.provider_override == "bedrock"
