"""`lobster config-test` tests the model the user's chat uses, not the provider default."""

import json

import pytest

from lobster.cli_internal.commands.light.config_commands import config_test_impl

pytestmark = pytest.mark.no_auto_config
USER_MODEL = "claude-sonnet-4-20250514"


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    """A folder whose workspace picks Anthropic and a model that is not the default."""
    from lobster.config import global_config
    from lobster.config.providers import get_provider
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.core.config_resolver import ConfigResolver

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global-config")
    for name in ("LOBSTER_LLM_PROVIDER", "NCBI_API_KEY", "NCBI_EMAIL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-key")
    ws = tmp_path / ".lobster_workspace"
    ws.mkdir()
    config = WorkspaceProviderConfig(global_provider="anthropic")
    config.set_model_for_provider("anthropic", USER_MODEL)
    config.save(ws)
    monkeypatch.setattr(ConfigResolver, "_instance", None)
    assert get_provider("anthropic").get_default_model() != USER_MODEL

    tested = []

    class _Reply:
        def invoke(self, prompt):
            return "ok"

    def create_chat_model(self, model_id, **kwargs):
        tested.append(model_id)
        return _Reply()

    monkeypatch.setattr(
        type(get_provider("anthropic")), "create_chat_model", create_chat_model
    )
    return tested


def test_config_test_tests_the_workspace_model(workspace, capsys):
    config_test_impl(output_json=True)
    result = json.loads(capsys.readouterr().out)
    assert workspace == [USER_MODEL]
    assert result["checks"]["llm_provider"]["status"] == "pass"
    assert result["checks"]["llm_provider"]["model"] == USER_MODEL
