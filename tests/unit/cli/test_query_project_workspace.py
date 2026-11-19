"""``lobster query -w <project folder>`` works in the project's own workspace."""

import os

import pytest
from rich.console import Console

from lobster.cli_internal.commands.heavy import query_commands, session_infra
from lobster.core import projects


# It patches ConfigResolver.get_instance itself; see test_startup_diagnostics.py.
@pytest.mark.no_auto_config
@pytest.mark.parametrize("session_id", [None, "latest"])
def test_query_with_a_project_folder_uses_the_project_workspace(
    monkeypatch, mocker, tmp_path, session_id
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    from lobster.config import global_config

    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "cfg" / "lobster")
    project = tmp_path / "study"
    projects.init_project(project)
    monkeypatch.chdir(tmp_path)

    seen = []

    class _FakeResolver:
        def resolve_provider(self, runtime_override=None):
            return "bedrock", "runtime flag --provider"

    class _FakeProvider:
        def check_dependencies(self):
            pass

    class _FakeClient:
        session_id = "session-test"
        token_tracker = None
        callbacks = []

        def query(self, question):
            return {"success": True, "response": "done", "text": "done"}

    monkeypatch.setattr(
        session_infra, "_maybe_seed_workspace_provider_config", seen.append
    )
    monkeypatch.setattr(
        "lobster.config.settings.settings.reload_credentials", lambda path: None
    )
    mocker.patch(
        "lobster.core.config_resolver.ConfigResolver.get_instance",
        lambda workspace_path: _FakeResolver(),
    )
    monkeypatch.setattr(
        "lobster.config.providers.registry.ProviderRegistry.get",
        lambda name: _FakeProvider(),
    )
    monkeypatch.setattr(
        query_commands, "init_client", lambda *args, **kwargs: _FakeClient()
    )
    monkeypatch.setattr(query_commands, "console", Console(record=True, width=120))
    monkeypatch.setattr(
        query_commands, "_maybe_print_timings", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(query_commands, "should_show_progress", lambda client: False)

    query_commands.query_impl(
        "hi",
        workspace=project,
        session_id=session_id,
        reasoning=False,
        verbose=False,
        debug=False,
        output=None,
        profile_timings=None,
        provider="bedrock",
        model=None,
        stream=False,
        json_output=False,
    )

    assert seen == [project / ".lobster_workspace"]
    assert os.listdir(project) == [".lobster_workspace"]
