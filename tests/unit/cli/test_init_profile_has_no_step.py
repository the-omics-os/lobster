"""A profile chooses no model, so no wizard asks for one; --profile still sets it."""

import pytest
from typer.testing import CliRunner

KEY = "sk-ant-synthetic-test-key"  # nosec B105 - synthetic value


@pytest.fixture
def proj(monkeypatch, tmp_path):
    from lobster.config import credentials, global_config

    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", tmp_path / "creds")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "creds" / "credentials.json"
    )
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)
    for name in (
        "OMICS_OS_API_KEY",
        "LOBSTER_LLM_PROVIDER",
        "LOBSTER_PROFILE",
        "ANTHROPIC_API_KEY",
        "LOBSTER_WORKSPACE",
    ):
        monkeypatch.delenv(name, raising=False)
    return proj


def test_the_manifest_asks_anthropic_and_bedrock_for_a_model():
    from lobster.ui.wizard.manifest import build_init_manifest

    providers = {p.name: p for p in build_init_manifest(detect_ollama=False).providers}
    for name in ("anthropic", "bedrock"):
        assert providers[name].model_selection == "explicit"
        assert any(m.is_default for m in providers[name].models)
    assert not any(
        p.profiles or p.model_selection == "profile" for p in providers.values()
    )


def test_the_profile_flag_still_sets_the_profile(proj):
    from lobster.cli import app
    from lobster.config.agent_defaults import get_current_profile

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--non-interactive",
            "--skip-extras",
            "--skip-ssl-test",
            "--anthropic-key",
            KEY,
            "--profile",
            "max",
        ],
    )
    assert result.exit_code == 0, result.output
    env = (proj / ".env").read_text()
    assert "LOBSTER_PROFILE=max" in env
    assert "it does not choose models" in env
    assert get_current_profile(proj / ".lobster_workspace") == "max"


def test_the_profile_flag_reaches_the_wizard_answers(monkeypatch, tmp_path):
    from lobster.config import credentials, global_config
    from lobster.config.agent_defaults import get_current_profile
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from tests.unit.cli.test_init_project import _init_impl, _wizard

    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", tmp_path / "creds")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "creds" / "credentials.json"
    )
    for name in ("LOBSTER_PROFILE", "LOBSTER_LLM_PROVIDER", "LOBSTER_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)
    study = tmp_path / "study"
    _wizard(
        monkeypatch,
        {"provider": "anthropic", "api_key": KEY, "agents": [], "cancelled": False},
    )
    _init_impl(study, profile="ultra")

    workspace = study / ".lobster_workspace"
    assert WorkspaceProviderConfig.load(workspace).profile == "performance"
    assert get_current_profile(workspace) == "performance"
