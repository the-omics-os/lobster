"""Tests for init protocol commands (npm CLI ↔ Python handoff)."""

import json

from typer.testing import CliRunner

from lobster.cli import app

runner = CliRunner()


def test_init_capabilities_returns_json():
    result = runner.invoke(app, ["init-capabilities"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["init_protocol_version"] == 1
    assert "lobster_ai_version" in data
    assert 1 in data["accepted_result_schemas"]


def test_init_manifest_schema_v1(monkeypatch):
    result = runner.invoke(app, ["init-manifest", "--schema-version", "1", "--no-detect-ollama"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["schema_version"] == 1
    assert "providers" in data
    assert "agent_packages" in data
    assert "ollama_status" in data
    assert "existing_state" in data
    assert "environment" in data
    assert isinstance(data["existing_state"]["npm_cli_available"], bool)
    assert isinstance(data["environment"]["is_venv"], bool)
    assert data["environment"]["platform"] == "darwin"


def test_init_manifest_rejects_unknown_schema():
    result = runner.invoke(app, ["init-manifest", "--schema-version", "99"])
    assert result.exit_code == 1


def test_apply_init_result_cancelled(tmp_path):
    f = tmp_path / "result.json"
    f.write_text(json.dumps({"cancelled": True}))
    result = runner.invoke(app, ["apply-init-result", "--result-file", str(f), "--schema-version", "1"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["applied"] is False
    assert data["reason"] == "cancelled"


def test_apply_init_result_writes_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda *a, **kw: tmp_path,
    )
    f = tmp_path / "result.json"
    f.write_text(json.dumps({
        "provider": "anthropic",
        "api_key": "sk-ant-test123",
        "api_key_secondary": "",
        "profile": "production",
        "agents": ["research_agent"],
        "ncbi_key": "",
        "cloud_key": "",
        "ollama_model": "",
        "smart_standardization_enabled": False,
        "smart_standardization_openai_key": "",
        "cancelled": False,
    }))
    result = runner.invoke(app, ["apply-init-result", "--result-file", str(f), "--schema-version", "1"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["applied"] is True
    assert data["provider"] == "anthropic"
    assert data["profile"] == "production"
    assert data["agents_configured"] == 1


def test_apply_init_result_rejects_bad_json(tmp_path):
    f = tmp_path / "result.json"
    f.write_text("not json at all")
    result = runner.invoke(app, ["apply-init-result", "--result-file", str(f), "--schema-version", "1"])
    assert result.exit_code == 1


def test_commands_hidden_from_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "init-capabilities" not in result.output
    assert "init-manifest" not in result.output
    assert "apply-init-result" not in result.output
