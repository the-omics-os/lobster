import json
import subprocess
from pathlib import Path

import pytest
import typer

from lobster.cli_internal.commands.heavy.init_commands import init_impl
from lobster.ui.bridge.go_tui_bridge import BridgeError, run_init_wizard


def test_init_force_go_skips_reconfirm_and_uses_go_wizard(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=test\n", encoding="utf-8")

    seen = {}

    def _unexpected_confirm(*args, **kwargs):
        raise AssertionError("Confirm.ask should not be called for --force")

    def _fake_run_init_wizard(binary_path, *, theme="lobster-dark", timeout=300):
        seen["binary_path"] = binary_path
        seen["theme"] = theme
        seen["timeout"] = timeout
        return {
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "api_key_secondary": "",
            "profile": "production",
            "agents": [],
            "ncbi_key": "",
            "cloud_key": "",
            "ollama_model": "",
            "cancelled": False,
        }

    def _fake_apply(result, workspace_path, env_path, global_config=False):
        seen["result"] = result
        seen["workspace_path"] = workspace_path
        seen["env_path"] = env_path
        seen["global_config"] = global_config

    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.init_commands.Confirm.ask",
        _unexpected_confirm,
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.binary_finder.find_tui_binary",
        lambda: "/tmp/fake-lobster-tui",
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.go_tui_bridge.run_init_wizard",
        _fake_run_init_wizard,
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.init_adapter.apply_tui_init_result",
        _fake_apply,
    )

    with pytest.raises(typer.Exit) as exc_info:
        init_impl(
            global_config=False,
            force=True,
            non_interactive=False,
            anthropic_key=None,
            bedrock_access_key=None,
            bedrock_secret_key=None,
            use_ollama=False,
            ollama_model=None,
            gemini_key=None,
            openai_key=None,
            profile=None,
            ncbi_key=None,
            cloud_key=None,
            cloud_endpoint=None,
            skip_ssl_test=False,
            ssl_verify=None,
            ssl_cert_path=None,
            agents=None,
            preset=None,
            auto_agents=False,
            agents_description=None,
            skip_docling=False,
            install_docling=False,
            install_vector_search=False,
            skip_extras=True,
            ui_mode="go",
        )

    assert exc_info.value.exit_code == 0
    assert seen["binary_path"] == "/tmp/fake-lobster-tui"
    assert seen["workspace_path"] == tmp_path / ".lobster_workspace"
    assert seen["env_path"] == tmp_path / ".env"
    assert seen["result"]["provider"] == "anthropic"
    assert len(list(tmp_path.glob(".env.backup.*"))) == 1


def test_init_go_cancel_exits_without_classic_fallback(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def _fake_run_init_wizard(binary_path, *, theme="lobster-dark", timeout=300):
        return {"cancelled": True}

    def _unexpected_classic_fallback(*args, **kwargs):
        raise AssertionError("classic init should not run after Go cancel")

    monkeypatch.setattr(
        "lobster.ui.bridge.binary_finder.find_tui_binary",
        lambda: "/tmp/fake-lobster-tui",
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.go_tui_bridge.run_init_wizard",
        _fake_run_init_wizard,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.init_commands._perform_agent_selection_interactive",
        _unexpected_classic_fallback,
    )

    with pytest.raises(typer.Exit) as exc_info:
        init_impl(
            global_config=False,
            force=False,
            non_interactive=False,
            anthropic_key=None,
            bedrock_access_key=None,
            bedrock_secret_key=None,
            use_ollama=False,
            ollama_model=None,
            gemini_key=None,
            openai_key=None,
            profile=None,
            ncbi_key=None,
            cloud_key=None,
            cloud_endpoint=None,
            skip_ssl_test=False,
            ssl_verify=None,
            ssl_cert_path=None,
            agents=None,
            preset=None,
            auto_agents=False,
            agents_description=None,
            skip_docling=False,
            install_docling=False,
            install_vector_search=False,
            skip_extras=True,
            ui_mode="go",
        )

    assert exc_info.value.exit_code == 0


def test_run_init_wizard_reads_result_file_without_capture(monkeypatch, tmp_path):
    seen = {}

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        result_path = Path(cmd[cmd.index("--result-file") + 1])
        result_path.write_text(
            json.dumps({"provider": "anthropic", "cancelled": False}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("lobster.ui.bridge.go_tui_bridge.subprocess.run", _fake_run)

    payload = run_init_wizard("/tmp/fake-lobster-tui")

    assert payload == {"provider": "anthropic", "cancelled": False}
    assert "--result-file" in seen["cmd"]
    assert seen["kwargs"]["text"] is True
    assert "capture_output" not in seen["kwargs"]


def test_run_init_wizard_returns_cancelled_payload_from_result_file(monkeypatch):
    def _fake_run(cmd, **kwargs):
        result_path = Path(cmd[cmd.index("--result-file") + 1])
        result_path.write_text(json.dumps({"cancelled": True}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr("lobster.ui.bridge.go_tui_bridge.subprocess.run", _fake_run)

    payload = run_init_wizard("/tmp/fake-lobster-tui")

    assert payload == {"cancelled": True}


def test_run_init_wizard_raises_when_result_missing(monkeypatch):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr("lobster.ui.bridge.go_tui_bridge.subprocess.run", _fake_run)

    with pytest.raises(BridgeError, match="exit code 1"):
        run_init_wizard("/tmp/fake-lobster-tui")
