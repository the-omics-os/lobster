import json
import os
import subprocess
from pathlib import Path

import pytest

from lobster.cli_internal.ink_launcher import (
    _PythonTerminalQuarantine,
    run_ink_init_wizard,
)


class _FakeManifest:
    def to_json(self) -> str:
        return "{}"


def test_python_terminal_quarantine_suppresses_stdout_and_stderr(capfd):
    quarantine = _PythonTerminalQuarantine.activate()
    try:
        os.write(1, b"hidden-stdout\n")
        os.write(2, b"hidden-stderr\n")
    finally:
        quarantine.restore()

    os.write(1, b"visible-stdout\n")
    os.write(2, b"visible-stderr\n")

    captured = capfd.readouterr()
    combined = captured.out + captured.err
    assert "hidden-stdout" not in combined
    assert "hidden-stderr" not in combined
    assert "visible-stdout" in captured.out
    assert "visible-stderr" in captured.err


def test_run_ink_init_wizard_reads_result_file_without_capture(monkeypatch):
    seen = {}

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        result_path = Path(cmd[cmd.index("--result-file") + 1])
        result_path.write_text(
            json.dumps(
                {
                    "provider": "anthropic",
                    "credentials": {"ANTHROPIC_API_KEY": "sk-ant-test"},
                    "selectedPackages": ["lobster-research"],
                    "optionalKeys": {"NCBI_API_KEY": "ncbi-test"},
                    "model": None,
                    "profile": "production",
                    "smartStandardization": True,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        "lobster.cli_internal.ink_launcher.find_ink_binary",
        lambda: "/tmp/fake-lobster-chat",
    )
    monkeypatch.setattr(
        "lobster.ui.wizard.manifest.build_init_manifest",
        lambda: _FakeManifest(),
    )
    monkeypatch.setattr("lobster.cli_internal.ink_launcher.subprocess.run", _fake_run)

    payload = run_ink_init_wizard()

    assert payload == {
        "provider": "anthropic",
        "api_key": "sk-ant-test",
        "api_key_secondary": "",
        "profile": "production",
        "agents": ["lobster-research"],
        "ncbi_key": "ncbi-test",
        "cloud_key": "",
        "ollama_model": "",
        "model_id": "",
        "smart_standardization_enabled": True,
        "smart_standardization_openai_key": "",
        "cancelled": False,
    }
    assert "--result-file" in seen["cmd"]
    assert seen["kwargs"]["text"] is True
    assert seen["kwargs"]["timeout"] == 300
    assert "capture_output" not in seen["kwargs"]


def test_run_ink_init_wizard_returns_cancelled_when_result_missing(monkeypatch):
    monkeypatch.setattr(
        "lobster.cli_internal.ink_launcher.find_ink_binary",
        lambda: "/tmp/fake-lobster-chat",
    )
    monkeypatch.setattr(
        "lobster.ui.wizard.manifest.build_init_manifest",
        lambda: _FakeManifest(),
    )
    monkeypatch.setattr(
        "lobster.cli_internal.ink_launcher.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0),
    )

    payload = run_ink_init_wizard()

    assert payload == {"cancelled": True}


def test_run_ink_init_wizard_raises_when_result_missing_and_exit_nonzero(monkeypatch):
    monkeypatch.setattr(
        "lobster.cli_internal.ink_launcher.find_ink_binary",
        lambda: "/tmp/fake-lobster-chat",
    )
    monkeypatch.setattr(
        "lobster.ui.wizard.manifest.build_init_manifest",
        lambda: _FakeManifest(),
    )
    monkeypatch.setattr(
        "lobster.cli_internal.ink_launcher.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1),
    )

    with pytest.raises(RuntimeError, match="exited with code 1"):
        run_ink_init_wizard()
