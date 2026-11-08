"""Observable combined locale/provider and CLI contracts."""

import json
import os
import re
import subprocess  # nosec B404 # test runs the CLI as a child process
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from lobster.i18n import (
    consume_lang_flag,
    current_lang,
    resolve_lang,
    set_lang,
    write_workspace_lang,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def reset_locale():
    set_lang("en")
    yield
    set_lang("en")


def cli(tmp_path, *args):
    env = os.environ.copy()
    env.update(
        HOME=str(tmp_path),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        PYTHONPATH=str(ROOT),
        LOBSTER_WORKSPACE=str(tmp_path / "workspace"),
        LOBSTER_LANG="en",
    )
    return subprocess.run(  # nosec B603 # fixed argv, no shell
        [sys.executable, "-m", "lobster.cli", *args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )


def test_language_flag_respects_literal_argument_separator():
    argv = ["lobster", "--lang", "zh", "query", "--", "--lang", "literal"]
    assert consume_lang_flag(argv) == "zh"
    assert argv == ["lobster", "query", "--", "--lang", "literal"]


@pytest.mark.parametrize("language", ["../phrases.zh", "/fake/zh", "../../en"])
def test_unsafe_locale_names_fall_back_to_english(language):
    assert set_lang(language) == "en"


def test_workspace_flag_and_env_select_saved_language(tmp_path):
    workspace = tmp_path / "selected"
    workspace.mkdir()
    (workspace / "provider_config.json").write_text('{"lang":"zh"}')
    for argv in (
        ["lobster", "chat", "-w", str(workspace)],
        ["lobster", "chat", "--workspace=" + str(workspace)],
    ):
        assert resolve_lang(list(argv), environ={}) == "zh"
    assert (
        resolve_lang(["lobster"], environ={"LOBSTER_WORKSPACE": str(workspace)}) == "zh"
    )
    assert (
        resolve_lang(["lobster", "--lang", "en"], environ={"LOBSTER_LANG": "zh"})
        == "en"
    )


def test_malformed_config_is_not_replaced(tmp_path):
    path = tmp_path / "provider_config.json"
    path.write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        write_workspace_lang(tmp_path, "zh")
    assert path.read_text() == "{broken"
    assert current_lang() == "en"


def test_failed_atomic_replace_preserves_config(tmp_path, monkeypatch):
    path = tmp_path / "provider_config.json"
    path.write_text('{"global_provider":"ollama-cloud"}')
    original = path.read_bytes()

    def fail(*args):
        raise OSError("read-only")

    monkeypatch.setattr("lobster.i18n.os.replace", fail)
    with pytest.raises(OSError):
        write_workspace_lang(tmp_path, "zh")
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".provider-config-*"))


def test_real_wizard_adapter_preserves_saved_language_and_new_provider(tmp_path):
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.ui.bridge.init_adapter import apply_tui_init_result

    workspace = tmp_path / "workspace"
    write_workspace_lang(workspace, "zh")
    # A plain English run with no language choice must keep the saved language.
    resolve_lang(["lobster"], {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}, tmp_path)
    apply_tui_init_result(
        {"provider": "ollama-cloud", "api_key": "test-only-key", "model_id": "glm-5.2"},
        workspace,
        tmp_path / ".env",
    )
    saved = WorkspaceProviderConfig.load(workspace)
    assert saved.lang == "zh"
    assert saved.global_provider == "ollama-cloud"
    assert saved.get_model_for_provider("ollama-cloud") == "glm-5.2"
    write_workspace_lang(workspace, "en")
    saved = WorkspaceProviderConfig.load(workspace)
    assert saved.global_provider == "ollama-cloud"
    assert saved.ollama_cloud_model == "glm-5.2"
    assert saved.lang == "en"


def test_init_and_chat_use_explicit_built_binary(tmp_path, monkeypatch):
    from lobster.cli_internal.go_tui_launcher import find_tui_binary_fast
    from lobster.ui.bridge.binary_finder import find_tui_binary

    binary = tmp_path / "local-tui"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o700)
    monkeypatch.setenv("LOBSTER_TUI_BINARY", str(binary))
    assert find_tui_binary() == find_tui_binary_fast() == str(binary)


def test_init_bridge_passes_resolved_language(tmp_path, monkeypatch):
    from lobster.ui.bridge.go_tui_bridge import run_init_wizard

    monkeypatch.setenv("LOBSTER_LANG", "en")
    set_lang("zh")

    def run(command, **kwargs):
        assert kwargs["env"]["LOBSTER_LANG"] == "zh"
        Path(command[command.index("--result-file") + 1]).write_text(
            '{"cancelled":true}'
        )
        return Mock(returncode=0)

    monkeypatch.setattr("lobster.ui.bridge.go_tui_bridge.subprocess.run", run)
    assert run_init_wizard("unused")["cancelled"] is True


@pytest.mark.parametrize(
    "lang,expected", [("en", "Upgrade lobster-ai"), ("zh", "把 lobster-ai 更新")]
)
def test_real_upgrade_help_is_localized(tmp_path, lang, expected):
    proc = cli(tmp_path, "--lang", lang, "upgrade", "--help")
    assert proc.returncode == 0, proc.stderr
    # CI may force colour; Rich then styles "--dry-run" in separate spans.
    stdout = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout)
    assert expected in stdout
    assert "--dry-run" in stdout


def test_missing_language_value_is_usage_error(tmp_path):
    proc = cli(tmp_path, "--lang")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_real_json_command_keeps_schema_in_both_languages(tmp_path):
    results = []
    for lang in ("en", "zh"):
        proc = cli(tmp_path, "--lang", lang, "command", "help", "--json")
        assert proc.returncode == 0, proc.stderr
        results.append(json.loads(proc.stdout))
    assert set(results[0]) == set(results[1])
    assert results[0]["success"] is results[1]["success"] is True
    assert results[0]["command"] == results[1]["command"] == "help"


def test_init_cancelled_and_status_not_initialized_are_localized():
    from lobster.i18n import t

    assert t("cli.init.cancelled") == "Setup cancelled."
    assert t("cli.status.not_initialized") == "Lobster is not initialized yet"
    en_hint = t("cli.status.init_hint")
    assert en_hint.startswith("Run `lobster init` to configure your LLM provider")

    set_lang("zh")
    assert t("cli.init.cancelled") == "已取消设置。"
    assert t("cli.status.not_initialized") == "Lobster 还没有设置"
    zh_hint = t("cli.status.init_hint")
    assert zh_hint != en_hint
    for token in (
        "`lobster init`",
        "--non-interactive",
        "--use-omics-os",
        "--skip-extras",
        "--skip-ssl-test",
        "OMICS_OS_API_KEY",
    ):
        assert token in zh_hint
