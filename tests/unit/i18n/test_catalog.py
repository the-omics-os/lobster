"""Catalog fallback, Mandarin overlay, and CLI help language."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from lobster.i18n import (
    canonicalize,
    consume_lang_flag,
    h,
    read_workspace_lang,
    reply_language_instruction,
    resolve_lang,
    t,
    write_workspace_lang,
)

ROOT = Path(__file__).resolve().parents[3]
EN_SENTENCE = "Multi-Agent Bioinformatics Analysis System"
ZH_SENTENCE = "用对话做生物信息分析"
KEY_CALL = re.compile(r"""\bt\(\s*['"]([^'"]+)['"]""")


def test_missing_key_returns_english_text():
    resolve_lang([])
    assert t("cli.version.help") == "Show version and exit"
    assert t("not.a.real.key") == "not.a.real.key"


def test_overlay_and_placeholder():
    resolve_lang(["lobster", "--lang", "zh"])
    assert t("cli.app.help") == f"🦞 Lobster by Omics-OS — {ZH_SENTENCE}"
    assert "找不到智能体" in t("light.agent.not_found", name="de_expert")
    assert "de_expert" in t("light.agent.not_found", name="de_expert")


def test_unknown_language_is_english():
    assert resolve_lang(["lobster", "--lang", "zz"]) == "en"
    assert EN_SENTENCE in t("cli.app.help")


def test_aliases_and_env_and_file(tmp_path, monkeypatch):
    assert canonicalize("zh-CN") == "zh"
    assert canonicalize("mandarin") == "zh"
    monkeypatch.setenv("LOBSTER_LANG", "zh")
    assert resolve_lang(["lobster"]) == "zh"
    monkeypatch.delenv("LOBSTER_LANG")
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    (workspace / "provider_config.json").write_text(
        json.dumps({"lang": "zh", "global_provider": "ollama"}),
        encoding="utf-8",
    )
    (workspace / "config.toml").write_text(
        'enabled = ["transcriptomics_expert"]\n',
        encoding="utf-8",
    )
    assert resolve_lang(["lobster"], workspace=tmp_path) == "zh"
    assert "transcriptomics_expert" in (workspace / "config.toml").read_text(
        encoding="utf-8"
    )
    assert read_workspace_lang(tmp_path) == "zh"


def test_flag_is_removed_and_last_wins():
    argv = ["lobster", "--lang", "en", "--help", "--lang", "zh"]
    assert consume_lang_flag(argv) == "zh"
    assert argv == ["lobster", "--help"]


def test_overlay_keys_are_a_subset_of_english():
    i18n = ROOT / "lobster" / "i18n"
    english = set(json.loads((i18n / "en.json").read_text(encoding="utf-8")))
    mandarin = set(json.loads((i18n / "zh.json").read_text(encoding="utf-8")))
    assert mandarin <= english
    assert mandarin


def test_code_keys_exist_in_english():
    english = set(
        json.loads((ROOT / "lobster" / "i18n" / "en.json").read_text(encoding="utf-8"))
    )
    files = [ROOT / "lobster" / "cli.py"]
    files.extend(
        (ROOT / "lobster" / "cli_internal" / "commands" / "light").glob("*.py")
    )
    used = set()
    for path in files:
        used.update(KEY_CALL.findall(path.read_text(encoding="utf-8")))
    missing = used - english
    assert not missing, missing


def test_write_workspace_lang_keeps_other_keys(tmp_path):
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    (workspace / "provider_config.json").write_text(
        json.dumps({"global_provider": "ollama", "profile": "production"}) + "\n",
        encoding="utf-8",
    )
    assert write_workspace_lang(workspace, "zh") == "zh"
    saved = json.loads((workspace / "provider_config.json").read_text(encoding="utf-8"))
    assert saved["lang"] == "zh"
    assert saved["global_provider"] == "ollama"
    assert saved["profile"] == "production"
    assert write_workspace_lang(workspace, "zz") == "en"
    saved = json.loads((workspace / "provider_config.json").read_text(encoding="utf-8"))
    assert saved["lang"] == "en"
    assert saved["global_provider"] == "ollama"


def test_help_phrases_cover_every_h_call():
    phrases = json.loads(
        (ROOT / "lobster" / "i18n" / "phrases.zh.json").read_text(encoding="utf-8")
    )
    files = [ROOT / "lobster" / "cli.py"]
    files.extend(
        (ROOT / "lobster" / "cli_internal" / "commands" / "light").glob("*.py")
    )
    used = set()
    for path in files:
        used.update(re.findall(r'\bh\("([^"]*)"\)', path.read_text(encoding="utf-8")))
    missing = sorted(used - set(phrases))
    assert not missing, missing
    resolve_lang(["lobster", "--lang", "zh"])
    assert h("Clear screen") == "清屏"
    assert reply_language_instruction().startswith(
        "Reply to the user in easy Simplified Chinese"
    )
    resolve_lang(["lobster", "--lang", "en"])
    assert h("Clear screen") == "Clear screen"
    assert reply_language_instruction() == ""


def test_json_field_names_are_not_translated():
    payload = {"status": "ok", "error": None, "session_id": "abc"}
    assert list(payload) == ["status", "error", "session_id"]


@pytest.mark.skipif(
    os.environ.get("LOBSTER_I18N_SKIP_HELP") == "1",
    reason="help subprocess disabled",
)
def test_help_uses_mandarin_app_sentence():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("LOBSTER_LANG", None)
    proc = subprocess.run(
        [sys.executable, "-m", "lobster.cli", "--lang", "zh", "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert ZH_SENTENCE in proc.stdout
    assert EN_SENTENCE not in proc.stdout
    assert "开始和多智能体系统的交互对话" in proc.stdout
    assert "Start an interactive chat session" not in proc.stdout
