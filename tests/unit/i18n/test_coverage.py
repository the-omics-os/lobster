"""Translation coverage across the whole ``lobster`` package."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from literal_scan import iter_calls, scan  # noqa: E402

from lobster.i18n import current_lang, set_lang  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "lobster"
I18N = PACKAGE / "i18n"

# Files whose user-visible text is fully translated. A new raw English literal
# in one of these fails the test. Extend this list as areas are translated.
STRICT_FILES: list[str] = []


def _py_files():
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def test_every_h_literal_has_a_chinese_phrase():
    phrases = json.loads((I18N / "phrases.zh.json").read_text(encoding="utf-8"))
    missing = [
        f"{path.relative_to(ROOT)}:{line}: {text!r}"
        for path in _py_files()
        for line, _name, text in iter_calls(path, {"h"})
        if text not in phrases
    ]
    assert not missing, "\n".join(missing)


def test_every_t_key_exists_in_english():
    english = json.loads((I18N / "en.json").read_text(encoding="utf-8"))
    missing = [
        f"{path.relative_to(ROOT)}:{line}: {text!r}"
        for path in _py_files()
        for line, _name, text in iter_calls(path, {"t"})
        if "." in text and " " not in text and text not in english
    ]
    assert not missing, "\n".join(missing)


@pytest.mark.parametrize("relpath", STRICT_FILES)
def test_strict_files_have_no_raw_english(relpath):
    found = scan(ROOT / relpath)
    assert not found, "\n".join(f"{relpath}:{line}: {text!r}" for line, text in found)


def test_lang_command_switches_go_tui(tmp_path):
    from lobster.cli_internal.commands.heavy.slash_commands import _execute_command
    from lobster.cli_internal.commands.output_adapter import ProtocolOutputAdapter

    sent = []
    output = ProtocolOutputAdapter(lambda kind, payload: sent.append((kind, payload)))
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    client = SimpleNamespace(workspace_path=workspace)
    try:
        _execute_command("/lang zh", client, "/lang zh", output=output)
        kinds = [kind for kind, _ in sent]
        assert ("set_lang", {"lang": "zh"}) in sent
        # The language switch reaches Go before the confirmation text.
        assert kinds.index("set_lang") < len(kinds) - 1
        saved = json.loads((workspace / "provider_config.json").read_text())
        assert saved["lang"] == "zh"
    finally:
        set_lang("en")


_CJK = re.compile(r"[㐀-鿿＀-￯　-〿]")


@pytest.mark.parametrize(
    "command",
    ["workspace", "config provider list", "queue list publication", "tokens"],
)
def test_json_command_output_stays_english_in_chinese_ui(
    command, tmp_path, monkeypatch, capsys
):
    import typer

    from lobster.cli_internal.commands.heavy.slash_commands import command_cmd_impl

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOBSTER_WORKSPACE", raising=False)
    set_lang("zh")
    try:
        command_cmd_impl(command, workspace=str(tmp_path / "ws"), json_output=True)
    except typer.Exit:
        pass
    finally:
        assert current_lang() == "zh"  # the JSON path restores the UI language
        set_lang("en")
    stdout = capsys.readouterr().out
    payload = json.loads(stdout.strip().splitlines()[-1])
    assert "command" in payload
    decoded = json.dumps(payload, ensure_ascii=False)
    assert not _CJK.search(decoded), decoded
