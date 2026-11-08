"""Language choice at ``lobster init`` and where it is stored and read back."""

import io
import json
from pathlib import Path

import pytest

import lobster.i18n as i18n
from lobster.i18n import (
    current_lang,
    lang_source,
    language_was_explicit,
    resolve_lang,
    set_lang,
    use_lang,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def global_dir(tmp_path):
    root = tmp_path / "xdg"
    return root, {"XDG_CONFIG_HOME": str(root), "APPDATA": str(root)}


def test_global_lang_used_when_workspace_has_none(tmp_path, global_dir):
    root, env = global_dir
    _write_json(root / "lobster" / "providers.json", {"lang": "zh"})
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert resolve_lang(["lobster"], env, workspace) == "zh"
    assert lang_source() == "global"
    assert not language_was_explicit()


def test_workspace_lang_wins_over_global(tmp_path, global_dir):
    root, env = global_dir
    _write_json(root / "lobster" / "providers.json", {"lang": "zh"})
    workspace = tmp_path / "ws"
    _write_json(workspace / "provider_config.json", {"lang": "en"})
    assert resolve_lang(["lobster"], env, workspace) == "en"
    assert lang_source() == "workspace"


def test_flag_and_env_are_explicit(tmp_path, global_dir):
    _, env = global_dir
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert resolve_lang(["lobster", "--lang", "zh", "init"], env, workspace) == "zh"
    assert lang_source() == "flag" and language_was_explicit()
    assert resolve_lang(["lobster"], {**env, "LOBSTER_LANG": "zh"}, workspace) == "zh"
    assert lang_source() == "env" and language_was_explicit()
    assert resolve_lang(["lobster"], env, workspace) == "en"
    assert lang_source() == "default" and not language_was_explicit()


def test_global_config_dir_matches_provider_config():
    from lobster.config.global_config import CONFIG_DIR

    assert i18n._global_config_dir() == CONFIG_DIR


def test_set_lang_is_explicit_and_use_lang_restores_source(tmp_path, global_dir):
    _, env = global_dir
    resolve_lang(["lobster"], env, tmp_path)
    with use_lang("zh"):
        assert current_lang() == "zh"
        assert lang_source() == "default"
    assert (current_lang(), lang_source()) == ("en", "default")
    set_lang("zh")
    assert language_was_explicit()


def test_init_lang_explicit_choice_overwrites_stored(tmp_path, global_dir):
    from lobster.cli_internal.commands.heavy.init_commands import _init_lang

    _, env = global_dir
    resolve_lang(["lobster"], env, tmp_path)
    assert _init_lang("zh") == "zh"  # not explicit: keep what is stored
    assert _init_lang(None) == "en"
    set_lang("zh")
    assert _init_lang("en") == "zh"  # explicit: the new choice wins


def test_reinit_with_new_language_updates_workspace(tmp_path, global_dir):
    from lobster.cli_internal.commands.heavy.init_commands import (
        _create_workspace_config,
    )

    _, env = global_dir
    workspace = tmp_path / ".lobster_workspace"
    _write_json(workspace / "provider_config.json", {"lang": "en"})
    resolve_lang(["lobster", "--lang", "zh"], env, workspace)
    _create_workspace_config({"provider": "ollama"}, workspace)
    saved = json.loads((workspace / "provider_config.json").read_text())
    assert saved["lang"] == "zh"


def test_global_init_saves_language(tmp_path, monkeypatch):
    import lobster.config.global_config as global_config
    from lobster.cli_internal.commands.heavy.init_commands import (
        _create_global_config,
    )

    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path)
    set_lang("zh")
    path = _create_global_config({"provider": "ollama"})
    assert json.loads(Path(path).read_text())["lang"] == "zh"


@pytest.mark.parametrize("explicit", [False, True])
def test_go_wizard_asked_for_language_only_without_explicit_choice(
    tmp_path, monkeypatch, global_dir, explicit
):
    import lobster.ui.bridge.go_tui_bridge as bridge

    _, env = global_dir
    argv = ["lobster", "--lang", "zh"] if explicit else ["lobster"]
    resolve_lang(argv, env, tmp_path)
    monkeypatch.setenv("LOBSTER_INIT_ASK_LANG", "1")  # stale value must not leak
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs["env"])
        result_file = cmd[cmd.index("--result-file") + 1]
        Path(result_file).write_text('{"provider": "ollama"}', encoding="utf-8")

        class Done:
            returncode = 0

        return Done()

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    assert bridge.run_init_wizard("lobster-tui") == {"provider": "ollama"}
    assert ("LOBSTER_INIT_ASK_LANG" in seen) is (not explicit)
    assert seen["LOBSTER_LANG"] == current_lang()


def test_ask_language_skipped_when_explicit(monkeypatch):
    from lobster.cli_internal.commands.heavy import init_commands

    set_lang("en")
    monkeypatch.setattr(
        init_commands.Prompt, "ask", lambda *a, **k: pytest.fail("asked again")
    )
    assert init_commands._ask_language(use_questionary=False) is None


def test_ask_language_classic_prompt_switches_language(
    tmp_path, monkeypatch, global_dir
):
    from lobster.cli_internal.commands.heavy import init_commands

    _, env = global_dir
    resolve_lang(["lobster"], env, tmp_path)
    monkeypatch.setattr(init_commands.Prompt, "ask", lambda *a, **k: "2")
    from rich.console import Console

    monkeypatch.setattr(init_commands, "console", Console(file=io.StringIO()))
    assert init_commands._ask_language(use_questionary=False) == "zh"
    assert current_lang() == "zh" and language_was_explicit()
