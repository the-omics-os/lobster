"""A Ctrl-C after the setup screens says the setup may be half saved.

The screens' own cancel says "Setup cancelled." because nothing is written yet;
after them init may already have written files, so it says how to finish.
"""

import pytest

from tests.unit.cli.test_init_project import (  # noqa: F401  (fixture)
    KEY,
    _init_impl,
    _wizard,
    dirs,
)

pytestmark = pytest.mark.unit

INTERRUPTED = (
    "Interrupted while saving the setup; run lobster init --force to finish it."
)
OLD_ENV = b"ANTHROPIC_API_KEY=sk-ant-old-synthetic\n"


def _interrupted_after_the_screens(monkeypatch, ui_mode, rewrite=None):
    from rich.console import Console

    from lobster.cli_internal.commands.heavy import init_commands

    def interrupted(*a, **k):
        if rewrite is not None:
            rewrite.write_bytes(b"ANTHROPIC_API_KEY=sk-ant-new-synthetic\n")
        raise KeyboardInterrupt

    console = Console(width=200, record=True)
    monkeypatch.setattr(init_commands, "console", console)
    answers = {"provider": "anthropic", "api_key": KEY, "cancelled": False}
    _wizard(monkeypatch, answers)
    if ui_mode == "auto":  # no lobster-tui: the questionary screens
        monkeypatch.setattr(
            "lobster.ui.bridge.binary_finder.find_tui_binary", lambda: None
        )
        monkeypatch.setattr(init_commands, "_ask_language", lambda **k: None)
        monkeypatch.setattr(
            "lobster.ui.bridge.questionary_fallback.run_questionary_init",
            lambda *a, **k: dict(answers),
        )
    monkeypatch.setattr(init_commands, "_postprocess_tui_init_result", interrupted)
    return console


def _run(folder, ui_mode):
    try:
        return _init_impl(folder, ui_mode=ui_mode)
    except KeyboardInterrupt:  # would also stop pytest itself
        pytest.fail("the Ctrl-C escaped init")


@pytest.mark.parametrize("ui_mode", ["go", "auto"])
def test_a_ctrl_c_while_init_saves_the_setup_says_how_to_finish(
    dirs, monkeypatch, ui_mode  # noqa: F811
):
    console = _interrupted_after_the_screens(monkeypatch, ui_mode)

    assert _run(dirs / "study", ui_mode) == 1

    said = console.export_text()
    assert INTERRUPTED in said
    assert "cancelled" not in said.lower()


def test_a_file_rewritten_before_the_ctrl_c_keeps_its_backup(
    dirs, monkeypatch  # noqa: F811
):
    study = dirs / "study"
    (study / ".lobster_workspace").mkdir(parents=True)
    (study / ".env").write_bytes(OLD_ENV)
    _interrupted_after_the_screens(monkeypatch, "go", rewrite=study / ".env")

    assert _run(study, "go") == 1

    [backup] = study.glob(".env.backup.*")
    assert backup.read_bytes() == OLD_ENV
