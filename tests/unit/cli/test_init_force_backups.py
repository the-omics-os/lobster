"""A cancelled ``init --force`` leaves no backup of files it never replaced."""

import pytest

from tests.unit.cli.test_init_project import (  # noqa: F401  (fixture)
    KEY,
    _init_impl,
    _wizard,
    dirs,
)

pytestmark = pytest.mark.unit

ENV = b"ANTHROPIC_API_KEY=sk-ant-old-synthetic\n"
PROVIDER = b'{"global_provider": "anthropic"}\n'


@pytest.fixture
def study(dirs):  # noqa: F811
    folder = dirs / "study"
    (folder / ".lobster_workspace").mkdir(parents=True)
    (folder / ".env").write_bytes(ENV)
    (folder / ".lobster_workspace" / "provider_config.json").write_bytes(PROVIDER)
    return folder


def _backups(folder):
    return sorted(
        p.name
        for p in [*folder.iterdir(), *(folder / ".lobster_workspace").iterdir()]
        if ".backup." in p.name
    )


def test_a_cancelled_go_wizard_leaves_no_backup(study, monkeypatch):
    _wizard(monkeypatch, {"cancelled": True})

    assert _init_impl(study) == 0

    assert _backups(study) == []
    assert (study / ".env").read_bytes() == ENV


def test_a_cancelled_questionary_wizard_leaves_no_backup(study, monkeypatch):
    from lobster.cli_internal.commands.heavy import init_commands

    _wizard(monkeypatch, {"cancelled": True})
    monkeypatch.setattr("lobster.ui.bridge.binary_finder.find_tui_binary", lambda: None)
    monkeypatch.setattr(init_commands, "_ask_language", lambda **k: None)
    asked = []
    monkeypatch.setattr(
        "lobster.ui.bridge.questionary_fallback.run_questionary_init",
        lambda *a, **k: asked.append(1) or {"cancelled": True},
    )

    assert _init_impl(study, ui_mode="auto") == 0
    assert asked == [1]

    assert _backups(study) == []


def test_a_run_that_fails_before_writing_leaves_no_backup(study, monkeypatch):
    import typer

    from lobster.cli_internal.commands.heavy import init_commands

    def interrupted(*a, **k):
        raise typer.Exit(1)

    _wizard(monkeypatch, {"provider": "anthropic", "api_key": KEY})
    monkeypatch.setattr(init_commands, "_postprocess_tui_init_result", interrupted)

    assert _init_impl(study) == 1

    assert _backups(study) == []
    assert (study / ".env").read_bytes() == ENV


def test_a_run_that_replaced_a_file_keeps_its_backup(study, monkeypatch):
    import typer

    from lobster.cli_internal.commands.heavy import init_commands

    _wizard(
        monkeypatch,
        {
            "provider": "anthropic",
            "api_key": KEY,
            "profile": "production",
            "model_id": "claude-test-model",
        },
    )

    def stop(*a, **k):
        raise typer.Exit(1)

    monkeypatch.setattr(init_commands, "_print_init_summary", stop)

    assert _init_impl(study) == 1

    assert (study / ".env").read_bytes() != ENV
    [env_backup] = [n for n in _backups(study) if n.startswith(".env.")]
    assert (study / env_backup).read_bytes() == ENV


def test_a_run_that_ends_normally_keeps_every_backup(dirs, tmp_path):  # noqa: F811
    import typer

    from lobster.cli_internal.commands.heavy import init_commands

    source, backup = tmp_path / "a", tmp_path / "a.backup"
    source.write_bytes(ENV)
    backup.write_bytes(ENV)

    def made(end):
        @init_commands._creates_project
        def run(**_):
            init_commands._init_backups.append((source, backup))
            end()

        return run

    def handled():
        raise init_commands._Handled(0)

    made(lambda: None)(global_config=True)
    assert backup.exists()
    with pytest.raises(typer.Exit):
        made(handled)(global_config=True)
    assert backup.exists()
