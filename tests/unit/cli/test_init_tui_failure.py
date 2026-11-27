"""When saving the Go wizard's answers fails, setup says so honestly.

The wizard worked, so it is not "unavailable"; the answers are kept for a
retry and no second wizard starts. Placeholder keys; no network.
"""

import inspect
import logging
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from lobster.cli_internal.commands.heavy import init_commands

pytestmark = pytest.mark.no_auto_config

ANSWERS = {"provider": "openai", "api_key": "not-a-key", "agents": []}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    from lobster.config import global_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global-config")
    for name in ("OMICS_OS_API_KEY", "OMICS_OS_ENDPOINT", "LOBSTER_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)


@pytest.fixture
def go_wizard(monkeypatch):
    """A Go wizard that answers ANSWERS; saving fails as told; record calls."""
    calls = SimpleNamespace(saves=0, fallback=False)

    def save(result, **kwargs):
        calls.saves += 1
        if calls.saves <= calls.failures:
            raise PermissionError("[Errno 13] Permission denied: '.env'")
        return result

    def fallback(*args, **kwargs):
        calls.fallback = True
        raise AssertionError("a second wizard started")

    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        "lobster.ui.bridge.binary_finder.find_tui_binary", lambda: "/fake/lobster-tui"
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.go_tui_bridge.run_init_wizard",
        lambda binary, **kwargs: dict(ANSWERS),
    )
    monkeypatch.setattr(
        "lobster.ui.bridge.questionary_fallback.run_questionary_init", fallback
    )
    monkeypatch.setattr(init_commands, "_postprocess_tui_init_result", save)
    monkeypatch.setattr(init_commands.Prompt, "ask", fallback)  # classic wizard
    monkeypatch.setattr(init_commands, "_print_init_summary", lambda *a, **k: None)
    calls.failures = 1
    return calls


def run_init(monkeypatch, capsys, retry):
    monkeypatch.setattr(init_commands.Confirm, "ask", lambda *a, **k: retry)
    options = {
        name: (
            p.default.default
            if isinstance(p.default, typer.models.OptionInfo)
            else p.default
        )
        for name, p in inspect.signature(init_commands.init_impl).parameters.items()
    }
    options.update(ui_mode="auto", skip_extras=True, skip_ssl_test=True)
    code = 0
    try:
        init_commands.init_impl(**options)
    except typer.Exit as error:
        code = error.exit_code
    return code, capsys.readouterr().out


def test_a_failed_save_shows_the_real_error_and_retries_with_the_same_answers(
    go_wizard, monkeypatch, capsys
):
    code, output = run_init(monkeypatch, capsys, retry=True)
    assert code == 0, output
    assert "could not save your answers" in output
    assert "Permission denied" in output
    assert "unavailable" not in output
    assert go_wizard.saves == 2
    assert not go_wizard.fallback


def test_declining_the_retry_stops_without_a_second_wizard(
    go_wizard, monkeypatch, capsys
):
    code, output = run_init(monkeypatch, capsys, retry=False)
    assert code == 1
    assert "lobster init" in output
    assert "unavailable" not in output
    assert go_wizard.saves == 1
    assert not go_wizard.fallback


def test_a_global_config_save_error_is_not_logged_as_a_developer_error(
    monkeypatch, tmp_path, caplog
):
    """The file itself refuses the write, as on the b1 walk (chmod a-w)."""
    from lobster.config import global_config

    real_write_text = global_config.Path.write_text

    def refuse(self, *args, **kwargs):
        if self.name == "providers.json":
            raise PermissionError(13, "Permission denied", str(self))
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(global_config.Path, "write_text", refuse)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(PermissionError):
            init_commands._create_global_config({"provider": "openai"})
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_global_cloud_setup_refuses_a_model_before_changing_anything(tmp_path):
    from lobster.cli import app
    from lobster.config.global_config import GlobalProviderConfig

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--non-interactive",
            "--skip-extras",
            "--skip-ssl-test",
            "--global",
            "--omics-os-key",
            "omk_PLACEHOLDER",
            "--model",
            "some-model",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "omics-os has no model setting; drop --model" in result.output
    assert "Traceback" not in result.output
    assert not GlobalProviderConfig.get_config_path().exists()
