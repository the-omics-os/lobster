"""`lobster init` no longer asks for a separate Omics-OS Cloud key.

The old optional question saved ``LOBSTER_CLOUD_KEY``, which switched the whole
chat to a retired service. An Omics-OS Cloud account is used through the
``omics-os`` provider or ``lobster cloud login --api-key <key>`` instead.
"""

import io
import sys
from types import ModuleType, SimpleNamespace

import pytest
import typer
from rich.console import Console

from lobster.cli_internal.commands.heavy import init_commands

pytestmark = pytest.mark.unit

QUESTION = "Omics-OS Cloud API key"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    import lobster.i18n as i18n
    from lobster.config import global_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(i18n, "_LANG", "en")
    monkeypatch.setattr(i18n, "_SOURCE", "default")
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global-config")
    for name in ("LOBSTER_CLOUD_KEY", "LOBSTER_WORKSPACE", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: False)
    monkeypatch.setattr(init_commands, "console", Console(file=io.StringIO()))


def _init_kwargs(**overrides):
    import inspect

    kwargs = {
        name: (
            param.default.default
            if isinstance(param.default, typer.models.OptionInfo)
            else param.default
        )
        for name, param in inspect.signature(init_commands.init_impl).parameters.items()
    }
    kwargs.update(overrides)
    return kwargs


def test_classic_wizard_does_not_ask_for_a_cloud_key(monkeypatch, tmp_path):
    asked = []

    def prompt(text, *args, **kwargs):
        asked.append(str(text))
        if "provider" in str(text):
            return "2"  # Claude API (Anthropic)
        if "Claude API key" in str(text):
            return "sk-ant-unit"
        if QUESTION in str(text):
            return "legacy_cloud_key"
        return kwargs.get("default", "1")

    def confirm(text, *args, **kwargs):
        asked.append(str(text))
        return QUESTION in str(text)  # a "y" to the old question

    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(init_commands.Prompt, "ask", prompt)
    monkeypatch.setattr(init_commands.Confirm, "ask", confirm)
    monkeypatch.setattr(
        init_commands,
        "_perform_agent_selection_interactive",
        lambda *args, **kwargs: ([], None),
    )
    try:
        init_commands.init_impl(
            **_init_kwargs(ui_mode="classic", skip_extras=True, skip_ssl_test=True)
        )
    except typer.Exit as error:
        assert error.exit_code == 0
    env = (tmp_path / ".env").read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-unit" in env
    assert "LOBSTER_CLOUD_KEY" not in env
    assert not [text for text in asked if QUESTION in text]


def test_questionary_wizard_does_not_ask_for_a_cloud_key(monkeypatch):
    from lobster.ui.bridge.questionary_fallback import run_questionary_init
    from lobster.ui.wizard.manifest import ProviderDef, WizardManifest

    asked = []
    questionary = ModuleType("questionary")
    questionary.Choice = lambda title, **kwargs: SimpleNamespace(title=title, **kwargs)
    questionary.select = lambda *a, **k: SimpleNamespace(ask=lambda: "ollama")
    questionary.checkbox = lambda *a, **k: SimpleNamespace(ask=lambda: [])
    questionary.text = lambda *a, **k: SimpleNamespace(ask=lambda: "")

    def confirm(prompt, **kwargs):
        asked.append(prompt)
        return SimpleNamespace(ask=lambda: True)  # "y" to every question

    def password(prompt, **kwargs):
        asked.append(prompt)
        return SimpleNamespace(ask=lambda: "legacy_cloud_key")

    questionary.confirm = confirm
    questionary.password = password
    monkeypatch.setitem(sys.modules, "questionary", questionary)
    manifest = WizardManifest(
        providers=[ProviderDef(name="ollama", display_name="Ollama")]
    )
    result = run_questionary_init(manifest)
    assert not [prompt for prompt in asked if QUESTION in prompt]
    assert "cloud_key" not in result


def test_a_cloud_key_from_an_older_go_wizard_is_not_saved(tmp_path):
    from lobster.ui.bridge.init_adapter import apply_tui_init_result

    result = {
        "provider": "anthropic",
        "api_key": "sk-ant-unit",
        "agents": [],
        "cloud_key": "legacy_cloud_key",
    }
    apply_tui_init_result(result, tmp_path / ".lobster_workspace", tmp_path / ".env")
    env = (tmp_path / ".env").read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-unit" in env
    assert "LOBSTER_CLOUD_KEY" not in env and "legacy_cloud_key" not in env
