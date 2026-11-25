"""/config provider reads the whole provider name, never just its first word."""

import json
import sys

import pytest

import lobster.cli_internal.commands.heavy.slash_commands as slash_commands


class _Output:
    def __init__(self):
        self.messages = []

    def print(self, message, style=None):
        self.messages.append((style, message))

    def print_table(self, table_data):
        self.messages.append(("table", table_data))


class _Client:
    def __init__(self, workspace_path):
        self.workspace_path = workspace_path
        self.provider_override = None
        self.switched_to = []

    def switch_provider(self, provider_name):
        self.switched_to.append(provider_name)
        self.provider_override = provider_name
        return {"success": True, "provider": provider_name}


def _run(command, client, output):
    return slash_commands._execute_command(
        command.lower(), client, original_command=command, output=output
    )


@pytest.mark.parametrize(
    "command",
    [
        "/config provider ollama cloud",
        "/config provider Ollama-Cloud",
        "/config provider ollama_cloud",
        "/config provider switch ollama cloud",
    ],
)
def test_two_word_ollama_cloud_selects_ollama_cloud(tmp_path, command):
    client, output = _Client(tmp_path), _Output()
    _run(command, client, output)
    assert client.switched_to == ["ollama-cloud"]


def test_plain_ollama_still_selects_local_ollama(tmp_path):
    client, output = _Client(tmp_path), _Output()
    _run("/config provider ollama", client, output)
    assert client.switched_to == ["ollama"]


@pytest.mark.parametrize(
    "command", ["/config provider foo bar", "/config provider ollama local"]
)
def test_unknown_name_is_refused_and_nothing_changes(tmp_path, command):
    client, output = _Client(tmp_path), _Output()
    assert _run(command, client, output) is None
    assert client.switched_to == []
    assert client.provider_override is None
    assert not (tmp_path / "provider_config.json").exists()
    text = " ".join(str(m) for _, m in output.messages)
    assert "ollama-cloud" in text and "anthropic" in text


def test_save_persists_ollama_cloud(tmp_path):
    client, output = _Client(tmp_path), _Output()
    _run("/config provider ollama cloud --save", client, output)
    assert client.switched_to == ["ollama-cloud"]
    saved = json.loads((tmp_path / "provider_config.json").read_text())
    assert saved["global_provider"] == "ollama-cloud"


def test_missing_langchain_ollama_message_is_short(monkeypatch):
    from lobster.config.providers.ollama_provider import OllamaProvider

    monkeypatch.setitem(sys.modules, "langchain_ollama", None)
    monkeypatch.setattr("lobster.core.uv_tool_env.is_uv_tool_env", lambda: True)
    monkeypatch.setattr(
        "lobster.core.uv_tool_env.build_tool_install_command",
        lambda **_: ["uv", "tool", "install", "x" * 500],
    )
    with pytest.raises(ImportError) as err:
        OllamaProvider().check_dependencies()
    message = str(err.value)
    assert "\n" not in message
    assert len(message) < 120
    assert "langchain-ollama" in message and "lobster init" in message


def _shown(output):
    from rich.text import Text

    return " ".join(Text.from_markup(str(m)).plain for _, m in output.messages)


@pytest.mark.parametrize(
    "command", ["/config provider switch", "/config provider --save"]
)
def test_no_name_asks_for_one(tmp_path, command):
    client, output = _Client(tmp_path), _Output()
    assert _run(command, client, output) is None
    assert client.switched_to == []
    text = _shown(output)
    assert "Name a provider." in text and "anthropic" in text
    assert "Unknown provider" not in text


def test_a_typed_name_is_shown_as_typed_not_as_markup(tmp_path):
    client, output = _Client(tmp_path), _Output()
    _run("/config provider [bold]x", client, output)
    assert client.switched_to == []
    assert "Unknown provider: [bold]x." in _shown(output)
