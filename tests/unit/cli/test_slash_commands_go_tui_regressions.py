from pathlib import Path

import pytest

import lobster.cli_internal.commands.heavy.slash_commands as slash_commands


class _DummyOutput:
    def __init__(self):
        self.messages = []

    def print(self, message, style=None):
        self.messages.append((style, message))

    def print_table(self, table_data):
        self.messages.append(("table", table_data))

    def confirm(self, question):
        self.messages.append(("confirm", question))
        return False

    def prompt(self, question, default=""):
        self.messages.append(("prompt", question))
        return default

    def print_code_block(self, code, language="python"):
        self.messages.append(("code", language, code))


class ProtocolOutputAdapter(_DummyOutput):
    pass


class _DummyClient:
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.session_id = "test_session"
        self.data_manager = type(
            "DM",
            (),
            {"workspace_path": workspace_path, "modalities": {}},
        )()


def test_open_command_executes_and_returns_summary(tmp_path, monkeypatch):
    output = _DummyOutput()
    client = _DummyClient(tmp_path)
    target = tmp_path / "demo.txt"
    target.write_text("hello")

    monkeypatch.setattr(slash_commands, "current_directory", tmp_path)

    called = {}

    def _fake_open_path(path):
        called["path"] = path
        return True, f"Opened {path.name}"

    import lobster.utils as lobster_utils

    monkeypatch.setattr(lobster_utils, "open_path", _fake_open_path)

    summary = slash_commands._execute_command(
        "/open demo.txt",
        client,
        original_command="/open demo.txt",
        output=output,
    )

    assert summary == "Opened demo.txt"
    assert called["path"] == target
    assert ("success", "Opened demo.txt") in output.messages


def test_restore_command_delegates_to_restore_handler(tmp_path, monkeypatch):
    output = _DummyOutput()
    client = _DummyClient(tmp_path)

    seen = {}

    def _fake_restore(_client, _output, pattern="recent"):
        seen["pattern"] = pattern
        return f"restored:{pattern}"

    monkeypatch.setattr(slash_commands, "_command_restore", _fake_restore)

    summary = slash_commands._execute_command(
        "/restore all",
        client,
        original_command="/restore all",
        output=output,
    )

    assert summary == "restored:all"
    assert seen["pattern"] == "all"


def test_config_provider_accepts_direct_provider_name(tmp_path, monkeypatch):
    output = _DummyOutput()
    client = _DummyClient(tmp_path)

    def _fake_provider_switch(_client, _output, provider_name, save):
        return f"provider:{provider_name}:save={save}"

    monkeypatch.setattr(slash_commands, "config_provider_switch", _fake_provider_switch)

    summary = slash_commands._execute_command(
        "/config provider openai --save",
        client,
        original_command="/config provider openai --save",
        output=output,
    )

    assert summary == "provider:openai:save=True"


def test_config_model_accepts_direct_model_name(tmp_path, monkeypatch):
    output = _DummyOutput()
    client = _DummyClient(tmp_path)

    def _fake_model_switch(_client, _output, model_name, save):
        return f"model:{model_name}:save={save}"

    monkeypatch.setattr(slash_commands, "config_model_switch", _fake_model_switch)

    summary = slash_commands._execute_command(
        "/config model sonnet-4",
        client,
        original_command="/config model sonnet-4",
        output=output,
    )

    assert summary == "model:sonnet-4:save=False"


def test_save_command_protocol_mode_avoids_direct_console_usage(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)
    client.data_manager.modalities = {"rna": object(), "atac": object()}

    seen = {}

    def _fake_auto_save_state(force=False):
        seen["force"] = force
        return ["rna.h5ad", "Skipped atac.h5ad"]

    client.data_manager.auto_save_state = _fake_auto_save_state

    class _FailingConsole:
        def print(self, *_args, **_kwargs):
            raise AssertionError("direct console.print should not be used in protocol mode")

        def status(self, *_args, **_kwargs):
            raise AssertionError("direct console.status should not be used in protocol mode")

    monkeypatch.setattr(slash_commands, "console", _FailingConsole())

    summary = slash_commands._execute_command(
        "/save --force",
        client,
        original_command="/save --force",
        output=output,
    )

    assert seen["force"] is True
    assert summary == "Saved 1 items, skipped 1 unchanged"
    assert ("info", "Saved: rna.h5ad") in output.messages
    assert ("info", "Skipped 1 unchanged modalities") in output.messages


def test_clear_command_protocol_mode_avoids_direct_console_usage(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    class _FailingConsole:
        def clear(self):
            raise AssertionError("direct console.clear should not be used in protocol mode")

    monkeypatch.setattr(slash_commands, "console", _FailingConsole())

    summary = slash_commands._execute_command(
        "/clear",
        client,
        original_command="/clear",
        output=output,
    )

    assert summary is None
    assert output.messages == []


def test_exit_command_protocol_mode_avoids_direct_console_and_confirm(tmp_path, monkeypatch):
    output = ProtocolOutputAdapter()
    client = _DummyClient(tmp_path)

    class _FailingConsole:
        def print(self, *_args, **_kwargs):
            raise AssertionError("direct console.print should not be used in protocol mode")

    class _FailingConfirm:
        @staticmethod
        def ask(*_args, **_kwargs):
            raise AssertionError("Confirm.ask should not be used in protocol mode")

    def _failing_display_goodbye(*_args, **_kwargs):
        raise AssertionError("display_goodbye should not be called in protocol mode")

    monkeypatch.setattr(slash_commands, "console", _FailingConsole())
    monkeypatch.setattr(slash_commands, "Confirm", _FailingConfirm)
    monkeypatch.setattr(slash_commands, "display_goodbye", _failing_display_goodbye)

    summary = slash_commands._execute_command(
        "/exit",
        client,
        original_command="/exit",
        output=output,
    )

    assert summary is None
    assert ("confirm", "exit?") in output.messages


def test_exit_command_protocol_mode_can_raise_without_rich_interactive_calls(tmp_path, monkeypatch):
    client = _DummyClient(tmp_path)
    output = ProtocolOutputAdapter()

    def _confirm_true(question):
        output.messages.append(("confirm", question))
        return True

    monkeypatch.setattr(output, "confirm", _confirm_true)

    class _FailingConsole:
        def print(self, *_args, **_kwargs):
            raise AssertionError("direct console.print should not be used in protocol mode")

    class _FailingConfirm:
        @staticmethod
        def ask(*_args, **_kwargs):
            raise AssertionError("Confirm.ask should not be used in protocol mode")

    def _failing_display_goodbye(*_args, **_kwargs):
        raise AssertionError("display_goodbye should not be called in protocol mode")

    monkeypatch.setattr(slash_commands, "console", _FailingConsole())
    monkeypatch.setattr(slash_commands, "Confirm", _FailingConfirm)
    monkeypatch.setattr(slash_commands, "display_goodbye", _failing_display_goodbye)

    with pytest.raises(KeyboardInterrupt):
        slash_commands._execute_command(
            "/exit",
            client,
            original_command="/exit",
            output=output,
        )
