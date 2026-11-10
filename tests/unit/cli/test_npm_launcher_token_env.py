"""Token handoff to the npm chat client: environment when the client says it can, argv otherwise."""

import os
import stat
import sys

import pytest

from lobster.cli_internal import npm_launcher
from lobster.cli_internal.npm_launcher import TOKEN_ENV_VAR, supports_token_env


def _client(tmp_path, body: str) -> str:
    """Write a fake client executable whose `capabilities` subcommand runs `body`."""
    script = tmp_path / "fake-client"
    script.write_text(
        f"#!{sys.executable}\nimport sys, time\n"
        "assert sys.argv[1:] == ['capabilities'], sys.argv\n"
        f"{body}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


@pytest.mark.parametrize(
    "body, expected",
    [
        ("print('{\"token-env\": true}')", True),
        ("print('{\"token-env\": false}')", False),
        ("print('{}')", False),
        ("print('[true]')", False),
        ("print('not json')", False),
        ('print(\'{"token-env": "true"}\')', False),
        ("print('{\"token-env\": true}'); sys.exit(1)", False),
        ("print('{\"token-env\": true}\\n{}')", False),
        ("sys.stdout.buffer.write(b'\\xff\\xfe\\x00')", False),
        ("sys.stdin.read(); print('{\"token-env\": true}')", True),
    ],
)
def test_supports_token_env_requires_exit_0_and_true_flag(tmp_path, body, expected):
    assert supports_token_env(_client(tmp_path, body)) is expected


def test_supports_token_env_false_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(npm_launcher, "CAPABILITIES_TIMEOUT_S", 0.2)
    slow = _client(tmp_path, "time.sleep(5); print('{\"token-env\": true}')")
    assert supports_token_env(slow) is False


def test_supports_token_env_false_when_binary_missing(tmp_path):
    assert supports_token_env(str(tmp_path / "absent")) is False


@pytest.mark.parametrize("capable", [True, False])
def test_cloud_chat_passes_token_by_env_only_when_capable(monkeypatch, capable):
    import typer

    from lobster.cli_internal.commands.light import cloud_commands

    seen = {}

    class FakeProc:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            seen["cmd"], seen["env"] = cmd, kwargs.get("env")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(npm_launcher, "find_npm_binary", lambda: "/bin/fake-client")
    monkeypatch.setattr(npm_launcher, "supports_token_env", lambda binary: capable)
    monkeypatch.setattr("os.isatty", lambda fd: True)
    monkeypatch.setattr("subprocess.Popen", FakeProc)

    with pytest.raises(typer.Exit):
        cloud_commands.chat(
            session_id=None,
            token="tok-123",
            endpoint=None,
            stream_endpoint=None,
            project_id=None,
        )

    on_argv = any("tok-123" in arg for arg in seen["cmd"])
    if capable:
        assert not on_argv
        assert seen["env"][TOKEN_ENV_VAR] == "tok-123"
        assert seen["env"]["PATH"] == os.environ["PATH"]
    else:
        assert on_argv
        assert seen["env"] is None


def test_cloud_chat_reads_token_from_env_var(monkeypatch):
    from typer.testing import CliRunner

    from lobster.cli_internal.commands.light import cloud_commands

    seen = {}

    class FakeProc:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            seen["cmd"], seen["env"] = cmd, kwargs.get("env")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(npm_launcher, "find_npm_binary", lambda: "/bin/fake-client")
    monkeypatch.setattr(npm_launcher, "supports_token_env", lambda binary: True)
    monkeypatch.setattr("os.isatty", lambda fd: True)
    monkeypatch.setattr("subprocess.Popen", FakeProc)

    result = CliRunner().invoke(
        cloud_commands.cloud_app, ["chat"], env={TOKEN_ENV_VAR: "tok-env"}
    )

    assert result.exit_code == 0, result.output
    assert not any("tok-env" in arg for arg in seen["cmd"])
    assert seen["env"][TOKEN_ENV_VAR] == "tok-env"
