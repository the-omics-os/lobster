import re

import pytest

from lobster.config import version_check


def test_get_update_message_uses_uv_tool_upgrade_when_in_uv_tool_env(monkeypatch):
    monkeypatch.setattr(
        "lobster.core.uv_tool_env.is_uv_tool_env",
        lambda: True,
    )

    message = version_check.get_update_message("9.9.9")

    assert "Update with: lobster upgrade" in message
    assert "(or: uv tool upgrade lobster-ai)" in message


def test_get_update_message_uses_uv_pip_when_not_in_uv_tool_env(monkeypatch):
    monkeypatch.setattr(
        "lobster.core.uv_tool_env.is_uv_tool_env",
        lambda: False,
    )

    message = version_check.get_update_message("9.9.9")

    assert "Update with: lobster upgrade" in message
    assert "(or: uv pip install --upgrade lobster-ai)" in message


def test_get_update_message_falls_back_to_uv_pip_on_detection_error(monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "lobster.core.uv_tool_env.is_uv_tool_env",
        _boom,
    )

    message = version_check.get_update_message("9.9.9")

    assert "Update with: lobster upgrade" in message
    assert "(or: uv pip install --upgrade lobster-ai)" in message


def _fake_update(monkeypatch, tmp_path):
    """An update is available; record whether anything reached the network or cache."""
    calls = []
    monkeypatch.setattr(version_check, "_get_cache_path", lambda: tmp_path / "c.json")

    def fetch():
        calls.append("fetch")
        return "9.9.9"

    monkeypatch.setattr(version_check, "_fetch_latest_version", fetch)
    return calls


def test_update_notice_goes_to_stderr_never_stdout(monkeypatch, tmp_path, capsys):
    _fake_update(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["lobster", "query", "hello"])
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)

    version_check.maybe_show_update_notification()

    out, err = capsys.readouterr()
    assert out == ""
    plain = re.sub(r"\x1b\[[0-9;]*m", "", err)  # stderr counts as a terminal here
    assert "A new version of lobster-ai is available: 9.9.9" in plain


@pytest.mark.parametrize(
    "argv, stderr_tty",
    [
        (["lobster", "query", "--json", "hello"], True),
        (["lobster", "query", "-j", "hello"], True),
        (["lobster", "chat", "--stdio"], True),
        (["lobster", "query", "hello"], False),
    ],
)
def test_update_check_skipped_for_json_stdio_or_non_tty_stderr(
    monkeypatch, tmp_path, capsys, argv, stderr_tty
):
    calls = _fake_update(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr("sys.stderr.isatty", lambda: stderr_tty)

    version_check.maybe_show_update_notification()

    assert capsys.readouterr() == ("", "")
    assert calls == []  # no network request
    assert not (tmp_path / "c.json").exists()  # no cache write
