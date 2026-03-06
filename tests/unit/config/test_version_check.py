from lobster.config import version_check


def test_get_update_message_uses_uv_tool_upgrade_when_in_uv_tool_env(monkeypatch):
    monkeypatch.setattr(
        "lobster.core.uv_tool_env.is_uv_tool_env",
        lambda: True,
    )

    message = version_check.get_update_message("9.9.9")

    assert "Update with: uv tool upgrade lobster-ai" in message


def test_get_update_message_uses_uv_pip_when_not_in_uv_tool_env(monkeypatch):
    monkeypatch.setattr(
        "lobster.core.uv_tool_env.is_uv_tool_env",
        lambda: False,
    )

    message = version_check.get_update_message("9.9.9")

    assert "Update with: uv pip install --upgrade lobster-ai" in message


def test_get_update_message_falls_back_to_uv_pip_on_detection_error(monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "lobster.core.uv_tool_env.is_uv_tool_env",
        _boom,
    )

    message = version_check.get_update_message("9.9.9")

    assert "Update with: uv pip install --upgrade lobster-ai" in message
