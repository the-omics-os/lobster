import sys

import pytest


def test_agent_config_endpoint_missing_httpx_uses_dynamic_install_hint(
    monkeypatch, caplog
):
    from lobster.config import agent_config_endpoint

    monkeypatch.setitem(sys.modules, "httpx", None)
    monkeypatch.setattr(
        "lobster.core.component_registry.get_install_command",
        lambda package, is_extra=False: "uv tool install lobster-ai --with httpx",
    )

    with caplog.at_level("WARNING"):
        result = agent_config_endpoint.suggest_agents("analyze scRNA-seq data")

    assert result is None
    assert "Install with: uv tool install lobster-ai --with httpx" in caplog.text


def test_openrouter_provider_missing_langchain_openai_uses_dynamic_install_hint(
    monkeypatch,
):
    from lobster.config.providers.openrouter_provider import OpenRouterProvider

    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    monkeypatch.setattr(
        "lobster.core.component_registry.get_install_command",
        lambda package, is_extra=False: "uv tool install lobster-ai --with langchain-openai",
    )

    provider = OpenRouterProvider()
    with pytest.raises(ImportError) as exc_info:
        provider.create_chat_model("openai/gpt-4o")

    assert (
        "Install with: uv tool install lobster-ai --with langchain-openai"
        in str(exc_info.value)
    )


def test_activate_license_missing_httpx_uses_dynamic_install_hint(monkeypatch):
    from lobster.core.governance.license_manager import activate_license

    monkeypatch.setitem(sys.modules, "httpx", None)
    monkeypatch.setattr(
        "lobster.core.component_registry.get_install_command",
        lambda package, is_extra=False: "uv pip install httpx",
    )

    result = activate_license("TEST-CLOUD-KEY")

    assert result["success"] is False
    assert "run 'uv pip install httpx' for license activation" in result["error"]
