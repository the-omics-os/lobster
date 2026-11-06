"""Hosted discovery, configuration, and inference routing regression tests."""

from unittest.mock import Mock, patch

import pytest
import requests
import typer

from lobster.config import provider_setup
from lobster.config.constants import VALID_PROVIDERS
from lobster.config.global_config import GlobalProviderConfig
from lobster.config.providers import get_provider
from lobster.config.providers.ollama_cloud_provider import OllamaCloudProvider
from lobster.config.workspace_config import WorkspaceProviderConfig


@pytest.fixture(autouse=True)
def clear_cache():
    OllamaCloudProvider._models_cache = None
    yield
    OllamaCloudProvider._models_cache = None


def reply(data, status=200):
    return Mock(status_code=status, json=Mock(return_value=data))


def tags(*items):
    return {"models": list(items)}


def model(name="glm-5.2", caps=None):
    return {
        "name": name,
        "capabilities": caps if caps is not None else ["completion", "tools"],
    }


@pytest.mark.parametrize("key,expected", [("", False), ("  ", False), ("test", True)])
def test_configuration(monkeypatch, key, expected):
    monkeypatch.setenv("OLLAMA_API_KEY", key)
    p = get_provider("ollama-cloud")
    assert p.is_configured() == expected
    assert p.is_available() == expected
    assert p.get_default_model() == "glm-5.2"
    assert provider_setup.create_ollama_cloud_config(key).success == expected


@patch("requests.get")
def test_discovery_metadata_filter_cache(get):
    get.side_effect = [
        reply(
            tags(
                model(),
                model(),
                model("new-model"),
                model("clef", ["decision"]),
                None,
                model("bad name"),
            )
        ),
        reply({"models": [{"slug": "glm-5.2", "context_window": 1000000}]}),
    ]
    p = OllamaCloudProvider()
    rows = p.list_models()
    assert [m.name for m in rows] == ["glm-5.2", "new-model"]
    assert rows[0].context_window == 1000000
    assert rows[0].input_cost_per_million > 0
    assert rows[1].input_cost_per_million is None
    assert rows[1].context_window == 8192
    rows[0].name = "mutated"
    assert p.list_models()[0].name == "glm-5.2"
    assert get.call_count == 2
    assert all(
        c.kwargs == {"timeout": 5, "allow_redirects": False} for c in get.call_args_list
    )


@pytest.mark.parametrize(
    "payload",
    [None, [], {}, {"models": None}, tags(), tags(model("clef", ["decision"]))],
)
@patch("requests.get")
def test_bad_catalog_fallback(get, payload):
    get.return_value = reply(payload)
    rows = OllamaCloudProvider().list_models()
    assert [m.name for m in rows] == ["glm-5.2", "gpt-oss:120b", "gpt-oss:20b"]
    assert sum(m.is_default for m in rows) == 1


@pytest.mark.parametrize("status", [301, 401, 429, 500])
@patch("requests.get")
def test_http_failure(get, status):
    get.return_value = reply({}, status)
    assert OllamaCloudProvider().list_models()[0].is_default


@patch("requests.get")
def test_metadata_failure_preserves_catalog(get):
    get.side_effect = [reply(tags(model("brand-new"))), requests.Timeout()]
    assert OllamaCloudProvider().list_models()[0].name == "brand-new"


@patch("requests.get", side_effect=requests.Timeout())
def test_timeout_fallback(get):
    assert len(OllamaCloudProvider().list_models()) == 3


@patch("langchain_openai.ChatOpenAI")
def test_client_routing(client, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", " test-key ")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9999")
    OllamaCloudProvider().create_chat_model("explicit-id", max_tokens=123)
    assert client.call_args.kwargs["base_url"] == "https://ollama.com/v1"
    assert client.call_args.kwargs["api_key"] == "test-key"
    assert client.call_args.kwargs["model"] == "explicit-id"
    assert client.call_args.kwargs["max_tokens"] == 123


def test_missing_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
        OllamaCloudProvider().create_chat_model("glm-5.2")


@pytest.mark.parametrize("cls", [WorkspaceProviderConfig, GlobalProviderConfig])
def test_model_persistence(cls):
    cfg = cls()
    cfg.set_model_for_provider("ollama-cloud", "glm-5.2")
    restored = cls.model_validate_json(cfg.model_dump_json())
    assert restored.get_model_for_provider("ollama-cloud") == "glm-5.2"
    cfg.reset()
    assert cfg.get_model_for_provider("ollama-cloud") is None


def test_provider_choice():
    assert provider_setup.validate_provider_choice(
        False, False, False, has_ollama_cloud=True
    )[0]
    assert (
        provider_setup.get_provider_priority_warning(
            False, False, False, has_ollama_cloud=True
        )
        is None
    )
    assert "Azure" in provider_setup.get_provider_priority_warning(
        False, False, False, has_azure=True, has_ollama_cloud=True
    )


def test_ollama_cloud_init_adapter_writes_env_and_config(tmp_path):
    """The wizard result for ollama_cloud must produce a usable .env + provider config.

    Regression: `ollama_cloud` was absent from _PROVIDER_ENV_KEYS, so this raised
    ValueError and `lobster init` could not complete for Nebius at all.
    """
    import json

    from lobster.ui.bridge.init_adapter import apply_tui_init_result

    workspace = tmp_path / ".lobster_workspace"
    env_path = tmp_path / ".env"

    apply_tui_init_result(
        {
            "provider": "ollama-cloud",
            "api_key": "test-key",
            "model_id": "Qwen/Qwen3-32B",
        },
        workspace_path=workspace,
        env_path=env_path,
    )

    assert "OLLAMA_API_KEY=test-key" in env_path.read_text()

    saved = json.loads((workspace / "provider_config.json").read_text())
    assert saved["global_provider"] == "ollama-cloud"
    assert saved["ollama_cloud_model"] == "Qwen/Qwen3-32B"


def test_ollama_cloud_present_in_init_wizard_manifest(monkeypatch):
    monkeypatch.setattr(
        OllamaCloudProvider, "list_models", lambda self: self.static_models()
    )
    """The wizard manifest must carry Ollama Cloud credentials + model catalog.

    Regression: without a _PROVIDER_CREDENTIALS entry the questionary flow
    prompted for no API key, producing an unconfigured provider.
    """
    from lobster.ui.wizard.manifest import build_init_manifest

    manifest = build_init_manifest(detect_ollama=False)
    ollama_cloud = next(
        (p for p in manifest.providers if p.name == "ollama-cloud"), None
    )

    assert ollama_cloud is not None
    assert ollama_cloud.display_name == "Ollama Cloud"
    assert ollama_cloud.model_selection == "explicit"
    assert [c.key for c in ollama_cloud.credentials] == ["OLLAMA_API_KEY"]
    assert len(ollama_cloud.models) >= 3
    assert any(m.is_default for m in ollama_cloud.models)


def test_ollama_cloud_provider_package_maps_cover_ollama_cloud():
    """Auto-install maps must resolve ollama_cloud -> langchain-openai."""
    from lobster.cli_internal.commands.heavy.init_commands import (
        _PROVIDER_IMPORT_NAMES,
        _PROVIDER_PACKAGES,
    )

    assert _PROVIDER_PACKAGES["ollama-cloud"] == "langchain-openai"
    assert _PROVIDER_IMPORT_NAMES["ollama-cloud"] == "langchain_openai"


def test_ollama_cloud_validates_as_a_provider_choice():
    """--ollama-cloud-key alone must satisfy the non-interactive provider check.

    Regression: validate_provider_choice() had no has_ollama_cloud parameter, so
    `lobster init --non-interactive --ollama-cloud-key=...` aborted with
    "No provider specified".
    """
    from lobster.config.provider_setup import (
        get_provider_priority_warning,
        validate_provider_choice,
    )

    valid, error = validate_provider_choice(
        has_anthropic=False,
        has_bedrock=False,
        has_ollama=False,
        has_ollama_cloud=True,
    )
    assert valid
    assert error is None

    # Nebius alone is not a conflict; Nebius + a higher-priority provider is.
    assert (
        get_provider_priority_warning(
            has_anthropic=False,
            has_bedrock=False,
            has_ollama=False,
            has_ollama_cloud=True,
        )
        is None
    )
    warning = get_provider_priority_warning(
        has_anthropic=True,
        has_bedrock=False,
        has_ollama=False,
        has_ollama_cloud=True,
    )
    assert warning is not None and "Claude API" in warning


def test_ollama_cloud_extra_is_declared_in_pyproject():
    """LLM_PROVIDER_PACKAGES must point at extras that actually exist.

    `_build_uv_tool_init_command` passes the provider name through as a
    lobster-ai extra. uv ignores unknown extras *without failing*, so an
    undeclared extra makes the uv-tool install a silent no-op that leaves
    langchain-openai missing.

    Asserts the ollama_cloud entry by name — a loop over whatever happens to be in
    LLM_PROVIDER_PACKAGES would still pass if the entry were deleted.
    """
    import tomllib
    from pathlib import Path

    from lobster.core.component_registry import LLM_PROVIDER_PACKAGES

    pyproject_data = tomllib.loads(
        (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text()
    )
    extras = pyproject_data["project"]["optional-dependencies"]

    assert LLM_PROVIDER_PACKAGES["ollama-cloud"] == ("openai", "langchain-openai")
    assert LLM_PROVIDER_PACKAGES["openrouter"] == ("openrouter", "langchain-openai")

    # Every mapped extra must exist AND actually ship its mapped package.
    for provider, (extra, pypi_package) in LLM_PROVIDER_PACKAGES.items():
        assert extra in extras, f"{provider} maps to undeclared extra '{extra}'"
        assert any(
            req.split(">")[0].split("=")[0].split("[")[0].strip() == pypi_package
            for req in extras[extra]
        ), f"extra '{extra}' does not install {pypi_package}"

    # Both gateway extras must be reachable from the all-providers meta-extra.
    all_providers = " ".join(extras["all-providers"])
    assert "openai" in all_providers
    assert "openrouter" in all_providers


def test_provider_icons_cover_every_provider_and_match_go_tui():
    """PROVIDER_ICONS is the shared palette — Go and Python must agree.

    The Go TUI's providerIcon() renders the same status line as the Python CLI,
    so a provider missing from either side, or mapped to a different glyph,
    shows up as an inconsistent UI.
    """
    import re
    from pathlib import Path

    from lobster.config.constants import PROVIDER_ICONS

    assert set(PROVIDER_ICONS) == set(VALID_PROVIDERS)

    go_source = (
        Path(__file__).resolve().parents[3]
        / "lobster-tui"
        / "internal"
        / "chat"
        / "model.go"
    )
    if not go_source.exists():  # pragma: no cover - Go TUI not vendored
        pytest.skip("lobster-tui sources not present")

    body = go_source.read_text().split("func providerIcon(")[1].split("\n}")[0]
    go_icons = dict(re.findall(r'case "([^"]+)":\s*\n\s*return "([^"]+)"', body))

    for provider, icon in PROVIDER_ICONS.items():
        assert go_icons.get(provider) == icon, (
            f"provider '{provider}': Python has {icon!r}, "
            f"Go has {go_icons.get(provider)!r}"
        )


def test_ollama_cloud_key_flag_is_exposed_and_forwarded():
    """`lobster init` must expose --ollama-cloud-key and forward it to init_impl.

    Regression: the flag did not exist, so Ollama Cloud was unreachable from CI/CD.
    """
    import inspect

    from lobster.cli import app
    from lobster.cli_internal.commands.heavy.init_commands import init_impl

    init_cmd = typer.main.get_command(app).commands["init"]
    flags = {opt for param in init_cmd.params for opt in param.opts}
    assert "--ollama-cloud-key" in flags

    # The flag is only useful if it reaches init_impl under the same name.
    assert "ollama_cloud_key" in inspect.signature(init_impl).parameters


def test_ollama_cloud_non_interactive_init_writes_credentials(tmp_path, monkeypatch):
    """--ollama-cloud-key must produce a usable .env + provider_config.json.

    Calls init_impl the way the CLI does. init_impl's defaults are
    ``typer.OptionInfo`` sentinels when it is invoked directly, so they are
    resolved to their real values first — passing them by hand would silently
    rot as the signature grows.
    """
    import inspect
    import io
    import json

    from rich.console import Console

    from lobster.cli_internal.commands.heavy import init_commands
    from lobster.cli_internal.commands.heavy.init_commands import init_impl

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(tmp_path / ".lobster_workspace"))
    # Own console: other CLI tests can leave the shared one writing to a
    # closed stream, which would surface here as an unrelated ValueError.
    monkeypatch.setattr(init_commands, "console", Console(file=io.StringIO()))

    kwargs = {}
    for name, param in inspect.signature(init_impl).parameters.items():
        default = param.default
        if isinstance(default, typer.models.OptionInfo):
            default = default.default
        kwargs[name] = None if default is inspect.Parameter.empty else default

    kwargs.update(
        non_interactive=True,
        force=True,
        skip_extras=True,
        skip_ssl_test=True,
        ollama_cloud_key="test-ollama_cloud-key",
    )

    with pytest.raises(typer.Exit) as exc_info:
        init_impl(**kwargs)
    assert exc_info.value.exit_code == 0

    env_text = (tmp_path / ".env").read_text()
    assert "LOBSTER_LLM_PROVIDER=ollama-cloud" in env_text
    assert "OLLAMA_API_KEY=test-ollama_cloud-key" in env_text

    saved = json.loads(
        (tmp_path / ".lobster_workspace" / "provider_config.json").read_text()
    )
    assert saved["global_provider"] == "ollama-cloud"


def test_cli_model_listing_includes_unknown_prices(monkeypatch):
    import io

    from rich.console import Console

    from lobster.cli_internal.commands.light.config_commands import list_models_impl

    stream = io.StringIO()
    monkeypatch.setattr(
        "lobster.ui.console_manager.get_console_manager",
        lambda: Mock(console=Console(file=stream, width=200)),
    )
    monkeypatch.setattr(
        OllamaCloudProvider, "list_models", lambda self: [self._model_info("new-model")]
    )
    list_models_impl()
    assert "new-model" in stream.getvalue()
    assert "unknown" in stream.getvalue()
