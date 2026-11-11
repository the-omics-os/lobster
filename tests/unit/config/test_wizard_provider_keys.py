"""The init wizard manifest agrees with the env names the engine reads.

Every init path reduces manifest credentials to two positional slots
(``api_key``, ``api_key_secondary``) in manifest order; ``init_adapter`` turns
the slots into env names. These tests pin that contract offline.
"""

import pytest
from dotenv import dotenv_values

from lobster.config import credentials, global_config
from lobster.config.providers.registry import ProviderRegistry
from lobster.ui.bridge import init_adapter
from lobster.ui.wizard.manifest import _PROVIDER_CREDENTIALS

SLOTS = ("api_key", "api_key_secondary")
# Engine fallbacks that could configure a provider without the written keys.
FALLBACK_ENV = ("GEMINI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")
WITH_CREDENTIALS = sorted(p for p, fields in _PROVIDER_CREDENTIALS.items() if fields)


@pytest.mark.parametrize("global_mode", [False, True], ids=["workspace", "global"])
@pytest.mark.parametrize("provider", WITH_CREDENTIALS)
def test_written_keys_configure_the_engine_provider(
    provider, global_mode, tmp_path, monkeypatch
):
    """T1: slots filled in manifest order configure the engine provider, in a
    workspace .env and in the global credentials file."""
    for names in (*init_adapter._PROVIDER_ENV_KEYS.values(), FALLBACK_ENV):
        for name in names:
            monkeypatch.delenv(name, raising=False)
    # No real login or global config may satisfy is_configured().
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", tmp_path / "omics-os")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "omics-os" / "credentials.json"
    )
    result = {"provider": provider, "agents": []}
    for slot, field in zip(SLOTS, _PROVIDER_CREDENTIALS[provider]):
        result[slot] = (
            "https://example.invalid/" if "ENDPOINT" in field.key else "test-value"
        )
    env_path = tmp_path / ".env"
    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    init_adapter.apply_tui_init_result(
        result, workspace, env_path, global_config=global_mode
    )

    written = global_config.get_global_credentials_path() if global_mode else env_path
    for name, value in dotenv_values(written).items():
        if value is not None:
            monkeypatch.setenv(name, value)
    assert ProviderRegistry.get(provider).is_configured()


@pytest.mark.parametrize("provider", WITH_CREDENTIALS)
def test_manifest_declares_the_written_env_names(provider):
    """T2: each field's env name is one the adapter writes for the provider."""
    written = set(init_adapter._PROVIDER_ENV_KEYS[provider])
    for field in _PROVIDER_CREDENTIALS[provider]:
        assert (field.env_var or field.key) in written, (provider, field.key)


@pytest.mark.parametrize("provider", WITH_CREDENTIALS)
def test_no_collected_field_is_dropped(provider):
    """T3: every field, optional or not, maps to one of the two slots."""
    assert len(_PROVIDER_CREDENTIALS[provider]) <= len(SLOTS), provider
