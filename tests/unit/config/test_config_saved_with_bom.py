"""A config file saved as UTF-8 with a BOM (PowerShell 5.1's Set-Content -Encoding
UTF8, Notepad's "UTF-8 with BOM") is read, not replaced by the defaults."""

import json

from lobster.config import global_config
from lobster.config.global_config import GlobalProviderConfig
from lobster.config.workspace_agent_config import WorkspaceAgentConfig
from lobster.config.workspace_config import WorkspaceProviderConfig

BOM = b"\xef\xbb\xbf"


def _save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(BOM + text.encode())
    return path


def test_the_workspace_provider_config(tmp_path):
    _save(tmp_path / "provider_config.json", json.dumps({"global_provider": "ollama"}))
    assert WorkspaceProviderConfig.load(tmp_path).global_provider == "ollama"


def test_the_user_wide_provider_config():
    _save(
        global_config.CONFIG_DIR / "providers.json",
        json.dumps({"default_provider": "ollama"}),
    )
    assert GlobalProviderConfig.load().default_provider == "ollama"


def test_the_workspace_agent_config(tmp_path):
    _save(tmp_path / "config.toml", 'enabled_agents = ["research_agent"]\n')
    assert WorkspaceAgentConfig.load(tmp_path).enabled_agents == ["research_agent"]


def test_the_mcp_overview_reads_the_workspace_provider_config(tmp_path):
    from lobster.cli_internal.mcp_overview import load_provider_config

    path = _save(
        tmp_path / "provider_config.json", json.dumps({"global_provider": "ollama"})
    )
    assert load_provider_config(path).global_provider == "ollama"


def test_the_mcp_overview_reads_the_workspace_agent_config(tmp_path):
    from lobster.cli_internal.mcp_overview import load_agent_config

    path = _save(tmp_path / "config.toml", 'enabled_agents = ["research_agent"]\n')
    assert load_agent_config(path).enabled_agents == ["research_agent"]


def test_the_catalog_reads_the_user_wide_config(tmp_path):
    from lobster.core.catalog.factory import _read_global

    path = _save(tmp_path / "providers.json", json.dumps({"lamin_enabled": True}))
    assert _read_global(path) == {"lamin_enabled": True}
