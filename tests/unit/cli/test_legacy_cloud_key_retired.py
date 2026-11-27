"""``LOBSTER_CLOUD_KEY`` and ``init --cloud-key`` no longer switch Lobster to a
cloud client.

The client they selected called a service that is retired. Lobster now says the
key is not used, keeps working with the chosen provider, and names
``lobster cloud login --api-key <key>`` for an Omics-OS Cloud account.
"""

import importlib
from types import SimpleNamespace

import pytest
import requests
from rich.console import Console
from typer.testing import CliRunner

from lobster.cli_internal.commands.heavy import session_infra

pytestmark = pytest.mark.unit

WAY_OUT = "lobster cloud login --api-key <key>"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    import lobster.i18n as i18n
    from lobster.config import global_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(i18n, "_LANG", "en")
    monkeypatch.setattr(i18n, "_SOURCE", "default")
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global-config")
    for name in ("LOBSTER_CLOUD_KEY", "LOBSTER_ENDPOINT", "LOBSTER_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("Lobster tried the retired cloud client")

    monkeypatch.setattr(requests.Session, "request", no_network)


def test_chat_with_a_legacy_cloud_key_starts_locally_and_says_so(monkeypatch, tmp_path):
    recorded = Console(record=True, width=200)
    local = SimpleNamespace(profile_timings_enabled=None)
    monkeypatch.setenv("LOBSTER_CLOUD_KEY", "legacy_cloud_key")
    monkeypatch.setattr(session_infra, "console", recorded)
    monkeypatch.setattr(
        session_infra,
        "validate_startup_or_raise_startup_diagnostic",
        lambda workspace=None, provider_override=None, credentials_from=None: tmp_path,
    )
    monkeypatch.setattr("lobster.ui.setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "lobster.core.data_manager_v2.DataManagerV2",
        lambda workspace_path, console, session_dir: SimpleNamespace(),
    )
    monkeypatch.setattr(
        session_infra, "_resolve_profile_timings_flag", lambda profile_timings: False
    )
    monkeypatch.setattr(
        session_infra, "_create_local_agent_client", lambda **kwargs: local
    )

    client = session_infra.init_client_or_raise_startup_diagnostic(workspace=tmp_path)

    assert client is local
    output = " ".join(recorded.export_text().split())
    assert "LOBSTER_CLOUD_KEY is no longer used" in output
    assert WAY_OUT in output
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lobster.lobster_cloud")


def test_init_cloud_key_is_not_saved_and_names_the_way_out(tmp_path):
    from lobster.cli import app

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--non-interactive",
            "--skip-extras",
            "--skip-ssl-test",
            "--anthropic-key",
            "sk-ant-unit",
            "--cloud-key",
            "legacy_cloud_key",
            "--cloud-endpoint",
            "https://legacy.example",
        ],
    )
    assert result.exit_code == 0, result.output
    env = (tmp_path / ".env").read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-unit" in env
    assert "LOBSTER_CLOUD_KEY" not in env and "LOBSTER_ENDPOINT" not in env
    assert "legacy_cloud_key" not in env
    output = " ".join(result.output.split())
    assert "--cloud-key is no longer used" in output
    assert WAY_OUT in output


def test_init_help_no_longer_offers_the_cloud_key():
    from lobster.cli import app

    result = CliRunner().invoke(app, ["init", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--cloud-key" not in result.output
    assert "--cloud-endpoint" not in result.output
    assert "premium" not in result.output.lower()


def test_memory_advice_does_not_name_the_cloud_key():
    from lobster.services.data_access.geo.parser import GEOParser

    advice = GEOParser._get_memory_recommendation(
        None, False, 8.0, 4.0, 10.0, 1000, 5000
    )
    assert "LOBSTER_CLOUD_KEY" not in advice and "cloud mode" not in advice
    assert "1. Subsample to ~1,000 cells" in advice
    assert "2. Increase system RAM to 10.0 GB" in advice
