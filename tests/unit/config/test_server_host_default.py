"""The server host setting and generated .env templates default to localhost."""

import pytest

from lobster.config.settings import Settings


def test_host_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    assert Settings().HOST == "127.0.0.1"


def test_host_env_override_is_kept(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")  # nosec B104 # test value; nothing binds
    assert Settings().HOST == "0.0.0.0"  # nosec B104


@pytest.mark.parametrize("generator", ["config_manager", "config_commands"])
def test_generated_env_template_uses_localhost(tmp_path, monkeypatch, generator):
    monkeypatch.chdir(tmp_path)
    if generator == "config_manager":
        from lobster.config.config_manager import generate_env_template

        generate_env_template()
    else:
        from lobster.cli_internal.commands.light.config_commands import (
            generate_env_impl,
        )

        generate_env_impl()

    lines = (tmp_path / ".env.template").read_text().splitlines()
    assert "HOST=127.0.0.1" in lines
    assert "HOST=0.0.0.0" not in lines
