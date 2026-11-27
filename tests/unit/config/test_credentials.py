"""Lobster keeps its own Omics-OS Cloud key.

`lobster cloud login --api-key` saves OMICS_OS_API_KEY in Lobster's
credentials.env; logout removes only that line. Lobster never reads or changes
the omics-os CLI's sign-in (~/.config/omics-os): a planted sign-in stays
byte-identical through every command. Plus the endpoint allowlist matrix.

All network is mocked; no test makes a live call. All keys are placeholders.
"""

import json
import stat

import pytest
from typer.testing import CliRunner

USAGE = {
    "user_id": "u-placeholder",
    "tier": "free",
    "period": "2026-11",
    "budget": {"remaining_usd": 5},
}
SIGN_IN = {"api_key": "omk_CLI_SIGN_IN", "endpoint": "https://tenant.omics-os.com"}


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    """A home with the omics-os CLI signed in and Lobster's settings in tmp."""
    from lobster.cli_internal.commands.light import cloud_commands
    from lobster.config import global_config

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / ".config" / "lobster")
    monkeypatch.delenv("OMICS_OS_API_KEY", raising=False)
    monkeypatch.delenv("OMICS_OS_ENDPOINT", raising=False)
    monkeypatch.chdir(tmp_path)
    sent = []

    def validate(endpoint, key, token_type="API key"):
        sent.append((endpoint, key))
        return dict(USAGE)

    monkeypatch.setattr(cloud_commands, "_validate_credentials", validate)
    sign_in = tmp_path / ".config" / "omics-os" / "credentials.json"
    sign_in.parent.mkdir(parents=True)
    sign_in.write_text(json.dumps(SIGN_IN))

    class Cloud:
        saved = global_config.get_global_credentials_path()
        before = sign_in.read_bytes()
        requests = sent

        @staticmethod
        def run(*args):
            result = CliRunner().invoke(cloud_commands.cloud_app, list(args))
            result.text = " ".join(result.output.split())  # Rich wraps lines
            return result

        @staticmethod
        def sign_in_untouched():
            return sign_in.read_bytes() == Cloud.before

    return Cloud


def test_login_saves_the_key_in_lobsters_own_settings(cloud):
    result = cloud.run("login", "--api-key", "omk_NEW")
    assert result.exit_code == 0, result.output
    assert "OMICS_OS_API_KEY=omk_NEW" in cloud.saved.read_text().splitlines()
    assert stat.S_IMODE(cloud.saved.stat().st_mode) == 0o600
    assert cloud.requests == [("https://app.omics-os.com", "omk_NEW")]
    assert cloud.sign_in_untouched()


def test_a_new_login_replaces_the_saved_key_and_keeps_other_keys(cloud, monkeypatch):
    cloud.saved.parent.mkdir(parents=True)
    cloud.saved.write_text("ANTHROPIC_API_KEY=sk-x\nOMICS_OS_API_KEY=omk_OLD\n")
    monkeypatch.setenv("OMICS_OS_API_KEY", "omk_OLD")  # as settings loaded it
    result = cloud.run("login", "--api-key", "omk_NEW")
    assert result.exit_code == 0, result.output
    lines = cloud.saved.read_text().splitlines()
    assert "ANTHROPIC_API_KEY=sk-x" in lines
    assert [line for line in lines if "OMICS_OS_API_KEY" in line] == [
        "OMICS_OS_API_KEY=omk_NEW"
    ]
    assert "used first" not in result.text  # the old key came from the file
    assert cloud.sign_in_untouched()


def test_login_says_a_key_in_the_shell_is_used_first(cloud, monkeypatch):
    monkeypatch.setenv("OMICS_OS_API_KEY", "omk_SHELL")
    result = cloud.run("login", "--api-key", "omk_NEW")
    assert result.exit_code == 0, result.output
    assert "used first" in result.text


def test_logout_removes_only_the_omics_os_line(cloud):
    cloud.saved.parent.mkdir(parents=True)
    cloud.saved.write_text("ANTHROPIC_API_KEY=sk-x\nOMICS_OS_API_KEY=omk_OLD\n")
    result = cloud.run("logout")
    assert result.exit_code == 0, result.output
    text = cloud.saved.read_text()
    assert "ANTHROPIC_API_KEY=sk-x" in text.splitlines()
    assert "OMICS_OS_API_KEY" not in text
    assert "Removed the Omics-OS Cloud key" in result.text
    assert "still set" not in result.text
    assert cloud.sign_in_untouched()


def test_logout_names_a_key_that_is_still_set_in_the_shell(cloud, monkeypatch):
    cloud.saved.parent.mkdir(parents=True)
    cloud.saved.write_text("OMICS_OS_API_KEY=omk_OLD\n")
    monkeypatch.setenv("OMICS_OS_API_KEY", "omk_SHELL")
    result = cloud.run("logout")
    assert "OMICS_OS_API_KEY" not in cloud.saved.read_text()
    assert "still set in your shell" in result.text


def test_logout_without_a_saved_key_changes_nothing(cloud):
    result = cloud.run("logout")
    assert result.exit_code == 0, result.output
    assert "nothing to remove" in result.text
    assert not cloud.saved.exists()
    assert cloud.sign_in_untouched()


@pytest.mark.parametrize("command", ["status", "account"])
def test_the_omics_os_cli_sign_in_is_no_key(cloud, command):
    result = cloud.run(command)
    assert result.exit_code == 0, result.output
    assert "lobster cloud login --api-key" in result.text
    assert "omk_CLI_SIGN_IN" not in result.output
    assert cloud.requests == []
    assert cloud.sign_in_untouched()


def test_account_shows_what_the_gateway_reports_for_the_key(cloud, monkeypatch):
    monkeypatch.setenv("OMICS_OS_API_KEY", "omk_SHELL")
    result = cloud.run("account")
    assert result.exit_code == 0, result.output
    assert "u-placeholder" in result.text
    assert cloud.requests == [("https://app.omics-os.com", "omk_SHELL")]


# ---------------------------------------------------------------------------
# Endpoint allowlist bypass matrix (PHASE 4 + Codex edge rows)
# ---------------------------------------------------------------------------


class TestEndpointAllowlist:
    @pytest.mark.parametrize(
        "url,allowed",
        [
            ("https://databiomix.omics-os.com", True),
            ("https://app.omics-os.com", True),
            ("https://stream.omics-os.com", True),
            ("http://localhost:8000", True),
            ("http://127.0.0.1", True),
            ("https://DATABIOMIX.OMICS-OS.COM", True),  # uppercase
            (
                "https://user:pass@app.omics-os.com",
                True,
            ),  # userinfo, host resolves clean
            ("https://app.omics-os.com.evil.com", False),  # suffix append
            ("https://evil-omics-os.com", False),  # no leading dot
            ("https://app.omics-os.com@evil.com", False),  # userinfo host-spoof
            ("https://evil.com/?x=app.omics-os.com", False),  # query smuggle
            ("http://databiomix.omics-os.com", False),  # http non-localhost
            ("https://omics-os.com", False),  # apex rejected
            ("https://x.omics-os.com.", False),  # trailing dot
            ("https://app.omics-os.com/path", False),  # path present
        ],
    )
    def test_validate_endpoint_matrix(self, url, allowed):
        from lobster.config.endpoint_policy import (
            EndpointPolicyError,
            validate_endpoint,
        )

        if allowed:
            validate_endpoint(url)  # must not raise
        else:
            with pytest.raises(EndpointPolicyError):
                validate_endpoint(url)

    def test_malformed_ipv6_raises_policy_error_not_valueerror(self):
        from lobster.config.endpoint_policy import (
            EndpointPolicyError,
            validate_endpoint,
        )

        for bad in ("https://[::1", "https://[gggg::1]"):
            with pytest.raises(EndpointPolicyError):
                validate_endpoint(bad)

    def test_cloud_query_wrapper_maps_to_cloudqueryerror(self):
        from lobster.cli_internal.commands.light.cloud_query import (
            CloudQueryError,
            _validate_endpoint,
        )

        with pytest.raises(CloudQueryError):
            _validate_endpoint("https://evil.com")
        # legit tenant passes
        _validate_endpoint("https://databiomix.omics-os.com")
