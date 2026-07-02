"""E2E endpoint security — the token-exfil gate.

Two complementary proofs that a poisoned endpoint cannot exfiltrate a token:

1. **Subprocess (real CLI):** with a genuinely-disallowed ``OMICS_OS_ENDPOINT``
   (host resolves to a non-omics, non-localhost attacker), every
   credential-bearing command must fail closed — refuse without a traceback. A
   refused endpoint never opens a connection, so there is nothing to record; the
   observable is "refused, did not crash, did not proceed to send".

2. **In-process (precise):** mock ``httpx`` so ANY outbound call is recorded,
   poison the endpoint, and assert the token-sender made ZERO calls. This is the
   exact "no bytes left the process" assertion.

Note on localhost: ``127.0.0.1``/``localhost`` are intentionally allowlisted
(dev override), so a localhost server is NOT a valid stand-in for an attacker —
a token sent there is legitimate dev behaviour, not exfil. The attacker must be a
disallowed host, which by definition is never connected to.

Also runs the allowlist bypass matrix against the shared policy.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


# Genuinely-disallowed endpoints (host resolves to attacker / non-omics).
_GENUINE_POISON = [
    "https://app.omics-os.com@evil.com",  # userinfo host-spoof → real host = evil.com
    "https://evil.com",  # bare attacker
    "https://app.omics-os.com.evil.com",  # suffix append
    "https://evil-omics-os.com",  # no leading dot
]

_TOKEN_COMMANDS = [
    ["cloud", "status"],
    ["cloud", "account"],
]


def _refused(res) -> bool:
    blob = (res.stdout + res.stderr).lower()
    return ("allowlist" in blob) or ("disallowed" in blob) or ("not in allow" in blob)


@pytest.mark.parametrize("command", _TOKEN_COMMANDS, ids=lambda c: "_".join(c))
@pytest.mark.parametrize("poison", _GENUINE_POISON)
def test_poisoned_endpoint_fails_closed(isolated_home, cred_factory, command, poison):
    """Genuine poison → command refuses, no traceback, no send."""
    isolated_home.write(cred_factory.v2_single())
    res = isolated_home.run_cli(command, endpoint_override=poison)
    assert (
        "Traceback" not in res.stderr
    ), f"{command} crashed on {poison}: {res.stderr[-300:]}"
    assert _refused(res), (
        f"{command} did NOT visibly refuse poison {poison}: "
        f"out={res.stdout[-200:]!r} err={res.stderr[-200:]!r}"
    )


def test_no_httpx_call_on_poisoned_endpoint_provider(
    tmp_path, monkeypatch, cred_factory
):
    """In-process: the LOCAL CHAT provider makes ZERO outbound calls on poison.

    This is the worst site — the default `lobster chat` path. A poisoned endpoint
    must raise before any client is constructed / any byte is sent.
    """
    from lobster.config import credentials
    from lobster.config.endpoint_policy import EndpointPolicyError

    cred_dir = tmp_path / "omics-os"
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", cred_dir)
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", cred_dir / "credentials.json")
    credentials.save_credentials(cred_factory.v2_single())
    monkeypatch.setenv("OMICS_OS_ENDPOINT", "https://app.omics-os.com@evil.com")

    from lobster.config.providers.omics_os_provider import OmicsOSProvider

    sentinel = MagicMock(
        side_effect=AssertionError("httpx must not be called on poison")
    )
    with patch("httpx.Client", sentinel):
        with pytest.raises(EndpointPolicyError):
            OmicsOSProvider().create_chat_model("some-model")
    sentinel.assert_not_called()


def test_no_httpx_call_on_poisoned_endpoint_validate(
    tmp_path, monkeypatch, cred_factory
):
    """In-process: login credential-validation sends nothing on poison."""
    from lobster.cli_internal.commands.light.cloud_commands import _validate_credentials

    sentinel = MagicMock(
        side_effect=AssertionError("httpx must not be called on poison")
    )
    with patch("httpx.Client", sentinel):
        result = _validate_credentials(
            "https://app.omics-os.com@evil.com", "omc_TESTFAKE"
        )
    assert result is None
    sentinel.assert_not_called()


# ---------------------------------------------------------------------------
# Allowlist bypass matrix (shared policy — source of truth for all senders)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("https://databiomix.omics-os.com", True),
        ("https://app.omics-os.com", True),
        ("https://DATABIOMIX.OMICS-OS.COM", True),
        ("https://app.omics-os.com.evil.com", False),
        ("https://evil-omics-os.com", False),
        ("https://app.omics-os.com@evil.com", False),
        ("https://evil.com/?x=app.omics-os.com", False),
        ("http://databiomix.omics-os.com", False),
        ("https://omics-os.com", False),
        ("https://x.omics-os.com.", False),
        ("https://app.omics-os.com/path", False),
    ],
)
def test_allowlist_matrix(url, allowed):
    from lobster.config.endpoint_policy import EndpointPolicyError, validate_endpoint

    if allowed:
        validate_endpoint(url)
    else:
        with pytest.raises(EndpointPolicyError):
            validate_endpoint(url)


def test_malformed_ipv6_is_typed_error():
    from lobster.config.endpoint_policy import EndpointPolicyError, validate_endpoint

    for bad in ("https://[::1", "https://[gggg::1]"):
        with pytest.raises(EndpointPolicyError):
            validate_endpoint(bad)
