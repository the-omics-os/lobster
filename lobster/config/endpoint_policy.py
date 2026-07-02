"""Endpoint allowlist policy — dependency-free host validation.

This module is the single source of truth for "is this URL a safe Omics-OS
origin to send a credential to?" It is deliberately dependency-free (stdlib
``urllib.parse`` only) so it can be imported from ``credentials.py``,
``cloud_query.py``, providers, and command modules WITHOUT creating an import
cycle. Do NOT import anything from ``cloud_query``/providers/commands here.

Security rationale: after the credential file became V2 profile-nested,
``get_endpoint()`` returns the active profile's endpoint (a tenant host, or a
poisoned value from a tampered file / ``OMICS_OS_ENDPOINT``). Multiple
token-carrying senders derive their host from ``get_endpoint()``. Every such
sender must validate the endpoint through :func:`validate_endpoint` BEFORE
attaching a bearer token, or a poisoned endpoint exfiltrates the token. (Codex P0)
"""

from urllib.parse import urlparse

_ALLOWED_HOSTS = frozenset(
    {
        "app.omics-os.com",
        "stream.omics-os.com",
        "localhost",
        "127.0.0.1",
        "::1",
    }
)

_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class EndpointPolicyError(Exception):
    """Raised when an endpoint fails the allowlist / origin policy."""

    pass


def is_allowed_host(hostname: str, scheme: str) -> bool:
    """Return True if (hostname, scheme) is an allowed Omics-OS origin.

    Operates on the ALREADY-PARSED hostname (never a substring of the raw URL)
    to stay bypass-safe. Accepts any ``*.omics-os.com`` subdomain over https
    (tenant hosts like ``databiomix.omics-os.com``), plus the exact platform
    hosts and localhost. The apex ``omics-os.com`` is intentionally rejected
    (no leading dot, not a static entry) — mirrors the npm CLI ``isAllowedOrigin``.
    """
    host = (hostname or "").lower()
    if host in _LOCALHOST_HOSTS:
        return True
    if scheme != "https":
        return False
    if host in _ALLOWED_HOSTS:
        return True
    # Leading dot => subdomain only. Rejects apex "omics-os.com",
    # suffix-spoof "evil-omics-os.com", and "x.omics-os.com.evil.com".
    return host.endswith(".omics-os.com")


def validate_endpoint(endpoint: str) -> None:
    """Reject endpoints not in the allowlist to prevent token exfiltration.

    Raises :class:`EndpointPolicyError` on any violation: disallowed host,
    non-https (except localhost), embedded path/query/fragment, or a malformed
    URL. Callers should let this propagate (fail closed) — never send a token
    to an endpoint that did not pass this check.
    """
    try:
        parsed = urlparse(endpoint)
        hostname = parsed.hostname or ""
    except ValueError as e:
        # Malformed URL (e.g. unterminated IPv6 "[::1") — urlparse/.hostname
        # raises ValueError. Convert to our typed error so callers that only
        # catch EndpointPolicyError don't crash on an unhandled ValueError.
        raise EndpointPolicyError(f"Malformed endpoint URL: {endpoint!r:.80}") from e

    if parsed.scheme not in ("https", "http"):
        raise EndpointPolicyError("Only https:// and http:// endpoints supported.")
    if not is_allowed_host(hostname, parsed.scheme):
        raise EndpointPolicyError(
            f"Endpoint '{hostname}' not in allowlist. "
            f"Allowed: *.omics-os.com (https), {', '.join(sorted(_ALLOWED_HOSTS))}."
        )
    if parsed.scheme == "http" and hostname not in _LOCALHOST_HOSTS:
        raise EndpointPolicyError("HTTP only allowed for localhost. Use HTTPS.")
    if parsed.path and parsed.path != "/":
        raise EndpointPolicyError(
            f"Endpoint must be an origin (no path). Got: {endpoint}"
        )
    if parsed.query or parsed.fragment:
        raise EndpointPolicyError(
            f"Endpoint must be an origin (no query/fragment). Got: {endpoint}"
        )
