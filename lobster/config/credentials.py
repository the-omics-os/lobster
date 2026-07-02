"""
Omics-OS Cloud credentials manager.

Manages API keys and auth tokens stored in ~/.config/omics-os/credentials.json.
Supports environment variable overrides for CI/CD and headless environments.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CREDENTIALS_DIR = Path.home() / ".config" / "omics-os"
CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials.json"

DEFAULT_ENDPOINT = "https://app.omics-os.com"
# Broker/refresh audience is ALWAYS the platform host, regardless of the active
# profile's data endpoint (which may be a tenant host like databiomix.omics-os.com).
# Verified live: the tenant host 401s the unauth refresh path. (D5)
PLATFORM_ENDPOINT = "https://app.omics-os.com"
DEFAULT_CLIENT_ID = "7lgldp8e72p2lmpmi3gjbnn9uk"


def load_credentials() -> Optional[dict]:
    """Load credentials from the credentials file.

    Returns:
        Parsed credentials dict, or None if file doesn't exist or is invalid.
    """
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Failed to load credentials: {e}")
        return None


def _resolve_active(raw: dict) -> tuple[Optional[str], Optional[dict]]:
    """Resolve the active profile from a raw V2 dict → (name, profile_dict).

    Selection order: ``active_profile`` → ``"default"`` → any single dict
    profile. Returns ``(None, None)`` if no usable profile exists. Used by BOTH
    the read view (:func:`load_active_profile`) and the writer
    (``_save_active_profile``) so the two can never diverge on which profile
    they operate against (Codex P2).
    """
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict):
        return None, None
    active = raw.get("active_profile")
    if active and isinstance(profiles.get(active), dict):
        return active, profiles[active]
    if isinstance(profiles.get("default"), dict):
        return "default", profiles["default"]
    for name, prof in profiles.items():
        if isinstance(prof, dict):
            return name, prof
    return None, None


def load_active_profile() -> Optional[dict]:
    """Return a FLAT view of the active credential profile.

    V2 files ({"version":2,"active_profile":..,"profiles":{..}}) are collapsed
    to the active profile's dict. V1 flat files pass through unchanged. Field
    aliasing: expose ``access_token`` from ``cli_token`` when the profile is a
    first-party ``omics_cli`` credential, so downstream Bearer readers stay
    shape-stable.

    This is a READ view — a copy, never the on-disk structure. Writers must use
    ``load_credentials()`` (raw) so their round-trip preserves sibling profiles.
    """
    raw = load_credentials()
    if not raw:
        return None
    # V1 passthrough (flat Cognito / api_key files)
    if raw.get("version") != 2 or "profiles" not in raw:
        return raw
    _, prof = _resolve_active(raw)
    if not isinstance(prof, dict):
        return None
    view = dict(prof)  # copy — never mutate on-disk structure
    # Alias cli_token -> access_token for omics_cli first-party creds.
    # UNCONDITIONAL assign (not setdefault): if a stale access_token coexists
    # with cli_token on the same profile, cli_token is authoritative. (Codex P2)
    if view.get("credential_type") == "omics_cli" and view.get("cli_token"):
        view["access_token"] = view["cli_token"]
    return view


def save_credentials(data: dict) -> None:
    """Save credentials atomically with secure permissions.

    Writes to a temp file (``O_CREAT|O_EXCL``, 0600) → ``fsync`` → ``os.replace``
    over the real path, mirroring the npm CLI's ``saveCredentialFileV2``. This
    makes both tools tear-safe: a crash or a concurrent read can never observe a
    torn file, and a concurrent npm↔Python write degrades to "last writer wins"
    (one update lost) rather than corruption. (D13 — shared advisory lock deferred.)

    Args:
        data: Credentials dict to persist.
    """
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CREDENTIALS_DIR, 0o700)

    tmp = CREDENTIALS_FILE.with_suffix(f".tmp.{os.getpid()}")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, CREDENTIALS_FILE)  # atomic rename over the real path
        os.chmod(CREDENTIALS_FILE, 0o600)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _save_active_profile(updates: dict, remove: tuple = ()) -> None:
    """Merge ``updates`` into the active profile, preserving the on-disk shape.

    V2: read raw, deep-merge ``updates`` into ``profiles[active_profile]`` (via
    the shared :func:`_resolve_active`), keep sibling profiles + ``version`` +
    ``active_profile``, write back V2. V1 (or no file): merge into the flat dict
    and write flat (legacy behaviour).

    ``remove`` names keys to delete from the target profile BEFORE applying
    ``updates`` — used by login writers to clear stale ``omics_cli`` fields
    (``cli_token``/``credential_id``) so a Cognito login over a V2 ``omics_cli``
    profile doesn't leave the profile token-model-incoherent (Codex P1).
    """
    raw = load_credentials() or {}
    if raw.get("version") == 2 and isinstance(raw.get("profiles"), dict):
        name, prof = _resolve_active(raw)
        if name is None:
            # Malformed V2 (no usable profile) — seed a default rather than
            # silently writing flat over a versioned file.
            name = raw.get("active_profile") or "default"
            prof = {}
        target = dict(prof or {})
        for key in remove:
            target.pop(key, None)
        target.update(updates)
        raw["profiles"][name] = target
        save_credentials(raw)
    else:
        for key in remove:
            raw.pop(key, None)
        raw.update(updates)
        save_credentials(raw)


def _clear_active_profile() -> None:
    """Log out of the ACTIVE profile only, preserving sibling profiles (Codex P1).

    V2: remove ``profiles[active_profile]``; if other profiles remain, repoint
    ``active_profile`` to a survivor and write back; if it was the last profile,
    delete the file. V1 (or no file): delete the file (legacy behaviour). A
    whole-file wipe is reserved for an explicit "logout --all" (not implemented).
    """
    raw = load_credentials()
    if not raw or raw.get("version") != 2 or not isinstance(raw.get("profiles"), dict):
        clear_credentials()
        return
    name, _ = _resolve_active(raw)
    profiles = raw["profiles"]
    if name is not None:
        profiles.pop(name, None)
    survivors = [n for n, p in profiles.items() if isinstance(p, dict)]
    if not survivors:
        clear_credentials()
        return
    if raw.get("active_profile") not in survivors:
        raw["active_profile"] = "default" if "default" in survivors else survivors[0]
    save_credentials(raw)


def clear_credentials() -> None:
    """Delete the credentials file."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()


def is_token_expired() -> bool:
    """Check if the OAuth access token is expired.

    Returns False for non-OAuth auth modes or if expiry is not set.
    """
    creds = load_active_profile()
    if not creds or creds.get("auth_mode") != "oauth":
        return False
    expiry = creds.get("token_expiry")
    if not expiry:
        return True
    try:
        from datetime import datetime, timedelta, timezone

        expiry_dt = datetime.fromisoformat(expiry)
        return expiry_dt < datetime.now(timezone.utc) + timedelta(seconds=60)
    except (ValueError, TypeError):
        return True


def _sync_credential_across_profiles(
    credential_id: Optional[str], token_updates: dict
) -> None:
    """Fan rotated token fields to every profile sharing ``credential_id``.

    A tenant login stores one platform-wide credential under BOTH ``default``
    and the tenant profile (same ``credential_id``, different ``endpoint``).
    When one is refreshed the server rotates the shared credential, so the other
    profile's stored tokens are now stale. Mirror npm's
    ``syncRefreshedCredentialAcrossProfiles``: rotate token fields only, preserve
    each profile's own ``endpoint``/``label``/``org_id``. (D8)

    Only ``auth_mode == "oauth"`` profiles with a matching ``credential_id`` are
    updated (exact match — NOT user_id/org_id, which repeat across creds).
    """
    if not credential_id:
        return
    raw = load_credentials()
    if not raw or raw.get("version") != 2 or not isinstance(raw.get("profiles"), dict):
        return
    fields = {
        k: token_updates[k]
        for k in (
            "cli_token",
            "access_token",
            "refresh_token",
            "token_expiry",
            "refresh_expiry",
        )
        if k in token_updates
    }
    changed = False
    for prof in raw["profiles"].values():
        if (
            isinstance(prof, dict)
            and prof.get("auth_mode") == "oauth"
            and prof.get("credential_id") == credential_id
        ):
            prof.update(fields)  # endpoint/label/org_id untouched
            changed = True
    if changed:
        save_credentials(raw)


def refresh_token() -> Optional[str]:
    """Refresh the OAuth access token using the stored refresh_token.

    Two token models coexist (Scope B):

    * First-party ``omics_cli`` (``omc_``/``omr_``) — refreshes via the
      platform-pinned OAuth-CLI endpoint (``app.omics-os.com``, NOT the profile's
      tenant endpoint — D5). The rotated token is parsed as ``cli_token`` ONLY
      (no ``access_token`` fallback — matches npm ``parseOAuthCredentialResponse``);
      the ``refresh_token`` IS rotated server-side (old ``omr_`` burns), so the
      new one must be persisted, and fanned across same-``credential_id`` profiles.
    * Legacy flat Cognito ``access_token`` — keeps the existing Cognito refresh.

    All writes route through ``_save_active_profile`` so sibling profiles survive.

    Returns:
        New Bearer token on success, None on failure.
    """
    creds = load_active_profile()  # flat view of the active profile
    if not creds or creds.get("auth_mode") != "oauth":
        return None

    refresh_tok = creds.get("refresh_token")
    if not refresh_tok:
        logger.debug("No refresh_token stored, cannot refresh.")
        return None

    is_omics_cli = creds.get("credential_type") == "omics_cli" or str(
        refresh_tok
    ).startswith("omr_")

    try:
        import httpx

        if is_omics_cli:
            # First-party omc_ refresh — pinned to PLATFORM, never the tenant endpoint.
            token_url = f"{PLATFORM_ENDPOINT}/api/v1/gateway/oauth/cli/token"
            payload = {
                "grant_type": "refresh_token",
                "credential_id": creds.get("credential_id"),
                "refresh_token": refresh_tok,
            }
        else:
            # Legacy Cognito refresh.
            endpoint = creds.get("endpoint", DEFAULT_ENDPOINT).rstrip("/")
            token_url = f"{endpoint}/api/v1/gateway/token/refresh"
            payload = {
                "refresh_token": refresh_tok,
                "client_id": creds.get("client_id", DEFAULT_CLIENT_ID),
            }

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(token_url, json=payload)

        if resp.status_code != 200:
            logger.debug(f"Token refresh failed: {resp.status_code} {resp.text}")
            return None

        data = resp.json()

    except Exception as e:
        logger.debug(f"Token refresh error: {e}")
        return None

    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)

    if is_omics_cli:
        # npm parses the rotated token as `cli_token` ONLY — no access_token
        # fallback. If absent, treat the refresh as FAILED.
        new_cli = data.get("cli_token")
        if not new_cli:
            logger.debug("omc_ refresh response missing cli_token.")
            return None
        updates = {
            "cli_token": new_cli,
            "access_token": new_cli,  # alias for Bearer readers
            "token_expiry": (now + timedelta(hours=1)).isoformat(),
        }
        # refresh_token IS rotated server-side — persist the new one + expiry.
        if data.get("refresh_token"):
            updates["refresh_token"] = data["refresh_token"]
        if data.get("refresh_expiry"):
            updates["refresh_expiry"] = data["refresh_expiry"]
        _save_active_profile(updates)
        _sync_credential_across_profiles(creds.get("credential_id"), updates)
        return new_cli

    new_access = data.get("access_token")
    if not new_access:
        logger.debug("Token refresh response missing access_token.")
        return None
    updates = {
        "access_token": new_access,
        "token_expiry": (now + timedelta(hours=1)).isoformat(),
    }
    if data.get("id_token"):
        updates["id_token"] = data["id_token"]
    _save_active_profile(updates)
    return new_access


def get_api_key() -> Optional[str]:
    """Get the auth token with env var taking priority over credentials file.

    Priority: OMICS_OS_API_KEY env > credentials file (api_key or access_token).
    For OAuth mode, auto-refreshes expired tokens before returning.
    """
    env_key = os.environ.get("OMICS_OS_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    creds = load_active_profile()
    if creds:
        # OAuth mode: use access_token, auto-refresh if expired
        if creds.get("auth_mode") == "oauth":
            if is_token_expired():
                new_token = refresh_token()
                if new_token:
                    return new_token
                # Refresh failed on an expired token — surface "logged out"
                # instead of serving a dead token the server will reject. (D9)
                return None
            token = creds.get("access_token")
            if token and str(token).strip():
                return str(token).strip()
        # API key mode
        key = creds.get("api_key")
        if key and str(key).strip():
            return str(key).strip()

    return None


def get_fallback_provider_name() -> Optional[str]:
    """Get the configured fallback provider name from credentials.

    .. deprecated::
        Fallback providers are no longer supported. All Omics-OS calls
        route through the gateway. This function always returns None for
        new credentials and will be removed in a future release.
    """
    return None


def get_endpoint() -> str:
    """Get the gateway endpoint URL.

    Priority: OMICS_OS_ENDPOINT env > credentials file > default.
    """
    env_endpoint = os.environ.get("OMICS_OS_ENDPOINT")
    if env_endpoint and env_endpoint.strip():
        return env_endpoint.strip().rstrip("/")

    creds = load_active_profile()
    if creds:
        endpoint = creds.get("endpoint")
        if endpoint and str(endpoint).strip():
            return str(endpoint).strip().rstrip("/")

    return DEFAULT_ENDPOINT


def get_validated_endpoint() -> str:
    """Return :func:`get_endpoint`, validated against the endpoint allowlist.

    Use this (not bare ``get_endpoint()``) wherever a credential-bearing request
    derives its host from the credential file / ``OMICS_OS_ENDPOINT`` — a
    poisoned endpoint would otherwise exfiltrate the token. Raises
    :class:`~lobster.config.endpoint_policy.EndpointPolicyError` on a disallowed
    host; callers should let it propagate (fail closed). (Codex P0)
    """
    from lobster.config.endpoint_policy import validate_endpoint

    endpoint = get_endpoint()
    validate_endpoint(endpoint)
    return endpoint


def get_platform_endpoint() -> str:
    """Return the PLATFORM endpoint (``app.omics-os.com``) for broker/auth flows.

    ``/auth/cli`` login and token refresh always target the platform host, NOT
    the active profile's data endpoint (which may be a tenant host). Use this for
    those flows; use :func:`get_validated_endpoint` for data/session/gateway. (D5)
    """
    return PLATFORM_ENDPOINT


def get_auth_headers() -> dict:
    """Return authorization headers if an API key is available."""
    api_key = get_api_key()
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}
