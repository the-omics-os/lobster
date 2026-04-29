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


def save_credentials(data: dict) -> None:
    """Save credentials to the credentials file with secure permissions.

    Args:
        data: Credentials dict to persist.
    """
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CREDENTIALS_DIR, 0o700)

    CREDENTIALS_FILE.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    os.chmod(CREDENTIALS_FILE, 0o600)


def clear_credentials() -> None:
    """Delete the credentials file."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()


def is_token_expired() -> bool:
    """Check if the OAuth access token is expired.

    Returns False for non-OAuth auth modes or if expiry is not set.
    """
    creds = load_credentials()
    if not creds or creds.get("auth_mode") != "oauth":
        return False
    expiry = creds.get("token_expiry")
    if not expiry:
        return True
    try:
        from datetime import datetime, timedelta, timezone

        expiry_dt = datetime.fromisoformat(expiry)
        # Consider expired 60s early to avoid edge-case failures
        return expiry_dt < datetime.now(timezone.utc) + timedelta(seconds=60)
    except (ValueError, TypeError):
        return True


DEFAULT_CLIENT_ID = "7lgldp8e72p2lmpmi3gjbnn9uk"


def refresh_token() -> Optional[str]:
    """Refresh the OAuth access token using the stored refresh_token.

    Calls the Cognito token endpoint to get a new access_token.
    Updates the credentials file on success.

    Returns:
        New access_token on success, None on failure.
    """
    creds = load_credentials()
    if not creds or creds.get("auth_mode") != "oauth":
        return None

    refresh_tok = creds.get("refresh_token")
    if not refresh_tok:
        logger.debug("No refresh_token stored, cannot refresh.")
        return None

    client_id = creds.get("client_id", DEFAULT_CLIENT_ID)
    endpoint = creds.get("endpoint", DEFAULT_ENDPOINT).rstrip("/")
    token_url = f"{endpoint}/api/v1/gateway/token/refresh"

    try:
        import httpx

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                token_url,
                json={"refresh_token": refresh_tok, "client_id": client_id},
            )

        if resp.status_code != 200:
            logger.debug(f"Token refresh failed: {resp.status_code} {resp.text}")
            return None

        data = resp.json()
        new_access_token = data.get("access_token")
        if not new_access_token:
            logger.debug("Token refresh response missing access_token.")
            return None

        from datetime import datetime, timedelta, timezone

        creds["access_token"] = new_access_token
        if data.get("id_token"):
            creds["id_token"] = data["id_token"]
        creds["token_expiry"] = (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat()

        save_credentials(creds)
        return new_access_token

    except Exception as e:
        logger.debug(f"Token refresh error: {e}")
        return None


def get_api_key() -> Optional[str]:
    """Get the auth token with env var taking priority over credentials file.

    Priority: OMICS_OS_API_KEY env > credentials file (api_key or access_token).
    For OAuth mode, auto-refreshes expired tokens before returning.
    """
    env_key = os.environ.get("OMICS_OS_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    creds = load_credentials()
    if creds:
        # OAuth mode: use access_token, auto-refresh if expired
        if creds.get("auth_mode") == "oauth":
            if is_token_expired():
                new_token = refresh_token()
                if new_token:
                    return new_token
                # Refresh failed — return stale token, let server reject it
                token = creds.get("access_token")
                if token and str(token).strip():
                    return str(token).strip()
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

    creds = load_credentials()
    if creds:
        endpoint = creds.get("endpoint")
        if endpoint and str(endpoint).strip():
            return str(endpoint).strip().rstrip("/")

    return DEFAULT_ENDPOINT


def get_auth_headers() -> dict:
    """Return authorization headers if an API key is available."""
    api_key = get_api_key()
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}
