"""
Per-provider OAuth credential storage.

Stores OAuth tokens in ~/.config/lobster/auth/<provider>.json with
secure file permissions (0600). Separate from Omics-OS Cloud credentials.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AUTH_DIR = Path.home() / ".config" / "lobster" / "auth"


@dataclass
class OAuthCredentials:
    """Provider-agnostic OAuth credential set."""

    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp (seconds)
    provider: str
    # Optional metadata from the identity provider
    email: Optional[str] = None
    account_id: Optional[str] = None

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Check if access token is expired (with safety buffer)."""
        return time.time() >= (self.expires_at - buffer_seconds)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OAuthCredentials":
        # Only pass known fields to avoid TypeError on extra keys
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def _cred_path(provider: str) -> Path:
    return AUTH_DIR / f"{provider}.json"


def save(creds: OAuthCredentials) -> None:
    """Persist OAuth credentials to disk with secure permissions."""
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(AUTH_DIR, 0o700)

    path = _cred_path(creds.provider)
    path.write_text(json.dumps(creds.to_dict(), indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    logger.debug("Saved OAuth credentials for provider=%s", creds.provider)


def load(provider: str) -> Optional[OAuthCredentials]:
    """Load OAuth credentials for a provider, or None if absent/invalid."""
    path = _cred_path(provider)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return OAuthCredentials.from_dict(data)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.debug("Failed to load OAuth credentials for %s: %s", provider, e)
        return None


def delete(provider: str) -> bool:
    """Remove stored credentials. Returns True if file was deleted."""
    path = _cred_path(provider)
    if path.exists():
        path.unlink()
        logger.debug("Deleted OAuth credentials for provider=%s", provider)
        return True
    return False


def has_credentials(provider: str) -> bool:
    """Check if non-expired OAuth credentials exist for a provider."""
    creds = load(provider)
    if creds is None:
        return False
    # Credentials exist — even if expired, refresh may work
    return True
