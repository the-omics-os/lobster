"""
PKCE (Proof Key for Code Exchange) utilities for OAuth 2.0.

Uses only Python stdlib (hashlib, secrets, base64).
Compatible with Python 3.12+.
"""

import base64
import hashlib
import secrets


def generate_pkce() -> tuple[str, str]:
    """Generate a PKCE code verifier and S256 challenge.

    Returns:
        (verifier, challenge) tuple of base64url-encoded strings.
    """
    verifier_bytes = secrets.token_bytes(32)
    verifier = base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode("ascii")

    challenge_digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(challenge_digest).rstrip(b"=").decode("ascii")

    return verifier, challenge
