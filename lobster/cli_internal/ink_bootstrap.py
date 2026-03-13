"""Bootstrap for the Ink TUI binary — download-on-first-run with SHA256 verification.

On first launch, detects platform, downloads the matching binary from GitHub Releases,
verifies SHA256, and caches at ~/.cache/lobster/bin/lobster-chat.
Subsequent launches use the cached binary directly.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import stat
import sys
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Version pinned here — update when releasing new Ink binary
INK_BINARY_VERSION = "0.1.0"

GITHUB_RELEASE_BASE = (
    "https://github.com/the-omics-os/lobster/releases/download"
)

# Platform → binary name mapping
PLATFORM_MAP = {
    ("Darwin", "arm64"): "lobster-chat-darwin-arm64",
    ("Darwin", "x86_64"): "lobster-chat-darwin-x64",
    ("Linux", "x86_64"): "lobster-chat-linux-x64",
    ("Linux", "aarch64"): "lobster-chat-linux-arm64",
    ("Windows", "AMD64"): "lobster-chat-windows-x64.exe",
    ("Windows", "ARM64"): "lobster-chat-windows-arm64.exe",
}

CACHE_DIR = Path.home() / ".cache" / "lobster" / "bin"
BINARY_NAME = "lobster-chat"


def _detect_platform_key() -> Optional[str]:
    """Detect the platform key for binary download."""
    system = platform.system()
    machine = platform.machine()
    return PLATFORM_MAP.get((system, machine))


def _download_url(binary_name: str) -> str:
    """Build the download URL for a specific binary."""
    tag = f"chat-ui/v{INK_BINARY_VERSION}"
    return f"{GITHUB_RELEASE_BASE}/{tag}/{binary_name}"


def _checksums_url() -> str:
    """Build the URL for the SHA256 checksums file."""
    tag = f"chat-ui/v{INK_BINARY_VERSION}"
    return f"{GITHUB_RELEASE_BASE}/{tag}/checksums.txt"


def _verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Verify SHA256 hash of a downloaded file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash


def _fetch_expected_hash(binary_name: str) -> Optional[str]:
    """Download checksums.txt and extract the hash for our binary."""
    try:
        url = _checksums_url()
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read().decode("utf-8")

        for line in content.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) == 2 and parts[1] == binary_name:
                return parts[0]

        return None
    except Exception as exc:
        logger.debug("Failed to fetch checksums: %s", exc)
        return None


def find_or_download_ink_binary() -> Optional[str]:
    """Find or download the Ink TUI binary.

    Search order:
    1. LOBSTER_INK_BINARY env var
    2. Dev build (lobster-tui-ink/dist/lobster-chat)
    3. Cached download (~/.cache/lobster/bin/lobster-chat)
    4. Download from GitHub Releases → cache → verify SHA256

    Returns the path to the binary, or None if unavailable.
    """
    # 1. Env override
    env = os.environ.get("LOBSTER_INK_BINARY")
    if env and os.path.isfile(env):
        return env

    # 2. Dev build
    dev_path = (
        Path(__file__).resolve().parents[2]
        / "lobster-tui-ink"
        / "dist"
        / "lobster-chat"
    )
    if dev_path.is_file():
        return str(dev_path)

    # 3. Cached binary (with version check)
    cached = CACHE_DIR / BINARY_NAME
    version_file = CACHE_DIR / f"{BINARY_NAME}.version"
    if cached.is_file() and version_file.is_file():
        try:
            cached_version = version_file.read_text().strip()
            if cached_version == INK_BINARY_VERSION:
                return str(cached)
        except OSError:
            pass

    # 4. Download
    binary_name = _detect_platform_key()
    if not binary_name:
        logger.warning(
            "Unsupported platform: %s %s",
            platform.system(),
            platform.machine(),
        )
        return None

    url = _download_url(binary_name)
    logger.info("Downloading Ink TUI binary from %s", url)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = cached.with_suffix(".tmp")

        # Download
        urllib.request.urlretrieve(url, str(tmp_path))

        # Verify SHA256
        expected_hash = _fetch_expected_hash(binary_name)
        if expected_hash:
            if not _verify_sha256(tmp_path, expected_hash):
                logger.error("SHA256 mismatch for %s", binary_name)
                tmp_path.unlink(missing_ok=True)
                return None
            logger.debug("SHA256 verified: %s", expected_hash[:16])
        else:
            logger.warning("Could not verify SHA256 (checksums unavailable)")

        # Make executable (Unix)
        if sys.platform != "win32":
            tmp_path.chmod(tmp_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        # Atomic move
        tmp_path.rename(cached)

        # Record version
        version_file.write_text(INK_BINARY_VERSION)

        logger.info("Ink TUI binary cached at %s", cached)
        return str(cached)

    except Exception as exc:
        logger.error("Failed to download Ink TUI binary: %s", exc)
        # Clean up partial download
        tmp_path = cached.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return None
