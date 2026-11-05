"""
Shared SOFT file pre-download helper.

Replaces 7 copy-pasted blocks across geo_service.py and geo_provider.py.
Downloads GEO SOFT files via HTTPS, bypassing GEOparse's FTP downloader
which fails in many network environments.
"""

import gzip
import os
import tempfile
import urllib.request
import zlib
from pathlib import Path
from typing import Optional

from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context, handle_ssl_error

logger = get_logger(__name__)


def build_soft_url(geo_id: str) -> str:
    """Build HTTPS URL for a GEO SOFT file.

    Handles both GSE (series) and GSM (sample) accessions.
    URL pattern follows NCBI FTP-over-HTTPS structure:
      https://ftp.ncbi.nlm.nih.gov/geo/{type}/{folder_prefix}/{geo_id}/soft/{geo_id}_family.soft.gz

    Args:
        geo_id: GEO accession (e.g. "GSE194247", "GSM1234567")

    Returns:
        Full HTTPS URL for the SOFT file
    """
    prefix = geo_id[:3]  # GSE or GSM
    num_str = geo_id[3:]

    if prefix == "GSM":
        folder_prefix = f"GSM{num_str[:-3]}nnn" if len(num_str) >= 3 else "GSMnnn"
        base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{folder_prefix}/{geo_id}"
    else:
        folder_prefix = f"GSE{num_str[:-3]}nnn" if len(num_str) >= 3 else "GSEnnn"
        base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{folder_prefix}/{geo_id}"

    return f"{base}/soft/{geo_id}_family.soft.gz"


def _valid_soft_file(path: Path, geo_id: str) -> bool:
    """Check complete gzip contents and the requested SOFT entity/accession.

    Consume to EOF even after finding the entity so truncation/CRC errors cannot
    publish a partial cache. This is format validation, not scientific validation.
    """
    entity = b"SAMPLE" if geo_id.startswith("GSM") else b"SERIES"
    accession = geo_id.encode("ascii")
    metadata_key = b"!" + entity.title() + b"_geo_accession"
    in_requested_entity = False
    found_accession = False
    try:
        with gzip.open(path, "rb") as contents:
            for line in contents:
                key, separator, value = line.strip().partition(b"=")
                key, value = key.strip(), value.strip()
                if key.startswith(b"^"):
                    in_requested_entity = (
                        separator == b"="
                        and key == b"^" + entity
                        and value == accession
                    )
                elif in_requested_entity and key == metadata_key:
                    if separator != b"=" or value != accession:
                        return False
                    found_accession = True
        return found_accession
    except (OSError, EOFError, zlib.error):
        return False


def pre_download_soft_file(
    geo_id: str,
    cache_dir: Path,
) -> Optional[Path]:
    """Pre-download SOFT file via HTTPS, bypassing GEOparse's FTP downloader.

    This function handles the common pattern of downloading a GEO SOFT file
    before passing it to GEOparse. Only complete gzip/SOFT content matching the
    requested accession is reused or atomically published; invalid caches are
    discarded and refetched.

    Args:
        geo_id: GEO accession (e.g. "GSE194247", "GSM1234567")
        cache_dir: Directory to store downloaded SOFT files

    Returns:
        Path to the downloaded file, or None if download fails
        (GEOparse will retry via FTP as fallback).

    Raises:
        Exception: On SSL certificate verification failure (not recoverable
        by FTP retry -- indicates environment-level SSL misconfiguration).
    """
    soft_file_path = cache_dir / f"{geo_id}_family.soft.gz"
    temporary_path = None
    soft_url = build_soft_url(geo_id)

    try:
        if soft_file_path.exists():
            if _valid_soft_file(soft_file_path, geo_id):
                logger.debug(f"Using cached SOFT file: {soft_file_path}")
                return soft_file_path
            soft_file_path.unlink()
            logger.debug(f"Discarded invalid cached SOFT file: {soft_file_path}")

        logger.debug(f"Pre-downloading SOFT file using HTTPS: {soft_url}")
        ssl_context = create_ssl_context()
        with urllib.request.urlopen(soft_url, context=ssl_context) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=cache_dir, prefix=f".{geo_id}_soft_", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(response.read())
        if not _valid_soft_file(temporary_path, geo_id):
            raise ValueError("Downloaded SOFT file is incomplete or invalid")
        os.replace(temporary_path, soft_file_path)
        logger.debug(f"Successfully pre-downloaded SOFT file to {soft_file_path}")
        return soft_file_path
    except Exception as e:
        error_str = str(e)
        if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
            handle_ssl_error(e, soft_url, logger)
            raise Exception(
                "SSL certificate verification failed when downloading SOFT file. "
                "See error message above for solutions."
            )
        logger.debug(
            f"Pre-download failed for {geo_id} via HTTPS: {e}. "
            "Falling back to GEOparse downloader."
        )
        return None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
