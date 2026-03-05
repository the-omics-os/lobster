"""
Shared utilities extracted from geo_service.py.

This module provides constants, enums, dataclasses, and helper functions
that are used across multiple GEO domain modules (metadata_fetch,
download_execution, archive_processing, matrix_parsing, concatenation).

Extracting these shared symbols to a single location eliminates duplication
and provides a clean import target for both geo_service.py (backward compat)
and the new domain modules.
"""

import ftplib
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Archive extension constant and helpers
# ---------------------------------------------------------------------------

ARCHIVE_EXTENSIONS = (".tar", ".tar.gz", ".tgz", ".tar.bz2")


def _is_archive_url(url: str) -> bool:
    """Check if URL points to a tar archive file."""
    lower = url.lower()
    return any(lower.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def _score_expression_file(filename: str) -> float:
    """Score a supplementary file for expression data likelihood.

    Positive scores indicate likely expression data files.
    Negative scores indicate metadata, feature lists, or annotation files.

    Scoring signals:
    - Expression keywords (count, expression, matrix, tpm, fpkm, rpkm) boost score
    - Metadata keywords (barcode, annotation, metadata, clinical, sample_info) penalize
    - Ambiguous keywords (gene, feature) are contextual: positive with expression
      signals, negative when alone (e.g. genes.tsv.gz is a 10X feature list)
    - Structured formats (.h5ad, .h5, .mtx) get bonuses over plain text
    """
    lower = filename.lower()

    # Expression signal keywords and their weights
    EXPRESSION_SIGNALS = {
        "count": 2.0,
        "expression": 2.0,
        "matrix": 1.5,
        "tpm": 2.0,
        "fpkm": 2.0,
        "rpkm": 2.0,
        "normalized": 1.5,
        "processed": 1.0,
    }

    # Metadata signal keywords and their weights (negative)
    METADATA_SIGNALS = {
        "barcode": -2.0,
        "annotation": -1.5,
        "metadata": -2.0,
        "sample_info": -1.5,
        "clinical": -1.5,
        "sample": -1.0,
    }

    # Format bonuses for structured formats
    FORMAT_BONUS = {
        ".h5ad": 1.0,
        ".h5": 0.8,
        ".mtx": 0.5,
    }

    score = 0.0

    # Apply expression signals
    has_expression_context = False
    for keyword, weight in EXPRESSION_SIGNALS.items():
        if keyword in lower:
            score += weight
            has_expression_context = True

    # Apply metadata signals
    for keyword, weight in METADATA_SIGNALS.items():
        if keyword in lower:
            score += weight  # weight is already negative

    # Ambiguous keyword handling: "gene" and "feature"
    # These are positive when combined with expression context,
    # but negative when alone (e.g. genes.tsv.gz, features.tsv.gz)
    for ambiguous in ("gene", "feature"):
        if ambiguous in lower:
            if has_expression_context:
                score += 0.5  # mild boost in expression context
            else:
                score -= 1.5  # penalize standalone (feature list / gene list)

    # Format bonuses
    for ext, bonus in FORMAT_BONUS.items():
        if lower.endswith(ext):
            score += bonus
            break

    return score


# ---------------------------------------------------------------------------
# Multi-modal file filter
# ---------------------------------------------------------------------------

# Patterns that indicate non-RNA modalities
_MODALITY_PATTERNS = {
    "atac": ["atac", "peaks", "fragments", "accessibility", "chromatin"],
    "protein": ["protein", "adt", "antibody", "cite"],
    "spatial": ["spatial", "visium", "slide"],
}


def _is_unsupported_modality_file(
    filename: str, unsupported_types: list
) -> bool:
    """Check if filename matches an unsupported modality pattern."""
    name_lower = filename.lower()
    for mod_type in unsupported_types:
        patterns = _MODALITY_PATTERNS.get(mod_type, [])
        if any(p in name_lower for p in patterns):
            return True
    return False


# ---------------------------------------------------------------------------
# Typed retry result (replaces string sentinel "SOFT_FILE_MISSING")
# ---------------------------------------------------------------------------


class RetryOutcome(Enum):
    """Outcome of a retry-with-backoff attempt."""

    SUCCESS = "success"
    EXHAUSTED = "exhausted"
    SOFT_FILE_MISSING = "soft_file_missing"


@dataclass
class RetryResult:
    """Typed result from _retry_with_backoff replacing mixed str/None returns."""

    outcome: RetryOutcome
    value: Any = None
    retries_used: int = 0

    @property
    def succeeded(self) -> bool:
        return self.outcome == RetryOutcome.SUCCESS

    @property
    def needs_fallback(self) -> bool:
        return self.outcome == RetryOutcome.SOFT_FILE_MISSING


# ---------------------------------------------------------------------------
# Data validity check (extracted from GEOService._is_data_valid)
# ---------------------------------------------------------------------------


def _is_data_valid(data: Optional[Any]) -> bool:
    """
    Check if data is valid (non-None and non-empty).

    Works for both DataFrame and AnnData objects, avoiding AttributeError
    when checking .empty on AnnData which doesn't have that attribute.

    Args:
        data: DataFrame, AnnData, or None

    Returns:
        True if data is valid (non-None and has content), False otherwise
    """
    if data is None:
        return False

    # Lazy imports to avoid heavy dependencies at module level
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return not data.empty
    except ImportError:
        pass

    try:
        import anndata

        if isinstance(data, anndata.AnnData):
            return data.n_obs > 0 and data.n_vars > 0
    except ImportError:
        pass

    # Assume other types are valid if not None
    return True


# ---------------------------------------------------------------------------
# Retry with exponential backoff (extracted from GEOService._retry_with_backoff)
# ---------------------------------------------------------------------------


def _retry_with_backoff(
    operation: Callable[[], Any],
    operation_name: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    is_ftp: bool = False,
    console: Optional[Any] = None,
) -> RetryResult:
    """
    Retry operation with exponential backoff, jitter, and progress reporting.

    Implements production-grade retry logic for transient failures:
    - Exponential backoff: 1s, 2s, 4s, 8s, 16s
    - Jitter: 0.5-1.5x random multiplier (prevents thundering herd)
    - Progress reporting: Updates console during retry delays
    - FTP optimization: Reduced retry count for fast-failing FTP
    - Typed return: Always returns RetryResult (never bare None or strings)

    Args:
        operation: Function to retry (must be idempotent)
        operation_name: Human-readable name for logging
        max_retries: Maximum number of attempts (default: 5, FTP: 2)
        base_delay: Base delay in seconds (default: 1.0)
        is_ftp: Whether this is an FTP operation (affects retry count)
        console: Optional rich console for progress reporting

    Returns:
        RetryResult with outcome, optional value, and retries_used
    """
    import requests

    # FTP connections often fail permanently, not transiently
    if is_ftp:
        max_retries = min(max_retries, 2)

    retry_count = 0
    total_delay = 0.0

    while retry_count < max_retries:
        try:
            result = operation()

            if retry_count > 0:
                logger.info(
                    f"{operation_name} succeeded after {retry_count} retries "
                    f"(total delay: {total_delay:.1f}s)"
                )

            return RetryResult(
                RetryOutcome.SUCCESS, value=result, retries_used=retry_count
            )

        except requests.exceptions.HTTPError as e:
            # Special handling for rate limiting
            if e.response and e.response.status_code == 429:
                delay = base_delay * 10  # Much longer backoff for rate limits
                retry_count += 1
                logger.warning(
                    f"{operation_name} rate limited (429). "
                    f"Waiting {delay:.0f}s before retry {retry_count}/{max_retries}..."
                )
                total_delay += delay

                # Progress reporting (if console available)
                if console:
                    console.print(
                        f"[yellow]⚠ {operation_name} rate limited (attempt {retry_count}/{max_retries})[/yellow]"
                    )
                    console.print(
                        f"[yellow]  Retrying in {delay:.1f}s...[/yellow]"
                    )

                time.sleep(delay)
                continue
            else:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(
                        f"{operation_name} failed after {max_retries} attempts: {e}"
                    )
                    return RetryResult(
                        RetryOutcome.EXHAUSTED, retries_used=retry_count
                    )

                delay = (
                    base_delay * (2 ** (retry_count - 1)) * (0.5 + random.random())
                )
                total_delay += delay

                # Progress reporting (if console available)
                if console:
                    console.print(
                        f"[yellow]⚠ {operation_name} failed (attempt {retry_count}/{max_retries})[/yellow]"
                    )
                    console.print(f"[yellow]  Error: {str(e)[:100]}[/yellow]")
                    console.print(
                        f"[yellow]  Retrying in {delay:.1f}s...[/yellow]"
                    )
                else:
                    logger.warning(
                        f"{operation_name} failed (attempt {retry_count}/{max_retries}). "
                        f"Retrying in {delay:.1f}s... Error: {e}"
                    )

                time.sleep(delay)

        except OSError as e:
            # GEOparse wraps ftplib.error_perm (550) as OSError with specific message
            error_str = str(e)
            if "Download failed" in error_str and (
                "No such file" in error_str or "not public yet" in error_str
            ):
                logger.warning(
                    f"{operation_name} OSError indicates missing file: {error_str[:100]}. "
                    "Skipping retries, triggering fallback mechanism."
                )
                return RetryResult(
                    RetryOutcome.SOFT_FILE_MISSING, retries_used=retry_count
                )
            # Other OSErrors may be transient, fall through to generic handler
            logger.warning(
                f"{operation_name} OSError (may retry): {error_str[:100]}"
            )
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(
                    f"{operation_name} failed after {max_retries} attempts: {e}"
                )
                return RetryResult(RetryOutcome.EXHAUSTED, retries_used=retry_count)
            # Continue with exponential backoff
            delay = base_delay * (2 ** (retry_count - 1)) * (0.5 + random.random())
            total_delay += delay
            logger.warning(
                f"{operation_name} retrying after OSError (attempt {retry_count}/{max_retries}) in {delay:.1f}s"
            )
            time.sleep(delay)

        except ftplib.error_perm as e:
            # Permanent FTP errors (550 = File not found) should not be retried
            error_str = str(e)
            if error_str.startswith("550"):
                logger.warning(
                    f"{operation_name} permanent FTP error: File not found (550). "
                    "Skipping retries, triggering fallback mechanism."
                )
                return RetryResult(
                    RetryOutcome.SOFT_FILE_MISSING, retries_used=retry_count
                )
            # Other FTP error codes may be transient, fall through to generic handler
            logger.warning(f"{operation_name} FTP error: {error_str}")
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(
                    f"{operation_name} failed after {max_retries} attempts: {e}"
                )
                return RetryResult(RetryOutcome.EXHAUSTED, retries_used=retry_count)
            # Continue with exponential backoff
            delay = base_delay * (2 ** (retry_count - 1)) * (0.5 + random.random())
            total_delay += delay
            logger.warning(
                f"{operation_name} retrying after FTP error (attempt {retry_count}/{max_retries}) in {delay:.1f}s"
            )
            time.sleep(delay)

        except Exception as e:
            retry_count += 1

            if retry_count >= max_retries:
                logger.error(
                    f"{operation_name} failed after {max_retries} attempts: {e}"
                )
                return RetryResult(RetryOutcome.EXHAUSTED, retries_used=retry_count)

            # Exponential backoff with jitter
            # Jitter range: 0.5-1.5x multiplier (standard practice)
            delay = base_delay * (2 ** (retry_count - 1)) * (0.5 + random.random())
            total_delay += delay

            # Progress reporting (if console available)
            if console:
                console.print(
                    f"[yellow]⚠ {operation_name} failed (attempt {retry_count}/{max_retries})[/yellow]"
                )
                console.print(f"[yellow]  Error: {str(e)[:100]}[/yellow]")
                console.print(
                    f"[yellow]  Retrying in {delay:.1f}s...[/yellow]"
                )
            else:
                logger.warning(
                    f"{operation_name} failed (attempt {retry_count}/{max_retries}). "
                    f"Retrying in {delay:.1f}s... Error: {e}"
                )

            time.sleep(delay)

    return RetryResult(RetryOutcome.EXHAUSTED, retries_used=retry_count)
