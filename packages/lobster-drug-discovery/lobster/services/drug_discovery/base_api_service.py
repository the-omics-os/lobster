"""
Shared HTTP base class for drug discovery API services.

Provides ``_get_json`` and ``_post_json`` with:
- Retry on 5xx status codes and timeouts (configurable ``max_retries``)
- Exponential backoff with jitter: ``2^(attempt+1) + random.uniform(0, 1)``
- 429 / Retry-After handling (parses header, sleeps, retries)
- Content-type validation before JSON parsing
- Safe JSON decoding (``try/except ValueError``)
- Connection reuse across retries via ``httpx.Client``
- Configurable ``service_name``, ``default_timeout``, ``default_headers``

Subclasses: ChEMBLService, PubChemService, OpenTargetsService.
"""

import json
import random
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Default headers sent with every request unless overridden.
_BASE_HEADERS: Dict[str, str] = {
    "Accept": "application/json",
    "User-Agent": "lobster-ai/1.0 (https://github.com/the-omics-os/lobster)",
}


class BaseAPIService:
    """
    Shared HTTP base class for drug discovery API services.

    Provides ``_get_json`` and ``_post_json`` helpers that handle retries,
    backoff, content-type validation, JSON decode safety, and 429 rate-limit
    responses. Subclasses inherit these methods and only implement
    domain-specific public API methods.

    Args:
        service_name: Human-readable name used in log messages
                      (e.g. ``"ChEMBL"``, ``"PubChem"``).
        default_timeout: HTTP timeout in seconds for all requests.
        default_headers: Extra headers merged on top of the base
                         ``Accept``/``User-Agent`` headers. Keys in
                         *default_headers* override base keys.
    """

    def __init__(
        self,
        service_name: str = "API",
        default_timeout: float = 45.0,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._service_name = service_name
        self._default_timeout = default_timeout
        # Merge base headers with caller-supplied overrides.
        self._headers: Dict[str, str] = {**_BASE_HEADERS}
        if default_headers:
            self._headers.update(default_headers)

    # =========================================================================
    # GET helper
    # =========================================================================

    def _get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Execute a GET request with retry logic for timeouts, 5xx, and 429.

        Retries up to *max_retries* times with exponential backoff plus
        jitter.  Client errors (4xx other than 429) are never retried.
        The ``httpx.Client`` is reused across retries for connection pooling.

        Returns:
            ``(json_data, None)`` on success, or ``(None, error_message)``
            on failure.
        """
        return self._request_json(
            method="GET",
            url=url,
            params=params,
            json_body=None,
            max_retries=max_retries,
        )

    # =========================================================================
    # POST helper
    # =========================================================================

    def _post_json(
        self,
        url: str,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 0,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Execute a POST request with optional retry logic.

        Supports both ``json=`` (request body) and ``params=`` (URL query
        parameters) simultaneously, which PubChem's async search requires.

        Returns:
            ``(json_data, None)`` on success, or ``(None, error_message)``
            on failure.
        """
        return self._request_json(
            method="POST",
            url=url,
            params=params,
            json_body=json_body,
            max_retries=max_retries,
        )

    # =========================================================================
    # Unified internal request engine
    # =========================================================================

    def _request_json(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]],
        json_body: Optional[Dict[str, Any]],
        max_retries: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Core request loop shared by ``_get_json`` and ``_post_json``.

        Handles:
        - Timeout retries with exponential backoff + jitter
        - 5xx retries
        - 429 / Retry-After header parsing
        - Content-type validation before JSON parsing
        - Safe JSON decoding
        - Connection reuse via a single ``httpx.Client``
        """
        last_error: Optional[str] = None
        label = f"{self._service_name} API"

        with httpx.Client(
            timeout=self._default_timeout, headers=self._headers
        ) as client:
            for attempt in range(max_retries + 1):
                try:
                    if method == "GET":
                        response = client.get(url, params=params)
                    else:
                        response = client.post(
                            url, json=json_body, params=params
                        )

                    # ----- 429 Rate-limit handling -----
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after is not None:
                            try:
                                wait = float(retry_after)
                            except (ValueError, TypeError):
                                wait = 2 ** (attempt + 1)
                        else:
                            wait = 2 ** (attempt + 1)
                        last_error = (
                            f"{label} rate-limited (429) for {method} {url}"
                        )
                        if attempt < max_retries:
                            logger.warning(
                                "%s (attempt %d/%d, retry after %.1fs)",
                                last_error,
                                attempt + 1,
                                max_retries + 1,
                                wait,
                            )
                            time.sleep(wait + random.uniform(0, 1))
                            continue
                        else:
                            logger.error(last_error)
                            return None, last_error

                    response.raise_for_status()

                    # ----- Content-type validation -----
                    content_type = response.headers.get("content-type", "")
                    if "json" not in content_type:
                        msg = (
                            f"{label} returned non-JSON response "
                            f"({content_type}) for {method} {url}"
                        )
                        if attempt < max_retries:
                            last_error = msg
                            logger.warning(
                                "%s (attempt %d/%d)",
                                msg,
                                attempt + 1,
                                max_retries + 1,
                            )
                        else:
                            logger.error(msg)
                            return None, msg

                    else:
                        # ----- Safe JSON decode -----
                        try:
                            return response.json(), None
                        except (ValueError, json.JSONDecodeError) as exc:
                            msg = (
                                f"{label} returned invalid JSON for "
                                f"{method} {url}: {exc}"
                            )
                            logger.error(msg)
                            return None, msg

                except httpx.TimeoutException:
                    last_error = (
                        f"{label} timeout after {self._default_timeout}s "
                        f"for {method} {url}"
                    )
                    logger.warning(
                        "%s (attempt %d/%d)",
                        last_error,
                        attempt + 1,
                        max_retries + 1,
                    )

                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status >= 500 and attempt < max_retries:
                        last_error = (
                            f"{label} HTTP {status} for {method} {url}"
                        )
                        logger.warning(
                            "%s (attempt %d/%d)",
                            last_error,
                            attempt + 1,
                            max_retries + 1,
                        )
                    else:
                        # 4xx (non-429) or final attempt -- do not retry
                        msg = (
                            f"{label} HTTP {status} for {method} {url}: "
                            f"{exc.response.text[:200]}"
                        )
                        logger.error(msg)
                        return None, msg

                except httpx.RequestError as exc:
                    msg = f"{label} request error for {method} {url}: {exc}"
                    logger.error(msg)
                    return None, msg

                # ----- Backoff before next attempt -----
                if attempt < max_retries:
                    time.sleep(2 ** (attempt + 1) + random.uniform(0, 1))

        # Exhausted all retries
        logger.error(last_error)
        return None, last_error
