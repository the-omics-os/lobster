"""Gateway Bedrock client shim -- routes converse/converse_stream through Omics-OS gateway."""

import json
import logging
import random
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class _EventIterator:
    """Wraps NDJSON lines into Bedrock EventStream-like dicts."""

    def __init__(self, lines_iter):
        self._lines = lines_iter

    def __iter__(self):
        return self

    def __next__(self):
        for line in self._lines:
            line = line.strip()
            if not line:
                continue
            if line == "[DONE]":
                raise StopIteration
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise StopIteration


class GatewayBedrockClient:
    """boto3 bedrock-runtime client shim routing through the Omics-OS gateway.

    Implements the converse() and converse_stream() surface that
    ChatBedrockConverse uses. All other methods raise NotImplementedError.

    Args:
        endpoint: Gateway base URL (e.g., "https://app.omics-os.com")
        token_fn: Callable returning current bearer token (handles refresh)
        timeout: HTTP read timeout in seconds (default: 600)
    """

    _MAX_RETRIES = 3

    def __init__(
        self,
        endpoint: str,
        token_fn: Callable[[], Optional[str]],
        timeout: float = 600.0,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._token_fn = token_fn
        self._timeout = timeout

    class meta:
        """Required by ChatBedrockConverse for region detection."""

        region_name = "us-east-1"

    def _build_headers(self) -> dict:
        token = self._token_fn()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Omics-Gateway-Contract": "bedrock-converse-v1",
        }

    def _should_retry(self, status_code: int) -> bool:
        return status_code in (429, 503)

    def _retry_delay(self, attempt: int) -> float:
        base = 2**attempt
        return base + random.uniform(0, 0.5)

    def _handle_error(self, status_code: int, body: str) -> None:
        """Raise botocore-compatible ClientError for ChatBedrockConverse error handling."""
        from botocore.exceptions import ClientError

        error_map = {
            401: (
                "UnauthorizedAccess",
                "Omics-OS authentication failed. Run 'lobster cloud login'.",
            ),
            402: (
                "BudgetExceeded",
                "Monthly budget exhausted. Visit https://app.omics-os.com/settings/billing",
            ),
            403: (
                "AccessDeniedException",
                "Model not available for your tier.",
            ),
            422: (
                "ValidationException",
                f"Invalid request: {body}",
            ),
        }

        if status_code in error_map:
            code, msg = error_map[status_code]
            # Try to extract detail from JSON body
            try:
                data = json.loads(body)
                if "error" in data:
                    msg = data["error"].get("message", msg)
                elif "detail" in data:
                    msg = data["detail"]
            except (json.JSONDecodeError, KeyError):
                pass

            raise ClientError(
                error_response={"Error": {"Code": code, "Message": msg}},
                operation_name="Converse",
            )

        if status_code >= 500:
            raise ClientError(
                error_response={
                    "Error": {
                        "Code": "ServiceError",
                        "Message": f"Gateway error ({status_code}): {body}",
                    }
                },
                operation_name="Converse",
            )

    def converse(self, **kwargs) -> dict:
        """Send a Bedrock Converse request through the gateway.

        Returns Bedrock-format response dict with output, stopReason, usage, etc.
        """
        import httpx

        url = f"{self._endpoint}/api/v1/gateway/bedrock/converse"
        headers = self._build_headers()

        last_exc: Optional[Exception] = None
        for attempt in range(self._MAX_RETRIES):
            try:
                with httpx.Client(timeout=self._timeout) as http:
                    resp = http.post(url, json=kwargs, headers=headers)

                if self._should_retry(resp.status_code):
                    last_exc = RuntimeError(f"HTTP {resp.status_code}")
                    time.sleep(self._retry_delay(attempt))
                    continue

                # One token refresh attempt on 401
                if resp.status_code == 401 and attempt == 0:
                    headers = self._build_headers()
                    continue

                if resp.status_code != 200:
                    self._handle_error(resp.status_code, resp.text)

                return resp.json()

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(self._retry_delay(attempt))
                    continue
                from botocore.exceptions import ClientError

                raise ClientError(
                    error_response={
                        "Error": {"Code": "ConnectionError", "Message": str(e)}
                    },
                    operation_name="Converse",
                ) from e

        if last_exc and "429" in str(last_exc):
            from lobster.config.providers.omics_os_provider import RateLimitError

            raise RateLimitError(
                "Rate limit exceeded. Too many API requests in a short window.\n"
                "This can happen when multiple browser tabs are open on app.omics-os.com.\n\n"
                "What to do:\n"
                "  \u2022 Wait 10-15 seconds and retry\n"
                "  \u2022 Close extra browser tabs on app.omics-os.com\n"
                "  \u2022 Run: lobster cloud status",
                retry_after_seconds=10.0,
            )
        raise last_exc or RuntimeError("Max retries exceeded")

    def converse_stream(self, **kwargs) -> dict:
        """Send a Bedrock ConverseStream request through the gateway.

        Returns dict with "stream" key containing an iterator of Bedrock event dicts,
        matching the shape that ChatBedrockConverse expects from converse_stream().
        """
        import httpx

        url = f"{self._endpoint}/api/v1/gateway/bedrock/converse-stream"
        headers = self._build_headers()

        last_exc: Optional[Exception] = None
        for attempt in range(self._MAX_RETRIES):
            try:
                # Use a persistent client for streaming
                http = httpx.Client(timeout=httpx.Timeout(self._timeout, connect=10.0))
                resp = http.send(
                    http.build_request("POST", url, json=kwargs, headers=headers),
                    stream=True,
                )

                if self._should_retry(resp.status_code):
                    resp.close()
                    http.close()
                    last_exc = RuntimeError(f"HTTP {resp.status_code}")
                    time.sleep(self._retry_delay(attempt))
                    continue

                if resp.status_code == 401 and attempt == 0:
                    resp.close()
                    http.close()
                    headers = self._build_headers()
                    continue

                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    resp.close()
                    http.close()
                    self._handle_error(resp.status_code, body)

                # Return Bedrock-compatible response shape
                return {
                    "stream": _EventIterator(resp.iter_lines()),
                    "ResponseMetadata": {"HTTPStatusCode": 200},
                }

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(self._retry_delay(attempt))
                    continue
                from botocore.exceptions import ClientError

                raise ClientError(
                    error_response={
                        "Error": {"Code": "ConnectionError", "Message": str(e)}
                    },
                    operation_name="ConverseStream",
                ) from e

        if last_exc and "429" in str(last_exc):
            from lobster.config.providers.omics_os_provider import RateLimitError

            raise RateLimitError(
                "Rate limit exceeded. Too many API requests in a short window.\n"
                "This can happen when multiple browser tabs are open on app.omics-os.com.\n\n"
                "What to do:\n"
                "  \u2022 Wait 10-15 seconds and retry\n"
                "  \u2022 Close extra browser tabs on app.omics-os.com\n"
                "  \u2022 Run: lobster cloud status",
                retry_after_seconds=10.0,
            )
        raise last_exc or RuntimeError("Max retries exceeded")
