"""Gateway Bedrock client shim -- routes converse/converse_stream through Omics-OS gateway."""

import json
import logging
import random
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class _EventIterator:
    """Wraps NDJSON lines into Bedrock EventStream-like dicts."""

    def __init__(self, lines_iter, response, http):
        self._lines = lines_iter
        self._response = response
        self._http = http
        self._closed = False
        self._events = 0

    def __iter__(self):
        return self

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._response.close()
        finally:
            self._http.close()

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            for line in self._lines:
                line = line.strip()
                if not line:
                    continue
                if line == "[DONE]":
                    self.close()
                    raise StopIteration
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._events += 1
                return event
            from botocore.exceptions import ClientError

            raise ClientError(
                error_response={
                    "Error": {
                        "Code": "StreamIncomplete",
                        "Message": "Gateway stream ended before [DONE].",
                    }
                },
                operation_name="ConverseStream",
            )
        except BaseException as error:
            self.close()
            import httpx

            if isinstance(error, httpx.TransportError):
                # Not retried: events may already have reached the caller, and
                # replaying the request would duplicate them.
                logger.warning(
                    "Omics-OS Cloud gateway stream stopped after %d event(s) (%s); "
                    "a stream is not retried once its response has started.",
                    self._events,
                    type(error).__name__,
                )
            raise


class GatewayBedrockClient:
    """boto3 bedrock-runtime client shim routing through the Omics-OS gateway.

    Implements the converse() and converse_stream() surface that
    ChatBedrockConverse uses. All other methods raise NotImplementedError.

    Connection failures before the response is received (dropped connection, no
    response in time, a body cut short) are retried with backoff, up to
    ``_MAX_RETRIES`` attempts.
    Trade-off: the gateway may already have run, and billed, a request whose
    response was lost on the way back, so a retry can charge the same call
    twice. Failing the whole agent run instead costs more. A stream is never
    retried once its response has started, because the caller may already have
    received part of the reply.

    Args:
        endpoint: Gateway base URL (e.g., "https://app.omics-os.com")
        token_fn: Callable returning current bearer token (handles refresh)
        timeout: Seconds to wait for a converse() response (default:
            ``_UNARY_RESPONSE_TIMEOUT``); streams use ``stream_idle_timeout``
        stream_idle_timeout: Seconds a converse_stream() response may stay
            silent between events (default: ``_STREAM_IDLE_TIMEOUT``)
    """

    _MAX_RETRIES = 3
    # converse() responses start only after the whole generation, and the hosted
    # endpoint enforces a shorter response limit, so a response cannot
    # legitimately arrive later than this.
    _UNARY_RESPONSE_TIMEOUT = 190.0
    # Bedrock sends stream events continuously; the slowest Bedrock call observed
    # through the gateway took about 10 s.
    _STREAM_IDLE_TIMEOUT = 90.0
    _CONNECT_TIMEOUT = 10.0

    def __init__(
        self,
        endpoint: str,
        token_fn: Callable[[], Optional[str]],
        timeout: Optional[float] = None,
        stream_idle_timeout: Optional[float] = None,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._token_fn = token_fn
        self._timeout = self._UNARY_RESPONSE_TIMEOUT if timeout is None else timeout
        self._stream_idle_timeout = (
            self._STREAM_IDLE_TIMEOUT
            if stream_idle_timeout is None
            else stream_idle_timeout
        )

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
        return base + random.uniform(
            0, 0.5
        )  # nosec B311 # Retry timing jitter, not cryptographic randomness.

    def _retry_transport_error(
        self, error: Exception, attempt: int, operation: str
    ) -> None:
        """Back off before the next attempt, or raise once attempts run out."""
        if attempt < self._MAX_RETRIES - 1:
            delay = self._retry_delay(attempt)
            logger.warning(
                "Omics-OS Cloud gateway connection failed before the response was "
                "received (%s); "
                "retrying (attempt %d of %d) in %.1f s. The failed request may "
                "still have been processed and billed.",
                type(error).__name__,
                attempt + 2,
                self._MAX_RETRIES,
                delay,
            )
            time.sleep(delay)
            return
        from botocore.exceptions import ClientError

        raise ClientError(
            error_response={
                "Error": {
                    "Code": "ConnectionError",
                    "Message": (
                        "Omics-OS Cloud gateway connection failed after "
                        f"{self._MAX_RETRIES} attempts "
                        f"({type(error).__name__}: {error})"
                    ),
                }
            },
            operation_name=operation,
        ) from error

    def _handle_error(self, status_code: int, body: str) -> None:
        """Raise botocore-compatible ClientError for ChatBedrockConverse error handling."""
        from botocore.exceptions import ClientError

        error_map = {
            401: (
                "UnauthorizedAccess",
                "Omics-OS Cloud API key rejected or expired: create a new one at app.omics-os.com.",
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
                detail = None
                if isinstance(data, dict):
                    error = data.get("error")
                    if isinstance(error, dict):
                        detail = error.get("message")
                    elif "error" not in data:
                        detail = data.get("detail")
                if isinstance(detail, str) and detail.strip():
                    msg = detail
                elif status_code == 403:
                    msg = "Request denied (HTTP 403)."
            except (json.JSONDecodeError, KeyError):
                pass

            if status_code == 401:
                msg = (
                    "Omics-OS Cloud API key rejected or expired: create a new one "
                    "at app.omics-os.com. Alternatively, use the npm "
                    "@omicsos/lobster client: lobster-cli cloud login --device."
                )
            else:
                # Preserve the gateway's tier/policy refusal and its detail;
                # a 403 is not evidence that replacing credentials will help.
                msg = f"Omics-OS Cloud: {msg}"

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
                timeout = httpx.Timeout(self._timeout, connect=self._CONNECT_TIMEOUT)
                with httpx.Client(timeout=timeout) as http:
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

            except (httpx.LocalProtocolError, httpx.UnsupportedProtocol):
                raise  # a malformed request fails the same way on every attempt
            except httpx.TransportError as e:
                self._retry_transport_error(e, attempt, "Converse")

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
            # Use a persistent client for streaming; the read timeout bounds both
            # the wait for headers and every gap between events.
            http = httpx.Client(
                timeout=httpx.Timeout(
                    self._stream_idle_timeout, connect=self._CONNECT_TIMEOUT
                )
            )
            try:
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
                    "stream": _EventIterator(resp.iter_lines(), resp, http),
                    "ResponseMetadata": {"HTTPStatusCode": 200},
                }

            except httpx.TransportError as e:
                http.close()
                if isinstance(e, (httpx.LocalProtocolError, httpx.UnsupportedProtocol)):
                    raise  # a malformed request fails the same way on every attempt
                self._retry_transport_error(e, attempt, "ConverseStream")
            except BaseException:
                http.close()
                raise

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
