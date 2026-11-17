"""Bounded waits for the OpenAI-compatible chat providers.

The openai SDK defaults wait up to 10 minutes for a response and retry a timed-out
request twice, so a host that goes quiet can hold a run for about 30 minutes. This
module bounds that:

- a stream may be quiet for at most the idle bound (180 s; time between received
  bytes, not the total length of the answer);
- a non-streaming call may wait at most the response bound (600 s); on a stream the
  idle bound also covers the wait for the first byte (with the idle bound off, the
  response bound does);
- a timed-out request is never sent again;
- only connection failures and 429/502/503/504 answers (no body byte consumed yet)
  are retried, at most ``LOBSTER_LLM_CONNECT_RETRIES`` times, honouring Retry-After (capped at 60 s). 408/409 are not retried.

Settings (seconds; ``0`` turns a bound off):
``LOBSTER_LLM_STREAM_IDLE_TIMEOUT_S`` (180), ``LOBSTER_LLM_RESPONSE_TIMEOUT_S`` (600),
``LOBSTER_LLM_CONNECT_RETRIES`` (2). A caller can still pass ``timeout`` or
``max_retries`` for one model; those win. Local Ollama uses 600/600 because a model
load can be slow.

shortcut: the idle bound is time between bytes. A host that keeps sending keep-alive
comments but no content is not caught on the sync path (the async path also gets
langchain-openai's own chunk-gap bound). Add a content-gap guard if that is seen.
shortcut: an environment proxy takes over from the retrying transport, so status
retries are skipped behind a proxy; the timeouts still apply.
"""

import asyncio
import functools
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10.0
_WRITE_TIMEOUT = 30.0
_POOL_TIMEOUT = 30.0
_RETRY_STATUSES = frozenset({429, 502, 503, 504})
_MAX_RETRY_DELAY = 60.0

IDLE_CODE = "provider_idle_timeout"
RESPONSE_CODE = "provider_response_timeout"


class ProviderIdleTimeout(TimeoutError):
    """The host stayed silent past the bound. Carries a stable ``code``."""

    def __init__(self, code: str, seconds: Optional[float]):
        what = "stream was idle" if code == IDLE_CODE else "response did not start"
        bound = "the bound" if seconds is None else f"{seconds:g} s"
        super().__init__(f"{code}: the provider {what} for more than {bound}")
        self.code = code


def _setting(name: str, default: float) -> Optional[float]:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw not in (None, "") else default
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, raw)
        value = default
    return value if value > 0 else None


def _retries() -> int:
    value = _setting("LOBSTER_LLM_CONNECT_RETRIES", 2)
    return int(value) if value else 0


def _openai_httpx():
    # openai 3.x is built on httpx2; older SDKs on httpx. Use the one the SDK uses.
    import openai

    try:
        import httpx2
    except ImportError:
        import httpx

        return httpx
    if issubclass(openai.DefaultHttpxClient, httpx2.Client):
        return httpx2
    import httpx

    return httpx


def _timeout(h, read: Optional[float]):
    return h.Timeout(
        connect=_CONNECT_TIMEOUT, read=read, write=_WRITE_TIMEOUT, pool=_POOL_TIMEOUT
    )


def _delay(response, attempt: int) -> float:
    try:
        wait = float(response.headers.get("Retry-After", ""))
    except ValueError:
        wait = 0.5 * 2**attempt
    return min(max(wait, 0.0), _MAX_RETRY_DELAY)


def _transports(h, retries: int, limits=None):
    limits = limits or h.Limits(max_connections=100, max_keepalive_connections=20)
    """A sync and an async transport of httpx flavour ``h`` that retry as above."""

    class Sync(h.BaseTransport):
        def __init__(self):
            self._inner = h.HTTPTransport(retries=retries, limits=limits)

        def handle_request(self, request):
            for attempt in range(retries + 1):
                response = self._inner.handle_request(request)
                if response.status_code not in _RETRY_STATUSES or attempt == retries:
                    return response
                wait = _delay(response, attempt)
                response.close()
                time.sleep(wait)

        def close(self):
            self._inner.close()

    class Async(h.AsyncBaseTransport):
        def __init__(self):
            self._inner = h.AsyncHTTPTransport(retries=retries, limits=limits)

        async def handle_async_request(self, request):
            for attempt in range(retries + 1):
                response = await self._inner.handle_async_request(request)
                if response.status_code not in _RETRY_STATUSES or attempt == retries:
                    return response
                wait = _delay(response, attempt)
                await response.aclose()
                await asyncio.sleep(wait)

        async def aclose(self):
            await self._inner.aclose()

    return Sync(), Async()


def chat_openai_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return ChatOpenAI ``kwargs`` with the bounds added; the caller's values win."""
    import openai

    h = _openai_httpx()
    # The SDK's own pool sizes; passing a transport would otherwise drop them.
    sync, aio = _transports(
        h, _retries(), h.Limits(max_connections=1000, max_keepalive_connections=100)
    )
    bounded = dict(kwargs)
    if "request_timeout" not in bounded:
        bounded.setdefault(
            "timeout", _timeout(h, _setting("LOBSTER_LLM_RESPONSE_TIMEOUT_S", 600))
        )
    bounded.setdefault("max_retries", 0)
    bounded.setdefault("http_client", openai.DefaultHttpxClient(transport=sync))
    bounded.setdefault(
        "http_async_client", openai.DefaultAsyncHttpxClient(transport=aio)
    )
    return bounded


def chat_ollama_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return ChatOllama ``kwargs`` with the bounds added (600 s read, same retries)."""
    import httpx

    sync, aio = _transports(httpx, _retries())
    timeout = _timeout(httpx, _setting("LOBSTER_LLM_RESPONSE_TIMEOUT_S", 600))
    bounded = dict(kwargs)
    bounded.setdefault("sync_client_kwargs", {"timeout": timeout, "transport": sync})
    bounded.setdefault("async_client_kwargs", {"timeout": timeout, "transport": aio})
    return bounded


def _is_read_timeout(exc: Optional[BaseException]) -> bool:
    # langchain re-raises SDK timeouts, so walk the whole cause chain. Match by class
    # name because the SDK may use httpx or httpx2.
    while exc is not None:
        if type(exc).__name__ == "ReadTimeout":
            return True
        exc = exc.__cause__
    return False


def _named(exc: Exception, code: str, seconds: Optional[float]) -> Exception:
    if isinstance(exc, ProviderIdleTimeout) or not _is_read_timeout(exc):
        return exc
    return ProviderIdleTimeout(code, seconds)


def with_request_bounds(
    llm: Any,
    *,
    idle_default: float = 180,
    response_default: float = 600,
    per_request_timeout: bool = True,
) -> Any:
    """Apply the idle bound to streaming and name read timeouts on every path.

    ``per_request_timeout=False`` (Ollama) leaves the client's read timeout as the only
    bound and passes no ``timeout`` argument to the model.

    Wrap this BEFORE ``with_stream_usage_hint``. shortcut: like that wrapper, these
    close over THIS instance; a ``model_copy`` would call the original's methods.
    """
    idle = _setting("LOBSTER_LLM_STREAM_IDLE_TIMEOUT_S", idle_default)
    response = _setting("LOBSTER_LLM_RESPONSE_TIMEOUT_S", response_default)
    if not per_request_timeout:
        idle = response  # the client's read timeout is the only bound
    sync_stream, async_stream = llm._stream, llm._astream
    sync_generate, async_generate = llm._generate, llm._agenerate

    def _stream_kwargs(kwargs):
        # A stream's read bound also covers waiting for the first byte. With the idle
        # bound off the client's response bound applies instead.
        if per_request_timeout and idle is not None:
            kwargs.setdefault("timeout", _timeout(_openai_httpx(), idle))
        return kwargs

    @functools.wraps(sync_stream)
    def _stream(*args, **kwargs):
        try:
            yield from sync_stream(*args, **_stream_kwargs(kwargs))
        except Exception as exc:
            named = _named(exc, IDLE_CODE, idle)
            if named is exc:
                raise
            raise named from exc

    @functools.wraps(async_stream)
    async def _astream(*args, **kwargs):
        try:
            async for chunk in async_stream(*args, **_stream_kwargs(kwargs)):
                yield chunk
        except Exception as exc:
            named = _named(exc, IDLE_CODE, idle)
            if named is exc:
                raise
            raise named from exc

    @functools.wraps(sync_generate)
    def _generate(*args, **kwargs):
        try:
            return sync_generate(*args, **kwargs)
        except Exception as exc:
            named = _named(exc, RESPONSE_CODE, response)
            if named is exc:
                raise
            raise named from exc

    @functools.wraps(async_generate)
    async def _agenerate(*args, **kwargs):
        try:
            return await async_generate(*args, **kwargs)
        except Exception as exc:
            named = _named(exc, RESPONSE_CODE, response)
            if named is exc:
                raise
            raise named from exc

    llm._stream, llm._astream = _stream, _astream
    llm._generate, llm._agenerate = _generate, _agenerate
    return llm
