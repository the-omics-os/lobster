"""A compatible host that rejects ``stream_options`` must not break a run that worked.

The OpenAI-compatible providers ask for streamed usage (``stream_usage=True``), which
sends ``stream_options.include_usage``. A host that does not know the field answers
400. When that 400 names ``stream_options`` and no chunk was yielded yet, streamed
usage is switched off on that model, one warning carries the hint, and the request is
sent ONCE more. Usage is then missing, which the token tracker counts (cost incomplete,
usage not reported). Any other 400, a 400 after chunks, or a second rejection is raised
with the host's message kept and the hint appended.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

HINT = (
    "This host does not accept stream_options (streamed token usage). Set "
    "stream_usage: false in the model config (additional_model_request_fields: "
    '{"stream_usage": false}) to skip the first request; usage is then not counted '
    "and the run is marked usage-unavailable."
)


def _rejects_stream_options(exc: Exception) -> bool:
    import openai

    return isinstance(exc, openai.BadRequestError) and "stream_options" in str(exc)


def _hinted(exc: Exception) -> Exception:
    if _rejects_stream_options(exc) and HINT not in str(exc):
        # shortcut: relies on BadRequestError(message, *, response, body) (openai SDK);
        # a signature change is caught by the reject tests.
        return type(exc)(f"{exc.message}\n{HINT}", response=exc.response, body=exc.body)
    return exc


def with_stream_usage_hint(llm: Any) -> Any:
    """Wrap this chat model's streaming: retry once without streamed usage when rejected.

    shortcut: the wrappers are instance attributes closing over THIS instance, so a
    ``model_copy(update=...)`` of the model would stream through the original's
    settings. Nothing copies these models today; bind per copy if that changes.
    """
    sync_stream, async_stream = llm._stream, llm._astream

    def _turn_off(exc: Exception) -> None:
        llm.stream_usage = False
        logger.warning("%s Retrying once without it. Host said: %s", HINT, exc)

    def _stream(*args, **kwargs):
        yielded = False
        try:
            for chunk in sync_stream(*args, **kwargs):
                yielded = True
                yield chunk
            return
        except Exception as exc:
            if yielded or not _rejects_stream_options(exc):
                hinted = _hinted(exc)
                if hinted is exc:
                    raise
                raise hinted from exc
            _turn_off(exc)
        try:
            yield from sync_stream(*args, **kwargs)
        except Exception as exc:
            hinted = _hinted(exc)
            if hinted is exc:
                raise
            raise hinted from exc

    async def _astream(*args, **kwargs):
        yielded = False
        try:
            async for chunk in async_stream(*args, **kwargs):
                yielded = True
                yield chunk
            return
        except Exception as exc:
            if yielded or not _rejects_stream_options(exc):
                hinted = _hinted(exc)
                if hinted is exc:
                    raise
                raise hinted from exc
            _turn_off(exc)
        try:
            async for chunk in async_stream(*args, **kwargs):
                yield chunk
        except Exception as exc:
            hinted = _hinted(exc)
            if hinted is exc:
                raise
            raise hinted from exc

    # Instance attributes shadow the class methods; the model class stays ChatOpenAI.
    object.__setattr__(llm, "_stream", _stream)
    object.__setattr__(llm, "_astream", _astream)
    return llm
