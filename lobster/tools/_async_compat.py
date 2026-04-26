"""Dual-mode async/sync compatibility for LangChain tools.

When a @tool is defined as `async def`, LangChain creates a StructuredTool
with `func=None` and `coroutine=<the async fn>`. This means `.invoke()` raises
`NotImplementedError: StructuredTool does not support sync invocation`.

Cloud uses `graph.astream()` which calls `tool.ainvoke()` — works fine.
CLI uses `graph.stream()` which calls `tool.invoke()` — breaks.

`enable_sync_fallback(tool)` patches `tool.func` with a sync wrapper that
detects whether an event loop is running and handles both cases:
- No loop: `asyncio.run(coroutine(...))`
- Loop running: offload to a thread via ThreadPoolExecutor
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from langchain_core.tools import BaseTool


def enable_sync_fallback(tool: BaseTool) -> BaseTool:
    """Patch an async-only tool to also support sync .invoke().

    Idempotent — safe to call multiple times on the same tool.
    """
    if tool.func is not None:
        return tool

    coroutine = tool.coroutine
    if coroutine is None:
        return tool

    def _sync_fallback(*args: Any, **kwargs: Any) -> Any:
        coro = coroutine(*args, **kwargs)
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    tool.func = _sync_fallback
    return tool
