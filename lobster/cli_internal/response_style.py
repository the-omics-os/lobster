"""``LOBSTER_RESPONSE_STYLE``: a clean error for a bad value, and the "run" style
for the entries whose reply is read after the work (``lobster query``, ``lobster
mcp --stdio``).

The supervisor prompt reads the variable through ``SupervisorConfig.from_env()``
when the client is built (``graph.py`` passes no config), so setting it around
client construction is enough; ``Settings.supervisor_config`` is built from the
same variable but nothing reads it for the prompt.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator

ENV = "LOBSTER_RESPONSE_STYLE"


def check_env() -> None:
    """Exit with code 2 and one line on stderr when the variable is not a style.

    Without this a bad value ends in a traceback from ``Settings()`` the first
    time the engine is imported.
    """
    import typer

    from lobster.config.supervisor_config import RESPONSE_STYLES

    value = os.environ.get(ENV, "")
    if value.strip() and value.strip().lower() not in RESPONSE_STYLES:
        print(
            f"lobster: {ENV}={value!r} is not valid; expected one of "
            f"{', '.join(RESPONSE_STYLES)}",
            file=sys.stderr,
        )
        raise typer.Exit(2)


@contextmanager
def forced(style: str = "run") -> Iterator[None]:
    """Build the client inside this block to get *style*, whatever the user set.

    The variable the user set is only a default; it is put back afterwards.
    """
    old = os.environ.get(ENV)
    os.environ[ENV] = style
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old
