"""Client session identity shown in notebook exports.

The client that owns a session (``AgentClient``, ``CommandClient``) binds its
session id to the data manager; the notebook exporter reads that binding. The
rule below is a display allowlist for that one purpose, not validation of how
sessions are created or selected: an id outside it is shown as unknown.
"""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DISPLAY_CLIENT_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def display_client_session_id(value: Any) -> Optional[str]:
    """The exact text of ``value`` when a notebook export can show it, else None.

    The result is always a plain ``str`` holding the characters that were
    checked, so a ``str`` subclass cannot change how it is compared or
    rendered later. Never raises.
    """
    try:
        if not isinstance(value, str):
            return None
        text = str.__str__(value)
        if type(text) is not str:
            return None
        if _DISPLAY_CLIENT_SESSION_ID.fullmatch(text) is None:
            return None
        return text
    except Exception:
        return None


def is_display_client_session_id(value: Any) -> bool:
    """True when a notebook export can show ``value`` as the client session id."""
    return display_client_session_id(value) is not None


def bind_client_session(
    data_manager: Any, session_id: Any, *, replace: bool = False
) -> None:
    """Bind ``session_id`` on ``data_manager`` for notebook export; never raises.

    A data manager without ``bind_client_session`` stays unbound, and exports
    from it show the identity as unknown. Only the owning client calls this;
    ``replace=True`` is reserved for restoring a saved session.
    """
    try:
        bind = getattr(data_manager, "bind_client_session", None)
        if callable(bind):
            bind(session_id, replace=replace)
    except Exception:
        logger.warning(
            "Notebook export: client session identity was not bound (binder_failed)"
        )
