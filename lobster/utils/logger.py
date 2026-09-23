"""
Logging configuration for the application.

Lobster attaches at most **one** handler, on the ``lobster`` namespace logger,
and never disables propagation. That lets whichever process hosts the engine own
its log output — the CLI's ConsoleManager, a uvicorn/FastAPI worker, pytest —
while a bare script still gets readable output with zero setup.

Two consequences worth knowing:

1. Module loggers are left at ``NOTSET`` so they inherit from the ``lobster``
   namespace logger. One ``LOBSTER_LOG_LEVEL`` (or one ``setLevel`` on
   ``logging.getLogger("lobster")``) therefore controls the whole engine,
   including the ~131 modules that call ``logging.getLogger(__name__)``
   directly instead of going through :func:`get_logger`.
2. Records propagate to the root logger, so a host that calls
   ``logging.basicConfig()`` before importing the engine captures them.
"""

import logging
import os
import sys
import threading

_NAMESPACE = "lobster"

# Unset LOBSTER_LOG_LEVEL keeps the historical behaviour (engine modules emit
# INFO). Setting it explicitly is what makes it a real knob — before this module
# was consolidated, get_logger hardcoded INFO per module and the variable only
# ever moved the root/Rich level, so it could not quieten the engine at all.
_DEFAULT_LEVEL = logging.INFO

_LEVEL_NAMES = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_configure_lock = threading.Lock()
_namespace_configured = False


class _FallbackHandler(logging.StreamHandler):
    """Stream handler that stands down once the host configures root logging.

    The stand-down cannot be decided at import time. ``lobster/cli.py`` imports
    engine modules — which call :func:`get_logger` — at lines 70-290, but only
    installs the RichHandler on root at ``cli.py:293``. An import-time check
    would see a bare root, attach this handler, and then double-print every
    record once Rich arrives. Deciding per-record is order-independent, so it
    also covers any other host that configures root after importing the engine.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Root's own handlers, not hasHandlers(): the latter walks up from this
        # logger and would count this handler itself.
        if logging.getLogger().handlers:
            return
        super().emit(record)


def _resolve_level() -> int:
    """Resolve the engine-wide log level from ``LOBSTER_LOG_LEVEL``."""
    return _LEVEL_NAMES.get(
        os.environ.get("LOBSTER_LOG_LEVEL", "").strip().upper(), _DEFAULT_LEVEL
    )


def _configure_namespace() -> logging.Logger:
    """Configure the ``lobster`` namespace logger exactly once per process."""
    global _namespace_configured

    namespace = logging.getLogger(_NAMESPACE)
    if _namespace_configured:
        return namespace

    with _configure_lock:
        if _namespace_configured:
            return namespace

        namespace.setLevel(_resolve_level())

        if not any(isinstance(h, _FallbackHandler) for h in namespace.handlers):
            handler = _FallbackHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)s - [%(name)s] - %(message)s"
                )
            )
            namespace.addHandler(handler)

        _namespace_configured = True

    return namespace


def setup_logger(name: str, level: int = None) -> logging.Logger:
    """
    Get a logger that participates in the engine's shared logging setup.

    Args:
        name: Name of the logger. Names under the ``lobster`` namespace inherit
            their level from it; anything else gets the resolved level set
            directly, since it has no lobster ancestor to inherit from.
        level: Optional explicit level for this logger only. Leave unset so the
            logger inherits — that is what keeps one knob in control.

    Returns:
        logging.Logger: Logger instance.
    """
    _configure_namespace()

    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)
    elif name != _NAMESPACE and not name.startswith(_NAMESPACE + "."):
        logger.setLevel(_resolve_level())

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Name of the logger

    Returns:
        logging.Logger: Logger instance
    """
    return setup_logger(name)
