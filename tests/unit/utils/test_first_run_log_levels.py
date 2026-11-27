"""Optional services that are missing on a normal install log at DEBUG, not WARNING.

A first `lobster chat` printed "Redis unavailable" (no Redis server) and
"Docling not installed" before the first answer. Neither is a problem for a
user: rate limiting is for servers, and a feature that needs Docling says so
with the install command when it is used.
"""

import logging
import sys
from unittest import mock

import pytest

pytestmark = pytest.mark.unit


def _records(caplog, name):
    return [r for r in caplog.records if r.name == name]


def test_no_redis_server_is_logged_at_debug(caplog):
    from lobster.tools import rate_limiter

    if not rate_limiter.REDIS_AVAILABLE:
        pytest.skip("the redis package is not installed")
    rate_limiter.reset_redis_pool()
    client = mock.Mock()
    client.ping.side_effect = rate_limiter.ConnectionError("refused")
    try:
        with (
            caplog.at_level(logging.DEBUG, logger=rate_limiter.__name__),
            mock.patch.object(rate_limiter.redis, "Redis", return_value=client),
        ):
            assert rate_limiter._create_connection_pool() is None
    finally:
        rate_limiter.reset_redis_pool()

    records = _records(caplog, rate_limiter.__name__)
    assert [r.levelno for r in records if "Redis unavailable" in r.getMessage()] == [
        logging.DEBUG
    ]
    assert not [r for r in records if r.levelno >= logging.WARNING]


def test_docling_not_installed_is_logged_at_debug(caplog):
    from lobster.services.data_access import docling_service

    service = object.__new__(docling_service.DoclingService)
    # None in sys.modules makes `import docling...` raise ImportError
    with (
        caplog.at_level(logging.DEBUG, logger=docling_service.__name__),
        mock.patch.dict(
            sys.modules,
            {
                "docling": None,
                "docling.datamodel": None,
                "docling.datamodel.base_models": None,
            },
        ),
    ):
        service._initialize_docling()

    assert service.converter is None
    records = _records(caplog, docling_service.__name__)
    assert [
        r.levelno for r in records if "Docling not installed" in r.getMessage()
    ] == [logging.DEBUG]
    assert not [r for r in records if r.levelno >= logging.WARNING]
