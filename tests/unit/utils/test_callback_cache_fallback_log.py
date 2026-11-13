"""A cache-cost fallback warns once per reason per session, then logs at DEBUG."""

import logging
from uuid import uuid4

import pytest

from tests.unit.utils.test_callback_cache_cost import MODEL, result, tracker

pytestmark = pytest.mark.no_auto_config


def _call(callback, response):
    run_id = uuid4()
    callback.on_chat_model_start(
        {},
        [[]],
        run_id=run_id,
        metadata={"ls_provider": "amazon_bedrock", "ls_model_name": MODEL},
    )
    callback.on_llm_end(response, run_id=run_id)


def _fallback_records(caplog):
    return [r for r in caplog.records if r.message.startswith("Cache cost fallback")]


def test_repeated_fallback_warns_once_then_debug(caplog):
    callback = tracker()
    with caplog.at_level(logging.DEBUG, logger="lobster.utils.callbacks"):
        for _ in range(3):
            _call(callback, result(ttl=False))
    records = _fallback_records(caplog)
    assert [r.levelno for r in records] == [
        logging.WARNING,
        logging.DEBUG,
        logging.DEBUG,
    ]
    assert "occurrence=3" in records[-1].message
    assert MODEL not in records[0].message
    # Accounting is per invocation, unchanged by the quieter log.
    summary = callback.get_usage_summary()
    assert [i["cache_cost_status"] for i in summary["invocations"]] == [
        "legacy_fallback"
    ] * 3


def test_each_reason_warns_once_and_reset_rearms(caplog):
    callback = tracker()
    with caplog.at_level(logging.WARNING, logger="lobster.utils.callbacks"):
        _call(callback, result(ttl=False))
        _call(callback, result(ttl=False))
        callback.reset()
        _call(callback, result(ttl=False))
    assert len(_fallback_records(caplog)) == 2
