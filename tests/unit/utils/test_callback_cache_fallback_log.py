"""A cache-cost fallback warns once per reason per session, then logs at DEBUG."""

import logging
from uuid import uuid4

import pytest

from tests.unit.utils.test_callback_cache_cost import MODEL, result, tracker

pytestmark = pytest.mark.no_auto_config


def _call(callback, response, provider="amazon_bedrock"):
    run_id = uuid4()
    callback.on_chat_model_start(
        {},
        [[]],
        run_id=run_id,
        metadata=(
            {}
            if provider is None
            else {"ls_provider": provider, "ls_model_name": MODEL}
        ),
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


@pytest.mark.parametrize("provider", [None, "anthropic", "openai", "omics-os"])
def test_a_provider_without_a_cache_price_identity_logs_only_at_debug(caplog, provider):
    # Every cached turn on such a provider takes this fallback: expected, not a warning
    from tests.unit.utils.test_callback_cache_cost import legacy_amount

    callback = tracker()
    with caplog.at_level(logging.DEBUG, logger="lobster.utils.callbacks"):
        _call(callback, result(), provider)
        _call(callback, result(), provider)
    summary = callback.get_usage_summary()
    records = _fallback_records(caplog)
    assert [r.levelno for r in records] == [logging.DEBUG, logging.DEBUG]
    assert "reason=pricing_identity_unknown" in records[0].message
    # Accounting is unchanged by the quieter log.
    assert [i["cache_cost_reason"] for i in summary["invocations"]] == [
        "pricing_identity_unknown"
    ] * 2
    assert callback.total_cost_usd == pytest.approx(2 * legacy_amount())
