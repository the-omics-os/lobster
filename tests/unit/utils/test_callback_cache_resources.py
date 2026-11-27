"""Cold pricing-resource failure must not change literal legacy cache-free usage."""

import functools
import json
import logging
import sys
from unittest.mock import MagicMock

import pytest

import lobster.pricing as pricing
from lobster.utils.callbacks import TokenTrackingCallback
from tests.unit.utils.test_callback_cache_cost import MODEL, record, result

pytestmark = pytest.mark.no_auto_config


@pytest.fixture
def cold_pricing(monkeypatch):
    # Keep the original book/cache/module intact for restoration. This fresh
    # wrapper has no cached entries; it executes the real packaged loader.
    original_loader = pricing.default_price_book
    fresh_loader = functools.cache(original_loader.__wrapped__)
    monkeypatch.setattr(pricing, "default_price_book", fresh_loader)
    monkeypatch.delitem(sys.modules, "lobster.pricing.cache_cost", raising=False)
    monkeypatch.delattr(pricing, "cache_cost", raising=False)
    yield
    fresh_loader.cache_clear()


def manual_tracker():
    # No get_rates(): deliberately avoid warming resources during construction.
    return TokenTrackingCallback(
        "synthetic", {MODEL: {"input_per_million": 2.0, "output_per_million": 10.0}}
    )


@pytest.mark.parametrize("details", [None, {}])
def test_cold_absent_zero_cache_never_opens_pricing_resources(
    cold_pricing, monkeypatch, details
):
    resource = MagicMock(side_effect=AssertionError("No-cache pricing resource I/O"))
    monkeypatch.setattr(pricing.importlib.resources, "files", resource)
    response = result(read=0, write=0, ttl=False)
    if details is None:
        response.generations[0][0].message.usage_metadata.pop("input_token_details")
    callback = manual_tracker()
    summary = record(callback, response)
    assert callback.total_cost_usd == (20 * 2 + 5 * 10) / 1_000_000
    assert summary["total_tokens"] == 25
    assert len(callback.invocations) == 1
    assert not any(key.startswith("cache_cost") for key in summary["invocations"][0])
    resource.assert_not_called()


@pytest.mark.parametrize("fault", ["missing", "malformed", "utf8"])
def test_cold_positive_cache_resource_failure_keeps_exact_legacy_and_one_warning(
    cold_pricing, monkeypatch, caplog, fault
):
    class Document:
        def __init__(self, name):
            self.name = name

        def read_bytes(self):
            if fault == "missing":
                raise FileNotFoundError("synthetic resource missing")
            return b"\xff" if fault == "utf8" else b"{malformed"

        def read_text(self, **_):
            return "{malformed"

    root = MagicMock()
    root.joinpath.side_effect = Document
    resource = MagicMock(return_value=root)
    monkeypatch.setattr(pricing.importlib.resources, "files", resource)
    callback = manual_tracker()
    with caplog.at_level(logging.WARNING, logger="lobster.utils.callbacks"):
        summary = record(callback, result())
    assert callback.total_cost_usd == (20 * 2 + 5 * 10) / 1_000_000
    assert len(callback.invocations) == 1
    assert summary["invocations"][0]["cache_cost_status"] == "legacy_fallback"
    assert summary["invocations"][0]["cache_cost_reason"] == "rates_unknown"
    records = [
        line
        for line in caplog.records
        if line.message.startswith("Cache cost fallback")
    ]
    assert len(records) == 1 and len(caplog.records) == 1
    assert "synthetic" not in records[0].message and MODEL not in records[0].message
    assert records[0].exc_info is None
    resource.assert_called()


def test_cold_oversized_json_rate_preserves_known_legacy_cost(
    cold_pricing, monkeypatch, caplog
):
    # JSON parses integers exactly; real schema math.isfinite(10**400) overflows.
    prices = {
        "schema_version": pricing.SCHEMA_VERSION,
        "snapshot_id": "synthetic-rate-overflow",
        "fetched_at": "2026-10-28T00:00:00Z",
        "source": {"url": "https://example.invalid/prices", "sha256": "0" * 64},
        "providers": {
            "bedrock": {
                "modelsdev_provider": pricing.PROVIDER_MAP["bedrock"],
                "models": {MODEL: {"cost": {"input": 10**400, "output": 10}}},
            }
        },
    }
    overrides = {"schema_version": pricing.SCHEMA_VERSION, "overrides": []}

    class Document:
        def __init__(self, name):
            self.name = name

        def read_bytes(self):
            assert self.name == "overrides.json"
            return json.dumps(overrides).encode("utf-8")

        def read_text(self, **_):
            assert self.name == "prices.json"
            return json.dumps(prices)

    root = MagicMock()
    root.joinpath.side_effect = Document
    resource = MagicMock(return_value=root)
    monkeypatch.setattr(pricing.importlib.resources, "files", resource)
    check = MagicMock(wraps=pricing.schema._check_rates)
    monkeypatch.setattr(pricing.schema, "_check_rates", check)
    callback = manual_tracker()
    expected = callback._calculate_cost(MODEL, 20, 5)
    assert expected is not None and expected > 0
    with caplog.at_level(logging.WARNING, logger="lobster.utils.callbacks"):
        summary = record(callback, result())
    check.assert_called_once()  # Nonvacuous: actual rate parsing was reached.
    assert callback.total_cost_usd == expected
    assert len(callback.invocations) == 1
    assert callback.invocations[0].cost_usd == expected
    assert callback.unpriced_invocations == 0 and summary["cost_complete"]
    assert summary["invocations"][0]["cache_cost_status"] == "legacy_fallback"
    assert summary["invocations"][0]["cache_cost_reason"] == "rates_unknown"
    records = [
        line
        for line in caplog.records
        if line.message.startswith("Cache cost fallback")
    ]
    assert len(records) == 1 and len(caplog.records) == 1
    assert records[0].exc_info is None
    assert "synthetic" not in records[0].message and MODEL not in records[0].message
    assert str(10**400) not in records[0].message
    resource.assert_called_once()


@pytest.mark.parametrize("flag", [None, "false"])
@pytest.mark.parametrize("provider", ["amazon_bedrock", "anthropic"])
def test_cache_off_still_accounts_observed_usage(monkeypatch, flag, provider, caplog):
    if flag is None:
        monkeypatch.delenv("LOBSTER_BEDROCK_PROMPT_CACHE", raising=False)
    else:
        monkeypatch.setenv("LOBSTER_BEDROCK_PROMPT_CACHE", flag)
    rates = pricing.get_rates("bedrock", MODEL)
    callback = manual_tracker()
    with caplog.at_level(logging.DEBUG, logger="lobster.utils.callbacks"):
        summary = record(callback, result(), provider)
    if provider == "amazon_bedrock":
        expected = float(
            (
                10 * rates.input
                + 5 * rates.output
                + 7 * rates.cache_read
                + 3 * rates.cache_write
            )
            / 1_000_000
        )
        assert callback.total_cost_usd == expected
        assert summary["invocations"][0]["cache_cost_status"] == "catalog"
        assert caplog.records == []
    else:
        assert callback.total_cost_usd == (20 * 2 + 5 * 10) / 1_000_000
        assert summary["invocations"][0]["cache_cost_status"] == "legacy_fallback"
        assert (
            summary["invocations"][0]["cache_cost_reason"] == "pricing_identity_unknown"
        )
        assert len(caplog.records) == 1
