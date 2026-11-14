"""Bedrock ``us.`` (geo / in-region) profiles carry a 10% premium over ``global.``.

AWS publishes the premium on its Bedrock price page (Global vs Geo and In-region
rows) and Anthropic's pricing doc states it applies to Opus 4.7, Haiku 4.5 and later
models. The static catalog used to price the ``us.`` Claude 4.x profiles at parity,
so a run on them showed a cost about 9% below the real one.
"""

import pytest

from lobster.config.providers.bedrock_provider import BedrockProvider
from lobster.pricing import get_rates

PRICES = {
    m.name: (m.input_cost_per_million, m.output_cost_per_million)
    for m in BedrockProvider.KNOWN_MODELS
}

# (profile id without the us./global. prefix, global rate, us rate)
CLAUDE_4X = [
    ("anthropic.claude-opus-4-5-20251101-v1:0", (5.0, 25.0), (5.5, 27.5)),
    ("anthropic.claude-opus-4-6-v1", (5.0, 25.0), (5.5, 27.5)),
    ("anthropic.claude-opus-4-7", (5.0, 25.0), (5.5, 27.5)),
    ("anthropic.claude-opus-4-8", (5.0, 25.0), (5.5, 27.5)),
    ("anthropic.claude-opus-5", (5.0, 25.0), (5.5, 27.5)),
    ("anthropic.claude-sonnet-4-5-20250929-v1:0", (3.0, 15.0), (3.3, 16.5)),
    ("anthropic.claude-sonnet-4-6", (3.0, 15.0), (3.3, 16.5)),
    ("anthropic.claude-haiku-4-5-20251001-v1:0", (1.0, 5.0), (1.1, 5.5)),
]


@pytest.mark.parametrize("suffix,global_rate,us_rate", CLAUDE_4X)
def test_us_profile_carries_the_regional_premium_literally(
    suffix, global_rate, us_rate
):
    assert PRICES["global." + suffix] == global_rate
    assert PRICES["us." + suffix] == us_rate


def test_every_anthropic_us_profile_is_the_global_rate_plus_ten_percent():
    pairs = [
        (name, "global." + name[len("us.") :])
        for name in PRICES
        if name.startswith("us.anthropic.") and "global." + name[3:] in PRICES
    ]
    assert len(pairs) >= len(CLAUDE_4X) + 4  # the 4.x set plus the 5.5 family
    divergent = {
        us: {"us": PRICES[us], "expected": tuple(round(v * 1.1, 4) for v in PRICES[g])}
        for us, g in pairs
        if PRICES[us] != tuple(round(v * 1.1, 4) for v in PRICES[g])
    }
    assert not divergent, divergent


@pytest.mark.parametrize("suffix,global_rate,us_rate", CLAUDE_4X)
def test_catalog_agrees_with_the_packaged_list_prices(suffix, global_rate, us_rate):
    """Cache rates come from the packaged table; input/output must not contradict it."""
    for prefix in ("global.", "us."):
        rates = get_rates("bedrock", prefix + suffix)
        assert rates is not None
        assert PRICES[prefix + suffix] == (float(rates.input), float(rates.output))


# Sonnet 4.5 is left out: the Omics-OS Cloud provider lists the same id at the price a
# Cloud user is charged (3/15) and wins in the merged MODEL_PRICING, shadowing the
# Bedrock entry. That key collision is a separate, commercial question.
SHADOWED = "anthropic.claude-sonnet-4-5-20250929-v1:0"


@pytest.mark.parametrize(
    "suffix,global_rate,us_rate", [c for c in CLAUDE_4X if c[0] != SHADOWED]
)
def test_the_merged_cost_table_carries_the_premium(suffix, global_rate, us_rate):
    from lobster.config.settings import MODEL_PRICING

    entry = MODEL_PRICING["us." + suffix]
    assert (entry["input_per_million"], entry["output_per_million"]) == us_rate
