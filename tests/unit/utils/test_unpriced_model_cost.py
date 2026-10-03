"""An unpriced model must report cost as unknown, never as zero.

A missing pricing entry is distinct from a free model. Preserve token counts, mark the
session cost incomplete, and avoid showing a misleading cost summary.
"""

from types import SimpleNamespace

import pytest

from lobster.config.settings import MODEL_PRICING
from lobster.utils.callbacks import TokenTrackingCallback

PRICED_HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
UNPRICED = "us.anthropic.claude-nonexistent-9-v1:0"


@pytest.fixture
def tracker():
    return TokenTrackingCallback(session_id="t", pricing_config=MODEL_PRICING)


def feed(tracker, model, input_tokens, output_tokens):
    """Drive a full invocation through the real `on_llm_end` path."""
    tracker.on_llm_end(
        SimpleNamespace(
            llm_output={
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                "model_name": model,
            },
            generations=[[]],
        )
    )


class TestUnpricedReturnsNone:
    def test_unpriced_model_returns_none_not_zero(self, tracker):
        assert tracker._calculate_cost(UNPRICED, 70_000, 5_000) is None

    def test_priced_model_returns_a_real_number(self, tracker):
        cost = tracker._calculate_cost(PRICED_HAIKU, 70_000, 5_044)
        assert cost is not None
        assert cost > 0

    def test_warns_once_per_model_not_once_per_call(self, tracker, caplog):
        """A per-invocation warning would flood the log for a whole session."""
        with caplog.at_level("WARNING"):
            for _ in range(4):
                tracker._calculate_cost(UNPRICED, 1_000, 1_000)
        assert sum("No pricing entry" in r.message for r in caplog.records) == 1

    def test_warning_names_the_offending_model(self, tracker, caplog):
        with caplog.at_level("WARNING"):
            tracker._calculate_cost(UNPRICED, 1_000, 1_000)
        assert UNPRICED in caplog.text


class TestLocalModelsAreStillFree:
    """Absence from pricing_config means 'free' for local models, 'unknown' otherwise.

    `get_all_models_with_pricing()` skips the ollama provider on purpose, so a blanket
    `None` would have mislabelled genuinely-free local runs as unpriced.
    """

    @pytest.mark.parametrize("model", ["ollama/llama3.3", "mistral-nemo", "qwen2.5-7b"])
    def test_local_model_costs_zero_not_none(self, tracker, model):
        assert tracker._calculate_cost(model, 5_000, 5_000) == 0.0

    def test_local_session_still_reads_as_local(self, tracker):
        feed(tracker, "ollama/llama3.3", 40_000, 5_000)
        assert tracker.get_usage_summary()["cost_complete"] is True
        assert "(local)" in tracker.get_minimal_summary()


class TestSessionTotalsStayHonest:
    def test_the_reported_scenario_no_longer_reports_zero(self, tracker):
        """A priced invocation contributes a nonzero cost to the session total."""
        feed(tracker, PRICED_HAIKU, 70_000, 5_044)
        summary = tracker.get_usage_summary()
        assert summary["total_cost_usd"] > 0
        assert summary["cost_complete"] is True

    def test_unpriced_invocation_does_not_crash_accumulation(self, tracker):
        """`total_cost_usd += None` raises TypeError; the guard must be present."""
        feed(tracker, UNPRICED, 70_000, 5_000)
        assert tracker.total_cost_usd == 0.0
        assert tracker.unpriced_invocations == 1

    def test_tokens_are_counted_even_when_cost_is_not(self, tracker):
        feed(tracker, UNPRICED, 70_000, 5_044)
        assert tracker.get_usage_summary()["total_tokens"] == 75_044

    def test_mixed_session_is_flagged_incomplete(self, tracker):
        feed(tracker, PRICED_HAIKU, 70_000, 5_044)
        feed(tracker, UNPRICED, 60_000, 4_000)
        summary = tracker.get_usage_summary()

        assert summary["cost_complete"] is False
        assert summary["unpriced_invocations"] == 1
        assert summary["unpriced_models"] == [UNPRICED]
        # The total remains a real sum over the priced invocations -- a lower bound.
        assert summary["total_cost_usd"] > 0

    def test_summary_never_calls_round_on_none(self, tracker):
        """`round(None)` raises TypeError, so serialization must handle it explicitly."""
        feed(tracker, UNPRICED, 1_000, 1_000)
        costs = [i["cost_usd"] for i in tracker.get_usage_summary()["invocations"]]
        assert costs == [None]

    def test_latest_cost_reports_completeness(self, tracker):
        feed(tracker, UNPRICED, 1_000, 1_000)
        latest = tracker.get_latest_cost()
        assert latest["latest_cost_usd"] is None
        assert latest["cost_complete"] is False

    def test_reset_clears_unpriced_state(self, tracker):
        feed(tracker, UNPRICED, 1_000, 1_000)
        tracker.reset()
        assert tracker.unpriced_invocations == 0
        assert tracker.get_usage_summary()["cost_complete"] is True

    def test_per_agent_breakdown_tracks_unpriced(self, tracker):
        feed(tracker, UNPRICED, 1_000, 1_000)
        by_agent = tracker.get_usage_summary()["by_agent"]
        assert any(s["unpriced_invocations"] == 1 for s in by_agent.values())


class TestHumanFacingSummaries:
    def test_unpriced_session_is_not_labelled_local(self, tracker):
        """An unpriced paid invocation must not be presented as local or free.

        `get_minimal_summary` branched on `total_cost_usd > 0`, so an unpriced paid run
        fell through to the "(local)" branch.
        """
        feed(tracker, UNPRICED, 70_000, 5_044)
        summary = tracker.get_minimal_summary()

        assert "(local)" not in summary
        assert "unknown" in summary.lower()

    def test_mixed_session_marks_the_total_as_a_floor(self, tracker):
        feed(tracker, PRICED_HAIKU, 70_000, 5_044)
        feed(tracker, UNPRICED, 60_000, 4_000)
        assert ">" in tracker.get_minimal_summary()

    def test_fully_priced_session_shows_a_plain_cost(self, tracker):
        feed(tracker, PRICED_HAIKU, 70_000, 5_044)
        summary = tracker.get_minimal_summary()
        assert summary.startswith("Session cost: $")
        assert ">" not in summary

    def test_verbose_summary_flags_unpriced(self, tracker):
        feed(tracker, UNPRICED, 1_000, 1_000)
        assert "unpriced" in tracker.get_verbose_summary()


class TestBedrockInferenceProfilesArePriced:
    """Bedrock requires 'us.'/'global.'-prefixed profile IDs for on-demand invocation.

    `list_foundation_models` still lists the BARE IDs, so catalog presence does not imply
    invocability -- only `list_inference_profiles` does. A pricing table keyed only on bare
    IDs therefore miss invocations, which is why matching needs the profile ID.
    """

    @pytest.mark.parametrize(
        "model_id",
        [
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us.anthropic.claude-opus-4-6-v1",
            "global.anthropic.claude-opus-4-6-v1",
        ],
    )
    def test_profile_id_has_pricing(self, model_id):
        assert model_id in MODEL_PRICING, (
            f"{model_id} is invocable on Bedrock but has no pricing entry, so real "
            f"calls would be recorded as unpriced."
        )

    def test_us_and_global_variants_are_priced_identically(self):
        us = MODEL_PRICING["us.anthropic.claude-haiku-4-5-20251001-v1:0"]
        gl = MODEL_PRICING["global.anthropic.claude-haiku-4-5-20251001-v1:0"]
        assert us["input_per_million"] == gl["input_per_million"]
        assert us["output_per_million"] == gl["output_per_million"]

    def test_haiku_rates_match_the_published_snapshot(self):
        """Provider list rates for the current catalog."""
        pricing = MODEL_PRICING["us.anthropic.claude-haiku-4-5-20251001-v1:0"]
        assert pricing["input_per_million"] == 1.0
        assert pricing["output_per_million"] == 5.0
