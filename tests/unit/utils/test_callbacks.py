"""
Unit tests for callback handlers, specifically TokenTrackingCallback.

Tests token extraction, cost calculation, and usage tracking across multiple providers.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.outputs import Generation, LLMResult

from lobster.utils.callbacks import TokenInvocation, TokenTrackingCallback


@pytest.fixture
def pricing_config():
    """Standard pricing configuration for tests - matches official AWS Bedrock pricing."""
    return {
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": {
            "input_per_million": 1.00,  # Corrected from 0.80
            "output_per_million": 5.00,  # Corrected from 4.00
            "display_name": "Claude 4.5 Haiku",
        },
        "us.anthropic.claude-sonnet-4-20250514-v1:0": {
            "input_per_million": 3.00,
            "output_per_million": 15.00,
            "display_name": "Claude 4 Sonnet",
        },
        "claude-4-5-haiku": {
            "input_per_million": 1.00,  # Corrected from 0.80
            "output_per_million": 5.00,  # Corrected from 4.00
            "display_name": "Claude 4.5 Haiku",
        },
    }


@pytest.fixture
def token_tracker(pricing_config):
    """Create a TokenTrackingCallback instance for testing."""
    return TokenTrackingCallback(
        session_id="test_session", pricing_config=pricing_config
    )


class TestTokenExtraction:
    """Test token extraction from different LLMResult formats."""

    def test_extract_anthropic_format(self, token_tracker):
        """Test extraction from Anthropic Direct API format."""
        result = LLMResult(
            generations=[[Generation(text="test response")]],
            llm_output={
                "usage": {"input_tokens": 1000, "output_tokens": 500},
                "model_name": "claude-4-5-haiku",
            },
        )

        usage = token_tracker._extract_token_usage(result)

        assert usage is not None
        assert usage["input_tokens"] == 1000
        assert usage["output_tokens"] == 500
        assert usage["total_tokens"] == 1500

    def test_extract_bedrock_format(self, token_tracker):
        """Test extraction from AWS Bedrock format."""
        result = LLMResult(
            generations=[[Generation(text="test response")]],
            llm_output={
                "usage": {"input_tokens": 2000, "output_tokens": 800},
                "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            },
        )

        usage = token_tracker._extract_token_usage(result)

        assert usage is not None
        assert usage["input_tokens"] == 2000
        assert usage["output_tokens"] == 800
        assert usage["total_tokens"] == 2800

    def test_extract_standard_langchain_format(self, token_tracker):
        """Test extraction from standard LangChain format (prompt_tokens/completion_tokens)."""
        result = LLMResult(
            generations=[[Generation(text="test response")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 1500,
                    "completion_tokens": 600,
                    "total_tokens": 2100,
                },
                "model_name": "test-model",
            },
        )

        usage = token_tracker._extract_token_usage(result)

        assert usage is not None
        assert usage["input_tokens"] == 1500
        assert usage["output_tokens"] == 600
        assert usage["total_tokens"] == 2100

    def test_extract_from_response_metadata(self, token_tracker):
        """Test extraction from response_metadata (Bedrock alternative)."""
        # Mock an LLMResult with response_metadata using MagicMock
        result = MagicMock(spec=LLMResult)
        result.generations = (
            []
        )  # Empty generations, will check response_metadata instead
        result.llm_output = {}
        result.response_metadata = {
            "usage": {"input_tokens": 3000, "output_tokens": 1200}
        }

        usage = token_tracker._extract_token_usage(result)

        assert usage is not None
        assert usage["input_tokens"] == 3000
        assert usage["output_tokens"] == 1200
        assert usage["total_tokens"] == 4200

    def test_extract_no_token_data(self, token_tracker):
        """Test extraction when no token data is available."""
        result = LLMResult(
            generations=[[Generation(text="test response")]], llm_output={}
        )

        usage = token_tracker._extract_token_usage(result)

        assert usage is None


class TestModelNameExtraction:
    """Test model name extraction from different LLMResult formats."""

    def test_extract_model_name_from_llm_output(self, token_tracker):
        """Test extraction from llm_output['model_name']."""
        result = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={"model_name": "claude-4-5-haiku"},
        )

        model = token_tracker._extract_model_name(result)
        assert model == "claude-4-5-haiku"

    def test_extract_model_id_from_llm_output(self, token_tracker):
        """Test extraction from llm_output['model_id']."""
        result = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
        )

        model = token_tracker._extract_model_name(result)
        assert model == "us.anthropic.claude-sonnet-4-20250514-v1:0"

    def test_extract_model_from_response_metadata(self, token_tracker):
        """Test extraction from response_metadata."""
        # Mock an LLMResult with response_metadata using MagicMock
        result = MagicMock(spec=LLMResult)
        result.generations = (
            []
        )  # Empty generations, will check response_metadata instead
        result.llm_output = {}
        result.response_metadata = {"model_id": "bedrock-model"}

        model = token_tracker._extract_model_name(result)
        assert model == "bedrock-model"

    def test_extract_unknown_model(self, token_tracker):
        """Test extraction when model name is not found."""
        result = LLMResult(generations=[[Generation(text="test")]], llm_output={})

        model = token_tracker._extract_model_name(result)
        assert model == "unknown"


class TestCostCalculation:
    """Test cost calculation based on model pricing."""

    def test_calculate_cost_haiku(self, token_tracker):
        """Test cost calculation for Haiku model."""
        model = "claude-4-5-haiku"
        input_tokens = 1_000_000  # 1M input tokens
        output_tokens = 1_000_000  # 1M output tokens

        cost = token_tracker._calculate_cost(model, input_tokens, output_tokens)

        # Expected: (1M / 1M) * $1.00 + (1M / 1M) * $5.00 = $6.00
        assert cost == pytest.approx(6.00)

    def test_calculate_cost_sonnet(self, token_tracker):
        """Test cost calculation for Sonnet model."""
        model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        input_tokens = 500_000  # 0.5M input tokens
        output_tokens = 250_000  # 0.25M output tokens

        cost = token_tracker._calculate_cost(model, input_tokens, output_tokens)

        # Expected: (0.5M / 1M) * $3.00 + (0.25M / 1M) * $15.00 = $1.50 + $3.75 = $5.25
        assert cost == pytest.approx(5.25)

    def test_calculate_cost_unknown_model(self, token_tracker):
        """Test cost calculation for unknown model (should return 0.0)."""
        model = "unknown-model"
        input_tokens = 1000
        output_tokens = 500

        cost = token_tracker._calculate_cost(model, input_tokens, output_tokens)

        assert cost == 0.0

    def test_calculate_cost_zero_tokens(self, token_tracker):
        """Test cost calculation with zero tokens."""
        model = "claude-4-5-haiku"

        cost = token_tracker._calculate_cost(model, 0, 0)

        assert cost == 0.0


class TestTokenTracking:
    """Test end-to-end token tracking functionality."""

    def test_on_llm_start_sets_current_agent(self, token_tracker):
        """Test that on_llm_start sets current agent context."""
        token_tracker.on_llm_start(
            serialized={"name": "test_agent"},
            prompts=["test prompt"],
            name="test_agent",
        )

        assert token_tracker.current_agent == "test_agent"

    def test_on_tool_start_sets_current_tool(self, token_tracker):
        """Test that on_tool_start sets current tool context."""
        token_tracker.on_tool_start(
            serialized={"name": "test_tool"}, input_str="test input"
        )

        assert token_tracker.current_tool == "test_tool"

    def test_on_tool_end_clears_current_tool(self, token_tracker):
        """Test that on_tool_end clears current tool context."""
        token_tracker.current_tool = "test_tool"
        token_tracker.on_tool_end(output="test output")

        assert token_tracker.current_tool is None

    def test_on_llm_end_tracks_invocation(self, token_tracker):
        """Test that on_llm_end creates invocation record."""
        token_tracker.current_agent = "test_agent"
        token_tracker.current_tool = "test_tool"

        result = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 1000, "output_tokens": 500},
                "model_name": "claude-4-5-haiku",
            },
        )

        token_tracker.on_llm_end(result)

        assert len(token_tracker.invocations) == 1
        invocation = token_tracker.invocations[0]
        assert invocation.agent == "test_agent"
        assert invocation.tool == "test_tool"
        assert invocation.model == "claude-4-5-haiku"
        assert invocation.input_tokens == 1000
        assert invocation.output_tokens == 500
        assert invocation.total_tokens == 1500
        assert invocation.cost_usd > 0

    def test_on_llm_end_accumulates_totals(self, token_tracker):
        """Test that multiple LLM calls accumulate totals correctly."""
        token_tracker.current_agent = "agent1"

        result1 = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 1000, "output_tokens": 500},
                "model_name": "claude-4-5-haiku",
            },
        )
        token_tracker.on_llm_end(result1)

        result2 = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 2000, "output_tokens": 800},
                "model_name": "claude-4-5-haiku",
            },
        )
        token_tracker.on_llm_end(result2)

        assert token_tracker.total_input_tokens == 3000
        assert token_tracker.total_output_tokens == 1300
        assert token_tracker.total_tokens == 4300
        assert token_tracker.total_cost_usd > 0

    def test_on_llm_end_no_token_data_skips_tracking(self, token_tracker):
        """Test that LLM calls without token data are skipped silently."""
        result = LLMResult(
            generations=[[Generation(text="test")]], llm_output={}  # No usage data
        )

        token_tracker.on_llm_end(result)

        assert len(token_tracker.invocations) == 0
        assert token_tracker.total_tokens == 0


class TestPerAgentTracking:
    """Test per-agent aggregation functionality."""

    def test_single_agent_tracking(self, token_tracker):
        """Test tracking for a single agent."""
        token_tracker.current_agent = "agent1"

        result = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 1000, "output_tokens": 500},
                "model_name": "claude-4-5-haiku",
            },
        )
        token_tracker.on_llm_end(result)

        assert "agent1" in token_tracker.by_agent
        agent_stats = token_tracker.by_agent["agent1"]
        assert agent_stats["input_tokens"] == 1000
        assert agent_stats["output_tokens"] == 500
        assert agent_stats["total_tokens"] == 1500
        assert agent_stats["invocation_count"] == 1
        assert agent_stats["cost_usd"] > 0

    def test_multiple_agents_tracking(self, token_tracker):
        """Test tracking across multiple agents."""
        # Agent 1
        token_tracker.current_agent = "agent1"
        result1 = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 1000, "output_tokens": 500},
                "model_name": "claude-4-5-haiku",
            },
        )
        token_tracker.on_llm_end(result1)

        # Agent 2
        token_tracker.current_agent = "agent2"
        result2 = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 2000, "output_tokens": 800},
                "model_name": "claude-4-5-haiku",
            },
        )
        token_tracker.on_llm_end(result2)

        assert len(token_tracker.by_agent) == 2
        assert token_tracker.by_agent["agent1"]["input_tokens"] == 1000
        assert token_tracker.by_agent["agent2"]["input_tokens"] == 2000

    def test_multiple_invocations_same_agent(self, token_tracker):
        """Test multiple invocations from the same agent accumulate correctly."""
        token_tracker.current_agent = "agent1"

        result1 = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 1000, "output_tokens": 500},
                "model_name": "claude-4-5-haiku",
            },
        )
        token_tracker.on_llm_end(result1)

        result2 = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 1500, "output_tokens": 700},
                "model_name": "claude-4-5-haiku",
            },
        )
        token_tracker.on_llm_end(result2)

        agent_stats = token_tracker.by_agent["agent1"]
        assert agent_stats["input_tokens"] == 2500
        assert agent_stats["output_tokens"] == 1200
        assert agent_stats["total_tokens"] == 3700
        assert agent_stats["invocation_count"] == 2


class TestUsageSummary:
    """Test usage summary generation."""

    def test_get_usage_summary_structure(self, token_tracker):
        """Test that usage summary has correct structure."""
        summary = token_tracker.get_usage_summary()

        assert "session_id" in summary
        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "total_tokens" in summary
        assert "total_cost_usd" in summary
        assert "by_agent" in summary
        assert "invocations" in summary

    def test_get_usage_summary_with_data(self, token_tracker):
        """Test usage summary with actual tracking data."""
        token_tracker.current_agent = "agent1"

        result = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 1000, "output_tokens": 500},
                "model_name": "claude-4-5-haiku",
            },
        )
        token_tracker.on_llm_end(result)

        summary = token_tracker.get_usage_summary()

        assert summary["session_id"] == "test_session"
        assert summary["total_input_tokens"] == 1000
        assert summary["total_output_tokens"] == 500
        assert summary["total_tokens"] == 1500
        assert summary["total_cost_usd"] > 0
        assert "agent1" in summary["by_agent"]
        assert len(summary["invocations"]) == 1

    def test_get_latest_cost(self, token_tracker):
        """Test getting latest cost information."""
        token_tracker.current_agent = "agent1"

        result = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 1000, "output_tokens": 500},
                "model_name": "claude-4-5-haiku",
            },
        )
        token_tracker.on_llm_end(result)

        latest = token_tracker.get_latest_cost()

        assert "latest_cost_usd" in latest
        assert "session_total_usd" in latest
        assert "total_tokens" in latest
        assert latest["latest_cost_usd"] > 0
        assert latest["session_total_usd"] == latest["latest_cost_usd"]
        assert latest["total_tokens"] == 1500


class TestWorkspacePersistence:
    """Test saving token usage to workspace."""

    def test_save_to_workspace(self, token_tracker, tmp_path):
        """Test saving usage data to workspace file."""
        token_tracker.current_agent = "agent1"

        result = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 1000, "output_tokens": 500},
                "model_name": "claude-4-5-haiku",
            },
        )
        token_tracker.on_llm_end(result)

        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()

        token_tracker.save_to_workspace(workspace_path)

        usage_file = workspace_path / "token_usage.json"
        assert usage_file.exists()

        with open(usage_file, "r") as f:
            saved_data = json.load(f)

        assert saved_data["session_id"] == "test_session"
        assert saved_data["total_tokens"] == 1500
        assert "agent1" in saved_data["by_agent"]

    def test_save_to_workspace_empty_tracking(self, token_tracker, tmp_path):
        """Test saving with no tracking data."""
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()

        token_tracker.save_to_workspace(workspace_path)

        usage_file = workspace_path / "token_usage.json"
        assert usage_file.exists()

        with open(usage_file, "r") as f:
            saved_data = json.load(f)

        assert saved_data["total_tokens"] == 0
        assert len(saved_data["invocations"]) == 0


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_all_data(self, token_tracker):
        """Test that reset clears all tracking data."""
        token_tracker.current_agent = "agent1"

        result = LLMResult(
            generations=[[Generation(text="test")]],
            llm_output={
                "usage": {"input_tokens": 1000, "output_tokens": 500},
                "model_name": "claude-4-5-haiku",
            },
        )
        token_tracker.on_llm_end(result)

        # Verify data exists
        assert token_tracker.total_tokens > 0
        assert len(token_tracker.invocations) > 0
        assert len(token_tracker.by_agent) > 0

        # Reset
        token_tracker.reset()

        # Verify all data cleared
        assert token_tracker.total_input_tokens == 0
        assert token_tracker.total_output_tokens == 0
        assert token_tracker.total_tokens == 0
        assert token_tracker.total_cost_usd == 0.0
        assert len(token_tracker.invocations) == 0
        assert len(token_tracker.by_agent) == 0
        assert token_tracker.current_agent is None
        assert token_tracker.current_tool is None
