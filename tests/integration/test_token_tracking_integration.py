"""
Integration tests for token tracking functionality.

These tests verify that token tracking works correctly in real agent execution
scenarios with actual LLM calls.

Tests marked with @pytest.mark.real_api require:
- AWS_BEDROCK_ACCESS_KEY or ANTHROPIC_API_KEY
- Will consume actual API tokens and incur costs
"""

import tempfile
from pathlib import Path

import pytest

from lobster.core.client import AgentClient


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory(prefix="lobster_token_test_") as temp_dir:
        yield Path(temp_dir)


@pytest.mark.integration
@pytest.mark.real_api
class TestTokenTrackingIntegration:
    """Integration tests for token tracking with real agent execution."""

    def test_token_tracking_simple_query(self, temp_workspace):
        """Test that token tracking works with a simple query."""
        # Create client (this will automatically set up token tracking)
        client = AgentClient(workspace_path=temp_workspace)

        # Run a simple query that should invoke the supervisor
        query = "Hello, please introduce yourself briefly."
        result = client.query(query)

        assert result["success"] is True
        assert "response" in result

        # Verify token tracking data is present
        assert "token_usage" in result
        token_info = result["token_usage"]

        assert "latest_cost_usd" in token_info
        assert "session_total_usd" in token_info
        assert "total_tokens" in token_info

        # Verify tokens were actually tracked
        assert token_info["total_tokens"] > 0
        assert token_info["session_total_usd"] > 0
        assert token_info["latest_cost_usd"] > 0

        # Verify comprehensive usage summary
        usage_summary = client.get_token_usage()
        assert usage_summary["session_id"] == client.session_id
        assert usage_summary["total_input_tokens"] > 0
        assert usage_summary["total_output_tokens"] > 0
        assert usage_summary["total_tokens"] > 0
        assert usage_summary["total_cost_usd"] > 0

        # Verify invocation log exists
        assert "invocations" in usage_summary
        assert len(usage_summary["invocations"]) > 0

        first_invocation = usage_summary["invocations"][0]
        assert "timestamp" in first_invocation
        assert "agent" in first_invocation
        assert "model" in first_invocation
        assert first_invocation["input_tokens"] > 0
        assert first_invocation["output_tokens"] > 0
        assert first_invocation["cost_usd"] > 0

    def test_token_tracking_multiple_queries(self, temp_workspace):
        """Test that token tracking accumulates across multiple queries."""
        client = AgentClient(workspace_path=temp_workspace)

        # Run first query
        result1 = client.query("What can you help me with?")
        assert result1["success"] is True

        usage1 = client.get_token_usage()
        total_tokens_1 = usage1["total_tokens"]
        total_cost_1 = usage1["total_cost_usd"]
        invocation_count_1 = len(usage1["invocations"])

        assert total_tokens_1 > 0
        assert total_cost_1 > 0

        # Run second query
        result2 = client.query("Thank you.")
        assert result2["success"] is True

        usage2 = client.get_token_usage()
        total_tokens_2 = usage2["total_tokens"]
        total_cost_2 = usage2["total_cost_usd"]
        invocation_count_2 = len(usage2["invocations"])

        # Verify accumulation
        assert total_tokens_2 > total_tokens_1
        assert total_cost_2 > total_cost_1
        assert invocation_count_2 > invocation_count_1

        # Verify per-agent breakdown exists
        assert "by_agent" in usage2
        assert len(usage2["by_agent"]) > 0

        # Verify session total matches sum of invocations
        invocation_cost_sum = sum(
            inv["cost_usd"] for inv in usage2["invocations"]
        )
        assert abs(invocation_cost_sum - total_cost_2) < 0.0001  # Allow small floating point difference

    def test_token_tracking_per_agent_breakdown(self, temp_workspace):
        """Test that per-agent token tracking works correctly."""
        client = AgentClient(workspace_path=temp_workspace)

        # Run query that will involve supervisor
        result = client.query("Hello, I need help with data analysis.")
        assert result["success"] is True

        usage = client.get_token_usage()

        # Verify per-agent data structure
        assert "by_agent" in usage
        by_agent = usage["by_agent"]

        # At minimum, supervisor should be present
        agent_names = list(by_agent.keys())
        assert len(agent_names) > 0

        # Verify each agent has correct fields
        for agent_name, agent_stats in by_agent.items():
            assert "input_tokens" in agent_stats
            assert "output_tokens" in agent_stats
            assert "total_tokens" in agent_stats
            assert "cost_usd" in agent_stats
            assert "invocation_count" in agent_stats

            # Verify values are reasonable
            assert agent_stats["input_tokens"] >= 0
            assert agent_stats["output_tokens"] >= 0
            assert agent_stats["total_tokens"] >= 0
            assert agent_stats["cost_usd"] >= 0
            assert agent_stats["invocation_count"] > 0

            # Verify tokens add up
            assert agent_stats["total_tokens"] == (
                agent_stats["input_tokens"] + agent_stats["output_tokens"]
            )

    def test_token_tracking_workspace_persistence(self, temp_workspace):
        """Test that token usage is saved to workspace on export."""
        client = AgentClient(workspace_path=temp_workspace)

        # Run query
        result = client.query("Hello")
        assert result["success"] is True

        # Export session (should save token usage)
        export_path = client.export_session()
        assert export_path.exists()

        # Verify token_usage.json was created
        token_file = client.data_manager.workspace_path / "token_usage.json"
        assert token_file.exists()

        # Verify content
        import json
        with open(token_file, "r") as f:
            saved_usage = json.load(f)

        assert "session_id" in saved_usage
        assert saved_usage["session_id"] == client.session_id
        assert "total_tokens" in saved_usage
        assert saved_usage["total_tokens"] > 0
        assert "total_cost_usd" in saved_usage
        assert saved_usage["total_cost_usd"] > 0
        assert "by_agent" in saved_usage
        assert "invocations" in saved_usage

    def test_token_tracking_with_error_recovery(self, temp_workspace):
        """Test that token tracking continues working after errors."""
        client = AgentClient(workspace_path=temp_workspace)

        # Run successful query
        result1 = client.query("Hello")
        assert result1["success"] is True

        usage1 = client.get_token_usage()
        tokens_before = usage1["total_tokens"]

        # Run query that might fail or have issues
        # (Even if it fails, token tracking should work for the attempted invocations)
        result2 = client.query("Test query 2")

        usage2 = client.get_token_usage()
        tokens_after = usage2["total_tokens"]

        # Tokens should have increased (even if query had issues)
        # At minimum, the second query attempt should consume some tokens
        assert tokens_after >= tokens_before

    def test_token_tracking_cost_calculation_accuracy(self, temp_workspace):
        """Test that cost calculations are accurate based on model pricing."""
        client = AgentClient(workspace_path=temp_workspace)

        # Run query
        result = client.query("Brief test query")
        assert result["success"] is True

        usage = client.get_token_usage()

        # Verify cost calculation makes sense
        # For Claude models:
        # - Haiku: ~$0.80/M input, ~$4.00/M output
        # - Sonnet: ~$3.00/M input, ~$15.00/M output

        total_tokens = usage["total_tokens"]
        total_cost = usage["total_cost_usd"]

        # Rough sanity check: cost should be proportional to tokens
        # For 1000 tokens on cheapest model (Haiku):
        # Input: 1000/1M * $1.00 = $0.001
        # Output: 1000/1M * $5.00 = $0.005
        # Average: ~$0.003 per 1000 tokens

        # So for any number of tokens, cost should be in reasonable range
        if total_tokens > 0:
            cost_per_1k_tokens = (total_cost / total_tokens) * 1000
            # Should be between $0.001 (cheapest input) and $0.0225 (most expensive output - long context)
            assert 0.0005 < cost_per_1k_tokens < 0.025, \
                f"Cost per 1k tokens ({cost_per_1k_tokens}) seems unreasonable"

    def test_token_tracking_model_identification(self, temp_workspace):
        """Test that models are correctly identified in token tracking."""
        client = AgentClient(workspace_path=temp_workspace)

        # Run query
        result = client.query("Test")
        assert result["success"] is True

        usage = client.get_token_usage()

        # Verify model information is captured
        assert "invocations" in usage
        assert len(usage["invocations"]) > 0

        for invocation in usage["invocations"]:
            assert "model" in invocation
            model = invocation["model"]

            # Model should not be "unknown"
            assert model != "unknown", "Model should be identified, not unknown"

            # Model should be one of our configured models
            # (either Claude variants or at least a recognized format)
            assert (
                "claude" in model.lower() or
                "anthropic" in model.lower() or
                "bedrock" in model.lower() or
                "us.anthropic" in model
            ), f"Unexpected model identifier: {model}"


@pytest.mark.integration
class TestTokenTrackingMocked:
    """Integration tests with mocked graph for faster testing."""

    def test_token_tracking_disabled_on_mock(self, temp_workspace):
        """Test that token tracking gracefully handles mocked scenarios."""
        from unittest.mock import Mock, patch

        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Mock response")]}}
        ]

        with patch(
            "lobster.core.client.create_bioinformatics_graph", return_value=mock_graph
        ):
            client = AgentClient(workspace_path=temp_workspace)

            # Token tracker should exist
            assert hasattr(client, "token_tracker")
            assert client.token_tracker is not None

            # Run query with mocked graph
            result = client.query("Test query")
            assert result["success"] is True

            # Token tracking should exist but may have zero values
            usage = client.get_token_usage()
            assert "total_tokens" in usage
            # In mocked scenario, no actual LLM calls, so tokens may be 0
            assert usage["total_tokens"] >= 0
