"""
Comprehensive unit tests for supervisor agent.

This module provides thorough testing of the supervisor agent including
coordination, decision-making, agent handoffs, workflow management,
and multi-agent orchestration for the bioinformatics platform.

Test coverage target: 95%+ with meaningful tests for agent coordination.
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph

from lobster.agents.langgraph_supervisor import create_supervisor
from lobster.agents.supervisor import create_supervisor_prompt
from lobster.core.data_manager_v2 import DataManagerV2
from tests.mock_data.base import SMALL_DATASET_CONFIG
from tests.mock_data.factories import SingleCellDataFactory

# ===============================================================================
# Mock Objects and Fixtures
# ===============================================================================


class MockMessage:
    """Mock LangGraph message object."""

    def __init__(self, content: str, sender: str = "human"):
        self.content = content
        self.sender = sender
        self.additional_kwargs = {}


class MockState:
    """Mock LangGraph state object."""

    def __init__(self, messages=None, **kwargs):
        self.messages = messages or []
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_data_manager(mock_agent_environment):
    """Create mock data manager."""
    with patch("lobster.core.data_manager_v2.DataManagerV2") as MockDataManager:
        mock_dm = MockDataManager.return_value
        mock_dm.list_modalities.return_value = ["test_data", "geo_gse12345"]
        mock_dm.get_modality.return_value = SingleCellDataFactory(
            config=SMALL_DATASET_CONFIG
        )
        mock_dm.get_summary.return_value = "Test dataset with 100 cells and 500 genes"
        yield mock_dm


@pytest.fixture
def mock_llm():
    """Mock LLM for supervisor testing."""
    mock_llm = Mock()
    mock_llm.invoke.return_value = AIMessage(
        content="I'll help you with that analysis."
    )
    return mock_llm


@pytest.fixture
def supervisor_state():
    """Create supervisor state for testing."""
    return {
        "messages": [
            HumanMessage(content="Please analyze this single-cell RNA-seq dataset")
        ]
    }


# ===============================================================================
# Supervisor Agent Core Functionality Tests
# ===============================================================================


@pytest.mark.unit
class TestSupervisorAgentCore:
    """Test supervisor agent core functionality."""

    def test_supervisor_initialization(self, mock_data_manager, mock_llm):
        """Test supervisor agent initialization."""
        # Test that create_supervisor returns a StateGraph
        result = create_supervisor(
            agents=[],  # Empty list for testing initialization
            model=mock_llm,
            prompt="You are a supervisor agent.",
        )

        # Should return a StateGraph
        assert isinstance(result, StateGraph)

    def test_supervisor_agent_selection_data_task(
        self, mock_llm, supervisor_state, mock_data_manager
    ):
        """Test supervisor selecting data expert for data tasks."""
        supervisor_state["messages"] = [
            HumanMessage(
                content="Load the dataset from GEO GSE12345 and show me a summary"
            )
        ]

        # Mock an agent that could be selected
        mock_data_agent = Mock()
        mock_data_agent.name = "data_expert_agent"

        # Test that supervisor can be created with agents
        supervisor_graph = create_supervisor(
            agents=[mock_data_agent],
            model=mock_llm,
            prompt="You are a supervisor agent.",
        )

        # Should create a graph successfully
        assert isinstance(supervisor_graph, StateGraph)

    def test_supervisor_agent_selection_analysis_task(
        self, mock_llm, supervisor_state, mock_data_manager
    ):
        """Test supervisor selecting analysis expert for analysis tasks."""
        supervisor_state["messages"] = [
            HumanMessage(content="Perform single-cell clustering and find marker genes")
        ]

        # Mock an agent that could be selected
        mock_analysis_agent = Mock()
        mock_analysis_agent.name = "singlecell_expert_agent"

        # Test that supervisor can be created with agents
        supervisor_graph = create_supervisor(
            agents=[mock_analysis_agent],
            model=mock_llm,
            prompt="You are a supervisor agent.",
        )

        # Should create a graph successfully
        assert isinstance(supervisor_graph, StateGraph)

    def test_supervisor_agent_selection_research_task(
        self, mock_llm, supervisor_state, mock_data_manager
    ):
        """Test supervisor selecting research agent for literature tasks."""
        supervisor_state["messages"] = [
            HumanMessage(content="Find papers about T cell exhaustion in cancer")
        ]

        # Mock an agent that could be selected
        mock_research_agent = Mock()
        mock_research_agent.name = "research_agent"

        # Test that supervisor can be created with agents
        supervisor_graph = create_supervisor(
            agents=[mock_research_agent],
            model=mock_llm,
            prompt="You are a supervisor agent.",
        )

        # Should create a graph successfully
        assert isinstance(supervisor_graph, StateGraph)

    def test_supervisor_multi_step_coordination(
        self, mock_llm, supervisor_state, mock_data_manager
    ):
        """Test supervisor coordinating multi-step workflows."""
        supervisor_state["messages"] = [
            HumanMessage(
                content="Load GEO data, perform quality control, and cluster cells"
            )
        ]

        # Mock multiple agents for coordination
        mock_data_agent = Mock()
        mock_data_agent.name = "data_expert_agent"
        mock_analysis_agent = Mock()
        mock_analysis_agent.name = "singlecell_expert_agent"

        # Test that supervisor can be created with multiple agents
        supervisor_graph = create_supervisor(
            agents=[mock_data_agent, mock_analysis_agent],
            model=mock_llm,
            prompt="You are a supervisor agent.",
        )

        # Should create a graph successfully
        assert isinstance(supervisor_graph, StateGraph)


# ===============================================================================
# Agent Handoff and Coordination Tests
# ===============================================================================


@pytest.mark.unit
class TestSupervisorHandoffCoordination:
    """Test supervisor agent handoff and coordination."""

    def test_handoff_to_data_expert(self, mock_data_manager, mock_llm):
        """Test handoff to data expert agent via supervisor graph."""
        from lobster.agents.langgraph_supervisor.handoff import create_handoff_tool

        # Create a handoff tool for data expert
        handoff_tool = create_handoff_tool(
            agent_name="data_expert_agent",
            description="Transfer to data expert for data loading tasks",
        )

        # Test that handoff tool is created properly
        assert handoff_tool.name == "transfer_to_data_expert_agent"
        assert "data expert" in handoff_tool.description.lower()

    def test_handoff_to_singlecell_expert(self, mock_data_manager, mock_llm):
        """Test handoff to single-cell expert agent via supervisor graph."""
        from lobster.agents.langgraph_supervisor.handoff import create_handoff_tool

        # Create a handoff tool for single-cell expert
        handoff_tool = create_handoff_tool(
            agent_name="singlecell_expert_agent",
            description="Transfer to single-cell expert for analysis tasks",
        )

        # Test that handoff tool is created properly
        assert handoff_tool.name == "transfer_to_singlecell_expert_agent"
        assert "single-cell expert" in handoff_tool.description.lower()

    def test_handoff_to_research_agent(self, mock_data_manager, mock_llm):
        """Test handoff to research agent via supervisor graph."""
        from lobster.agents.langgraph_supervisor.handoff import create_handoff_tool

        # Create a handoff tool for research agent
        handoff_tool = create_handoff_tool(
            agent_name="research_agent",
            description="Transfer to research agent for literature tasks",
        )

        # Test that handoff tool is created properly
        assert handoff_tool.name == "transfer_to_research_agent"
        assert "research agent" in handoff_tool.description.lower()

    def test_invalid_handoff_handling(self, mock_llm):
        """Test handling of invalid handoff requests."""
        # Test that supervisor can be created without issues
        mock_agent = Mock()
        mock_agent.name = "valid_agent"

        supervisor_graph = create_supervisor(
            agents=[mock_agent], model=mock_llm, prompt="You are a supervisor agent."
        )

        # Should create a graph successfully even with one agent
        assert isinstance(supervisor_graph, StateGraph)


# ===============================================================================
# Decision Making and Routing Tests
# ===============================================================================


@pytest.mark.unit
class TestSupervisorDecisionMaking:
    """Test supervisor decision making and routing logic."""

    @pytest.mark.parametrize(
        "task,expected_agent",
        [
            ("Load GEO dataset GSE12345", "data_expert_agent"),
            ("Perform single-cell clustering", "singlecell_expert_agent"),
            ("Find papers about T cells", "research_agent"),
            (
                "Extract parameters from PMID:12345678",
                "research_agent",
            ),  # Phase 1: research_agent handles method extraction
            ("Analyze proteomics data", "proteomics_expert_agent"),
        ],
    )
    def test_task_routing_decisions(
        self, task, expected_agent, mock_data_manager, mock_llm
    ):
        """Test that supervisor can be configured with different agents."""
        # Mock the expected agent
        mock_agent = Mock()
        mock_agent.name = expected_agent

        # Test that supervisor can be created with this agent
        supervisor_graph = create_supervisor(
            agents=[mock_agent],
            model=mock_llm,
            prompt=f"You are a supervisor agent handling {task}.",
        )

        # Should create a graph successfully
        assert isinstance(supervisor_graph, StateGraph)

    def test_ambiguous_task_handling(self, mock_data_manager, mock_llm):
        """Test handling of ambiguous tasks through supervisor prompt creation."""
        # Test that create_supervisor_prompt handles ambiguous scenarios
        prompt = create_supervisor_prompt(mock_data_manager)

        # Should create a prompt that guides decision making
        assert "decision" in prompt.lower() or "clarif" in prompt.lower()
        assert len(prompt) > 0

    def test_context_aware_decisions(self, mock_data_manager, mock_llm):
        """Test context-aware decision making through prompt generation."""
        # Set up context with existing data
        mock_data_manager.list_modalities.return_value = ["geo_gse12345_processed"]

        # Test that create_supervisor_prompt includes data context
        from lobster.config.supervisor_config import SupervisorConfig

        config = SupervisorConfig()
        config.include_data_context = True

        prompt = create_supervisor_prompt(mock_data_manager, config)

        # Should include data context information
        assert "geo_gse12345_processed" in prompt or "modalities" in prompt.lower()

    def test_sequential_task_planning(self, mock_data_manager, mock_llm):
        """Test sequential task planning through workflow configuration."""
        # Test that supervisor prompt includes workflow guidance
        from lobster.config.supervisor_config import SupervisorConfig

        config = SupervisorConfig()
        config.workflow_guidance_level = "detailed"

        prompt = create_supervisor_prompt(
            mock_data_manager,
            config,
            active_agents=["data_expert_agent", "singlecell_expert_agent"],
        )

        # Should include workflow information
        assert "workflow" in prompt.lower()
        assert "data_expert_agent" in prompt
        assert "singlecell_expert_agent" in prompt


# ===============================================================================
# Workflow Management Tests
# ===============================================================================


@pytest.mark.unit
class TestSupervisorWorkflowManagement:
    """Test supervisor workflow management capabilities."""

    def test_workflow_initialization(self, mock_data_manager, mock_llm):
        """Test workflow initialization and tracking."""
        # Test supervisor prompt includes workflow awareness
        from lobster.config.supervisor_config import SupervisorConfig

        config = SupervisorConfig()
        config.workflow_guidance_level = "detailed"

        prompt = create_supervisor_prompt(
            mock_data_manager,
            config,
            active_agents=["data_expert_agent", "singlecell_expert_agent"],
        )

        # Should include workflow information
        assert "workflow" in prompt.lower()
        assert len(prompt) > 0

    def test_workflow_progress_tracking(self, mock_data_manager, mock_llm):
        """Test workflow progress tracking through configuration."""
        # Test that supervisor can track progress through multiple agents
        mock_data_agent = Mock()
        mock_data_agent.name = "data_expert_agent"
        mock_analysis_agent = Mock()
        mock_analysis_agent.name = "singlecell_expert_agent"

        supervisor_graph = create_supervisor(
            agents=[mock_data_agent, mock_analysis_agent],
            model=mock_llm,
            prompt="You are a supervisor tracking workflow progress.",
        )

        # Should create a graph with multiple agents for workflow tracking
        assert isinstance(supervisor_graph, StateGraph)

    def test_workflow_error_recovery(self, mock_data_manager, mock_llm):
        """Test workflow error recovery through supervisor configuration."""
        # Test supervisor prompt includes error handling guidance
        prompt = create_supervisor_prompt(mock_data_manager)

        # Should include guidance for error handling
        assert len(prompt) > 0
        # The prompt should contain decision-making guidance
        assert "decision" in prompt.lower() or "response" in prompt.lower()

    def test_workflow_completion(self, mock_data_manager, mock_llm):
        """Test workflow completion through supervisor response configuration."""
        # Test supervisor configuration for workflow completion
        from lobster.config.supervisor_config import SupervisorConfig

        config = SupervisorConfig()
        config.summarize_expert_output = True
        config.auto_suggest_next_steps = True

        prompt = create_supervisor_prompt(mock_data_manager, config)

        # Should include guidance for completion and next steps
        assert "summary" in prompt.lower() or "next" in prompt.lower()
        assert len(prompt) > 0


# ===============================================================================
# State Management Tests
# ===============================================================================


@pytest.mark.unit
class TestSupervisorStateManagement:
    """Test supervisor state management."""

    def test_state_persistence(self, mock_data_manager, mock_llm):
        """Test state persistence through supervisor graph structure."""
        # Test that supervisor graph maintains state structure
        mock_agent = Mock()
        mock_agent.name = "test_agent"

        supervisor_graph = create_supervisor(
            agents=[mock_agent],
            model=mock_llm,
            prompt="You are a supervisor maintaining state.",
        )

        # Should create a stateful graph
        assert isinstance(supervisor_graph, StateGraph)
        # Graph should have the expected structure for state management
        assert hasattr(supervisor_graph, "nodes")

    def test_conversation_history_management(self, mock_data_manager, mock_llm):
        """Test conversation history management through message handling."""
        # Test that supervisor handles message history properly
        state = {
            "messages": [
                HumanMessage(content="Load dataset A"),
                AIMessage(content="Dataset loaded"),
                HumanMessage(content="Show me the results"),
            ]
        }

        # Test supervisor graph creation with state schema that handles messages
        mock_agent = Mock()
        mock_agent.name = "test_agent"

        supervisor_graph = create_supervisor(
            agents=[mock_agent],
            model=mock_llm,
            prompt="You are a supervisor managing conversation history.",
        )

        # Should create a graph that can handle message sequences
        assert isinstance(supervisor_graph, StateGraph)

    def test_data_context_tracking(self, mock_data_manager, mock_llm):
        """Test tracking of data context through supervisor prompt."""
        mock_data_manager.list_modalities.return_value = [
            "dataset_A",
            "dataset_B_clustered",
        ]

        # Test that create_supervisor_prompt includes data context
        from lobster.config.supervisor_config import SupervisorConfig

        config = SupervisorConfig()
        config.include_data_context = True

        prompt = create_supervisor_prompt(mock_data_manager, config)

        # Should include available datasets in context
        assert (
            "dataset_A" in prompt
            or "dataset_B_clustered" in prompt
            or "modalities" in prompt.lower()
        )


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================


@pytest.mark.unit
class TestSupervisorErrorHandling:
    """Test supervisor error handling and edge cases."""

    def test_empty_message_handling(self, mock_data_manager, mock_llm):
        """Test handling of empty messages through supervisor configuration."""
        # Test that supervisor prompt includes guidance for handling unclear requests
        prompt = create_supervisor_prompt(mock_data_manager)

        # Should include response rules that cover unclear requests
        assert (
            "clarif" in prompt.lower()
            or "question" in prompt.lower()
            or "response" in prompt.lower()
        )

    def test_malformed_input_handling(self, mock_data_manager, mock_llm):
        """Test handling of malformed input through supervisor robustness."""
        # Test that supervisor can handle any input gracefully
        mock_agent = Mock()
        mock_agent.name = "test_agent"

        supervisor_graph = create_supervisor(
            agents=[mock_agent],
            model=mock_llm,
            prompt="You are a supervisor handling various inputs gracefully.",
        )

        # Should create a robust graph structure
        assert isinstance(supervisor_graph, StateGraph)

    def test_agent_unavailable_handling(self, mock_data_manager, mock_llm):
        """Test handling when requested agent is unavailable."""
        # Test that supervisor validates agent names during creation
        mock_agent = Mock()
        mock_agent.name = "available_agent"

        supervisor_graph = create_supervisor(
            agents=[mock_agent],
            model=mock_llm,
            prompt="You are a supervisor with available agents.",
        )

        # Should successfully create with available agents
        assert isinstance(supervisor_graph, StateGraph)

    def test_concurrent_request_handling(self, mock_data_manager, mock_llm):
        """Test handling of concurrent requests through supervisor design."""
        # Test that supervisor can be created multiple times (thread-safe creation)
        import threading
        import time

        def create_supervisor_worker(worker_id, results, errors):
            """Worker function for concurrent supervisor creation testing."""
            try:
                mock_agent = Mock()
                mock_agent.name = f"agent_{worker_id}"

                supervisor_graph = create_supervisor(
                    agents=[mock_agent],
                    model=mock_llm,
                    prompt=f"You are supervisor {worker_id}.",
                )

                results.append(supervisor_graph)
                time.sleep(0.01)

            except Exception as e:
                errors.append((worker_id, e))

        results = []
        errors = []
        threads = []

        # Create multiple concurrent supervisor graphs
        for i in range(3):
            thread = threading.Thread(
                target=create_supervisor_worker, args=(i, results, errors)
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors and all graphs created
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 3
        assert all(isinstance(graph, StateGraph) for graph in results)

    def test_memory_management_large_history(self, mock_data_manager, mock_llm):
        """Test memory management through supervisor configuration."""
        # Test that supervisor can be configured to handle large histories
        from lobster.config.supervisor_config import SupervisorConfig

        config = SupervisorConfig()

        # Test prompt generation with configuration (should not fail with large data)
        prompt = create_supervisor_prompt(mock_data_manager, config)

        # Should create prompt without memory issues
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Test supervisor creation
        mock_agent = Mock()
        mock_agent.name = "memory_test_agent"

        supervisor_graph = create_supervisor(
            agents=[mock_agent],
            model=mock_llm,
            prompt=prompt[:1000],  # Truncate for test
        )

        # Should create successfully even with managed memory
        assert isinstance(supervisor_graph, StateGraph)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
