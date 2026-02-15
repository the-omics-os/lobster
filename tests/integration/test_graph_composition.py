"""Integration tests for Phase 5: Graph Composition Refactor.

Tests verify:
- GRAPH-01: create_bioinformatics_graph() accepts parsed config object
- GRAPH-02: Graph loads only agents specified in config
- GRAPH-03: Single-pass agent creation (no re-creation)
- GRAPH-04: Factory functions receive standardized parameters
- GRAPH-05: Graph builder resolves agents from ComponentRegistry
- GRAPH-06: Lazy delegation tools resolve child agents at invocation time
- GRAPH-07: Returns (graph, GraphMetadata) tuple
"""

import inspect
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lobster.agents.graph import (
    AgentInfo,
    GraphMetadata,
    _create_lazy_delegation_tool,
    create_bioinformatics_graph,
)
from lobster.config.workspace_agent_config import WorkspaceAgentConfig


class TestGraphCompositionPhase5:
    """Test suite for Phase 5 graph composition requirements."""

    @pytest.fixture
    def mock_data_manager(self, tmp_path):
        """Create a minimal mock DataManagerV2 with Path workspace."""
        dm = MagicMock()
        # workspace_path must be a Path object, not string
        dm.workspace_path = tmp_path / "test_workspace"
        dm.workspace_path.mkdir(exist_ok=True)
        dm.get_modality_ids.return_value = []
        return dm

    # =========================================================================
    # GRAPH-01: Accepts parsed config object
    # =========================================================================
    def test_graph_accepts_config_parameter(self, mock_data_manager):
        """Verify create_bioinformatics_graph accepts WorkspaceAgentConfig."""
        # Should not raise TypeError - verify parameter exists
        sig = inspect.signature(create_bioinformatics_graph)
        assert "config" in sig.parameters, "config parameter missing"
        assert sig.parameters["config"].default is None, "config should default to None"

    def test_graph_accepts_enabled_agents_parameter(self, mock_data_manager):
        """Verify create_bioinformatics_graph accepts enabled_agents list."""
        sig = inspect.signature(create_bioinformatics_graph)
        assert "enabled_agents" in sig.parameters, "enabled_agents parameter missing"
        assert (
            sig.parameters["enabled_agents"].default is None
        ), "enabled_agents should default to None"

    def test_graph_accepts_config_object_at_runtime(self, mock_data_manager):
        """Verify config object can be passed at runtime without error."""
        config = WorkspaceAgentConfig(enabled_agents=["research_agent"])

        # Should not raise when passed config object
        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            config=config,
        )

        assert graph is not None
        assert metadata is not None

    # =========================================================================
    # GRAPH-02: Loads only agents specified in config
    # =========================================================================
    def test_graph_filters_agents_by_enabled_list(self, mock_data_manager, caplog):
        """Verify graph creates only enabled agents via enabled_agents param."""
        # Use agents that are actually registered in ComponentRegistry
        enabled = ["research_agent", "transcriptomics_expert"]

        with caplog.at_level(logging.DEBUG):
            graph, metadata = create_bioinformatics_graph(
                data_manager=mock_data_manager,
                enabled_agents=enabled,
            )

        # Verify only requested agents are in metadata
        agent_names = [a.name for a in metadata.available_agents]
        for name in enabled:
            assert name in agent_names, f"Expected {name} in available agents"

        # Verify excluded agents are NOT in metadata (any agent not in enabled list)
        for agent_info in metadata.available_agents:
            assert (
                agent_info.name in enabled
            ), f"Unexpected agent {agent_info.name} in metadata"

    def test_graph_filters_agents_by_config_object(self, mock_data_manager):
        """Verify graph creates only enabled agents via config object."""
        config = WorkspaceAgentConfig(enabled_agents=["research_agent"])

        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            config=config,
        )

        agent_names = [a.name for a in metadata.available_agents]
        assert "research_agent" in agent_names

        # Verify it filtered to just the enabled agent
        assert len(agent_names) == 1, f"Expected 1 agent, got {len(agent_names)}"

    def test_enabled_agents_param_overrides_config(self, mock_data_manager):
        """Verify enabled_agents param takes precedence over config.enabled_agents."""
        config = WorkspaceAgentConfig(enabled_agents=["transcriptomics_expert"])

        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            config=config,
            enabled_agents=["research_agent"],  # This should override
        )

        agent_names = [a.name for a in metadata.available_agents]
        assert "research_agent" in agent_names
        assert "transcriptomics_expert" not in agent_names

    # =========================================================================
    # GRAPH-03: Single-pass agent creation
    # =========================================================================
    def test_graph_single_pass_creation_log_marker(self, mock_data_manager):
        """Verify agents log '[single pass]' marker during creation."""
        import io
        import logging as stdlib_logging

        # Capture logs from graph module directly
        log_capture = io.StringIO()
        handler = stdlib_logging.StreamHandler(log_capture)
        handler.setLevel(stdlib_logging.DEBUG)
        handler.setFormatter(stdlib_logging.Formatter("%(message)s"))

        graph_logger = stdlib_logging.getLogger("lobster.agents.graph")
        original_level = graph_logger.level
        graph_logger.setLevel(stdlib_logging.DEBUG)
        graph_logger.addHandler(handler)

        try:
            graph, metadata = create_bioinformatics_graph(
                data_manager=mock_data_manager,
                enabled_agents=["research_agent"],
            )
        finally:
            graph_logger.removeHandler(handler)
            graph_logger.setLevel(original_level)

        # Check logs for single pass marker
        log_text = log_capture.getvalue().lower()
        assert (
            "single pass" in log_text
        ), f"Expected '[single pass]' in logs, got: {log_text[:500]}"

    def test_graph_no_recreation_logs(self, mock_data_manager, caplog):
        """Verify agents are NOT re-created (no two-pass pattern)."""
        with caplog.at_level(logging.DEBUG, logger="lobster.agents.graph"):
            graph, metadata = create_bioinformatics_graph(
                data_manager=mock_data_manager,
                enabled_agents=["research_agent"],
            )

        # Check NO re-creation logs exist
        log_text = caplog.text.lower()
        assert (
            "re-created" not in log_text
        ), "Found re-creation logs - two-pass not eliminated"
        assert (
            "recreated" not in log_text
        ), "Found recreated logs - two-pass not eliminated"

    # =========================================================================
    # GRAPH-04: Factory functions receive standardized parameters
    # =========================================================================
    def test_factory_functions_have_standardized_signature(self):
        """Verify agent factory functions have standardized signature."""
        # Factory functions are named after the agent (e.g., research_agent not create_research_agent)
        from lobster.agents.research.research_agent import research_agent

        sig = inspect.signature(research_agent)
        required_params = ["data_manager", "callback_handler", "agent_name"]

        for param in required_params:
            assert param in sig.parameters, f"Missing required param: {param}"

    def test_factory_functions_accept_delegation_tools(self):
        """Verify factory functions can accept delegation_tools parameter."""
        # Factory functions are named after the agent
        from lobster.agents.transcriptomics.transcriptomics_expert import (
            transcriptomics_expert,
        )

        sig = inspect.signature(transcriptomics_expert)
        assert "delegation_tools" in sig.parameters, "delegation_tools param missing"

    # =========================================================================
    # GRAPH-05: Graph builder resolves agents from ComponentRegistry
    # =========================================================================
    def test_graph_uses_component_registry(self, mock_data_manager):
        """Verify graph builder uses ComponentRegistry for agent discovery."""
        # Verify by checking that ComponentRegistry.list_agents() provides the agents
        from lobster.core.component_registry import component_registry

        # Get agents from ComponentRegistry
        registry_agents = component_registry.list_agents()

        # Create graph and verify it uses agents from registry
        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=["research_agent"],
        )

        # The available agents should come from ComponentRegistry
        available_names = [a.name for a in metadata.available_agents]
        assert "research_agent" in available_names

        # research_agent should exist in ComponentRegistry
        assert "research_agent" in registry_agents

    def test_graph_source_code_uses_component_registry(self):
        """Verify graph.py imports and uses component_registry."""
        import lobster.agents.graph as graph_module

        source = inspect.getsource(graph_module.create_bioinformatics_graph)

        # Should import from component_registry
        assert "component_registry" in source, "Should use component_registry"
        assert "list_agents" in source, "Should call list_agents()"

    # =========================================================================
    # GRAPH-06: Lazy delegation tools resolve at invocation time
    # =========================================================================
    def test_lazy_delegation_tool_handles_missing_agent(self):
        """Verify lazy tool returns graceful message when agent not in dict."""
        agents_dict = {}  # Empty - agent not yet created

        tool = _create_lazy_delegation_tool(
            agent_name="test_child",
            agents_dict=agents_dict,
            description="Test child agent",
        )

        assert tool is not None
        # Tool should have invoke method (from @tool decorator)
        assert hasattr(tool, "invoke") or callable(tool)

        # Invoke tool - should return graceful message since agent not in dict
        result = tool.invoke("test task")
        # The message contains "not available" - check case-insensitively
        assert (
            "not available" in result.lower() or "is not available" in result
        ), f"Expected 'not available' in: {result}"

    def test_lazy_delegation_tool_resolves_after_creation(self):
        """Verify lazy tool resolves agent after it's added to dict."""
        agents_dict = {}

        tool = _create_lazy_delegation_tool(
            agent_name="test_child",
            agents_dict=agents_dict,
            description="Test child agent",
        )

        # First invocation - agent not yet created
        result_before = tool.invoke("test task")
        assert "not available" in result_before.lower()

        # Now add agent to dict (simulating creation)
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="Success from child")]
        }
        agents_dict["test_child"] = mock_agent

        # Second invocation - should now work
        result_after = tool.invoke("test task")
        assert (
            result_after == "Success from child"
        ), f"Expected 'Success from child', got: {result_after}"

    def test_lazy_delegation_tool_closure_captures_dict_reference(self):
        """Verify tool captures dict by reference, not by value."""
        agents_dict = {}

        tool = _create_lazy_delegation_tool(
            agent_name="child", agents_dict=agents_dict, description="Test"
        )

        # Verify tool was created with tool decorator
        assert hasattr(tool, "invoke"), "Tool should have invoke method"

        # The dict should be captured by reference
        # Adding to dict after tool creation should affect tool behavior
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [MagicMock(content="OK")]}
        agents_dict["child"] = mock_agent

        result = tool.invoke("task")
        assert result == "OK", "Lazy resolution should work after dict update"

    # =========================================================================
    # GRAPH-07: Returns (graph, GraphMetadata) tuple
    # =========================================================================
    def test_graph_returns_tuple(self, mock_data_manager):
        """Verify return type is tuple."""
        result = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=["research_agent"],
        )

        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return 2-tuple"

    def test_graph_returns_graph_and_metadata(self, mock_data_manager):
        """Verify second element is GraphMetadata."""
        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=["research_agent"],
        )

        assert graph is not None, "Graph should not be None"
        assert isinstance(
            metadata, GraphMetadata
        ), "Second element should be GraphMetadata"

    def test_metadata_has_required_attributes(self, mock_data_manager):
        """Verify GraphMetadata has all required attributes."""
        _, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            enabled_agents=["research_agent"],
        )

        assert hasattr(metadata, "subscription_tier")
        assert hasattr(metadata, "available_agents")
        assert hasattr(metadata, "supervisor_accessible_agents")
        assert hasattr(metadata, "filtered_out_agents")
        assert hasattr(metadata, "agent_count")
        assert hasattr(metadata, "supervisor_accessible_count")
        assert hasattr(metadata, "to_dict")

    # =========================================================================
    # Backward compatibility
    # =========================================================================
    def test_backward_compatibility_no_config(self, mock_data_manager):
        """Verify function works without config/enabled_agents params."""
        # Should not raise when called without new params
        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
        )

        assert graph is not None
        assert metadata is not None
        # When no filter, should have multiple agents
        assert len(metadata.available_agents) >= 1

    def test_backward_compatibility_existing_params(self, mock_data_manager):
        """Verify existing parameters still work (agent_filter, subscription_tier)."""
        graph, metadata = create_bioinformatics_graph(
            data_manager=mock_data_manager,
            subscription_tier="free",
            agent_filter=lambda name, config: name == "research_agent",
        )

        assert metadata.subscription_tier == "free"


class TestGraphMetadata:
    """Tests for GraphMetadata dataclass."""

    def test_metadata_construction(self):
        """Verify GraphMetadata can be constructed."""
        metadata = GraphMetadata(
            subscription_tier="free",
            available_agents=[],
            supervisor_accessible_agents=[],
            filtered_out_agents=[],
        )

        assert metadata.subscription_tier == "free"
        assert metadata.available_agents == []

    def test_metadata_agent_count_property(self):
        """Verify agent_count property."""
        agent = AgentInfo(
            name="test",
            display_name="Test",
            description="Test agent",
            is_supervisor_accessible=True,
        )
        metadata = GraphMetadata(
            subscription_tier="free",
            available_agents=[agent],
            supervisor_accessible_agents=["test"],
            filtered_out_agents=[],
        )

        assert metadata.agent_count == 1

    def test_metadata_supervisor_accessible_count_property(self):
        """Verify supervisor_accessible_count property."""
        metadata = GraphMetadata(
            subscription_tier="pro",
            available_agents=[],
            supervisor_accessible_agents=["agent1", "agent2"],
            filtered_out_agents=[],
        )

        assert metadata.supervisor_accessible_count == 2

    def test_metadata_to_dict_serialization(self):
        """Verify GraphMetadata serializes correctly."""
        agent = AgentInfo(
            name="test_agent",
            display_name="Test Agent",
            description="A test agent",
            is_supervisor_accessible=True,
        )
        metadata = GraphMetadata(
            subscription_tier="pro",
            available_agents=[agent],
            supervisor_accessible_agents=["test_agent"],
            filtered_out_agents=["filtered_agent"],
        )

        d = metadata.to_dict()

        assert d["subscription_tier"] == "pro"
        assert len(d["available_agents"]) == 1
        assert d["available_agents"][0]["name"] == "test_agent"
        assert d["supervisor_accessible_agents"] == ["test_agent"]
        assert d["filtered_out_agents"] == ["filtered_agent"]
        assert d["agent_count"] == 1
        assert d["supervisor_accessible_count"] == 1

    def test_metadata_get_agent_by_name(self):
        """Verify get_agent_by_name lookup."""
        agent = AgentInfo(
            name="target",
            display_name="Target",
            description="Target agent",
            is_supervisor_accessible=True,
        )
        metadata = GraphMetadata(
            subscription_tier="free",
            available_agents=[agent],
            supervisor_accessible_agents=[],
            filtered_out_agents=[],
        )

        found = metadata.get_agent_by_name("target")
        assert found is not None
        assert found.name == "target"

        not_found = metadata.get_agent_by_name("nonexistent")
        assert not_found is None


class TestAgentInfo:
    """Tests for AgentInfo dataclass."""

    def test_agent_info_construction(self):
        """Verify AgentInfo can be constructed."""
        info = AgentInfo(
            name="test",
            display_name="Test Agent",
            description="A test agent",
            is_supervisor_accessible=True,
            parent_agent=None,
            child_agents=["child1", "child2"],
            handoff_tool_name="handoff_to_test",
        )

        assert info.name == "test"
        assert info.display_name == "Test Agent"
        assert info.child_agents == ["child1", "child2"]

    def test_agent_info_to_dict(self):
        """Verify AgentInfo serializes correctly."""
        info = AgentInfo(
            name="test",
            display_name="Test Agent",
            description="A test agent",
            is_supervisor_accessible=True,
            parent_agent="parent",
            child_agents=None,
            handoff_tool_name="handoff_to_test",
        )

        d = info.to_dict()

        assert d["name"] == "test"
        assert d["display_name"] == "Test Agent"
        assert d["description"] == "A test agent"
        assert d["is_supervisor_accessible"] is True
        assert d["parent_agent"] == "parent"
        assert d["child_agents"] is None
        assert d["handoff_tool_name"] == "handoff_to_test"
