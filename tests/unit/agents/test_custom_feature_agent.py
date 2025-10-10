"""
Unit tests for custom feature agent.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lobster.agents.custom_feature_agent import (
    CustomFeatureError,
    FeatureCreationError,
    SDKConnectionError,
    ValidationError,
    custom_feature_agent,
)
from lobster.agents.state import CustomFeatureAgentState
from lobster.core.data_manager_v2 import DataManagerV2


@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2 for testing."""
    data_manager = MagicMock(spec=DataManagerV2)
    data_manager.modalities = {}
    data_manager.log_tool_usage = MagicMock()
    return data_manager


@pytest.fixture
def temp_lobster_root(tmp_path):
    """Create temporary Lobster directory structure for testing."""
    # Create directory structure
    agents_dir = tmp_path / "lobster" / "agents"
    agents_dir.mkdir(parents=True)

    tools_dir = tmp_path / "lobster" / "tools"
    tools_dir.mkdir(parents=True)

    tests_agents = tmp_path / "tests" / "unit" / "agents"
    tests_agents.mkdir(parents=True)

    tests_tools = tmp_path / "tests" / "unit" / "tools"
    tests_tools.mkdir(parents=True)

    wiki_dir = tmp_path / "lobster" / "wiki"
    wiki_dir.mkdir(parents=True)

    # Create CLAUDE.md
    claude_md = agents_dir / "CLAUDE.md"
    claude_md.write_text("# Test CLAUDE.md\nTest content")

    return tmp_path


def test_agent_creation(mock_data_manager):
    """Test that agent can be created successfully."""
    agent = custom_feature_agent(mock_data_manager)
    assert agent is not None
    assert hasattr(agent, 'name')


def test_agent_has_required_tools(mock_data_manager):
    """Test that agent has all required tools."""
    agent = custom_feature_agent(mock_data_manager)

    # Get tool names
    tool_names = [tool.name for tool in agent.tools]

    # Check required tools exist
    required_tools = [
        "create_new_feature",
        "list_existing_patterns",
        "validate_feature_name_tool",
        "create_feature_summary"
    ]

    for required_tool in required_tools:
        assert required_tool in tool_names, f"Tool '{required_tool}' not found in agent tools"


def test_validate_feature_name_tool_valid(mock_data_manager, temp_lobster_root):
    """Test feature name validation with valid name."""
    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)

        # Extract the validation tool
        validate_tool = next(t for t in agent.tools if t.name == "validate_feature_name_tool")

        result = validate_tool.invoke({"feature_name": "spatial_transcriptomics"})

        assert isinstance(result, str)
        assert "valid" in result.lower()


def test_validate_feature_name_tool_invalid_uppercase(mock_data_manager, temp_lobster_root):
    """Test feature name validation with invalid uppercase name."""
    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        validate_tool = next(t for t in agent.tools if t.name == "validate_feature_name_tool")

        result = validate_tool.invoke({"feature_name": "SpatialTranscriptomics"})

        assert "invalid" in result.lower() or "error" in result.lower()
        assert "lowercase" in result.lower()


def test_validate_feature_name_tool_invalid_hyphen(mock_data_manager, temp_lobster_root):
    """Test feature name validation with hyphens."""
    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        validate_tool = next(t for t in agent.tools if t.name == "validate_feature_name_tool")

        result = validate_tool.invoke({"feature_name": "spatial-transcriptomics"})

        assert "invalid" in result.lower() or "error" in result.lower()


def test_validate_feature_name_tool_reserved_name(mock_data_manager, temp_lobster_root):
    """Test feature name validation with reserved name."""
    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        validate_tool = next(t for t in agent.tools if t.name == "validate_feature_name_tool")

        result = validate_tool.invoke({"feature_name": "supervisor"})

        assert "reserved" in result.lower() or "invalid" in result.lower()


def test_validate_feature_name_tool_existing_files(mock_data_manager, temp_lobster_root):
    """Test feature name validation when files already exist."""
    # Create existing agent file
    agent_file = temp_lobster_root / "lobster" / "agents" / "test_feature_expert.py"
    agent_file.write_text("# Existing agent")

    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        validate_tool = next(t for t in agent.tools if t.name == "validate_feature_name_tool")

        result = validate_tool.invoke({"feature_name": "test_feature"})

        assert "existing" in result.lower()


def test_list_existing_patterns(mock_data_manager, temp_lobster_root):
    """Test listing existing patterns."""
    # Create some example agent files
    (temp_lobster_root / "lobster" / "agents" / "example_expert.py").write_text("# Example agent")
    (temp_lobster_root / "lobster" / "tools" / "example_service.py").write_text("# Example service")

    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        list_tool = next(t for t in agent.tools if t.name == "list_existing_patterns")

        result = list_tool.invoke({})

        assert isinstance(result, str)
        assert "example" in result.lower() or "agents" in result.lower()


def test_create_feature_summary_empty(mock_data_manager):
    """Test creating summary when no features created."""
    agent = custom_feature_agent(mock_data_manager)
    summary_tool = next(t for t in agent.tools if t.name == "create_feature_summary")

    result = summary_tool.invoke({})

    assert isinstance(result, str)
    assert "no features" in result.lower() or "not" in result.lower()


@patch('lobster.agents.custom_feature_agent.asyncio.new_event_loop')
@patch('lobster.agents.custom_feature_agent.ClaudeSDKClient')
def test_create_new_feature_invalid_type(mock_sdk_client, mock_event_loop, mock_data_manager, temp_lobster_root):
    """Test create_new_feature with invalid feature type."""
    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        create_tool = next(t for t in agent.tools if t.name == "create_new_feature")

        result = create_tool.invoke({
            "feature_type": "invalid_type",
            "feature_name": "test_feature",
            "requirements": "Create a test feature for testing purposes"
        })

        assert "invalid" in result.lower() or "error" in result.lower()
        assert "feature type" in result.lower()


@patch('lobster.agents.custom_feature_agent.asyncio.new_event_loop')
@patch('lobster.agents.custom_feature_agent.ClaudeSDKClient')
def test_create_new_feature_invalid_name(mock_sdk_client, mock_event_loop, mock_data_manager, temp_lobster_root):
    """Test create_new_feature with invalid feature name."""
    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        create_tool = next(t for t in agent.tools if t.name == "create_new_feature")

        result = create_tool.invoke({
            "feature_type": "agent",
            "feature_name": "Invalid-Name",
            "requirements": "Create a test feature"
        })

        assert "invalid" in result.lower() or "error" in result.lower()


@patch('lobster.agents.custom_feature_agent.asyncio.new_event_loop')
@patch('lobster.agents.custom_feature_agent.ClaudeSDKClient')
def test_create_new_feature_insufficient_requirements(mock_sdk_client, mock_event_loop, mock_data_manager, temp_lobster_root):
    """Test create_new_feature with insufficient requirements."""
    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        create_tool = next(t for t in agent.tools if t.name == "create_new_feature")

        result = create_tool.invoke({
            "feature_type": "agent",
            "feature_name": "test_feature",
            "requirements": "Short"  # Too brief
        })

        assert "brief" in result.lower() or "requirements" in result.lower()


@patch('lobster.agents.custom_feature_agent.asyncio.new_event_loop')
def test_create_new_feature_existing_files(mock_event_loop, mock_data_manager, temp_lobster_root):
    """Test create_new_feature when files already exist."""
    # Create existing file
    agent_file = temp_lobster_root / "lobster" / "agents" / "test_feature_expert.py"
    agent_file.write_text("# Existing")

    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        create_tool = next(t for t in agent.tools if t.name == "create_new_feature")

        result = create_tool.invoke({
            "feature_type": "agent",
            "feature_name": "test_feature",
            "requirements": "Create a test feature for analyzing spatial data"
        })

        assert "existing" in result.lower()


@patch('lobster.agents.custom_feature_agent.asyncio.new_event_loop')
def test_create_new_feature_sdk_import_error(mock_event_loop, mock_data_manager, temp_lobster_root):
    """Test create_new_feature when SDK not installed."""
    # Mock import error
    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        # Mock the import to raise ImportError
        with patch('lobster.agents.custom_feature_agent.asyncio') as mock_asyncio:
            mock_loop = MagicMock()
            mock_asyncio.new_event_loop.return_value = mock_loop
            mock_asyncio.set_event_loop = MagicMock()

            # Create coroutine that returns SDK error
            async def mock_create_coroutine(*args, **kwargs):
                return {
                    "success": False,
                    "created_files": [],
                    "error": "Claude Agent SDK not installed. Please install with: pip install claude-agent-sdk",
                    "sdk_output": ""
                }

            mock_loop.run_until_complete.return_value = asyncio.run(mock_create_coroutine())

            agent = custom_feature_agent(mock_data_manager)
            create_tool = next(t for t in agent.tools if t.name == "create_new_feature")

            result = create_tool.invoke({
                "feature_type": "agent",
                "feature_name": "test_feature",
                "requirements": "Create a test feature for analyzing spatial transcriptomics data"
            })

            assert "sdk" in result.lower() or "install" in result.lower()


@patch('lobster.agents.custom_feature_agent.asyncio.new_event_loop')
@patch('lobster.agents.custom_feature_agent.ClaudeSDKClient')
def test_create_new_feature_success_agent(mock_sdk_client, mock_event_loop, mock_data_manager, temp_lobster_root):
    """Test successful agent creation."""
    # Setup mock SDK client
    mock_client_instance = AsyncMock()
    mock_sdk_client.return_value.__aenter__.return_value = mock_client_instance
    mock_sdk_client.return_value.__aexit__.return_value = None

    # Mock the async iterator for receive_response
    async def mock_messages():
        yield MagicMock(content="Creating agent file at lobster/agents/test_feature_expert.py")
        yield MagicMock(content="Creating tests at tests/unit/agents/test_test_feature_expert.py")

    mock_client_instance.receive_response.return_value = mock_messages()

    # Create the expected files
    agent_file = temp_lobster_root / "lobster" / "agents" / "test_feature_expert.py"
    agent_file.write_text("# Created agent")

    test_file = temp_lobster_root / "tests" / "unit" / "agents" / "test_test_feature_expert.py"
    test_file.write_text("# Created tests")

    wiki_file = temp_lobster_root / "lobster" / "wiki" / "Test-Feature.md"
    wiki_file.write_text("# Wiki")

    # Mock event loop
    mock_loop = MagicMock()
    mock_event_loop.return_value = mock_loop
    mock_loop.run_until_complete.return_value = {
        "success": True,
        "created_files": [str(agent_file), str(test_file), str(wiki_file)],
        "error": None,
        "sdk_output": "Files created successfully"
    }

    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        create_tool = next(t for t in agent.tools if t.name == "create_new_feature")

        result = create_tool.invoke({
            "feature_type": "agent",
            "feature_name": "test_feature",
            "requirements": "Create a test feature agent for analyzing spatial transcriptomics data with clustering and visualization tools"
        })

        assert "success" in result.lower()
        assert "test_feature" in result.lower()
        # Should have registry instructions
        assert "registry" in result.lower()


@patch('lobster.agents.custom_feature_agent.asyncio.new_event_loop')
@patch('lobster.agents.custom_feature_agent.ClaudeSDKClient')
def test_create_new_feature_success_service(mock_sdk_client, mock_event_loop, mock_data_manager, temp_lobster_root):
    """Test successful service creation."""
    # Create the expected files
    service_file = temp_lobster_root / "lobster" / "tools" / "test_service.py"
    service_file.write_text("# Created service")

    test_file = temp_lobster_root / "tests" / "unit" / "tools" / "test_test_service.py"
    test_file.write_text("# Created tests")

    wiki_file = temp_lobster_root / "lobster" / "wiki" / "Test-Service.md"
    wiki_file.write_text("# Wiki")

    # Mock event loop
    mock_loop = MagicMock()
    mock_event_loop.return_value = mock_loop
    mock_loop.run_until_complete.return_value = {
        "success": True,
        "created_files": [str(service_file), str(test_file), str(wiki_file)],
        "error": None,
        "sdk_output": "Service created"
    }

    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        create_tool = next(t for t in agent.tools if t.name == "create_new_feature")

        result = create_tool.invoke({
            "feature_type": "service",
            "feature_name": "test_service",
            "requirements": "Create a stateless service for processing spatial transcriptomics data"
        })

        assert "success" in result.lower()
        # Services don't need registry update
        assert "service" in result.lower()


def test_custom_feature_error_hierarchy():
    """Test custom exception hierarchy."""
    assert issubclass(SDKConnectionError, CustomFeatureError)
    assert issubclass(FeatureCreationError, CustomFeatureError)
    assert issubclass(ValidationError, CustomFeatureError)


def test_agent_state_class():
    """Test that CustomFeatureAgentState is properly defined."""
    # This will raise an error if the state class is not properly defined
    from lobster.agents.state import CustomFeatureAgentState

    assert hasattr(CustomFeatureAgentState, 'next')
    assert hasattr(CustomFeatureAgentState, 'task_description')
    assert hasattr(CustomFeatureAgentState, 'feature_type')
    assert hasattr(CustomFeatureAgentState, 'feature_name')
    assert hasattr(CustomFeatureAgentState, 'created_files')


@pytest.mark.parametrize("feature_name,expected_valid", [
    ("spatial_transcriptomics", True),
    ("metabolomics", True),
    ("variant_calling_123", True),
    ("SpatialTranscriptomics", False),  # Uppercase
    ("spatial-transcriptomics", False),  # Hyphen
    ("spatial_transcriptomics_", False),  # Trailing underscore
    ("123_analysis", False),  # Starts with number
    ("supervisor", False),  # Reserved
    ("data", False),  # Reserved
])
def test_validate_feature_name_parametrized(feature_name, expected_valid, mock_data_manager, temp_lobster_root):
    """Test feature name validation with multiple cases."""
    with patch('lobster.agents.custom_feature_agent.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = temp_lobster_root

        agent = custom_feature_agent(mock_data_manager)
        validate_tool = next(t for t in agent.tools if t.name == "validate_feature_name_tool")

        result = validate_tool.invoke({"feature_name": feature_name})

        if expected_valid:
            assert "valid" in result.lower() and "invalid" not in result.lower()
        else:
            assert "invalid" in result.lower() or "error" in result.lower() or "reserved" in result.lower()
