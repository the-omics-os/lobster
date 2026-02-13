"""
Unit tests for todo_tools.

Tests the todo planning tools that implement LangGraph Command pattern:
- create_todo_tools factory
- write_todos tool (Command pattern with state updates)
- read_todos tool
- Validation logic for todo structure
"""

from unittest.mock import Mock

import pytest

from lobster.tools.todo_tools import create_todo_tools


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def todo_tools():
    """Create todo tools for testing."""
    write_todos, read_todos = create_todo_tools()
    return write_todos, read_todos


# =============================================================================
# Tests for create_todo_tools factory
# =============================================================================


class TestCreateTodoTools:
    """Test todo tools factory function."""

    def test_factory_creates_two_tools(self):
        """Verify factory creates both write_todos and read_todos tools."""
        write_todos, read_todos = create_todo_tools()

        # Verify both are callable tools
        assert hasattr(write_todos, "name")
        assert hasattr(write_todos, "func")
        assert write_todos.name == "write_todos"

        assert hasattr(read_todos, "name")
        assert hasattr(read_todos, "func")
        assert read_todos.name == "read_todos"

    def test_tools_have_docstrings(self, todo_tools):
        """Verify tools have proper documentation."""
        write_todos, read_todos = todo_tools

        assert write_todos.description is not None
        assert len(write_todos.description) > 0
        assert "todo" in write_todos.description.lower()

        assert read_todos.description is not None
        assert len(read_todos.description) > 0


# =============================================================================
# Tests for write_todos tool
# =============================================================================


class TestWriteTodos:
    """Test write_todos tool with various inputs."""

    def test_write_valid_todos(self, todo_tools):
        """Write valid todo list."""
        write_todos, _ = todo_tools

        todos = [
            {
                "content": "Task 1",
                "status": "completed",
                "activeForm": "Completing Task 1",
            },
            {
                "content": "Task 2",
                "status": "in_progress",
                "activeForm": "Working on Task 2",
            },
            {"content": "Task 3", "status": "pending", "activeForm": "Task 3 pending"},
        ]

        # Call tool with mocked tool_call_id
        result = write_todos.func(todos=todos, tool_call_id="test-123")

        # Verify Command structure
        assert hasattr(result, "update")
        assert "todos" in result.update
        assert "messages" in result.update
        assert result.update["todos"] == todos

        # Verify message content
        messages = result.update["messages"]
        assert len(messages) == 1
        message_content = messages[0].content

        assert "3 tasks" in message_content
        assert "1 completed" in message_content
        assert "1 in progress" in message_content
        assert "1 pending" in message_content

    def test_write_empty_todos(self, todo_tools):
        """Write empty todo list."""
        write_todos, _ = todo_tools

        result = write_todos.func(todos=[], tool_call_id="test-empty")

        assert hasattr(result, "update")
        assert result.update["todos"] == []

        message_content = result.update["messages"][0].content
        assert "0 tasks" in message_content

    def test_write_todos_missing_required_keys(self, todo_tools):
        """Write todos with missing required keys should return error."""
        write_todos, _ = todo_tools

        # Missing 'activeForm' key
        invalid_todos = [
            {"content": "Task 1", "status": "pending"}  # Missing activeForm
        ]

        result = write_todos.func(todos=invalid_todos, tool_call_id="test-invalid")

        # Should return error message
        message_content = result.update["messages"][0].content
        assert "Error" in message_content
        assert "missing required keys" in message_content
        assert "activeForm" in message_content

    def test_write_todos_invalid_status(self, todo_tools):
        """Write todos with invalid status should return error."""
        write_todos, _ = todo_tools

        invalid_todos = [
            {
                "content": "Task 1",
                "status": "invalid_status",  # Invalid status
                "activeForm": "Working on Task 1",
            }
        ]

        result = write_todos.func(todos=invalid_todos, tool_call_id="test-bad-status")

        # Should return error message
        message_content = result.update["messages"][0].content
        assert "Error" in message_content
        assert "Invalid status" in message_content
        assert "invalid_status" in message_content

    def test_write_todos_multiple_in_progress_warning(self, todo_tools):
        """Write todos with multiple in_progress should succeed but warn."""
        write_todos, _ = todo_tools

        todos = [
            {
                "content": "Task 1",
                "status": "in_progress",
                "activeForm": "Working on Task 1",
            },
            {
                "content": "Task 2",
                "status": "in_progress",
                "activeForm": "Working on Task 2",
            },
            {"content": "Task 3", "status": "pending", "activeForm": "Task 3 pending"},
        ]

        result = write_todos.func(todos=todos, tool_call_id="test-multi-progress")

        # Should succeed but include warning
        assert hasattr(result, "update")
        assert result.update["todos"] == todos

        message_content = result.update["messages"][0].content
        assert "Warning" in message_content
        assert "2 tasks marked as in_progress" in message_content

    def test_write_todos_all_statuses(self, todo_tools):
        """Test status counting with various combinations."""
        write_todos, _ = todo_tools

        todos = [
            {"content": "T1", "status": "completed", "activeForm": "T1 done"},
            {"content": "T2", "status": "completed", "activeForm": "T2 done"},
            {"content": "T3", "status": "in_progress", "activeForm": "T3 active"},
            {"content": "T4", "status": "pending", "activeForm": "T4 pending"},
            {"content": "T5", "status": "pending", "activeForm": "T5 pending"},
            {"content": "T6", "status": "pending", "activeForm": "T6 pending"},
        ]

        result = write_todos.func(todos=todos, tool_call_id="test-counts")

        message_content = result.update["messages"][0].content
        assert "6 tasks" in message_content
        assert "2 completed" in message_content
        assert "1 in progress" in message_content
        assert "3 pending" in message_content


# =============================================================================
# Tests for read_todos tool
# =============================================================================


class TestReadTodos:
    """Test read_todos tool."""

    def test_read_todos_returns_guidance(self, todo_tools):
        """Read todos returns guidance message."""
        _, read_todos = todo_tools

        result = read_todos.func(tool_call_id="test-read")

        # Verify Command structure
        assert hasattr(result, "update")
        assert "messages" in result.update

        # Verify message content
        message_content = result.update["messages"][0].content
        assert "todos" in message_content.lower()
        assert "state" in message_content.lower()

    def test_read_todos_command_structure(self, todo_tools):
        """Verify read_todos returns proper Command."""
        _, read_todos = todo_tools

        result = read_todos.func(tool_call_id="test-structure")

        # Should only update messages, not todos
        assert "messages" in result.update
        assert "todos" not in result.update
        assert len(result.update["messages"]) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestTodoToolsIntegration:
    """Integration tests for todo tools workflow."""

    def test_typical_workflow(self, todo_tools):
        """Test typical workflow: create plan, update, complete."""
        write_todos, read_todos = todo_tools

        # Step 1: Create initial plan
        initial_todos = [
            {"content": "Task 1", "status": "pending", "activeForm": "Task 1"},
            {"content": "Task 2", "status": "pending", "activeForm": "Task 2"},
            {"content": "Task 3", "status": "pending", "activeForm": "Task 3"},
        ]

        result1 = write_todos.func(todos=initial_todos, tool_call_id="step1")
        assert result1.update["todos"] == initial_todos
        assert "3 tasks" in result1.update["messages"][0].content
        assert "0 completed" in result1.update["messages"][0].content

        # Step 2: Start first task
        updated_todos = [
            {
                "content": "Task 1",
                "status": "in_progress",
                "activeForm": "Working on Task 1",
            },
            {"content": "Task 2", "status": "pending", "activeForm": "Task 2"},
            {"content": "Task 3", "status": "pending", "activeForm": "Task 3"},
        ]

        result2 = write_todos.func(todos=updated_todos, tool_call_id="step2")
        assert result2.update["todos"] == updated_todos
        assert "1 in progress" in result2.update["messages"][0].content

        # Step 3: Complete first, start second
        final_todos = [
            {"content": "Task 1", "status": "completed", "activeForm": "Task 1 done"},
            {
                "content": "Task 2",
                "status": "in_progress",
                "activeForm": "Working on Task 2",
            },
            {"content": "Task 3", "status": "pending", "activeForm": "Task 3"},
        ]

        result3 = write_todos.func(todos=final_todos, tool_call_id="step3")
        assert result3.update["todos"] == final_todos
        assert "1 completed" in result3.update["messages"][0].content
        assert "1 in progress" in result3.update["messages"][0].content

        # Step 4: Read todos for status check
        read_result = read_todos.func(tool_call_id="step4")
        assert "todos" in read_result.update["messages"][0].content.lower()

    def test_error_recovery_workflow(self, todo_tools):
        """Test workflow with error and recovery."""
        write_todos, _ = todo_tools

        # Attempt to write invalid todos
        invalid = [{"content": "Task", "status": "pending"}]  # Missing activeForm

        error_result = write_todos.func(todos=invalid, tool_call_id="error")
        assert "Error" in error_result.update["messages"][0].content

        # Recover with valid todos
        valid = [{"content": "Task", "status": "pending", "activeForm": "Pending Task"}]

        success_result = write_todos.func(todos=valid, tool_call_id="recovery")
        assert success_result.update["todos"] == valid
        assert "Error" not in success_result.update["messages"][0].content
