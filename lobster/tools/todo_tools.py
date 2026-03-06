"""
Todo tools for planning multi-step tasks.

Implements the TodoListMiddleware pattern from LangChain/DeepAgents using
LangGraph's Command pattern for atomic state updates.

Following the pattern from:
- https://docs.langchain.com/oss/python/deepagents/overview
- langchain.agents.middleware.todo (TodoListMiddleware)
"""

from typing import Any, Dict, List

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from typing_extensions import Annotated

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def create_todo_tools():
    """Factory to create write_todos and read_todos tools using Command pattern.

    These tools enable the supervisor to plan and track multi-step tasks,
    preventing impulsive actions on complex requests.

    Returns:
        Tuple[Callable, Callable]: (write_todos, read_todos) tool functions
    """

    @tool
    def write_todos(
        todos: List[Dict[str, str]],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[Dict[str, Any]]:
        """Create or update todo list for multi-step tasks (3+ steps or multi-agent).

        Each item needs: content (imperative), status (pending/in_progress/completed),
        activeForm (present continuous). Keep exactly 1 task in_progress at a time.

        Args:
            todos: List of todo items with content, status, activeForm keys
        """
        # Validate todos structure
        valid_statuses = {"pending", "in_progress", "completed"}

        for i, todo in enumerate(todos):
            required_keys = {"content", "status", "activeForm"}
            if missing := required_keys - set(todo.keys()):
                error_msg = f"Error: Todo item {i} missing required keys: {missing}"
                logger.warning(error_msg)
                return Command(
                    update={
                        "messages": [
                            ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                        ]
                    }
                )

            if todo["status"] not in valid_statuses:
                error_msg = f"Error: Invalid status '{todo['status']}' in todo {i}. Must be one of: {valid_statuses}"
                logger.warning(error_msg)
                return Command(
                    update={
                        "messages": [
                            ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                        ]
                    }
                )

        # Count status distribution
        total = len(todos)
        completed = sum(1 for t in todos if t["status"] == "completed")
        in_progress = sum(1 for t in todos if t["status"] == "in_progress")
        pending = sum(1 for t in todos if t["status"] == "pending")

        # Warn if multiple tasks are in_progress (soft enforcement)
        warning = ""
        if in_progress > 1:
            warning = f" Warning: {in_progress} tasks marked as in_progress (recommend: 1 at a time)."
            logger.info(f"Multiple in_progress tasks: {in_progress}")

        # Format response
        response = (
            f"Todo list updated: {total} tasks "
            f"({completed} completed, {in_progress} in progress, {pending} pending).{warning}"
        )

        logger.debug(
            f"write_todos: {total} tasks, {completed} done, {in_progress} active"
        )

        # Return Command to update both todos and messages atomically
        # The _todo_reducer in state.py will handle the todos update
        return Command(
            update={
                "todos": todos,  # Triggers _todo_reducer (replace semantics)
                "messages": [ToolMessage(content=response, tool_call_id=tool_call_id)],
            }
        )

    @tool
    def read_todos(
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[Dict[str, Any]]:
        """Read current todo list to check planning status.

        Use this to review the current plan before proceeding or to remind
        yourself of pending tasks.

        Note: The actual todos are stored in state.todos and will be visible
        in your context. This tool provides a formatted view.

        Returns:
            Command with formatted todo list or guidance message
        """
        # Note: In the current implementation, the agent sees state.todos in context.
        # This tool serves as a reminder/check mechanism.
        # Future enhancement: Could read from state directly if passed via runtime.

        response = (
            "To see current todos, check the 'todos' field in your state. "
            "Use write_todos to create or update your plan. "
            "Current todos should be visible in your working context."
        )

        logger.debug("read_todos called")

        return Command(
            update={
                "messages": [ToolMessage(content=response, tool_call_id=tool_call_id)]
            }
        )

    # AQUADIF metadata (UTILITY — planning tools, no provenance)
    write_todos.metadata = {"categories": ["UTILITY"], "provenance": False}
    write_todos.tags = ["UTILITY"]
    read_todos.metadata = {"categories": ["UTILITY"], "provenance": False}
    read_todos.tags = ["UTILITY"]

    return write_todos, read_todos
