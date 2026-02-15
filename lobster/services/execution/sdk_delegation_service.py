"""
SDK Delegation Service for complex reasoning tasks.

This service integrates Claude Agent SDK to provide sub-agent reasoning
capabilities for Lobster agents. It allows agents to delegate complex
multi-step tasks to specialized Claude SDK agents with read-only access
to Lobster's data.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add Claude SDK to path if not installed
sdk_path = Path("/Users/tyo/GITHUB/omics-os/tmp_folder/claude-agent-sdk-python/src")
if sdk_path.exists() and str(sdk_path) not in sys.path:
    sys.path.insert(0, str(sdk_path))

try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        create_sdk_mcp_server,
    )
    from claude_agent_sdk import tool as sdk_tool

    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    sdk_tool = None
    create_sdk_mcp_server = None
    ClaudeSDKClient = None
    ClaudeAgentOptions = None

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class SDKDelegationError(Exception):
    """Raised when SDK delegation fails."""

    pass


class SDKDelegationService:
    """
    Service for delegating complex reasoning to Claude Agent SDK.

    This service creates a Claude SDK client with read-only Lobster tools,
    allowing sub-agents to inspect workspace data while performing complex
    multi-step reasoning.

    Design:
    - Read-only access to modalities (list, inspect, summarize)
    - Read-only access to workspace files
    - No write/modification capabilities (safe delegation)
    - Returns reasoning result as text
    """

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize SDK delegation service.

        Args:
            data_manager: DataManagerV2 instance for workspace access

        Raises:
            SDKDelegationError: If Claude Agent SDK is not available
        """
        if not SDK_AVAILABLE:
            raise SDKDelegationError(
                "Claude Agent SDK not available. "
                f"Please ensure it's installed or available at {sdk_path}"
            )

        self.data_manager = data_manager
        self.workspace_path = data_manager.workspace_path
        logger.debug("Initialized SDKDelegationService")

    def create_lobster_sdk_tools(self):
        """
        Create SDK tools that have read-only access to Lobster data.

        These tools are exposed to the Claude SDK sub-agent and provide
        inspection capabilities via closure access to data_manager.

        Returns:
            List of SDK tool functions
        """

        @sdk_tool("list_modalities", "List available datasets", {})
        async def list_modalities_tool(args):
            """List all available modalities in the workspace."""
            try:
                modalities = self.data_manager.list_modalities()
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Available modalities ({len(modalities)}): {modalities}",
                        }
                    ]
                }
            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Error listing modalities: {e}"}
                    ],
                    "is_error": True,
                }

        @sdk_tool(
            "get_modality_info",
            "Get detailed information about a specific dataset",
            {"modality_name": str},
        )
        async def get_modality_info_tool(args):
            """Get summary information about a modality."""
            try:
                modality_name = args["modality_name"]

                if modality_name not in self.data_manager.list_modalities():
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Modality '{modality_name}' not found. "
                                f"Available: {self.data_manager.list_modalities()}",
                            }
                        ],
                        "is_error": True,
                    }

                adata = self.data_manager.get_modality(modality_name)

                info = {
                    "name": modality_name,
                    "shape": f"{adata.n_obs} observations × {adata.n_vars} variables",
                    "obs_columns": list(adata.obs.columns)[:10],  # First 10
                    "var_columns": list(adata.var.columns)[:10],  # First 10
                    "layers": list(adata.layers.keys()) if adata.layers else [],
                    "obsm_keys": list(adata.obsm.keys()) if adata.obsm else [],
                    "uns_keys": list(adata.uns.keys()) if adata.uns else [],
                }

                # Add quality metrics if available
                try:
                    metrics = self.data_manager.get_quality_metrics(modality_name)
                    if metrics:
                        info["quality_metrics"] = {
                            k: v
                            for k, v in list(metrics.items())[:5]  # First 5 metrics
                        }
                except Exception:
                    pass

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Modality Info:\n{self._format_dict(info)}",
                        }
                    ]
                }
            except Exception as e:
                logger.error(f"Error getting modality info: {e}")
                return {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "is_error": True,
                }

        @sdk_tool("list_workspace_files", "List files in workspace", {})
        async def list_workspace_files_tool(args):
            """List CSV and JSON files in workspace."""
            try:
                csv_files = list(self.workspace_path.glob("*.csv"))
                json_files = list(self.workspace_path.glob("*.json"))
                jsonl_files = list(self.workspace_path.glob("*.jsonl"))

                file_list = {
                    "csv_files": [f.name for f in csv_files],
                    "json_files": [
                        f.name for f in json_files if not f.name.startswith(".")
                    ],
                    "jsonl_files": [f.name for f in jsonl_files],
                }

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Workspace Files:\n{self._format_dict(file_list)}",
                        }
                    ]
                }
            except Exception as e:
                return {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "is_error": True,
                }

        return [list_modalities_tool, get_modality_info_tool, list_workspace_files_tool]

    async def _async_delegate(
        self, task: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Async implementation of SDK delegation.

        Args:
            task: Task description for sub-agent
            context: Additional context information

        Returns:
            Dictionary with reasoning_result and metadata
        """
        # Create SDK MCP server with Lobster tools
        lobster_tools_server = create_sdk_mcp_server(
            name="lobster", version="1.0.0", tools=self.create_lobster_sdk_tools()
        )

        # Configure SDK options
        options = ClaudeAgentOptions(
            mcp_servers={"lobster": lobster_tools_server},
            allowed_tools=[
                "mcp__lobster__list_modalities",
                "mcp__lobster__get_modality_info",
                "mcp__lobster__list_workspace_files",
            ],
            max_turns=15,  # Limit reasoning iterations
            model="claude-sonnet-4-5",  # Latest model
            cwd=str(self.workspace_path),
        )

        # Build full prompt
        full_prompt = f"""{task}

{f"Context: {context}" if context else ""}

You have access to Lobster workspace data through these tools:
- list_modalities: See available datasets
- get_modality_info: Get details about a specific dataset
- list_workspace_files: See available CSV/JSON files

Available modalities: {self.data_manager.list_modalities()}
"""

        # Execute reasoning task
        logger.info(f"Delegating to SDK: {task[:100]}...")
        response_parts = []

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(full_prompt)

                async for msg in client.receive_response():
                    # Extract text from AssistantMessage
                    if hasattr(msg, "content"):
                        for block in msg.content:
                            if hasattr(block, "text"):
                                response_parts.append(block.text)

            reasoning_result = "\n".join(response_parts)

            if not reasoning_result:
                reasoning_result = "No response received from SDK agent."

            logger.info(f"SDK delegation complete ({len(reasoning_result)} chars)")

            return {
                "reasoning_result": reasoning_result,
                "status": "completed",
                "response_length": len(reasoning_result),
            }

        except Exception as e:
            logger.error(f"SDK delegation failed: {e}")
            raise SDKDelegationError(f"SDK execution failed: {e}")

    def delegate(
        self,
        task: str,
        context: Optional[str] = None,
        persist: bool = False,
        description: str = "Complex reasoning delegation",
    ) -> Tuple[str, Dict[str, Any], AnalysisStep]:
        """
        Delegate complex reasoning task to Claude Agent SDK.

        Args:
            task: Task description for sub-agent
            context: Optional additional context
            persist: Whether to include in notebook export
            description: Human-readable description

        Returns:
            Tuple of (reasoning_result, stats, ir)

        Raises:
            SDKDelegationError: If delegation fails
        """
        try:
            # Run async SDK call in sync context
            result = asyncio.run(self._async_delegate(task, context))

            # Build stats
            stats = {
                "success": True,
                "response_length": result["response_length"],
                "task": task[:200],  # Truncate for stats
                "persisted": persist,
            }

            # Create IR for provenance
            ir = self._create_ir(
                task, description, result["reasoning_result"], persist, stats
            )

            return result["reasoning_result"], stats, ir

        except Exception as e:
            logger.error(f"Delegation failed: {e}")

            # Still return IR even on failure
            stats = {"success": False, "error": str(e), "persisted": persist}
            ir = self._create_ir(task, description, str(e), persist, stats)

            raise SDKDelegationError(f"Delegation failed: {e}")

    def _create_ir(
        self,
        task: str,
        description: str,
        result: str,
        persist: bool,
        stats: Dict[str, Any],
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for provenance tracking.

        Args:
            task: Original task description
            description: Human-readable description
            result: Reasoning result
            persist: Whether to include in notebook export
            stats: Execution statistics

        Returns:
            AnalysisStep object
        """
        parameter_schema = {
            "task": ParameterSpec(
                param_type="str",
                papermill_injectable=False,
                default_value=task,
                required=True,
                description="Task description for SDK agent",
            ),
            "description": ParameterSpec(
                param_type="str",
                papermill_injectable=False,
                default_value=description,
                required=False,
                description="Human-readable description",
            ),
        }

        # Code template shows what happened (not executable Python)
        code_template = f"""# Claude Agent SDK Delegation
# Task: {task[:100]}
#
# This operation delegated reasoning to a Claude SDK sub-agent
# with read-only access to Lobster workspace data.
#
# Result summary:
# {result[:500]}
"""

        return AnalysisStep(
            operation="sdk_delegation",
            tool_name="delegate_complex_reasoning",
            description=description,
            library="claude_agent_sdk",
            code_template=code_template,
            imports=[],  # No imports for delegation
            parameters={"task": task, "description": description},
            parameter_schema=parameter_schema,
            input_entities=[],
            output_entities=[],
            execution_context={
                "persist": persist,
                "success": stats.get("success", False),
                "response_length": stats.get("response_length", 0),
            },
            validates_on_export=False,  # Not executable code
            requires_validation=False,
            exportable=persist,  # Only include if persist=True
        )

    def _format_dict(self, d: Dict[str, Any], indent: int = 2) -> str:
        """Format dictionary for readable text output."""
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{' ' * indent}{key}:")
                lines.append(self._format_dict(value, indent + 2))
            elif isinstance(value, list):
                lines.append(
                    f"{' ' * indent}{key}: {value[:10]}"
                )  # Truncate long lists
            else:
                lines.append(f"{' ' * indent}{key}: {value}")
        return "\n".join(lines)
