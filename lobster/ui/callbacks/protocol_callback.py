"""Protocol callback handler for forwarding agent activity to Go TUI."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class ProtocolCallbackHandler(BaseCallbackHandler):
    """Emit tool and handoff telemetry through a callable emitter."""

    def __init__(self, emit_event):
        self.emit_event = emit_event
        self.current_agent: Optional[str] = None
        self.current_tool: Optional[str] = None
        self.start_times: Dict[str, datetime] = {}

    def _emit(self, msg_type: str, payload: Dict[str, Any]) -> None:
        try:
            self.emit_event(msg_type, payload)
        except Exception:
            logger.debug("Failed to emit protocol callback event", exc_info=True)

    def _extract_task_description(
        self, input_str: str, inputs: Dict[str, Any] | None = None
    ) -> str:
        if inputs and isinstance(inputs, dict):
            if "task_description" in inputs:
                return str(inputs["task_description"])
            if len(inputs) == 1:
                return str(next(iter(inputs.values()), ""))
        return input_str or ""

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        inputs: Dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        tool_name = (serialized or {}).get("name", "unknown_tool")
        self.current_tool = tool_name
        self.start_times[f"tool_{tool_name}"] = datetime.now()

        if tool_name.startswith("handoff_to_"):
            target = tool_name.replace("handoff_to_", "")
            task = self._extract_task_description(input_str, inputs)
            from_agent = self.current_agent or "supervisor"
            self.current_agent = target
            self._emit(
                "agent_transition",
                {"from": from_agent, "to": target, "reason": task},
            )
            return

        if tool_name.startswith("transfer_back_to_"):
            from_agent = self.current_agent or "unknown"
            self.current_agent = "supervisor"
            self._emit(
                "agent_transition",
                {"from": from_agent, "to": "supervisor", "reason": "return"},
            )
            return

        self._emit(
            "tool_execution",
            {
                "tool": tool_name,
                "agent": self.current_agent or "unknown",
                "status": "running",
            },
        )

    def on_tool_end(self, output: Any, **kwargs) -> None:
        tool_name = self.current_tool or "unknown_tool"
        duration_ms = None
        key = f"tool_{tool_name}"
        if key in self.start_times:
            duration_ms = (datetime.now() - self.start_times[key]).total_seconds() * 1000
            self.start_times.pop(key, None)

        self._emit(
            "tool_execution",
            {
                "tool": tool_name,
                "agent": self.current_agent or "unknown",
                "status": "complete",
                "duration_ms": duration_ms,
            },
        )
        self.current_tool = None

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        tool_name = self.current_tool or kwargs.get("name", "unknown_tool")
        self._emit(
            "tool_execution",
            {
                "tool": tool_name,
                "agent": self.current_agent or "unknown",
                "status": "error",
                "result": str(error)[:200],
            },
        )

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        return

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        return

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        chain_name = (serialized or {}).get("name", "").lower()
        if not chain_name:
            return

        # Best-effort agent detection from chain names.
        for candidate in (
            "supervisor",
            "research_agent",
            "data_expert",
            "transcriptomics_expert",
            "de_analysis_expert",
            "annotation_expert",
            "visualization_expert_agent",
            "genomics_expert",
            "proteomics_expert",
            "metabolomics_expert",
            "machine_learning_expert",
        ):
            if candidate in chain_name and candidate != self.current_agent:
                self._emit(
                    "agent_transition",
                    {
                        "from": self.current_agent or "supervisor",
                        "to": candidate,
                        "reason": "chain_start",
                    },
                )
                self.current_agent = candidate
                break
