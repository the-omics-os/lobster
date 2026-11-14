"""Deterministic replay chat model for protocol fixtures and equivalence tests.

Set ``LOBSTER_TEST_REPLAY_FIXTURE`` (tests and fixtures only) to a ``lobster.replay/1`` JSON script and every
model the factory creates serves the script's responses in order instead of
calling a provider. A request beyond the script, or one made by a different
agent than the script expects, raises ``ReplayMismatchError`` so a changed
call sequence fails loudly instead of drifting.

Script shape::

    {
      "schema": "lobster.replay/1",
      "responses": [
        {"agent": "supervisor", "content": "Hello"},
        {"agent": "supervisor", "tool_calls": [
            {"name": "ask_user", "args": {"question": "..."}}]},
        {"error": "Simulated provider failure"},
        {"content": "slow", "delay_s": 5},
        {"content": "costed", "usage": {"input_tokens": 1000, "output_tokens": 100}}
      ]
    }

``agent`` is optional; when present it must equal the ``agent_name`` the
factory received. ``usage`` is optional; when present it becomes the scripted
message's ``usage_metadata``, when absent the provider reports no usage (so the
run's cost reads as incomplete, as for any provider that sends none). All models in one process share one cursor, because the
graph calls them in a single deterministic order.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

REPLAY_ENV = "LOBSTER_TEST_REPLAY_FIXTURE"
REPLAY_SCHEMA = "lobster.replay/1"


class ReplayMismatchError(RuntimeError):
    """The engine asked for a response the replay script does not hold."""


class _ReplayScript:
    """Shared, thread-safe cursor over one script file."""

    def __init__(self, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != REPLAY_SCHEMA:
            raise ReplayMismatchError(
                f"{path}: expected schema {REPLAY_SCHEMA!r}, got {raw.get('schema')!r}"
            )
        self.path = path
        self.responses: List[Dict[str, Any]] = list(raw.get("responses", []))
        self.cursor = 0
        self._lock = threading.Lock()

    def next(self, agent_name: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            if self.cursor >= len(self.responses):
                raise ReplayMismatchError(
                    f"{self.path.name}: request {self.cursor + 1} from "
                    f"{agent_name!r} is beyond the {len(self.responses)} scripted responses"
                )
            entry = self.responses[self.cursor]
            expected = entry.get("agent")
            if expected and agent_name and expected != agent_name:
                raise ReplayMismatchError(
                    f"{self.path.name}: request {self.cursor + 1} came from "
                    f"{agent_name!r}, script expects {expected!r}"
                )
            self.cursor += 1
            return entry


_scripts: Dict[str, _ReplayScript] = {}
_scripts_lock = threading.Lock()


def _script_for(path: str) -> _ReplayScript:
    with _scripts_lock:
        script = _scripts.get(path)
        if script is None:
            script = _ReplayScript(Path(path))
            _scripts[path] = script
            # stderr only: stdout belongs to the protocol in machine modes.
            print(
                f"Replay fixture active: answers come from {path}, not from a model.",
                file=sys.stderr,
            )
        return script


def reset_replay_scripts() -> None:
    """Forget loaded scripts so the next model starts at response one (tests)."""
    with _scripts_lock:
        _scripts.clear()


class ReplayChatModel(BaseChatModel):
    """Chat model that returns scripted ``AIMessage`` objects in order."""

    script_path: str
    agent_name: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "lobster-replay"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ReplayChatModel":
        # Tool schemas do not change scripted answers.
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        script = _script_for(self.script_path)
        entry = script.next(self.agent_name)
        delay = float(entry.get("delay_s", 0) or 0)
        if delay:
            time.sleep(delay)
        if entry.get("error"):
            raise RuntimeError(str(entry["error"]))
        tool_calls = [
            {
                "name": call["name"],
                "args": call.get("args", {}),
                "id": call.get("id") or f"replay_call_{script.cursor}_{index}",
                "type": "tool_call",
            }
            for index, call in enumerate(entry.get("tool_calls", []))
        ]
        usage = entry.get("usage")
        usage_metadata = (
            {
                "input_tokens": int(usage["input_tokens"]),
                "output_tokens": int(usage["output_tokens"]),
                "total_tokens": int(usage["input_tokens"])
                + int(usage["output_tokens"]),
            }
            if usage
            else None
        )
        message = AIMessage(
            content=entry.get("content", ""),
            tool_calls=tool_calls,
            id=f"replay-{script.cursor}",
            usage_metadata=usage_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def replay_model_from_env(
    agent_name: Optional[str] = None,
) -> Optional[ReplayChatModel]:
    """Return a replay model when ``LOBSTER_TEST_REPLAY_FIXTURE`` is set, else None."""
    path = os.environ.get(REPLAY_ENV)
    if not path:
        return None
    _script_for(path)  # fail at creation time on a missing or malformed script
    return ReplayChatModel(script_path=path, agent_name=agent_name)
