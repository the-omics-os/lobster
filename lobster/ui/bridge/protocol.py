"""JSON-lines protocol message definitions for lobster-tui bridge."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


PROTOCOL_VERSION = 1


class MessageType(str, Enum):
    # Python -> Go
    HANDSHAKE = "handshake"
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    TABLE = "table"
    FORM = "form"
    CONFIRM = "confirm"
    SELECT = "select"
    PROGRESS = "progress"
    ALERT = "alert"
    SPINNER = "spinner"
    STATUS = "status"
    CLEAR = "clear"
    DONE = "done"
    AGENT_TRANSITION = "agent_transition"
    MODALITY_LOADED = "modality_loaded"
    TOOL_EXECUTION = "tool_execution"
    SUSPEND = "suspend"
    RESUME = "resume"
    READY = "ready"
    HEARTBEAT = "heartbeat"

    # Go -> Python
    INPUT = "input"
    FORM_RESPONSE = "form_response"
    CONFIRM_RESPONSE = "confirm_response"
    SELECT_RESPONSE = "select_response"
    CANCEL = "cancel"
    QUIT = "quit"
    RESIZE = "resize"
    SLASH_COMMAND = "slash_command"


@dataclass
class ProtocolMessage:
    """Wire-format message envelope."""

    type: str
    id: Optional[str] = None
    version: int = PROTOCOL_VERSION
    payload: Optional[dict[str, Any]] = None

    def to_json_line(self) -> str:
        body: dict[str, Any] = {
            "type": self.type,
            "version": self.version,
        }
        if self.id is not None:
            body["id"] = self.id
        if self.payload is not None:
            body["payload"] = self.payload
        return json.dumps(body, separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> "ProtocolMessage":
        raw = json.loads(line)
        return cls(
            type=raw.get("type", ""),
            id=raw.get("id"),
            version=raw.get("version", PROTOCOL_VERSION),
            payload=raw.get("payload"),
        )


def new_message(msg_type: str, payload: Optional[dict[str, Any]] = None) -> ProtocolMessage:
    return ProtocolMessage(type=msg_type, payload=payload)


def new_request(msg_type: str, payload: Optional[dict[str, Any]] = None) -> ProtocolMessage:
    return ProtocolMessage(type=msg_type, id=str(uuid.uuid4()), payload=payload)
