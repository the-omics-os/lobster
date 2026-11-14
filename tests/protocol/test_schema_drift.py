"""The engine-owned protocol schema is canonical; the two hand-kept copies
(``lobster/ui/bridge/protocol.py`` and ``lobster-tui/internal/protocol/types.go``)
must stay inside it, and their known gaps are listed here on purpose."""

import json
import re

from lobster.ui.bridge.protocol import MessageType
from tests.protocol.stdio_driver import ROOT, SCHEMAS

SCHEMA = json.loads((SCHEMAS / "protocol-v1.schema.json").read_text(encoding="utf-8"))
SCHEMA_TYPES = set(SCHEMA["properties"]["type"]["enum"])
GO_TYPES = set(
    re.findall(
        r'^\tType\w+\s*=\s*"(\w+)"',
        (ROOT / "lobster-tui/internal/protocol/types.go").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
)

# Types the stdio interface adds; the Go TUI ignores unknown types.
ENGINE_ONLY = {"run_state", "decision_answer"}
# Sent or handled as string literals in go_tui_launcher.py but absent from the
# Python MessageType enum (component_set_data: declared in Go, sent by nobody).
PYTHON_ENUM_GAP = {
    "component_render",
    "component_response",
    "completion_request",
    "completion_response",
    "component_set_data",
}


def test_python_enum_is_inside_schema():
    assert {m.value for m in MessageType} <= SCHEMA_TYPES


def test_go_types_are_inside_schema():
    assert GO_TYPES and GO_TYPES <= SCHEMA_TYPES


def test_schema_adds_only_engine_types_to_go():
    assert SCHEMA_TYPES - GO_TYPES == ENGINE_ONLY


def test_python_enum_gap_is_known():
    assert (
        SCHEMA_TYPES - ENGINE_ONLY - {m.value for m in MessageType} == PYTHON_ENUM_GAP
    )


def test_every_type_names_its_emitter():
    listed = {m["type"]: m for m in SCHEMA["x-messages"]}
    assert set(listed) == SCHEMA_TYPES
    assert all(m["direction"] and m["emitted_by"] for m in listed.values())
