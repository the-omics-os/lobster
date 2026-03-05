from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lobster.cli_internal.commands.output_adapter import ProtocolOutputAdapter
import lobster.cli_internal.commands.heavy.slash_commands as slash_commands


pytestmark = pytest.mark.integration

_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "slash_commands"
_UPDATE_GOLDENS = os.getenv("LOBSTER_UPDATE_GOLDENS", "").lower() in {"1", "true", "yes"}


class _DummyDataManager:
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.modalities = {"rna": object(), "atac": object()}


class _DummyClient:
    def __init__(self, workspace_path: Path):
        self.session_id = "sess_123"
        self.workspace_path = workspace_path
        self.data_manager = _DummyDataManager(workspace_path)
        self.provider_override = "openai"
        self.model_override = "gpt-4.1"

    def get_token_usage(self):
        return {
            "session_id": "sess_123",
            "total_input_tokens": 123,
            "total_output_tokens": 456,
            "total_tokens": 579,
            "total_cost_usd": 0.0123,
            "by_agent": {
                "research_agent": {
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "total_tokens": 300,
                    "cost_usd": 0.005,
                    "invocation_count": 2,
                },
                "data_expert": {
                    "input_tokens": 23,
                    "output_tokens": 256,
                    "total_tokens": 279,
                    "cost_usd": 0.0073,
                    "invocation_count": 1,
                },
            },
        }


@pytest.mark.parametrize(
    ("command", "golden_name"),
    [
        ("/help", "help.json"),
        ("/session", "session.json"),
        ("/status", "status.json"),
        ("/tokens", "tokens.json"),
    ],
)
def test_slash_command_protocol_golden_transcripts(command: str, golden_name: str, monkeypatch):
    try:
        import lobster.core.license_manager as license_manager

        monkeypatch.setattr(license_manager, "get_current_tier", lambda: "free")
    except Exception:
        pass

    workspace_path = Path("/tmp/lobster_ws")
    client = _DummyClient(workspace_path)
    events = []
    output = ProtocolOutputAdapter(lambda msg_type, payload: events.append({"type": msg_type, "payload": payload}))

    summary = slash_commands._execute_command(
        command,
        client,
        original_command=command,
        output=output,
    )

    actual = {"summary": summary, "events": events}
    golden_path = _GOLDEN_DIR / golden_name

    if _UPDATE_GOLDENS:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"updated golden file: {golden_path}")

    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    assert actual == expected
