"""The seam must not change what the Go TUI receives.

Each scenario replays a scripted model through the Go launcher's turn handler
and compares the protocol transcript and the conversation it leaves with the
baseline recorded before the seam existed (``baseline/*.launcher.json``).

Record a new baseline only on purpose:
``LOBSTER_UPDATE_SEAM_BASELINE=1 uv run pytest tests/equivalence -q``
"""

from __future__ import annotations

import json
import os

import pytest

from tests.equivalence.launcher_harness import (
    SCENARIOS,
    baseline_path,
    drive_launcher_turn,
    make_client,
    run_launcher_turn,
    workspace_files,
    workspace_fingerprint,
)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_launcher_transcript_matches_pre_seam_baseline(scenario, tmp_path, monkeypatch):
    client = make_client(tmp_path / "workspace", monkeypatch, scenario)
    observed = {
        "transcript": run_launcher_turn(client, scenario),
        "workspace": workspace_fingerprint(client),
    }

    path = baseline_path(scenario)
    if os.environ.get("LOBSTER_UPDATE_SEAM_BASELINE") == "1":
        path.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"baseline written: {path.name}")

    expected = json.loads(path.read_text(encoding="utf-8"))
    # tool_execution comes from LangChain callbacks on the tool thread, so its
    # position relative to streamed text is not deterministic today. Each
    # channel keeps its own order.
    for channel in ("stream", "tools"):
        observed_part = _channel(observed["transcript"], channel)
        expected_part = _channel(expected["transcript"], channel)
        assert [m["type"] for m in observed_part] == [
            m["type"] for m in expected_part
        ], f"{channel}: message types or their order changed"
        assert observed_part == expected_part, f"{channel}: payloads changed"
    assert observed["workspace"] == expected["workspace"]


def test_decision_answer_echoes_the_render_id(tmp_path, monkeypatch):
    from lobster.core.runs import runs_for

    client = make_client(tmp_path / "workspace", monkeypatch, "decision")
    bridge, _ = drive_launcher_turn(client, "decision")
    renders = [m["id"] for m in bridge.sent if m["type"] == "component_render"]
    answered = [m["id"] for m in bridge.answered]
    assert answered and answered == renders[: len(answered)]
    assert set(answered) <= set(runs_for(client).decisions)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_turn_leaves_the_same_files_and_status_every_time(
    scenario, tmp_path, monkeypatch
):
    """Live check, no stored baseline: workspace file names and return status."""
    outcomes = []
    for attempt in ("first", "second"):
        workspace = tmp_path / attempt / "workspace"
        client = make_client(workspace, monkeypatch, scenario)
        _, returned = drive_launcher_turn(client, scenario)
        outcomes.append((workspace_files(workspace), returned))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][1] is None
    assert f".lobster/sessions/fixture-{scenario}" in outcomes[0][0]


def _channel(transcript, channel):
    is_tool = lambda m: m["type"] == "tool_execution"  # noqa: E731
    return [m for m in transcript if is_tool(m) == (channel == "tools")]
