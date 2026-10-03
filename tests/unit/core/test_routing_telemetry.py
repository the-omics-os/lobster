"""Tests for routing telemetry.

The recorder must classify pre/post-handoff code calls accurately, report when it cannot
assess artifact changes, and remain bounded and fail-open so telemetry cannot break a
session.

Deterministic; no model calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from lobster.core.governance.routing_telemetry import (
    CODE_EXEC_TOOL,
    EVENT_SESSION_SUMMARY,
    RoutingTelemetryRecorder,
    summarize_sessions,
)


def recorder(modality_sequence=None, **kwargs):
    """Recorder with a scripted DataManagerV2 stand-in."""
    dm = None
    if modality_sequence is not None:
        dm = MagicMock()
        dm.list_modality_records.side_effect = [
            [{"name": n} for n in names] for names in modality_sequence
        ]
    return RoutingTelemetryRecorder(data_manager=dm, **kwargs)


class TestThreeWaySplit:
    def test_code_after_handoff_is_post(self):
        rec = recorder()
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        split = rec.three_way_split()
        assert split["code_calls_post_handoff"] == 1
        assert split["code_calls_pre_handoff"] == 0

    def test_code_before_any_handoff_is_pre(self):
        rec = recorder()
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        split = rec.three_way_split()
        assert split["code_calls_pre_handoff"] == 1
        assert split["code_calls_post_handoff"] == 0

    def test_interleaved_sequence_splits_exactly(self):
        rec = recorder()
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        rec.record_tool_invocation("handoff_to_research_agent", "supervisor")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        split = rec.three_way_split()
        assert split["code_calls_pre_handoff"] == 1
        assert split["code_calls_post_handoff"] == 2
        assert split["code_calls_total"] == 3

    def test_records_which_agent_preceded_the_code_call(self):
        """Needed so an offline analysis with ground truth can judge mis-routes."""
        rec = recorder()
        rec.record_tool_invocation("handoff_to_data_expert_agent", "supervisor")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        code_event = next(e for e in rec.events if e.tool_name == CODE_EXEC_TOOL)
        assert code_event.preceding_handoff_agent == "data_expert_agent"

    def test_handoff_destinations_recorded_in_order(self):
        rec = recorder()
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        rec.record_tool_invocation("handoff_to_research_agent", "supervisor")
        assert rec.three_way_split()["handoffs"] == [
            "transcriptomics_expert",
            "research_agent",
        ]

    def test_no_code_calls_is_zero(self):
        rec = recorder()
        rec.record_tool_invocation("handoff_to_research_agent", "supervisor")
        assert rec.three_way_split()["code_calls_total"] == 0


class TestArtifactDetection:
    def test_detects_artifact_written_between_handoff_and_code(self):
        """The sharpest signal available without intent inference: the specialist
        demonstrably produced something and the supervisor recomputed anyway."""
        rec = recorder(modality_sequence=[[], ["pbmc_qc"]])
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        assert rec.three_way_split()["post_handoff_with_artifact_present"] == 1

    def test_no_artifact_written_is_recorded_as_false(self):
        rec = recorder(modality_sequence=[["existing"], ["existing"]])
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        event = next(e for e in rec.events if e.tool_name == CODE_EXEC_TOOL)
        assert event.artifact_written_since_handoff is False
        assert rec.three_way_split()["post_handoff_with_artifact_present"] == 0

    def test_unmeasurable_is_none_not_false(self):
        """'Could not measure' must never read as 'nothing was written'.

        Collapsing those two would understate how often the supervisor recomputed work
        that had in fact been done.
        """
        rec = recorder()  # no data manager at all
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        event = next(e for e in rec.events if e.tool_name == CODE_EXEC_TOOL)
        assert event.artifact_written_since_handoff is None

    def test_data_manager_failure_is_none_not_false(self):
        dm = MagicMock()
        dm.list_modality_records.side_effect = RuntimeError("backend down")
        rec = RoutingTelemetryRecorder(data_manager=dm)
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        event = next(e for e in rec.events if e.tool_name == CODE_EXEC_TOOL)
        assert event.artifact_written_since_handoff is None


class TestAttributionHealth:
    def test_recorder_with_no_events_declares_itself_blind(self):
        """A blind recorder reports a clean system, inverting the conclusion."""
        health = recorder().attribution_health
        assert health["blind"] is True
        assert health["trustworthy"] is False

    def test_healthy_recorder_is_trustworthy(self):
        rec = recorder()
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        assert rec.attribution_health["trustworthy"] is True

    def test_supervisor_between_handoffs_is_not_a_conflict(self):
        """Between handoffs the supervisor is legitimately the actor.

        Flagging that as disagreement would make every healthy session look broken.
        """
        rec = recorder()
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        rec.record_tool_invocation("list_available_modalities", "supervisor")
        assert rec.attribution_health["attribution_conflicts"] == 0

    def test_genuinely_conflicting_signals_are_counted(self):
        rec = recorder()
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        # metadata says a different specialist than the handoff chain
        rec.record_tool_invocation("run_pca", "proteomics_expert")
        assert rec.attribution_health["attribution_conflicts"] == 1

    def test_persistent_disagreement_marks_recorder_untrustworthy(self):
        rec = recorder()
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        for _ in range(10):
            rec.record_tool_invocation("run_pca", "proteomics_expert")
        health = rec.attribution_health
        assert health["conflict_rate"] > 0.25
        assert health["trustworthy"] is False


class TestFailOpenAndBounds:
    def test_recording_never_raises(self):
        rec = recorder()
        rec.record_tool_invocation(None, None)  # type: ignore[arg-type]
        rec.record_tool_invocation("", "")
        assert len(rec.events) == 2

    def test_event_count_is_bounded(self):
        rec = recorder(max_events=10)
        for _ in range(50):
            rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        assert len(rec.events) == 10
        assert rec.attribution_health["truncated"] is True

    def test_flush_without_session_dir_returns_none(self):
        assert recorder().flush() is None

    def test_flush_failure_returns_none_rather_than_raising(self, tmp_path):
        target = tmp_path / "file_not_dir"
        target.write_text("blocker")
        rec = RoutingTelemetryRecorder(session_dir=target / "nested")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        assert rec.flush() is None


class TestPersistence:
    def test_writes_events_and_summary(self, tmp_path):
        rec = RoutingTelemetryRecorder(session_dir=tmp_path)
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        path = rec.flush()
        assert path is not None and path.exists()

        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        assert all("v" in r for r in records), "schema version on every record"
        summaries = [r for r in records if r.get("kind") == EVENT_SESSION_SUMMARY]
        assert len(summaries) == 1
        assert summaries[0]["split"]["code_calls_post_handoff"] == 1

    def test_lands_beside_provenance(self, tmp_path):
        rec = RoutingTelemetryRecorder(session_dir=tmp_path)
        rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        assert rec.flush().name == "routing_events.jsonl"

    def test_records_no_data_values(self, tmp_path):
        """Names, ordering and counts only — never payloads or user prose."""
        rec = RoutingTelemetryRecorder(session_dir=tmp_path)
        rec.record_tool_invocation("handoff_to_transcriptomics_expert", "supervisor")
        text = rec.flush().read_text()
        for leak in ("prompt", "response", "content", "message"):
            assert leak not in text.lower()


class TestRoutingDiversity:
    def test_reports_top_tool_share(self):
        """Nothing currently reports this, so 'not 100% one tool' was uncheckable."""
        rec = recorder()
        for _ in range(3):
            rec.record_tool_invocation(CODE_EXEC_TOOL, "supervisor")
        rec.record_tool_invocation("handoff_to_research_agent", "supervisor")
        diversity = rec.routing_diversity()
        assert diversity["distinct_tools"] == 2
        assert diversity["top_tool_share"] == pytest.approx(0.75)


class TestAggregation:
    @staticmethod
    def _write(tmp_path, name, pre, post, trustworthy=True):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "kind": EVENT_SESSION_SUMMARY,
                    "v": 1,
                    "split": {
                        "code_calls_total": pre + post,
                        "code_calls_pre_handoff": pre,
                        "code_calls_post_handoff": post,
                        "post_handoff_with_artifact_present": post,
                    },
                    "attribution_health": {"trustworthy": trustworthy},
                }
            )
            + "\n"
        )
        return path

    def test_computes_post_handoff_share(self, tmp_path):
        paths = [
            self._write(tmp_path, "a.jsonl", pre=1, post=3),
            self._write(tmp_path, "b.jsonl", pre=1, post=1),
        ]
        totals = summarize_sessions(paths)
        assert totals["sessions_read"] == 2
        assert totals["code_calls_post_handoff"] == 4
        # Shares are rounded to 4dp for a compact JSON record.
        assert totals["post_handoff_share"] == pytest.approx(4 / 6, abs=1e-4)

    def test_untrustworthy_sessions_are_excluded_and_counted(self, tmp_path):
        """Pooling a blind session would bias the result toward 'no escalation'."""
        paths = [
            self._write(tmp_path, "good.jsonl", pre=0, post=2),
            self._write(tmp_path, "blind.jsonl", pre=0, post=0, trustworthy=False),
        ]
        totals = summarize_sessions(paths)
        assert totals["sessions_read"] == 1
        assert totals["sessions_excluded_untrustworthy"] == 1
        assert totals["post_handoff_share"] == pytest.approx(1.0)

    def test_absent_code_use_is_undefined_not_zero_percent(self, tmp_path):
        """No code calls is a legitimate reading but says nothing about the split.

        Reporting 0% would look like evidence that post-handoff escalation does not
        happen, when in fact nothing was apportioned at all.
        """
        totals = summarize_sessions([self._write(tmp_path, "q.jsonl", pre=0, post=0)])
        assert totals["post_handoff_share"] is None
        assert "undefined" in totals["note"]

    def test_unreadable_file_does_not_stop_the_sweep(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not json")
        good = self._write(tmp_path, "good.jsonl", pre=0, post=1)
        assert summarize_sessions([bad, good])["sessions_read"] == 1

    def test_missing_file_is_skipped(self, tmp_path):
        assert summarize_sessions([tmp_path / "nope.jsonl"])["sessions_read"] == 0


class TestCallbackWiring:
    """The recorder is inert unless the callback feeds it."""

    def test_token_tracker_exposes_the_hook(self):
        from lobster.utils.callbacks import TokenTrackingCallback

        tracker = TokenTrackingCallback(session_id="t")
        assert hasattr(tracker, "routing_telemetry")
        assert tracker.routing_telemetry is None, "must be opt-in"

    def test_tool_calls_reach_the_recorder(self):
        from lobster.utils.callbacks import TokenTrackingCallback

        tracker = TokenTrackingCallback(session_id="t")
        rec = recorder()
        tracker.routing_telemetry = rec
        tracker.on_tool_start({"name": "handoff_to_transcriptomics_expert"}, "")
        tracker.on_tool_start({"name": CODE_EXEC_TOOL}, "")
        assert rec.three_way_split()["code_calls_post_handoff"] == 1

    def test_recorder_exception_does_not_break_tool_invocation(self):
        """Telemetry must never be able to fail a session."""
        from lobster.utils.callbacks import TokenTrackingCallback

        tracker = TokenTrackingCallback(session_id="t")
        exploding = MagicMock()
        exploding.record_tool_invocation.side_effect = RuntimeError("boom")
        tracker.routing_telemetry = exploding
        tracker.on_tool_start({"name": CODE_EXEC_TOOL}, "")  # must not raise


class TestClientWiring:
    """AgentClient must attach the recorder only when asked."""

    @staticmethod
    def _client(tmp_path, enabled, monkeypatch):
        from unittest.mock import patch

        import lobster.core.client as client_module

        monkeypatch.setenv("LOBSTER_ROUTING_TELEMETRY", "1" if enabled else "0")
        with patch.object(
            client_module,
            "create_bioinformatics_graph",
            return_value=(MagicMock(), MagicMock()),
        ):
            return client_module.AgentClient(workspace_path=tmp_path / str(enabled))

    def test_disabled_by_default(self, tmp_path, monkeypatch):
        """Opt-in: an unset flag must cost nothing."""
        monkeypatch.delenv("LOBSTER_ROUTING_TELEMETRY", raising=False)
        client = self._client(tmp_path, False, monkeypatch)
        assert client.routing_telemetry is None
        assert client.token_tracker.routing_telemetry is None

    def test_enabled_attaches_and_wires(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, True, monkeypatch)
        assert isinstance(client.routing_telemetry, RoutingTelemetryRecorder)
        assert client.token_tracker.routing_telemetry is client.routing_telemetry
        assert client.routing_telemetry._data_manager is client.data_manager

    def test_persists_into_the_session_directory(self, tmp_path, monkeypatch):
        """Records must land beside provenance.jsonl so they survive the session."""
        client = self._client(tmp_path, True, monkeypatch)
        client.token_tracker.on_tool_start(
            {"name": "handoff_to_transcriptomics_expert"}, ""
        )
        client.token_tracker.on_tool_start({"name": CODE_EXEC_TOOL}, "")
        path = client.routing_telemetry.flush()
        assert path is not None and path.exists()
        assert ".lobster/sessions" in str(path)

    def test_real_callback_path_produces_the_split(self, tmp_path, monkeypatch):
        """Drive the actual callback, not the recorder directly.

        Guards the wiring itself: if on_tool_start stopped feeding the recorder, unit
        tests on the recorder would still pass while production recorded nothing.
        """
        client = self._client(tmp_path, True, monkeypatch)
        for name in (
            "list_available_modalities",
            "handoff_to_transcriptomics_expert",
            CODE_EXEC_TOOL,
        ):
            client.token_tracker.on_tool_start({"name": name}, "")
        split = client.routing_telemetry.three_way_split()
        assert split["code_calls_post_handoff"] == 1
        assert split["handoffs"] == ["transcriptomics_expert"]
