"""Tests for unsupported-claim detection.

The detector must distinguish an unsupported claim from an inconclusive check. It should
remain conservative when evidence is incomplete and avoid allowing missing inputs to appear
as a clean result.
"""

from __future__ import annotations

import json

import pytest

from lobster.agents.artifact_manifest import ArtifactDiff, render_manifest
from lobster.core.governance.claim_verification import (
    EVENT_CLAIM_CHECK,
    TURN_ANALYTICAL,
    TURN_CONVERSATIONAL,
    VERDICT_NO_CLAIM,
    VERDICT_SUPPORTED,
    VERDICT_UNDETERMINED,
    VERDICT_UNSUPPORTED,
    ClaimVerificationRecorder,
    classify_turn,
    find_completion_claim,
    read_manifests,
    summarize_claim_checks,
    verify_claim,
)

EMPTY_MANIFEST = render_manifest(ArtifactDiff())
FULL_MANIFEST = render_manifest(
    ArtifactDiff(created=["pbmc_qc (2700 obs x 1838 vars)"])
)

ANALYSIS_REQUEST = (
    "Run quality control on my single-cell data and filter low-quality cells"
)


class TestTheTargetCase:
    """The detector must flag an unsupported success claim."""

    def test_success_prose_with_empty_manifest_is_flagged(self):
        check = verify_claim(
            ANALYSIS_REQUEST,
            "Successfully calculated the QC metrics for your dataset.",
            [EMPTY_MANIFEST],
        )
        assert check.verdict == VERDICT_UNSUPPORTED
        assert check.is_flagged
        assert check.claim_excerpt

    def test_success_prose_with_artifacts_is_supported(self):
        check = verify_claim(
            ANALYSIS_REQUEST,
            "Quality control complete. 2500 cells remain after filtering.",
            [FULL_MANIFEST],
        )
        assert check.verdict == VERDICT_SUPPORTED
        assert not check.is_flagged

    @pytest.mark.parametrize(
        "prose",
        [
            "Successfully calculated the QC metrics.",
            "I have completed the clustering analysis.",
            "Normalization is now complete.",
            "The analysis has been generated as requested.",
            "I successfully filtered the cells.",
        ],
    )
    def test_recognises_common_completion_phrasings(self, prose):
        assert verify_claim(ANALYSIS_REQUEST, prose, [EMPTY_MANIFEST]).is_flagged


class TestFalsePositiveSuite:
    """The expensive error. Each of these must stay silent."""

    @pytest.mark.parametrize(
        "request_text",
        [
            "What is ambient RNA?",
            "what are highly variable genes",
            "Explain how leiden clustering works",
            "Why is normalization necessary?",
            "How do I load a GEO dataset?",
            "Which agents are available?",
            "What can you do?",
            "Hello",
            "hi there",
            "Thanks!",
            "Tell me about pseudobulk aggregation",
            "list the available modalities",
        ],
    )
    def test_conversational_turns_are_never_flagged(self, request_text):
        """Category A turns correctly write nothing.

        These are the primary false-positive risk: the response to a concept question
        can easily contain completion-shaped language.
        """
        check = verify_claim(
            request_text,
            "Successfully explained the concept. The analysis is complete.",
            [EMPTY_MANIFEST],
        )
        assert check.turn_type == TURN_CONVERSATIONAL
        assert check.verdict == VERDICT_NO_CLAIM
        assert not check.is_flagged

    @pytest.mark.parametrize(
        "prose",
        [
            "I could not complete the analysis: the data is not loaded.",
            "Unable to run clustering because scikit-survival is not installed.",
            "The analysis failed to complete due to a missing dependency.",
            "No data is available. Please upload a dataset first.",
            "I cannot compute QC metrics without a loaded modality.",
            "An error occurred: traceback shows a missing module.",
            "I will run the QC analysis once the data is loaded.",
        ],
    )
    def test_honest_failure_reports_are_never_flagged(self, prose):
        """A supervisor reporting failure is being truthful.

        An empty manifest is the CORRECT outcome here, so flagging it would punish
        exactly the behaviour we want.
        """
        check = verify_claim(ANALYSIS_REQUEST, prose, [EMPTY_MANIFEST])
        assert check.verdict == VERDICT_NO_CLAIM
        assert not check.is_flagged

    def test_neutral_summary_without_a_completion_claim_is_not_flagged(self):
        check = verify_claim(
            ANALYSIS_REQUEST,
            "Here is what I found in the workspace: one modality named pbmc.",
            [EMPTY_MANIFEST],
        )
        assert not check.is_flagged

    def test_empty_response_is_not_flagged(self):
        assert not verify_claim(ANALYSIS_REQUEST, "", [EMPTY_MANIFEST]).is_flagged

    def test_empty_request_abstains(self):
        """With no request we cannot establish an expectation, so abstain."""
        check = verify_claim(
            "", "Successfully calculated the metrics.", [EMPTY_MANIFEST]
        )
        assert not check.is_flagged


class TestUndeterminedIsNotClean:
    def test_absent_manifest_is_undetermined(self):
        """Not measured must never read as verified.

        This is the case where the  flag is off, and it must be visibly distinct
        from a verified-clean turn.
        """
        check = verify_claim(
            ANALYSIS_REQUEST, "Successfully calculated the QC metrics.", []
        )
        assert check.verdict == VERDICT_UNDETERMINED
        assert not check.is_flagged
        assert "manifest" in check.reason.lower()

    def test_undetermined_is_distinct_from_supported(self):
        absent = verify_claim(ANALYSIS_REQUEST, "Analysis is complete.", [])
        present = verify_claim(
            ANALYSIS_REQUEST, "Analysis is complete.", [FULL_MANIFEST]
        )
        assert absent.verdict != present.verdict

    def test_handoff_return_without_a_manifest_block_is_undetermined(self):
        check = verify_claim(
            ANALYSIS_REQUEST, "Analysis is complete.", ["Plain prose, no manifest."]
        )
        assert check.verdict == VERDICT_UNDETERMINED


class TestManifestParsing:
    def test_absent_is_distinct_from_empty(self):
        """The distinction the whole detector rests on."""
        assert read_manifests([]).present is False
        empty = read_manifests([EMPTY_MANIFEST])
        assert empty.present is True and empty.has_artifacts is False

    def test_detects_artifacts(self):
        state = read_manifests([FULL_MANIFEST])
        assert state.present and state.has_artifacts

    def test_artifacts_from_any_of_several_handoffs_count(self):
        state = read_manifests([EMPTY_MANIFEST, FULL_MANIFEST])
        assert state.has_artifacts

    def test_manifest_embedded_in_surrounding_prose_is_found(self):
        text = f"QC complete.\n\n{FULL_MANIFEST}\n\n[store_key=abc123]"
        assert read_manifests([text]).has_artifacts

    def test_excerpts_are_capped(self):
        big = render_manifest(
            ArtifactDiff(created=[f"m{i} (10 obs x 5 vars)" for i in range(50)])
        )
        for excerpt in read_manifests([big]).excerpts:
            assert len(excerpt) <= 300


class TestTurnClassification:
    def test_analysis_request_is_analytical(self):
        assert classify_turn(ANALYSIS_REQUEST) == TURN_ANALYTICAL

    def test_question_is_conversational(self):
        assert classify_turn("What is a UMAP?") == TURN_CONVERSATIONAL

    def test_claim_detection_checks_negation_first(self):
        """Negation must win over a completion phrase in the same sentence."""
        assert (
            find_completion_claim("I could not successfully calculate the metrics.")
            is None
        )


class TestRecording:
    def test_record_stores_the_evidence(self):
        recorder = ClaimVerificationRecorder()
        recorder.record(
            ANALYSIS_REQUEST,
            "Successfully calculated the QC metrics.",
            [EMPTY_MANIFEST],
            ["handoff_to_transcriptomics_expert"],
        )
        check = recorder.checks[0]
        assert check.request_excerpt
        assert check.claim_excerpt
        assert check.manifest_excerpts
        assert check.tool_sequence == ["handoff_to_transcriptomics_expert"]

    def test_record_never_raises(self):
        recorder = ClaimVerificationRecorder()
        assert recorder.record(None, None, None, None) is not None  # type: ignore[arg-type]

    def test_writes_into_the_shared_routing_stream(self, tmp_path):
        """One stream per session; claim checks share the routing event record."""
        recorder = ClaimVerificationRecorder(session_dir=tmp_path)
        recorder.record(ANALYSIS_REQUEST, "Analysis is complete.", [EMPTY_MANIFEST])
        path = recorder.flush()
        assert path is not None and path.name == "routing_events.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        assert records[0]["kind"] == EVENT_CLAIM_CHECK
        assert records[0]["v"] == 1

    def test_coexists_with_routing_telemetry_in_one_file(self, tmp_path):
        """A reader must get both event kinds from a single file."""
        from lobster.core.governance.routing_telemetry import RoutingTelemetryRecorder

        telemetry = RoutingTelemetryRecorder(session_dir=tmp_path)
        telemetry.record_tool_invocation("execute_custom_code", "supervisor")
        telemetry.flush()

        claims = ClaimVerificationRecorder(session_dir=tmp_path)
        claims.record(ANALYSIS_REQUEST, "Analysis is complete.", [EMPTY_MANIFEST])
        path = claims.flush()

        kinds = {
            json.loads(line)["kind"]
            for line in path.read_text().splitlines()
            if line.strip()
        }
        assert EVENT_CLAIM_CHECK in kinds
        assert "tool_call" in kinds

    def test_flush_without_session_dir_returns_none(self):
        recorder = ClaimVerificationRecorder()
        recorder.record(ANALYSIS_REQUEST, "Analysis is complete.", [EMPTY_MANIFEST])
        assert recorder.flush() is None

    def test_records_survive_reread(self, tmp_path):
        recorder = ClaimVerificationRecorder(session_dir=tmp_path)
        recorder.record(ANALYSIS_REQUEST, "Analysis is complete.", [EMPTY_MANIFEST])
        path = recorder.flush()
        assert summarize_claim_checks([path])["counts"][VERDICT_UNSUPPORTED] == 1


class TestBaseRate:
    def test_rate_over_verifiable_turns_only(self):
        """Conversational turns must not dilute the denominator.

        A rate diluted by chat traffic is useless for judging whether a reviewer agent
        earns its per-handoff cost.
        """
        recorder = ClaimVerificationRecorder()
        recorder.record(ANALYSIS_REQUEST, "Analysis is complete.", [EMPTY_MANIFEST])
        recorder.record(ANALYSIS_REQUEST, "Analysis is complete.", [FULL_MANIFEST])
        recorder.record("What is a UMAP?", "It is a projection.", [])
        rate = recorder.base_rate()
        assert rate["turns_total"] == 3
        assert rate["turns_verifiable"] == 2
        assert rate["unsupported_rate"] == pytest.approx(0.5)

    def test_no_verifiable_turns_reports_none_not_zero(self):
        """0% would read as 'no fabrication found' when nothing was checked."""
        recorder = ClaimVerificationRecorder()
        recorder.record("What is a UMAP?", "It is a projection.", [])
        assert recorder.base_rate()["unsupported_rate"] is None

    def test_flagged_list_exposes_the_corpus(self):
        recorder = ClaimVerificationRecorder()
        recorder.record(ANALYSIS_REQUEST, "Analysis is complete.", [EMPTY_MANIFEST])
        recorder.record(ANALYSIS_REQUEST, "Analysis is complete.", [FULL_MANIFEST])
        assert len(recorder.flagged) == 1


class TestAggregation:
    @staticmethod
    def _session(tmp_path, name, verdicts):
        path = tmp_path / name
        with path.open("w") as handle:
            for verdict in verdicts:
                handle.write(
                    json.dumps(
                        {
                            "kind": EVENT_CLAIM_CHECK,
                            "v": 1,
                            "verdict": verdict,
                            "claim_excerpt": "Analysis is complete.",
                            "request_excerpt": "run qc",
                            "tool_sequence": ["handoff_to_transcriptomics_expert"],
                        }
                    )
                    + "\n"
                )
        return path

    def test_aggregates_a_production_base_rate(self, tmp_path):
        paths = [
            self._session(
                tmp_path, "a.jsonl", [VERDICT_UNSUPPORTED, VERDICT_SUPPORTED]
            ),
            self._session(tmp_path, "b.jsonl", [VERDICT_SUPPORTED, VERDICT_SUPPORTED]),
        ]
        result = summarize_claim_checks(paths)
        assert result["sessions_with_checks"] == 2
        assert result["turns_verifiable"] == 4
        assert result["unsupported_rate"] == pytest.approx(0.25)

    def test_collects_flagged_examples_for_the_corpus(self, tmp_path):
        path = self._session(tmp_path, "a.jsonl", [VERDICT_UNSUPPORTED])
        examples = summarize_claim_checks([path])["flagged_examples"]
        assert len(examples) == 1
        assert examples[0]["claim_excerpt"]

    def test_all_undetermined_reports_undefined_and_explains_why(self, tmp_path):
        path = self._session(tmp_path, "a.jsonl", [VERDICT_UNDETERMINED] * 3)
        result = summarize_claim_checks([path])
        assert result["unsupported_rate"] is None
        assert "LOBSTER_HANDOFF_MANIFEST" in result["note"]

    def test_unreadable_file_does_not_stop_the_sweep(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not json")
        good = self._session(tmp_path, "good.jsonl", [VERDICT_UNSUPPORTED])
        assert summarize_claim_checks([bad, good])["sessions_with_checks"] == 1

    def test_ignores_other_event_kinds(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            json.dumps({"kind": "tool_call", "v": 1, "tool_name": "x"}) + "\n"
        )
        assert summarize_claim_checks([path])["sessions_with_checks"] == 0


class TestCallbackCapture:
    """Manifests appear only in handoff returns, so capture is load-bearing."""

    @staticmethod
    def _tracker():
        from lobster.utils.callbacks import TokenTrackingCallback

        return TokenTrackingCallback(session_id="t")

    def test_capture_is_off_by_default(self):
        assert self._tracker().handoff_returns is None

    def test_handoff_return_is_captured(self):
        tracker = self._tracker()
        tracker.handoff_returns = []
        tracker.current_tool = "handoff_to_transcriptomics_expert"
        tracker.on_tool_end(f"QC done.\n\n{FULL_MANIFEST}")
        assert len(tracker.handoff_returns) == 1
        assert read_manifests(tracker.handoff_returns).has_artifacts

    def test_non_handoff_tool_output_is_not_captured(self):
        """Only delegation returns carry manifests; capturing code output would add
        noise and could pull data values into the record."""
        tracker = self._tracker()
        tracker.handoff_returns = []
        tracker.current_tool = "execute_custom_code"
        tracker.on_tool_end("printed some dataframe rows")
        assert tracker.handoff_returns == []

    def test_capture_is_bounded(self):
        tracker = self._tracker()
        tracker.handoff_returns = []
        tracker.max_handoff_returns = 3
        for _ in range(10):
            tracker.current_tool = "handoff_to_x"
            tracker.on_tool_end(EMPTY_MANIFEST)
        assert len(tracker.handoff_returns) == 3

    def test_non_string_output_does_not_raise(self):
        tracker = self._tracker()
        tracker.handoff_returns = []
        tracker.current_tool = "handoff_to_x"
        tracker.on_tool_end(None)  # type: ignore[arg-type]
        tracker.on_tool_end({"unexpected": "shape"})  # type: ignore[arg-type]

    def test_current_tool_is_still_cleared(self):
        """The pre-existing responsibility of on_tool_end must not regress."""
        tracker = self._tracker()
        tracker.current_tool = "handoff_to_x"
        tracker.on_tool_end("out")
        assert tracker.current_tool is None


class TestClientWiring:
    @staticmethod
    def _client(tmp_path, monkeypatch, enabled=True):
        from unittest.mock import MagicMock, patch

        import lobster.core.client as client_module

        monkeypatch.setenv("LOBSTER_CLAIM_VERIFICATION", "1" if enabled else "0")
        with patch.object(
            client_module,
            "create_bioinformatics_graph",
            return_value=(MagicMock(), MagicMock()),
        ):
            return client_module.AgentClient(workspace_path=tmp_path / str(enabled))

    def test_disabled_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LOBSTER_CLAIM_VERIFICATION", raising=False)
        client = self._client(tmp_path, monkeypatch, enabled=False)
        assert client.claim_verification is None
        assert client.token_tracker.handoff_returns is None

    def test_enabled_attaches_recorder_and_capture(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        assert isinstance(client.claim_verification, ClaimVerificationRecorder)
        assert client.token_tracker.handoff_returns == []

    def test_end_to_end_flags_a_fabricated_claim(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        client = self._client(tmp_path, monkeypatch)

        def fake_run(_graph_input, _config):
            client.token_tracker.current_tool = "handoff_to_transcriptomics_expert"
            client.token_tracker.on_tool_end(f"Looked at data.\n\n{EMPTY_MANIFEST}")
            return {
                "success": True,
                "response": "Successfully calculated the QC metrics.",
            }

        with patch.object(client, "_run_query", side_effect=fake_run):
            result = client.query("Run quality control on my single-cell data")

        assert result["claim_check"] == VERDICT_UNSUPPORTED
        assert len(client.claim_verification.flagged) == 1

    def test_end_to_end_passes_a_supported_claim(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        client = self._client(tmp_path, monkeypatch)

        def fake_run(_graph_input, _config):
            client.token_tracker.current_tool = "handoff_to_transcriptomics_expert"
            client.token_tracker.on_tool_end(f"Done.\n\n{FULL_MANIFEST}")
            return {"success": True, "response": "Quality control complete."}

        with patch.object(client, "_run_query", side_effect=fake_run):
            result = client.query("Run quality control on my single-cell data")

        assert result["claim_check"] == VERDICT_SUPPORTED
        assert client.claim_verification.flagged == []

    def test_capture_is_reset_between_turns(self, tmp_path, monkeypatch):
        """Turn N's artifacts must not support turn N+1's claim.

        Without the reset, a fabricated claim in a later turn would look verified
        because an earlier turn happened to write something.
        """
        from unittest.mock import patch

        client = self._client(tmp_path, monkeypatch)

        def writing_turn(_gi, _cfg):
            client.token_tracker.current_tool = "handoff_to_transcriptomics_expert"
            client.token_tracker.on_tool_end(f"Done.\n\n{FULL_MANIFEST}")
            return {"success": True, "response": "Quality control complete."}

        def non_writing_turn(_gi, _cfg):
            client.token_tracker.current_tool = "handoff_to_transcriptomics_expert"
            client.token_tracker.on_tool_end(f"Nothing to do.\n\n{EMPTY_MANIFEST}")
            return {"success": True, "response": "Successfully calculated the metrics."}

        with patch.object(client, "_run_query", side_effect=writing_turn):
            first = client.query("Run quality control on my single-cell data")
        with patch.object(client, "_run_query", side_effect=non_writing_turn):
            second = client.query("Now normalize the counts and scale the data")

        assert first["claim_check"] == VERDICT_SUPPORTED
        assert second["claim_check"] == VERDICT_UNSUPPORTED

    def test_verification_failure_does_not_break_the_query(self, tmp_path, monkeypatch):
        """A detection bug must never affect the response the user receives."""
        from unittest.mock import patch

        client = self._client(tmp_path, monkeypatch)
        client.claim_verification.record = lambda **_kw: (_ for _ in ()).throw(
            RuntimeError("boom")
        )

        with patch.object(
            client,
            "_run_query",
            side_effect=lambda _g, _c: {"success": True, "response": "Done."},
        ):
            result = client.query("Run quality control")

        assert result["success"] is True
        assert "claim_check" not in result

    def test_records_persist_to_the_session_directory(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        client = self._client(tmp_path, monkeypatch)

        def fake_run(_gi, _cfg):
            client.token_tracker.current_tool = "handoff_to_transcriptomics_expert"
            client.token_tracker.on_tool_end(f"x\n\n{EMPTY_MANIFEST}")
            return {"success": True, "response": "Analysis is complete."}

        with patch.object(client, "_run_query", side_effect=fake_run):
            client.query("Run quality control on my single-cell data")

        path = client.data_manager.provenance.session_dir / "routing_events.jsonl"
        assert path.exists()
        kinds = {
            json.loads(line)["kind"]
            for line in path.read_text().splitlines()
            if line.strip()
        }
        assert EVENT_CLAIM_CHECK in kinds


class TestClientCompletedClaimGuard:
    """Only completed textual non-stream results receive advisory checks."""

    @staticmethod
    def _client(result, enabled=True):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from lobster.core.client import AgentClient

        client = AgentClient.__new__(AgentClient)
        client.messages = []
        client.session_id = "claim-guard-test"
        client.callbacks = []
        client.token_tracker = SimpleNamespace(handoff_returns=[])
        client.routing_telemetry = None
        client.claim_verification = Mock() if enabled else None
        if enabled:
            client.claim_verification.record.return_value = SimpleNamespace(
                verdict="supported"
            )
        client._run_query = Mock(return_value=result)
        return client

    @pytest.mark.parametrize(
        "result",
        [
            {"success": False, "interrupts": [{"interrupt_id": "pause"}]},
            {"success": False, "error": "failed"},
            {"success": False, "response": "Incomplete"},
            {"success": True, "cancelled": True, "response": "Partial"},
            {
                "success": True,
                "interrupts": [{"interrupt_id": "pause"}],
                "response": "Partial",
            },
            {"success": True, "error": "failed", "response": "Partial"},
            {},
            {"response": "No completion status"},
            {"success": True},
            {"success": True, "response": None},
            {"success": True, "response": ""},
            {"success": True, "response": "  \n\t"},
            {"success": True, "response": ["not textual"]},
        ],
    )
    def test_nonfinal_result_is_not_recorded(self, result):
        original = dict(result)
        client = self._client(result)
        assert client.query("Analyze synthetic data") is result
        client.claim_verification.record.assert_not_called()
        client.claim_verification.flush.assert_not_called()
        assert result == original
        assert "claim_check" not in result

    def test_completed_text_is_checked_once(self):
        result = {"success": True, "response": "Analysis complete."}
        client = self._client(result)
        assert client.query("Analyze synthetic data") is result
        client.claim_verification.record.assert_called_once_with(
            user_request="Analyze synthetic data",
            response_text="Analysis complete.",
            handoff_returns=[],
            tool_sequence=[],
        )
        client.claim_verification.flush.assert_called_once_with()
        assert result["claim_check"] == "supported"

    def test_disabled_verifier_preserves_result(self):
        result = {"success": True, "response": "Analysis complete."}
        client = self._client(result, enabled=False)
        assert client.query("Analyze synthetic data") == result
        assert "claim_check" not in result

    @pytest.mark.parametrize("operation", ["record", "flush"])
    def test_verifier_error_is_fail_open(self, operation):
        result = {"success": True, "response": "Analysis complete."}
        original = dict(result)
        client = self._client(result)
        getattr(client.claim_verification, operation).side_effect = RuntimeError(
            "synthetic verifier failure"
        )
        assert client.query("Analyze synthetic data") is result
        assert result == original


class TestScopeIsHonest:
    def test_does_not_claim_to_catch_wrong_values(self):
        """Artifacts written but described incorrectly is OUT of scope.

        Pinning this prevents a later reader assuming the detector validates content.
        """
        check = verify_claim(
            ANALYSIS_REQUEST,
            "Successfully calculated the QC metrics: 9999 cells remain.",  # wrong number
            [FULL_MANIFEST],
        )
        assert check.verdict == VERDICT_SUPPORTED, (
            "an artifact exists, so this detector passes it — value correctness needs "
            "the reviewer agent"
        )
