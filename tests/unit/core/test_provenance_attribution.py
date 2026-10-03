"""Provenance attribution, ordering and DAG edges.

Tests cover actor attribution, sequence ordering, input/output edges, and compatibility
with records that omit newer optional fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.provenance.provenance import ProvenanceTracker

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ir(inputs=("adata",), outputs=("adata",), **kw) -> AnalysisStep:
    return AnalysisStep(
        operation="scanpy.tl.pca",
        tool_name="run_pca",
        description="d",
        library="scanpy",
        code_template="pass",
        imports=[],
        parameters={},
        parameter_schema={},
        input_entities=list(inputs),
        output_entities=list(outputs),
        **kw,
    )


@pytest.fixture
def tracker(tmp_path) -> ProvenanceTracker:
    return ProvenanceTracker(session_dir=tmp_path)


class TestAgentAttribution:
    def test_agent_is_recorded_as_given(self, tracker):
        tracker.create_activity(
            activity_type="run_pca", agent="transcriptomics_expert", parameters={}
        )
        assert tracker.activities[-1]["agent"] == "transcriptomics_expert"

    def test_log_tool_usage_forwards_the_agent(self, tmp_path):
        """The gap: `create_activity` always accepted an agent; `log_tool_usage` could not
        forward one, so it passed a constant."""
        from lobster.core.runtime.data_manager import DataManagerV2

        dm = DataManagerV2(workspace_path=tmp_path)
        dm.log_tool_usage(
            tool_name="run_pca", parameters={}, agent="transcriptomics_expert"
        )
        assert dm.provenance.activities[-1]["agent"] == "transcriptomics_expert"

    def test_omitting_agent_keeps_the_historical_literal(self, tmp_path):
        """Omitting an agent preserves the default for existing callers."""
        from lobster.core.runtime.data_manager import DataManagerV2

        dm = DataManagerV2(workspace_path=tmp_path)
        dm.log_tool_usage(tool_name="run_pca", parameters={})
        assert dm.provenance.activities[-1]["agent"] == "data_manager"
        assert DataManagerV2.DEFAULT_PROVENANCE_AGENT == "data_manager"


class TestSequenceOrdering:
    def test_seq_is_total_and_gap_free(self, tracker):
        for i in range(5):
            tracker.create_activity(activity_type=f"t{i}", agent="a", parameters={})
        assert [a["seq"] for a in tracker.activities] == [0, 1, 2, 3, 4]

    def test_seq_orders_events_that_share_a_timestamp(self, tracker):
        """Sequence values provide a total order when timestamps are identical."""
        a = tracker.create_activity(activity_type="list", agent="x", parameters={})
        b = tracker.create_activity(activity_type="details", agent="x", parameters={})
        first, second = tracker.activities[-2], tracker.activities[-1]
        assert first["seq"] < second["seq"]
        assert a != b


class TestDagEdges:
    def test_ir_entities_are_lifted_to_inputs_outputs(self, tracker):
        tracker.create_activity(
            activity_type="get_modality_details",
            agent="data_expert_agent",
            parameters={},
            ir=_ir(inputs=("pbmc3k_pca",), outputs=("modality_info",)),
        )
        act = tracker.activities[-1]
        assert [e["name"] for e in act["inputs"]] == ["pbmc3k_pca"]
        assert [e["name"] for e in act["outputs"]] == ["modality_info"]

    def test_explicit_arguments_win_over_the_ir(self, tracker):
        tracker.create_activity(
            activity_type="t",
            agent="x",
            inputs=[{"name": "explicit"}],
            parameters={},
            ir=_ir(inputs=("pbmc3k",), outputs=("out",)),
        )
        act = tracker.activities[-1]
        assert act["inputs"] == [{"name": "explicit"}]
        assert [e["name"] for e in act["outputs"]] == ["out"], "outputs still lifted"

    def test_no_ir_leaves_edges_empty(self, tracker):
        tracker.create_activity(activity_type="t", agent="x", parameters={})
        act = tracker.activities[-1]
        assert act["inputs"] == [] and act["outputs"] == []


class TestPlaceholderEntityResolution:
    """Resolve code-local IR placeholders to modality names when available.

    Keep unresolved placeholders marked as such; dropping them would hide incomplete edges.
    """

    def test_placeholder_is_replaced_by_the_real_modality(self, tracker):
        tracker.create_activity(
            activity_type="run_pca",
            agent="transcriptomics_expert",
            parameters={"modality_name": "pbmc3k_filtered_normalized"},
            ir=_ir(),
        )
        act = tracker.activities[-1]
        assert [e["name"] for e in act["inputs"]] == ["pbmc3k_filtered_normalized"]
        assert act["inputs"][0]["resolved"] is True
        assert act["inputs"][0]["ir_name"] == "adata", "keep what the IR claimed"

    def test_real_names_are_not_rewritten(self, tracker):
        tracker.create_activity(
            activity_type="remove_modality",
            agent="data_expert_agent",
            parameters={"modality_name": "pbmc3k_pca"},
            ir=_ir(inputs=("pbmc3k_pca",), outputs=()),
        )
        assert tracker.activities[-1]["inputs"] == [
            {"name": "pbmc3k_pca", "resolved": True}
        ]

    def test_unresolvable_placeholder_is_kept_and_flagged(self, tracker):
        """Dropping it would make the DAG look complete when it is not."""
        tracker.create_activity(
            activity_type="run_pca", agent="x", parameters={}, ir=_ir()
        )
        edge = tracker.activities[-1]["inputs"][0]
        assert edge == {"name": "adata", "resolved": False}

    def test_a_consumer_can_tell_resolved_from_declared(self, tracker):
        """The whole point of the flag: a DAG builder must know which edges to trust."""
        tracker.create_activity(
            activity_type="run_pca", agent="x", parameters={}, ir=_ir()
        )
        tracker.create_activity(
            activity_type="run_pca",
            agent="x",
            parameters={"modality_name": "pbmc3k"},
            ir=_ir(),
        )
        flags = [a["inputs"][0]["resolved"] for a in tracker.activities[-2:]]
        assert flags == [False, True]


class TestPersistedShape:
    def test_new_fields_survive_a_round_trip(self, tmp_path):
        pt = ProvenanceTracker(session_dir=tmp_path)
        pt.create_activity(
            activity_type="run_pca",
            agent="transcriptomics_expert",
            parameters={"modality_name": "pbmc3k"},
            ir=_ir(),
        )
        line = json.loads((tmp_path / "provenance.jsonl").read_text().splitlines()[0])
        assert line["agent"] == "transcriptomics_expert"
        assert line["seq"] == 0
        assert line["inputs"][0]["name"] == "pbmc3k"


class TestWiredToolCoverage:
    """Wired tools name their executing agent as a first-class provenance field.

    A missing agent uses the visible ``DEFAULT_PROVENANCE_AGENT`` marker,
    rather than a guessed identity that would read as recorded fact.
    """

    @pytest.mark.parametrize(
        "rel_path,expected_min",
        [
            ("lobster/tools/custom_code_tool.py", 1),
            ("lobster/tools/workspace_tool.py", 4),
            (
                "packages/lobster-transcriptomics/lobster/agents/transcriptomics/shared_tools.py",
                5,
            ),
        ],
    )
    def test_module_passes_agent_at_every_log_site(self, rel_path, expected_min):
        import ast

        path = REPO_ROOT / rel_path
        tree = ast.parse(path.read_text())
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and (getattr(n.func, "attr", None) or getattr(n.func, "id", None))
            == "log_tool_usage"
        ]
        assert (
            len(calls) >= expected_min
        ), f"{rel_path}: expected >= {expected_min} sites"
        missing = [
            c.lineno
            for c in calls
            if "agent" not in {kw.arg for kw in c.keywords if kw.arg}
        ]
        assert not missing, f"{rel_path}: no agent= at line(s) {missing}"

    def test_shared_workspace_factories_default_to_unattributed(self):
        """They must NOT default to a plausible name.

        `create_list_modalities_tool` and friends are called by five different agents
        (supervisor, data_expert_agent, research_agent, metadata_assistant,
        feature_selection_expert). Any default would be wrong for four of them — attribution
        that reads as fact while being false, which is worse than a visible default.
        """
        import inspect

        from lobster.tools.workspace_tool import (
            create_delete_from_workspace_tool,
            create_get_content_from_workspace_tool,
            create_list_modalities_tool,
        )

        for factory in (
            create_list_modalities_tool,
            create_get_content_from_workspace_tool,
            create_delete_from_workspace_tool,
        ):
            default = inspect.signature(factory).parameters["agent_name"].default
            assert default is None, (
                f"{factory.__name__} defaults agent_name to {default!r}; it is called by "
                f"five different agents, so any default is wrong for four of them"
            )


class TestRunIdInTelemetry:
    """`index` gives order; `run_id` gives nesting and a cross-stream join key.

    A counter cannot express that a tool call happened *inside* a particular agent's run. The
    callback layer already computed `run_id`/`parent_run_id` (it maintains `run_to_agent` and
    `current_run_id`) and dropped them before telemetry.
    """

    def _recorder(self, tmp_path):
        from lobster.core.governance.routing_telemetry import RoutingTelemetryRecorder

        return RoutingTelemetryRecorder(session_dir=tmp_path, data_manager=None)

    def test_run_id_and_parent_are_recorded(self, tmp_path):
        rec = self._recorder(tmp_path)
        rec.record_tool_invocation(
            tool_name="run_pca",
            current_agent="transcriptomics_expert",
            run_id="child-1",
            parent_run_id="parent-1",
        )
        event = rec.events[-1].to_dict()
        assert event["run_id"] == "child-1"
        assert event["parent_run_id"] == "parent-1"

    def test_run_id_is_optional(self, tmp_path):
        """Nothing may depend on run-ID metadata being present."""
        rec = self._recorder(tmp_path)
        rec.record_tool_invocation(tool_name="run_pca", current_agent="x")
        event = rec.events[-1].to_dict()
        assert event["run_id"] is None and event["parent_run_id"] is None

    def test_ids_are_stringified(self, tmp_path):
        """LangChain passes UUID objects; the stream is JSON."""
        import uuid

        rec = self._recorder(tmp_path)
        rid = uuid.uuid4()
        rec.record_tool_invocation(tool_name="t", current_agent="x", run_id=rid)
        assert rec.events[-1].run_id == str(rid)

    def test_nesting_is_expressible(self, tmp_path):
        """The capability a bare counter lacks: child calls point at their parent."""
        rec = self._recorder(tmp_path)
        rec.record_tool_invocation(
            tool_name="handoff_to_transcriptomics_expert",
            current_agent="supervisor",
            run_id="A",
        )
        rec.record_tool_invocation(
            tool_name="run_pca",
            current_agent="transcriptomics_expert",
            run_id="B",
            parent_run_id="A",
        )
        parent, child = rec.events[-2], rec.events[-1]
        assert child.parent_run_id == parent.run_id


class TestArtifactEdgeResolution:
    """Output edges must identify the artifact produced by each activity.

    Analysis tools may choose output names that differ from their input names, so their
    recorded edges must use the stored result name rather than a placeholder.
    """

    def test_output_resolves_to_the_stored_artifact(self, tracker):
        tracker.create_activity(
            activity_type="run_pca",
            agent="transcriptomics_expert",
            parameters={
                "modality_name": "pbmc3k_filtered_normalized",
                "result_modality_name": "pbmc3k_filtered_normalized_pca",
            },
            ir=_ir(),
        )
        act = tracker.activities[-1]
        assert [e["name"] for e in act["inputs"]] == ["pbmc3k_filtered_normalized"]
        assert [e["name"] for e in act["outputs"]] == [
            "pbmc3k_filtered_normalized_pca"
        ], "output must be the artifact, not the input"

    def test_no_self_loop(self, tracker):
        """The precise defect: input and output resolved to the same node."""
        tracker.create_activity(
            activity_type="run_pca",
            agent="x",
            parameters={"modality_name": "m", "result_modality_name": "m_pca"},
            ir=_ir(),
        )
        act = tracker.activities[-1]
        assert act["inputs"][0]["name"] != act["outputs"][0]["name"]

    def test_missing_result_name_falls_back_visibly(self, tracker):
        """An unwired tool must not silently claim its input as its output."""
        tracker.create_activity(
            activity_type="run_pca",
            agent="x",
            parameters={"modality_name": "m"},
            ir=_ir(),
        )
        out = tracker.activities[-1]["outputs"][0]
        assert out == {"name": "adata", "resolved": False}, (
            "without a result name the placeholder must be RETAINED and flagged, not "
            "resolved to the input — otherwise the DAG grows a false edge"
        )

    def test_real_output_names_are_not_rewritten(self, tracker):
        tracker.create_activity(
            activity_type="get_modality_details",
            agent="x",
            parameters={"modality_name": "m", "result_modality_name": "ignored"},
            ir=_ir(inputs=("m",), outputs=("modality_info",)),
        )
        assert tracker.activities[-1]["outputs"] == [
            {"name": "modality_info", "resolved": True}
        ]

    def test_a_chain_is_walkable(self, tracker):
        """An artifact from one step can be the input of the next."""
        tracker.create_activity(
            activity_type="filter_and_normalize",
            agent="transcriptomics_expert",
            parameters={"modality_name": "pbmc3k", "result_modality_name": "pbmc3k_fn"},
            ir=_ir(),
        )
        tracker.create_activity(
            activity_type="run_pca",
            agent="transcriptomics_expert",
            parameters={
                "modality_name": "pbmc3k_fn",
                "result_modality_name": "pbmc3k_fn_pca",
            },
            ir=_ir(),
        )
        a, b = tracker.activities[-2], tracker.activities[-1]
        assert a["outputs"][0]["name"] == b["inputs"][0]["name"] == "pbmc3k_fn"
        assert b["outputs"][0]["name"] == "pbmc3k_fn_pca"


class TestAnalysisToolCoverage:
    """Listed analysis tools must identify themselves in provenance.

    An incorrect attribution is recorded as fact, so checks focus on audited tool
    factories.
    """

    ANALYSIS_TOOLS = {
        "execute_custom_code",
        "run_pca",
        "list_available_modalities",
        "get_modality_details",
        "assess_data_quality",
        "filter_and_normalize",
        "remove_modality",
        "select_variable_features",
    }

    def _wired_tool_names(self) -> set[str]:
        import ast

        names: set[str] = set()
        for rel in (
            "lobster/tools/custom_code_tool.py",
            "lobster/tools/workspace_tool.py",
            "packages/lobster-transcriptomics/lobster/agents/transcriptomics/shared_tools.py",
            "packages/lobster-research/lobster/agents/data_expert/data_expert.py",
        ):
            tree = ast.parse((REPO_ROOT / rel).read_text())
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and (
                        getattr(node.func, "attr", None)
                        or getattr(node.func, "id", None)
                    )
                    == "log_tool_usage"
                ):
                    continue
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                if "agent" in kw and isinstance(kw.get("tool_name"), ast.Constant):
                    names.add(kw["tool_name"].value)
        return names

    def test_analysis_tools_are_wired(self):
        wired = self._wired_tool_names()
        missing = sorted(self.ANALYSIS_TOOLS - wired)
        assert not missing, (
            f"analysis tools missing provenance attribution: {missing}. "
            f"Each one would leave its activity without an identifiable agent."
        )
