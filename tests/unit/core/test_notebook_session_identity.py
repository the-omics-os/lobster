"""Canonical client session identity in notebook export (F25).

The client that owns a session binds its id to the data manager; an export
reads that binding once and shows it in the integrity manifest, the hash input,
the notebook metadata and the provenance summary. Unknown is null, never the
provenance namespace, the data manager's timestamp or a directory name.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import nbformat
import pytest

from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2

SID_A = "session_20261030_101500"
SID_B = "3F2B8C1E-5d4a-4e7b-9a10-2c6d8e9f0a1b"
DIR_NAME = "dirname-not-identity"
UNKNOWN_SUMMARY = "Session ID: unknown (the client session identity was unavailable or ambiguous at export time)"
NEUTRAL_LINE = "- Activity records are tracked separately from the notebook; disk persistence is optional."


@pytest.fixture(autouse=True)
def _no_host_dependency_scan(monkeypatch):
    # Host package metadata is an external boundary; everything else is real.
    monkeypatch.setattr(NotebookExporter, "_snapshot_dependencies", lambda self: {})


def _ir() -> AnalysisStep:
    return AnalysisStep(
        operation="scanpy.pp.normalize_total",
        tool_name="normalize_data",
        description="Normalize counts",
        library="scanpy",
        code_template="sc.pp.normalize_total(adata, target_sum={{ target_sum }})",
        imports=["import scanpy as sc"],
        parameters={"target_sum": 1e4},
        parameter_schema={},
        input_entities=["adata"],
        output_entities=["adata"],
        execution_context={"agent": "transcriptomics_expert"},
        exportable=True,
    )


def _with_activity(dm):
    ir = _ir()
    dm.provenance.activities.append(
        {
            "id": f"{dm.provenance.namespace}:activity:normalize",
            "type": ir.tool_name,
            "agent": f"{dm.provenance.namespace}:agent:transcriptomics_expert",
            "timestamp": datetime.now().isoformat(),
            "inputs": [{"entity": "adata", "role": "input"}],
            "outputs": [{"entity": "adata", "role": "output"}],
            "parameters": ir.parameters,
            "description": ir.description,
            "ir": ir.to_dict(),
        }
    )
    return dm


def _dm(tmp_path: Path, name: str = "ws", namespace: str = "lobster") -> DataManagerV2:
    workspace = tmp_path / name
    workspace.mkdir(parents=True, exist_ok=True)
    dm = DataManagerV2(
        workspace_path=workspace,
        session_dir=workspace / ".lobster" / "sessions" / DIR_NAME,
    )
    if namespace != "lobster":
        dm.provenance.namespace = namespace
    return _with_activity(dm)


def _read(path: Path):
    nb = nbformat.read(path, as_version=4)
    manifest = next(
        c.source
        for c in nb.cells
        if c.cell_type == "markdown" and "data_integrity_manifest" in c.source
    )
    record = json.loads(manifest.split("```json\n", 1)[1].split("\n```", 1)[0])[
        "data_integrity_manifest"
    ]["provenance"]
    summary = next(
        c.source
        for c in nb.cells
        if c.cell_type == "markdown" and "## Provenance" in c.source
    )
    return record, nb.metadata["lobster"], summary, manifest


def _expected_hash(session_id, namespace, activities, entities) -> str:
    data = {
        "session_id": session_id,
        "namespace": namespace,
        "activities_count": activities,
        "entities_count": entities,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def _assert_identity(path: Path, dm, expected) -> None:
    record, meta, summary, _ = _read(path)
    ns = dm.provenance.namespace
    assert set(record) == {
        "session_id",
        "namespace",
        "sha256",
        "activities",
        "entities",
    }
    assert record["activities"] == len(dm.provenance.activities)
    assert record["entities"] == len(dm.provenance.entities)
    assert record["session_id"] == expected
    assert record["namespace"] == ns
    assert record["sha256"] == _expected_hash(
        expected, ns, len(dm.provenance.activities), len(dm.provenance.entities)
    )
    assert meta["source_session_id"] == expected
    assert meta["provenance_namespace"] == ns
    if expected is None:
        assert UNKNOWN_SUMMARY in summary
    else:
        assert f"- Session ID: `{expected}`" in summary


# ---------------------------------------------------------------- T1–T3, T6, T7


def test_t1_unbound_export_is_null_and_keeps_namespace(tmp_path):
    dm = _dm(tmp_path)
    path = dm.notebook_exporter.export("nb")
    _assert_identity(path, dm, None)


def test_t2_bound_export_names_the_client_session(tmp_path):
    dm = _dm(tmp_path)
    assert dm.bind_client_session(SID_A) == SID_A
    _assert_identity(dm.notebook_exporter.export("nb"), dm, SID_A)


def test_t3_identity_fields_never_take_namespace_timestamp_or_directory(tmp_path):
    dm = _dm(tmp_path)
    record, meta, summary, _ = _read(dm.notebook_exporter.export("nb"))
    for value in (record["session_id"], meta["source_session_id"]):
        assert value not in ("lobster", str(dm.session_id), DIR_NAME)
    assert "Session ID: `lobster`" not in summary
    assert (
        record["namespace"] == "lobster" and meta["provenance_namespace"] == "lobster"
    )


@pytest.mark.parametrize("sid", ["session_alpha", "session_beta"])
def test_t6_two_real_session_contexts_report_their_own_ids(tmp_path, sid):
    from lobster.cli_internal.commands.heavy.session_infra import CommandClient

    workspace = tmp_path / sid
    (workspace / ".lobster" / "sessions" / sid).mkdir(parents=True)
    client = CommandClient(workspace, session_id=sid)
    _with_activity(client.data_manager)
    _assert_identity(
        client.data_manager.notebook_exporter.export("nb"), client.data_manager, sid
    )


def test_t7_namespace_prefixes_and_tracker_namespace_unchanged(tmp_path):
    dm = _dm(tmp_path, namespace="study-ns")
    dm.bind_client_session(SID_A)
    activity = dm.provenance.create_activity(
        "normalize",
        "transcriptomics_expert",
        inputs=[],
        outputs=[],
        parameters={},
        description="d",
    )
    assert activity.startswith("study-ns:activity:")
    entity = dm.provenance.create_entity("dataset", uri=None)
    assert entity.startswith("study-ns:entity:")
    agent = dm.provenance.create_agent("transcriptomics_expert")
    assert agent.startswith("study-ns:agent:")
    assert dm.provenance.namespace == "study-ns"
    import anndata
    import numpy as np

    adata = dm.provenance.add_to_anndata(anndata.AnnData(np.zeros((2, 2))))
    assert adata.uns["provenance"]["tracker_namespace"] == "study-ns"
    _assert_identity(dm.notebook_exporter.export("nb"), dm, SID_A)


# ---------------------------------------------------------------- T4, T5 (bind table, allowlist)


def test_t4_bind_transitions(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path / "w")
    assert dm.client_session_id is None
    assert dm.bind_client_session(SID_A) == SID_A
    assert dm.bind_client_session(SID_A) == SID_A  # identical: no-op
    assert dm.bind_client_session(SID_B) is None  # conflict
    assert dm.bind_client_session(SID_A) is None  # sticky
    assert (
        dm.bind_client_session(SID_B, replace=True) == SID_B
    )  # the one sanctioned reset
    assert dm.bind_client_session(SID_A) is None  # conflict again
    assert dm.bind_client_session("bad value") is None  # conflict -> invalid
    assert dm.bind_client_session(SID_A) is None  # still sticky
    assert (
        dm.bind_client_session("bad value", replace=True) is None
    )  # invalid replace never clears
    assert dm.bind_client_session(SID_A) is None
    assert dm.bind_client_session(SID_A, replace=True) == SID_A
    with pytest.raises(AttributeError):
        dm.client_session_id = SID_B  # read-only property
    assert dm.client_session_id == SID_A


@pytest.mark.parametrize(
    "value",
    [
        None,
        7,
        b"session",
        "",
        "a`b",
        "a b",
        "A\n",
        "_study",
        ".study",
        "study\u03b1",
        "a" * 129,
        ["x"],
    ],
)
def test_t5_display_allowlist_rejects_without_raising(tmp_path, value):
    dm = _dm(tmp_path)
    assert dm.bind_client_session(value) is None
    _assert_identity(dm.notebook_exporter.export("nb"), dm, None)


@pytest.mark.parametrize(
    "value",
    ["a" * 128, SID_A, SID_B, SID_B.lower(), "lobster", "1730281234.56", "x:y.z-1"],
)
def test_t5_display_allowlist_accepts(tmp_path, value):
    dm = _dm(tmp_path)
    assert dm.bind_client_session(value) == value
    _assert_identity(dm.notebook_exporter.export("nb"), dm, value)


# ---------------------------------------------------------------- T8–T12 (exporter observation)


def test_t8_mock_data_manager_is_null(tmp_path):
    dm = MagicMock()
    dm.workspace_path = tmp_path / "w"
    dm.workspace_path.mkdir()
    dm.modalities = {"m": MagicMock()}
    real = _dm(tmp_path, "src")
    exporter = NotebookExporter(real.provenance, dm)
    record, meta, summary, _ = _read(exporter.export("nb"))
    assert (
        record["session_id"] is None
        and meta["source_session_id"] is None
        and UNKNOWN_SUMMARY in summary
    )


def test_t8_raising_identity_property_still_exports_null(tmp_path, caplog):
    dm = _dm(tmp_path)
    dm.bind_client_session(SID_A)
    with patch.object(
        DataManagerV2,
        "client_session_id",
        property(lambda self: (_ for _ in ()).throw(RuntimeError("boom-secret"))),
    ):
        with caplog.at_level(logging.WARNING):
            path = dm.notebook_exporter.export("nb")
    _assert_identity(path, dm, None)
    assert "boom-secret" not in caplog.text
    reasons = [
        r.getMessage() for r in caplog.records if "client session" in r.getMessage()
    ]
    assert len(reasons) == 1 and "lookup_failed" in reasons[0]


def test_t9_cached_exporter_sees_a_later_bind(tmp_path):
    dm = _dm(tmp_path)
    exporter = dm.notebook_exporter
    _assert_identity(exporter.export("before"), dm, None)
    dm.bind_client_session(SID_A)
    assert dm.notebook_exporter is exporter
    _assert_identity(exporter.export("after"), dm, SID_A)


@pytest.mark.parametrize("bound", [True, False])
def test_t10_hash_fallback_keeps_identity_and_namespace(tmp_path, bound):
    dm = _dm(tmp_path)
    if bound:
        dm.bind_client_session(SID_A)
    with patch(
        "lobster.core.notebooks.exporter.hashlib.sha256",
        side_effect=RuntimeError("no hash"),
    ):
        path = dm.notebook_exporter.export("nb")
    record, meta, summary, _ = _read(path)
    expected = SID_A if bound else None
    assert record == {
        "session_id": expected,
        "namespace": "lobster",
        "sha256": "unavailable",
    }
    assert meta["source_session_id"] == expected
    assert meta["provenance_namespace"] == "lobster"
    if bound:
        assert f"- Session ID: `{SID_A}`" in summary
    else:
        assert UNKNOWN_SUMMARY in summary


def test_t11_rebind_during_export_does_not_split_fields(tmp_path):
    dm = _dm(tmp_path)
    dm.bind_client_session(SID_A)
    exporter = dm.notebook_exporter
    original = exporter._create_imports_cell

    def rebind_then_continue(irs):
        dm.bind_client_session(
            SID_B, replace=True
        )  # after the manifest, before summary and metadata
        return original(irs)

    exporter._create_imports_cell = rebind_then_continue
    path = exporter.export("nb")
    record, meta, summary, _ = _read(path)
    assert record["session_id"] == SID_A and meta["source_session_id"] == SID_A
    assert f"- Session ID: `{SID_A}`" in summary
    assert record["sha256"] == _expected_hash(
        SID_A, "lobster", len(dm.provenance.activities), len(dm.provenance.entities)
    )
    assert dm.client_session_id == SID_B


def test_t12_concurrent_different_binds_end_unknown(tmp_path):
    import lobster.core.runtime.data_manager as dm_module

    for _ in range(25):
        dm = DataManagerV2(workspace_path=tmp_path / "c")
        gate = threading.Barrier(2)
        check = dm_module._display_client_session_id

        def slow_check(value):
            ok = check(value)
            gate.wait(timeout=5)
            return ok

        with patch.object(dm_module, "_display_client_session_id", slow_check):
            threads = [
                threading.Thread(target=dm.bind_client_session, args=(s,))
                for s in (SID_A, SID_B)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
        assert dm.client_session_id is None


# ---------------------------------------------------------------- T13, T14 (logs, wording)


def test_t13_warnings_have_fixed_reasons_and_no_raw_values(tmp_path, caplog):
    dm = _dm(tmp_path)
    with caplog.at_level(logging.WARNING):
        dm.bind_client_session(SID_A)
        dm.bind_client_session(SID_A)
        assert not [r for r in caplog.records if "client session" in r.getMessage()]
        dm.bind_client_session("raw secret\nvalue")
        dm.bind_client_session(12345)
    dm2 = _dm(tmp_path, "w2")
    with caplog.at_level(logging.WARNING):
        dm2.bind_client_session(SID_A)
        dm2.bind_client_session(SID_B)
    messages = [
        r.getMessage() for r in caplog.records if "client session" in r.getMessage()
    ]
    assert len(messages) == 3
    assert any("invalid_format" in m for m in messages)
    assert any("invalid_type" in m for m in messages)
    assert any("conflict" in m for m in messages)
    assert (
        "raw secret" not in caplog.text
        and SID_B not in caplog.text
        and "12345" not in caplog.text
    )


@pytest.mark.parametrize("bound", [True, False])
def test_t14_summary_wording_is_neutral(tmp_path, bound):
    dm = _dm(tmp_path)
    if bound:
        dm.bind_client_session(SID_A)
    path = dm.notebook_exporter.export("nb")
    _, _, summary, manifest = _read(path)
    assert NEUTRAL_LINE in summary
    assert "regulatory compliance" not in summary
    assert (
        "SHA-256 of the session ID (`null` when the client session identity was unavailable or ambiguous at export time), the provenance namespace and the activity and entity counts only."
        in manifest
    )
    nb = nbformat.read(path, as_version=4)
    assert "**Verification:** not run." in nb.cells[0].source


# ---------------------------------------------------------------- T15 (persistence invariants)


@pytest.mark.parametrize("bound", [None, SID_A])
@pytest.mark.parametrize("mode", ["recent", "all", "named"])
def test_t15_restore_session_never_binds_saved_values(tmp_path, bound, mode):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".session.json").write_text(json.dumps({"session_id": "1730000000.5"}))
    (workspace / "saved").mkdir()
    (workspace / "saved" / ".session.json").write_text(
        json.dumps({"session_id": "1730000001.5"})
    )
    dm = DataManagerV2(workspace_path=workspace)
    assert dm.client_session_id is None
    if bound:
        dm.bind_client_session(bound)
    if mode == "named":
        dm.restore_session("recent", workspace_name="saved")
    else:
        dm.restore_session(mode)
    assert dm.client_session_id == bound


# ---------------------------------------------------------------- T16–T19 (owners, real constructors)


@pytest.fixture
def no_graph():
    with patch(
        "lobster.core.client.create_bioinformatics_graph",
        return_value=(MagicMock(), {}),
    ) as fake:
        yield fake


def test_t16_agent_client_binds_created_and_injected(tmp_path, no_graph):
    from lobster.core.client import AgentClient

    created = AgentClient(workspace_path=tmp_path / "a", session_id=SID_A)
    assert created.data_manager.client_session_id == SID_A
    injected = DataManagerV2(workspace_path=tmp_path / "b")
    AgentClient(workspace_path=tmp_path / "b", session_id=SID_A, data_manager=injected)
    assert injected.client_session_id == SID_A
    other = DataManagerV2(workspace_path=tmp_path / "c")
    other.bind_client_session(SID_B)
    AgentClient(workspace_path=tmp_path / "c", session_id=SID_A, data_manager=other)
    assert other.client_session_id is None


class _NoBinder:
    def __init__(self, base):
        self.__dict__["_base"] = base

    def __getattr__(self, name):
        if name == "bind_client_session":
            raise AttributeError(name)
        return getattr(self._base, name)


class _RaisingBinderProperty(_NoBinder):
    def __getattr__(self, name):
        if name == "bind_client_session":
            raise RuntimeError("binder-lookup-secret")
        return getattr(self._base, name)


class _RaisingBinderCall(_NoBinder):
    def __getattr__(self, name):
        if name == "bind_client_session":

            def fail(*a, **k):
                raise RuntimeError("binder-call-secret")

            return fail
        return getattr(self._base, name)


@pytest.mark.parametrize(
    "wrapper", [_NoBinder, _RaisingBinderProperty, _RaisingBinderCall]
)
def test_t17_agent_client_binder_is_fail_open(tmp_path, no_graph, caplog, wrapper):
    from lobster.core.client import AgentClient

    base = DataManagerV2(workspace_path=tmp_path / "w")
    with caplog.at_level(logging.WARNING):
        client = AgentClient(
            workspace_path=tmp_path / "w", session_id=SID_A, data_manager=wrapper(base)
        )
        session_file = tmp_path / "w" / "restore.json"
        session_file.write_text(
            json.dumps({"session_id": SID_B, "conversation": [], "metadata": {}})
        )
        assert client.load_session(session_file)["success"] is True
    assert client.session_id == SID_B
    assert base.client_session_id is None
    assert "secret" not in caplog.text


def _session_file(tmp_path, payload) -> Path:
    path = tmp_path / f"s{abs(hash(json.dumps(payload, sort_keys=True)))}.json"
    path.write_text(json.dumps({"conversation": [], "metadata": {}, **payload}))
    return path


def test_t18_restore_rebinds_and_cached_exporter_follows(tmp_path, no_graph):
    from lobster.core.client import AgentClient

    client = AgentClient(workspace_path=tmp_path / "w", session_id=SID_A)
    dm = _with_activity(client.data_manager)
    exporter = dm.notebook_exporter
    _assert_identity(exporter.export("a"), dm, SID_A)
    client.load_session(_session_file(tmp_path, {"session_id": SID_B}))
    assert client.session_id == SID_B and dm.notebook_exporter is exporter
    _assert_identity(exporter.export("b"), dm, SID_B)


def test_t18_restore_clears_a_conflict(tmp_path, no_graph):
    from lobster.core.client import AgentClient

    client = AgentClient(workspace_path=tmp_path / "w", session_id=SID_A)
    dm = _with_activity(client.data_manager)
    exporter = dm.notebook_exporter
    dm.bind_client_session(SID_B)
    assert dm.client_session_id is None
    _assert_identity(exporter.export("conflict"), dm, None)
    client.load_session(_session_file(tmp_path, {"session_id": SID_B}))
    assert dm.client_session_id == SID_B
    _assert_identity(exporter.export("restored"), dm, SID_B)


def test_t18_truthy_invalid_restored_id_leaves_no_stale_identity(tmp_path, no_graph):
    from lobster.core.client import AgentClient

    client = AgentClient(workspace_path=tmp_path / "w", session_id=SID_A)
    dm = _with_activity(client.data_manager)
    exporter = dm.notebook_exporter
    _assert_identity(exporter.export("before"), dm, SID_A)
    client.load_session(_session_file(tmp_path, {"session_id": "my study"}))
    assert client.session_id == "my study"
    assert dm.client_session_id is None
    _assert_identity(exporter.export("after"), dm, None)


@pytest.mark.parametrize("payload", [{"session_id": ""}, {"session_id": None}, {}])
def test_t18_falsy_or_missing_restored_id_changes_nothing(tmp_path, no_graph, payload):
    from lobster.core.client import AgentClient

    client = AgentClient(workspace_path=tmp_path / "w", session_id=SID_A)
    client.load_session(_session_file(tmp_path, payload))
    assert client.session_id == SID_A
    assert client.data_manager.client_session_id == SID_A


@pytest.mark.parametrize(
    "module", ["lobster.cli", "lobster.cli_internal.commands.heavy.session_infra"]
)
def test_t19_command_clients_bind_only_a_concrete_session(tmp_path, module):
    import importlib

    CommandClient = importlib.import_module(module).CommandClient
    workspace = tmp_path / "w"
    (workspace / ".lobster" / "sessions" / SID_A).mkdir(parents=True)
    assert (
        CommandClient(workspace, session_id=SID_A).data_manager.client_session_id
        == SID_A
    )
    assert CommandClient(workspace).data_manager.client_session_id is None
    assert (
        CommandClient(workspace, session_id="command").data_manager.client_session_id
        is None
    )


# ---------------------------------------------------------------- review r1 hardening


class _EqualsEverything(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


class _FormatsOtherText(str):
    def __format__(self, spec):
        return "x`\n- injected line"

    def __str__(self):
        return "other text"


class _FormatRaises(str):
    def __format__(self, spec):
        raise RuntimeError("format-secret")


@pytest.mark.parametrize("replace", [False, True])
def test_r1_failed_validation_from_a_bound_state_is_sticky_unknown(
    tmp_path, caplog, replace
):
    dm = _dm(tmp_path)
    exporter = dm.notebook_exporter
    assert dm.bind_client_session(SID_A) == SID_A
    with caplog.at_level(logging.WARNING):
        # Claims to be a str but is not one; checking it raises inside the regex.
        assert dm.bind_client_session(MagicMock(spec=str), replace=replace) is None
    assert dm.client_session_id is None
    assert dm.bind_client_session(SID_A) is None
    assert dm.client_session_id is None
    _assert_identity(exporter.export("nb"), dm, None)
    assert dm.bind_client_session(SID_B, replace=True) == SID_B
    reasons = [
        r.getMessage() for r in caplog.records if "client session" in r.getMessage()
    ]
    assert len(reasons) == 1


def test_r1_comparison_override_cannot_hide_a_conflict(tmp_path):
    dm = _dm(tmp_path)
    dm.bind_client_session(SID_A)
    assert dm.bind_client_session(_EqualsEverything(SID_B)) is None
    assert dm.client_session_id is None
    _assert_identity(dm.notebook_exporter.export("nb"), dm, None)


def test_r1_bound_identity_is_an_exact_str_copy(tmp_path):
    dm = _dm(tmp_path)
    assert dm.bind_client_session(_EqualsEverything(SID_A)) == SID_A
    assert type(dm.client_session_id) is str
    assert dm.bind_client_session(SID_A) == SID_A  # same contents: no conflict


@pytest.mark.parametrize("cls", [_FormatsOtherText, _FormatRaises])
def test_r1_string_subclass_renders_the_checked_contents(tmp_path, caplog, cls):
    dm = _dm(tmp_path)
    assert dm.bind_client_session(cls("safe-id")) == "safe-id"
    assert type(dm.client_session_id) is str
    with caplog.at_level(logging.WARNING):
        path = dm.notebook_exporter.export("nb")
    _assert_identity(path, dm, "safe-id")
    _, _, summary, _ = _read(path)
    assert "injected" not in summary and "other text" not in summary
    assert "format-secret" not in caplog.text


@pytest.mark.parametrize("cls", [_FormatsOtherText, _FormatRaises])
def test_r1_exporter_renders_checked_contents_from_any_data_manager(tmp_path, cls):
    real = _dm(tmp_path, "src")
    foreign = MagicMock()
    foreign.workspace_path = tmp_path / "foreign"
    foreign.workspace_path.mkdir()
    foreign.modalities = {}
    foreign.client_session_id = cls("safe-id")
    path = NotebookExporter(real.provenance, foreign).export("nb")
    record, meta, summary, _ = _read(path)
    assert record["session_id"] == "safe-id" and meta["source_session_id"] == "safe-id"
    assert "- Session ID: `safe-id`" in summary and "injected" not in summary


@pytest.mark.parametrize("value", ["bad value", 7, MagicMock()])
def test_r1_exporter_logs_one_fixed_reason_for_an_unexpected_value(
    tmp_path, caplog, value
):
    real = _dm(tmp_path, "src")
    foreign = MagicMock()
    foreign.workspace_path = tmp_path / "foreign"
    foreign.workspace_path.mkdir()
    foreign.modalities = {}
    foreign.client_session_id = value
    with caplog.at_level(logging.WARNING):
        path = NotebookExporter(real.provenance, foreign).export("nb")
    record, meta, summary, _ = _read(path)
    assert record["session_id"] is None and meta["source_session_id"] is None
    reasons = [
        r.getMessage() for r in caplog.records if "client session" in r.getMessage()
    ]
    assert len(reasons) == 1 and "invalid_value" in reasons[0]
    assert "bad value" not in caplog.text


def test_r1_exporter_is_quiet_when_unbound(tmp_path, caplog):
    dm = _dm(tmp_path)
    with caplog.at_level(logging.WARNING):
        dm.notebook_exporter.export("nb")
    assert not [r for r in caplog.records if "client session" in r.getMessage()]


def test_r1_one_identity_read_per_export(tmp_path):
    dm = _dm(tmp_path)
    dm.bind_client_session(SID_A)
    reads = []
    original = DataManagerV2.client_session_id

    def counted(self):
        reads.append(1)
        return original.fget(self)

    with patch.object(DataManagerV2, "client_session_id", property(counted)):
        path = dm.notebook_exporter.export("nb")
    assert len(reads) == 1
    _assert_identity(path, dm, SID_A)


class _OwnedLock:
    """A lock that records its owner and can hold the first holder in place."""

    def __init__(self):
        self._lock = threading.Lock()
        self.owner = None
        self.acquisitions = 0
        self.first_inside = threading.Event()
        self.release_first = threading.Event()

    def __enter__(self):
        self._lock.acquire()
        self.owner = threading.get_ident()
        self.acquisitions += 1
        if self.acquisitions == 1:
            self.first_inside.set()
            self.release_first.wait(timeout=10)
        return self

    def __exit__(self, *exc):
        self.owner = None
        self._lock.release()
        return False


_GUARDED = {"_client_session_id", "_client_session_unknown"}


class _GuardedDataManager(DataManagerV2):
    """Records any identity-state access made without holding the identity lock."""

    def _check(self, name):
        if name in _GUARDED and object.__getattribute__(self, "__dict__").get("_armed"):
            lock = object.__getattribute__(self, "_client_session_lock")
            if getattr(lock, "owner", None) != threading.get_ident():
                object.__getattribute__(self, "violations").append(name)

    def __getattribute__(self, name):
        if name in _GUARDED:
            object.__getattribute__(self, "_check")(name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name in _GUARDED:
            self._check(name)
        object.__setattr__(self, name, value)


def test_r1_conflicting_binds_are_serialized_by_the_identity_lock(tmp_path):
    dm = _GuardedDataManager(
        workspace_path=tmp_path / "g",
        session_dir=tmp_path / "g" / ".lobster" / "sessions" / DIR_NAME,
    )
    _with_activity(dm)
    lock = _OwnedLock()
    object.__setattr__(dm, "_client_session_lock", lock)
    object.__setattr__(dm, "violations", [])
    object.__setattr__(dm, "_armed", True)
    results, errors = {}, []

    def bind(label, value):
        try:
            results[label] = dm.bind_client_session(value)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    first = threading.Thread(target=bind, args=("first", SID_A))
    second = threading.Thread(target=bind, args=("second", SID_B))
    first.start()
    assert lock.first_inside.wait(
        timeout=10
    ), "the first bind never took the identity lock"
    second.start()
    second.join(timeout=0.3)
    assert (
        second.is_alive()
    ), "the second bind ran while the first held the identity lock"
    lock.release_first.set()
    first.join(timeout=10)
    second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert results == {"first": SID_A, "second": None}
    assert lock.acquisitions == 2
    lock.acquisitions = 99  # later acquisitions must not pause
    assert dm.client_session_id is None  # the property reads under the lock too
    object.__setattr__(dm, "_armed", False)
    assert dm.violations == []
    assert dm.client_session_id is None
    _assert_identity(dm.notebook_exporter.export("nb"), dm, None)


@pytest.mark.parametrize(
    "module",
    ["lobster.cli", "lobster.cli_internal.commands.heavy.session_infra"],
)
def test_r1_command_client_binder_is_fail_open(tmp_path, caplog, module):
    import importlib

    CommandClient = importlib.import_module(module).CommandClient
    workspace = tmp_path / "w"
    (workspace / ".lobster" / "sessions" / SID_A).mkdir(parents=True)
    with patch.object(
        DataManagerV2, "bind_client_session", side_effect=RuntimeError("cc-secret")
    ):
        with caplog.at_level(logging.WARNING):
            client = CommandClient(workspace, session_id=SID_A)
    assert client.data_manager.client_session_id is None
    assert "cc-secret" not in caplog.text and "binder_failed" in caplog.text
