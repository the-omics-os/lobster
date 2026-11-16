"""Offline synthetic contained-download -> real queue/tool/H5/IR requirement proof."""

import gzip
import json
from unittest.mock import MagicMock, patch

import anndata
import numpy as np
import pandas as pd
import pytest

# GEO download imports geo/loaders/tenx.py -> scanpy (optional extra). Without it
# the package-only CI env cannot even collect this module; the full env runs it.
pytest.importorskip("scanpy")

from lobster.agents.data_expert.data_expert import data_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus
from lobster.services.data_access.geo.parser import GEOParser
from lobster.services.data_access.geo_download_service import GEODownloadService
from lobster.testing.provenance_replay import factory_tools, replay_ir
from lobster.tools.download_orchestrator import DownloadOrchestrator

MODULE = "lobster.agents.data_expert.data_expert"
GSE = "GSE900000000001"
A, B = "GSM900000000001", "GSM900000000002"
CANONICAL = "geo_" + GSE.lower() + "_transcriptomics_bulk"
BASE = "https://example.test/"


def table(titles=False):
    return pd.DataFrame(
        [[1, 3], [2, 4], [5, 6], [2, 5]],
        index=["synthetic_a", "synthetic_b", "synthetic_c", "synthetic_d"],
        columns=["Synthetic A", "Synthetic B"] if titles else [A, B],
    )


@pytest.fixture
def fixture_import(tmp_path):
    manager = DataManagerV2(workspace_path=tmp_path)
    metadata = {
        "title": "Synthetic series",
        "samples": {
            B: {"title": "Synthetic B", "characteristics_ch1": ["condition: second"]},
            A: {"title": "Synthetic A", "characteristics_ch1": ["condition: first"]},
        },
        "supplementary_file": [],
    }
    manager.metadata_store[GSE] = {"metadata": metadata, "strategy_config": {}}
    wrapper = GEODownloadService(manager)
    geo = wrapper.geo_service
    geo._check_platform_compatibility = MagicMock(return_value=(True, "synthetic"))
    payloads, downloaded = {}, []

    def transport(url, destination):
        # The only acquisition seam: bounded generated text bytes, no live requests.
        data = payloads[url]
        assert len(data) <= 256 * 1024
        destination.write_bytes(data)
        downloaded.append((url, destination))
        return True

    geo.geo_downloader.download_file = MagicMock(side_effect=transport)
    manager.log_tool_usage = MagicMock(wraps=manager.log_tool_usage)
    manager.on_modality_loaded = MagicMock()
    manager.provenance.create_entity = MagicMock(wraps=manager.provenance.create_entity)
    entry = DownloadQueueEntry(
        entry_id="synthetic_entry",
        dataset_id=GSE,
        database="geo",
        modality_name="synthetic_queue_alias",
    )
    manager.download_queue.add_entry(entry)
    with patch.object(DownloadOrchestrator, "_register_default_services"):
        orchestrator = DownloadOrchestrator(manager)
    orchestrator.register_service(wrapper)
    tool = factory_tools(data_expert, MODULE, manager)["execute_download_from_queue"]
    return manager, metadata, geo, payloads, downloaded, entry, orchestrator, tool


def add_file(setup, name, frame=None, *, compression=False):
    manager, metadata, geo, payloads, downloaded, entry, orchestrator, tool = setup
    url = BASE + name
    content = (frame if frame is not None else table()).to_csv(sep="\t").encode()
    payloads[url] = gzip.compress(content) if compression else content
    metadata["supplementary_file"].append(url)
    return url


def invoke(setup, **arguments):
    manager, metadata, geo, payloads, downloaded, entry, orchestrator, tool = setup
    with patch(f"{MODULE}.DownloadOrchestrator", return_value=orchestrator):
        return tool.invoke(
            {"entry_id": entry.entry_id, "data_requirement": "gene_counts", **arguments}
        )


def assert_refusal(setup, result):
    manager, metadata, geo, payloads, downloaded, entry, orchestrator, tool = setup
    assert "Import Refused" in result
    assert "choose another dataset" in result and "report" in result
    assert "Do not retry this entry unchanged" in result
    assert "Try a different" not in result
    assert "different download strategy" not in result
    assert "different concatenation" not in result
    assert 'strategy_override="' not in result
    assert "retry the same entry" not in result.lower()
    assert (
        manager.download_queue.get_entry(entry.entry_id).status == DownloadStatus.FAILED
    )
    assert manager.list_modalities() == []
    manager.on_modality_loaded.assert_not_called()
    manager.log_tool_usage.assert_not_called()
    manager.provenance.create_entity.assert_not_called()


@pytest.mark.parametrize("titles", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_complete_tool_import_actual_parser_h5_and_rendered_ir_replay(
    fixture_import, titles, compressed, monkeypatch
):
    setup = fixture_import
    manager, metadata, geo, payloads, downloaded, entry, orchestrator, tool = setup
    name = "Synthetic_Gene_Counts%20v1%2Fpart.tsv" + (".gz" if compressed else "")
    frame = table(titles)
    url = add_file(setup, name, frame, compression=compressed)
    add_file(setup, "synthetic_transcript.tsv")
    result = invoke(setup)
    assert "Download completed successfully" in result and name in result
    assert "whole-number counts (numeric check)" in result
    assert manager.list_modalities() == [CANONICAL]
    current = manager.download_queue.get_entry(entry.entry_id)
    assert (
        current.status == DownloadStatus.COMPLETED
        and current.modality_name == CANONICAL
    )
    assert [u for u, _ in downloaded] == [url]
    assert downloaded[0][1].name != name  # Hash cache is not the logical source name.
    data = manager.get_modality(CANONICAL)
    np.testing.assert_array_equal(data.X, frame.T.to_numpy())
    assert list(data.obs["source_sample_id"]) == [A, B]
    assert list(data.obs["condition"]) == ["first", "second"]
    expected_keys = {
        "schema_version",
        "data_requirement",
        "representation",
        "level",
        "accepted_file",
        "sample_rule",
        "candidates_json",
        "numeric_check",
    }
    assert set(data.uns["geo_import"]) == expected_keys
    assert data.uns["geo_import"]["accepted_file"] == name
    facts = json.loads(data.uns["geo_import"]["candidates_json"])
    assert len(facts) == 2 and sum(f["verdict"] == "accepted" for f in facts) == 1
    restored = anndata.read_h5ad(manager.data_dir / (CANONICAL + "_raw.h5ad"))
    np.testing.assert_array_equal(restored.X, frame.T.to_numpy())
    assert list(restored.obs["source_sample_id"]) == [A, B]
    assert all(type(value) is str for value in restored.obs["source_sample_id"])
    assert restored.uns["geo_import"]["accepted_file"] == name
    assert json.loads(restored.uns["geo_import"]["candidates_json"]) == facts
    calls = [
        call
        for call in manager.log_tool_usage.call_args_list
        if call.kwargs.get("tool_name") == "execute_download_from_queue"
    ]
    assert len(calls) == 1
    ir = calls[0].kwargs["ir"]
    assert ir.execution_context["data_requirement"] == "gene_counts"
    assert ir.output_entities == [CANONICAL]
    monkeypatch.chdir(manager.workspace_path)
    raw_file = manager.data_dir / (CANONICAL + "_raw.h5ad")
    replayed = replay_ir(ir, input_data=str(raw_file))["adata"]
    np.testing.assert_array_equal(replayed.X, restored.X)
    assert list(replayed.obs["source_sample_id"]) == [A, B]
    assert json.loads(replayed.uns["geo_import"]["candidates_json"]) == facts
    # Completed request returns the same canonical proof without acquisition/logging again.
    downloads, logs = len(downloaded), manager.log_tool_usage.call_count
    again = invoke(setup)
    assert "already imported" in again and name in again
    assert len(downloaded) == downloads and manager.log_tool_usage.call_count == logs
    assert manager.list_modalities() == [CANONICAL]


@pytest.mark.parametrize(
    "kind, reason",
    [
        ("fractional", "whole-number"),
        ("negative", "whole-number"),
        ("nonfinite", "whole-number"),
        ("constant", "constant sample totals"),
        ("transcript", "transcript-level"),
        ("unknown_level", "gene level not established"),
        ("subset", "sample identity"),
        ("extra", "sample identity"),
        ("ambiguous", "ambiguous"),
        ("high", "value_fidelity"),
        ("clinical", "characteristics_ch1"),
    ],
)
def test_refusal_model_result_exact_evidence_zero_registration_and_reclaimable(
    fixture_import, kind, reason
):
    setup = fixture_import
    manager, metadata, geo, payloads, downloaded, entry, orchestrator, tool = setup
    frame, name = table(), "synthetic_gene.tsv"
    if kind == "fractional":
        frame = frame.astype(float)
        frame.iloc[0, 0] = 1.5
    elif kind == "negative":
        frame.iloc[0, 0] = -1
    elif kind == "nonfinite":
        frame = frame.astype(float)
        frame.iloc[0, 0] = np.inf
    elif kind == "constant":
        frame[B] = frame[A]
    elif kind == "transcript":
        name = "synthetic_transcript.tsv"
    elif kind == "unknown_level":
        name = "synthetic_matrix.tsv"
    elif kind == "subset":
        frame = frame[[A]]
    elif kind == "extra":
        frame["GSM900000000003"] = [1, 2, 4, 5]
    elif kind == "high":
        frame.iloc[0, 0] = 2**24 + 1
    elif kind == "clinical":
        metadata["samples"][A]["characteristics_ch1"] = "malformed"
    add_file(setup, name, frame)
    if kind == "ambiguous":
        add_file(setup, "synthetic_gene_second.tsv")
    for attempt in range(2):
        result = invoke(setup)
        assert_refusal(setup, result)
        assert reason in result and name in result
        assert (
            len(manager.download_queue.get_entry(entry.entry_id).error_log)
            == attempt + 1
        )


@pytest.mark.parametrize("kind", ["overlong", "short", "mixed_r", "quoted_whitespace"])
@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
@pytest.mark.parametrize("chunked", [False, True], ids=["eager", "chunked"])
def test_malformed_original_records_refuse_before_registration_twice(
    fixture_import, monkeypatch, kind, compressed, chunked
):
    setup = fixture_import
    manager, metadata, geo, payloads, downloaded, entry, orchestrator, tool = setup
    frame = pd.DataFrame(
        np.tile([[1, 3]], (2001, 1)),
        index=[f"synthetic_gene_{i}" for i in range(2001)],
        columns=[A, B],
    )
    name = "synthetic_gene_records.tsv" + (".gz" if compressed else "")
    url = add_file(setup, name, frame)
    text = frame.to_csv(sep="\t")
    if kind == "overlong":
        text = text.replace(
            "synthetic_gene_2000\t1\t3", "synthetic_gene_2000\t1\t3\t999"
        )
    elif kind == "short":
        text = text.replace("synthetic_gene_2000\t1\t3", "synthetic_gene_2000\t1")
    elif kind == "mixed_r":
        text = text.removeprefix("\t").replace("synthetic_gene_2000\t1\t3", "1\t3")
    else:
        text += '"   "\n'
    payloads[url] = gzip.compress(text.encode()) if compressed else text.encode()
    actual_memory_check = geo.geo_parser.check_memory_for_dimensions

    def resource_mode(*dimensions):
        check = actual_memory_check(*dimensions)
        check["can_load"] = not chunked
        return check

    monkeypatch.setattr(geo.geo_parser, "check_memory_for_dimensions", resource_mode)
    monkeypatch.setattr(geo.geo_parser, "_get_adaptive_chunk_size", lambda *_: 1000)
    monkeypatch.setattr(
        geo.geo_parser, "_check_memory_availability", lambda *_, **__: True
    )
    # No actual parse flags are manufactured: the real parser consumes original bytes.
    for attempt in range(2):
        result = invoke(setup)
        assert_refusal(setup, result)
        assert "file could not be read completely" in result
        assert name in result
        assert (
            len(manager.download_queue.get_entry(entry.entry_id).error_log)
            == attempt + 1
        )
        assert list(manager.data_dir.glob("*.h5ad")) == []


@pytest.mark.parametrize(
    "kind",
    [
        "rounded_fraction",
        "duplicate_header",
        "nul_feature",
        "nul_sample",
        "nul_header",
        "nul_value",
    ],
)
@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
@pytest.mark.parametrize("chunked", [False, True], ids=["eager", "chunked"])
def test_original_token_fidelity_refuses_before_registration_twice(
    fixture_import, monkeypatch, kind, compressed, chunked
):
    setup = fixture_import
    manager, metadata, geo, payloads, downloaded, entry, orchestrator, tool = setup
    name = "synthetic_gene_original.tsv" + (".gz" if compressed else "")
    url = add_file(setup, name)
    if kind == "rounded_fraction":
        text = (
            table()
            .to_csv(sep="\t")
            .replace("synthetic_a\t1\t3", "synthetic_a\t1.0000000000000000001\t3")
        )
        reason = "values are not finite non-negative whole-number counts"
    elif kind == "duplicate_header":
        text = f"sample\tsynthetic_a\tsynthetic_a\tsynthetic_b\n{A}\t1\t2\t3\n{B}\t4\t5\t6\n"
        reason = "file could not be read completely"
    else:
        text = table().to_csv(sep="\t")
        if kind == "nul_feature":
            text = text.replace("synthetic_a", "synthetic_a\x00suffix")
        elif kind == "nul_sample":
            text = f"sample\tsynthetic_a\tsynthetic_b\n{A}\x00suffix\t1\t3\n{B}\t2\t4\n"
        elif kind == "nul_header":
            text = text.replace(A, A + "\x00suffix")
        else:
            text = text.replace("synthetic_a\t1\t3", "synthetic_a\t1\x00suffix\t3")
        reason = "file could not be read completely"
    payloads[url] = gzip.compress(text.encode()) if compressed else text.encode()
    monkeypatch.setattr(
        geo.geo_parser,
        "check_memory_for_dimensions",
        lambda *_: {"can_load": not chunked},
    )
    monkeypatch.setattr(geo.geo_parser, "_get_adaptive_chunk_size", lambda *_: 1000)
    monkeypatch.setattr(
        geo.geo_parser, "_check_memory_availability", lambda *_, **__: True
    )
    for attempt in range(2):
        result = invoke(setup)
        assert_refusal(setup, result)
        assert reason in result
        assert (
            len(manager.download_queue.get_entry(entry.entry_id).error_log)
            == attempt + 1
        )
        assert list(manager.data_dir.glob("*.h5ad")) == []


def test_corrupt_gzip_is_incomplete_read_not_manufactured_partial_flag(fixture_import):
    setup = fixture_import
    url = add_file(setup, "synthetic_gene.tsv.gz", compression=True)
    setup[3][url] = setup[3][url][:-8]
    result = invoke(setup)
    assert_refusal(setup, result)
    assert "file could not be read completely" in result


def test_real_chunk_resource_probe_partial_result_is_refused_by_real_guard(
    fixture_import, tmp_path, monkeypatch
):
    # Explicit UNIT resource probe seam. No OOM, public knob or full-run seam claim.
    setup = fixture_import
    manager, metadata, geo, payloads, downloaded, entry, orchestrator, tool = setup
    frame = pd.DataFrame(
        np.tile([[1, 3]], (2001, 1)),
        index=[f"synthetic_gene_{i}" for i in range(2001)],
        columns=[A, B],
    )
    path = tmp_path / "synthetic_gene_partial.tsv"
    frame.to_csv(path, sep="\t")
    parser = GEOParser()
    monkeypatch.setattr(parser, "_get_adaptive_chunk_size", lambda *_: 1000)
    probe = MagicMock(side_effect=[True, False])
    monkeypatch.setattr(parser, "_check_memory_availability", probe)
    partial = parser.parse_large_file_in_chunks(path, "\t", None)
    assert partial.is_partial is True and partial.is_complete is False
    assert partial.rows_read == 1000 and partial.data.shape == (1000, 2)
    assert "Memory limit" in partial.truncation_reason
    assert probe.call_count == 2
    url = add_file(setup, "synthetic_gene_partial.tsv", frame)
    # Read the same bounded transport; this parser entry seam chooses real chunk code.
    probe.side_effect = [True, False]
    monkeypatch.setattr(
        parser, "check_memory_for_dimensions", lambda *_: {"can_load": False}
    )
    monkeypatch.setattr(
        geo.geo_parser,
        "parse_required_expression_file",
        parser.parse_required_expression_file,
    )
    result = invoke(setup)
    assert_refusal(setup, result)
    assert "file could not be read completely" in result
    assert downloaded[0][0] == url
