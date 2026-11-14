"""Peptide library tools log IR that rebuilds the same library."""

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.proteomics.peptide_tools import create_peptide_tools
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.services.peptidomics.peptide_library_service import (
    PeptideLibraryService,
)
from lobster.testing.provenance_replay import RecordingDataManager, replay_ir

FASTA = ">p1 first\nGIGKFLHSAKKFGKAFVGEIMNS\n>p2\nKWKLFKKIGAVLKVL\n>bad\nXXZZ\n"


@pytest.fixture
def library(tmp_path):
    path = tmp_path / "peptides.fasta"
    path.write_text(FASTA)
    dm = RecordingDataManager(workspace_path=tmp_path)
    tools = {t.name: t for t in create_peptide_tools(dm, workspace_path=tmp_path)}
    tools["import_peptide_sequences"].invoke({"file_path": str(path)})
    return dm, tools, path


def _ir(dm, tool_name):
    ir = dm.last_ir(tool_name)
    assert isinstance(ir, AnalysisStep) and ir.replayable
    ir.validate_rendered_code()
    return ir


def test_import_replays(library):
    dm, _, _ = library
    produced = dm.modalities["peptides"]
    assert list(produced.obs_names) == ["p1", "p2"]  # invalid sequence dropped
    replayed = replay_ir(_ir(dm, "import_peptide_sequences"))["adata"]
    pd.testing.assert_frame_equal(replayed.obs, produced.obs)


def test_filter_replays(library):
    dm, tools, _ = library
    tools["filter_peptides"].invoke({"modality_name": "peptides", "max_length": 20})
    produced = dm.modalities["peptides_filtered"]
    assert list(produced.obs_names) == ["p2"]
    source = dm.modalities["peptides"].copy()
    replayed = replay_ir(_ir(dm, "filter_peptides"), adata=source)["adata"]
    pd.testing.assert_frame_equal(replayed.obs, produced.obs)


@pytest.mark.parametrize(
    "variant_type", ["alanine_scan", "point_mutations", "scrambled"]
)
def test_variants_replay(library, variant_type):
    dm, tools, _ = library
    tools["generate_peptide_variants"].invoke(
        {
            "modality_name": "peptides",
            "variant_type": variant_type,
            "target_peptide": "p2",
        }
    )
    produced = dm.modalities[f"peptides_{variant_type}"]
    source = dm.modalities["peptides"].copy()
    ir = _ir(dm, "generate_peptide_variants")
    assert ir.parameters["parent_name"] == "p2"
    assert ir.parameters["parent_sequence"] == "KWKLFKKIGAVLKVL"
    assert ir.parameters["parent_position"] == 1
    assert ir.execution_context["replay_policy"] == "current_library_only"
    if variant_type == "scrambled":
        # The drawn seed is recorded, so the scrambled set can be regenerated.
        assert isinstance(ir.parameters["random_seed"], int)
    scope = replay_ir(ir, adata=source)
    replayed = scope["adata"]
    assert scope["variant_stats"]["parent_position"] == 1
    assert scope["variant_stats"]["parent_sequence"] == ir.parameters["parent_sequence"]
    assert scope["variant_stats"]["n_variants"] == produced.n_obs
    assert scope["_"].parameters == ir.parameters
    pd.testing.assert_frame_equal(replayed.obs, produced.obs)


def test_activity_annotation_reapplies_recorded_hits(library):
    dm, _, _ = library
    seq = dm.modalities["peptides"].obs.loc["p1", "sequence"]

    class FakeProvider:
        def search_sequences(self, query, max_results=3):
            return [SimpleNamespace(sequence=seq, activities=["antibacterial"])]

    adata_out, stats, ir = PeptideLibraryService().annotate_known_activity(
        dm.modalities["peptides"], providers=[FakeProvider], database="dbaasp"
    )
    assert stats["n_annotated"] == 1
    assert ir.replayable
    # Replay needs no database: it re-applies the recorded hits.
    replayed = replay_ir(ir, adata=dm.modalities["peptides"].copy())["adata"]
    pd.testing.assert_series_equal(
        replayed.obs["known_activities"], adata_out.obs["known_activities"]
    )
    assert list(adata_out.obs["known_activities"]) == ["antibacterial", "unknown"]


def test_duplicate_sequences_get_one_consistent_outcome(library):
    """Duplicates are looked up once, so the sequence-keyed IR replays every row."""
    import anndata as ad

    calls = []

    class FlakyProvider:
        def search_sequences(self, query, max_results=3):
            calls.append(query)
            if len(calls) > 1:
                raise RuntimeError("transient provider failure")
            return [SimpleNamespace(sequence="ACDE", activities=["antibacterial"])]

    peptides = ad.AnnData(
        obs=pd.DataFrame({"sequence": ["ACDE", "ACDE"]}, index=["a", "b"])
    )
    adata_out, _, ir = PeptideLibraryService().annotate_known_activity(
        peptides, providers=[FlakyProvider], database="dbaasp"
    )
    assert len(calls) == 1
    assert list(adata_out.obs["known_activities"]) == ["antibacterial", "antibacterial"]
    replayed = replay_ir(ir, adata=peptides.copy())["adata"]
    pd.testing.assert_series_equal(
        replayed.obs["known_activities"], adata_out.obs["known_activities"]
    )


def test_unbounded_filter_replays(library):
    """inf bounds must render as runnable Python, not the undefined name `inf`."""
    dm, tools, _ = library
    tools["filter_peptides"].invoke(
        {
            "modality_name": "peptides",
            "min_charge": float("-inf"),
            "max_charge": float("inf"),
        }
    )
    produced = dm.modalities["peptides_filtered"]
    source = dm.modalities["peptides"].copy()
    replayed = replay_ir(_ir(dm, "filter_peptides"), adata=source)["adata"]
    pd.testing.assert_frame_equal(replayed.obs, produced.obs)


def test_import_only_notebook_runs_without_an_h5ad_input(tmp_path, monkeypatch):
    """The exported notebook must build adata from the FASTA, not load dataset.h5ad."""
    import nbformat

    monkeypatch.chdir(tmp_path)  # the notebook's save cell writes to the cwd

    from lobster.core.runtime.data_manager import DataManagerV2

    fasta = tmp_path / "lib.fasta"
    fasta.write_text(FASTA)
    dm = DataManagerV2(workspace_path=tmp_path / "ws", auto_scan=False)
    tools = {t.name: t for t in create_peptide_tools(dm, workspace_path=tmp_path)}
    tools["import_peptide_sequences"].invoke({"file_path": str(fasta)})
    tools["filter_peptides"].invoke({"modality_name": "peptides", "max_length": 20})

    nb = nbformat.read(
        dm.export_notebook(name="peptides", filter_strategy="replay"), as_version=4
    )
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "read_h5ad(input_data)" not in code
    scope = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "<cell>", "exec"), scope)  # nosec B102
    assert list(scope["adata"].obs_names) == ["p2"]


def test_later_peptide_import_keeps_the_input_load(tmp_path):
    """Only a first-step import replaces the notebook's input load."""
    from types import SimpleNamespace as NS

    from lobster.core.notebooks.exporter import NotebookExporter

    def pair(op):
        return ({}, NS(operation=op))

    exporter = NotebookExporter.__new__(NotebookExporter)
    first = [pair("peptidomics.import_sequences"), pair("peptidomics.filter_peptides")]
    later = [
        pair("lobster.ml.select_ml_features"),
        pair("peptidomics.import_sequences"),
    ]
    assert exporter._has_download_activity(first) is True
    assert exporter._has_download_activity(later) is False


def test_report_export_is_utility(library):
    _, tools, _ = library
    tool = tools["export_peptide_report"]
    assert tool.metadata == {"categories": ["UTILITY"], "provenance": False}


def _sequence_file(tmp_path, format, labels, sequences):
    path = tmp_path / f"library.{format}"
    if format == "fasta":
        path.write_text("".join(f">{n}\n{s}\n" for n, s in zip(labels, sequences)))
    else:
        pd.DataFrame({"name": labels, "sequence": sequences}).to_csv(path, index=False)
    return path


def _notebook_tools(tmp_path, monkeypatch):
    """Real storage/provenance/export; only environment metadata is stubbed."""
    from lobster.core.notebooks.exporter import NotebookExporter
    from lobster.core.runtime.data_manager import DataManagerV2

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(NotebookExporter, "_get_system_info", lambda self: {})
    dm = DataManagerV2(workspace_path=tmp_path / "ws", auto_scan=False)
    return dm, {t.name: t for t in create_peptide_tools(dm, workspace_path=tmp_path)}


def _export_and_run(dm, strategy, scope):
    import nbformat

    nb = nbformat.read(
        dm.export_notebook(name="variants", filter_strategy=strategy), as_version=4
    )
    for number, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            exec(
                compile(cell.source, f"<notebook-cell:{number}>", "exec"), scope
            )  # nosec B102


def _assert_variants_equal(replayed, produced):
    assert replayed.shape == produced.shape
    assert all(isinstance(s, str) for s in replayed.obs["sequence"])
    # H5AD writing may categorize strings; biological values and order must agree.
    # Only the index name is set aside: where anndata cannot keep an unnamed index,
    # the session writer names it 'index' (obs) or 'gene_id' (var) while the runtime
    # object keeps None. Column labels and values still compare.
    expected, actual = produced.obs.astype(object), replayed.obs.astype(object)
    expected.index.name = actual.index.name = None
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    np.testing.assert_array_equal(replayed.X, produced.X)
    assert list(replayed.var_names) == list(produced.var_names)


@pytest.mark.parametrize("format", ["fasta", "csv"])
def test_duplicate_labels_default_records_positional_parent(tmp_path, format):
    path = _sequence_file(tmp_path, format, ["dup", "dup"], ["ACDE", "FGHI"])
    dm = RecordingDataManager(workspace_path=tmp_path)
    tools = {t.name: t for t in create_peptide_tools(dm, workspace_path=tmp_path)}
    tools["import_peptide_sequences"].invoke({"file_path": str(path)})
    result = tools["generate_peptide_variants"].invoke({"modality_name": "peptides"})
    assert "parent_sequence=ACDE" in result
    ir = _ir(dm, "generate_peptide_variants")
    assert ir.parameters["parent_name"] == "dup"
    assert ir.parameters["parent_sequence"] == "ACDE"
    assert ir.parameters["parent_position"] == 0
    source = dm.modalities["peptides"].copy()
    replayed = replay_ir(ir, adata=source)["adata"]
    assert list(replayed.obs["sequence"]) == ["AADE", "ACAE", "ACDA"]
    _assert_variants_equal(replayed, dm.modalities["peptides_alanine_scan"])
    # Reordering duplicate labels must not silently change the positional parent.
    with pytest.raises(ValueError, match="replay input mismatch"):
        replay_ir(ir, adata=source[::-1].copy())


@pytest.mark.parametrize("format", ["fasta", "csv"])
@pytest.mark.parametrize(
    "target,error", [("dup", "Ambiguous"), ("absent", "not found")]
)
def test_explicit_invalid_target_does_not_store_or_log(tmp_path, format, target, error):
    path = _sequence_file(tmp_path, format, ["dup", "dup"], ["ACDE", "FGHI"])
    dm = RecordingDataManager(workspace_path=tmp_path)
    tools = {t.name: t for t in create_peptide_tools(dm, workspace_path=tmp_path)}
    tools["import_peptide_sequences"].invoke({"file_path": str(path)})
    with pytest.raises(ValueError, match=error):
        PeptideLibraryService().generate_variants(
            dm.modalities["peptides"], target_peptide=target
        )
    before_calls, before_modalities = len(dm.calls), list(dm.modalities)
    response = tools["generate_peptide_variants"].invoke(
        {"modality_name": "peptides", "target_peptide": target}
    )
    assert error in response and "STATUS: complete" not in response
    assert len(dm.calls) == before_calls
    assert list(dm.modalities) == before_modalities


def test_empty_parent_library_fails_clearly():
    empty = ad.AnnData(obs=pd.DataFrame({"sequence": []}))
    with pytest.raises(ValueError, match="empty"):
        PeptideLibraryService().generate_variants(empty)


@pytest.mark.parametrize(
    "change", ["missing", "sequence", "label", "other_row", "no_sequence"]
)
def test_replay_rejects_changed_input_even_when_parent_survives(library, change):
    dm, tools, _ = library
    tools["generate_peptide_variants"].invoke({"modality_name": "peptides"})
    ir = _ir(dm, "generate_peptide_variants")
    source = dm.modalities["peptides"].copy()
    if change == "missing":
        source = source[1:].copy()
    elif change == "sequence":
        source.obs.iloc[0, source.obs.columns.get_loc("sequence")] = "ACDE"
    elif change == "label":
        source.obs_names = ["renamed", "p2"]
    elif change == "no_sequence":
        del source.obs["sequence"]
    else:
        source = source[:1].copy()  # same parent, not the recorded input library
    scope = {"adata": source}
    with pytest.raises(ValueError, match="replay input mismatch"):
        exec(ir.render(), scope)  # nosec B102
    assert scope["adata"] is source  # fail before replacing the current library


@pytest.mark.parametrize("strategy", ["successful", "replay"])
@pytest.mark.parametrize("format", ["fasta", "csv"])
def test_duplicate_default_full_notebook_matches_runtime(
    tmp_path, monkeypatch, strategy, format
):
    dm, tools = _notebook_tools(tmp_path, monkeypatch)
    path = _sequence_file(tmp_path, format, ["dup", "dup"], ["ACDE", "FGHI"])
    tools["import_peptide_sequences"].invoke({"file_path": str(path)})
    tools["generate_peptide_variants"].invoke({"modality_name": "peptides"})
    produced = dm.get_modality("peptides_alanine_scan").copy()
    scope = {}
    _export_and_run(dm, strategy, scope)
    _assert_variants_equal(scope["adata"], produced)
    saved = ad.read_h5ad(scope["output_path"])
    _assert_variants_equal(saved, produced)
    assert saved.n_obs == 3
    assert list(saved.obs["parent"]) == ["dup"] * 3


@pytest.mark.parametrize("strategy", ["successful", "replay"])
@pytest.mark.parametrize(
    "variant_type", ["alanine_scan", "point_mutations", "scrambled"]
)
def test_straight_line_variants_full_notebook_parity(
    tmp_path, monkeypatch, strategy, variant_type
):
    dm, tools = _notebook_tools(tmp_path, monkeypatch)
    path = _sequence_file(
        tmp_path, "fasta", ["p1", "p2"], ["ACDEFGHIK", "KWKLFKKIGAVLKVL"]
    )
    tools["import_peptide_sequences"].invoke({"file_path": str(path)})
    tools["filter_peptides"].invoke({"modality_name": "peptides", "min_length": 10})
    result = tools["generate_peptide_variants"].invoke(
        {"modality_name": "peptides_filtered", "variant_type": variant_type}
    )
    assert "parent_sequence=KWKLFKKIGAVLKVL" in result
    produced = dm.get_modality(f"peptides_filtered_{variant_type}").copy()
    scope = {}
    _export_and_run(dm, strategy, scope)
    _assert_variants_equal(scope["adata"], produced)
    _assert_variants_equal(ad.read_h5ad(scope["output_path"]), produced)
    expected_count = {"alanine_scan": 14, "point_mutations": 285, "scrambled": 10}[
        variant_type
    ]
    assert produced.n_obs == expected_count
    assert list(produced.obs["parent"]) == ["p2"] * expected_count


@pytest.mark.parametrize("strategy", ["successful", "replay"])
@pytest.mark.parametrize("target", ["p1", "p2"])
def test_original_library_branch_full_notebook_fails_closed(
    tmp_path, monkeypatch, strategy, target
):
    """Canonical short-p1 branch, and a surviving-p2 branch, cannot alias the filter."""
    dm, tools = _notebook_tools(tmp_path, monkeypatch)
    path = _sequence_file(
        tmp_path, "fasta", ["p1", "p2"], ["ACDEFGHIK", "KWKLFKKIGAVLKVL"]
    )
    tools["import_peptide_sequences"].invoke({"file_path": str(path)})
    tools["filter_peptides"].invoke({"modality_name": "peptides", "min_length": 10})
    result = tools["generate_peptide_variants"].invoke(
        {"modality_name": "peptides", "target_peptide": target}
    )
    produced = dm.get_modality("peptides_alanine_scan")
    assert "STATUS: complete" in result
    assert produced.n_obs == (8 if target == "p1" else 14)
    assert list(produced.obs["parent"]) == [target] * produced.n_obs
    scope = {}
    with pytest.raises(ValueError, match="replay input mismatch.*non-current library"):
        _export_and_run(dm, strategy, scope)
    assert list(scope["adata"].obs_names) == ["p2"]  # no variant output substituted
    assert "output_path" not in scope  # saving must never be reached
    assert not list(tmp_path.glob("*_processed.h5ad"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("parent_position", 0),
        ("parent_position", 99),
        ("parent_name", "absent"),
        ("parent_sequence", "ACDE"),
    ],
)
def test_recorded_parent_must_match_its_position(library, field, value):
    dm, tools, _ = library
    tools["generate_peptide_variants"].invoke(
        {"modality_name": "peptides", "target_peptide": "p2"}
    )
    ir = _ir(dm, "generate_peptide_variants")
    ir.parameters[field] = value
    with pytest.raises(ValueError, match="replay input mismatch"):
        replay_ir(ir, adata=dm.modalities["peptides"].copy())


@pytest.mark.parametrize("strategy", ["successful", "replay"])
def test_original_branch_rejected_with_unchanged_positional_parent(
    tmp_path, monkeypatch, strategy
):
    """Checking only the selected row cannot certify the actual input library."""
    dm, tools = _notebook_tools(tmp_path, monkeypatch)
    path = _sequence_file(
        tmp_path, "fasta", ["p1", "p2"], ["KWKLFKKIGAVLKVL", "ACDEFGHIK"]
    )
    tools["import_peptide_sequences"].invoke({"file_path": str(path)})
    tools["filter_peptides"].invoke({"modality_name": "peptides", "min_length": 10})
    result = tools["generate_peptide_variants"].invoke(
        {"modality_name": "peptides", "target_peptide": "p1"}
    )
    assert "STATUS: complete" in result and "variants_generated=14" in result
    scope = {}
    with pytest.raises(ValueError, match="replay input mismatch.*non-current library"):
        _export_and_run(dm, strategy, scope)
    assert list(scope["adata"].obs_names) == ["p1"]
    assert list(scope["adata"].obs["sequence"]) == ["KWKLFKKIGAVLKVL"]
    assert "output_path" not in scope


def test_notebook_file_keeps_an_unnamed_obs_index_unnamed(tmp_path, monkeypatch):
    dm, tools = _notebook_tools(tmp_path, monkeypatch)
    path = _sequence_file(tmp_path, "fasta", ["p1", "p2"], ["ACDE", "FGHI"])
    tools["import_peptide_sequences"].invoke({"file_path": str(path)})
    scope = {}
    _export_and_run(dm, "successful", scope)
    assert ad.read_h5ad(scope["output_path"]).obs.index.name is None
