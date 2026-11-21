"""A pseudobulk matrix made from single-cell data replays to MATCH.

``create_pseudobulk_matrix`` sums the cells of each sample and cell type into one
pseudobulk sample and saves it as ``<modality>_pseudobulk``. The exported cell
must make that same object (its rows, the ``raw_aggregated`` layer, the obs, var
and uns the service writes) with every setting the session used, so that
``lobster replay --verify`` compares the saved file with it, also when a DE step
changes the pseudobulk matrix after the session saved it. Skipped without the
transcriptomics package, pyDESeq2 or a Jupyter kernel.
"""

import json

import numpy as np
import pandas as pd
import pytest

from tests.unit.cli.test_replay_commands import app, needs_kernel, runner

pytest.importorskip("pydeseq2")
de_module = pytest.importorskip("lobster.agents.transcriptomics.de_analysis_expert")
pseudobulk = pytest.importorskip("lobster.services.analysis.pseudobulk_service")

import anndata as ad  # noqa: E402

from lobster.core.runtime.data_manager import DataManagerV2  # noqa: E402
from lobster.testing.provenance_replay import factory_tools  # noqa: E402

DEFAULTS = {}
CHANGED = {
    "layer": "counts",
    "min_cells": 20,
    "aggregation_method": "mean",
    "min_genes": 50,
    "filter_zeros": False,
}


def _single_cell():
    """8 donors (4 ctrl, 4 treat) x 80 cells of one cell type, 400 genes, some
    genes never seen (so filter_zeros has genes to drop)."""
    rng = np.random.default_rng(3)
    donors = [f"D{i}" for i in range(1, 9)]
    obs = pd.DataFrame(
        {
            "donor": np.repeat(donors, 80),
            "condition": np.repeat(["ctrl"] * 4 + ["treat"] * 4, 80),
            "cell_type": "T",
        },
        index=[f"c{i}" for i in range(640)],
    )
    mu = rng.gamma(2.0, 1.0, size=400)[None, :] * np.ones((640, 1))
    mu[320:, :40] *= 3
    mu[:, -5:] = 0
    counts = rng.poisson(mu).astype(np.float32)
    adata = ad.AnnData(
        X=counts.copy(),
        obs=obs,
        var=pd.DataFrame(index=[f"G{i:03d}" for i in range(400)]),
    )
    adata.layers["counts"] = counts
    return adata


def _session(tmp_path, monkeypatch, **settings):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "in" / "sc.h5ad"
    source.parent.mkdir()
    _single_cell().write_h5ad(source)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.load_modality("sc", str(source), adapter="transcriptomics_single_cell")
    tools = factory_tools(de_module.de_analysis_expert, de_module.__name__, dm)
    reply = tools["create_pseudobulk_matrix"].invoke(
        {
            "modality_name": "sc",
            "sample_col": "donor",
            "celltype_col": "cell_type",
            **settings,
        }
    )
    assert "Pseudobulk matrix created" in reply, reply
    return dm, source, tools


def _replay(dm, source):
    notebook = dm.export_notebook("pb", "pseudobulk", filter_strategy="successful")
    result = runner.invoke(
        app,
        ["--verify", str(notebook), "--input", str(source),
         "-w", str(dm.workspace_path), "--json"],
    )  # fmt: skip
    out = json.loads(result.stdout)
    return result.exit_code, out


@needs_kernel
@pytest.mark.parametrize("settings", [DEFAULTS, CHANGED], ids=["defaults", "changed"])
def test_the_exported_pseudobulk_matrix_replays_to_match(
    tmp_path, monkeypatch, settings
):
    dm, source, _ = _session(tmp_path, monkeypatch, **settings)
    code, out = _replay(dm, source)
    assert (code, out["verdict"]) == (0, "MATCH"), out


def _with_de(tmp_path, monkeypatch):
    """The pseudobulk session, then a DE step on the saved pseudobulk matrix: it
    writes its results into that matrix after the session saved it."""
    dm, source, tools = _session(tmp_path, monkeypatch)
    reply = tools["run_differential_expression"].invoke(
        {
            "modality_name": "sc_pseudobulk",
            "groupby": "condition",
            "group1": "ctrl",
            "group2": "treat",
            "method": "deseq2",
            "save_result": True,
        }
    )
    assert "Complete" in reply, reply
    return dm, source


def _code_cells(notebook):
    return [
        "".join(cell["source"])
        for cell in json.loads(notebook.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]


def _assert_saves_in_place(dm):
    """The pseudobulk matrix is saved right after its own cell, before the DE cell
    changes it; the DE result is saved after the DE cell."""
    notebook = dm.export_notebook("pb", "pseudobulk", filter_strategy="successful")
    code = _code_cells(notebook)

    def where(text):
        (found,) = [i for i, source in enumerate(code) if text in source]
        return found

    pseudobulk_cell = where("PseudobulkService().aggregate_to_pseudobulk(")
    de_cell = where("run_pydeseq2_from_pseudobulk(")
    assert where("'sc_pseudobulk' }.h5ad") == pseudobulk_cell + 1
    assert de_cell > pseudobulk_cell + 1
    assert where("'sc_pseudobulk_de_ctrl_vs_treat' }.h5ad") > de_cell


def test_a_save_is_written_where_the_session_saved_it(tmp_path, monkeypatch):
    dm, _ = _with_de(tmp_path, monkeypatch)
    _assert_saves_in_place(dm)


def test_a_record_without_what_each_step_stored_places_saves_the_same(
    tmp_path, monkeypatch
):
    """An older session's steps do not record what they stored: a save just
    before a step is still taken as made inside it."""
    dm, _ = _with_de(tmp_path, monkeypatch)
    for activity in dm.provenance.activities:
        activity.pop("stored_modalities", None)
    _assert_saves_in_place(dm)


def test_a_save_made_between_two_steps_is_written_after_the_first(
    tmp_path, monkeypatch
):
    """Saved on its own after the pseudobulk step: the DE step after it stores
    the same name (its results go into it), but did not make the save."""
    dm, _, tools = _session(tmp_path, monkeypatch, save_result=False)
    dm.save_modality("sc_pseudobulk", "sc_pseudobulk.h5ad")
    reply = tools["run_differential_expression"].invoke(
        {
            "modality_name": "sc_pseudobulk",
            "groupby": "condition",
            "group1": "ctrl",
            "group2": "treat",
            "method": "deseq2",
            "save_result": True,
        }
    )
    assert "Complete" in reply, reply
    _assert_saves_in_place(dm)


@needs_kernel
def test_a_de_step_after_the_saved_pseudobulk_matrix_replays_to_match(
    tmp_path, monkeypatch
):
    dm, source = _with_de(tmp_path, monkeypatch)
    code, out = _replay(dm, source)
    assert (code, out["verdict"]) == (0, "MATCH"), out


def test_the_step_records_what_it_read_and_made(tmp_path, monkeypatch):
    """The session's record names the modality the step read and the one it made,
    so the notebook knows which modality ``adata`` holds after the cell."""
    dm, _, _ = _session(tmp_path, monkeypatch)
    (step,) = [
        a
        for a in dm.provenance.activities
        if a.get("type") == "create_pseudobulk_matrix"
    ]
    assert [e["name"] for e in step["inputs"]] == ["sc"]
    assert [e["name"] for e in step["outputs"]] == ["sc_pseudobulk"]


@pytest.mark.parametrize(
    "gene_subset", [None, [f"G{i:03d}" for i in range(100)]], ids=["all", "subset"]
)
def test_the_step_records_every_setting_the_session_used(gene_subset):
    """The tool never passes a gene subset; a Python caller can, and its cell gets
    the same subset."""
    service = pseudobulk.PseudobulkService()
    _, _, ir = service.aggregate_to_pseudobulk(
        _single_cell(),
        sample_col="donor",
        celltype_col="cell_type",
        gene_subset=gene_subset,
        **CHANGED,
    )
    assert ir.parameters == {
        "sample_col": "donor",
        "celltype_col": "cell_type",
        **CHANGED,
        "gene_subset": gene_subset,
    }
    code = ir.render()
    for name, value in ir.parameters.items():
        assert f"{name}={value!r}" in code, (name, code)
