"""import_bulk_counts must put samples in obs for a CSV in either layout."""

import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.transcriptomics_expert import (
    transcriptomics_expert,
)
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

MODULE = "lobster.agents.transcriptomics.transcriptomics_expert"


@pytest.mark.parametrize("genes_as_rows", [True, False])
def test_csv_import_puts_samples_in_obs(tmp_path, genes_as_rows):
    samples = [f"S{i}" for i in range(1, 7)]
    genes = [f"GENE{i:04d}" for i in range(1_000)]
    counts = pd.DataFrame(
        np.arange(len(genes) * len(samples)).reshape(len(genes), len(samples)),
        index=genes,
        columns=samples,
    )
    path = tmp_path / "counts.csv"
    (counts if genes_as_rows else counts.T).to_csv(path)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    tools = factory_tools(transcriptomics_expert, MODULE, dm)

    tools["import_bulk_counts"].invoke(
        {"modality_name": "bulk", "file_path": str(path)}
    )

    adata = dm.get_modality("bulk")
    assert list(adata.obs_names) == samples
    assert list(adata.var_names) == genes
    np.testing.assert_array_equal(adata.X, counts.T.to_numpy())


def _import(tmp_path, genes, samples, genes_as_rows, **args):
    counts = pd.DataFrame(
        np.arange(len(genes) * len(samples)).reshape(len(genes), len(samples)),
        index=genes,
        columns=samples,
    )
    path = tmp_path / "counts.csv"
    (counts if genes_as_rows else counts.T).to_csv(path)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    tools = factory_tools(transcriptomics_expert, MODULE, dm)
    reply = tools["import_bulk_counts"].invoke(
        {"modality_name": "bulk", "file_path": str(path), **args}
    )
    return dm, counts, reply


def _assert_imported(dm, counts, reply, axis, reason, requested="auto"):
    adata = dm.get_modality("bulk")
    assert list(adata.obs_names) == list(counts.columns)
    assert list(adata.var_names) == list(counts.index)
    np.testing.assert_array_equal(adata.X, counts.T.to_numpy())
    assert f"- Genes axis: {axis} ({reason})" in reply
    activity = [
        a for a in dm.provenance.activities if a["type"] == "import_bulk_counts"
    ][-1]
    for recorded in (activity["parameters"], activity["ir"]["parameters"]):
        assert recorded["genes_axis"] == requested
        assert recorded["genes_axis_used"] == axis


def _assert_refused(dm, reply, rows, columns):
    assert "bulk" not in dm.list_modalities()
    assert "Cannot tell which axis of counts.csv holds the genes" in reply
    # The agent sees the labels, so it can choose (or ask) instead of guessing.
    assert f"Rows start: {', '.join(rows[:3])}." in reply
    assert f"Columns start: {', '.join(columns[:3])}." in reply
    assert "Pick the axis whose labels are genes, or ask the user" in reply
    assert "genes_axis='rows' or 'columns'" in reply


def _side(genes_as_rows):
    return "rows" if genes_as_rows else "columns"


SYMBOLS = [f"G{i}" for i in range(10_000)]
SAMPLES = [f"P{i:04d}" for i in range(1_000)]


@pytest.mark.parametrize("genes_as_rows", [True, False])
def test_filtered_genes_by_few_samples_follow_the_shape(tmp_path, genes_as_rows):
    dm, counts, reply = _import(tmp_path, SYMBOLS[:5_000], SAMPLES[:6], genes_as_rows)
    _assert_imported(dm, counts, reply, _side(genes_as_rows), "shape")


@pytest.mark.parametrize("genes_as_rows", [True, False])
def test_a_whole_transcriptome_of_symbols_follows_the_shape(tmp_path, genes_as_rows):
    dm, counts, reply = _import(tmp_path, SYMBOLS, SAMPLES[:50], genes_as_rows)
    _assert_imported(dm, counts, reply, _side(genes_as_rows), "shape")


@pytest.mark.parametrize(
    "n_genes,n_samples",
    [(500, 1_000), (5_000, 50), (8, 40)],
    ids=["panel", "filtered", "small_panel"],
)
@pytest.mark.parametrize("genes_as_rows", [True, False])
def test_symbols_that_shape_cannot_settle_are_refused(
    tmp_path, genes_as_rows, n_genes, n_samples
):
    genes, samples = SYMBOLS[:n_genes], SAMPLES[:n_samples]
    dm, _, reply = _import(tmp_path, genes, samples, genes_as_rows)
    rows, columns = (genes, samples) if genes_as_rows else (samples, genes)
    _assert_refused(dm, reply, rows, columns)


@pytest.mark.parametrize("genes_as_rows", [True, False])
def test_genes_axis_settles_a_refused_layout(tmp_path, genes_as_rows):
    side = _side(genes_as_rows)
    dm, counts, reply = _import(
        tmp_path, SYMBOLS[:500], SAMPLES, genes_as_rows, genes_axis=side
    )
    _assert_imported(dm, counts, reply, side, "given", requested=side)


@pytest.mark.parametrize(
    "n_genes,n_samples,genes_as_rows",
    [(25, 30, False), (30, 120, True), (30, 120, False)],
    ids=["ids_on_columns_30_rows", "panel_ids_on_rows", "panel_ids_on_columns"],
)
def test_gene_ids_decide_the_axis(tmp_path, n_genes, n_samples, genes_as_rows):
    genes = [f"ENSG{i:011d}.4" for i in range(n_genes)]
    dm, counts, reply = _import(tmp_path, genes, SAMPLES[:n_samples], genes_as_rows)
    _assert_imported(dm, counts, reply, _side(genes_as_rows), "Ensembl/RefSeq gene IDs")


def test_an_unknown_genes_axis_is_refused(tmp_path):
    dm, _, reply = _import(tmp_path, ["A", "B"], ["S1"], True, genes_axis="cols")
    assert "bulk" not in dm.list_modalities()
    assert "genes_axis must be 'auto', 'rows' or 'columns'" in reply
