"""A session that imported a bulk count table (CSV or TSV) replays from that table.

import_bulk_counts read the table itself (layout detected, genes on columns,
raw counts in layers['counts']), but its step recorded
``sc.read_csv(file_path)`` and the notebook began with ``ad.read_h5ad(input_data)``:
replayed with the table as ``--input``, the notebook failed at its first cell.
The import step now reads ``input_data`` the way the tool did, the export lists
the table as the file the session loaded, and the replay compares the files the
session saved. Skipped without the transcriptomics package or a Jupyter kernel.
"""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from tests.unit.cli.test_replay_commands import app, needs_kernel, runner

tx = pytest.importorskip("lobster.agents.transcriptomics.transcriptomics_expert")
de = pytest.importorskip("lobster.agents.transcriptomics.de_analysis_expert")

from lobster.core.runtime.data_manager import DataManagerV2  # noqa: E402
from lobster.testing.provenance_replay import factory_tools  # noqa: E402

SAMPLES = [f"C{i}" for i in range(1, 7)] + [f"T{i}" for i in range(1, 7)]
# Ensembl IDs: the tool finds the genes axis itself (genes_axis="auto")
GENES = [f"ENSG{i:011d}" for i in range(300)]
# layout -> (file name, separator): genes on rows in a CSV, samples on rows in a TSV
LAYOUTS = {"genes_rows": ("counts.csv", ","), "samples_rows": ("counts.tsv", "\t")}


def _counts():
    rng = np.random.default_rng(5)
    mu = rng.gamma(2.0, 50.0, size=len(GENES))[:, None] * np.ones(len(SAMPLES))
    mu[:30, 6:] *= 4
    return pd.DataFrame(
        rng.negative_binomial(5, 5 / (5 + mu)), index=GENES, columns=SAMPLES
    )


def _write(table, folder, layout):
    name, sep = LAYOUTS[layout]
    if layout == "samples_rows":
        table = table.T
    table.index.name = "gene_id" if layout == "genes_rows" else "sample"
    path = folder / name
    table.to_csv(path, sep=sep)
    return path


def _session(tmp_path, monkeypatch, layout, method):
    """Import the table, merge the sample sheet, run DE with a save, export."""
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / "in"
    inputs.mkdir()
    source = _write(_counts(), inputs, layout)
    meta = inputs / "meta.csv"
    pd.DataFrame(
        {"sample_id": SAMPLES, "condition": ["ctrl"] * 6 + ["treat"] * 6}
    ).to_csv(meta, index=False)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    tools = {
        **factory_tools(tx.transcriptomics_expert, tx.__name__, dm),
        **factory_tools(de.de_analysis_expert, de.__name__, dm),
    }
    reply = tools["import_bulk_counts"].invoke(
        {"modality_name": "bulk", "file_path": str(source)}
    )
    assert "Successfully imported" in reply, reply
    reply = tools["merge_sample_metadata"].invoke(
        {"modality_name": "bulk", "metadata_file": str(meta)}
    )
    assert "Successfully merged" in reply, reply
    reply = tools["run_differential_expression"].invoke(
        {
            "modality_name": "bulk",
            "groupby": "condition",
            "group1": "ctrl",
            "group2": "treat",
            "method": method,
            "save_result": True,
        }
    )
    assert "Complete" in reply, reply
    notebook = dm.export_notebook("de", "bulk import", filter_strategy="successful")
    return dm, notebook, source


def _verify(dm, notebook, source):
    result = runner.invoke(
        app,
        ["--verify", str(notebook), "--input", str(source),
         "-w", str(dm.workspace_path), "--json"],
    )  # fmt: skip
    return result, json.loads(result.stdout)


@needs_kernel
@pytest.mark.parametrize("method", ["wilcoxon", "deseq2"])
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_an_imported_count_table_replays_to_match(
    tmp_path, monkeypatch, layout, method
):
    if method == "deseq2":
        pytest.importorskip("pydeseq2")
    dm, notebook, source = _session(tmp_path, monkeypatch, layout, method)
    # the replay reads the file it is given: a copy elsewhere, the original gone
    copy = tmp_path / "elsewhere" / source.name
    copy.parent.mkdir()
    shutil.copy(source, copy)
    source.unlink()

    result, out = _verify(dm, notebook, copy)

    assert out["verdict"] == "MATCH", out
    assert result.exit_code == 0, (result.output, result.exception)


@needs_kernel
def test_a_tampered_table_is_never_a_match(tmp_path, monkeypatch):
    """The imported table with one count changed: refused or a mismatch."""
    dm, notebook, source = _session(tmp_path, monkeypatch, "genes_rows", "wilcoxon")
    other = _counts()
    other.iloc[0, 0] += 7
    changed = _write(other, tmp_path, "genes_rows")

    result, out = _verify(dm, notebook, changed)

    assert out["verdict"] in ("MISMATCH", "REFUSED"), out
    assert result.exit_code != 0


def test_the_export_lists_the_table_as_the_file_loaded(tmp_path, monkeypatch):
    """The record names the table (its SHA-256), the notebook starts by reading
    ``input_data`` with the layout the tool found, and no cell reads an h5ad."""
    import hashlib

    dm, notebook, source = _session(tmp_path, monkeypatch, "samples_rows", "wilcoxon")
    text = open(notebook).read()
    cells = json.loads(text)["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")

    assert "read_h5ad(input_data)" not in code
    assert "genes_axis='columns'" in code
    assert hashlib.sha256(source.read_bytes()).hexdigest() in text


def _cells_until_last_import(notebook, input_data):
    """Run the notebook's code cells (``input_data`` as given) through its last
    count-table import; return the ``adata`` each import cell left."""
    cells = [c for c in json.loads(open(notebook).read())["cells"]]
    code = [c for c in cells if c["cell_type"] == "code"]
    imports = [
        i
        for i, c in enumerate(code)
        if "read_bulk_count_table(" in "".join(c["source"])
    ]
    scope, made = {}, []
    for i, cell in enumerate(code[: imports[-1] + 1]):
        exec("".join(cell["source"]), scope)  # nosec B102 - the notebook under test
        if "parameters" in cell["metadata"].get("tags", []):
            scope["input_data"] = str(input_data)
        if i in imports:
            made.append(scope["adata"].copy())
    return made


@pytest.mark.parametrize("first", ["table", "h5ad"])
def test_a_later_table_import_reads_its_own_file(tmp_path, monkeypatch, first):
    """Two files in one session (a table or an h5ad, then a table holding three
    times its counts): run outside a replay, the notebook builds the later
    modality from the later file, never from the notebook's input."""
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / "in"
    inputs.mkdir()
    counts = _counts()
    a = _write(counts.copy(), inputs, "genes_rows")
    b = inputs / "b.csv"
    (counts * 3).rename_axis("gene_id").to_csv(b)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    tools = factory_tools(tx.transcriptomics_expert, tx.__name__, dm)
    if first == "table":
        reply = tools["import_bulk_counts"].invoke(
            {"modality_name": "a", "file_path": str(a)}
        )
        assert "Successfully imported" in reply, reply
    else:
        import anndata

        a = inputs / "a.h5ad"
        anndata.AnnData(counts.T.astype(np.float32)).write_h5ad(a)
        dm.load_modality("a", a, adapter="transcriptomics_bulk")
    reply = tools["import_bulk_counts"].invoke(
        {"modality_name": "b", "file_path": str(b)}
    )
    assert "Successfully imported" in reply, reply
    notebook = dm.export_notebook("two", "two files", filter_strategy="successful")

    made = _cells_until_last_import(notebook, a)

    # known answer: the later file's counts, samples on rows (numpy, not the reader)
    expected = (counts.to_numpy() * 3).T
    np.testing.assert_array_equal(np.asarray(made[-1].X), expected)
    assert list(made[-1].obs_names) == SAMPLES
    if first == "table":  # the notebook's input stays the first file
        np.testing.assert_array_equal(np.asarray(made[0].X), counts.to_numpy().T)
