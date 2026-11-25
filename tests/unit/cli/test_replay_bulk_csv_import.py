"""A session that imported a bulk count table (CSV or TSV) replays from that table.

import_bulk_counts read the table itself (layout detected, genes on columns,
raw counts in layers['counts']), but its step recorded
``sc.read_csv(file_path)`` and the notebook began with ``ad.read_h5ad(input_data)``:
replayed with the table as ``--input``, the notebook failed at its first cell.
The import step now reads ``input_data`` the way the tool did, the export lists
the table as the file the session loaded, and the replay compares the files the
session saved. A merged sample sheet stays a path in the notebook, never its
content, so a replay refuses that session and names the sheet and the step.
Skipped without the transcriptomics package or a Jupyter kernel.
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


# a cell value of the sheet that must never leave the session
PRIVATE = "patient-7731-tumour-site"


def _sheet(path):
    pd.DataFrame(
        {
            "sample_id": SAMPLES,
            "condition": ["ctrl"] * 6 + ["treat"] * 6,
            "note": [PRIVATE] + [""] * 11,
        }
    ).to_csv(path, index=False)
    return path


def _session(tmp_path, monkeypatch, layout, sheet=None):
    """Import the table, save it, and export; ``sheet`` ('outside' or
    'workspace'): merge a sample sheet kept there and run DE with a save."""
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / "in"
    inputs.mkdir()
    source = _write(_counts(), inputs, layout)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    tools = {
        **factory_tools(tx.transcriptomics_expert, tx.__name__, dm),
        **factory_tools(de.de_analysis_expert, de.__name__, dm),
    }
    reply = tools["import_bulk_counts"].invoke(
        {"modality_name": "bulk", "file_path": str(source)}
    )
    assert "Successfully imported" in reply, reply
    if sheet is None:
        dm.save_modality("bulk", dm.workspace_path / "data" / "bulk.h5ad")
    else:
        folder = inputs if sheet == "outside" else dm.workspace_path
        reply = tools["merge_sample_metadata"].invoke(
            {"modality_name": "bulk", "metadata_file": str(_sheet(folder / "meta.csv"))}
        )
        assert "Successfully merged" in reply, reply
        reply = tools["run_differential_expression"].invoke(
            {
                "modality_name": "bulk",
                "groupby": "condition",
                "group1": "ctrl",
                "group2": "treat",
                "method": "wilcoxon",
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
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_an_imported_count_table_replays_to_match(tmp_path, monkeypatch, layout):
    dm, notebook, source = _session(tmp_path, monkeypatch, layout)
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
    dm, notebook, source = _session(tmp_path, monkeypatch, "genes_rows")
    other = _counts()
    other.iloc[0, 0] += 7
    changed = _write(other, tmp_path, "genes_rows")

    result, out = _verify(dm, notebook, changed)

    assert out["verdict"] in ("MISMATCH", "REFUSED"), out
    assert result.exit_code != 0


@needs_kernel
@pytest.mark.parametrize("sheet", ["outside", "workspace"])
def test_a_merged_sample_sheet_is_refused_naming_the_sheet_and_step(
    tmp_path, monkeypatch, sheet
):
    """The merge cell reads the sheet from its path, which a replay never reads:
    refused, wherever the sheet is, with the step and the sheet named."""
    dm, notebook, source = _session(tmp_path, monkeypatch, "genes_rows", sheet)

    result, out = _verify(dm, notebook, source)

    assert out["verdict"] == "REFUSED", out
    assert result.exit_code == 4
    reason = out["report"]["refusal"]["reason"]
    assert "merge_sample_metadata" in reason and "meta.csv" in reason, reason


def test_no_sample_sheet_content_leaves_the_session(tmp_path, monkeypatch):
    """The notebook and the provenance record name the sheet's path; none of its
    cells is written into either."""
    dm, notebook, source = _session(tmp_path, monkeypatch, "genes_rows", "outside")
    record = json.dumps([a for a in dm.provenance.activities], default=str) + "".join(
        p.read_text() for p in dm.workspace_path.rglob("*.jsonl")
    )

    assert PRIVATE not in open(notebook).read()
    assert PRIVATE not in record
    assert str((tmp_path / "in" / "meta.csv").resolve()) in open(notebook).read()


def test_the_export_lists_the_table_as_the_file_loaded(tmp_path, monkeypatch):
    """The record names the table (its SHA-256), the notebook starts by reading
    ``input_data`` with the layout the tool found, and no cell reads an h5ad."""
    import hashlib

    dm, notebook, source = _session(tmp_path, monkeypatch, "samples_rows")
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
