"""Each exported preprocessing step prints its own settings.

The step code printed ``method`` and ``log_transform`` through Python f-string
braces, so at replay they read the notebook-level names from the parameters
cell, which holds only the first step's value. Every later step then reported
the first step's method.
"""

import contextlib
import io
from types import SimpleNamespace

import anndata
import nbformat
import numpy as np
import pandas as pd

from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.provenance.provenance import ProvenanceTracker
from lobster.services.quality.metabolomics_preprocessing_service import (
    MetabolomicsPreprocessingService,
)

X = np.array(
    [
        [1.0, 4.0, np.nan],
        [2.0, np.nan, 6.0],
        [np.nan, 8.0, 3.0],
        [4.0, 2.0, 5.0],
    ]
)


def _adata():
    adata = anndata.AnnData(X.copy())
    adata.obs_names = [f"s{i}" for i in range(X.shape[0])]
    adata.var_names = ["m0", "m1", "m2"]
    adata.obs["sample_type"] = pd.Categorical(["sample"] * X.shape[0])
    return adata


def _export_and_run(tmp_path, monkeypatch):
    service = MetabolomicsPreprocessingService()
    adata = _adata()
    imputed, _, ir_impute = service.impute_missing_values(adata, method="min")
    normalized, _, ir_norm = service.normalize(
        imputed, method="tic", log_transform=False
    )
    _, _, ir_norm2 = service.normalize(normalized, method="median", log_transform=True)

    adata.write_h5ad(tmp_path / "dataset.h5ad")
    tracker = ProvenanceTracker()
    tracker.create_activity(
        activity_type="handle_missing_values", agent="test", ir=ir_impute
    )
    tracker.create_activity(
        activity_type="normalize_metabolomics", agent="test", ir=ir_norm
    )
    tracker.create_activity(
        activity_type="normalize_metabolomics", agent="test", ir=ir_norm2
    )
    context = SimpleNamespace(
        workspace_path=tmp_path, workspace_dir=tmp_path, modalities={"m": adata}
    )
    path = NotebookExporter(tracker, context).export("metab_steps")
    notebook = nbformat.read(path, as_version=4)

    monkeypatch.chdir(tmp_path)
    namespace, out = {}, io.StringIO()
    with contextlib.redirect_stdout(out):
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                exec(
                    compile(cell.source, f"metab_cell_{index}", "exec"), namespace
                )  # nosec B102
    return notebook, namespace, out.getvalue(), normalized


def test_each_step_prints_its_own_method(tmp_path, monkeypatch):
    notebook, _, printed, _ = _export_and_run(tmp_path, monkeypatch)

    assert "using min" in printed
    assert "Normalized using tic, log_transform=False" in printed
    assert "Normalized using median, log_transform=True" in printed
    assert "Normalized using min" not in printed
    # No step reads a notebook-level ``method`` any more, so none is offered.
    params = next(
        c.source for c in notebook.cells if "parameters" in c.metadata.get("tags", [])
    )
    assert "\nmethod = " not in params


def test_printed_lines_and_values_match_hand_computation(tmp_path, monkeypatch):
    _, namespace, printed, normalized = _export_and_run(tmp_path, monkeypatch)

    # 3 of the 12 values are missing: 25.0%.
    assert "Imputed 3 values (25.0%) using min\n" in printed
    assert "Normalized using tic, log_transform=False\n" in printed

    # min imputation: each missing value becomes half its column's observed minimum.
    imputed = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        imputed[np.isnan(col), j] = np.min(col[~np.isnan(col)]) / 2
    np.testing.assert_allclose(np.asarray(namespace["adata_imputed"].X), imputed)
    # TIC: divide each row by (row total / median row total).
    totals = imputed.sum(axis=1)
    expected = imputed / (totals / np.median(totals))[:, None]
    np.testing.assert_allclose(np.asarray(normalized.X), expected)


def test_batch_step_prints_its_own_method():
    ir = MetabolomicsPreprocessingService()._create_ir_correct_batch_effects(
        batch_key="batch", method="median_centering", qc_label="QC"
    )
    namespace = {"method": "combat", "stats": {"n_batches": 2}}
    out = io.StringIO()
    line = next(l for l in ir.render().splitlines() if l.startswith("print("))
    with contextlib.redirect_stdout(out):
        exec(line, namespace)  # nosec B102
    assert out.getvalue() == "Batch correction: median_centering, 2 batches\n"
