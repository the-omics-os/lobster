"""Artifact writer — runs INSIDE the production image (pandas 3 / anndata 0.13).

Invoked by ``tests/integration/test_pandas3_artifact_gate.py`` via
``docker run --network none --entrypoint /opt/venv/bin/python``. Writes
proteomics-shaped H5AD (and Zarr) artifacts through the RAW ``write_h5ad`` path
— the same call the three direct writers use (custom_code_execution_service.py,
core/notebooks/exporter.py, core/runtime/data_manager.py) that bypass the
managed backend's column sanitizer. This reproduces the pandas-3 string
representation customer artifacts carry across a downgrade.

Feature layout per file (4 vars): F0=contaminant, F1=clean, F2=contaminant,
F3=clean. A correct reader keeps {F1, F3} and leaves the numeric matrix intact.
"""
import json
import sys

import numpy as np
import pandas as pd
import anndata as ad

OUT = "/artifacts"
FLAG = [True, False, True, False]


def contaminant_col(encoding):
    plus = {True: "+", False: ""}
    tf = {True: "True", False: "False"}
    tens = {True: "1", False: "0"}
    builders = {
        "text_plus": lambda: pd.Series([plus[v] for v in FLAG]),
        "text_10": lambda: pd.Series([tens[v] for v in FLAG]),
        "text_tf": lambda: pd.Series([tf[v] for v in FLAG]),
        "bool": lambda: pd.Series(FLAG, dtype=bool),
        "numeric": lambda: pd.Series([1 if v else 0 for v in FLAG], dtype="int64"),
        "cat_text": lambda: pd.Series(pd.Categorical([plus[v] for v in FLAG])),
        "cat_num": lambda: pd.Series(pd.Categorical([1 if v else 0 for v in FLAG])),
    }
    return builders[encoding]()


ENCODINGS = ["text_plus", "text_10", "text_tf", "bool", "numeric", "cat_text", "cat_num"]

manifest = {
    "stack": {"pandas": pd.__version__, "anndata": ad.__version__},
    "flag_rows_true": FLAG,
    "expected_kept": ["F1", "F3"],
    "files": [],
}

for enc in ENCODINGS:
    X = np.arange(4 * 3, dtype=np.float64).reshape(3, 4) + 0.5  # 3 obs x 4 var
    var = pd.DataFrame(index=[f"F{i}" for i in range(4)])
    var["is_contaminant"] = contaminant_col(enc).values
    var["is_reverse"] = contaminant_col(enc).values
    var["extra_label"] = pd.Series(["batchA", "batchB", "batchA", "batchC"]).values
    var["num_as_text"] = pd.Series(["1.5", "2.5", "3.5", "4.5"]).values
    adata = ad.AnnData(
        X=X, obs=pd.DataFrame(index=[f"S{i}" for i in range(3)]), var=var
    )
    written = {c: str(adata.var[c].dtype) for c in adata.var.columns}
    h5ad = f"{OUT}/contam_{enc}.h5ad"
    adata.write_h5ad(h5ad)  # RAW write — the bypass path
    entry = {"encoding": enc, "h5ad": f"contam_{enc}.h5ad",
             "written_var_dtypes": written}
    try:
        adata.write_zarr(f"{OUT}/contam_{enc}.zarr")
        entry["zarr"] = f"contam_{enc}.zarr"
    except Exception as e:  # noqa: BLE001
        entry["zarr_error"] = f"{type(e).__name__}: {e}"
    manifest["files"].append(entry)

with open(f"{OUT}/manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
print(f"wrote {len(manifest['files'])} artifact sets under {OUT} "
      f"(pandas {pd.__version__} / anndata {ad.__version__})")
sys.exit(0)
