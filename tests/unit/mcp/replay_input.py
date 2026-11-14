"""A real, small input file and replay scripts that load it, for the tests that
run the engine: a first run that loads no data from its input fails."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

DATA = Path(__file__).parent / "data"


def script(name: str, tmp_path: Path, data: Path) -> Path:
    """The replay script *name* of ``data/`` with ``__INPUT__`` set to *data*."""
    text = (DATA / name).read_text().replace("__INPUT__", str(data))
    target = tmp_path / name
    target.write_text(text)
    return target


def make_input(path: Path) -> None:
    counts = np.zeros((128, 2601))
    counts[:, :240] = 1
    counts[65, 1] = 1000
    genes = ["MT-ND1", "RPS3", "ACTB"] + [f"GENE{i}" for i in range(2598)]
    adata = ad.AnnData(
        sparse.csr_matrix(counts),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(128)]),
        var=pd.DataFrame({"source": ["synthetic"] * 2601}, index=genes),
    )
    adata.write_h5ad(path)


def loading(name: str, tmp_path: Path, data: Path) -> Path:
    """Write a real input to *data* and return the script *name* for it."""
    from tests.unit.mcp.transcriptomics import quality_service

    quality_service()  # the replay loads it with the transcriptomics adapter
    make_input(data)
    return script(name, tmp_path, data)
