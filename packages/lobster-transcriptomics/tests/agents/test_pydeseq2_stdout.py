"""A pyDESeq2 run writes nothing to stdout.

``lobster query --json`` keeps stdout for the JSON document; pyDESeq2's own
``print`` of the results table (``DeseqStats.summary`` and ``lfc_shrink``) broke
``lobster query ... --json | jq``. Checked at the file-descriptor level.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

pytest.importorskip("pydeseq2")

DE = "lobster.agents.transcriptomics.de_analysis_expert"
SAMPLES = [f"C{i}" for i in range(1, 5)] + [f"T{i}" for i in range(1, 5)]
CALLS = {
    "run_bulk_de_direct": {"group_key": "condition", "group1": "ctrl"}
    | {"group2": "treat"},
    "run_differential_expression": {"groupby": "condition", "group1": "ctrl"}
    | {"group2": "treat", "method": "deseq2"},
    "run_de_with_formula": {"formula": "~condition"}
    | {"contrast": ["condition", "treat", "ctrl"]},
}


def _bulk():
    rng = np.random.default_rng(3)
    mu = rng.gamma(2.0, 50.0, size=200)[None, :] * np.ones((len(SAMPLES), 1))
    mu[4:, :20] *= 4
    adata = ad.AnnData(
        X=rng.negative_binomial(5, 5 / (5 + mu)).astype(np.float32),
        obs=pd.DataFrame({"condition": ["ctrl"] * 4 + ["treat"] * 4}, index=SAMPLES),
        var=pd.DataFrame(index=[f"GENE{i:04d}" for i in range(200)]),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


@pytest.mark.parametrize("tool", sorted(CALLS))
def test_a_pydeseq2_run_prints_nothing_to_stdout(tmp_path, monkeypatch, capfd, tool):
    monkeypatch.chdir(tmp_path)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    dm.store_modality("bulk", _bulk())
    tools = factory_tools(de_analysis_expert, DE, dm)
    capfd.readouterr()  # what set-up printed is not the run's
    key = "pseudobulk_modality" if tool == "run_de_with_formula" else "modality_name"
    reply = tools[tool].invoke({key: "bulk", "save_result": False, **CALLS[tool]})
    out, _ = capfd.readouterr()
    assert "Complete" in reply or "complete" in reply, reply
    assert out == "", out[:500]
