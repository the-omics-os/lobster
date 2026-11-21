"""The notebook cell of a DE export writes the table the session wrote.

The session renames columns to the publication names (logFC to
log2FoldChange, FDR to padj, ...), sorts only by a column the table has, and
names the gene index ``gene``. The cell did none of these: it sorted by a name
the table may not have (a KeyError, or a sort before the rename), and wrote an
unnamed index. It now does what the session does.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

pytestmark = pytest.mark.unit

DE = "lobster.agents.transcriptomics.de_analysis_expert"


def _table(columns):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {name: rng.uniform(0, 1, 6) for name in columns},
        index=[f"G{i}" for i in range(6)],
    )


def _export(tmp_path, monkeypatch, table, sort_by):
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    adata = ad.AnnData(np.ones((2, 6)), var=pd.DataFrame(index=table.index))
    adata.uns["de_results_a_vs_b"] = {"results_df": table}
    dm.store_modality("bulk", adata)
    steps = []
    log = dm.log_tool_usage
    monkeypatch.setattr(
        dm, "log_tool_usage", lambda **kw: steps.append(kw["ir"]) or log(**kw)
    )
    reply = factory_tools(de_analysis_expert, DE, dm)["export_de_results"].invoke(
        {
            "modality_name": "bulk",
            "de_results_key": "de_results_a_vs_b",
            "output_path": "t.csv",
            "sort_by": sort_by,
        }
    )
    session = (dm.exports_dir / "t.csv").read_text()

    replay = tmp_path / "replay"
    replay.mkdir()
    monkeypatch.chdir(replay)
    from pathlib import Path

    exec(steps[-1].render(), {"adata": adata, "pd": pd, "Path": Path})
    return reply, session, (replay / "exports" / "t.csv").read_text()


@pytest.mark.parametrize(
    "columns, sort_by",
    [
        (["logFC", "FDR", "AveExpr", "pvalue"], "padj"),  # renamed, then sorted
        (["log2FoldChange", "padj", "pvalue"], "padj"),
        (["log2FoldChange", "padj", "pvalue"], "no_such_column"),  # not sorted
    ],
)
def test_the_cell_writes_the_table_the_session_wrote(
    tmp_path, monkeypatch, columns, sort_by
):
    reply, session, cell = _export(tmp_path, monkeypatch, _table(columns), sort_by)

    assert session.splitlines()[0].startswith("gene,"), reply
    assert cell == session
