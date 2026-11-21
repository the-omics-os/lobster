"""A DE export takes only csv or xlsx, and its notebook cell names the writer.

``output_format`` was not checked and the cell rendered it raw, as
``results_df.to_{{ output_format }}(...)``: any text in it became code when the
notebook ran. A format other than csv or xlsx is now refused before anything is
written or logged, and the cell writes with ``to_csv`` or ``to_excel``, never
with the given text (an xlsx cell called ``to_xlsx``, which pandas does not
have).
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.testing.provenance_replay import RecordingDataManager, factory_tools

MODULE = "lobster.agents.transcriptomics.de_analysis_expert"
KEY = "de_results_treat_vs_ctrl"
TOOLS = {
    "export_de_results": {},
    "extract_and_export_de_results": {"shrink_lfc": False, "padj_threshold": 1.0},
}
INJECTED = "csv(__import__('os').getcwd()) or results_df.to_csv"


def _session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    table = pd.DataFrame(
        {"padj": [0.01, 0.2], "log2FoldChange": [2.0, -0.5], "baseMean": [95.5, 30.25]},
        index=["GENE1", "GENE2"],
    )
    adata = ad.AnnData(np.zeros((2, 2)))
    adata.uns[KEY] = {"results_df": table}
    adata.uns["de_current_results_key"] = KEY
    dm = RecordingDataManager({"de": adata}, workspace_path=tmp_path / "ws")
    dm.exports_dir = dm.workspace_path / "exports"
    return dm, factory_tools(de_analysis_expert, MODULE, dm), table


@pytest.mark.parametrize("output_format", [INJECTED, "parquet", "CSV"])
@pytest.mark.parametrize("tool", list(TOOLS))
def test_another_format_is_refused_before_anything_is_written(
    tmp_path, monkeypatch, tool, output_format
):
    dm, tools, _ = _session(tmp_path, monkeypatch)

    reply = tools[tool].invoke(
        {"modality_name": "de", "output_format": output_format, **TOOLS[tool]}
    )

    assert reply == (
        f"Unknown output_format {output_format!r}: use 'csv' or 'xlsx'."
    ), reply
    assert dm.calls == []
    assert not dm.exports_dir.exists() or not any(dm.exports_dir.iterdir())


@pytest.mark.parametrize(
    "output_format, writer, read",
    [("csv", "to_csv", pd.read_csv), ("xlsx", "to_excel", pd.read_excel)],
)
@pytest.mark.parametrize("tool", list(TOOLS))
def test_the_cell_writes_with_the_method_of_its_format(
    tmp_path, monkeypatch, tool, output_format, writer, read
):
    dm, tools, table = _session(tmp_path, monkeypatch)
    reply = tools[tool].invoke(
        {"modality_name": "de", "output_format": output_format, **TOOLS[tool]}
    )
    ir = dm.last_ir(tool)
    written = tmp_path / f"replayed.{output_format}"
    code = ir.render(output_path=str(written))
    assert f"results_df.{writer}(" in code, reply

    exec("\n".join([*ir.imports, code]), {"adata": dm.modalities["de"]})

    pd.testing.assert_frame_equal(
        read(written, index_col=0), table.sort_values("padj"), check_names=False
    )


@pytest.mark.parametrize("tool", list(TOOLS))
def test_the_cell_never_renders_the_format_text(tmp_path, monkeypatch, tool):
    dm, tools, _ = _session(tmp_path, monkeypatch)
    tools[tool].invoke({"modality_name": "de", **TOOLS[tool]})

    code = dm.last_ir(tool).render(output_format=INJECTED)

    assert "getcwd" not in code
    assert "results_df.to_csv(" in code


def test_a_filtered_export_cell_writes_the_filtered_rows(tmp_path, monkeypatch):
    """The session exported the filtered table, its notebook cell the unfiltered one."""
    dm, tools, table = _session(tmp_path, monkeypatch)
    reply = tools["filter_de_results"].invoke(
        {"modality_name": "de", "padj_threshold": 0.05, "lfc_threshold": 1.0}
    )
    adata = dm.modalities["de"]
    filtered = adata.uns[f"filtered_{KEY}"]["results_df"]
    assert list(filtered.index) == ["GENE1"], reply

    tools["export_de_results"].invoke({"modality_name": "de", "filtered_only": True})
    ir = dm.last_ir("export_de_results")
    written = tmp_path / "replayed.csv"
    exec(
        "\n".join([*ir.imports, ir.render(output_path=str(written))]),
        {"adata": adata},
    )

    assert list(pd.read_csv(written, index_col=0).index) == ["GENE1"]
    assert ir.parameters["de_results_key"] == f"filtered_{KEY}"
