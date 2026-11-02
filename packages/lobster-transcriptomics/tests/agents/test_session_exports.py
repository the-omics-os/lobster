"""Exports remain visible when the process cwd is not the session workspace."""

import ast
import json
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.annotation_expert import annotation_expert
from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

DE_MODULE = "lobster.agents.transcriptomics.de_analysis_expert"
ANNOTATION_MODULE = "lobster.agents.transcriptomics.annotation_expert"
RESULTS_KEY = "de_results_condition_sample_b_vs_sample_a"


@pytest.fixture
def session(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    adata = ad.AnnData(
        X=np.ones((6, 2), dtype=int),
        obs=pd.DataFrame(
            {
                "condition": ["sample_a"] * 3 + ["sample_b"] * 3,
                "cell_type_manual": ["type_a"] * 3 + ["type_b"] * 3,
            },
            index=[f"sample_{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    adata.uns["is_pseudobulk"] = True
    results = pd.DataFrame(
        {"log2FoldChange": [1.0, -1.0], "padj": [0.01, 0.02]},
        index=pd.Index(["gene_a", "gene_b"], name="gene"),
    )
    adata.uns[RESULTS_KEY] = {"results_df": results, "parameters": {}}
    dm.store_modality("de_results", adata)
    return dm, cwd, results


def assert_export(dm, cwd, filename, response, marker):
    path = dm.exports_dir / filename
    assert path.is_file(), response
    assert path.read_bytes()
    assert list(cwd.iterdir()) == []
    reported = Path(response.split(marker, 1)[1].splitlines()[0].split(" & ")[0])
    if not reported.is_absolute():
        reported = dm.workspace_path / reported
    assert reported.resolve() == path.resolve()
    listing = dm.list_workspace_files()["exports"]
    assert any(Path(item["path"]).resolve() == path.resolve() for item in listing)
    return path


@pytest.mark.parametrize("formula", [False, True], ids=["ordinary", "formula"])
def test_de_save_uses_session_exports(session, monkeypatch, formula):
    dm, cwd, results = session
    science = Mock()
    science.validate_experimental_design.return_value = {"valid": True}
    science.create_formula_design.return_value = {}
    science.run_pydeseq2_from_pseudobulk.return_value = (
        results,
        {
            "n_genes_tested": 2,
            "n_significant_genes": 2,
            "n_upregulated": 1,
            "n_downregulated": 1,
            "top_upregulated": ["gene_a"],
            "top_downregulated": ["gene_b"],
        },
    )
    monkeypatch.setattr(f"{DE_MODULE}.BulkRNASeqService", lambda **_: science)
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    if formula:
        response = tools["run_de_with_formula"].invoke(
            {
                "pseudobulk_modality": "de_results",
                "formula": "~condition",
                "contrast": ["condition", "sample_b", "sample_a"],
                "save_results": True,
            }
        )
        filename = "de_results_formula_de_results.csv"
        marker = "- CSV file: "
    else:
        response = tools["run_differential_expression"].invoke(
            {
                "modality_name": "de_results",
                "groupby": "condition",
                "group1": "sample_a",
                "group2": "sample_b",
                "save_result": True,
            }
        )
        filename = "de_results_de_results.csv"
        marker = "**Saved to**: "
    path = assert_export(dm, cwd, filename, response, marker)
    pd.testing.assert_frame_equal(pd.read_csv(path, index_col=0), results)


@pytest.mark.parametrize(
    "tool_name,prefix",
    [
        ("export_de_results", "de_results"),
        ("extract_and_export_de_results", "de_publication"),
    ],
)
@pytest.mark.parametrize("output", ["default", "relative", "absolute"])
def test_de_export_path(session, tool_name, prefix, output):
    dm, cwd, results = session
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    args = {"modality_name": "de_results"}
    filename = f"{prefix}_condition_sample_b_vs_sample_a.csv"
    if output != "default":
        filename = "custom_de_results.csv"
        args["output_path"] = (
            filename if output == "relative" else str(dm.exports_dir / filename)
        )
    if tool_name == "extract_and_export_de_results":
        args["shrink_lfc"] = False
    response = tools[tool_name].invoke(args)
    path = assert_export(dm, cwd, filename, response, "- Output: ")
    pd.testing.assert_frame_equal(pd.read_csv(path, index_col=0), results)


@pytest.mark.parametrize(
    "tool_name", ["export_de_results", "extract_and_export_de_results"]
)
def test_de_export_generated_import_cell_is_import_statements(
    session, monkeypatch, tool_name
):
    """Exercise both real IR producers and only their generated import cell.

    This does not execute a notebook kernel or certify the analysis replay.
    The existing factory fixture replaces the LLM; no provider call is made.
    """
    from lobster.core.notebooks.exporter import NotebookExporter

    dm, _, _ = session
    recorded = []
    original_log = dm.log_tool_usage

    def capture_ir(*args, **kwargs):
        recorded.append(kwargs.get("ir"))
        return original_log(*args, **kwargs)

    monkeypatch.setattr(dm, "log_tool_usage", capture_ir)
    tools = factory_tools(de_analysis_expert, DE_MODULE, dm)
    args = {"modality_name": "de_results"}
    if tool_name == "extract_and_export_de_results":
        args["shrink_lfc"] = False
    response = tools[tool_name].invoke(args)
    produced = [ir for ir in recorded if ir is not None and ir.tool_name == tool_name]
    assert len(produced) == 1, response
    cell = NotebookExporter(dm.provenance, dm)._create_imports_cell(produced)
    statements = ast.parse(cell.source).body
    assert statements and all(
        isinstance(statement, (ast.Import, ast.ImportFrom)) for statement in statements
    ), cell.source
    namespace = {}
    exec(compile(cell.source, "<generated-de-import-cell>", "exec"), namespace)
    assert namespace["pd"] is pd


@pytest.mark.parametrize("format", ["json", "csv"])
@pytest.mark.parametrize("output", ["default", "relative", "absolute"])
def test_annotation_export_path(session, format, output):
    dm, cwd, _ = session
    tools = factory_tools(annotation_expert, ANNOTATION_MODULE, dm)
    args = {"modality_name": "de_results", "format": format}
    filename = "annotation_mapping.json"
    if output != "default":
        filename = f"custom_mapping.{format}"
        args["output_filename"] = (
            filename if output == "relative" else str(dm.exports_dir / filename)
        )
    response = tools["export_annotation_mapping"].invoke(args)
    path = assert_export(dm, cwd, filename, response, "- Output file: ")
    if format == "json":
        assert json.loads(path.read_text())["cell_type_counts"] == {
            "type_a": 3,
            "type_b": 3,
        }
    else:
        assert pd.read_csv(path)["cell_count"].tolist() == [3, 3]
