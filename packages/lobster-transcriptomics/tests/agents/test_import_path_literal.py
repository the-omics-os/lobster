"""The bulk import and metadata merge cells write no path of the session.

Both templates once put the path inside double quotes in an f-string: a ``"`` in the
path broke the cell, and a ``{{`` was read as a template expression, which
Lobster evaluated in its own process when it rendered the cell.
"""

import ast

import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.transcriptomics_expert import (
    transcriptomics_expert,
)
from lobster.core.notebooks.written_templates import outdated_tool, template_hash
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

MODULE = "lobster.agents.transcriptomics.transcriptomics_expert"


def _literals(code):
    return [
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


@pytest.mark.parametrize(
    "name", ['a"b {{ c }}', "x{{ cycler.__init__.__globals__.os.getpid() }}y"]
)
def test_a_quote_or_braces_in_the_path_stay_in_the_literal(tmp_path, monkeypatch, name):
    folder = tmp_path / name
    folder.mkdir()
    samples = [f"S{i}" for i in range(1, 7)]
    counts = pd.DataFrame(
        np.arange(6000).reshape(1000, 6),
        index=[f"GENE{i:04d}" for i in range(1000)],
        columns=samples,
    )
    counts.to_csv(folder / "counts.csv")
    pd.DataFrame({"sample": samples, "group": ["a", "b"] * 3}).to_csv(
        folder / "meta.csv", index=False
    )
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    irs = {}
    log = dm.log_tool_usage

    def recording(tool_name, *args, **kwargs):
        irs[tool_name] = kwargs.get("ir")
        return log(tool_name, *args, **kwargs)

    monkeypatch.setattr(dm, "log_tool_usage", recording)
    tools = factory_tools(transcriptomics_expert, MODULE, dm)
    reply = tools["import_bulk_counts"].invoke(
        {"modality_name": "bulk", "file_path": str(folder / "counts.csv")}
    )
    assert "import_bulk_counts" in irs, reply
    reply = tools["merge_sample_metadata"].invoke(
        {
            "modality_name": "bulk",
            "metadata_file": str(folder / "meta.csv"),
            "sample_id_column": "sample",
        }
    )

    assert "merge_sample_metadata" in irs, reply
    # neither cell names a path of this session: the import reads the notebook's
    # input file, the merge keeps the sheet itself
    for tool in ("import_bulk_counts", "merge_sample_metadata"):
        assert name not in irs[tool].render()
    merge = _literals(irs["merge_sample_metadata"].render())
    assert "meta.csv" in merge
    assert "sample" in merge
    # the templates are the ones the exporter knows, whatever the path
    for tool in ("import_bulk_counts", "merge_sample_metadata"):
        assert outdated_tool(irs[tool]) is None, (
            irs[tool].operation,
            template_hash(irs[tool].code_template),
        )
