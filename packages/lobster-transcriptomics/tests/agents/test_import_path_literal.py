"""The bulk import and metadata merge cells write their paths as Python literals.

Both templates put the path inside double quotes in an f-string: a ``"`` in the
path broke the cell, and a ``{{`` was read as a template expression.
"""

import ast

import numpy as np
import pandas as pd

from lobster.agents.transcriptomics.transcriptomics_expert import (
    transcriptomics_expert,
)
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

MODULE = "lobster.agents.transcriptomics.transcriptomics_expert"


def _literals(code):
    return [
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_a_quote_or_braces_in_the_path_stay_in_the_literal(tmp_path, monkeypatch):
    folder = tmp_path / 'a"b {{ c }}'
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
    assert str(folder / "counts.csv") in _literals(irs["import_bulk_counts"].render())
    merge = _literals(irs["merge_sample_metadata"].render())
    assert str(folder / "meta.csv") in merge
    assert "sample" in merge
