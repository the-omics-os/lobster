"""Generic table plotting uses the production save path without a modality."""

import base64
import json

import numpy as np
import plotly.io as pio
import pytest

from lobster.core.runtime.data_manager import DataManagerV2
from lobster.tools.plot_tools import create_plot_tools
from lobster.tools.filesystem_tools import create_filesystem_tools

pytestmark = pytest.mark.no_auto_config


def test_csv_scatter_saves_managed_figure_without_modality(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path / ".lobster_workspace")
    csv = b"point_id,group,effect,evidence_score\nP01,A,-1.9,2.0\nP02,A,-1.2,4.0\nP03,B,0.5,7.0\n"
    path = dm.workspace_path / "data" / "points.csv"
    path.write_bytes(csv)
    read = next(t for t in create_filesystem_tools(tmp_path) if t.name == "read_file")
    assert "point_id,group,effect,evidence_score" in read.invoke({"path": str(path)})
    tool, inspect = create_plot_tools(dm, dm.workspace_path)
    details = json.loads(inspect.invoke({"data_file": "data/points.csv"}))
    assert details["numeric_columns"] == ["effect", "evidence_score"]
    assert details["columns"] == ["point_id", "group", "effect", "evidence_score"]
    assert details["rows"] == 3
    reply = tool.invoke(
        {
            "type": "tool_call",
            "name": tool.name,
            "id": "csv_scatter",
            "args": {
                "plot_type": "scatter",
                "data_file": "data/points.csv",
                "x": "effect",
                "y": "evidence_score",
                "color": "group",
            },
        }
    )
    assert "saved:" in reply.content, reply.content
    records = dm.figure_store.list().items
    assert len(records) == 1
    record = records[0]
    ref = record["figure_id"] + "@" + record["revision_id"]
    assert ref in reply.content
    _, figure_bytes, _ = dm.figure_store.read(ref)
    figure = pio.from_json(figure_bytes.decode())
    assert figure.layout.xaxis.title.text == "effect"
    assert figure.layout.yaxis.title.text == "evidence_score"

    def values(array):
        if isinstance(array, dict):
            return np.frombuffer(base64.b64decode(array["bdata"]), dtype=array["dtype"])
        return array

    points = {t.name: list(zip(values(t.x), values(t.y))) for t in figure.data}
    assert points == {"A": [(-1.9, 2.0), (-1.2, 4.0)], "B": [(0.5, 7.0)]}
    assert path.read_bytes() == csv
    assert not dm.modalities
