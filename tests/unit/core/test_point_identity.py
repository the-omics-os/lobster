"""Source IDs survive point grouping, sampling and trusted revision saves."""

import json

import pandas as pd
import pytest

from lobster.core.figures.point_identity import resolve_identity_column
from lobster.figure_templates import get_template
from lobster.tools.plot_tools import _create_figure

pytestmark = pytest.mark.no_auto_config


def test_ids_follow_grouped_points_even_when_coordinates_are_identical():
    table = pd.DataFrame(
        {
            "point_id": ["P1", "P2", "P3", "P4"],
            "group": ["B", "A", "B", "A"],
            "x": [1, 1, 1, 1],
            "y": [2, 2, 2, 2],
        }
    )
    figure = _create_figure(
        "scatter",
        table,
        "x",
        "y",
        "",
        "group",
        "",
        "Points",
        id_col=resolve_identity_column(table),
    )
    assert {trace.name: list(trace.ids) for trace in figure.data} == {
        "B": ["P1", "P3"],
        "A": ["P2", "P4"],
    }
    assert "custom_data" not in table.columns
    assert list(table.point_id) == ["P1", "P2", "P3", "P4"]


@pytest.mark.parametrize("ids", [["P1", "P1"], ["P1", None], ["P1", " "], [1, "1"]])
def test_invalid_explicit_ids_are_rejected(ids):
    table = pd.DataFrame({"point_id": ids})
    assert resolve_identity_column(table) is None
    with pytest.raises(ValueError, match="Entity IDs"):
        resolve_identity_column(table, "point_id")


def test_missing_or_ambiguous_ids_are_never_guessed():
    assert resolve_identity_column(pd.DataFrame({"x": [1, 2], "y": [3, 4]})) is None
    table = pd.DataFrame({"point_id": ["P1", "P2"], "sample_id": ["S1", "S2"]})
    with pytest.raises(ValueError, match="Specify id_col"):
        resolve_identity_column(table)
    assert resolve_identity_column(table, "sample_id") == "sample_id"
    with pytest.raises(ValueError, match="not found"):
        resolve_identity_column(table, "missing")


def test_generic_template_also_selects_and_declares_ids_automatically():
    from lobster.core.figures.data_view import column_mapping
    from lobster.core.figures.serialize import figure_to_bytes

    data = pd.DataFrame({"point_id": ["A", "B"], "x": [1, 2], "y": [3, 4]})
    figure = get_template("generic.scatter")(data, x="x", y="y")
    assert list(figure.data[0].ids) == ["A", "B"]
    parsed = json.loads(figure_to_bytes(figure))
    roles = column_mapping(parsed, list(data.columns))["roles"]
    assert {"column": "point_id", "role": "id"} in roles


def test_numeric_id_column_is_stored_as_strings_without_changing_source():
    data = pd.DataFrame(
        {"point_id": [9007199254740993, 9007199254740995], "x": [1, 2], "y": [3, 4]}
    )
    figure = get_template("generic.scatter")(data, x="x", y="y")
    assert list(figure.data[0].ids) == ["9007199254740993", "9007199254740995"]
    assert data.point_id.iloc[0] == 9007199254740993


def test_tool_preserves_ids_through_sampling_and_a_saved_visual_edit(
    tmp_path, monkeypatch
):
    import lobster.tools.plot_tools as tools
    from lobster.core.plot_editor.service import EditorService
    from lobster.core.runtime.data_manager import DataManagerV2

    monkeypatch.setattr(tools, "SAMPLE_THRESHOLD", 3)
    data = pd.DataFrame(
        {
            "point_id": ["P" + str(i) for i in range(10)],
            "x": range(10),
            "y": range(10),
            "group": ["A", "B"] * 5,
        }
    )
    dm = DataManagerV2(workspace_path=tmp_path)
    path = tmp_path / "data" / "points.csv"
    data.to_csv(path, index=False)
    plot, inspect = tools.create_plot_tools(dm, tmp_path)
    summary = json.loads(inspect.invoke({"data_file": "data/points.csv"}))
    assert summary["suggested_id_column"] == "point_id"
    reply = plot.invoke(
        {
            "type": "tool_call",
            "name": plot.name,
            "id": "identity-test",
            "args": {
                "plot_type": "scatter",
                "data_file": "data/points.csv",
                "x": "x",
                "y": "y",
                "color": "group",
            },
        }
    )
    assert "saved:" in reply.content, reply.content
    row = dm.figure_store.list().items[0]
    ref = row["figure_id"] + "@" + row["revision_id"]
    service = EditorService(dm.figure_store, ref)
    figure = service.load(ref)["figure"]
    expected = set(data.sample(n=3, random_state=42).point_id)
    assert {value for trace in figure["data"] for value in trace["ids"]} == expected
    result = service.edit_visuals(
        ref,
        generation=service.history()["generation"],
        request_id="identityedit" * 3,
        layout_updates={"title": {"text": "Edited"}},
    )
    assert service.load(result["ref"])["figure"]["data"] == figure["data"]


@pytest.mark.parametrize("path", ["tool", "template"])
def test_identity_column_used_as_numeric_axis_keeps_original_values(path):
    data = pd.DataFrame(
        {"point_id": [10, 20], "y": [3, 4], "_lobster_point_id": ["existing", "column"]}
    )
    if path == "tool":
        figure = _create_figure(
            "scatter", data, "point_id", "y", "", "", "", "Points", id_col="point_id"
        )
    else:
        figure = get_template("generic.scatter")(data, x="point_id", y="y")
    assert list(figure.data[0].ids) == ["10", "20"]
    assert list(figure.data[0].x) == [10, 20]
    assert list(data["_lobster_point_id"]) == ["existing", "column"]


def test_explicit_bad_ids_fail_before_sampling_or_saving(tmp_path, monkeypatch):
    import lobster.tools.plot_tools as tools
    from lobster.core.runtime.data_manager import DataManagerV2

    monkeypatch.setattr(tools, "SAMPLE_THRESHOLD", 1)
    dm = DataManagerV2(workspace_path=tmp_path)
    plot, _ = tools.create_plot_tools(dm, tmp_path)
    reply = plot.invoke(
        {
            "type": "tool_call",
            "name": plot.name,
            "id": "invalid-identity",
            "args": {
                "plot_type": "scatter",
                "data_json": json.dumps(
                    [{"id": "same", "x": 1, "y": 2}, {"id": "same", "x": 2, "y": 4}]
                ),
                "x": "x",
                "y": "y",
                "id_col": "id",
            },
        }
    )
    assert "ERROR:" in reply.content and "unique" in reply.content
    assert dm.figure_store.list().items == []
