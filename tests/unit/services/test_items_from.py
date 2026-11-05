"""items_from (a choice field's items read from a workspace table) and item
positions on a plate. Synthetic tables only."""

import pytest
from pydantic import ValidationError

from lobster.services.interaction.component_schemas import (
    ITEMS_FROM_MAX_ROWS,
    AskUserContext,
    Item,
    expand_items_from,
)

LAYOUT = """\
sample,well,condition,sample_type,batch,note,warning
TW.M01,D1,vehicle,study,B1,,
TW.M02,D2,TDX 9,study,B1,re-plated,
TW.M03,D3,?,study,B2,,low RNA yield
TW.M04,D4,unknown,study,B2,,
TW.M05,D5,,study,B2,,
QC.P1,C6,pool,qc,B1,,
"""

OPTIONS = ["vehicle", {"id": "tdx9", "label": "TDX-9"}, "leave_out"]


def _context(spec, **field):
    return {
        "fields": [
            {
                "key": "groups",
                "type": "choice",
                "view": "plate_map",
                "options": OPTIONS,
                "items_from": spec,
                **field,
            }
        ]
    }


SPEC = {
    "file": "data/layout.csv",
    "id": "sample",
    "default": "condition",
    "details": ["batch", "note"],
    "warning": "warning",
    "position": "well",
    "where": {"sample_type": "study"},
    "match": "ignore_case_and_spaces",
}


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "layout.csv").write_text(LAYOUT)
    return tmp_path


def _expand(workspace, spec=SPEC, **field):
    return expand_items_from(_context(spec, **field), workspace)


def test_items_and_defaults_come_from_the_table(workspace):
    context, problems = _expand(workspace)
    assert problems == []
    (field,) = context["fields"]
    # Kept with the question, so the table can be read again if it changes;
    # the screens get an ordinary for_each.
    assert field["items_from"]["file"] == "data/layout.csv"
    assert [item["id"] for item in field["for_each"]] == [
        "TW.M01",
        "TW.M02",
        "TW.M03",
        "TW.M04",
        "TW.M05",
    ]  # QC.P1 left out by "where"
    assert field["for_each"][0] == {
        "id": "TW.M01",
        "details": "batch: B1",
        "position": "D1",
    }
    assert field["for_each"][2] == {
        "id": "TW.M03",
        "details": "batch: B2",
        "warning": "low RNA yield",
        "position": "D3",
    }
    # "TDX 9" is the id tdx9 once case and spaces are ignored; "?", "unknown"
    # and blank are not options, so those samples start unresolved.
    assert field["default"] == {"TW.M01": {"id": "vehicle"}, "TW.M02": {"id": "tdx9"}}
    assert context["data_files"] == ["data/layout.csv"]
    AskUserContext.model_validate(context)  # a valid question


def test_ignore_case_and_spaces_matches_a_label_written_differently(workspace):
    (workspace / "data" / "layout.csv").write_text(
        LAYOUT.replace("TDX 9", " tdx-9 ").replace("vehicle,", " Vehicle ,")
    )
    context, problems = _expand(workspace)
    assert problems == []
    assert context["fields"][0]["default"] == {
        "TW.M01": {"id": "vehicle"},
        "TW.M02": {"id": "tdx9"},
    }


def test_exact_match_is_the_default(workspace):
    (workspace / "data" / "layout.csv").write_text(
        LAYOUT.replace("vehicle,", "Vehicle,")
    )
    spec = {k: v for k, v in SPEC.items() if k != "match"}
    context, problems = _expand(workspace, spec)
    assert problems == []
    assert "default" not in context["fields"][0]  # "Vehicle" is not "vehicle"


def test_a_tsv_file_and_a_label_column(workspace):
    (workspace / "data" / "sheet.tsv").write_text("id\tname\nS1\tDonor A\nS2\t\n")
    spec = {"file": "data/sheet.tsv", "id": "id", "label": "name"}
    context, problems = _expand(workspace, spec, view="default")
    assert problems == []
    assert context["fields"][0]["for_each"] == [
        {"id": "S1", "label": "Donor A"},
        {"id": "S2"},
    ]


def test_the_file_is_added_to_data_files_once(workspace):
    context = _context(SPEC)
    context["data_files"] = ["data/layout.csv"]
    expanded, problems = expand_items_from(context, workspace)
    assert problems == [] and expanded["data_files"] == ["data/layout.csv"]


def test_a_context_without_items_from_is_unchanged(workspace):
    context = {"fields": [{"key": "a", "type": "text"}]}
    assert expand_items_from(context, workspace) == (context, [])


@pytest.mark.parametrize(
    "change, path, message",
    [
        ({"file": "data/missing.csv"}, "fields.0.items_from.file", "can't read"),
        (
            {"file": "../outside.csv"},
            "fields.0.items_from.file",
            "inside the workspace",
        ),
        (
            {"file": "data/layout.xlsx"},
            "fields.0.items_from.file",
            "not a .csv or .tsv",
        ),
        ({"id": "sample_id"}, "fields.0.items_from", "no column 'sample_id'"),
        ({"details": ["batch", "dose"]}, "fields.0.items_from", "no column 'dose'"),
        ({"where": {"kind": "study"}}, "fields.0.items_from", "no column 'kind'"),
        ({"where": {"sample_type": "none"}}, "fields.0.items_from", "no rows"),
        ({"id": "batch"}, "fields.0.items_from", "repeats ids B1, B2"),
        ({"id": "note"}, "fields.0.items_from", "have no 'note'"),
        ({"colour": "x"}, "fields.0.items_from.colour", "Extra inputs"),
        ({"match": "fuzzy"}, "fields.0.items_from.match", "exact"),
    ],
)
def test_a_broken_table_is_never_shown(workspace, change, path, message):
    context, problems = _expand(workspace, {**SPEC, **change})
    assert len(problems) == 1, problems
    assert problems[0].path == path
    assert message in problems[0].message
    assert "items_from" in context["fields"][0]  # unchanged


@pytest.mark.parametrize(
    "field, message",
    [
        ({"for_each": ["S1"]}, "not both"),
        ({"default": {"S1": "vehicle"}}, "leave the field's default out"),
        ({"type": "text", "options": None}, "only for a choice"),
    ],
)
def test_items_from_with_a_conflicting_field_is_refused(workspace, field, message):
    _, problems = _expand(workspace, **field)
    assert message in problems[0].message


def test_without_a_workspace_the_file_cannot_be_read():
    _, problems = expand_items_from(_context(SPEC), None)
    assert "no workspace is open" in problems[0].message


def test_too_many_rows_for_a_plate_or_a_list(workspace):
    rows = "\n".join(f"S{n:03d},x" for n in range(97))
    (workspace / "data" / "big.csv").write_text("id,kind\n" + rows + "\n")
    spec = {"file": "data/big.csv", "id": "id"}
    _, problems = _expand(workspace, spec)
    assert problems[0].message == (
        "97 rows of data/big.csv match; a plate holds at most 96 samples; "
        "leave the view out to ask them as a list"
    )
    _, problems = _expand(workspace, spec, view="default")
    assert problems == []

    rows = "\n".join(f"S{n:04d},x" for n in range(ITEMS_FROM_MAX_ROWS + 1))
    (workspace / "data" / "big.csv").write_text("id,kind\n" + rows + "\n")
    _, problems = _expand(workspace, spec, view="default")
    assert f"one question takes at most {ITEMS_FROM_MAX_ROWS}" in problems[0].message


# --- positions ---


@pytest.mark.parametrize("given", ["D1", "d01", " D1 ", {"row": "D", "column": 1}])
def test_a_position_is_a_well(given):
    assert Item(id="S1", position=given).position == "D1"


@pytest.mark.parametrize("given", ["1D", "D", "D0", "", 5, {"row": "D"}])
def test_a_position_that_is_not_a_well_is_refused(given):
    with pytest.raises(ValidationError, match="position|row"):
        Item(id="S1", position=given)


def _plate(*items):
    return {
        "fields": [
            {
                "key": "g",
                "type": "choice",
                "view": "plate_map",
                "for_each": list(items),
                "options": ["a", "b"],
            }
        ]
    }


@pytest.mark.parametrize(
    "items, message",
    [
        (
            [{"id": "S1", "position": "A1"}, "S2"],
            "give every item a position, or none; S2 have none",
        ),
        (
            [{"id": "S1", "position": "I1"}],
            "item 'S1' is at I1, outside the plate (A1 to H12)",
        ),
        (
            [{"id": "S1", "position": "A13"}],
            "item 'S1' is at A13, outside the plate (A1 to H12)",
        ),
        (
            [{"id": "S1", "position": "A1"}, {"id": "S2", "position": "a01"}],
            "items 'S1' and 'S2' are both in well A1",
        ),
    ],
)
def test_plate_positions_are_checked(items, message):
    with pytest.raises(
        ValidationError, match=message.replace("(", r"\(").replace(")", r"\)")
    ):
        AskUserContext.model_validate(_plate(*items))


def test_an_option_has_no_position():
    context = _plate("S1")
    context["fields"][0]["options"] = [{"id": "a", "position": "A1"}, "b"]
    with pytest.raises(ValidationError, match="an option has no position"):
        AskUserContext.model_validate(context)


def test_unavailable_rows_sit_greyed_with_their_reason(workspace):
    spec = {k: v for k, v in SPEC.items() if k != "where"}
    spec["unavailable"] = {
        "column": "sample_type",
        "values": {"QC": "pooled QC, never grouped", "blank": "blank"},
    }
    context, problems = _expand(workspace, spec)
    assert problems == []
    field = context["fields"][0]
    pool = next(item for item in field["for_each"] if item["id"] == "QC.P1")
    assert pool["unavailable"] == "pooled QC, never grouped"  # "qc" ignoring case
    assert pool["position"] == "C6"
    assert "QC.P1" not in field.get("default", {})
    AskUserContext.model_validate(context)


def test_an_unavailable_column_that_is_missing_is_refused(workspace):
    spec = {**SPEC, "unavailable": {"column": "kind", "values": {"qc": "qc"}}}
    _, problems = _expand(workspace, spec)
    assert "no column 'kind'" in problems[0].message
