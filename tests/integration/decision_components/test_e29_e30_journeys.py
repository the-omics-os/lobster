"""E29–E30: full journeys on both synthetic datasets (review8 and review72).

E29 asks one question at a time (three ask_user calls: source, then groups
with the comparison direction, then QC). E30 asks everything on one
dashboard and submits through its review screen. Both check the final
answers in the graph state and what the scripted supervisor was shown.
"""

from __future__ import annotations

from tests.integration.decision_components._harness import (
    QC_CHECKS,
    answer_rows,
    ask,
    dashboard_context,
    direction_field,
    groups_field,
    groups_kept,
    new_id,
    no_default_for_unknown,
    press_in_plate,
    qc_field,
    source_field,
    tool_results,
)

# The person resolves the unknown conditions to the hidden truth of each case
# (review8: S08 treated; review72: odd numbers control, even treated).
RESOLVE = {
    "review8": {"S08": "treated"},
    "review72": {
        "S15": "control",
        "S30": "treated",
        "S49": "control",
        "S68": "treated",
    },
}


def _qc_rows(ds):
    """Exclude the first warned sample (with a reason), retain the rest."""
    first, *rest = ds.warned
    reason = f"{ds.samples[ds.ids.index(first)]['supplied_warning']} (e2e)"
    rows = [("exclude", reason)] + [("retain", None)] * len(rest)
    expected = {first: {"id": "exclude", "fields": {"reason": reason}}}
    expected.update({sid: {"id": "retain"} for sid in rest})
    return rows, expected


def _expected_groups(ds):
    groups = {sid: {"id": ds.condition(sid)} for sid in ds.ids}
    groups.update({sid: {"id": c} for sid, c in RESOLVE[ds.case].items()})
    return groups


def test_e29_single_question_journey(start_tui, any_dataset):
    """E29: source → groups (+ direction) → QC, one screen at a time."""
    ds = any_dataset
    ids = {"source": new_id("src"), "groups": new_id("grp"), "qc": new_id("qc")}
    groups_context = {
        "fields": [groups_field(ds), direction_field()],
        "checks": [groups_kept(), no_default_for_unknown(ds)],
        "data_files": [ds.rel("sample_metadata.csv")],
    }
    tui = start_tui(
        [
            ask((ids["source"], "Which count source?", {"fields": [source_field(ds)]})),
            ask((ids["groups"], "Set up the comparison groups", groups_context)),
            ask(
                (
                    ids["qc"],
                    "Keep or exclude each flagged sample?",
                    {"fields": [qc_field(ds)], "checks": QC_CHECKS},
                )
            ),
        ]
    )
    tui.ask_supervisor()
    assert tui.wait_render(1).component == "choice"
    tui.key("1")
    plate = tui.wait_render(2)
    assert plate.component == "plate_map" and len(plate.data["wells"]) == len(ds.ids)
    press_in_plate(tui, ds.ids, RESOLVE[ds.case])
    assert tui.wait_render(3).component == "choice"  # the direction
    tui.key("1")
    qc = tui.wait_render(4)
    assert qc.component == "choice_list"
    rows, expected_qc = _qc_rows(ds)
    answer_rows(tui, rows)
    assert tui.wait_turn_end() == ""

    results = tool_results(tui.client)
    assert results[ids["source"]]["answer"] == {"source": {"id": "SRC-A"}}
    assert results[ids["groups"]]["answer"] == {
        "groups": _expected_groups(ds),
        "direction": {"id": "treated_vs_control"},
    }
    assert results[ids["qc"]]["answer"] == {"qc": expected_qc}
    assert all(results[i]["attempts"] == 1 for i in ids.values())
    # What the scripted supervisor was shown is exactly the graph's results.
    seen = {e["tool_call_id"]: e["content"] for e in tui.client.e2e_model.seen}
    assert seen == results
    assert all(e["raw"].startswith("<tool_data>") for e in tui.client.e2e_model.seen)


def test_e30_dashboard_journey_with_review_screen(start_tui, any_dataset):
    """E30: one dashboard, every section, the review screen, confirm."""
    ds = any_dataset
    cid = new_id()
    tui = start_tui([ask((cid, "Set up the comparison", dashboard_context(ds)))])
    tui.ask_supervisor()
    render = tui.wait_render(1)
    assert [s["key"] for s in render.data["sections"]] == [
        "source",
        "groups",
        "direction",
        "qc",
    ]

    tui.key("1")
    press_in_plate(tui, ds.ids, RESOLVE[ds.case])
    tui.key("1")
    rows, expected_qc = _qc_rows(ds)
    answer_rows(tui, rows)
    tui.wait_screen("Every section is answered")
    tui.key("ctrl+s")
    review = tui.wait_screen("This is exactly what will be sent")
    # "lines a–b of N" is shown only when the review needs scrolling.
    total = int(review.rsplit(" of ", 1)[1].split()[0]) if " of " in review else 0
    seen_lines = review
    for _ in range(total // 10 + 1):  # scroll through the whole review
        tui.key("pgdown")
        seen_lines += " " + tui.flat()
    first_excluded = ds.warned[0]
    assert f"({first_excluded}) → exclude · reason:" in seen_lines
    assert "treated compared with control (treated_vs_control)" in seen_lines
    for sid in RESOLVE[ds.case]:
        assert sid in seen_lines
    tui.key("enter")
    assert tui.wait_turn_end() == ""

    result = tool_results(tui.client)[cid]
    assert result["action"] == "submit" and result["attempts"] == 1
    assert result["answer"] == {
        "source": {"id": "SRC-A"},
        "groups": _expected_groups(ds),
        "direction": {"id": "treated_vs_control"},
        "qc": expected_qc,
    }
    seen = {e["tool_call_id"]: e["content"] for e in tui.client.e2e_model.seen}
    assert seen == {cid: result}
