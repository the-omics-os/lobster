"""E21–E23: dashboard staleness, unsaved drafts and the review screen."""

from __future__ import annotations

import re

from tests.integration.decision_components._harness import (
    LEAVE_OUT,
    answer_rows,
    ask,
    dashboard_context,
    new_id,
    press_in_plate,
    tool_results,
)
from tests.integration.decision_components.test_e01_e10_identity_groups_qc import (
    active_section,
)

SOURCE = "Which count source should the analysis use?"
GROUPS = "Which group is each sample in?"
DIRECTION = "Compare which group with which?"
QC = "Keep or exclude each flagged sample?"


def _start(start_tui, ds):
    cid = new_id()
    tui = start_tui([ask((cid, "Set up the comparison", dashboard_context(ds)))])
    tui.ask_supervisor()
    tui.wait_render(1)
    return cid, tui


def _answer_all(tui, ds, qc_rows=(("retain", None), ("retain", None))):
    tui.key("1")  # SRC-A
    press_in_plate(tui, ds.ids, {"S08": "treated"})
    tui.key("1")  # treated compared with control
    answer_rows(tui, list(qc_rows))
    tui.wait_screen("Every section is answered")


def _review_and_send(tui):
    tui.key("ctrl+s")
    tui.wait_screen("This is exactly what will be sent")
    tui.key("enter")


def test_e21_changing_the_source_marks_groups_and_qc_stale_transitively(
    start_tui, review8
):
    """E21: source changed after groups + QC → both ⚠ with the reason; submit
    blocked until each is checked again; QC is stale through groups."""
    ds = review8
    cid, tui = _start(start_tui, ds)
    _answer_all(tui, ds)

    tui.key("shift+tab", "shift+tab", "shift+tab")  # qc → direction → groups → source
    tui.wait_screen(f"▸ ✓ 1. {SOURCE}")
    tui.key("2")  # SRC-B
    why = (
        f'"{SOURCE}" changed: Author-supplied matrix (SRC-A) → '
        "Repository-generated matrix (SRC-B)"
    )

    # Wait for the whole redrawn frame, not the first matching line: the
    # screen arrives in pieces, so a partial frame would fail the checks.
    def all_three_stale(text: str) -> bool:
        return (
            text.count(f"Re-check: {why}") == 3  # groups, direction and QC
            and all(
                f"⚠ {n}. {t}" in text for n, t in ((2, GROUPS), (3, DIRECTION), (4, QC))
            )
            and "3 to re-check" in text
        )

    text = tui.wait_screen(all_three_stale)
    assert "1 to re-check" not in text

    tui.key("ctrl+s")
    tui.wait_screen(
        lambda t: all(
            f'"{title}" needs re-checking ({why})' in t
            for title in (GROUPS, DIRECTION, QC)
        )
    )
    assert tui.responses() == []

    # Re-check each stale section (unchanged), then submit.
    assert active_section(tui.flat()) == "groups"
    tui.key("enter")
    tui.wait_screen(f"✓ 2. {GROUPS}")
    tui.key("1")
    tui.wait_screen(f"✓ 3. {DIRECTION}")
    tui.key("enter")
    tui.wait_screen("Every section is answered")
    assert "Re-check:" not in tui.flat()
    _review_and_send(tui)

    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "submit"
    assert result["answer"]["source"] == {"id": "SRC-B"}
    assert result["answer"]["direction"] == {"id": "treated_vs_control"}
    assert result["answer"]["groups"]["S08"] == {"id": "treated"}


def test_e22_edited_section_not_saved_blocks_submit(start_tui, review8):
    """E22: a QC row changed without Enter, then tab away: ctrl+s is blocked
    with "unsaved changes"; Enter in the section keeps the change."""
    ds = review8
    cid, tui = _start(start_tui, ds)
    _answer_all(tui, ds)
    assert active_section(tui.flat()) == "qc"

    tui.key("2")  # S06 (the cursor's row) → defer, not saved
    tui.key("tab")  # away to the source section
    tui.wait_screen(f"▸ ✓ 1. {SOURCE}")
    tui.key("ctrl+s")
    tui.wait_screen(
        f'Can\'t submit yet: unsaved changes in "{QC}": press enter in that section '
        "to keep them"
    )
    assert tui.responses() == []

    tui.key("shift+tab")  # back to QC: the draft is still there
    tui.wait_screen(f"▸ ✓ 4. {QC}")
    tui.key("enter")
    tui.wait_screen("Every section is answered")
    _review_and_send(tui)
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["answer"]["qc"] == {"S02": {"id": "retain"}, "S06": {"id": "defer"}}


def test_e23_review_screen_shows_exactly_what_is_sent(start_tui, review8):
    """E23: every answer on the review screen; back; confirm; the reply and the
    supervisor's answer are what the screen showed."""
    ds = review8
    cid, tui = _start(start_tui, ds)
    reason = "low library size, e2e"
    _answer_all(tui, ds, qc_rows=[("exclude", reason), ("retain", None)])

    tui.key("ctrl+s")
    shown = tui.wait_screen("This is exactly what will be sent")
    expected_lines = [
        "Review before sending Set up the comparison",  # title, then the question
        f"1. {SOURCE} Author-supplied matrix (SRC-A)",
        f"2. {GROUPS} control (4): S01, S02, S03, S04 treated (4): S05, S06, S07, S08",
        "changed from the data: S08 (blank) → treated",
        f"3. {DIRECTION} treated compared with control (treated_vs_control)",
        f"4. {QC} Donor B (S02) → exclude · reason: {reason} Donor F (S06) → retain",
        "Check your answers, then press enter to send them.",  # the guide
        "enter send",  # the key legend
    ]
    for line in expected_lines:
        assert line in shown, line

    tui.key("b")
    tui.wait_screen("4/4 answered")
    assert tui.responses() == []
    tui.key("ctrl+s")
    shown_again = tui.wait_screen("This is exactly what will be sent")
    tui.key("enter")
    reply = tui.wait_response(tui.renders()[0].msg_id)["payload"]["data"]
    assert tui.wait_turn_end() == ""

    # The Go reply is what was shown ...
    answers = reply["answers"]
    plate = answers["groups"]["assignments"]
    for choice, ids in re.findall(
        r"(control|treated|leave_out) \(\d+\): ((?:S\d\d, )*S\d\d)", shown_again
    ):
        assert all(plate[sid] == choice for sid in ids.split(", "))
    assert answers["source"]["selected"] == [{"id": "SRC-A"}]
    assert answers["qc"]["answers"]["S02"] == {
        "id": "exclude",
        "fields": {"reason": reason},
    }
    # ... and so is what the supervisor received.
    result = tool_results(tui.client)[cid]
    assert result["answer"] == {
        "source": {"id": "SRC-A"},
        "groups": {sid: {"id": plate[sid]} for sid in ds.ids},
        "direction": {"id": "treated_vs_control"},
        "qc": {
            "S02": {"id": "exclude", "fields": {"reason": reason}},
            "S06": {"id": "retain"},
        },
    }


def test_e07_reconfirming_an_unchanged_plate_on_a_reask_marks_nothing_stale(
    start_tui, review8
):
    """ROADMAP §4.3: a re-asked section with problems uses its previous answer
    as the staleness baseline, so recording the same value changes nothing.
    Checked here for the plate section of the E7 re-ask."""
    ds = review8
    cid, tui = _start(start_tui, ds)
    tui.key("1")
    press_in_plate(
        tui,
        ds.ids,
        {"S01": LEAVE_OUT, "S03": LEAVE_OUT, "S04": LEAVE_OUT, "S08": "treated"},
    )
    tui.key("1")
    answer_rows(tui, [("exclude", "e2e"), ("retain", None)])
    tui.wait_screen("Every section is answered")
    _review_and_send(tui)
    tui.wait_render(2)
    tui.wait_screen(f"▸ ○ 2. {GROUPS}")

    tui.key("enter")  # the plate, unchanged
    text = tui.wait_screen(f"✓ 2. {GROUPS}")
    assert "Re-check:" not in text, text
    assert active_section(text) == "qc"
    assert f"✓ 3. {DIRECTION}" in text
