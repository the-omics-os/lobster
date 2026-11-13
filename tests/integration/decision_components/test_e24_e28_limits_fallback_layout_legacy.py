"""E24, E25, E27, E28: terminal limits, an older TUI, layout, legacy payloads."""

from __future__ import annotations

import json
import re
import time
from typing import Annotated

import pytest
from langchain_core.tools import InjectedToolCallId, tool

from tests.integration.decision_components._harness import (
    QC_CHECKS,
    answer_rows,
    ask,
    call,
    groups_field,
    new_id,
    pending_interrupts,
    qc_field,
    source_field,
    tool_results,
)

_WORD = re.compile(r"[\w][\w'’-]*")


def _words(text):
    return set(_WORD.findall(text))


def _missing_words(source_text, screen_text):
    """Words of ``source_text`` that are not whole words on the screen."""
    return sorted(_words(source_text) - _words(screen_text))


# --- E24 -------------------------------------------------------------------


def test_e24_a_list_of_numbers_per_sample_is_refused_with_no_retry_comment(
    start_tui, review8
):
    """E24 (N2 shape): the terminal can't draw it → alert, cancel + comment."""
    ds = review8
    cid = new_id()
    context = {
        "fields": [
            {
                "key": "library_sizes",
                "type": "number",
                "question": "Library sizes per sample (one per lane)",
                "count": {"min": 1, "max": 4},
                "for_each": ds.ids,
            }
        ]
    }
    tui = start_tui([ask((cid, "Enter the library sizes", context))])
    tui.ask_supervisor()
    tui.wait_screen("This terminal can't show")
    assert tui.wait_turn_end() == ""
    alerts = [m["payload"]["message"] for m in tui.messages("to_go", "alert")]
    assert any("library_sizes: a list of number values (count)" in a for a in alerts)
    assert tui.renders() == []
    result = tool_results(tui.client)[cid]
    assert result["action"] == "cancel"
    assert "Don't ask again with the same shape" in result["comment"]
    assert "library_sizes: a list of number values (count)" in result["comment"]


# --- E25 -------------------------------------------------------------------


def test_e25_older_tui_shows_the_whole_question_and_answers_free_text(
    start_tui, review8, monkeypatch
):
    """E25: a component the binary doesn't know (an unregistered name stands in
    for an older binary) → its text box with the whole question; the typed
    answer reaches the supervisor as free_text, checked: false."""
    from lobster.cli_internal import go_tui_launcher

    real = go_tui_launcher._render_component

    def as_older_binary(bridge, component, data, fallback=None):
        return real(bridge, f"e2e_unregistered_{component}", data, fallback)

    monkeypatch.setattr(go_tui_launcher, "_render_component", as_older_binary)
    ds = review8
    cid = new_id()
    context = {"fields": [qc_field(ds)], "checks": QC_CHECKS}
    tui = start_tui([ask((cid, "Keep or exclude each flagged sample?", context))])
    tui.ask_supervisor()
    render = tui.wait_render(1)
    assert render.component == "e2e_unregistered_choice_list"
    screen = tui.wait_screen("Write your answer in your own words.")
    for line in render.fallback.splitlines():
        assert " ".join(line.split()) in screen, line
    assert "S02" in render.fallback and "then give: Why exclude it?" in render.fallback

    words = "Keep S02 but exclude S06 because the mito fraction is far too high"
    tui.type(words)
    tui.key("enter")
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "free_text"
    assert result["text"] == words
    assert result["checked"] is False
    assert "not checked" in result["note"]
    assert result["attempts"] == 1


# --- E27 -------------------------------------------------------------------


def _start_72_list(start_tui, ds):
    cid = new_id()
    field = groups_field(ds, view=None)
    tui = start_tui([ask((cid, "Which group is each sample in?", {"fields": [field]}))])
    tui.ask_supervisor()
    render = tui.wait_render(1)
    assert render.component == "choice_list" and len(render.data["items"]) == 72
    return cid, tui


def test_e27_72_rows_on_a_40_row_pty_keep_the_cursor_row_visible(start_tui, review72):
    """E27: 72 samples as one list on 40 rows: the cursor row is always on
    screen, the frame fits the terminal, and the whole list is answerable."""
    ds = review72
    cid, tui = _start_72_list(start_tui, ds)
    picks = {"S15": "1", "S30": "2", "S49": "1", "S68": "2"}
    for index, sid in enumerate(ds.ids):
        if index:
            tui.key("down")
        if sid in picks:
            tui.key(picks[sid])
        if index % 9 == 0 or sid in picks or index == 71:
            label = ds.label(sid)
            text = tui.wait_screen(
                lambda t, s=sid, lab=label: f"> ● {lab} ({s})" in t
                or f"> ○ {lab} ({s})" in t
            )
            assert "undecided" in text  # the summary line under the rows
            assert len(tui.screen.lines()) == 40
    tui.key("enter")
    assert tui.wait_turn_end() == ""
    groups = tool_results(tui.client)[cid]["answer"]["groups"]
    assert len(groups) == 72
    for sid, number in picks.items():
        assert groups[sid] == {"id": ("control", "treated")[int(number) - 1]}
    assert groups["S01"] == {"id": "control"} and groups["S72"] == {"id": "treated"}


def test_e27_72_rows_on_a_40_row_pty_keep_the_question_on_screen(start_tui, review72):
    """E27: the question and every option line of a 72-row list stay on a
    40-row screen (only the rows scroll), and the screen is not repainted
    while nothing changes."""
    ds = review72
    _, tui = _start_72_list(start_tui, ds)
    assert tui.output_rate() == 0, "the screen is repainted continuously"
    screen = tui.flat()
    for text in (
        "Which group is each sample in?",
        "[1] control",
        "[2] treated",
        "[3] leave out (leave_out)",
    ):
        assert text in screen, tui.screen.text()


_LONG_QUESTION = (
    "Which count source should the differential expression comparison between "
    "the treated and the control samples of this fictional study use?"
)


def _long_sources(ds):
    field = source_field(ds)
    field["question"] = _LONG_QUESTION
    for option in field["options"]:
        option["details"] += (
            "; produced for this synthetic example only, with every value made up "
            "and checked against the example's own analysis policy"
        )
    return field


def _visible_text_of(field):
    parts = [field["question"]]
    for option in field["options"]:
        parts += [
            option["id"],
            option["label"],
            option["details"],
            option.get("unavailable", ""),
        ]
    return " ".join(parts)


def test_e27_narrow_60_column_pty_cuts_no_words(start_tui, review8):
    """E27: on 60 columns every word of a long question and its options shows."""
    ds = review8
    field = _long_sources(ds)
    cid = new_id()
    tui = start_tui([ask((cid, "Which count source?", {"fields": [field]}))], cols=60)
    tui.ask_supervisor()
    tui.wait_render(1)
    screen = tui.wait_screen("1/3")
    assert _missing_words(_visible_text_of(field), screen) == []
    assert max(len(line) for line in tui.screen.lines()) <= 60
    tui.key("1")
    assert tui.wait_turn_end() == ""
    assert tool_results(tui.client)[cid]["answer"] == {"source": {"id": "SRC-A"}}


@pytest.mark.parametrize("cols", [120, 60])
def test_e27_plate_sample_details_are_not_cut_off(start_tui, review8, cols):
    """E27: the plate shows the sample under the cursor with its details (the
    warning and the suggestion note); none of it may be cut off."""
    ds = review8
    cid = new_id()
    tui = start_tui(
        [ask((cid, "Which group is each sample in?", {"fields": [groups_field(ds)]}))],
        cols=cols,
    )
    tui.ask_supervisor()
    render = tui.wait_render(1)
    note = next(w for w in render.data["wells"] if w["sample_id"] == "S08")["details"][
        "note"
    ]
    tui.key(*["right"] * 7)
    screen = tui.wait_screen("A8  S08")  # the current-well line
    assert _missing_words(note, screen) == [], tui.screen.text()


def test_e27_resize_mid_question_reflows_and_stays_answerable(start_tui, review8):
    """E27: 120x40 → 60x30 while a list is shown: the binary reflows (SIGWINCH
    delivered as a terminal does to its foreground process), nothing is cut
    off, and the question is still answered normally."""
    ds = review8
    cid = new_id()
    field = qc_field(ds)
    field["question"] = (
        "Keep or exclude each flagged sample before the treated versus control "
        "comparison of this fictional study?"
    )
    context = {"fields": [field], "checks": QC_CHECKS}
    tui = start_tui([ask((cid, "Keep or exclude?", context))])
    tui.ask_supervisor()
    tui.wait_render(1)
    tui.key("1")  # S02 → retain before the resize
    tui.resize(30, 60)
    tui._wait(
        lambda: {"width": 60, "height": 30}
        in [m["payload"] for m in tui.messages("from_go", "resize")],
        10,
        "the binary's resize message",
    )
    time.sleep(0.5)
    tui.key("down")  # any key: the screen is redrawn at the new size
    screen = tui.wait_screen("retain (1) · defer (0) · exclude (0) · undecided (1)")
    visible = " ".join([field["question"]] + [i["details"] for i in field["for_each"]])
    assert _missing_words(visible, screen) == []
    assert max(len(line) for line in tui.screen.lines()) <= 60
    answer_rows(tui, [("exclude", "high mito after resize")])
    assert tui.wait_turn_end() == ""
    assert tool_results(tui.client)[cid]["answer"]["qc"] == {
        "S02": {"id": "retain"},
        "S06": {"id": "exclude", "fields": {"reason": "high mito after resize"}},
    }


@pytest.mark.xfail(strict=True, reason="E2E_REPORT #3")
def test_e27_resize_reaches_the_binary_as_the_launcher_starts_it(start_tui, review8):
    """E27: started as launch_go_tui_chat starts it (os.setsid, no controlling
    terminal), the binary must still learn about a terminal resize."""
    ds = review8
    tui = start_tui(
        [ask((new_id(), "Which count source?", {"fields": [source_field(ds)]}))]
    )
    tui.ask_supervisor()
    tui.wait_render(1)
    tui._winsize(tui.master, 30, 60)  # what a terminal window resize does
    time.sleep(2.0)
    sizes = [m["payload"] for m in tui.messages("from_go", "resize")]
    assert {"width": 60, "height": 30} in sizes, sizes


# --- E28 -------------------------------------------------------------------


@tool
def e2e_confirm_deletion(tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Test-only tool: pause with the figure-deletion interrupt payload shape
    (lobster/core/figures/deletion.py render_warning) and report the resume."""
    from langgraph.types import interrupt

    value = interrupt(
        {
            "type": "confirm_deletion",
            "question": "Delete dataset 'syn_rnaseq' (instance 1).",
            "fingerprint": "e2e-fingerprint",
        }
    )
    return json.dumps({"resumed_with": value})


def test_e28_legacy_deletion_payload_is_handled_as_upstream(start_tui, review8):
    """E28: a non-ask_user payload goes through upstream _handle_interrupt:
    rendered as sent, a single pending one resumes with the bare reply."""
    cid = new_id("del")
    tui = start_tui(
        [call("e2e_confirm_deletion", cid)], extra_tools=[e2e_confirm_deletion]
    )
    tui.ask_supervisor()
    (render,) = [tui.wait_render(1)]
    # Upstream sends the payload itself as the component_render payload.
    raw = tui.messages("to_go", "component_render")[0]["payload"]
    assert raw == {
        "type": "confirm_deletion",
        "question": "Delete dataset 'syn_rnaseq' (instance 1).",
        "fingerprint": "e2e-fingerprint",
    }
    tui.wait_screen("Please provide input")  # Go's text box for a nameless component
    tui.type("yes")
    tui.key("enter")
    assert tui.wait_turn_end() == ""
    (resumed,) = tui.resumes
    assert resumed == {"answer": "yes", "action": "submit"}  # bare, not keyed
    assert tool_results(tui.client)[cid] == {
        "resumed_with": {"answer": "yes", "action": "submit"}
    }
    assert render.msg_id


def test_e28_legacy_payload_cancelled_ends_the_turn_without_resuming(
    start_tui, review8
):
    """E28: as upstream, cancelling a legacy prompt does not resume the graph."""
    cid = new_id("del")
    tui = start_tui(
        [call("e2e_confirm_deletion", cid)], extra_tools=[e2e_confirm_deletion]
    )
    tui.ask_supervisor()
    tui.wait_render(1)
    tui.wait_screen("Please provide input")
    tui.key("esc")
    assert tui.wait_turn_end() == "cancelled"
    assert tui.resumes == []
    assert len(pending_interrupts(tui.client)) == 1
