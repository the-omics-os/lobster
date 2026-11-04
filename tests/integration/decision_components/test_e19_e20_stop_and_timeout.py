"""E19–E20: cancel, defer, quit and esc at each stage; the answer timeout."""

from __future__ import annotations

import time

import pytest
from langchain_core.messages import AIMessage

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
    pending_interrupts,
    press_in_plate,
    qc_field,
    source_field,
    tool_results,
)


def _unless_stopped(message):
    """A scripted step that only asks when no earlier answer was a stop."""

    def step(messages):
        for m in messages:
            if m.type == "tool" and ('"cancel"' in m.content or '"defer"' in m.content):
                return AIMessage(
                    content="Scripted supervisor: stopped, nothing decided"
                )
        return message

    return step


def _journey(ds):
    """source → groups + direction (one request, two screens) → QC."""
    ids = {"source": new_id("src"), "groups": new_id("grp"), "qc": new_id("qc")}
    steps = [
        ask((ids["source"], "Which count source?", {"fields": [source_field(ds)]})),
        _unless_stopped(
            ask(
                (
                    ids["groups"],
                    "Set up the comparison groups",
                    {
                        "fields": [groups_field(ds), direction_field()],
                        "checks": [groups_kept(), no_default_for_unknown(ds)],
                    },
                )
            )
        ),
        _unless_stopped(
            ask(
                (
                    ids["qc"],
                    "Keep or exclude each flagged sample?",
                    {"fields": [qc_field(ds)], "checks": QC_CHECKS},
                )
            )
        ),
    ]
    return ids, steps


@pytest.mark.parametrize("stage", ["source", "groups", "direction", "qc"])
def test_e19_esc_cancels_a_single_question_at_each_stage(start_tui, review8, stage):
    """E19: esc on any single screen cancels that whole ask_user request. The
    plate (groups) and the choice_list (qc) ask first: one esc only warns."""
    ds = review8
    ids, steps = _journey(ds)
    tui = start_tui(steps)
    tui.ask_supervisor()
    order = ["source", "groups", "direction", "qc"]
    for number, screen in enumerate(order, 1):
        tui.wait_render(number)
        if screen == stage:
            if stage in ("groups", "qc"):
                sent = len(tui.responses())
                tui.key("esc")
                tui.wait_screen(
                    "press esc again to cancel this question; nothing will be sent"
                )
                assert len(tui.responses()) == sent  # a single esc sent nothing
            tui.key("esc")
            break
        if screen == "source":
            tui.key("1")
        elif screen == "groups":
            press_in_plate(tui, ds.ids, {"S08": "treated"})
        elif screen == "direction":
            tui.key("1")
    assert tui.wait_turn_end() == ""
    results = tool_results(tui.client)
    request = "groups" if stage == "direction" else stage
    assert results[ids[request]]["action"] == "cancel"
    assert "answer" not in results[ids[request]]
    before = order[: order.index(stage)]
    for done in {"groups" if s == "direction" else s for s in before} - {request}:
        assert results[ids[done]]["action"] == "submit"
    later = {"groups": ["qc"], "source": ["groups", "qc"]}.get(request, [])
    for skipped in later:
        assert ids[skipped] not in results  # the supervisor was told and stopped
    assert len(tui.renders()) == order.index(stage) + 1


def _start_dashboard(start_tui, ds):
    cid = new_id()
    tui = start_tui([ask((cid, "Set up the comparison", dashboard_context(ds)))])
    tui.ask_supervisor()
    tui.wait_render(1)
    return cid, tui


def _answer_whole_dashboard(tui, ds):
    tui.key("1")
    press_in_plate(tui, ds.ids, {"S08": "treated"})
    tui.key("1")
    answer_rows(tui, [("retain", None), ("retain", None)])
    tui.wait_screen("Every section is answered")


def test_e19_dashboard_single_esc_does_not_cancel_double_esc_does(start_tui, review8):
    """E19: on the sections one esc only warns; a second esc cancels."""
    cid, tui = _start_dashboard(start_tui, review8)
    tui.key("esc")
    tui.wait_screen("press esc again to cancel everything; nothing will be applied")
    tui.key("1")  # any other key disarms it; the dashboard still works
    tui.wait_screen("✓ 1. Which count source should the analysis use?")
    assert tui.responses() == []
    tui.key("esc")
    tui.wait_screen("press esc again to cancel everything")
    tui.key("esc")
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "cancel"
    assert "answer" not in result


@pytest.mark.parametrize("where", ["sections", "review"])
def test_e19_dashboard_ctrl_d_defers(start_tui, review8, where):
    """E19: ctrl+d defers from the sections and from the review screen."""
    ds = review8
    cid, tui = _start_dashboard(start_tui, ds)
    if where == "review":
        _answer_whole_dashboard(tui, ds)
        tui.key("ctrl+s")
        tui.wait_screen("This is exactly what will be sent")
    else:
        tui.key("1")  # part-way through
    tui.key("ctrl+d")
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "defer"
    assert "answer" not in result


def test_e19_esc_on_the_review_screen_goes_back_not_cancel(start_tui, review8):
    """E19: esc on the review screen returns to the sections."""
    ds = review8
    cid, tui = _start_dashboard(start_tui, ds)
    _answer_whole_dashboard(tui, ds)
    tui.key("ctrl+s")
    tui.wait_screen("This is exactly what will be sent")
    tui.key("esc")
    tui.wait_screen("4/4 answered")
    assert tui.responses() == []
    tui.key("esc")
    tui.wait_screen("press esc again")
    tui.key("esc")
    assert tui.wait_turn_end() == ""
    assert tool_results(tui.client)[cid]["action"] == "cancel"


def test_e19_ctrl_c_cancels_the_question_and_turn_second_ctrl_c_quits(
    start_tui, review8
):
    """E19 quit: the first ctrl+c on a question sends the question's cancel and
    cancels the running turn (the paused turn is still active, so the TUI's
    upstream ctrl+c means "cancel the turn"); ask_user records cancel. A second
    ctrl+c at the idle prompt quits the TUI and the launcher's loop ends."""
    ds = review8
    cid = new_id()
    tui = start_tui([ask((cid, "Which count source?", {"fields": [source_field(ds)]}))])
    tui.ask_supervisor()
    render = tui.wait_render(1)
    tui.key("ctrl+c")
    response = tui.wait_response(render.msg_id)
    assert response["payload"]["data"]["action"] == "cancel"
    assert tui.wait_turn_end() == "cancelled"
    assert tui.messages("from_go", "cancel")
    (resumed,) = tui.resumes
    assert [v["action"] for v in resumed.values()] == ["cancel"]
    assert tool_results(tui.client)[cid]["action"] == "cancel"
    assert tui.proc.poll() is None  # still running

    # Let the TUI take the launcher's last spinner-off, then it is idle.
    tui._wait(
        lambda: tui.messages("to_go")[-1]
        == {"version": 1, "type": "spinner", "payload": {"active": False}},
        10,
        "the launcher's final spinner-off",
    )
    time.sleep(0.5)
    tui.key("ctrl+c")
    tui.proc.wait(timeout=10)
    tui.loop_thread.join(timeout=20)
    assert not tui.loop_thread.is_alive()
    assert tui.messages("from_go", "quit")


def test_e19_protocol_quit_while_asking_ends_the_turn_without_resuming(
    start_tui, review8
):
    """E19 quit: a quit from the TUI mid-question ends the turn; nothing resumes."""
    ds = review8
    cid = new_id()
    tui = start_tui([ask((cid, "Which count source?", {"fields": [source_field(ds)]}))])
    tui.ask_supervisor()
    tui.wait_render(1)
    tui.inject("quit", {})
    assert tui.wait_turn_end() == "cancelled"
    assert tui.resumes == []
    assert tool_results(tui.client) == {}
    assert len(pending_interrupts(tui.client)) == 1  # paused, never answered


def test_e20_answer_timeout_ends_the_turn_and_a_late_reply_is_ignored(
    start_tui, review8, monkeypatch
):
    """E20: with a 2 s timeout the turn ends as cancelled, the supervisor is
    not resumed, and a reply sent afterwards is ignored (design.md)."""
    from lobster.cli_internal import go_tui_launcher

    monkeypatch.setattr(go_tui_launcher, "_ANSWER_TIMEOUT_S", 2)
    ds = review8
    cid = new_id()
    tui = start_tui([ask((cid, "Which count source?", {"fields": [source_field(ds)]}))])
    tui.ask_supervisor()
    render = tui.wait_render(1)
    started = time.time()
    assert tui.wait_turn_end(timeout=15) == "cancelled"
    assert 1.5 <= time.time() - started <= 10
    assert tui.resumes == []
    assert tool_results(tui.client) == {}

    tui.key("1")  # the question is still drawn; answering it now does nothing
    tui.wait_response(render.msg_id)
    time.sleep(1.0)
    assert tui.resumes == []
    assert tool_results(tui.client) == {}
    assert len(pending_interrupts(tui.client)) == 1
