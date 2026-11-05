"""E11–E18: request rules, re-ask, the re-ask cap, stale / misrouted /
duplicate replies and two pending questions.

"Crafted" replies are written into the protocol pipe towards the bridge (as
if the binary had sent them), or made by rewriting the answer envelope the
launcher hands to the real ``client.resume_from_interrupt``.
"""

from __future__ import annotations

import re

import time

from tests.integration.decision_components._harness import (
    QC_CHECKS,
    answer_rows,
    ask,
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


def _groups_context(ds):
    return {
        "fields": [groups_field(ds)],
        "checks": [groups_kept(), no_default_for_unknown(ds)],
    }


# --- E11 -------------------------------------------------------------------


def test_e11_prefilled_unknown_sample_is_invalid_context_and_nothing_shown(
    start_tui, review8
):
    """E11a: the supervisor pre-fills S08 although no_default_for names it."""
    ds = review8
    field = groups_field(ds)
    field["default"]["S08"] = {"id": "treated"}  # the suggestion, applied
    context = {"fields": [field], "checks": [groups_kept(), no_default_for_unknown(ds)]}
    cid = new_id()
    tui = start_tui([ask((cid, "Which group is each sample in?", context))])
    tui.ask_supervisor()
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "invalid_context"
    assert any("S08" in p["message"] for p in result["problems"])
    assert tui.renders() == []
    assert "Which group is each sample in?" not in tui.flat()


def test_e11_prefilled_exclude_for_a_warned_sample_is_invalid_context(
    start_tui, review8
):
    """E11b: a warning alone may not pre-select exclude (no_default_for)."""
    ds = review8
    field = qc_field(
        ds, default={"S02": {"id": "exclude", "fields": {"reason": "low library"}}}
    )
    cid = new_id()
    tui = start_tui(
        [ask((cid, "Keep or exclude?", {"fields": [field], "checks": QC_CHECKS}))]
    )
    tui.ask_supervisor()
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "invalid_context"
    assert any(
        "exclude" in p["message"] and "S02" in p["message"] for p in result["problems"]
    )
    assert tui.renders() == []


# --- E12 / E13 -------------------------------------------------------------


def test_e12_after_a_rejection_every_other_answer_is_still_prefilled_72(
    start_tui, review72
):
    """E12: 72 samples: the re-asked plate keeps all 72 submitted answers."""
    ds = review72
    cid = new_id()
    tui = start_tui([ask((cid, "Which group is each sample in?", _groups_context(ds)))])
    tui.ask_supervisor()
    first = tui.wait_render(1)
    assert len(first.data["wells"]) == 72
    # Resolve the four unknowns and move every control to treated.
    assignments = {sid: "treated" for sid in ds.of("control") + ds.unknown}
    press_in_plate(tui, ds.ids, dict(sorted(assignments.items())))

    second = tui.wait_render(2)
    assert "'control' has no items" in second.data["question"]
    tui.wait_screen("'control' has no items")
    kept = {w["sample_id"]: w["condition"] for w in second.data["wells"]}
    assert kept == {sid: "treated" for sid in ds.ids}
    assert len(kept) == 72

    press_in_plate(tui, ds.ids, {"S01": "control"})
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["attempts"] == 2
    groups = result["answer"]["groups"]
    assert len(groups) == 72
    assert groups["S01"] == {"id": "control"}
    assert all(g == {"id": "treated"} for sid, g in groups.items() if sid != "S01")


def test_e13_five_rejected_answers_in_a_row_return_rejected(start_tui, review8):
    """E13: the same refused answer five times → the supervisor gets rejected."""
    ds = review8
    cid = new_id()
    tui = start_tui([ask((cid, "Which group is each sample in?", _groups_context(ds)))])
    tui.ask_supervisor()
    tui.wait_render(1)
    press_in_plate(
        tui, ds.ids, {**{s: "treated" for s in ds.of("control")}, "S08": "treated"}
    )
    for number in range(2, 6):
        tui.wait_render(number)
        tui.key("enter")  # the pre-filled (refused) answer again
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "rejected"
    assert result["attempts"] == 5
    assert any("'control' has no items" in p["message"] for p in result["problems"])
    assert len(tui.renders()) == 5


# --- E14 / E15 / E16 -------------------------------------------------------


def test_e14_reply_to_version_1_after_version_2_is_shown_is_reasked(start_tui, review8):
    """E14: a stale render id is ignored; a stale version is re-asked."""
    ds = review8
    cid = new_id()
    tui = start_tui([ask((cid, "Which group is each sample in?", _groups_context(ds)))])
    tui.ask_supervisor()
    first = tui.wait_render(1)
    press_in_plate(
        tui, ds.ids, {**{s: "treated" for s in ds.of("control")}, "S08": "treated"}
    )
    second = tui.wait_render(2)

    # (a) A crafted reply to version 1's render while version 2 is shown.
    valid = {sid: ds.condition(sid) for sid in ds.ids if sid != "S08"}
    tui.inject(
        "component_response",
        {
            "id": first.msg_id,
            "data": {"action": "submit", "assignments": {**valid, "S08": "treated"}},
        },
        msg_id=first.msg_id,
    )
    time.sleep(1.0)
    assert len(tui.renders()) == 2 and len(tui.resumes) == 1  # ignored

    # (b) Version 2's answer, rewritten to claim version 1.
    def to_version_1(response):
        (envelope,) = response.values()
        envelope["responds_to"]["request_version"] = 1
        return response

    tui.rewrite_resume = to_version_1
    press_in_plate(tui, ds.ids, {"S01": "control"})
    third = tui.wait_render(3)
    tui.rewrite_resume = None
    assert "version 1 of the question" in third.data["question"]
    tui.wait_screen("this answer is for version 1 of the question")
    kept = {w["sample_id"]: w["condition"] for w in third.data["wells"]}
    assert kept["S01"] == "control"  # the answer to version 2 is kept
    assert tool_results(tui.client) == {}  # not consumed

    tui.key("enter")
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["attempts"] == 3
    assert result["answer"]["groups"]["S01"] == {"id": "control"}
    assert second.msg_id != first.msg_id


def test_e15_data_file_changed_before_submit_is_reasked_answers_kept(
    start_tui, review8
):
    """E15: a data_files file edited while the question is shown."""
    ds = review8
    cid = new_id()
    context = {
        "fields": [qc_field(ds)],
        "checks": QC_CHECKS,
        "data_files": [ds.rel("qc_metrics.csv")],
    }
    tui = start_tui([ask((cid, "Keep or exclude each flagged sample?", context))])
    tui.ask_supervisor()
    first = tui.wait_render(1)
    path = ds.path("qc_metrics.csv")
    path.write_text(path.read_text() + "S99,1,0.0,1,FICTIONAL edited during review\n")

    answer_rows(tui, [("exclude", "low library size"), ("retain", None)])
    second = tui.wait_render(2)
    message = (
        f"the data changed while you were reviewing it: {ds.rel('qc_metrics.csv')}"
    )
    assert message in second.data["details"]
    tui.wait_screen("the data changed while you were reviewing it")
    assert second.data["default"] == {
        "S02": {"id": "exclude", "fields": {"reason": "low library size"}},
        "S06": {"id": "retain"},
    }
    assert first.msg_id != second.msg_id

    tui.key("enter")  # the kept answers, now against the current file
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "submit"
    assert result["attempts"] == 2
    assert result["answer"]["qc"]["S02"]["fields"]["reason"] == "low library size"


def test_e16_reply_with_another_render_id_is_ignored(start_tui, review8):
    """E16a: a component_response for another render never answers this one."""
    ds = review8
    cid = new_id()
    tui = start_tui([ask((cid, "Which count source?", {"fields": [source_field(ds)]}))])
    tui.ask_supervisor()
    render = tui.wait_render(1)
    tui.inject(
        "component_response",
        {
            "id": "render-of-something-else",
            "data": {"action": "submit", "selected": [{"id": "SRC-B"}]},
        },
        msg_id="render-of-something-else",
    )
    time.sleep(1.0)
    assert tui.resumes == [] and tool_results(tui.client) == {}
    tui.key("1")
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["answer"] == {"source": {"id": "SRC-A"}}
    assert result["attempts"] == 1
    assert render.msg_id


def test_e16_answer_for_another_decision_is_reasked_not_consumed(start_tui, review8):
    """E16b: an answer whose decision_id is another one's is re-asked."""
    ds = review8
    cid = new_id()
    tui = start_tui([ask((cid, "Which count source?", {"fields": [source_field(ds)]}))])
    tui.ask_supervisor()
    tui.wait_render(1)

    def to_other_decision(response):
        (envelope,) = response.values()
        envelope["responds_to"]["decision_id"] = "dec-some-other-call"
        return response

    tui.rewrite_resume = to_other_decision
    tui.key("2")
    second = tui.wait_render(2)
    tui.rewrite_resume = None
    assert "dec-some-other-call" in second.data["details"]
    tui.wait_screen("this answer is for decision dec-some-other-call")
    assert "default" not in second.data  # a misrouted reply pre-fills nothing
    assert tool_results(tui.client) == {}

    tui.key("1")
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["answer"] == {"source": {"id": "SRC-A"}}
    assert result["attempts"] == 2


# --- E17 -------------------------------------------------------------------


def test_e17_duplicate_reply_and_reply_after_the_turn_are_ignored(start_tui, review8):
    """E17: the same reply twice, and once more after the turn ended."""
    ds = review8
    cid = new_id()
    tui = start_tui([ask((cid, "Which count source?", {"fields": [source_field(ds)]}))])
    tui.ask_supervisor()
    render = tui.wait_render(1)
    tui.key("1")
    reply = tui.wait_response(render.msg_id)
    tui.inject("component_response", reply["payload"], msg_id=render.msg_id)
    assert tui.wait_turn_end() == ""
    tui.inject("component_response", reply["payload"], msg_id=render.msg_id)
    time.sleep(1.0)

    assert len(tui.resumes) == 1
    results = tool_results(tui.client)
    assert list(results) == [cid]
    assert results[cid]["attempts"] == 1
    assert tui.turn_ends() == [""]
    assert pending_interrupts(tui.client) == []

    # The loop is still alive and nothing resumes: a new turn runs normally.
    tui.ask_supervisor("anything else?")
    assert tui.wait_turn_end(2) == ""
    assert len(tui.resumes) == 1
    assert list(tool_results(tui.client)) == [cid]


# --- E18 -------------------------------------------------------------------


def test_e18_two_parallel_questions_answered_in_reverse_order(start_tui, review8):
    """E18: two ask_user calls in one message; resumed in reverse key order."""
    ds = review8
    cid_source, cid_qc = new_id("src"), new_id("qc")
    qc_context = {"fields": [qc_field(ds)], "checks": QC_CHECKS}
    tui = start_tui(
        [
            ask(
                (cid_source, "Which count source?", {"fields": [source_field(ds)]}),
                (cid_qc, "Keep or exclude each flagged sample?", qc_context),
            )
        ]
    )
    tui.rewrite_resume = lambda response: dict(reversed(list(response.items())))
    tui.ask_supervisor()

    first = tui.wait_render(1)
    if first.component == "choice":
        tui.key("2")
        tui.wait_render(2)
        answer_rows(tui, [("defer", None), ("retain", None)])
    else:
        answer_rows(tui, [("defer", None), ("retain", None)])
        tui.wait_render(2)
        tui.key("2")
    assert tui.wait_turn_end() == ""

    (response,) = tui.resumes
    assert len(response) == 2
    by_decision = {v["responds_to"]["decision_id"]: k for k, v in response.items()}
    # decision_id is dec-<tool_call_id>-<8 hex> (U-26 #3): one per call.
    assert sorted(re.sub(r"-[0-9a-f]{8}$", "", d) for d in by_decision) == sorted(
        [f"dec-{cid_source}", f"dec-{cid_qc}"]
    )
    assert all(re.fullmatch(r"dec-.+-[0-9a-f]{8}", d) for d in by_decision)
    results = tool_results(tui.client)
    assert results[cid_source]["answer"] == {"source": {"id": "SRC-B"}}
    assert results[cid_qc]["answer"] == {
        "qc": {"S02": {"id": "defer"}, "S06": {"id": "retain"}}
    }
