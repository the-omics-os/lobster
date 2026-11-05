"""E1–E10: identity, unavailable source, group assignment and QC review.

Every test drives the real lobster-tui binary by keys on a pty and asserts on
the tool result in the supervisor's graph state (what the supervisor gets).
"""

from __future__ import annotations

from tests.integration.decision_components._harness import (
    LEAVE_OUT,
    LEAVE_OUT_REASON,
    QC_CHECKS,
    answer_rows,
    ask,
    dashboard_context,
    groups_field,
    groups_kept,
    new_id,
    no_default_for_unknown,
    press_in_plate,
    qc_field,
    source_field,
    tool_results,
)


def _shared_label_pair(ds):
    by_label = {}
    for s in ds.samples:
        by_label.setdefault(s["display_label"], []).append(s["sample_id"])
    return next(ids for ids in by_label.values() if len(ids) == 2)


def test_e01_shuffled_items_and_shared_label_are_keyed_by_id(start_tui, any_dataset):
    """E1: items in shuffled order, two of them sharing a display label."""
    ds = any_dataset
    first, second = _shared_label_pair(ds)
    others = [s for s in ds.warned if s not in (first, second)][:2]
    order = [second, others[0], first, others[1]]  # not the data's order
    cid = new_id()
    context = {"fields": [qc_field(ds, items=order)], "checks": QC_CHECKS}
    tui = start_tui([ask((cid, "Keep or exclude each flagged sample?", context))])
    tui.ask_supervisor()

    render = tui.wait_render(1)
    assert render.component == "choice_list"
    assert [item["id"] for item in render.data["items"]] == order
    label = ds.label(first)
    assert label == ds.label(second)
    tui.wait_screen(f"{label} ({first})")
    tui.wait_screen(f"{label} ({second})")

    reason = f"e2e reason for {second} only"
    answer_rows(
        tui,
        [("exclude", reason), ("retain", None), ("retain", None), ("defer", None)],
    )
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "submit"
    assert result["answer"]["qc"] == {
        second: {"id": "exclude", "fields": {"reason": reason}},
        others[0]: {"id": "retain"},
        first: {"id": "retain"},
        others[1]: {"id": "defer"},
    }


def test_e02_crafted_answer_with_an_unknown_id_is_reasked(start_tui, review8):
    """E2: a reply naming an id that is not in the question is never accepted."""
    ds = review8
    cid = new_id()
    context = {"fields": [qc_field(ds)], "checks": QC_CHECKS}
    tui = start_tui([ask((cid, "Keep or exclude each flagged sample?", context))])
    tui.ask_supervisor()
    first = tui.wait_render(1)

    crafted = {"S99": {"id": "retain"}, **{sid: {"id": "retain"} for sid in ds.warned}}
    tui.inject(
        "component_response",
        {"id": first.msg_id, "data": {"action": "submit", "answers": crafted}},
        msg_id=first.msg_id,
    )
    second = tui.wait_render(2)
    assert second.component == "choice_list"
    assert "S99" in second.data.get("details", "")
    tui.wait_screen("S99")

    answer_rows(tui, [("retain", None), ("defer", None)])
    assert tui.wait_turn_end() == ""
    results = tool_results(tui.client)
    result = results[cid]
    assert result["action"] == "submit"
    assert result["attempts"] == 2
    assert result["answer"]["qc"] == {"S02": {"id": "retain"}, "S06": {"id": "defer"}}
    assert "S99" not in str(results)


def test_e03_unavailable_source_is_refused_on_screen(start_tui, any_dataset):
    """E3: Enter (and its number key) on SRC-C is refused; SRC-C is in no answer."""
    ds = any_dataset
    cid = new_id()
    tui = start_tui([ask((cid, "Which count source?", {"fields": [source_field(ds)]}))])
    tui.ask_supervisor()
    render = tui.wait_render(1)
    assert [o["id"] for o in render.data["options"]] == ["SRC-A", "SRC-B", "SRC-C"]
    assert render.data["options"][2]["unavailable"]

    tui.key("down", "down", "enter")
    tui.wait_screen("Author summary table can't be picked")
    tui.key("3")
    tui.wait_screen("can't be picked: normalized expression")
    assert tui.responses() == []  # nothing was sent for the refused picks

    tui.key("1")
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["answer"] == {"source": {"id": "SRC-A"}}
    assert "SRC-C" not in str([r["payload"] for r in tui.responses()])


def _groups_question(ds, checks=True):
    context = {"fields": [groups_field(ds)]}
    if checks:
        context["checks"] = [groups_kept(), no_default_for_unknown(ds)]
    return context


def test_e04_unresolved_sample_blocks_the_plate_submit(start_tui, review8):
    """E4: S08 starts unresolved; Enter is refused until it is assigned."""
    ds = review8
    cid = new_id()
    tui = start_tui(
        [ask((cid, "Which group is each sample in?", _groups_question(ds)))]
    )
    tui.ask_supervisor()
    render = tui.wait_render(1)
    assert render.component == "plate_map"
    wells = {w["sample_id"]: w["condition"] for w in render.data["wells"]}
    assert wells["S08"] == ""  # no default: the suggestion is not applied

    tui.key("enter")
    tui.wait_screen("Resolve before submitting (press a number): A8 S08")
    assert tui.responses() == []

    # The refused enter moved the cursor to S08, the sample it named.
    press_in_plate(tui, ds.ids, {"S08": "treated"}, start="S08")
    assert tui.wait_turn_end() == ""
    groups = tool_results(tui.client)[cid]["answer"]["groups"]
    assert groups["S08"] == {"id": "treated"}
    assert {sid: g["id"] for sid, g in groups.items() if sid != "S08"} == {
        sid: ds.condition(sid) for sid in ds.ids if sid != "S08"
    }


def test_e05_unresolved_sample_left_out_is_in_no_group(start_tui, review8):
    """E5: S08 put in "leave out" is accepted and belongs to no group."""
    ds = review8
    cid = new_id()
    tui = start_tui(
        [ask((cid, "Which group is each sample in?", _groups_question(ds)))]
    )
    tui.ask_supervisor()
    tui.wait_render(1)
    press_in_plate(tui, ds.ids, {"S08": LEAVE_OUT})
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "submit"
    groups = result["answer"]["groups"]
    assert groups["S08"] == {"id": LEAVE_OUT, "fields": {"reason": LEAVE_OUT_REASON}}
    assert "S08" not in [
        s for s, g in groups.items() if g["id"] in ("control", "treated")
    ]


def test_e06_every_control_moved_is_reasked_with_answers_kept(start_tui, review8):
    """E6: an empty control group comes back with the problem, answers kept."""
    ds = review8
    cid = new_id()
    tui = start_tui(
        [ask((cid, "Which group is each sample in?", _groups_question(ds)))]
    )
    tui.ask_supervisor()
    tui.wait_render(1)
    moved = {sid: "treated" for sid in ds.of("control")}
    press_in_plate(tui, ds.ids, {**moved, "S08": "treated"})

    second = tui.wait_render(2)
    assert second.component == "plate_map"
    assert "'control' has no items" in second.data["question"]
    tui.wait_screen("'control' has no items")
    kept = {w["sample_id"]: w["condition"] for w in second.data["wells"]}
    assert kept == {sid: "treated" for sid in ds.ids}  # every answer kept

    press_in_plate(tui, ds.ids, {"S01": "control"})
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["attempts"] == 2
    assert result["answer"]["groups"] == {
        sid: {"id": "control" if sid == "S01" else "treated"} for sid in ds.ids
    }


def _dashboard(ds):
    return dashboard_context(ds)


def _first_dashboard_pass(tui, ds, qc_rows, groups):
    """source → groups (plate) → direction → QC, then ctrl+s and Enter."""
    tui.key("1")  # source: SRC-A
    press_in_plate(tui, ds.ids, groups)
    tui.key("1")  # direction: treated compared with control
    answer_rows(tui, qc_rows)
    tui.wait_screen("Every section is answered")
    tui.key("ctrl+s")
    tui.wait_screen("This is exactly what will be sent")
    tui.key("enter")


def _empty_control_via_qc(start_tui, ds, qc_option):
    cid = new_id()
    tui = start_tui([ask((cid, "Set up the comparison", _dashboard(ds)))])
    tui.ask_supervisor()
    tui.wait_render(1)
    # Only S02 stays control; QC then takes S02 out.
    groups = {"S01": LEAVE_OUT, "S03": LEAVE_OUT, "S04": LEAVE_OUT, "S08": "treated"}
    reason = "e2e: low library size" if qc_option == "exclude" else None
    _first_dashboard_pass(tui, ds, [(qc_option, reason), ("retain", None)], groups)
    return cid, tui, groups


_TITLES = {
    "source": "Which count source should the analysis use?",
    "groups": "Which group is each sample in?",
    "direction": "Compare which group with which?",
    "qc": "Keep or exclude each flagged sample?",
}


def active_section(flat_text):
    """The dashboard section the pointer (▸) is on, from the screen."""
    for number, (key, title) in enumerate(_TITLES.items(), 1):
        for icon in ("○", "⚠", "✓"):
            if f"▸ {icon} {number}. {title}" in flat_text:
                return key
    return None


def recheck_open_sections(tui, qc_rows):
    """Answer the active section until every section is answered.

    Re-confirming an unchanged plate may still mark its dependents stale
    (see E2E_REPORT.md); each stale section is then checked again here, so
    the journey itself can finish. That contract is asserted separately.
    Returns the sections visited, in order.
    """
    visited = []
    for _ in range(6):
        text = tui.flat()
        if "Every section is answered" in text:
            return visited
        key = active_section(text)
        visited.append(key)
        if key == "qc":
            answer_rows(tui, qc_rows)
        elif key in ("source", "direction"):
            tui.key("1")
        elif key == "groups":
            tui.key("enter")
        else:
            raise AssertionError(f"no active section on screen:\n{tui.screen.text()}")
        tui.wait_screen(lambda t, before=text: t != before, timeout=10)
    tui.wait_screen("Every section is answered")
    return visited


def _fix_and_resubmit(tui, ds):
    """On the re-asked dashboard: keep groups, set S02 to retain, submit."""
    tui.wait_screen("▸ ○ 2. Which group is each sample in?")
    visited = recheck_open_sections(tui, [("retain", None), ("retain", None)])
    tui.key("ctrl+s")
    tui.wait_screen("This is exactly what will be sent")
    tui.key("enter")
    return visited


def _assert_problem_on_groups_and_qc(render):
    sections = {s["key"]: s for s in render.data["sections"]}
    for key in ("groups", "qc"):
        assert any(
            "'control' has no items left" in p for p in sections[key]["problems"]
        )
    for key in ("source", "direction"):
        assert "problems" not in sections[key]


def test_e07_excluding_every_remaining_control_is_reasked_then_fixed(
    start_tui, review8
):
    """E7: dashboard: QC excludes the last control → problem on groups and QC."""
    ds = review8
    cid, tui, groups = _empty_control_via_qc(start_tui, ds, "exclude")
    second = tui.wait_render(2)
    assert second.component == "dashboard"
    _assert_problem_on_groups_and_qc(second)
    tui.wait_screen(
        "'control' has no items left after removing S02 (Donor B) (qc: exclude)"
    )

    _fix_and_resubmit(tui, ds)
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["action"] == "submit"
    assert result["attempts"] == 2
    assert result["answer"]["qc"] == {"S02": {"id": "retain"}, "S06": {"id": "retain"}}
    assert result["answer"]["groups"]["S02"] == {"id": "control"}
    for sid, choice in groups.items():
        expected = {"id": choice}
        if choice == LEAVE_OUT:  # a removal carries its reason
            expected["fields"] = {"reason": LEAVE_OUT_REASON}
        assert result["answer"]["groups"][sid] == expected


def test_e08_deferring_the_last_control_is_refused_like_e7(start_tui, review8):
    """E8 (U-13): a deferred sample does not count toward its group."""
    ds = review8
    cid, tui, _ = _empty_control_via_qc(start_tui, ds, "defer")
    second = tui.wait_render(2)
    _assert_problem_on_groups_and_qc(second)
    tui.wait_screen("(qc: defer)")
    _fix_and_resubmit(tui, ds)
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["attempts"] == 2
    assert result["answer"]["qc"]["S02"] == {"id": "retain"}


def test_e09_exclude_with_an_empty_or_blank_reason_is_refused(start_tui, review8):
    """E9: the reason box refuses empty and whitespace-only reasons."""
    ds = review8
    cid = new_id()
    context = {"fields": [qc_field(ds)], "checks": QC_CHECKS}
    tui = start_tui([ask((cid, "Keep or exclude each flagged sample?", context))])
    tui.ask_supervisor()
    tui.wait_render(1)

    tui.key("3")  # S02: exclude → the reason box opens
    tui.wait_screen("enter save · esc discard")
    tui.key("enter")
    tui.wait_screen("A reason is needed for exclude (S02)")
    tui.type("   ")
    tui.key("enter")
    tui.wait_screen("A reason is needed for exclude (S02)")
    tui.key("esc")  # discard: S02 goes back to undecided
    tui.wait_screen("not decided yet")
    tui.key("enter")  # submit with nothing decided is refused too
    tui.wait_screen("Not finished: choose an option for S02, S06")
    assert tui.responses() == []

    answer_rows(tui, [("exclude", "  low library size  "), ("retain", None)])
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["answer"]["qc"]["S02"] == {
        "id": "exclude",
        "fields": {"reason": "low library size"},
    }


def test_e10_retain_and_defer_need_no_reason(start_tui, review8):
    """E10: retain and defer are accepted without a reason."""
    ds = review8
    cid = new_id()
    context = {"fields": [qc_field(ds)], "checks": QC_CHECKS}
    tui = start_tui([ask((cid, "Keep or exclude each flagged sample?", context))])
    tui.ask_supervisor()
    tui.wait_render(1)
    answer_rows(tui, [("retain", None), ("defer", None)])
    assert tui.wait_turn_end() == ""
    result = tool_results(tui.client)[cid]
    assert result["attempts"] == 1
    assert result["answer"]["qc"] == {"S02": {"id": "retain"}, "S06": {"id": "defer"}}
