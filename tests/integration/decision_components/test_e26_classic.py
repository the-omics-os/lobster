"""E26: the same journeys through the classic (non-Go) chat UI.

No binary here: the classic UI is plain prompts on stdin/stdout. The real
AgentClient with the scripted supervisor runs through chat_commands'
``_query_classic`` (non-streaming query, resume) and
``classic_interaction.handle_interrupt_classic``; stdin is a scripted text
stream, stdout is captured.
"""

from __future__ import annotations

import io
import sys

import pytest

from tests.integration.decision_components._harness import (
    QC_CHECKS,
    ask,
    dashboard_context,
    groups_field,
    groups_kept,
    make_client,
    new_id,
    no_default_for_unknown,
    qc_field,
    tool_results,
)


def _run_classic(monkeypatch, client, lines):
    from lobster.cli_internal.commands.heavy.chat_commands import _query_classic

    stdin = io.StringIO("".join(f"{line}\n" for line in lines))
    monkeypatch.setattr(sys, "stdin", stdin)
    result = _query_classic(client, "run the scripted review")
    assert stdin.read() == "", "not every scripted line was asked for"
    return result


@pytest.fixture
def classic_client(workspace):
    def build(steps):
        return make_client(workspace, steps)

    return build


def test_e26_classic_plate_is_rechecked_and_reasked_with_answers_kept(
    monkeypatch, capsys, classic_client, review8
):
    """E26: structured prompts per sample; the check refuses an empty control
    group; the re-ask shows the problem and keeps every answer as default."""
    ds = review8
    cid = new_id()
    context = {
        "fields": [groups_field(ds)],
        "checks": [groups_kept(), no_default_for_unknown(ds)],
    }
    client = classic_client([ask((cid, "Which group is each sample in?", context))])
    first = [
        "2",
        "2",
        "2",
        "2",
        "",
        "",
        "",
        "2",
    ]  # every control → treated; S08 treated
    second = ["1"] + [""] * 7  # S01 back to control; Enter keeps the rest
    result = _run_classic(monkeypatch, client, first + second)

    out = capsys.readouterr().out
    assert "(At any prompt: d = decide later, q = cancel.)" in out
    assert "1. control" in out
    assert "3. leave out (leave_out) (asks for a reason)" in out
    assert (
        "S08 Donor H (Condition label unresolved; a second manifest suggests treated"
        in out
    )
    assert "⚠ 'control' has no items; give it at least one" in out
    assert out.count("S05 Donor A") == 2  # asked twice ...
    assert "S05 Donor A [Enter = 2]" in out  # ... the second time with its answer kept
    tool = tool_results(client)[cid]
    assert tool["attempts"] == 2
    assert tool["answer"]["groups"] == {
        sid: {"id": "control" if sid == "S01" else "treated"} for sid in ds.ids
    }
    assert result.get("success", True)


def test_e26_classic_dashboard_refuses_unavailable_and_blank_reason(
    monkeypatch, capsys, classic_client, review8
):
    """E26: the dashboard as parts; SRC-C refused; a reason is required; the
    summary is confirmed; the answer is checked like the TUI's."""
    ds = review8
    cid = new_id()
    client = classic_client(
        [ask((cid, "Set up the comparison", dashboard_context(ds)))]
    )
    lines = (
        ["3", "1"]  # source: SRC-C refused, then SRC-A
        + [""] * 7
        + ["2"]  # plate: keep the data's groups, S08 → treated
        + ["1"]  # direction
        + [
            "3",
            "",
            "low library size",
            "1",
        ]  # QC: S02 exclude (blank reason refused), S06 retain
        + ["y"]  # confirm all
    )
    _run_classic(monkeypatch, client, lines)
    out = capsys.readouterr().out
    assert "SRC-C can't be picked: normalized expression (log2 CPM)" in out
    assert "— Part 4 of 4 —" in out
    assert "An answer is needed." in out
    assert "Your answers:" in out
    result = tool_results(client)[cid]
    assert result["action"] == "submit"
    assert result["answer"]["source"] == {"id": "SRC-A"}
    assert result["answer"]["groups"]["S08"] == {"id": "treated"}
    assert result["answer"]["direction"] == {"id": "treated_vs_control"}
    assert result["answer"]["qc"] == {
        "S02": {"id": "exclude", "fields": {"reason": "low library size"}},
        "S06": {"id": "retain"},
    }


def test_e26_classic_dashboard_reask_after_emptying_control(
    monkeypatch, capsys, classic_client, review8
):
    """E26: QC excludes the last control → re-asked with the problem on groups
    and QC, answers kept; fixed → accepted (the E7 journey, classic)."""
    ds = review8
    cid = new_id()
    client = classic_client(
        [ask((cid, "Set up the comparison", dashboard_context(ds)))]
    )
    first = (
        ["1"]
        # only S02 stays control; each sample left out is asked a reason
        + ["3", "pilot", "", "3", "pilot", "3", "pilot", "", "", "", "2"]
        + ["1"]
        + ["3", "low library size", "1"]
        + ["y"]
    )
    second = (
        [""]  # source kept
        + [""] * 11  # plate kept: 8 samples, and the 3 kept reasons
        + [""]  # direction kept
        + ["1", ""]  # S02 → retain; S06 kept
        + ["y"]
    )
    _run_classic(monkeypatch, client, first + second)
    out = capsys.readouterr().out
    problem = "'control' has no items left after removing S02 (Donor B) (qc: exclude)"
    assert out.count(f"⚠ {problem}") == 2  # under groups and under QC
    result = tool_results(client)[cid]
    assert result["attempts"] == 2
    assert result["answer"]["qc"]["S02"] == {"id": "retain"}
    assert result["answer"]["groups"]["S01"] == {
        "id": "leave_out",
        "fields": {"reason": "pilot"},
    }


@pytest.mark.parametrize("key, action", [("d", "defer"), ("q", "cancel")])
def test_e26_classic_d_defers_and_q_cancels(
    monkeypatch, classic_client, review8, key, action
):
    """E26: d (decide later) and q (cancel) at any prompt stop the question."""
    ds = review8
    cid = new_id()
    context = {"fields": [qc_field(ds)], "checks": QC_CHECKS}
    client = classic_client(
        [ask((cid, "Keep or exclude each flagged sample?", context))]
    )
    _run_classic(monkeypatch, client, ["1", key])
    result = tool_results(client)[cid]
    assert result["action"] == action
    assert "answer" not in result
