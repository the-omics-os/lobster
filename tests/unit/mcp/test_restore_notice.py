"""A session restored from ``output_dir`` has only what was saved. The turn that
continues it says which modalities earlier turns made and the restore did not
bring back: in its warnings, in ``result.json``, and in its prompt."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from lobster.cli_internal import mcp_record, mcp_stdio, mcp_turns
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_continue import ERROR, go_on
from tests.unit.mcp.test_publish import H5AD, hidden, scan  # noqa: F401  (fixture)
from tests.unit.mcp.test_restore import earlier  # noqa: F401  (fixture)
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    call,
    data,
    engine,
    wait_terminal,
)

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "lobster/schemas/lobster.result.v1.schema.json"
    ).read_text()
)


def made(out: Path, *names: str, number: int = 1) -> None:
    """Record, in a result file, modalities that earlier turns made."""
    path = out / ("result.json" if number == 1 else f"result_{number}.json")
    record = json.loads(path.read_text())
    record["modalities"] = ["pbmc", *names]
    record["modality_details"] = [{"name": n} for n in record["modalities"]]
    path.write_text(json.dumps(record))
    workspace = out / mcp_record.WORKSPACE_DIR
    state = mcp_turns.load(workspace)
    state["turns"][-1]["modalities"] = record["modalities"]
    (workspace / mcp_turns.STATE_NAME).write_text(json.dumps(state))


def notice_lines(prompt: str) -> list:
    return [ln for ln in prompt.split("\n") if ln.startswith(mcp_record.NOTICE_LEAD)]


# the record ------------------------------------------------------------------


def test_the_modalities_earlier_turns_recorded_are_read_from_the_result_files(tmp_path):
    for name, names in (
        ("result.json", ["a", "b"]),
        ("result_2.json", ["b", "c"]),
        ("result_3.json", ["x y", "d\n"]),
    ):
        details = [{"name": n} for n in names]
        (tmp_path / name).write_text(
            json.dumps({"modalities": names, "modality_details": details})
        )
    (tmp_path / "result_4.json").write_text("{not json")
    (tmp_path / "result_x.json").write_text(json.dumps({"modalities": ["z"]}))
    assert mcp_record.recorded_modalities(tmp_path) == ["a", "b", "c", "xy", "d"]


def test_only_what_the_restore_lacks_is_not_restored():
    assert mcp_record.not_restored(["a", "b", "c"], ["c", "a", "new"]) == ["b"]
    assert mcp_record.not_restored(["a"], ["a"]) == []


# the prompt ------------------------------------------------------------------


def test_the_notice_is_one_fixed_line_after_the_instruction_block():
    plain = mcp_record.compose_continue_prompt("cluster it")
    assert notice_lines(plain) == []
    prompt = mcp_record.compose_continue_prompt("cluster it", ["raw", 'x"y'])
    assert prompt.startswith(plain + "\n")
    (notice,) = prompt[len(plain) + 1 :].split("\n")
    assert notice.startswith(mcp_record.NOTICE_LEAD)
    assert "written by Lobster" in notice and "not by the host" in notice
    assert "run again" in notice
    assert json.dumps(["raw", 'x"y']) in notice


def test_the_notice_is_not_part_of_what_the_host_wrote():
    prompt = mcp_record.compose_continue_prompt("cluster it", ["raw"])
    lines = prompt.split("\n")
    quoted = json.dumps({"instruction": "cluster it"})
    assert lines.index(quoted) == 2  # the instruction line is unchanged
    assert "raw" not in lines[2]


def test_a_long_list_is_cut_and_the_rest_counted():
    names = [f"m{i}" for i in range(mcp_record.MAX_NOT_RESTORED + 5)]
    shown, total = mcp_record.bounded_names(names)
    assert shown == names[: mcp_record.MAX_NOT_RESTORED] and total == len(names)
    prompt = mcp_record.compose_continue_prompt("x", names)
    assert "m9" in prompt and f"m{mcp_record.MAX_NOT_RESTORED}" not in prompt
    assert "5 more" in prompt


# the record of the turn ------------------------------------------------------


def test_the_request_lists_what_was_not_restored():
    plain = mcp_record.request_record("scrna_qc", "/in", "x", "mcp:h", 2)
    assert "not_restored" not in plain
    request = mcp_record.request_record(
        "scrna_qc", "/in", "x", "mcp:h", 2, not_restored=["a", "b"]
    )
    assert request["not_restored"] == ["a", "b"]
    assert "not_restored_total" not in request
    many = [f"m{i}" for i in range(mcp_record.MAX_NOT_RESTORED + 3)]
    request = mcp_record.request_record(
        "scrna_qc", "/in", "x", "mcp:h", 2, not_restored=many
    )
    assert len(request["not_restored"]) == mcp_record.MAX_NOT_RESTORED
    assert request["not_restored_total"] == len(many)
    record = mcp_record.result_record(
        {
            "schema_version": "lobster.result/1",
            "engine": {"name": "lobster-ai", "version": "1"},
            "session_id": None,
            "run_id": "run_x",
            "response": "",
        },
        {
            "metrics": {},
            "warnings": [{"code": "modalities_not_restored", "message": "x"}],
        },
        request,
    )
    Draft202012Validator(SCHEMA).validate(record)


def test_the_warning_code_is_in_the_closed_list():
    assert "modalities_not_restored" in mcp_record.WARNING_CODES


# the server ------------------------------------------------------------------


def test_the_turn_after_a_restart_names_the_modalities_that_did_not_come_back(earlier):
    out, first_id, opened, script = earlier
    made(out, "pbmc_filtered", "pbmc_clustered")
    second = McpServer(ThreadRunner())
    done = wait_terminal(
        second, go_on(second, "label them", output_dir=str(out))["run_id"]
    )
    assert done["status"] == "completed"
    (warning,) = [w for w in done["warnings"] if w["code"] == "modalities_not_restored"]
    assert (
        "pbmc_filtered" in warning["message"] and "pbmc_clustered" in warning["message"]
    )
    assert "pbmc," not in warning["message"]  # that one came back
    record = json.loads((out / "result_2.json").read_text())
    assert record["request"]["not_restored"] == ["pbmc_filtered", "pbmc_clustered"]
    assert record["warnings"] == done["warnings"]
    Draft202012Validator(SCHEMA).validate(record)
    (prompt,) = opened[1]["client"].prompts
    assert prompt == mcp_record.compose_continue_prompt(
        "label them", ["pbmc_filtered", "pbmc_clustered"]
    )
    assert len(notice_lines(prompt)) == 1


def test_a_restore_that_lost_nothing_says_nothing(earlier):
    out, _first_id, opened, script = earlier
    second = McpServer(ThreadRunner())
    done = wait_terminal(second, go_on(second, output_dir=str(out))["run_id"])
    assert "modalities_not_restored" not in [w["code"] for w in done["warnings"]]
    assert (
        "not_restored" not in json.loads((out / "result_2.json").read_text())["request"]
    )
    assert notice_lines(opened[1]["client"].prompts[0]) == []


def test_the_notice_is_raised_once_per_restore_whatever_became_of_the_turn(earlier):
    out, _first_id, opened, script = earlier
    made(out, "gone")
    second = McpServer(ThreadRunner())
    script["events"] = [ERROR]
    failed = wait_terminal(second, go_on(second, output_dir=str(out))["run_id"])
    assert failed["status"] == "failed"
    opened[1]["client"].events = [COMPLETE]
    done = wait_terminal(second, go_on(second, "again", output_dir=str(out))["run_id"])
    assert "modalities_not_restored" not in [w["code"] for w in done["warnings"]]
    third = wait_terminal(
        second, go_on(second, "once more", run_id=done["run_id"])["run_id"]
    )
    assert "modalities_not_restored" not in [w["code"] for w in third["warnings"]]
    prompts = opened[1]["client"].prompts
    assert [len(notice_lines(p)) for p in prompts] == [1, 0, 0]


def test_a_session_kept_in_memory_is_not_told_anything(earlier):
    out, _first_id, opened, script = earlier
    made(out, "gone")
    server = McpServer(ThreadRunner())
    restored = wait_terminal(server, go_on(server, output_dir=str(out))["run_id"])
    done = wait_terminal(
        server, go_on(server, "next", run_id=restored["run_id"])["run_id"]
    )
    assert "modalities_not_restored" not in [w["code"] for w in done["warnings"]]


# where the notice comes from ---------------------------------------------------


def test_a_modality_that_came_back_as_an_autosave_is_not_lost():
    assert mcp_record.not_restored(["foo", "bar"], ["foo_autosave", "bar"]) == []
    assert mcp_record.not_restored(["foo_autosave"], ["foo"]) == []
    assert mcp_record.not_restored(["foo"], ["bar_autosave"]) == ["foo"]


def test_a_modality_recorded_twice_is_listed_once():
    assert mcp_record.not_restored(["a", "a_autosave", "b", "a"], ["b"]) == ["a"]


def client_with(*restored):
    manager = SimpleNamespace(list_modalities=lambda: list(restored))
    return SimpleNamespace(data_manager=manager)


def test_the_notice_follows_the_newest_turn_end_in_the_record(tmp_path):
    workspace = tmp_path / mcp_record.WORKSPACE_DIR
    workspace.mkdir()
    (tmp_path / "result.json").write_text(
        json.dumps({"modalities": ["pbmc", "removed_on_purpose", "gone"]})
    )
    mcp_turns.record(workspace, 1, "completed", ["pbmc", "removed_on_purpose"])
    mcp_turns.record(workspace, 2, "cancelled", ["pbmc", "made_by_the_stopped_turn"])
    assert mcp_stdio._not_restored(client_with("pbmc"), tmp_path) == (
        "made_by_the_stopped_turn",
    )
    assert (
        mcp_stdio._not_restored(
            client_with("pbmc", "made_by_the_stopped_turn_autosave"), tmp_path
        )
        == ()
    )


def test_a_folder_of_an_older_build_falls_back_to_the_result_files(tmp_path):
    (tmp_path / mcp_record.WORKSPACE_DIR).mkdir()
    (tmp_path / "result.json").write_text(json.dumps({"modalities": ["pbmc", "gone"]}))
    assert mcp_stdio._not_restored(client_with("pbmc"), tmp_path) == ("gone",)


def test_a_stopped_turns_modalities_are_named_after_a_restart(earlier):
    out, _first_id, opened, script = earlier
    workspace = out / mcp_record.WORKSPACE_DIR
    # The process ended while a cancelled turn's modality was in memory only.
    mcp_turns.record(workspace, 2, "cancelled", ["pbmc", "made_then_stopped"])
    assert "made_then_stopped" not in (out / "result.json").read_text()
    second = McpServer(ThreadRunner())
    done = wait_terminal(second, go_on(second, output_dir=str(out))["run_id"])
    (warning,) = [w for w in done["warnings"] if w["code"] == "modalities_not_restored"]
    assert "made_then_stopped" in warning["message"]
    (prompt,) = opened[1]["client"].prompts
    assert json.dumps(["made_then_stopped"]) in notice_lines(prompt)[0]
