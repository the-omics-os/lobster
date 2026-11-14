"""What a completed MCP run reports: open question, files, response style."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from lobster.cli_internal import mcp_contract, mcp_stdio
from lobster.cli_internal.mcp_stdio import McpServer, RunRequest, ThreadRunner
from lobster.core.figures.errors import StorageError
from lobster.core.runs import new_run_id
from tests.unit.mcp.test_run import (  # noqa: F401  (fixtures)
    COMPLETE,
    Act,
    FakeDataManager,
    Hold,
    call,
    data,
    engine,
    run_args,
    wait_terminal,
)


def run_engine(tmp_path, data):
    request = RunRequest(
        new_run_id(), str(data), str(tmp_path / "out"), "What is in this file?"
    )
    return mcp_stdio.run_engine(request, threading.Event())


def complete(text):
    return {**COMPLETE, "response": text}


# open_question ---------------------------------------------------------------


@pytest.mark.parametrize(
    "summary, expected",
    [
        ("QC kept 2638 of 2700 cells.\n\nWant me to cluster them next?", True),
        ("Done.\n\n**Shall I continue?**", True),
        ("Done.\n\n_Shall I continue?_  \n", True),
        ('Done.\n\n"Is that enough?"', True),
        ("Done.\n\nIs that enough? ``", True),
        ("Done.\n\nWhat now?\n\n\n", True),
        ("Notes first.\nDo you want X?", True),
        ("Is this OK? Yes, it is.", False),
        ("Is that all?\n\nPossible next steps: clustering.", False),
        ("Why?\n\nBecause the cells were dead.", False),
        ("QC kept 2638 of 2700 cells.", False),
        ("", False),
        ("   \n\n ", False),
        ("?", True),
    ],
)
def test_open_question_is_the_last_paragraph_ending_in_a_question_mark(
    summary, expected
):
    assert mcp_stdio._open_question(summary) is expected


def test_completed_status_carries_open_question_only_when_true(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [complete("QC done.\n\nShould I cluster the cells next?")]
    assert run_engine(tmp_path, data)["open_question"] is True
    script["events"] = [complete("QC done.")]
    assert "open_question" not in run_engine(tmp_path, data)


def test_open_question_is_judged_on_the_whole_summary_not_the_cut_one(
    engine, tmp_path, data
):
    _opened, script = engine
    long_text = "x" * 9000 + "\n\nShall I go on?"
    script["events"] = [complete(long_text)]
    status = mcp_stdio._bounded(run_engine(tmp_path, data))
    assert status["open_question"] is True
    assert status["summary"].endswith(mcp_stdio.SUMMARY_CUT)


# files -------------------------------------------------------------------------


def written_file(tmp_path, name, size=10, category="data"):
    """A file in the session folder of ``run_engine`` and its engine entry."""
    path = tmp_path / "out" / ".lobster_workspace" / category / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return {
        "path": str(path),
        "relative": f"{category}/{name}",
        "category": category,
        "size_bytes": size,
        "modified": "2026-11-13T10:00:00+00:00",
    }


@pytest.fixture
def written(monkeypatch):
    """FakeDataManager gets files_written_since; returns the recorded cutoffs."""
    calls = {"since": [], "files": [], "error": None}

    def files_written_since(self, since=None, **kwargs):
        calls["since"].append(since)
        if calls["error"]:
            raise calls["error"]
        return calls["files"]

    monkeypatch.setattr(
        FakeDataManager, "files_written_since", files_written_since, raising=False
    )
    return calls


def test_completed_status_lists_the_files_in_output_dir(
    engine, written, tmp_path, data
):
    _opened, script = engine
    written["files"] = [
        written_file(tmp_path, "a.h5ad", 11),
        written_file(tmp_path, "qc.csv", 12, "exports"),  # stays in the session
    ]
    script["events"] = [COMPLETE]
    status = run_engine(tmp_path, data)
    out = tmp_path / "out"
    assert status["files"] == [
        {
            "path": str(out / "result.json"),
            "category": "result",
            "size_bytes": (out / "result.json").stat().st_size,
        },
        {
            "path": str(out / "lobster_run.ipynb"),
            "category": "notebook",
            "size_bytes": 2,
        },
        {"path": str(out / "a.h5ad"), "category": "data", "size_bytes": 11},
    ]
    assert "files_truncated" not in status
    assert (
        Path(status["result_path"]).exists() and Path(status["notebook_path"]).exists()
    )
    # result.json lists every file the engine reported; the status does not repeat it.
    on_disk = json.loads(Path(status["result_path"]).read_text())
    assert {f["relative"] for f in on_disk["files_written"]} == {
        "a.h5ad",
        ".lobster_workspace/exports/qc.csv",
        "lobster_run.ipynb",
    }
    assert "files_written" not in status["result"]
    assert written["since"][0].tzinfo is not None


def test_no_saved_file_still_lists_the_result_and_the_notebook(
    engine, written, tmp_path, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]
    status = run_engine(tmp_path, data)
    assert [f["category"] for f in status["files"]] == ["result", "notebook"]
    assert "no_data_file_saved" in [w["code"] for w in status["warnings"]]


def test_a_saved_copy_of_the_loaded_input_is_not_published(
    engine, written, tmp_path, data
):
    """The raw modality is the input as loaded: its file is not a result."""
    _opened, script = engine
    written["files"] = [
        written_file(tmp_path, "raw.h5ad", 11),
        written_file(tmp_path, "pbmc.h5ad", 12),
    ]
    script["events"] = [Act("load_dataset", name="raw", source=str(data)), COMPLETE]
    status = run_engine(tmp_path, data)
    out = tmp_path / "out"
    assert sorted(p.name for p in out.glob("*.h5ad")) == ["pbmc.h5ad"]
    assert [f["category"] for f in status["files"]] == ["result", "notebook", "data"]
    assert "no_data_file_saved" not in [w["code"] for w in status["warnings"]]
    # result.json still lists what the engine wrote, where it is.
    record = json.loads(Path(status["result_path"]).read_text())
    assert ".lobster_workspace/data/raw.h5ad" in {
        f["relative"] for f in record["files_written"]
    }


def test_only_a_copy_of_the_input_saved_means_no_data_file(
    engine, written, tmp_path, data
):
    _opened, script = engine
    written["files"] = [written_file(tmp_path, "raw.h5ad", 11)]
    script["events"] = [Act("load_dataset", name="raw", source=str(data)), COMPLETE]
    status = run_engine(tmp_path, data)
    assert list((tmp_path / "out").glob("*.h5ad")) == []
    assert "no_data_file_saved" in [w["code"] for w in status["warnings"]]


@pytest.mark.parametrize("count, truncated", [(48, False), (49, True)])
def test_files_are_cut_at_fifty(engine, written, tmp_path, data, count, truncated):
    """The result and the notebook come first, then the saved files."""
    _opened, script = engine
    written["files"] = [written_file(tmp_path, f"f{i}.h5ad") for i in range(count)]
    script["events"] = [COMPLETE]
    status = run_engine(tmp_path, data)
    assert len(status["files"]) == min(count + 2, 50)
    assert status.get("files_truncated", False) is truncated
    if truncated:
        assert status["files_total"] == count + 2
    on_disk = json.loads(Path(status["result_path"]).read_text())
    assert len(on_disk["files_written"]) == count + 1  # and the notebook


def test_a_storage_error_leaves_the_run_completed_without_a_data_list(
    engine, written, tmp_path, data
):
    _opened, script = engine
    written["error"] = StorageError("disk full at /secret/place")
    script["events"] = [COMPLETE]
    status = run_engine(tmp_path, data)
    assert status["status"] == "completed"
    assert [f["category"] for f in status["files"]] == ["result", "notebook"]
    result = status["result"]
    assert "files_written" not in result  # left out, as it is when it is a list
    error = result["files_written_error"]
    assert error and "\n" not in error and "/secret" not in error
    on_disk = json.loads(Path(status["result_path"]).read_text())
    assert on_disk["files_written"] is None
    codes = [w["code"] for w in status["warnings"]]
    assert "metrics_unavailable" in codes and "no_data_file_saved" not in codes


def test_without_a_file_source_there_is_no_engine_file_list(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [COMPLETE]
    status = run_engine(tmp_path, data)
    assert [f["category"] for f in status["files"]] == ["result", "notebook"]
    assert not {"files_written", "files_written_error"} & set(status["result"])
    on_disk = json.loads(Path(status["result_path"]).read_text())
    assert "files_written" not in on_disk


# response style ----------------------------------------------------------------


@pytest.mark.parametrize("user_value", [None, "chat", "run", "bogus"])
def test_an_mcp_run_asks_for_the_run_style_whatever_the_environment_says(
    engine, monkeypatch, tmp_path, data, user_value
):
    from lobster.cli_internal.commands.heavy import session_infra
    from lobster.config.supervisor_config import SupervisorConfig

    _opened, script = engine
    script["events"] = [COMPLETE]
    monkeypatch.setenv("LOBSTER_RESPONSE_STYLE", "chat")  # restores on teardown
    if user_value is None:
        monkeypatch.delenv("LOBSTER_RESPONSE_STYLE")
    else:
        monkeypatch.setenv("LOBSTER_RESPONSE_STYLE", user_value)
    seen = []
    fake_init = session_infra.init_client_or_raise_startup_diagnostic

    def spy(**kwargs):  # what the supervisor prompt is built from, at client start
        seen.append(SupervisorConfig.from_env().response_style)
        return fake_init(**kwargs)

    monkeypatch.setattr(session_infra, "init_client_or_raise_startup_diagnostic", spy)
    assert run_engine(tmp_path, data)["status"] == "completed"
    assert seen == ["run"]
    assert os.environ.get("LOBSTER_RESPONSE_STYLE") == user_value  # put back


# structured result -------------------------------------------------------------


def test_completed_status_carries_metrics_modalities_and_warnings(
    engine, written, tmp_path, data
):
    _opened, script = engine
    written["files"] = [written_file(tmp_path, "pbmc.h5ad")]
    script["events"] = [COMPLETE]
    status = run_engine(tmp_path, data)

    assert status["metrics"] == {
        "cells_before": 3,
        "cells_after": 3,
        "genes_before": 2,
        "genes_after": 2,
        "tokens": {
            "total": None,
            "cost_usd": None,
            "cost_complete": None,
            "usage_complete": None,
            "scope": "turn",
        },
    }
    assert status["modality_details"] == [
        {
            "name": "pbmc",
            "stage": "raw",
            "parent": None,
            "n_obs": 3,
            "n_vars": 2,
            "file": str(tmp_path / "out" / "pbmc.h5ad"),
        }
    ]
    assert status["warnings"] == []
    # The engine's own result is left as it was, minus what travels elsewhere.
    assert status["result"]["has_data"] is True
    # The engine's names stay in the result; the objects are modality_details.
    assert status["result"]["modalities"] == ["pbmc"] and "steps" not in status


def test_result_json_has_the_request_the_steps_and_the_modality_detail(
    engine, written, tmp_path, data
):
    _opened, script = engine
    hold = Hold("assess_data_quality", "filter_and_normalize")
    hold.release.set()
    script["events"] = [hold, COMPLETE]
    status = run_engine(tmp_path, data)
    on_disk = json.loads(Path(status["result_path"]).read_text())
    assert on_disk["request"] == {
        "question": "What is in this file?",
        "input_path": str(data),
        "plan": False,  # run_engine directly: no lobster_run default
        "instruction": None,
        "channel": "mcp:unknown",
        "turn": 1,
    }
    assert [s["tool"] for s in on_disk["steps"]] == [
        "load_dataset",
        "assess_data_quality",
        "filter_and_normalize",
    ]
    assert on_disk["modalities"] == ["pbmc"]
    assert on_disk["modality_details"][0]["name"] == "pbmc"
    assert on_disk["metrics"] == status["metrics"]
    assert on_disk["warnings"] == status["warnings"]
    assert on_disk["run_id"] == status["result"]["run_id"]


def test_an_open_question_is_a_field_and_a_warning(engine, tmp_path, data):
    _opened, script = engine
    script["events"] = [complete("Done.\n\nShall I cluster?")]
    status = run_engine(tmp_path, data)
    assert status["open_question"] is True
    assert "open_question" in [w["code"] for w in status["warnings"]]
    script["events"] = [complete("Done.")]
    status = run_engine(tmp_path, data)
    assert "open_question" not in status
    assert "open_question" not in [w["code"] for w in status["warnings"]]


def test_an_unreadable_section_leaves_the_run_completed(
    engine, monkeypatch, tmp_path, data
):
    _opened, script = engine
    script["events"] = [COMPLETE]

    def broken(self):
        raise RuntimeError("spilled")

    monkeypatch.setattr(FakeDataManager, "list_modalities_with_lineage", broken)
    status = run_engine(tmp_path, data)
    assert status["status"] == "completed"
    assert "modality_details" not in status
    assert status["result"]["modalities"] == ["pbmc"]  # the engine's list stays
    assert [w["code"] for w in status["warnings"]] == ["metrics_unavailable"]
    assert status["warnings"][0]["message"] == "Could not read: modalities."
