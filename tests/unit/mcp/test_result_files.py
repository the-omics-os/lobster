"""One way to find the result files of a folder, used by everything that reads them."""

from __future__ import annotations

import inspect
import json

from lobster.cli_internal import mcp_record


def test_the_result_file_pattern_is_defined_once():
    source = inspect.getsource(mcp_record)
    assert source.count("\n_RESULT_FILE = ") == 1


def test_result_files_are_listed_by_turn_and_nothing_else_counts(tmp_path):
    for name in (
        "result.json",
        "result_2.json",
        "result_10.json",
        "result_final.json",
        "result.json.bak",
        "result_3.jsonl",
    ):
        (tmp_path / name).write_text(json.dumps({"modalities": [name]}))
    found = mcp_record.result_files(tmp_path)
    assert {n: p.name for n, p in found.items()} == {
        1: "result.json",
        2: "result_2.json",
        10: "result_10.json",
    }
    # the readers see the same files
    assert mcp_record.recorded_modalities(tmp_path) == [
        "result.json",
        "result_2.json",
        "result_10.json",
    ]


def test_the_readers_use_the_one_helper():
    for reader in (
        mcp_record.recorded_modalities,
        mcp_record.read_disk_session,
        mcp_record.output_dir_in_use,
    ):
        assert "result_files(" in inspect.getsource(reader), reader.__name__
