"""How the files a turn read are told apart from the input: the same file by
another route is the input, a file of the same size and other bytes is not, and
the host's words only count when they name the path whole."""

from __future__ import annotations

import builtins
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lobster.cli_internal import mcp_contract, mcp_record


def judged(tmp_path, given, sources, instruction=None):
    activities = [
        {
            "type": "load_dataset",
            "parameters": {"name": f"m{n}", "source": str(s), "adapter": "x"},
        }
        for n, s in enumerate(sources)
    ]
    dm = SimpleNamespace(provenance=SimpleNamespace(activities=activities))
    turn = mcp_record.Turn(datetime.now(timezone.utc), 0, 2)
    workspace = tmp_path / "out" / mcp_record.WORKSPACE_DIR
    return mcp_record.judge_loads(dm, turn, str(given), workspace, instruction)


# the same file by another route ------------------------------------------------


def test_a_folder_input_whose_members_are_links_is_still_the_input(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "matrix.mtx").write_text("counts")
    given = tmp_path / "tenx"
    given.mkdir()
    (given / "matrix.mtx").symlink_to(elsewhere / "matrix.mtx")
    result = judged(tmp_path, given, [given / "matrix.mtx"])
    assert result.foreign == [] and result.input_loaded


def test_a_folder_named_in_another_letter_case_is_the_input(tmp_path):
    given = tmp_path / "tenx"
    given.mkdir()
    (given / "matrix.mtx").write_text("counts")
    if not (tmp_path / "TENX").exists():
        pytest.skip("a case-sensitive file system: TENX is another folder")
    result = judged(tmp_path, given, [tmp_path / "TENX" / "matrix.mtx"])
    assert result.foreign == [] and result.input_loaded


def test_a_hard_link_to_the_input_is_the_input_and_is_not_read(tmp_path, monkeypatch):
    given = tmp_path / "in.h5ad"
    given.write_bytes(b"x" * 4096)
    linked = tmp_path / "linked.h5ad"
    os.link(given, linked)
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(
        builtins,
        "open",
        lambda file, *a, **k: opened.append(str(file)) or real_open(file, *a, **k),
    )
    result = judged(tmp_path, given, [linked])
    monkeypatch.undo()
    assert result.foreign == [] and not [p for p in opened if p.endswith(".h5ad")]


def test_a_loader_that_reads_the_folder_of_the_input_file_reads_the_input(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    given = folder / "matrix.mtx"
    given.write_bytes(b"x")
    result = judged(tmp_path, given, [folder])
    assert result.foreign == [] and result.input_loaded


def test_a_folder_that_is_not_the_inputs_is_another_file(tmp_path):
    given = tmp_path / "data" / "matrix.mtx"
    given.parent.mkdir()
    given.write_bytes(b"x")
    (tmp_path / "other").mkdir()
    result = judged(tmp_path, given, [tmp_path / "other"])
    assert result.foreign == [str((tmp_path / "other").resolve())]


# the bytes ---------------------------------------------------------------------


def spied_open(monkeypatch):
    """Every path opened for reading, with the bytes read from it."""
    read = {}
    real_open = builtins.open

    class Counted:
        def __init__(self, handle, name):
            self.handle, self.name = handle, name

        def read(self, size=-1):
            data = self.handle.read(size)
            read[self.name] = read.get(self.name, 0) + len(data)
            return data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()

    def spy(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if "b" not in mode or not str(file).endswith(".h5ad"):
            return handle
        return Counted(handle, str(file))

    monkeypatch.setattr(builtins, "open", spy)
    return read


def test_files_of_different_sizes_are_not_read(tmp_path, monkeypatch):
    first, second = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    first.write_bytes(b"x" * 5000)
    second.write_bytes(b"x" * 5001)
    read = spied_open(monkeypatch)
    assert mcp_record._same_content(second, first) is False
    assert read == {}


def test_files_of_one_size_are_compared_in_big_pieces_and_stop_at_the_first_difference(
    tmp_path, monkeypatch
):
    size = 6 * 1024 * 1024
    first, second = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    first.write_bytes(b"a" + b"x" * (size - 1))
    second.write_bytes(b"b" + b"x" * (size - 1))
    read = spied_open(monkeypatch)
    assert mcp_record._same_content(second, first) is False
    assert sum(read.values()) <= 2 * mcp_record.COMPARE_CHUNK_BYTES
    assert max(read.values()) < size


def test_a_copy_is_read_to_its_end_and_is_the_same(tmp_path):
    first, second = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    first.write_bytes(os.urandom(3 * 1024 * 1024 + 7))
    second.write_bytes(first.read_bytes())
    assert mcp_record._same_content(second, first) is True
    body = first.read_bytes()
    second.write_bytes(body[:-1] + bytes([body[-1] ^ 1]))  # differs in the last byte
    assert mcp_record._same_content(second, first) is False


def test_a_missing_file_is_not_the_same(tmp_path):
    given = tmp_path / "a.h5ad"
    given.write_bytes(b"x")
    assert mcp_record._same_content(tmp_path / "gone.h5ad", given) is False


# the host's words --------------------------------------------------------------


@pytest.mark.parametrize(
    "instruction, asked",
    [
        ("compare with {p}", True),
        ("compare with {p}.", True),
        ("compare with '{p}', please", True),
        ('use "{p}" too', True),
        ("compare with ({p})", True),
        ("compare with {p}.bak", False),
        ("compare with {p}2", False),
        ("compare with /mnt{p}", False),
        ("compare with {p}/inner", False),
        ("compare with other.h5ad", False),
    ],
)
def test_the_instruction_names_a_file_only_by_its_whole_path(
    tmp_path, instruction, asked
):
    given = tmp_path / "in.h5ad"
    given.write_bytes(b"x")
    other = tmp_path / "other.h5ad"
    other.write_bytes(b"y")
    result = judged(tmp_path, given, [other], instruction.format(p=other.resolve()))
    assert (result.foreign == []) is asked


# what the texts claim ----------------------------------------------------------


def test_the_texts_say_the_check_covers_the_loading_tools_only():
    tools = {t["name"]: t["description"] for t in mcp_contract.list_tools()}
    for name in ("lobster_run", "lobster_continue"):
        text = tools[name]
        assert "Lobster's loading tools" in text
    sentence = (
        "Code the model writes can read other files, which this check does not see."
    )
    for name in ("lobster_run", "lobster_continue"):
        assert sentence in tools[name]
