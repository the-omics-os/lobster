"""get_workspace_files_metadata lists only files inside the workspace."""

import os

import pytest

from lobster.utils.file_analyzer import get_workspace_files_metadata


@pytest.fixture
def layout(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "inside.csv").write_text("a,b\n1,2\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not yours\n")
    return workspace, outside


@pytest.mark.parametrize("directory", ["../outside", "data/../../outside"])
def test_a_directory_outside_the_workspace_lists_nothing(layout, directory):
    workspace, _ = layout
    assert get_workspace_files_metadata(workspace, directory) == []


def test_an_absolute_directory_outside_the_workspace_lists_nothing(layout):
    workspace, outside = layout
    assert get_workspace_files_metadata(workspace, str(outside)) == []


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_a_link_out_of_the_workspace_is_not_followed(layout):
    workspace, outside = layout
    (workspace / "data" / "link").symlink_to(outside, target_is_directory=True)
    (workspace / "data" / "file_link.txt").symlink_to(outside / "secret.txt")
    names = {m["relative_path"] for m in get_workspace_files_metadata(workspace)}
    assert names == {os.path.join("data", "inside.csv")}


def test_a_directory_inside_the_workspace_is_listed(layout):
    workspace, _ = layout
    names = [
        m["relative_path"] for m in get_workspace_files_metadata(workspace, "data")
    ]
    assert names == [os.path.join("data", "inside.csv")]
