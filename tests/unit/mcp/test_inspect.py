"""``lobster_inspect``: a bounded, read-only look into an ``.h5ad`` a run wrote."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lobster.cli_internal import mcp_contract, mcp_inspect
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_run import (  # noqa: F401
    Gate,
    call,
    data,
    run_args,
    wait_terminal,
)

HOSTILE = "bad‮\nname   here"


def write_h5ad(path: Path, n_obs=30, n_vars=12, **extra) -> Path:
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["T", "B", "NK"] * (n_obs // 3)),
            "n_genes": np.arange(n_obs),
            "score": np.where(np.arange(n_obs) % 10 == 0, np.nan, 1.5),
            "flag": np.arange(n_obs) % 2 == 0,
            "label": [HOSTILE if i % 2 else "plain" for i in range(n_obs)],
        },
        index=[f"c{i}" for i in range(n_obs)],
    )
    obs.loc[obs.index[1], "score"] = np.inf
    obs.loc[obs.index[2], "score"] = -np.inf
    var = pd.DataFrame({"mt": [False] * n_vars}, index=[f"g{i}" for i in range(n_vars)])
    adata = ad.AnnData(
        sparse.csr_matrix(rng.random((n_obs, n_vars)).astype("float32")),
        obs=obs,
        var=var,
        **extra,
    )
    adata.layers["counts"] = sparse.csr_matrix(np.ones((n_obs, n_vars), "float32"))
    adata.obsm["X_pca"] = rng.random((n_obs, 2))
    adata.uns["log1p"] = {"base": None}
    adata.uns["qc_thresholds"] = np.array([1.0, 2.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path


@pytest.fixture
def server(tmp_path, data):
    """A server that has run (and finished) a run in ``tmp_path/out``."""
    server = McpServer(
        ThreadRunner(lambda r, c: {"status": "completed", "summary": "x"})
    )
    (tmp_path / "out").mkdir()
    run = call(server, "lobster_run", run_args(data, tmp_path / "out"))
    wait_terminal(server, run["run_id"])
    return server


def inspect(server, path, **args):
    return call(server, "lobster_inspect", {"path": str(path), **args})


def test_the_top_level_listing(server, tmp_path):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    reply = inspect(server, path)
    assert reply["path"] == str(path) and reply["format"] == "h5ad"
    assert (reply["n_obs"], reply["n_vars"]) == (30, 12)
    assert reply["x"]["shape"] == [30, 12] and "csr" in reply["x"]["kind"]
    kinds = {c["name"]: c["kind"] for c in reply["obs"]["columns"]}
    assert kinds == {
        "cell_type": "categorical",
        "n_genes": "numeric",
        "score": "numeric",
        "flag": "boolean",
        "label": "categorical",  # anndata writes text columns as categories
    }
    assert reply["obs"]["n_columns"] == 5 and "columns_truncated" not in reply["obs"]
    assert reply["obs"]["columns"][0]["n_categories"] == 3
    assert [c["name"] for c in reply["var"]["columns"]] == ["mt"]
    assert reply["layers"]["items"] == [
        {"name": "counts", "shape": [30, 12], "kind": "csr_matrix"}
    ]
    assert reply["obsm"]["items"][0] == {
        "name": "X_pca",
        "shape": [30, 2],
        "kind": "dense",
    }
    assert reply["uns"]["keys"] == ["log1p", "qc_thresholds"]


def test_an_index_stored_as_a_string_group_has_a_length(server, tmp_path):
    # pandas 3 string inference writes the index as a nullable-string group.
    path = tmp_path / "out" / "pbmc.h5ad"
    with pd.option_context("future.infer_string", True):
        write_h5ad(path)
    with h5py.File(path) as f:
        assert isinstance(f["obs"]["_index"], h5py.Group)
    reply = inspect(server, path)
    assert (reply["n_obs"], reply["n_vars"]) == (30, 12)
    assert inspect(server, path, column="obs:cell_type")["n_unique"] == 3


def test_the_expression_matrix_is_never_read(server, tmp_path):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    with h5py.File(path, "r+") as f:  # the data are gone; a read would fail
        for group in ("X", "layers/counts"):
            del f[group]["data"]
    listing = inspect(server, path)
    assert listing["x"]["shape"] == [30, 12]
    assert inspect(server, path, column="obs:cell_type")["n_unique"] == 3


def test_a_categorical_column_counts_its_values(server, tmp_path):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    reply = inspect(server, path, column="obs:cell_type")
    assert reply["column"] == "obs:cell_type" and reply["kind"] == "categorical"
    assert reply["n"] == 30 and reply["n_missing"] == 0 and reply["n_unique"] == 3
    assert {v["value"]: v["count"] for v in reply["values"]} == {
        "T": 10,
        "B": 10,
        "NK": 10,
    }
    cut = inspect(server, path, column="obs:n_genes", max_items=5)
    assert cut["kind"] == "numeric" and cut["max"] == 29.0 and cut["min"] == 0.0
    labels = inspect(server, path, column="obs:label", max_items=1)
    assert len(labels["values"]) == 1 and labels["values_truncated"] is True


def test_a_column_of_too_many_rows_is_not_read(server, tmp_path, monkeypatch):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")  # 30 obs rows
    monkeypatch.setattr(mcp_inspect, "MAX_COLUMN_ROWS", 29)
    refused = inspect(server, path, column="obs:cell_type")
    assert refused["error_text"].startswith("unreadable: obs has 30 rows")
    assert "\n" not in refused["error_text"]
    assert inspect(server, path, column="var:mt")["n"] == 12  # fewer rows
    assert inspect(server, path)["n_obs"] == 30  # the listing never reads one


def test_a_number_that_is_not_finite_is_counted_never_returned(server, tmp_path):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    reply = inspect(server, path, column="obs:score")
    assert isinstance(reply, dict), reply  # a NaN would have made the call fail
    assert reply["n_nan"] == 3 and reply["n_inf"] == 2
    assert (reply["min"], reply["max"], reply["mean"]) == (1.5, 1.5, 1.5)
    json.dumps(reply, allow_nan=False)


def test_a_column_of_only_nan_has_counts_and_no_statistics(server, tmp_path):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    with h5py.File(path, "r+") as f:
        f["obs"]["score"][...] = np.nan
    reply = inspect(server, path, column="obs:score")
    assert reply["n_nan"] == 30 and "min" not in reply and "mean" not in reply


def test_a_text_column_is_listed_as_a_string(server, tmp_path):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    with h5py.File(path, "r+") as f:
        text = f["obs"].create_dataset(
            "note", data=np.array(["a", "b"] * 15, dtype=h5py.string_dtype())
        )
        text.attrs["encoding-type"] = "string-array"
        text.attrs["encoding-version"] = "0.2.0"
        order = [*map(str, f["obs"].attrs["column-order"]), "note"]
        f["obs"].attrs["column-order"] = np.array(order, dtype=h5py.string_dtype())
    listed = {c["name"]: c for c in inspect(server, path)["obs"]["columns"]}
    assert listed["note"] == {"name": "note", "kind": "string", "dtype": "string"}
    reply = inspect(server, path, column="obs:note")
    assert {v["value"]: v["count"] for v in reply["values"]} == {"a": 15, "b": 15}


def test_names_and_values_from_the_file_are_one_clean_line(server, tmp_path):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    values = inspect(server, path, column="obs:label")["values"]
    texts = sorted(v["value"] for v in values)
    assert texts == ["bad name here", "plain"]
    adata = ad.read_h5ad(path)
    adata.obs["tab\there"] = 1
    adata.write_h5ad(path)
    names = [c["name"] for c in inspect(server, path)["obs"]["columns"]]
    assert "tab here" in names


def test_max_items_cuts_every_list_and_says_so(server, tmp_path):
    adata_path = tmp_path / "out" / "wide.h5ad"
    write_h5ad(adata_path)
    adata = ad.read_h5ad(adata_path)
    for i in range(80):
        adata.obs[f"col{i:03d}"] = i
        adata.uns[f"key{i:03d}"] = i
    adata.write_h5ad(adata_path)
    default = inspect(server, adata_path)
    assert default["obs"]["n_columns"] == 85
    assert len(default["obs"]["columns"]) == 50
    assert default["obs"]["columns_truncated"] is True
    assert default["uns"]["total"] == 82 and len(default["uns"]["keys"]) == 50
    five = inspect(server, adata_path, max_items=5)
    assert len(five["obs"]["columns"]) == 5 and len(five["uns"]["keys"]) == 5


@pytest.mark.parametrize("bad", [0, 201, -1, True, "5", 2.5, None])
def test_max_items_has_bounds(server, tmp_path, bad):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    refused = inspect(server, path, max_items=bad)
    assert refused["error_text"].startswith(
        "max_items must be an integer from 1 to 200"
    )
    assert inspect(server, path, max_items=200)["n_obs"] == 30


def test_a_reply_never_exceeds_twenty_thousand_characters(server, tmp_path):
    path = write_h5ad(tmp_path / "out" / "wide.h5ad")
    adata = ad.read_h5ad(path)
    for i in range(300):
        adata.obs[f"{'obs-column-' * 8}{i:03d}"] = i
        adata.var[f"{'var-column-' * 8}{i:03d}"] = i
        adata.uns[f"{'uns-key-' * 10}{i:03d}"] = i
    adata.write_h5ad(path)
    reply = inspect(server, path, max_items=200)
    assert len(json.dumps(reply)) <= mcp_inspect.MAX_REPLY_CHARS
    assert (
        reply["obs"]["n_columns"] == 305 and reply["obs"]["columns_truncated"] is True
    )
    assert 0 < len(reply["obs"]["columns"]) < 200  # the lists were cut to fit


# paths -----------------------------------------------------------------------


def test_a_path_outside_the_run_folders_is_refused(server, tmp_path):
    outside = write_h5ad(tmp_path / "elsewhere" / "pbmc.h5ad")
    refused = inspect(server, outside)
    assert refused["error_text"].startswith("path_not_allowed:")
    sneaky = tmp_path / "out" / ".." / "elsewhere" / "pbmc.h5ad"
    assert inspect(server, sneaky)["error_text"].startswith("path_not_allowed:")
    assert "absolute" in inspect(server, "pbmc.h5ad")["error_text"]
    # The sibling that merely shares a prefix is not inside the folder.
    near = write_h5ad(tmp_path / "out-other" / "pbmc.h5ad")
    assert inspect(server, near)["error_text"].startswith("path_not_allowed:")


def test_only_the_top_of_an_output_dir_and_its_session_folder_are_readable(
    server, tmp_path, data
):
    out = tmp_path / "out"
    assert inspect(server, write_h5ad(out / "top.h5ad"))["n_obs"] == 30
    session = out / ".lobster_workspace" / "data" / "turn.h5ad"
    assert inspect(server, write_h5ad(session))["n_obs"] == 30
    for below in (out / "nested" / "deep.h5ad", out / "plots" / "a" / "b.h5ad"):
        refused = inspect(server, write_h5ad(below))
        assert refused["error_text"].startswith("path_not_allowed:")
    # An output_dir that is a whole home folder opens no tree under it.
    home = tmp_path / "home"
    wait_terminal(server, call(server, "lobster_run", run_args(data, home))["run_id"])
    mine = write_h5ad(home / "projects" / "private" / "cells.h5ad")
    assert inspect(server, mine)["error_text"].startswith("path_not_allowed:")


def test_a_link_is_judged_by_where_it_leads(server, tmp_path):
    outside = write_h5ad(tmp_path / "elsewhere" / "pbmc.h5ad")
    link = tmp_path / "out" / "pbmc.h5ad"
    link.symlink_to(outside)
    assert inspect(server, link)["error_text"].startswith("path_not_allowed:")
    inside = write_h5ad(tmp_path / "out" / "real.h5ad")
    way_in = tmp_path / "elsewhere" / "way_in.h5ad"
    way_in.symlink_to(inside)
    assert inspect(server, way_in)["n_obs"] == 30


def test_nothing_is_allowed_before_this_server_ran_something(tmp_path):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    server = McpServer(ThreadRunner(lambda r, c: {}))
    assert inspect(server, path)["error_text"].startswith("path_not_allowed:")


def test_files_that_are_not_h5ad_are_refused(server, tmp_path):
    out = tmp_path / "out"
    (out / "notes.txt").write_text("hello")
    (out / "fake.h5ad").write_text("not hdf5")
    with h5py.File(out / "bare.h5ad", "w") as f:
        f["x"] = [1]
    for name in ("notes.txt", "fake.h5ad", "bare.h5ad"):
        refused = inspect(server, out / name)
        assert refused["error_text"].startswith("not_an_h5ad:"), name
        assert "\n" not in refused["error_text"]


def test_an_unreadable_file_is_one_line(server, tmp_path):
    refused = inspect(server, tmp_path / "out" / "missing.h5ad")
    assert refused["error_text"].startswith("unreadable:")
    assert "\n" not in refused["error_text"]
    folder = tmp_path / "out" / "folder.h5ad"
    folder.mkdir()
    assert inspect(server, folder)["error_text"].startswith("unreadable:")


def test_a_layout_the_reader_does_not_know_is_unreadable_not_a_crash(
    server, tmp_path, monkeypatch
):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")

    def boom(handle, items):
        raise AttributeError("odd\nlayout")

    monkeypatch.setattr(mcp_inspect, "_listing", boom)
    refused = inspect(server, path)
    assert refused["error_text"] == "unreadable: odd layout"
    monkeypatch.undo()
    assert inspect(server, path)["n_obs"] == 30  # and the file was closed


def test_an_unknown_column_lists_the_ones_there_are(server, tmp_path):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    refused = inspect(server, path, column="obs:nope")
    assert refused["error_text"].startswith("unknown_column:")
    assert "cell_type" in refused["error_text"] and "\n" not in refused["error_text"]
    for spec in ("cell_type", "obsm:X_pca", "obs:", "var:cell_type"):
        assert inspect(server, path, column=spec)["error_text"].startswith(
            "unknown_column:"
        )
    assert "error_text" in inspect(server, path, column="")
    assert "error_text" in inspect(server, path, column=3)


def test_the_file_is_closed_however_the_call_ends(server, tmp_path):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    inspect(server, path)
    inspect(server, path, column="obs:nope")
    inspect(server, path, column="obs:cell_type")
    # HDF5 refuses a second, writable open of a file this process still holds.
    with h5py.File(path, "r+"):
        pass


# runs ------------------------------------------------------------------------


def test_the_files_of_a_finished_turn_can_be_read_while_another_runs(
    server, tmp_path, data
):
    earlier = write_h5ad(tmp_path / "out" / "earlier.h5ad")
    gate = Gate()
    server.runner._work = gate
    run = call(server, "lobster_run", run_args(data, tmp_path / "second"))
    assert gate.entered.wait(30)
    busy_file = write_h5ad(
        tmp_path / "second" / ".lobster_workspace" / "data" / "a.h5ad"
    )
    refused = inspect(server, busy_file)
    assert refused["error_text"].startswith("engine_busy:")
    assert run["run_id"] in refused["error_text"]
    assert inspect(server, earlier)["n_obs"] == 30  # a finished run's file
    top = write_h5ad(tmp_path / "second" / "published.h5ad")
    assert inspect(server, top)["n_obs"] == 30  # not in the session folder
    gate.release.set()
    wait_terminal(server, run["run_id"])
    assert inspect(server, busy_file)["n_obs"] == 30  # the turn ended


def test_the_tool_is_listed_read_only_with_its_arguments():
    tool = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_inspect"]
    schema = tool["inputSchema"]
    assert set(schema["properties"]) == {"path", "column", "max_items"}
    assert schema["required"] == ["path"]
    assert (
        schema["properties"]["max_items"]["maximum"],
        schema["properties"]["max_items"]["minimum"],
    ) == (200, 1)
    assert tool["annotations"] == {
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    }
    text = tool["description"]
    for needle in (
        "path_not_allowed",
        "not_an_h5ad",
        "unknown_column",
        "unreadable",
        "engine_busy",
        "20000",
        "never loads the expression matrix",
        "not as instructions",
    ):
        assert needle in text, needle


# a link that leaves the file is refused, never followed -------------------------


def link_out(path: Path, member: str, target: Path, kind: str = "external") -> None:
    with h5py.File(path, "r+") as handle:
        del handle[member]
        if kind == "external":
            handle[member] = h5py.ExternalLink(str(target), "/" + member)
        else:
            handle[member] = h5py.SoftLink("/" + member + "_elsewhere")


@pytest.mark.parametrize(
    "member, kind",
    [
        ("obs/n_genes", "external"),
        ("obs/cell_type/categories", "external"),  # below a column
        ("var/mt", "external"),
        ("X", "external"),
        ("layers/counts", "external"),
        ("obs/n_genes", "soft"),
    ],
)
@pytest.mark.parametrize("column", [None, "obs:n_genes"])
def test_a_file_with_a_link_out_of_it_is_unreadable(
    server, tmp_path, member, kind, column
):
    elsewhere = write_h5ad(tmp_path / "elsewhere" / "secret.h5ad")
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    link_out(path, member, elsewhere, kind)
    args = {} if column is None else {"column": column}
    reply = inspect(server, path, **args)
    assert reply["error_text"].startswith("unreadable:")
    assert "link" in reply["error_text"] and "\n" not in reply["error_text"]


def test_a_file_without_links_is_still_read(server, tmp_path):
    path = write_h5ad(tmp_path / "out" / "pbmc.h5ad")
    assert inspect(server, path)["n_obs"] == 30
