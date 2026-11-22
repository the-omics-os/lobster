"""A replay names the custom code of an earlier session that it shows and does not run.

The fixture is a notebook exported by Lobster 1.3.0 (from PyPI) from a session
written by Lobster 1.2.2 (from PyPI): the session loads a bulk .h5ad and runs one
custom-code step that writes ``exports/mine.csv``. The exporter shows that code in
a markdown cell and never runs it, so the replay cannot list the file it wrote.
"""

import json
import shutil
from pathlib import Path

import anndata as ad
import nbformat
import numpy as np
import pandas as pd
import pytest

from lobster.cli_internal.commands.light import replay_commands
from lobster.i18n import h, use_lang
from tests.unit.cli import test_replay_commands as base

pytestmark = pytest.mark.unit

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "replay"
    / "migrated_1_2_2_custom_code.ipynb"
)
LABEL = (
    "Not executed: custom code from a {source} "
    "(it may use paths of the machine it ran on)"
)
NOTE = (
    "Cell {n}: custom code from an earlier session is shown, not run. Files it "
    "changed are compared without its changes, and files only it wrote are not "
    "listed."
)


def _copy(tmp_path, edit=None):
    nb = nbformat.read(FIXTURE, as_version=4)
    if edit is not None:
        edit(nb)
    path = tmp_path / "migrated.ipynb"
    nbformat.write(nb, path)
    return path


def test_the_fixture_shows_the_custom_code_and_runs_none_of_it():
    nb = nbformat.read(FIXTURE, as_version=4)
    shown = [c for c in nb.cells if LABEL.split("{")[0] in c.source]
    assert len(shown) == 1 and shown[0].cell_type == "markdown"
    assert "mine.csv" in shown[0].source
    assert not any("mine.csv" in c.source for c in nb.cells if c.cell_type == "code")


def test_the_cell_showing_custom_code_that_was_not_run_is_found(tmp_path):
    assert replay_commands._unrun_custom_cells(_copy(tmp_path)) == [7]


def test_the_cell_is_found_in_every_language_lobster_writes_it_in(tmp_path):
    with use_lang("zh"):
        zh = h(LABEL)
    assert zh != LABEL

    def to_zh(nb):
        cell = nb.cells[6]
        english = LABEL.format(source="Lobster 1.2.2 session")
        assert english in cell.source
        cell.source = cell.source.replace(
            english, zh.format(source="Lobster 1.2.2 会话")
        )

    assert replay_commands._unrun_custom_cells(_copy(tmp_path, to_zh)) == [7]


def test_a_notebook_without_the_cell_has_none(tmp_path):
    def drop(nb):
        del nb.cells[6]

    assert replay_commands._unrun_custom_cells(_copy(tmp_path, drop)) == []


def test_the_label_in_a_code_cell_is_not_the_cell(tmp_path):
    """Only the exporter's markdown cell is shown and not run; a code cell runs."""

    def as_code(nb):
        nb.cells[6] = nbformat.v4.new_code_cell(nb.cells[6].source)

    assert replay_commands._unrun_custom_cells(_copy(tmp_path, as_code)) == []


# --- replayed ------------------------------------------------------------------


def _input(path: Path) -> None:
    """The input the session loaded, written as the session's script wrote it."""
    rng = np.random.default_rng(0)
    ad.AnnData(
        rng.poisson(20, size=(6, 40)).astype(np.float32),
        obs=pd.DataFrame(
            {"condition": ["ctrl"] * 3 + ["treat"] * 3},
            index=[f"s{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(40)]),
    ).write_h5ad(path)


@base.needs_kernel
def test_a_replay_names_the_custom_code_it_did_not_run(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "notebooks").mkdir(parents=True)
    notebook = workspace / "notebooks" / "migrated.ipynb"
    shutil.copy(FIXTURE, notebook)
    source = tmp_path / "input.h5ad"
    _input(source)

    result = base.runner.invoke(
        base.app,
        ["--verify", str(notebook), "-w", str(workspace), "--input", str(source)]
        + ["--json"],
    )

    out = json.loads(result.stdout)
    assert (result.exit_code, out["verdict"]) == (3, "NOT_COMPARED"), out
    assert out["report"]["execution"] == "OK", out
    assert NOTE.format(n=7) in out["notes"], out["notes"]
    # the file the custom code wrote is nowhere in the run or its report
    assert not list(tmp_path.rglob("mine.csv"))
    assert "mine.csv" not in result.stdout
