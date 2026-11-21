"""A notebook that leaves out a step recorded with an earlier template is never run.

The exporter puts a "Not shown" cell in place of a step whose stored code is
not what this Lobster writes for its tool. What the step did is not in the
notebook, so the replay refuses it before anything runs, and says why. The
cell's mark, its label or the export record naming the step is each enough.
"""

import json

import nbformat
import pytest

from lobster.cli_internal.commands.light import replay_commands
from lobster.core.notebooks import exporter
from lobster.i18n import h, set_lang
from tests.unit.cli.test_replay_commands import (
    COPY,
    SESSION_SHIFT,
    _adata,
    _notebook,
    _save,
    _verify,
)

pytestmark = pytest.mark.unit

LEAVES_OUT = (
    "This notebook leaves out a step recorded with code this Lobster no longer "
    "writes, so a replay cannot check it; it was not run."
)
LABEL = (
    "Not shown: a {tool} step recorded with code this Lobster no longer writes "
    "(run the step again to include it)"
)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """The workspace of test_replay_commands: a saved file and an input."""
    for name in ("LOBSTER_LAMIN", "LOBSTER_LAMIN_INSTANCE", "LAMIN_CURRENT_INSTANCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "lobster.core.catalog.factory._global_config_path",
        lambda: tmp_path / "no-providers.json",
    )
    path = tmp_path / ".lobster_workspace"
    (path / "data").mkdir(parents=True)
    _save(_adata(shift=SESSION_SHIFT, record=True), path / "data" / "pbmc.h5ad")
    _save(_adata(), tmp_path / "input.h5ad")
    return path


def _with_cell(notebook, mark=True, label=None):
    nb = nbformat.read(notebook, as_version=4)
    text = label or LABEL.format(tool="concatenate_samples")
    cell = nbformat.v4.new_markdown_cell(f"**{text}**\n")
    if mark:
        cell.metadata["lobster"] = {
            exporter.OMITTED_CUSTOM_CODE_KEY: {"step": 2, "reason": "outdated_template"}
        }
    nb.cells.insert(len(nb.cells) - 1, cell)
    nbformat.write(nb, notebook)
    return notebook


def _refused(ws, notebook):
    result = _verify(ws, COPY, notebook=notebook)
    return result.exit_code, json.loads(result.stdout)


def _ran(ws):
    return list((ws / replay_commands.REPLAYS_DIR).glob("*/executed.ipynb"))


def test_the_marked_cell_refuses_the_notebook_and_says_why(ws):
    nb = _with_cell(_notebook(ws / "notebooks" / "nb.ipynb", COPY))
    code, out = _refused(ws, nb)
    assert code == 4 and out["verdict"] == "REFUSED", out
    assert out["report"]["refusal"]["reason"] == LEAVES_OUT
    assert not _ran(ws), "the notebook ran"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_label_refuses_it_without_the_mark(ws, lang):
    set_lang(lang)
    try:
        label = h(LABEL).format(tool="export_de_results")
    finally:
        set_lang("en")
    nb = _with_cell(_notebook(ws / "notebooks" / "nb.ipynb", COPY), False, label)
    code, out = _refused(ws, nb)
    assert code == 4 and out["report"]["refusal"]["reason"] == LEAVES_OUT, out
    assert not _ran(ws), "the notebook ran"


def test_the_export_record_naming_the_step_refuses_it(ws):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    data = json.loads(nb.read_text())
    for cell in data["cells"]:
        source = "".join(cell["source"])
        if "data_integrity_manifest" in source:
            cell["source"] = source.replace(
                '"data_integrity_manifest": {',
                '"data_integrity_manifest": {"outdated_steps": [2], ',
                1,
            )
            assert cell["source"] != source
    nb.write_text(json.dumps(data))
    code, out = _refused(ws, nb)
    assert code == 4 and out["report"]["refusal"]["reason"] == LEAVES_OUT, out
