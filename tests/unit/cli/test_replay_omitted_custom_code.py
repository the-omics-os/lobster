"""A notebook that leaves out custom code Lobster did not write is never run.

A session an earlier version wrote may hold custom code in a record of a form
Lobster did not write. Its notebook shows a "Not shown" cell in place of the
code, so what the code did is not in the notebook: the replay refuses it before
anything runs. With the cell's mark removed the label still refuses it; with the
label gone too, the export record still says the session was migrated, and a
migrated record is never a MATCH.
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
    _migrate,
    _notebook,
    _save,
    _verify,
    needs_kernel,
)

pytestmark = pytest.mark.unit

OMITS = (
    "This notebook omits custom code Lobster did not write, so a replay cannot "
    "check it; it was not run."
)
LABEL = (
    "Not shown: custom code from a {source} whose record has a form Lobster did "
    "not write"
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


def _with_omitted_cell(notebook, mark=True, label=None, migrated=True):
    """The notebook with the exporter's "Not shown" cell before its code."""
    if migrated:
        _migrate(notebook)
    nb = nbformat.read(notebook, as_version=4)
    text = label or LABEL.format(source="Lobster 1.2.2 session")
    cell = nbformat.v4.new_markdown_cell(f"**{text}**\n")
    if mark:
        cell.metadata["lobster"] = {
            exporter.OMITTED_CUSTOM_CODE_KEY: {
                "step": 1,
                "reason": "unrecognised_record",
            }
        }
    nb.cells.insert(len(nb.cells) - 1, cell)
    nbformat.write(nb, notebook)
    return notebook


def _refused(ws, notebook):
    result = _verify(ws, COPY, notebook=notebook)
    out = json.loads(result.stdout)
    return result.exit_code, out


def test_the_marked_cell_refuses_the_notebook_before_it_runs(ws):
    nb = _with_omitted_cell(_notebook(ws / "notebooks" / "nb.ipynb", COPY))
    code, out = _refused(ws, nb)
    assert code == 4 and out["verdict"] == "REFUSED", out
    assert out["report"]["refusal"]["reason"] == OMITS
    assert not list(
        (ws / replay_commands.REPLAYS_DIR).glob("*/executed.ipynb")
    ), "the notebook ran"


def test_the_mark_refuses_it_with_its_label_changed(ws):
    nb = _with_omitted_cell(
        _notebook(ws / "notebooks" / "nb.ipynb", COPY),
        label="Step 2 needs no code",
        migrated=False,
    )
    code, out = _refused(ws, nb)
    assert code == 4 and out["report"]["refusal"]["reason"] == OMITS, out


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize(
    "source", ["Lobster 1.2.2 session", "Lobster session of unknown version"]
)
def test_the_label_refuses_it_without_the_mark(ws, lang, source):
    set_lang(lang)
    try:
        label = h(LABEL).format(source=h(source))
    finally:
        set_lang("en")
    nb = _with_omitted_cell(
        _notebook(ws / "notebooks" / "nb.ipynb", COPY),
        mark=False,
        label=label,
        migrated=False,
    )
    code, out = _refused(ws, nb)
    assert code == 4 and out["verdict"] == "REFUSED", out
    assert out["report"]["refusal"]["reason"] == OMITS


def test_another_markdown_cell_is_not_the_label(ws):
    other = (
        "**Not shown here: the custom code is in the appendix**\n",
        "Not shown: custom code from a Lobster 1.2.2 session whose record has a "
        "form Lobster did not write\n",  # not the bold cell the exporter writes
    )
    for text in other:
        nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY)
        cells = nbformat.read(nb, as_version=4)
        cells.cells.insert(1, nbformat.v4.new_markdown_cell(text))
        nbformat.write(cells, nb)
        assert replay_commands._omits_custom_code(nb) is False, text


def test_the_export_record_naming_an_omitted_step_refuses_it(ws):
    nb = _migrate(_notebook(ws / "notebooks" / "nb.ipynb", COPY))
    data = json.loads(nb.read_text())
    for cell in data["cells"]:
        source = "".join(cell["source"])
        if "data_integrity_manifest" in source:
            cell["source"] = source.replace(
                '"migrated": {', '"migrated": {"omitted_custom_code": [2], ', 1
            )
            assert cell["source"] != source
    nb.write_text(json.dumps(data))
    code, out = _refused(ws, nb)
    assert code == 4 and out["report"]["refusal"]["reason"] == OMITS, out


@needs_kernel
def test_without_mark_and_label_a_migrated_record_still_never_matches(ws):
    """The fail-closed floor: the export record of a migrated session is never a
    MATCH, whatever the notebook's cells say."""
    nb = _migrate(_notebook(ws / "notebooks" / "nb.ipynb", COPY))
    result = _verify(ws, COPY, notebook=nb)
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["verdict"] == "NOT_COMPARED"


def test_the_exporter_marks_the_cell_it_writes(tmp_path):
    """The cell the exporter writes for a record of another form carries the
    mark, and the replay refuses that notebook."""
    from lobster.core.provenance.analysis_ir import AnalysisStep
    from lobster.core.runtime.data_manager import DataManagerV2

    ws_path = tmp_path / "ws"
    folder = ws_path / ".lobster" / "sessions" / "s1"
    dm = DataManagerV2(workspace_path=ws_path, session_dir=folder)
    dm.provenance.create_activity(
        "execute_custom_code",
        "a",
        description="custom",
        ir=AnalysisStep(
            operation="custom_code_execution",
            tool_name="execute_custom_code",
            description="custom",
            library="custom",
            code_template="WORKSPACE = Path('{{ workspace_path }}')\nprint(1)",
            imports=[],
            parameters={"workspace_path": "/old/ws", "code": "print(1)"},
            parameter_schema={},
            exportable=True,
            replayable=True,
        ),
    )
    (folder / "metadata.json").write_text(
        json.dumps({"software_versions": {"lobster": "1.2.2"}})
    )
    dm = DataManagerV2(workspace_path=ws_path, session_dir=folder)
    path = dm.notebook_exporter.export("nb", filter_strategy="successful")
    marked = [
        c
        for c in nbformat.read(path, as_version=4).cells
        if (c.metadata.get("lobster") or {}).get(exporter.OMITTED_CUSTOM_CODE_KEY)
    ]
    assert len(marked) == 1 and marked[0].cell_type == "markdown", marked
    assert replay_commands._omits_custom_code(path) is True


def test_the_reason_has_its_chinese_line():
    from pathlib import Path

    root = Path(replay_commands.__file__).resolve().parents[4]
    for catalog in (
        root / "lobster" / "i18n" / "phrases.zh.json",
        root / "lobster-tui" / "internal" / "i18n" / "catalogs" / "phrases.zh.json",
    ):
        assert OMITS in json.loads(catalog.read_text())
