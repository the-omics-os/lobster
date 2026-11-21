"""Gene ID conversion takes only the ID types it knows; its cell quotes them.

``source_type`` and ``target_type`` were not checked: an unknown type was passed
on as the mygene scope or field, and an f-string put it between double quotes in
the cell's template. A quote and a newline in it became code in the notebook,
and a ``{{ }}`` in it ran when Lobster rendered the cell. An unknown type is now
refused before anything is stored or logged, and the cell writes the scope and
field as Python literals.
"""

import ast
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.transcriptomics_expert import (
    transcriptomics_expert,
)
from lobster.core.notebooks.written_templates import outdated_tool, template_hash
from lobster.testing.provenance_replay import RecordingDataManager, factory_tools

MODULE = "lobster.agents.transcriptomics.transcriptomics_expert"
INJECTED = 'x")\n__import__("os").getcwd()  # {{ 7 * 191 }}'


class _MyGene:
    def querymany(self, ids, **_):
        return {"out": [{"query": i, "symbol": f"S{i}"} for i in ids]}


def _session(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "mygene", types.SimpleNamespace(MyGeneInfo=_MyGene)
    )
    genes = [f"ENSG0000000000{i}" for i in range(4)]
    dm = RecordingDataManager(
        {"bulk": ad.AnnData(np.ones((3, 4)), var=pd.DataFrame(index=genes))}
    )
    return dm, factory_tools(transcriptomics_expert, MODULE, dm)


@pytest.mark.parametrize(
    "arguments, reply",
    [
        (
            {"source_type": INJECTED},
            f"Unknown source_type {INJECTED!r}: use 'auto', 'ensembl', 'entrez' or "
            "'symbol'.",
        ),
        (
            {"target_type": INJECTED},
            f"Unknown target_type {INJECTED!r}: use 'symbol', 'ensembl' or 'entrez'.",
        ),
        (
            {"source_type": "Ensembl"},
            "Unknown source_type 'Ensembl': use 'auto', 'ensembl', 'entrez' or "
            "'symbol'.",
        ),
        (
            {"target_type": "auto"},
            "Unknown target_type 'auto': use 'symbol', 'ensembl' or 'entrez'.",
        ),
    ],
)
def test_an_unknown_id_type_is_refused_before_anything_is_stored(
    monkeypatch, arguments, reply
):
    dm, tools = _session(monkeypatch)
    before = dm.modalities["bulk"]

    answer = tools["convert_gene_identifiers"].invoke(
        {"modality_name": "bulk", **arguments}
    )

    assert answer == reply, answer
    assert dm.calls == []
    assert dm.modalities["bulk"] is before


def _keywords(code):
    call = ast.parse(code).body[-1].value
    return {k.arg: ast.literal_eval(k.value) for k in call.keywords}


def test_the_cell_writes_the_scope_and_field_as_literals(monkeypatch):
    dm, tools = _session(monkeypatch)
    tools["convert_gene_identifiers"].invoke({"modality_name": "bulk"})
    ir = dm.last_ir("convert_gene_identifiers")

    assert _keywords(ir.render()) == {
        "scopes": "ensembl.gene",
        "fields": "symbol",
        "species": "human",
    }
    # The scope and field are template values, not text baked into the template.
    hostile = ir.render(scope=INJECTED, field=INJECTED)
    assert _keywords(hostile)["scopes"] == INJECTED
    assert _keywords(hostile)["fields"] == INJECTED
    assert "1337" not in hostile


@pytest.mark.parametrize(
    "arguments",
    [{}, {"source_type": "ensembl", "target_type": "entrez"}],
)
def test_the_template_is_the_one_the_exporter_knows(monkeypatch, arguments):
    dm, tools = _session(monkeypatch)
    tools["convert_gene_identifiers"].invoke({"modality_name": "bulk", **arguments})
    ir = dm.last_ir("convert_gene_identifiers")

    assert outdated_tool(ir) is None, (ir.operation, template_hash(ir.code_template))
