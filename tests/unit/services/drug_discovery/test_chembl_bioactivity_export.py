"""The exported ChEMBL bioactivity cell runs, with or without a target.

With a target the cell began with a stray quote, ``"params["target_chembl_id"]
= ...``: a SyntaxError, so the exported notebook could not run.
"""

import ast

import httpx
import pytest

from lobster.services.drug_discovery.chembl_service import ChEMBLService
from lobster.testing.provenance_replay import replay_ir


class _Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"activities": [{"activity_id": 1}]}


@pytest.mark.parametrize("target", ["CHEMBL240", None])
def test_the_exported_bioactivity_cell_runs(monkeypatch, target):
    sent = {}

    def get(url, params, timeout):
        sent.update(params)
        return _Response()

    monkeypatch.setattr(httpx, "get", get)
    ir = ChEMBLService()._create_ir_get_bioactivity("CHEMBL25", target)
    ast.parse(ir.render())

    scope = replay_ir(ir)

    assert scope["activities"] == [{"activity_id": 1}]
    assert sent["molecule_chembl_id"] == "CHEMBL25"
    assert sent.get("target_chembl_id") == target
