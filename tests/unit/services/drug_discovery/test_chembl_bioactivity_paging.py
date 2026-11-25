"""ChEMBL bioactivity reads every page, and says so when a cap stops it early.

ChEMBL serves activity records in pages. Reading only the first page and
reporting it as the whole set hides most of the data for well-studied
compounds.
"""

from unittest.mock import MagicMock

import pytest

from lobster.services.drug_discovery.chembl_service import ChEMBLService

TOTAL = 250


def _fake_chembl(requests):
    """A ChEMBL /activity endpoint holding TOTAL records, paged by offset/limit."""

    def get_json(url, params=None, max_retries=2):
        requests.append(dict(params))
        offset = int(params.get("offset", 0))
        limit = int(params["limit"])
        records = [
            {"activity_id": i, "standard_type": "IC50"}
            for i in range(offset, min(offset + limit, TOTAL))
        ]
        has_next = offset + limit < TOTAL
        return {
            "activities": records,
            "page_meta": {
                "total_count": TOTAL,
                "limit": limit,
                "offset": offset,
                "next": "/chembl/api/data/activity.json?next" if has_next else None,
            },
        }, None

    return get_json


def test_every_page_is_read(monkeypatch):
    requests = []
    service = ChEMBLService()
    monkeypatch.setattr(service, "_get_json", _fake_chembl(requests))

    _, stats, _ = service.get_bioactivity("CHEMBL25")

    ids = [a["activity_id"] for a in stats["activities"]]
    assert ids == list(range(TOTAL))
    assert stats["n_activities"] == TOTAL
    assert stats["total_count"] == TOTAL
    assert stats["truncated"] is False
    assert stats["activity_type_counts"] == {"IC50": TOTAL}
    assert len(requests) == 3


def test_a_cap_is_said_plainly(monkeypatch):
    requests = []
    service = ChEMBLService()
    monkeypatch.setattr(service, "_get_json", _fake_chembl(requests))

    _, stats, _ = service.get_bioactivity("CHEMBL25", max_records=150)

    assert stats["n_activities"] == 150
    assert stats["total_count"] == TOTAL
    assert stats["truncated"] is True
    assert "150 of 250 activities (stopped at the limit of 150)" in stats["coverage"]
    assert len(requests) == 2


def test_a_failed_later_page_keeps_what_was_read_and_says_so(monkeypatch):
    requests = []
    service = ChEMBLService()
    fake = _fake_chembl(requests)

    def flaky(url, params=None, max_retries=2):
        if int(params.get("offset", 0)) > 0:
            return None, "ChEMBL API timeout"
        return fake(url, params, max_retries)

    monkeypatch.setattr(service, "_get_json", flaky)

    _, stats, _ = service.get_bioactivity("CHEMBL25")

    assert "error" not in stats
    assert stats["n_activities"] == 100
    assert stats["truncated"] is True
    assert "100 of 250 activities" in stats["coverage"]


def _capped_service(monkeypatch):
    service = ChEMBLService()
    monkeypatch.setattr(service, "_get_json", _fake_chembl([]))
    real = service.get_bioactivity
    monkeypatch.setattr(
        service,
        "get_bioactivity",
        lambda chembl_id, target_chembl_id=None: real(
            chembl_id, target_chembl_id, max_records=150
        ),
    )
    return service


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


@pytest.mark.parametrize(
    "tool_name, args",
    [
        ("get_compound_bioactivity", {"chembl_id": "CHEMBL25"}),
        ("search_clinical_trials", {"chembl_id": "CHEMBL25"}),
        ("get_pharmacogenomic_evidence", {"chembl_id": "CHEMBL25"}),
        ("compare_drug_candidates", {"candidates": "CHEMBL25, CHEMBL1201585"}),
    ],
)
def test_the_tool_reply_names_the_cap(monkeypatch, tool_name, args):
    from lobster.agents.drug_discovery.clinical_tools import create_clinical_tools
    from lobster.agents.drug_discovery.pharmacogenomics_tools import (
        create_pharmacogenomics_tools,
    )
    from lobster.agents.drug_discovery.shared_tools import create_shared_tools

    service = _capped_service(monkeypatch)
    dm = MagicMock()
    tools = (
        create_shared_tools(dm, service, MagicMock(), MagicMock(), MagicMock())
        + create_clinical_tools(dm, MagicMock(), MagicMock(), service, MagicMock())
        + create_pharmacogenomics_tools(dm, MagicMock(), service)
    )

    reply = _tool(tools, tool_name).invoke(args)

    assert "150 of 250 activities (stopped at the limit of 150)" in reply


def test_the_exported_cell_reads_every_page(monkeypatch):
    import httpx

    from lobster.testing.provenance_replay import replay_ir

    serve = _fake_chembl([])

    class _Response:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    monkeypatch.setattr(
        httpx, "get", lambda url, params, timeout: _Response(serve(url, params)[0])
    )
    ir = ChEMBLService()._create_ir_get_bioactivity("CHEMBL25", None)

    scope = replay_ir(ir)

    assert [a["activity_id"] for a in scope["activities"]] == list(range(TOTAL))
