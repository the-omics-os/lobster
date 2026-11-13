"""Platform gate decisions for 30 real GEO series, offline.

The fixture holds each series' esummary fields and the GEO platform records
(title, technology) of its platforms. The modality check is stubbed to return
None (its permissive fallback), so these tests pin the deterministic platform
tier only.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# GEOService imports scanpy (the transcriptomics extra); skip, not error, without it.
pytest.importorskip("scanpy")

from lobster.core.exceptions import UnsupportedPlatformError
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.data_access.geo_service import GEOService

FIXTURE = (
    Path(__file__).parents[3] / "fixtures" / "geo_platform_gate" / "core_series.json"
)
ROWS = json.loads(FIXTURE.read_text())["rows"]

# Rows the gate gets wrong today: {row: reason}. strict xfail, so a fix that
# makes a row pass must remove it here.
KNOWN_WRONG: dict = {}


def _params():
    for row in ROWS:
        marks = []
        if row["row"] in KNOWN_WRONG:
            marks.append(
                pytest.mark.xfail(
                    strict=True, reason=f"row {row['row']}: {KNOWN_WRONG[row['row']]}"
                )
            )
        yield pytest.param(
            row, id=f"row{row['row']:02d}-{row['accession']}", marks=marks
        )


def _metadata(row: dict) -> dict:
    return {
        "title": row["esummary"]["title"],
        "type": row["esummary"]["gdsType"],
        "platforms": {
            gpl: {"title": p["title"], "technology": p["technology"]}
            for gpl, p in row["platforms"].items()
        },
        "samples": {},
    }


@pytest.fixture(scope="module")
def geo_service(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("ws")
    return GEOService(data_manager=DataManagerV2(workspace_path=workspace))


def test_fixture_has_the_30_series():
    assert [r["row"] for r in ROWS] == list(range(1, 31))


@pytest.mark.parametrize("row", list(_params()))
def test_platform_gate_decision(geo_service, row):
    expected = row["expected_gate"]
    with patch.object(geo_service, "_data_expert_assistant", create=True) as llm:
        llm.detect_modality.return_value = None
        if expected == "pass":
            ok, _ = geo_service._check_platform_compatibility(
                row["accession"], _metadata(row)
            )
            assert ok is True
            llm.detect_modality.assert_called_once()
        else:
            platform_type = expected.split(":", 1)[1]
            with pytest.raises(UnsupportedPlatformError) as err:
                geo_service._check_platform_compatibility(
                    row["accession"], _metadata(row)
                )
            assert err.value.details["platform_type"] == platform_type
            llm.detect_modality.assert_not_called()
