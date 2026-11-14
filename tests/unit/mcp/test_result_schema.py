"""``lobster.result/1`` describes what an MCP run adds to ``result.json``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from lobster.cli_internal import mcp_record

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "lobster/schemas/lobster.result.v1.schema.json"
    ).read_text()
)
BASE = {
    "schema_version": "lobster.result/1",
    "engine": {"name": "lobster-ai", "version": "1"},
    "session_id": None,
    "run_id": "run_x",
    "response": "",
}
REQUEST = {
    "analysis": "scrna_qc",
    "input_path": "/in/pbmc.h5ad",
    "instruction": None,
    "channel": "mcp:claude-code",
    "turn": 1,
}
validator = Draft202012Validator(SCHEMA)


def test_nothing_became_required():
    assert SCHEMA["required"] == [
        "schema_version",
        "engine",
        "session_id",
        "run_id",
        "response",
    ]
    validator.validate(BASE)


def test_the_warning_codes_are_the_closed_list():
    codes = SCHEMA["properties"]["warnings"]["items"]["properties"]["code"]["enum"]
    assert codes == list(mcp_record.WARNING_CODES)


def test_a_record_built_for_result_json_validates():
    sections = {
        "metrics": {
            "cells_before": 2700,
            "cells_after": 2699,
            "genes_before": 32738,
            "genes_after": 13714,
            "qc": {
                "thresholds_assessed": {"max_ribo_pct": 50.0},
                "thresholds_applied": {"max_ribo_percent": None},
                "flagged_kept": 87,
            },
            "tokens": {"total": 1, "cost_usd": 0.1, "usage_complete": True},
        },
        "modality_details": [
            {"name": "m", "stage": "raw", "parent": None, "n_obs": 3, "n_vars": 2}
        ],
        "steps": [{"seq": 0, "tool": "t", "agent": None, "parameters": {"a": "1"}}],
        "steps_total": 60,
        "warnings": [{"code": "qc_flagged_cells_kept", "message": "87 cells."}],
    }
    record = mcp_record.result_record({**BASE, "modalities": ["m"]}, sections, REQUEST)
    assert record["modalities"] == ["m"]  # the engine's names, strings only
    validator.validate(record)
    validator.validate({**record, "continued_from": "run_y"})
    validator.validate({**record, "continued_from": None})
    validator.validate({**record, "request": {**REQUEST, "instruction": "x"}})


def test_files_and_plot_counts_validate():
    record = {
        **BASE,
        "files_written": [
            {
                "path": "/out/plots/qc.html",
                "relative": "plots/qc.html",
                "category": "plots",
                "size_bytes": 5,
                "format": "html",
                "title": "QC plots",
                "figure_ref": "fig@rev",
            }
        ],
        "plots_truncated": True,
        "plots_total": 9,
        "continued_from": "run_y",
    }
    validator.validate(record)


def test_the_engine_modality_names_still_validate():
    validator.validate({**BASE, "modalities": ["pbmc"], "has_data": True})


@pytest.mark.parametrize(
    "bad",
    [
        {"warnings": [{"code": "made_up", "message": "x"}]},
        {"warnings": [{"code": "open_question"}]},
        {"warnings": "none"},
        {"modalities": [3]},
        {"modalities": [{"name": "pbmc"}]},  # names only; objects are modality_details
        {"modality_details": ["pbmc"]},
        {"modality_details": [{"stage": "raw"}]},
        {"modality_details": [{"name": "m", "n_obs": "3"}]},
        {"request": {**REQUEST, "turn": "2"}},
        {
            "files_written": [
                {
                    "path": "p",
                    "relative": "p",
                    "category": "c",
                    "size_bytes": 1,
                    "format": 3,
                }
            ]
        },
        {
            "files_written": [
                {
                    "path": "p",
                    "relative": "p",
                    "category": "c",
                    "size_bytes": 1,
                    "title": 3,
                }
            ]
        },
        {"plots_truncated": "yes"},
        {"plots_total": "9"},
        {"steps": [{"seq": "0", "tool": "t", "parameters": {}}]},
        {"steps": [{"seq": 0, "parameters": {}}]},
        {"steps_total": "60"},
        {"continued_from": 7},
        {"request": {**REQUEST, "instruction": 3}},
        {"request": {"analysis": "scrna_qc"}},
        {"metrics": []},
        {"metrics": {"cells_before": "3"}},
    ],
)
def test_wrong_shapes_are_refused(bad):
    with pytest.raises(ValidationError):
        validator.validate({**BASE, **bad})


def test_token_usage_says_what_its_figures_are():
    usage = SCHEMA["properties"]["token_usage"]
    assert "properties" not in usage  # the engine's own keys stay open
    text = usage["description"]
    assert "Cost is the list value of the tokens" in text
    assert "cached input is priced at the full input rate" in text
    assert "run_cost_complete false means at least one call was unpriced" in text
    assert "lower bound" in text
    assert "latest_cost_usd is the cost of the last model call, not of the run" in text
    validator.validate({**BASE, "token_usage": {"run_cost_usd": 0.1, "other": 1}})
    validator.validate({**BASE, "token_usage": None})
