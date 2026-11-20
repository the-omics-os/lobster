"""Replicates are counted in biological units (donors), not in rows (libraries).

Several sequencing libraries from one donor are technical replicates. Counting
them as biological replicates inflates n and the confidence of the DE result.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.de_analysis_expert import (
    _count_biological_replicates,
    _find_biological_unit_column,
    _replicate_summary_text,
    de_analysis_expert,
)
from lobster.testing.provenance_replay import RecordingDataManager, factory_tools

DE_MODULE = "lobster.agents.transcriptomics.de_analysis_expert"


def _metadata(donors_per_group, libraries_per_donor, unit_col="donor_id"):
    rows = []
    for condition in ("control", "treated"):
        for d in range(donors_per_group):
            for lib in range(libraries_per_donor):
                rows.append(
                    {
                        "condition": condition,
                        unit_col: f"{condition}_d{d}",
                        "library_id": f"{condition}_d{d}_L{lib}",
                    }
                )
    return pd.DataFrame(rows)


class TestCountBiologicalReplicates:
    def test_three_libraries_from_one_donor_fail(self):
        result = _count_biological_replicates(_metadata(1, 3), "condition")

        assert result["valid"] is False
        assert result["unit_col"] == "donor_id"
        assert result["group_counts"] == {"control": 1, "treated": 1}
        assert result["row_counts"] == {"control": 3, "treated": 3}
        assert "3 libraries/samples -> 1 donor(s)" in result["errors"][0]
        assert "pseudoreplication" in result["errors"][0]
        assert "technical" in result["errors"][0]

    def test_three_donors_pass(self):
        result = _count_biological_replicates(_metadata(3, 1), "condition")

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["group_counts"] == {"control": 3, "treated": 3}
        assert "Group 'control': 3 libraries/samples -> 3 donor(s)" in (
            result["summary"][0]
        )

    def test_libraries_above_donors_warn_to_sum_per_donor(self):
        result = _count_biological_replicates(_metadata(4, 2), "condition")

        assert result["valid"] is True
        assert result["group_counts"] == {"control": 4, "treated": 4}
        assert any(
            "8 rows come from 4 donor(s)" in w and "sample_col='donor_id'" in w
            for w in result["warnings"]
        )

    def test_without_unit_column_rows_are_counted_and_flagged(self):
        meta = _metadata(1, 3).drop(columns=["donor_id"])

        result = _count_biological_replicates(meta, "condition")

        assert result["unit_col"] is None
        assert result["group_counts"] == {"control": 3, "treated": 3}
        assert result["valid"] is True
        assert result["summary"] == []
        assert any("No biological-unit column" in w for w in result["warnings"])

    def test_two_donors_below_minimum_fail(self):
        result = _count_biological_replicates(_metadata(2, 1), "condition")

        assert result["valid"] is False
        assert "only 2 biological replicates" in result["errors"][0]

    @pytest.mark.parametrize(
        "name", ["donor", "Donor ID", "patient_id", "subject-id", "animal", "mouseID"]
    )
    def test_unit_column_names_are_recognized(self, name):
        meta = _metadata(1, 3, unit_col=name)
        assert _find_biological_unit_column(meta, "condition") == name

    @pytest.mark.parametrize("name", ["sample", "library_id", "run", "batch"])
    def test_library_like_columns_are_not_units(self, name):
        meta = pd.DataFrame({"condition": ["a", "b"], name: ["x", "y"]})
        assert _find_biological_unit_column(meta, "condition") is None

    def test_several_unit_columns_use_the_stricter_one(self):
        meta = _metadata(3, 1)
        meta["patient"] = meta["condition"] + "_p0"  # one patient per condition

        result = _count_biological_replicates(meta, "condition")

        assert result["unit_col"] == "patient"
        assert result["valid"] is False

    def test_summary_text(self):
        text = _replicate_summary_text(
            _count_biological_replicates(_metadata(3, 2), "condition")
        )
        assert "n counted in donors" in text
        assert "6 libraries/samples -> 3 donor(s)" in text
        assert _replicate_summary_text({"summary": []}) == ""


def _pseudobulk(donors_per_group, libraries_per_donor):
    meta = _metadata(donors_per_group, libraries_per_donor)
    meta.index = meta["library_id"]
    rng = np.random.default_rng(0)
    counts = rng.poisson(20, size=(len(meta), 40)).astype(float)
    return ad.AnnData(
        X=counts, obs=meta, var=pd.DataFrame(index=[f"gene_{i}" for i in range(40)])
    )


class TestDesignToolUsesDonors:
    def _tools(self, adata, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dm = RecordingDataManager({"pb": adata})
        return factory_tools(de_analysis_expert, DE_MODULE, dm)

    def test_one_donor_with_three_libraries_is_refused(self, tmp_path, monkeypatch):
        tools = self._tools(_pseudobulk(1, 3), tmp_path, monkeypatch)

        result = tools["prepare_de_design"].invoke(
            {
                "modality_name": "pb",
                "formula": "~condition",
                "contrast": ["condition", "treated", "control"],
            }
        )

        assert "1 donor(s)" in result
        assert "pseudoreplication" in result
        assert "1 biological replicate (" in result

    def test_three_donors_report_libraries_to_donors(self, tmp_path, monkeypatch):
        tools = self._tools(_pseudobulk(3, 2), tmp_path, monkeypatch)

        result = tools["prepare_de_design"].invoke(
            {
                "modality_name": "pb",
                "formula": "~condition",
                "contrast": ["condition", "treated", "control"],
            }
        )

        assert "6 libraries/samples -> 3 donor(s)" in result
        assert "sample_col='donor_id'" in result
