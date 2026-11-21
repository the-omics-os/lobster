"""run_metabolomics_statistics takes a fold_change_threshold it does not apply:
up/down are counted at |log2FC| > 1. The record logged the value as a parameter
(the exported notebook lists it under the step) and the reply did not say it
had no effect. The record now leaves it out and the reply names it."""

from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.agents.metabolomics.shared_tools import create_shared_tools
from lobster.services.analysis.metabolomics_analysis_service import (
    MetabolomicsAnalysisService,
)
from lobster.testing.provenance_replay import RecordingDataManager


def _run(tmp_path, **arguments):
    rng = np.random.default_rng(0)
    x = rng.lognormal(3, 0.2, size=(6, 5))
    x[3:, 0] *= 4  # 4-fold up
    adata = ad.AnnData(
        x,
        obs=pd.DataFrame(
            {"condition": ["a"] * 3 + ["b"] * 3}, index=[f"S{i}" for i in range(6)]
        ),
        var=pd.DataFrame(index=[f"M{i}" for i in range(5)]),
    )
    dm = RecordingDataManager({"met": adata}, workspace_path=tmp_path / "ws")
    tools = create_shared_tools(
        dm, MagicMock(), MagicMock(), MetabolomicsAnalysisService(), MagicMock()
    )
    tool = {t.name: t for t in tools}["run_metabolomics_statistics"]
    reply = tool.invoke(
        {"modality_name": "met", "group_column": "condition", **arguments}
    )
    return reply, dm.calls_for("run_metabolomics_statistics")


@pytest.mark.parametrize("arguments", [{}, {"fold_change_threshold": 3.0}])
def test_the_unapplied_threshold_is_named_and_not_recorded(tmp_path, arguments):
    reply, calls = _run(tmp_path, **arguments)

    value = arguments.get("fold_change_threshold", 1.5)
    assert (
        f"- fold_change_threshold {value}: not applied; up/down use "
        "|log2FC| > 1 (2-fold)" in reply
    ), reply
    (call,) = calls
    assert "fold_change_threshold" not in call.parameters
    assert "fold_change_threshold" not in call.ir.parameters
    assert "fold_change_threshold" not in call.ir.code_template
